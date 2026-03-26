"""
MiniMind On-policy 白盒蒸馏入口：get_distillation_config() 加载参数，DDP + 混合精度 + swanlab。
"""

import os
import time
from contextlib import nullcontext

import swanlab
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from myminimind.config import DistillationConfig, get_distillation_config
from myminimind.data.lm_dataset import SFTDataset
from myminimind.model.configuration_myminimind import MyMiniMindConfig as MiniMindConfig
from myminimind.model.modeling_myminimind import (
    MyCausalLMOutputWithPast as CausalLMOutputWithPast,
    MyMiniMindForCausalLM as MiniMindForCausalLM,
)
from myminimind.utils.logger import logger
from myminimind.utils.train_utils import (
    SkipBatchSampler,
    get_model_weight_path,
    init_distributed,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)


def train_epoch(
    cfg: DistillationConfig,
    epoch: int,
    loader: DataLoader,
    model: MiniMindForCausalLM,
    optimizer: optim.AdamW,
    lr_scheduler: optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    autocast_ctx: nullcontext | torch.amp.autocast,
    lm_config: MiniMindConfig,
    last_end_step: int = 0,
    swanlab_: swanlab.Run | None = None,
) -> None:
    model.train()
    start_time = time.time()

    total_iters = len(loader) + last_end_step + 1
    pbar = tqdm(loader, total=total_iters, initial=last_end_step, desc=f"Epoch[{epoch + 1}/{cfg.epochs}]", leave=True)

    epoch_avg_loss = 0.0
    epoch_avg_aux_loss = 0.0
    cur_step = 0
    for step, (input_ids, labels) in enumerate(pbar, start=last_end_step + 1):
        input_ids: torch.Tensor
        labels: torch.Tensor
        input_ids = input_ids.to(cfg.device)
        labels = labels.to(cfg.device)

        with autocast_ctx:
            res: CausalLMOutputWithPast = model(input_ids=input_ids, labels=labels)
            loss: torch.Tensor = res.loss + res.aux_loss
            cur_loss = loss.item()
            cur_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0

            loss = loss / cfg.accumulation_steps
            epoch_avg_loss += cur_loss
            epoch_avg_aux_loss += cur_aux_loss
            cur_step += 1
            pbar.set_postfix({"batch_loss": cur_loss, "epoch_avg_loss": epoch_avg_loss / cur_step, "batch_aux_loss": cur_aux_loss, "epoch_avg_aux_loss": epoch_avg_aux_loss / cur_step})

        # 累计梯度
        scaler.scale(loss).backward()

        # 梯度累计结束，更新参数
        if (step + 1) % cfg.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 每一步结束，更新学习率
        lr_scheduler.step()

        if step % cfg.log_interval == 0 or step == total_iters - 1:
            spend_time = time.time() - start_time
            cur_logits_loss = cur_loss - cur_aux_loss
            current_lr = lr_scheduler.get_last_lr()[0]
            eta_min = spend_time / (step + 1) * total_iters // 60 - spend_time // 60
            if swanlab_:
                swanlab_.log({"loss": cur_loss, "logits_loss": cur_logits_loss, "aux_loss": cur_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % cfg.save_interval == 0 or step == total_iters - 1) and is_main_process():
            model.eval()
            ckp = get_model_weight_path(
                save_dir=cfg.save_dir,
                weight=cfg.save_weight,
                hidden_size=lm_config.hidden_size,
                use_moe=lm_config.use_moe,
                attention_type=getattr(lm_config, "attention_type", "gqa"),
            )
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config=lm_config,
                weight=cfg.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                swanlab_=swanlab_,
                save_dir=cfg.save_dir,
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss


def train(
    cfg: DistillationConfig,
    model: MiniMindForCausalLM,
    optimizer: optim.AdamW,
    lr_scheduler: optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    autocast_ctx,
    lm_config: MiniMindConfig,
    train_sampler: DistributedSampler | None,
    train_dataset: SFTDataset,
    last_end_epoch: int,
    last_end_step: int,
    swanlab_=None,
):
    start_epoch = last_end_epoch
    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_dataset)).tolist()
        skip = (last_end_step + 1) if (epoch == start_epoch and last_end_step >= 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, cfg.batch_size, skip)
        loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        if skip > 0:
            logger.info(f"Epoch[{epoch + 1}/{cfg.epochs}] 跳过前 {last_end_step} step，从 {last_end_step + 1} 开始")
            train_epoch(cfg, epoch, loader, model, optimizer, lr_scheduler, scaler, autocast_ctx, lm_config, last_end_step, swanlab_)
        else:
            logger.info(f"Epoch[{epoch + 1}/{cfg.epochs}] 从头开始训练")
            train_epoch(cfg, epoch, loader, model, optimizer, lr_scheduler, scaler, autocast_ctx, lm_config, 0, swanlab_)


def main():
    cfg = get_distillation_config()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed()
    if dist.is_initialized():
        cfg.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(cfg.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(**cfg.to_lm_config_kwargs())
    ckp_data = lm_checkpoint(lm_config, weight=cfg.save_weight, save_dir=cfg.save_dir) if cfg.from_resume else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in cfg.device else "cpu"
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4. 配swanlab ==========
    swanlab_ = None
    if cfg.use_swanlab and is_main_process():
        swanlab_id = ckp_data.get("swanlab_id", None) if ckp_data else None
        resume = "must" if swanlab_id else None
        model_name = f"MiniMind{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}"
        name = f"{model_name}-Distill-E{cfg.epochs}-B{cfg.batch_size}-LR{cfg.learning_rate}"
        swanlab.init(project=cfg.swanlab_project, name=name, id=swanlab_id, resume=resume)
        swanlab_ = swanlab

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(
        lm_config=lm_config,
        from_weight=cfg.from_weight,
        tokenizer_path=cfg.tokenizer_path,
        save_dir=cfg.save_dir,
        device=cfg.device,
    )
    if cfg.use_compile:
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    train_dataset = SFTDataset(cfg.data_path, tokenizer, max_length=cfg.max_seq_len)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    cur_rank_total_samples = len(train_sampler) if train_sampler is not None else len(train_dataset)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * ((cur_rank_total_samples + cfg.batch_size - 1) // cfg.batch_size), eta_min=0.1 * cfg.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    last_end_epoch, last_end_step = 0, -1
    if ckp_data is not None:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        last_end_epoch = ckp_data.get("epoch", 0)
        last_end_step = ckp_data.get("step", -1)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"cos_phi", "sin_phi"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    train(
        cfg,
        model,
        optimizer,
        lr_scheduler,
        scaler,
        autocast_ctx,
        lm_config,
        train_sampler,
        train_dataset,
        last_end_epoch,
        last_end_step,
        swanlab_,
    )

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
