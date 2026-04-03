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
from myminimind.data import SFTDataset
from myminimind.model.configuration_myminimind import MyMiniMindConfig as MiniMindConfig
from myminimind.model.modeling_myminimind import (
    MyCausalLMOutputWithPast as CausalLMOutputWithPast,
)
from myminimind.model.modeling_myminimind import (
    MyMiniMindForCausalLM as MiniMindForCausalLM,
)
from myminimind.utils.logger import logger
from myminimind.utils.train_utils import (
    SkipBatchSampler,
    init_distributed,
    init_model,
    is_main_process,
    lm_checkpoint,
    log_swanlab_training_metrics,
    resolve_lm_config_and_tokenizer,
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
    start_step: int = 0,
    swanlab_: swanlab.Run | None = None,
) -> None:
    model.train()
    start_time = time.time()

    total_iters = len(loader) + start_step
    pbar = tqdm(loader, total=total_iters, initial=start_step, desc=f"Epoch[{epoch + 1}/{cfg.epochs}]", leave=True)

    optimizer.zero_grad(set_to_none=True)

    epoch_avg_loss = 0.0
    epoch_avg_aux_loss = 0.0
    cur_step = 0
    for step, (input_ids, labels) in enumerate(pbar, start=start_step):
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

        # 只在真正发生参数更新时推进 scheduler；
        # epoch 末尾不足 accumulation_steps 的剩余梯度也要补一次更新。
        should_update = ((step + 1) % cfg.accumulation_steps == 0) or (step == total_iters - 1)
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        if step % cfg.log_interval == 0 or step == total_iters - 1:
            spend_time = time.time() - start_time
            cur_logits_loss = cur_loss - cur_aux_loss
            current_lr = lr_scheduler.get_last_lr()[0]
            eta_min = spend_time / (step + 1) * total_iters // 60 - spend_time // 60
            log_swanlab_training_metrics(
                swanlab_,
                epoch=epoch,
                step=step,
                steps_per_epoch=total_iters,
                total_epochs=cfg.epochs,
                learning_rate=current_lr,
                elapsed_seconds=spend_time,
                eta_minutes=float(eta_min),
                train_metrics={
                    "loss": cur_loss,
                    "logits_loss": cur_logits_loss,
                    "aux_loss": cur_aux_loss,
                },
            )

        if (step % cfg.save_interval == 0 or step == total_iters - 1) and is_main_process():
            model.eval()
            lm_checkpoint(
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                step=step,
                swanlab_=swanlab_,
            )
            model.train()

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
            logger.info(f"Epoch[{epoch + 1}/{cfg.epochs}] 跳过前 {skip} step，从 step {skip} 开始")
            train_epoch(cfg, epoch, loader, model, optimizer, lr_scheduler, scaler, autocast_ctx, lm_config, skip, swanlab_)
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
    lm_config, tokenizer = resolve_lm_config_and_tokenizer(cfg)
    ckp_data = lm_checkpoint(cfg) if cfg.from_resume else None

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
        cfg=cfg,
        lm_config=lm_config,
        tokenizer=tokenizer,
    )
    if cfg.use_compile:
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    train_dataset = SFTDataset(cfg.data_path, tokenizer, max_length=cfg.max_seq_len)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    cur_rank_total_samples = len(train_sampler) if train_sampler is not None else len(train_dataset)
    micro_batches_per_epoch = (cur_rank_total_samples + cfg.batch_size - 1) // cfg.batch_size
    update_steps_per_epoch = max(1, (micro_batches_per_epoch + cfg.accumulation_steps - 1) // cfg.accumulation_steps)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * update_steps_per_epoch, eta_min=0.1 * cfg.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    last_end_epoch, last_end_step = 0, -1
    if ckp_data is not None:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        if "lr_scheduler" in ckp_data:
            lr_scheduler.load_state_dict(ckp_data["lr_scheduler"])
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
