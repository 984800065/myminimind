"""
MiniMind 预训练入口：get_pretrain_config() 加载参数，DDP + 混合精度 + swanlab。
"""

import os
import time
from contextlib import nullcontext
from typing import Any

import swanlab
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from myminimind.config import PretrainConfig, get_pretrain_config
from myminimind.data import PretrainDataset
from myminimind.model.configuration_myminimind import MyMiniMindConfig
from myminimind.model.modeling_myminimind import MyCausalLMOutputWithPast, MyMiniMindForCausalLM
from myminimind.training.deepspeed_utils import (
    load_deepspeed_engine_checkpoint,
    load_deepspeed_resume_metadata,
    require_deepspeed,
    resolve_pretrain_deepspeed_config,
    save_deepspeed_checkpoint,
    save_resolved_deepspeed_config,
)
from myminimind.utils.logger import logger
from myminimind.utils.train_utils import (
    SkipBatchSampler,
    get_model_weight_path,
    init_distributed,
    init_model,
    is_main_process,
    lm_checkpoint,
    resolve_lm_config_and_tokenizer,
    setup_seed,
)


def train_epoch(
    cfg: PretrainConfig,
    epoch: int,
    loader: DataLoader,
    model: Any,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler | None,
    autocast_ctx: nullcontext | torch.autocast,
    lm_config: MyMiniMindConfig,
    start_step: int = 0,
    swanlab_: swanlab.Run | None = None,
) -> None:
    model.train()
    start_time = time.time()

    total_iters = len(loader) + start_step
    pbar = tqdm(loader, total=total_iters, initial=start_step, desc=f"Epoch[{epoch + 1}/{cfg.epochs}]", leave=True)

    if not cfg.use_deepspeed:
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

            res: MyCausalLMOutputWithPast = model(input_ids=input_ids, labels=labels)
            assert res.loss is not None, "模型前向传播未返回loss"
            if res.aux_loss is None:
                loss = res.loss
            else:
                loss: torch.Tensor = res.loss + res.aux_loss
                
            cur_loss = loss.item()
            cur_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0

            if not cfg.use_deepspeed:
                loss = loss / cfg.accumulation_steps
            epoch_avg_loss += cur_loss
            epoch_avg_aux_loss += cur_aux_loss
            cur_step += 1
            pbar.set_postfix({"batch_loss": cur_loss, "epoch_avg_loss": epoch_avg_loss / cur_step, "batch_aux_loss": cur_aux_loss, "epoch_avg_aux_loss": epoch_avg_aux_loss / cur_step})

        if cfg.use_deepspeed:
            model.backward(loss)
            model.step()
        else:
            assert scaler is not None, "非 DeepSpeed 模式下 scaler 不应为空"
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
            current_lr = optimizer.param_groups[0]["lr"]
            eta_min = spend_time / (step + 1) * total_iters // 60 - spend_time // 60
            if swanlab_:
                swanlab_.log({"loss": cur_loss, "logits_loss": cur_logits_loss, "aux_loss": cur_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if step % cfg.save_interval == 0 or step == total_iters - 1:
            model.eval()
            debug_suffix = "_debug" if cfg.debug else ""
            ckp = get_model_weight_path(
                save_dir=cfg.save_dir,
                weight=cfg.save_weight,
                hidden_size=lm_config.hidden_size,
                use_moe=lm_config.use_moe,
                attention_type=lm_config.attention_type,
            ).removesuffix(".pth") + f"{debug_suffix}.pth"
            raw_model = model.module if cfg.use_deepspeed or isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            if is_main_process():
                state_dict = raw_model.state_dict()
                torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
                del state_dict
            if cfg.use_deepspeed:
                save_deepspeed_checkpoint(
                    engine=model,
                    lm_config=lm_config,
                    weight=cfg.save_weight,
                    save_dir=cfg.save_dir,
                    epoch=epoch,
                    step=step,
                    swanlab_=swanlab_,
                )
            elif is_main_process():
                lm_checkpoint(
                    lm_config=lm_config,
                    weight=cfg.save_weight,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    swanlab_=swanlab_,
                    save_dir=cfg.save_dir,
                )
            model.train()

        del input_ids, labels, res, loss


def train(
    cfg: PretrainConfig,
    model: MyMiniMindForCausalLM,
    optimizer: optim.AdamW,
    lr_scheduler: optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.GradScaler,
    autocast_ctx,
    lm_config: MyMiniMindConfig,
    train_sampler: DistributedSampler | None,
    train_dataset: PretrainDataset,
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
    cfg = get_pretrain_config()
    if cfg.use_deepspeed:
        require_deepspeed()
    if cfg.use_deepspeed and cfg.use_compile:
        raise ValueError("当前预训练入口暂不建议同时启用 DeepSpeed 与 torch.compile，请先关闭其中一个。")

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed(use_deepspeed=cfg.use_deepspeed)
    if dist.is_initialized():
        cfg.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(cfg.save_dir, exist_ok=True)
    lm_config, tokenizer = resolve_lm_config_and_tokenizer(cfg.to_lm_config_kwargs(), cfg.tokenizer_path)
    ckp_data = None
    ds_resume_meta = None
    if cfg.from_resume:
        if cfg.use_deepspeed:
            ds_resume_meta = load_deepspeed_resume_metadata(lm_config, weight=cfg.save_weight, save_dir=cfg.save_dir)
        else:
            ckp_data = lm_checkpoint(lm_config, weight=cfg.save_weight, save_dir=cfg.save_dir)

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in cfg.device else "cpu"
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if (device_type == "cpu" or cfg.use_deepspeed) else torch.autocast(device_type=device_type, dtype=dtype)

    # ========== 4. 配swanlab ==========
    swanlab_ = None
    if cfg.use_swanlab and is_main_process():
        resume_meta = ds_resume_meta if cfg.use_deepspeed else ckp_data
        swanlab_id = resume_meta.get("swanlab_id", None) if resume_meta else None
        resume = "must" if swanlab_id else None
        model_name = f"MiniMind{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}"
        name = f"{model_name}-Pretrain-E{cfg.epochs}-B{cfg.batch_size}-LR{cfg.learning_rate}"
        swanlab.init(project=cfg.swanlab_project, name=name, id=swanlab_id, resume=resume)
        swanlab_ = swanlab

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(
        lm_config=lm_config,
        from_weight=cfg.from_weight,
        tokenizer_path=cfg.tokenizer_path,
        save_dir=cfg.save_dir,
        device=cfg.device,
        tokenizer=tokenizer,
    )
    if cfg.use_compile:
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    train_dataset = PretrainDataset(cfg.data_path, tokenizer, max_length=cfg.max_seq_len)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    scaler = None if cfg.use_deepspeed else torch.GradScaler(enabled=(cfg.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    cur_rank_total_samples = len(train_sampler) if train_sampler is not None else len(train_dataset)
    micro_batches_per_epoch = (cur_rank_total_samples + cfg.batch_size - 1) // cfg.batch_size
    update_steps_per_epoch = max(1, (micro_batches_per_epoch + cfg.accumulation_steps - 1) // cfg.accumulation_steps)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * update_steps_per_epoch, eta_min=0.1 * cfg.learning_rate)

    if cfg.use_deepspeed:
        deepspeed = require_deepspeed()
        ds_config = resolve_pretrain_deepspeed_config(cfg)
        save_resolved_deepspeed_config(ds_config, lm_config, cfg.save_weight, cfg.save_dir)
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
        )

    # ========== 6. 从ckp恢复状态 ==========
    last_end_epoch, last_end_step = 0, -1
    if cfg.use_deepspeed:
        client_state = load_deepspeed_engine_checkpoint(model, lm_config, weight=cfg.save_weight, save_dir=cfg.save_dir) if cfg.from_resume else None
        if client_state is not None:
            last_end_epoch = client_state.get("epoch", 0)
            last_end_step = client_state.get("step", -1)
    elif ckp_data is not None:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        if "lr_scheduler" in ckp_data:
            lr_scheduler.load_state_dict(ckp_data["lr_scheduler"])
        assert scaler is not None
        scaler.load_state_dict(ckp_data["scaler"])
        last_end_epoch = ckp_data.get("epoch", 0)
        last_end_step = ckp_data.get("step", -1)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized() and not cfg.use_deepspeed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

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
