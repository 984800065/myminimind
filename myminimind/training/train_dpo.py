"""
MiniMind DPO 入口：get_dpo_config() 加载参数，DDP + 混合精度 + swanlab。
"""

import os
import time
from contextlib import nullcontext

import swanlab
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from myminimind.config import DPOConfig, get_dpo_config
from myminimind.data import DPODataset
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


def logits_to_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits.shape == (batch_size, seq_len, vocab_size)
    # labels.shape == (batch_size, seq_len)
    assert len(labels.shape) == 2, f"labels.shape: {labels.shape}"

    # (batch_size, seq_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)
    # (batch_size, seq_len, 1)
    log_probs = log_probs.gather(dim=-1, index=labels[:, :, None])
    # (batch_size, seq_len)
    log_probs = log_probs.squeeze(-1)
    return log_probs


def get_dop_loss(ref_log_probs: torch.Tensor, policy_log_probs: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    # ref_log_probs.shape == (batch_size, seq_len)
    # policy_log_probs.shape == (batch_size, seq_len)
    # mask.shape == (batch_size, seq_len)

    # 防止零长度mask导致除零NaN
    # (batch_size, )
    seq_lengths = mask.sum(dim=-1).clamp_min(1e-8)

    # (batch_size, )
    ref_log_probs = (ref_log_probs * mask).sum(dim=-1) / seq_lengths
    # (batch_size, )
    policy_log_probs = (policy_log_probs * mask).sum(dim=-1) / seq_lengths

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_log_probs = ref_log_probs[: batch_size // 2]
    rejected_log_probs = ref_log_probs[batch_size // 2 :]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]
    rejected_policy_log_probs = policy_log_probs[batch_size // 2 :]

    pi_logratios = chosen_policy_log_probs - rejected_policy_log_probs
    ref_logratios = chosen_log_probs - rejected_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(
    cfg: DPOConfig,
    epoch: int,
    loader: DataLoader,
    model: MiniMindForCausalLM,
    ref_model: MiniMindForCausalLM,
    optimizer: optim.AdamW,
    lr_scheduler: optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    autocast_ctx: nullcontext | torch.amp.autocast,
    lm_config: MiniMindConfig,
    last_end_step: int = 0,
    swanlab_: swanlab.Run | None = None,
    beta: float = 0.1,
) -> None:
    model.train()
    start_time = time.time()

    total_iters = len(loader) + last_end_step + 1
    pbar = tqdm(loader, total=total_iters, initial=last_end_step, desc=f"Epoch[{epoch + 1}/{cfg.epochs}]", leave=True)

    epoch_avg_loss = 0.0
    epoch_avg_aux_loss = 0.0
    cur_step = 0
    for step, sample_pair in enumerate(pbar, start=last_end_step):
        # (batch_size, seq_len)
        x_chosen: torch.Tensor = sample_pair["x_chosen"]
        y_chosen: torch.Tensor = sample_pair["y_chosen"]
        mask_chosen: torch.Tensor = sample_pair["mask_chosen"]
        x_rejected: torch.Tensor = sample_pair["x_rejected"]
        y_rejected: torch.Tensor = sample_pair["y_rejected"]
        mask_rejected: torch.Tensor = sample_pair["mask_rejected"]

        x = torch.cat([x_chosen, x_rejected], dim=0).to(cfg.device)
        y = torch.cat([y_chosen, y_rejected], dim=0).to(cfg.device)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0).to(cfg.device)

        with autocast_ctx:
            with torch.no_grad():
                ref_outputs: CausalLMOutputWithPast = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            outputs: CausalLMOutputWithPast = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            dpo_loss = get_dop_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss: torch.Tensor = dpo_loss + outputs.aux_loss

            cur_loss = loss.item()
            cur_aux_loss = outputs.aux_loss.item() if outputs.aux_loss is not None else 0.0

            epoch_avg_loss += cur_loss
            epoch_avg_aux_loss += cur_aux_loss
            loss = loss / cfg.accumulation_steps
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
            cur_dpo_loss = cur_loss - cur_aux_loss
            current_lr = lr_scheduler.get_last_lr()[0]
            eta_min = spend_time / (step + 1) * total_iters // 60 - spend_time // 60
            if swanlab_:
                swanlab_.log({"total_loss": cur_loss, "dpo_loss": cur_dpo_loss, "aux_loss": cur_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min}, step=step)

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

        del x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, dpo_loss, loss


def train(
    cfg: DPOConfig,
    model: MiniMindForCausalLM,
    ref_model: MiniMindForCausalLM,
    optimizer: optim.AdamW,
    lr_scheduler: optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    autocast_ctx,
    lm_config: MiniMindConfig,
    train_sampler: DistributedSampler | None,
    train_dataset: DPODataset,
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
            train_epoch(cfg, epoch, loader, model, ref_model, optimizer, lr_scheduler, scaler, autocast_ctx, lm_config, last_end_step, swanlab_, cfg.beta)
        else:
            logger.info(f"Epoch[{epoch + 1}/{cfg.epochs}] 从头开始训练")
            train_epoch(cfg, epoch, loader, model, ref_model, optimizer, lr_scheduler, scaler, autocast_ctx, lm_config, 0, swanlab_, cfg.beta)


def main():
    cfg = get_dpo_config()

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
    )
    if cfg.use_compile:
        model = torch.compile(model)
        logger.info("torch.compile enabled")
    logger.info(f"策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    ref_model, _ = init_model(
        lm_config=lm_config,
        from_weight=cfg.from_weight,
        tokenizer_path=cfg.tokenizer_path,
        save_dir=cfg.save_dir,
        device=cfg.device,
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    logger.info(f"参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M")

    train_dataset = DPODataset(cfg.data_path, tokenizer, max_length=cfg.max_seq_len)
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
        ref_model,
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
