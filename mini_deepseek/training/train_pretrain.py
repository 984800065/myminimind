"""
MiniDeepSeek 预训练入口：get_pretrain_config() 加载参数，DDP + 混合精度 + swanlab。
"""

import os
import time
from contextlib import nullcontext

import deepspeed
import swanlab
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from mini_deepseek.config import PretrainConfig, get_pretrain_config
from mini_deepseek.data import PretrainDataset
from mini_deepseek.model.configuration_mini_deepseek import MiniDeepSeekConfig
from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekCausalLMOutputWithPast, MiniDeepSeekForCausalLM
from mini_deepseek.utils.deepspeed_utils import (
    load_deepspeed_engine_checkpoint,
    load_deepspeed_resume_metadata,
    resolve_deepspeed_config,
    save_deepspeed_checkpoint,
    save_resolved_deepspeed_config,
)
from mini_deepseek.utils.logger import logger
from mini_deepseek.utils.train_utils import (
    SkipBatchSampler,
    get_model_weight_path,
    get_swanlab_experiment_name,
    init_distributed,
    init_model,
    is_main_process,
    lm_checkpoint,
    log_swanlab_training_metrics,
    resolve_lm_config_and_tokenizer,
    restore_rng_state,
    save_model_config,
    setup_seed,
)


def train_epoch(
    cfg: PretrainConfig,
    epoch: int,
    loader: DataLoader,
    model: MiniDeepSeekForCausalLM | DistributedDataParallel | deepspeed.DeepSpeedEngine,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler | None,
    autocast_ctx: nullcontext | torch.autocast,
    lm_config: MiniDeepSeekConfig,
    start_step: int = 0,
    swanlab_: swanlab.Run | None = None,
) -> None:
    """
    执行单个 epoch 的预训练。

    逐个 batch 完成前向计算、反向传播、梯度累积、参数更新、指标记录和
    checkpoint 保存。DeepSpeed 模式由引擎管理反向传播和参数更新，原生
    PyTorch 模式使用 GradScaler 管理混合精度，并显式处理梯度累积。

    续训时，`loader` 已跳过当前 epoch 中完成的 batch，`start_step` 保留其
    原始编号，使进度条、日志和 checkpoint 中的 step 在恢复前后保持连续。

    Args:
        cfg: 预训练配置，包含设备、梯度累积、日志和 checkpoint 等参数。
        epoch: 当前 epoch 的零基索引。
        loader: 当前 epoch 的 DataLoader；续训时仅包含尚未处理的 batch。
        model: 待训练模型，可以是原始模型、DDP 包装模型或 DeepSpeed 引擎。
        optimizer: 负责更新模型参数的优化器。
        lr_scheduler: 按 optimizer update 调整学习率的调度器。
        scaler: 原生混合精度训练使用的梯度缩放器；DeepSpeed 模式下为 `None`。
        autocast_ctx: 原生混合精度上下文；CPU 或 DeepSpeed 模式下为空上下文。
        lm_config: 模型结构配置，保存 checkpoint 时与模型权重一起持久化。
        start_step: 当前 loader 首个 batch 在原 epoch 中的零基编号；默认从 `0` 开始。
        swanlab_: SwanLab 实验实例；未启用实验跟踪时为 `None`。

    Returns:
        None。
    """
    model.train()
    start_time = time.time()

    # 加上 start_step 得到原 epoch 的完整 step 数，使进度条和日志在续训前后保持连续。
    total_iters = len(loader) + start_step
    # tqdm 会在每个 batch 迭代时更新进度条，显示当前 step、总 step、耗时和预计剩余时间。
    pbar = tqdm(loader, total=total_iters, initial=start_step, desc=f"Epoch[{epoch + 1}/{cfg.epochs}]", leave=True)

    # DeepSpeedEngine 自己管理清零梯度；原生训练需要在进入循环前显式清零。
    if not cfg.use_deepspeed:
        optimizer.zero_grad(set_to_none=True)

    # 这些统计量只用于展示当前进程在本 epoch 内的平均 loss。
    epoch_avg_loss = 0.0
    epoch_avg_aux_loss = 0.0
    cur_step = 0
    for step, (input_ids, labels) in enumerate(pbar, start=start_step):
        input_ids: torch.Tensor
        labels: torch.Tensor

        # 在分布式训练时，main() 函数在 init_distributed() 后按 LOCAL_RANK 设置 cfg.device，此处将当前 rank 的 batch 移到对应 GPU。如果没有分布式则使用 config 中指定的GPU设备。
        input_ids = input_ids.to(cfg.device)
        labels = labels.to(cfg.device)

        # 原生 CUDA 训练使用 autocast；CPU 和自行管理混合精度的 DeepSpeed 使用空上下文。
        with autocast_ctx:
            res: MiniDeepSeekCausalLMOutputWithPast = model(input_ids=input_ids, labels=labels)
            assert res.loss is not None, "模型前向传播未返回loss"

            # MoE 模型将专家负载均衡辅助损失与语言模型 loss 相加，Dense 模型只使用语言模型 loss。
            if res.aux_loss is None:
                loss = res.loss
            else:
                loss: torch.Tensor = res.loss + res.aux_loss

            # 在缩放梯度前记录原始 loss，避免日志展示累积后的缩放值。
            cur_loss = loss.item()
            cur_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0

            # 原生训练按当前累积窗口的实际 batch 数缩放 loss；DeepSpeed 会自行完成同等处理。
            if not cfg.use_deepspeed:
                window_start = (step // cfg.accumulation_steps) * cfg.accumulation_steps
                accumulation_divisor = min(
                    cfg.accumulation_steps,
                    total_iters - window_start,
                )
                loss = loss / accumulation_divisor
            epoch_avg_loss += cur_loss
            epoch_avg_aux_loss += cur_aux_loss
            cur_step += 1
            pbar.set_postfix({"batch_loss": cur_loss, "epoch_avg_loss": epoch_avg_loss / cur_step, "batch_aux_loss": cur_aux_loss, "epoch_avg_aux_loss": epoch_avg_aux_loss / cur_step})

        # DeepSpeedEngine 自行管理反向传播和累积边界；原生训练通过 GradScaler 执行反向传播。
        if cfg.use_deepspeed:
            assert isinstance(model, deepspeed.DeepSpeedEngine), "启用 DeepSpeed 时，模型应为 DeepSpeedEngine 实例"
            model.backward(loss)
            model.step()
        else:
            assert scaler is not None, "非 DeepSpeed 模式下 scaler 不应为空"
            scaler.scale(loss).backward()

            # 仅在完整累积窗口或 epoch 最后一个 batch 更新参数，并同步推进 scheduler。
            should_update = ((step + 1) % cfg.accumulation_steps == 0) or (step == total_iters - 1)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

        # 按配置频率记录 batch 指标、学习率、耗时和预计剩余时间；未启用 SwanLab 时函数直接返回。
        if step % cfg.log_interval == 0 or step == total_iters - 1:
            spend_time = time.time() - start_time
            cur_logits_loss = cur_loss - cur_aux_loss
            current_lr = optimizer.param_groups[0]["lr"]
            # 表示按当前平均 batch 耗时估算的本 epoch 剩余训练分钟数。
            eta_minutes = spend_time / (step + 1) * total_iters // 60 - spend_time // 60
            log_swanlab_training_metrics(
                swanlab_,
                epoch=epoch,
                step=step,
                steps_per_epoch=total_iters,
                total_epochs=cfg.epochs,
                learning_rate=current_lr,
                elapsed_seconds=spend_time,
                eta_minutes=float(eta_minutes),
                train_metrics={
                    "loss": cur_loss,
                    "logits_loss": cur_logits_loss,
                    "aux_loss": cur_aux_loss,
                },
            )

        # 原生 checkpoint 只在参数更新后保存以避开未序列化的累积梯度；DeepSpeed 可保存完整内部状态。
        checkpoint_due = (step + 1) % cfg.save_interval == 0 or step == total_iters - 1
        checkpoint_safe = cfg.use_deepspeed or should_update
        if checkpoint_due and checkpoint_safe:
            model.eval()

            check_point_path: str = get_model_weight_path(cfg)

            if cfg.use_deepspeed:
                # 主进程额外保存完整推理权重，所有 rank 协同保存 DeepSpeed 分片训练状态。
                raw_model = model.module if cfg.use_deepspeed or isinstance(model, DistributedDataParallel) else model
                raw_model = getattr(raw_model, "_orig_mod", raw_model)
                if is_main_process():
                    torch.save(raw_model.state_dict(), check_point_path)
                    save_model_config(cfg, lm_config)
                    logger.info(f"当前training step为 {step}，已保存模型权重到 {check_point_path}")
                save_deepspeed_checkpoint(
                    engine=model,
                    epoch=epoch,
                    step=step,
                    swanlab_=swanlab_,
                    cfg=cfg,
                )
            elif is_main_process():
                # 原生 checkpoint 保存推理权重及续训所需的优化器、scheduler、GradScaler、随机数状态和训练位置。
                lm_checkpoint(
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    lm_config=lm_config,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    swanlab_=swanlab_,
                )
            model.train()

        # 尽早释放本 batch 的大 tensor 引用，降低下一轮前向前的显存峰值。
        del input_ids, labels, res, loss


def train(
    cfg: PretrainConfig,
    model: MiniDeepSeekForCausalLM | DistributedDataParallel | deepspeed.DeepSpeedEngine,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler | None,
    autocast_ctx: nullcontext | torch.autocast,
    lm_config: MiniDeepSeekConfig,
    train_sampler: DistributedSampler | None,
    train_dataset: PretrainDataset,
    last_end_epoch: int,
    last_end_step: int,
    swanlab_: swanlab.Run | None = None,
) -> None:
    """
    执行完整的预训练 epoch 循环。

    每个 epoch 都使用固定种子构造数据顺序，并将 DataLoader 交给
    `train_epoch` 完成具体训练。从 epoch 中间续训时，会重建原始数据顺序并
    跳过已经完成的 batch，确保恢复前后的样本顺序保持一致。

    Args:
        cfg: 预训练配置，包含 epoch、batch size、数据加载和设备等参数。
        model: 待训练模型，可以是原始模型、DDP 包装模型或 DeepSpeed 引擎。
        optimizer: 负责更新模型参数的优化器。
        lr_scheduler: 按 optimizer update 调整学习率的调度器。
        scaler: 原生混合精度训练使用的梯度缩放器；DeepSpeed 模式下为 `None`。
        autocast_ctx: 原生混合精度上下文；CPU 或 DeepSpeed 模式下为空上下文。
        lm_config: 模型结构配置，保存 checkpoint 时与模型权重一起持久化。
        train_sampler: 分布式数据采样器；单进程训练时为 `None`。
        train_dataset: 预训练数据集。
        last_end_epoch: checkpoint 记录的最后训练 epoch，也是恢复训练的起始 epoch。
        last_end_step: checkpoint 在起始 epoch 中完成的最后一个 batch 编号；未续训时为 `-1`。
        swanlab_: SwanLab 实验实例；未启用实验跟踪时为 `None`。

    Returns:
        None。
    """
    start_epoch = last_end_epoch
    for epoch in range(start_epoch, cfg.epochs):
        # DistributedSampler 依赖 epoch 生成每轮不同且各 rank 一致的数据顺序。
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 仅恢复首个 epoch 时需要跳过已完成 batch；后续 epoch 都从头训练。
        skip = (last_end_step + 1) if (epoch == start_epoch and last_end_step >= 0) else 0

        # 新 epoch 重置全局随机源；续训首个 epoch 使用 checkpoint 恢复的随机状态。
        if skip == 0:
            setup_seed(42 + epoch)

        # 使用独立 Generator 固定数据排列和 DataLoader worker seed，避免全局 RNG 改变样本顺序。
        epoch_generator = torch.Generator().manual_seed(42 + epoch)
        indices = torch.randperm(
            len(train_dataset),
            generator=epoch_generator,
        ).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, cfg.batch_size, skip)
        loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            generator=epoch_generator,
        )

        # start_step 只负责恢复进度编号；实际跳批已经由 SkipBatchSampler 完成。
        if skip > 0:
            logger.info(f"Epoch[{epoch + 1}/{cfg.epochs}] 跳过前 {skip} step，从 step {skip} 开始")
            train_epoch(cfg, epoch, loader, model, optimizer, lr_scheduler, scaler, autocast_ctx, lm_config, skip, swanlab_)
        else:
            logger.info(f"Epoch[{epoch + 1}/{cfg.epochs}] 从头开始训练")
            train_epoch(cfg, epoch, loader, model, optimizer, lr_scheduler, scaler, autocast_ctx, lm_config, 0, swanlab_)


def main():
    # 加载运行配置。配置由代码默认值、环境变量、可选配置文件和命令行参数分层合并。
    cfg: PretrainConfig = get_pretrain_config()
    if cfg.use_deepspeed and cfg.use_compile:
        raise ValueError("当前预训练入口暂不建议同时启用 DeepSpeed 与 torch.compile，请先关闭其中一个。")

    # 初始化分布式环境和随机种子。分布式训练中，每个进程绑定 LOCAL_RANK 对应的 GPU；随机种子叠加全局 rank，使不同进程使用相互独立但可复现的随机序列。
    local_rank = init_distributed(use_deepspeed=cfg.use_deepspeed)
    if dist.is_initialized():
        cfg.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 配置混合精度上下文。DeepSpeed 启用时由其管理 autocast 和 loss scaling；原生 CUDA 训练使用 PyTorch autocast，CPU 训练则保持默认精度。
    device_type = "cuda" if "cuda" in cfg.device else "cpu"
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if (device_type == "cpu" or cfg.use_deepspeed) else torch.autocast(device_type=device_type, dtype=dtype)

    # 准备输出目录、模型配置和续训元数据。创建模型前先解析 tokenizer，并把词表大小及特殊 token ID 同步到 MiniDeepSeekConfig，确保模型结构与 tokenizer 一致。
    os.makedirs(cfg.save_dir, exist_ok=True)
    lm_config, tokenizer = resolve_lm_config_and_tokenizer(cfg)
    ckp_data = None
    ds_resume_meta = None
    if cfg.from_resume:
        # DeepSpeed 从专用目录读取分片状态；原生 DDP 或单卡训练读取项目自己保存的 resume checkpoint。
        if cfg.use_deepspeed:
            ds_resume_meta = load_deepspeed_resume_metadata(cfg)
        else:
            ckp_data = lm_checkpoint(cfg)

    # 仅在主进程初始化 SwanLab；续训时复用 run ID，将新指标继续写入原实验。
    swanlab_ = None
    if cfg.use_swanlab and is_main_process():
        resume_meta = ds_resume_meta if cfg.use_deepspeed else ckp_data
        swanlab_id = resume_meta.get("swanlab_id", None) if resume_meta else None
        resume = "must" if swanlab_id else None
        name = get_swanlab_experiment_name(cfg)
        swanlab.init(project=cfg.swanlab_project, name=name, id=swanlab_id, resume=resume)
        swanlab_ = swanlab

    # 构建模型、数据集和优化器；调度器的 T_max 按梯度累积后的参数更新次数计算。
    model, tokenizer = init_model(
        cfg=cfg,
        lm_config=lm_config,
        tokenizer=tokenizer,
    )
    if cfg.use_compile:
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    train_dataset = PretrainDataset(cfg.data_path, tokenizer, max_length=cfg.data_max_seq_len)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    scaler = None if cfg.use_deepspeed else torch.GradScaler(enabled=(cfg.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    cur_rank_total_samples = len(train_sampler) if train_sampler is not None else len(train_dataset)
    # 计算当前 rank 每个 epoch 的 micro-batch 数，并向上取整以包含不足 batch_size 的最后一批样本。
    micro_batches_per_epoch = (cur_rank_total_samples + cfg.batch_size - 1) // cfg.batch_size
    # 将 micro-batch 数按梯度累积轮数折算为实际参数更新次数，供学习率调度器计算 T_max。
    update_steps_per_epoch = max(1, (micro_batches_per_epoch + cfg.accumulation_steps - 1) // cfg.accumulation_steps)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * update_steps_per_epoch, eta_min=0.1 * cfg.learning_rate)

    if cfg.use_deepspeed:
        # DeepSpeedEngine 负责梯度累积、参数更新、精度管理和分布式训练状态。
        ds_config = resolve_deepspeed_config(cfg)
        save_resolved_deepspeed_config(ds_config, cfg)
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
        )

    # DeepSpeed 从 engine 恢复分片状态；原生 checkpoint 恢复模型、优化器、调度器、GradScaler 和训练位置。
    last_end_epoch, last_end_step = 0, -1
    if cfg.use_deepspeed:
        client_state = load_deepspeed_engine_checkpoint(model, cfg) if cfg.from_resume else None
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

    # 未启用 DeepSpeed 时使用原生 DDP 包装模型，避免叠加两种分布式引擎。
    if dist.is_initialized() and not cfg.use_deepspeed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    # 在模型和 DDP 初始化后恢复 RNG，避免初始化过程消耗续训所需的随机序列。
    if ckp_data is not None:
        restore_rng_state(ckp_data)

    # 训练主循环。每个 epoch 内部会按 batch 执行前向、反向、参数更新、日志记录和 checkpoint 保存。
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

    # 训练结束后释放分布式进程组资源。
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
