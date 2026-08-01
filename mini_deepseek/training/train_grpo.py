"""
MiniDeepSeek GRPO 训练入口：get_grpo_config() 加载参数，DDP + 混合精度 + swanlab。
"""

import os
import re
import time
from contextlib import nullcontext
from pathlib import Path

import swanlab
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from mini_deepseek.config import GRPOConfig, get_grpo_config
from mini_deepseek.data import RLAIFDataset
from mini_deepseek.model.configuration_mini_deepseek import (
    MiniDeepSeekConfig,
    load_mini_deepseek_config,
)
from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekForCausalLM
from mini_deepseek.utils.logger import logger
from mini_deepseek.utils.train_utils import (
    SkipBatchSampler,
    get_swanlab_experiment_name,
    init_distributed,
    init_model,
    is_main_process,
    load_tokenizer,
    lm_checkpoint,
    log_swanlab_training_metrics,
    resolve_lm_config_and_tokenizer,
    resolve_model_weight_path,
    restore_rng_state,
    sync_lm_config_with_tokenizer,
    setup_seed,
)


def build_completion_mask(
    completion_ids: torch.Tensor,
    eos_token_id: int | None,
) -> torch.Tensor:
    """
    Mark generated tokens through and including the first EOS.

    Tokens after the first EOS are generation padding and must not contribute to
    policy/KL loss. Including EOS itself is important: otherwise GRPO never
    reinforces the action that terminates a response.
    """
    if eos_token_id is None:
        return torch.ones_like(completion_ids, dtype=torch.bool)

    is_eos = completion_ids.eq(eos_token_id)
    sequence_length = completion_ids.size(1)
    first_eos = torch.full(
        (completion_ids.size(0),),
        sequence_length,
        dtype=torch.long,
        device=completion_ids.device,
    )
    has_eos = is_eos.any(dim=1)
    first_eos[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    positions = torch.arange(sequence_length, device=completion_ids.device)
    return positions.unsqueeze(0) <= first_eos.unsqueeze(1)


def get_completion_log_probs(
    model: MiniDeepSeekForCausalLM | DistributedDataParallel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return completion log-probs and the model's optional MoE auxiliary loss."""
    if completion_length <= 0:
        raise ValueError("GRPO requires at least one generated completion token.")

    # Keep one extra position: logits at sequence position t predict token t+1.
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=completion_length + 1,
    )
    prediction_logits = outputs.logits[:, :-1, :].float()
    completion_ids = input_ids[:, -completion_length:]
    if prediction_logits.shape[:2] != completion_ids.shape:
        raise RuntimeError(
            "Completion log-prob alignment failed: "
            f"logits={tuple(prediction_logits.shape)}, "
            f"tokens={tuple(completion_ids.shape)}."
        )
    log_probs = torch.gather(
        prediction_logits.log_softmax(dim=-1),
        dim=-1,
        index=completion_ids.unsqueeze(-1),
    ).squeeze(-1)
    return log_probs, outputs.aux_loss


def resolve_grpo_model_config(
    cfg: GRPOConfig,
    base_weight: str,
) -> tuple[MiniDeepSeekConfig, AutoTokenizer]:
    """
    Rebuild policy/reference from the base checkpoint's architecture sidecar.

    GRPO changes optimization and context length, not hidden dimensions or
    attention/MoE structure. Falling back to GRPO defaults for those fields can
    make a valid SFT checkpoint fail with size mismatches.
    """
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    base_weight_path = Path(
        resolve_model_weight_path(
            cfg,
            weight=base_weight,
            include_debug=False,
        )
    )
    config_path = base_weight_path.with_suffix(".config.json")
    if not config_path.exists():
        logger.warning(
            f"基础权重缺少模型配置 sidecar: {config_path}，回退到 GRPO 配置。"
        )
        return resolve_lm_config_and_tokenizer(cfg)

    lm_config = load_mini_deepseek_config(config_path)
    sync_lm_config_with_tokenizer(lm_config, tokenizer)
    context_length = cfg.data_max_seq_len + cfg.max_gen_len
    lm_config.max_seq_len = context_length
    lm_config.max_position_embeddings = context_length
    if lm_config.dropout != 0.0:
        logger.warning(
            f"GRPO 将模型 dropout 从 {lm_config.dropout} 设为 0，"
            "确保 rollout 与 policy log-prob 重算一致。"
        )
        lm_config.dropout = 0.0

    # MTP is a pretraining-only auxiliary objective. Keeping its layers during
    # GRPO would execute unnecessary future-token heads in policy train mode.
    lm_config.mtp_level = 0

    # Keep output/checkpoint naming aligned with the architecture actually loaded.
    for field_name in cfg.__class__.model_fields:
        if hasattr(lm_config, field_name):
            setattr(cfg, field_name, getattr(lm_config, field_name))
    return lm_config, tokenizer


def calculate_rewards(
    cfg: GRPOConfig,
    prompts: list[str],
    responses: list[str],
    reward_model,
    reward_tokenizer,
) -> torch.Tensor:
    # len(prompts) == batch_size
    # len(responses) == batch_size * num_generations

    def reasoning_model_reward(rewards: torch.Tensor) -> torch.Tensor:
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern_2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern_2 in zip(matches_pattern, matches_pattern_2, strict=True):
            if match_pattern or match_pattern_2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=cfg.device)

        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=cfg.device)
        return rewards

    # (batch_size * num_generations, )
    rewards = torch.zeros(len(responses), device=cfg.device)
    if cfg.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0

        for i in range(batch_size):
            for j in range(cfg.num_generations):
                # response.shape == (batch_size * num_generations, response_len)
                response_idx = i * cfg.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches: list[dict] = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                tmp_chat: list[dict] = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)

                if cfg.reasoning == 1:
                    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        # (batch_size * num_generations, )
        reward_model_scores = torch.tensor(reward_model_scores, device=cfg.device)
        rewards += reward_model_scores
        # (batch_size * num_generations, )
        return rewards


def grpo_train_epoch(
    cfg: GRPOConfig,
    epoch: int,
    loader: DataLoader,
    total_iters: int,
    model: MiniDeepSeekForCausalLM,
    ref_model: MiniDeepSeekForCausalLM,
    reward_model,
    reward_tokenizer,
    tokenizer: AutoTokenizer,
    optimizer: optim.AdamW,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    autocast_ctx,
    lm_config: MiniDeepSeekConfig,
    zero_based_start_step: int = 0,
    swanlab_: swanlab.Run | None = None,
) -> None:
    model.train()
    start_time = time.time()
    pbar = tqdm(loader, total=total_iters, initial=zero_based_start_step, desc=f"Epoch[{epoch + 1}/{cfg.epochs}]", leave=True)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, start=zero_based_start_step):
        # len(prompts) == batch_size
        prompts: list[str] = batch["prompt"]
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False, padding_side="left", add_special_tokens=False).to(cfg.device)

        if cfg.data_max_seq_len:
            # (batch_size, data_max_seq_len)
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -cfg.data_max_seq_len :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -cfg.data_max_seq_len :]
        reward_prompts = [
            tokenizer.decode(
                row_ids[row_mask.bool()],
                skip_special_tokens=False,
            )
            for row_ids, row_mask in zip(
                prompt_inputs["input_ids"],
                prompt_inputs["attention_mask"],
                strict=True,
            )
        ]

        model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
        model_for_gen.eval()
        with torch.no_grad():
            # (batch_size * num_generations, prompt_len + response_len)
            outputs = model_for_gen.generate(**prompt_inputs, max_new_tokens=cfg.max_gen_len, do_sample=True, temperature=0.8, num_return_sequences=cfg.num_generations, pad_token_id=tokenizer.pad_token_id)
        # Some generation backends return inference tensors, which autograd
        # cannot save for the policy backward pass.
        if outputs.is_inference():
            outputs = outputs.clone()

        # (batch_size * num_generations, response_len)
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1) :]
        completion_mask = build_completion_mask(
            completion_ids,
            tokenizer.eos_token_id,
        )
        repeated_prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(
            cfg.num_generations,
            dim=0,
        )
        full_attention_mask = torch.cat(
            [repeated_prompt_mask, completion_mask.to(repeated_prompt_mask.dtype)],
            dim=1,
        )

        with autocast_ctx:
            # This project fixes model dropout at zero, so switching back to
            # train mode preserves rollout probabilities while enabling MoE to
            # report its load-balancing auxiliary loss.
            model.train()
            per_token_logps, aux_loss = get_completion_log_probs(
                model,
                outputs,
                full_attention_mask,
                completion_ids.size(1),
            )
            if aux_loss is None:
                aux_loss = torch.tensor(0.0, device=cfg.device)

        with torch.no_grad():
            with autocast_ctx:
                ref_per_token_logps, _ = get_completion_log_probs(
                    ref_model,
                    outputs,
                    full_attention_mask,
                    completion_ids.size(1),
                )

        # len(completions) == batch_size * num_generations
        completions: list[str] = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # (batch_size * num_generations, )
        rewards: torch.Tensor = calculate_rewards(
            cfg,
            reward_prompts,
            completions,
            reward_model,
            reward_tokenizer,
        ).to(cfg.device)

        # (batch_size, num_generations)
        grouped_rewards = rewards.reshape(-1, cfg.num_generations)
        # (batch_size * num_generations, )
        inner_batch_mean_reward = grouped_rewards.mean(dim=1).repeat_interleave(cfg.num_generations)
        # (batch_size * num_generations, )
        inner_batch_std_reward = grouped_rewards.std(dim=1, correction=0).repeat_interleave(cfg.num_generations)
        # (batch_size * num_generations, )
        advantages = torch.clamp((rewards - inner_batch_mean_reward) / (inner_batch_std_reward + 1e-8), min=-10, max=10)

        # (batch_size * num_generations, response_len)
        kl_div = ref_per_token_logps - per_token_logps
        # (batch_size * num_generations, response_len)
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # (batch_size * num_generations, response_len)
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages[:, None] - cfg.beta * per_token_kl)
        # \frac{1}{T} * \frac{1}{|o|} * \sum per_token_loss
        # ()
        completion_mask_float = completion_mask.to(per_token_loss.dtype)
        completion_lengths = completion_mask_float.sum(dim=1).clamp_min(1.0)
        policy_loss = (
            (per_token_loss * completion_mask_float).sum(dim=1)
            / completion_lengths
        ).mean()

        window_start = (step // cfg.accumulation_steps) * cfg.accumulation_steps
        accumulation_divisor = min(
            cfg.accumulation_steps,
            total_iters - window_start,
        )
        should_update = ((step + 1) % cfg.accumulation_steps == 0) or (step == total_iters - 1)
        loss = (policy_loss + aux_loss) / accumulation_divisor
        scaler.scale(loss).backward()

        if should_update:
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_interval == 0 or step == total_iters - 1:
            policy_loss_val = policy_loss.item()
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_lengths.mean().item()
            current_lr = optimizer.param_groups[0]["lr"]
            spend_time = time.time() - start_time
            eta_min = spend_time / (step + 1) * total_iters // 60 - spend_time // 60

            # logger.info(
            #     f"Epoch:[{epoch + 1}/{cfg.epochs}]({step}/{iters}), "
            #     f"Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, "
            #     f"Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}"
            # )
            pbar.set_postfix(
                policy_loss=policy_loss_val,
                aux_loss=current_aux_loss,
                reward=avg_reward_val,
                avg_response_len=avg_len_val,
                learning_rate=current_lr,
            )

            log_swanlab_training_metrics(
                swanlab_ if is_main_process() else None,
                epoch=epoch,
                step=step,
                steps_per_epoch=total_iters,
                total_epochs=cfg.epochs,
                learning_rate=current_lr,
                elapsed_seconds=spend_time,
                eta_minutes=float(eta_min),
                train_metrics={
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len (tokens)": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                },
            )

        if should_update and ((step + 1) % cfg.save_interval == 0 or step == total_iters - 1) and is_main_process():
            model.eval()
            lm_checkpoint(
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                lm_config=lm_config,
                epoch=epoch,
                step=step,
                scheduler=scheduler,
                scaler=scaler,
                swanlab_=swanlab_,
            )
            model.train()

        del prompt_inputs, reward_prompts, outputs, completion_ids, full_attention_mask
        del per_token_logps, ref_per_token_logps, completions, rewards
        del grouped_rewards, inner_batch_mean_reward, inner_batch_std_reward
        del advantages, completion_mask, completion_mask_float, completion_lengths


def train(
    cfg: GRPOConfig,
    model: MiniDeepSeekForCausalLM,
    ref_model: MiniDeepSeekForCausalLM,
    reward_model,
    reward_tokenizer,
    tokenizer: AutoTokenizer,
    optimizer: optim.AdamW,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    autocast_ctx,
    lm_config: MiniDeepSeekConfig,
    train_sampler: DistributedSampler | None,
    train_dataset: RLAIFDataset,
    last_end_epoch: int,
    last_end_step: int,
    swanlab_: swanlab.Run | None = None,
) -> None:
    start_epoch = last_end_epoch
    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        skip = (last_end_step + 1) if (epoch == start_epoch and last_end_step >= 0) else 0
        if skip == 0:
            setup_seed(42 + epoch)
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

        if skip > 0:
            logger.info(f"Epoch[{epoch + 1}/{cfg.epochs}] 跳过前 {skip} step，从 step {skip} 开始")
            grpo_train_epoch(
                cfg,
                epoch,
                loader,
                len(loader) + skip,
                model,
                ref_model,
                reward_model,
                reward_tokenizer,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
                autocast_ctx,
                lm_config,
                zero_based_start_step=skip,
                swanlab_=swanlab_,
            )
        else:
            logger.info(f"Epoch[{epoch + 1}/{cfg.epochs}] 从头开始训练")
            grpo_train_epoch(
                cfg,
                epoch,
                loader,
                len(loader),
                model,
                ref_model,
                reward_model,
                reward_tokenizer,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
                autocast_ctx,
                lm_config,
                zero_based_start_step=0,
                swanlab_=swanlab_,
            )


def main() -> None:
    cfg = get_grpo_config()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed()
    if dist.is_initialized():
        cfg.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(cfg.save_dir, exist_ok=True)
    base_weight = cfg.from_weight if cfg.from_weight != "none" else ("reason" if cfg.reasoning == 1 else "full_sft")
    lm_config, tokenizer = resolve_grpo_model_config(cfg, base_weight)
    ckp_data = lm_checkpoint(cfg) if cfg.from_resume else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in cfg.device else "cpu"
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4. 配 swanlab ==========
    swanlab_ = None
    if cfg.use_swanlab and is_main_process():
        swanlab_id = ckp_data.get("swanlab_id", None) if ckp_data else None
        resume = "must" if swanlab_id else None
        name = get_swanlab_experiment_name(cfg)
        swanlab.init(project=cfg.swanlab_project, name=name, id=swanlab_id, resume=resume)
        swanlab_ = swanlab

    # ========== 5. 初始化模型、Reward 与数据 ==========
    # Policy 模型
    model, tokenizer = init_model(
        cfg=cfg,
        lm_config=lm_config,
        tokenizer=tokenizer,
        from_weight=base_weight,
    )
    if cfg.use_compile:
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    # Reference 模型
    ref_model, _ = init_model(
        cfg=cfg,
        lm_config=lm_config,
        tokenizer=tokenizer,
        from_weight=base_weight,
    )
    ref_model = ref_model.eval().requires_grad_(False)

    # Reward 模型
    # reward_model = AutoModel.from_pretrained(
    #     cfg.reward_model_path,
    #     torch_dtype=torch.float16,
    #     trust_remote_code=True,
    # )
    reward_model = AutoModel.from_pretrained(cfg.reward_model_name, trust_remote_code=True, torch_dtype=torch.float16)
    reward_model = reward_model.to(cfg.device).eval().requires_grad_(False)
    # reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_path, trust_remote_code=True)
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_tokenizer_name, trust_remote_code=True)

    # 数据与优化器
    train_dataset = RLAIFDataset(cfg.data_path, tokenizer, max_length=cfg.data_max_seq_len)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))
    loader_for_count = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = max(1, ((iters + cfg.accumulation_steps - 1) // cfg.accumulation_steps) * cfg.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=cfg.learning_rate / 10)

    # ========== 6. 从 ckp 恢复状态 ==========
    last_end_epoch, last_end_step = 0, -1
    if ckp_data is not None:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        if "scheduler" in ckp_data:
            scheduler.load_state_dict(ckp_data["scheduler"])
        if "scaler" in ckp_data:
            scaler.load_state_dict(ckp_data["scaler"])
        last_end_epoch = ckp_data.get("epoch", 0)
        last_end_step = ckp_data.get("step", -1)

    # ========== 7. DDP 包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"cos_phi", "sin_phi"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    if ckp_data is not None:
        restore_rng_state(ckp_data)
    train(
        cfg,
        model,
        ref_model,
        reward_model,
        reward_tokenizer,
        tokenizer,
        optimizer,
        scheduler,
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
