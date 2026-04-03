"""
MiniMind GRPO 训练入口：get_grpo_config() 加载参数，DDP + 混合精度 + swanlab。
"""

import os
import re
from contextlib import nullcontext

import swanlab
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from myminimind.config import GRPOConfig, get_grpo_config
from myminimind.data import RLAIFDataset
from myminimind.model.configuration_myminimind import MyMiniMindConfig as MiniMindConfig
from myminimind.model.modeling_myminimind import MyMiniMindForCausalLM as MiniMindForCausalLM
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
    model: MiniMindForCausalLM,
    ref_model: MiniMindForCausalLM,
    reward_model,
    reward_tokenizer,
    tokenizer: AutoTokenizer,
    optimizer: optim.AdamW,
    scheduler: CosineAnnealingLR,
    autocast_ctx,
    lm_config: MiniMindConfig,
    zero_based_start_step: int = 0,
    swanlab_: swanlab.Run | None = None,
) -> None:
    model.train()

    pbar = tqdm(loader, total=total_iters, initial=zero_based_start_step, desc=f"Epoch[{epoch + 1}/{cfg.epochs}]", leave=True)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, start=zero_based_start_step):
        # len(prompts) == batch_size
        prompts: list[str] = batch["prompt"]
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False, padding_side="left", add_special_tokens=False).to(cfg.device)

        if cfg.max_seq_len:
            # (batch_size, max_seq_len)
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -cfg.max_seq_len :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -cfg.max_seq_len :]

        with torch.no_grad():
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # (batch_size * num_generations, prompt_len + response_len)
            outputs = model_for_gen.generate(**prompt_inputs, max_new_tokens=cfg.max_gen_len, do_sample=True, temperature=0.8, num_return_sequences=cfg.num_generations, pad_token_id=tokenizer.pad_token_id)

        # (batch_size * num_generations, response_len)
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1) :]

        def get_per_tokne_logps(
            model: MiniMindForCausalLM,
            input_ids: torch.Tensor,
            n_keep: int,
        ):
            # (batch_size * num_generations, total_seq_len)
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            # (batch_size * num_generations, n_keep, vocab_size)
            logits = model(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:], strict=True):
                # (n_keep, vocab_size)
                logits_row: torch.Tensor
                # (n_keep, )
                ids_row: torch.Tensor
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # (n_keep, )
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), dim=-1, index=ids_row[:, None]).squeeze(1))

            # (batch_size * num_generations, n_keep)
            return torch.stack(per_token_logps)

        with autocast_ctx:
            # completion_ids.shape == (batch_size * num_generations, response_len)
            # (batch_size * num_generations, response_len)
            per_token_logps = get_per_tokne_logps(model, outputs, completion_ids.size(1))
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=cfg.device)

        with torch.no_grad():
            # (batch_size * num_generations, response_len)
            ref_per_token_logps = get_per_tokne_logps(ref_model, outputs, completion_ids.size(1))

        # len(completions) == batch_size * num_generations
        completions: list[str] = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # (batch_size * num_generations, )
        rewards: torch.Tensor = calculate_rewards(cfg, prompts, completions, reward_model, reward_tokenizer).to(cfg.device)

        # (batch_size, num_generations)
        grouped_rewards = rewards.reshape(-1, cfg.num_generations)
        # (batch_size * num_generations, )
        inner_batch_mean_reward = grouped_rewards.mean(dim=1).repeat_interleave(cfg.num_generations)
        # (batch_size * num_generations, )
        inner_batch_std_reward = grouped_rewards.std(dim=1).repeat_interleave(cfg.num_generations)
        # (batch_size * num_generations, )
        advantages = torch.clamp((rewards - inner_batch_mean_reward) / (inner_batch_std_reward + 1e-8), min=-10, max=10)
        # (batch_size * num_generations, )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # (batch_size * num_generations, response_len), dtype == torch.bool
        is_eos: torch.Tensor = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=cfg.device)
        # 找每条生成序列里第一个 EOS token 的位置
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # (batch_size * num_generations, response_len), dtype == torch.int
        completion_mask = (torch.arange(is_eos.size(1), device=cfg.device) < eos_idx[:, None]).int()

        # (batch_size * num_generations, response_len)
        kl_div = ref_per_token_logps - per_token_logps
        # (batch_size * num_generations, response_len)
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # (batch_size * num_generations, response_len)
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages[:, None] - cfg.beta * per_token_kl)
        # \frac{1}{T} * \frac{1}{|o|} * \sum per_token_loss
        # ()
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        loss = (policy_loss + aux_loss) / cfg.accumulation_steps
        loss.backward()

        should_update = ((step + 1) % cfg.accumulation_steps == 0) or (step == total_iters - 1)
        if should_update:
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_interval == 0 or step == total_iters - 1:
            policy_loss_val = loss.item() * cfg.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
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

        if (step % cfg.save_interval == 0 or step == total_iters - 1) and is_main_process():
            model.eval()
            lm_checkpoint(
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                scheduler=scheduler,
            )
            model.train()

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, inner_batch_mean_reward, inner_batch_std_reward, advantages, completion_mask


def train(
    cfg: GRPOConfig,
    model: MiniMindForCausalLM,
    ref_model: MiniMindForCausalLM,
    reward_model,
    reward_tokenizer,
    tokenizer: AutoTokenizer,
    optimizer: optim.AdamW,
    scheduler: CosineAnnealingLR,
    autocast_ctx,
    lm_config: MiniMindConfig,
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
    lm_config, tokenizer = resolve_lm_config_and_tokenizer(cfg)
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
        model_name = f"MiniMind{cfg.hidden_size}{'_moe' if cfg.use_moe else ''}"
        name = f"{model_name}-GRPO-E{cfg.epochs}-B{cfg.batch_size}-LR{cfg.learning_rate}"
        swanlab.init(project=cfg.swanlab_project, name=name, id=swanlab_id, resume=resume)
        swanlab_ = swanlab

    # ========== 5. 初始化模型、Reward 与数据 ==========
    base_weight = cfg.from_weight if cfg.from_weight != "none" else ("reason" if cfg.reasoning == 1 else "full_sft")
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
    train_dataset = RLAIFDataset(cfg.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
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
        last_end_epoch = ckp_data.get("epoch", 0)
        last_end_step = ckp_data.get("step", -1)

    # ========== 7. DDP 包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"cos_phi", "sin_phi"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    train(
        cfg,
        model,
        ref_model,
        reward_model,
        reward_tokenizer,
        tokenizer,
        optimizer,
        scheduler,
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
