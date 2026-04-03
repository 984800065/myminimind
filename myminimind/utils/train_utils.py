import os
import random
from typing import Any

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer

from myminimind.config.schema import BaseConfig, InferConfig, TrainConfig
from myminimind.model.configuration_myminimind import MyMiniMindConfig
from myminimind.model.modular_myminimind import MyMiniMindForCausalLM
from myminimind.utils.logger import logger


def get_universial_name(cfg: TrainConfig | InferConfig, weight: str | None = None) -> str:
    """
    universial file name without file type suffix

    pattern: <save_weight>_<hidden_size>_moe-<moe_type>_attn-<attention_type>
    """
    save_weight = weight
    if save_weight is None:
        save_weight = cfg.save_weight if isinstance(cfg, TrainConfig) else cfg.weight
    hidden_size = cfg.hidden_size
    moe_type = "common" if cfg.use_moe else "none"
    attention_type = cfg.attention_type

    weight_file_name = f"{save_weight}_{hidden_size}_moe-{moe_type}_attn-{attention_type}"
    return weight_file_name


def get_universial_save_path(cfg: TrainConfig | InferConfig, universial_file_name: str) -> str:
    """
    save_dir pattern: <save_dir>/<universial_file_name>
    """
    save_dir = cfg.save_dir
    return os.path.join(save_dir, universial_file_name)


def get_model_weight_path(
    cfg: TrainConfig | InferConfig,
    *,
    weight: str | None = None,
    include_debug: bool | None = None,
) -> str:
    """
    Build the raw model weight path used by training/eval/export scripts.

    pattern: <save_dir>/<universial_file_name>[_debug].pth
    """
    if include_debug is None:
        include_debug = isinstance(cfg, TrainConfig) and cfg.debug

    weight_file_name = get_universial_name(cfg, weight=weight)

    if include_debug:
        weight_file_name = weight_file_name + "_debug"

    weight_file_name = weight_file_name + ".pth"
    path = get_universial_save_path(cfg, weight_file_name)
    return path


def get_resume_weight_path(cfg: TrainConfig, *, weight: str | None = None) -> str:
    """
    Build the resume checkpoint path used by non-DeepSpeed training.

    We intentionally keep model weights and resume payload separate:
    - `*.pth`: raw model parameters, convenient for inference/export
    - `*_resume.pth`: optimizer/scaler/scheduler/step metadata for training resume
    """
    resume_file_name = get_universial_name(cfg, weight=weight)
    if cfg.debug:
        resume_file_name = resume_file_name + "_debug"
    resume_file_name = resume_file_name + "_resume.pth"
    return get_universial_save_path(cfg, resume_file_name)


def init_distributed(use_deepspeed: bool = False) -> int:
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    if use_deepspeed:
        deepspeed.init_distributed(dist_backend="nccl")
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)

    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _unwrap_stateful_object(value: Any) -> Any:
    """
    Unwrap DDP / compiled wrappers before reading `.state_dict()`.

    The training code may hand us:
    - plain modules / optimizers
    - `DistributedDataParallel` wrappers
    - compiled modules exposing `_orig_mod`
    """
    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
    return getattr(raw_value, "_orig_mod", raw_value)


def _to_cpu_state_dict(value: Any) -> dict[str, Any]:
    """
    Convert a model state dict to CPU tensors so checkpoints are device-agnostic.

    We also cast floating-point tensors to fp16 to keep raw weight files smaller,
    which matches the project's previous behavior.
    """
    raw_value = _unwrap_stateful_object(value)
    state_dict = raw_value.state_dict()
    return {key: tensor.half().cpu() for key, tensor in state_dict.items()}


def _serialize_checkpoint_value(value: Any) -> Any:
    """
    Serialize a checkpoint field.

    - Objects with `state_dict()` are saved as their state dict
    - Plain python values are saved as-is
    """
    if hasattr(value, "state_dict"):
        return _unwrap_stateful_object(value).state_dict()
    return value


def _atomic_torch_save(payload: Any, path: str) -> None:
    """Write a checkpoint atomically to avoid partial files on interruption."""
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _get_swanlab_id(swanlab_: Any | None) -> str | None:
    """Extract the current swanlab run id when available."""
    if swanlab_ is None:
        return None
    if hasattr(swanlab_, "get_run"):
        run = swanlab_.get_run()
        return getattr(run, "id", None) if run else None
    return getattr(swanlab_, "id", None)


def lm_checkpoint(
    cfg: TrainConfig,
    model: Any | None = None,
    optimizer: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    swanlab_: Any | None = None,
    **extra_state: Any,
) -> dict | None:
    """
    Save or load a resumable non-DeepSpeed checkpoint.

    Save mode:
    - Triggered when `model` is not None
    - Writes both raw model weights (`*.pth`) and the resume payload (`*_resume.pth`)

    Load mode:
    - Triggered when `model` is None
    - Returns the resume payload if it exists, otherwise returns None
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    weight_path = get_model_weight_path(cfg)
    resume_path = get_resume_weight_path(cfg)

    if model is None:
        if not os.path.exists(resume_path):
            return None

        checkpoint_data = torch.load(resume_path, map_location="cpu")
        saved_world_size = checkpoint_data.get("world_size", 1)
        current_world_size = dist.get_world_size() if dist.is_initialized() else 1
        if saved_world_size != current_world_size:
            checkpoint_data["step"] = checkpoint_data["step"] * saved_world_size // current_world_size
            logger.warning(f"GPU数量变化({saved_world_size}→{current_world_size})，step已自动转换为{checkpoint_data['step']}")
        return checkpoint_data

    model_state_dict = _to_cpu_state_dict(model)
    _atomic_torch_save(model_state_dict, weight_path)

    resume_data: dict[str, Any] = {
        "model": model_state_dict,
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "swanlab_id": _get_swanlab_id(swanlab_),
    }
    if optimizer is not None:
        resume_data["optimizer"] = _serialize_checkpoint_value(optimizer)

    for key, value in extra_state.items():
        if value is not None:
            resume_data[key] = _serialize_checkpoint_value(value)

    _atomic_torch_save(resume_data, resume_path)
    torch.cuda.empty_cache()
    return None


def log_swanlab_training_metrics(
    swanlab_run: Any | None,
    *,
    epoch: int,
    step: int,
    steps_per_epoch: int,
    total_epochs: int,
    learning_rate: float,
    elapsed_seconds: float,
    eta_minutes: float,
    train_metrics: dict[str, float],
) -> None:
    """
    Log training metrics to SwanLab with explicit units in metric names.

    Why this helper exists:
    - SwanLab 0.7.13 exposes `log(step=...)` but does not provide a local
      `define_metric(unit=...)`-style API.
    - To make charts easier to read, we put units directly into metric names,
      and we always log with an explicit monotonically increasing global step.

    Resulting chart groups look like:
    - `train.loss`
    - `progress.epoch_progress (%)`
    - `progress.global_step (step)`
    - `time.eta (min)`
    - `optimizer.learning_rate`
    """
    if swanlab_run is None:
        return

    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}.")
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be > 0, got {total_epochs}.")

    # Use a 1-based global step so the x-axis is more human-friendly and never
    # reuses step 0 after resuming a run mid-epoch.
    global_step = epoch * steps_per_epoch + step + 1
    epoch_step = step + 1

    # Percentage-style metrics are stored in [0, 100], and the unit is written
    # into the metric name so the SwanLab chart title itself carries the unit.
    epoch_progress_pct = 100.0 * epoch_step / steps_per_epoch
    total_training_steps = total_epochs * steps_per_epoch
    training_progress_pct = 100.0 * global_step / total_training_steps

    swanlab_run.log(
        {
            "train": train_metrics,
            "progress": {
                "epoch (idx)": epoch + 1,
                "epoch_step (step)": epoch_step,
                "global_step (step)": global_step,
                "epoch_progress (%)": epoch_progress_pct,
                "training_progress (%)": min(training_progress_pct, 100.0),
            },
            "time": {
                "elapsed (s)": elapsed_seconds,
                "eta (min)": eta_minutes,
            },
            "optimizer": {
                "learning_rate": learning_rate,
            },
        },
        step=global_step,
    )


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_model_params(model: MyMiniMindForCausalLM, config: MyMiniMindConfig):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "num_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_token", 0)
    n_shared = getattr(config, "num_shared_experts", 0)
    attention_type = getattr(config, "attention_type", "gqa")
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.shared_experts.0." in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        logger.info(f"Model Params: {total:.2f}M-A{active:.2f}M ({attention_type})")
    else:
        logger.info(f"Model Params: {total:.2f}M ({attention_type})")


def load_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer {tokenizer_path!r} 缺少 pad_token 和 eos_token，当前训练入口无法自动补齐。")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def sync_lm_config_with_tokenizer(lm_config: MyMiniMindConfig, tokenizer: AutoTokenizer) -> MyMiniMindConfig:
    lm_config.vocab_size = len(tokenizer)
    lm_config.bos_token_id = tokenizer.bos_token_id
    lm_config.eos_token_id = tokenizer.eos_token_id
    lm_config.pad_token_id = tokenizer.pad_token_id
    return lm_config


def resolve_lm_config_and_tokenizer(cfg: BaseConfig) -> tuple[MyMiniMindConfig, AutoTokenizer]:
    """Build the LM config from training/infer config and sync tokenizer metadata."""
    lm_config_kwargs: dict = cfg.to_lm_config_kwargs()
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    lm_config = MyMiniMindConfig(**lm_config_kwargs)
    sync_lm_config_with_tokenizer(lm_config, tokenizer)
    return lm_config, tokenizer


def init_model(
    cfg: TrainConfig,
    lm_config: MyMiniMindConfig,
    tokenizer: AutoTokenizer | None = None,
    *,
    from_weight: str | None = None,
) -> tuple[MyMiniMindForCausalLM, AutoTokenizer]:
    """
    Initialize a trainable model and optionally load raw `.pth` weights.

    `from_weight` allows training scripts such as DPO/GRPO to override the
    source checkpoint name without mutating `cfg.save_weight`.
    """
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(cfg.tokenizer_path)
    model = MyMiniMindForCausalLM(lm_config)

    source_weight = cfg.from_weight if from_weight is None else from_weight
    if source_weight != "none":
        weight_path = get_model_weight_path(cfg, weight=source_weight, include_debug=False)
        weights: dict = torch.load(weight_path, map_location=cfg.device)

        ignore_keys = {
            "model.position_embeddings.cos_phi",
            "model.position_embeddings.sin_phi",
        }

        for k in list(weights.keys()):
            if k in ignore_keys:
                weights.pop(k)

        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    logger.info(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")
    return model.to(cfg.device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, skip_batches: int = 0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    batch = []
                    skipped += 1
                else:
                    yield batch
                    batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
