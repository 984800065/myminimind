import os
import random
from pathlib import Path
from typing import Any

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer

from mini_deepseek.config.schema import BaseConfig, InferConfig, TrainConfig
from mini_deepseek.model.configuration_mini_deepseek import MiniDeepSeekConfig
from mini_deepseek.model.modular_mini_deepseek import MiniDeepSeekForCausalLM
from mini_deepseek.utils.logger import logger

PROJECT_MODEL_NAME = "mini_deepseek"


def get_universal_name(cfg: TrainConfig | InferConfig, weight: str | None = None) -> str:
    """
    universal file name without file type suffix

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


def get_universal_save_path(cfg: TrainConfig | InferConfig, universal_file_name: str) -> str:
    """
    save_dir pattern: <save_dir>/<universal_file_name>
    """
    save_dir = cfg.save_dir
    return os.path.join(save_dir, universal_file_name)


def build_swanlab_experiment_name(
    *,
    project_model_name: str,
    universal_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> str:
    """Build a consistent SwanLab run name for all training entrypoints."""

    return (
        f"{project_model_name}_{universal_name}"
        f"_E{epochs}"
        f"_B{batch_size}"
        f"_LR{learning_rate}"
    )


def get_swanlab_experiment_name(
    cfg: TrainConfig,
    *,
    project_model_name: str = PROJECT_MODEL_NAME,
    weight: str | None = None,
) -> str:
    """Build the SwanLab experiment name from the shared project naming rules."""

    universal_name = get_universal_name(cfg, weight=weight)
    return build_swanlab_experiment_name(
        project_model_name=project_model_name,
        universal_name=universal_name,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
    )


def get_model_weight_path(
    cfg: TrainConfig | InferConfig,
    *,
    weight: str | None = None,
    include_debug: bool | None = None,
) -> str:
    """
    Build the raw model weight path used by training/eval/export scripts.

    pattern: <save_dir>/<universal_file_name>[_debug].pth
    """
    if include_debug is None:
        include_debug = isinstance(cfg, TrainConfig) and cfg.debug

    weight_file_name = get_universal_name(cfg, weight=weight)

    if include_debug:
        weight_file_name = weight_file_name + "_debug"

    weight_file_name = weight_file_name + ".pth"
    path = get_universal_save_path(cfg, weight_file_name)
    return path


def resolve_model_weight_path(
    cfg: TrainConfig | InferConfig,
    *,
    weight: str | None = None,
    include_debug: bool | None = None,
) -> str:
    """
    Resolve a raw checkpoint even when caller-side architecture defaults differ.

    The exact structured name remains preferred. If it does not exist, search by
    the human-selected weight prefix. A unique match is safe because its adjacent
    config sidecar supplies the actual architecture; multiple matches require the
    caller to disambiguate explicitly.
    """
    exact_path = Path(
        get_model_weight_path(
            cfg,
            weight=weight,
            include_debug=include_debug,
        )
    )
    if exact_path.exists():
        return str(exact_path)

    weight_prefix = weight
    if weight_prefix is None:
        weight_prefix = cfg.save_weight if isinstance(cfg, TrainConfig) else cfg.weight
    candidates = sorted(
        path
        for path in Path(cfg.save_dir).glob(
            f"{weight_prefix}_*_moe-*_attn-*.pth"
        )
        if not path.name.endswith("_resume.pth")
    )
    if len(candidates) == 1:
        logger.info(f"按权重前缀自动定位 checkpoint: {candidates[0]}")
        return str(candidates[0])
    if len(candidates) > 1:
        candidate_text = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(
            f"权重前缀 {weight_prefix!r} 匹配到多个 checkpoint，请显式指定结构或路径:\n"
            f"{candidate_text}"
        )
    return str(exact_path)


def get_resume_weight_path(cfg: TrainConfig, *, weight: str | None = None) -> str:
    """
    Build the resume checkpoint path used by non-DeepSpeed training.

    We intentionally keep model weights and resume payload separate:
    - `*.pth`: raw model parameters, convenient for inference/export
    - `*_resume.pth`: optimizer/scaler/scheduler/step metadata for training resume
    """
    resume_file_name = get_universal_name(cfg, weight=weight)
    if cfg.debug:
        resume_file_name = resume_file_name + "_debug"
    resume_file_name = resume_file_name + "_resume.pth"
    return get_universal_save_path(cfg, resume_file_name)


def get_model_config_path(
    cfg: TrainConfig | InferConfig,
    *,
    weight: str | None = None,
    include_debug: bool | None = None,
) -> str:
    """
    Return the architecture sidecar path associated with a raw weight file.

    A raw PyTorch state dict does not contain model hyperparameters. Keeping a
    same-stem JSON sidecar prevents inference/export from accidentally rebuilding
    an MLA checkpoint as GQA, or dropping MoE/MTP layers because infer defaults
    changed after training.
    """
    weight_path = get_model_weight_path(
        cfg,
        weight=weight,
        include_debug=include_debug,
    )
    return str(Path(weight_path).with_suffix(".config.json"))


def save_model_config(
    cfg: TrainConfig | InferConfig,
    lm_config: MiniDeepSeekConfig,
    *,
    weight: str | None = None,
    include_debug: bool | None = None,
) -> str:
    """Atomically save the model architecture next to its raw state dict."""
    config_path = get_model_config_path(
        cfg,
        weight=weight,
        include_debug=include_debug,
    )
    tmp_path = config_path + ".tmp"
    Path(tmp_path).write_text(lm_config.to_json_string(use_diff=False))
    os.replace(tmp_path, config_path)
    return config_path


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


def setup_seed(seed: int) -> None:
    """
    统一设置训练过程中使用的随机数种子，尽量保证实验可复现。

    这里覆盖四类随机数来源：
    - Python `random`：控制 Python 代码中的随机采样。
    - NumPy：控制数据预处理等 NumPy 操作的随机行为。
    - PyTorch CPU：控制 CPU tensor 初始化和随机算子。
    - PyTorch CUDA：控制当前 GPU 及所有可见 GPU 上的随机算子。

    cuDNN 的 deterministic 模式优先选择确定性算法，关闭 benchmark 则避免
    cuDNN 根据运行时性能测试动态选择算法。这会提高相同环境下的可复现性，
    但可能降低训练速度；部分 GPU 算子仍可能存在非确定性实现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 同时设置当前 CUDA 设备和所有可见 CUDA 设备，兼容单卡与多卡训练。
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 固定 cuDNN 算法选择，避免相同输入在不同运行中选到不同实现。
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


def _to_cpu_state_dict(
    value: Any,
    *,
    cast_floating_to_fp16: bool,
) -> dict[str, Any]:
    """
    Convert a model state dict to CPU tensors so checkpoints are device-agnostic.

    Raw inference weights may be cast to fp16 for size. Resume checkpoints keep
    original precision so an interruption does not perturb optimizer training.
    """
    raw_value = _unwrap_stateful_object(value)
    state_dict = raw_value.state_dict()
    cpu_state_dict: dict[str, Any] = {}
    for key, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu()
        if cast_floating_to_fp16 and torch.is_floating_point(cpu_tensor):
            cpu_tensor = cpu_tensor.half()
        cpu_state_dict[key] = cpu_tensor
    return cpu_state_dict


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


def _capture_rng_state() -> dict[str, Any]:
    """Capture RNG streams needed to continue dropout/sampling exactly."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint_data: dict[str, Any]) -> None:
    """Restore RNG state from a resume payload when the fields are available."""
    state = checkpoint_data.get("rng_state")
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def lm_checkpoint(
    cfg: TrainConfig,
    model: Any | None = None,
    optimizer: Any | None = None,
    lm_config: MiniDeepSeekConfig | None = None,
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

    raw_model_state_dict = _to_cpu_state_dict(
        model,
        cast_floating_to_fp16=True,
    )
    _atomic_torch_save(raw_model_state_dict, weight_path)
    if lm_config is not None:
        save_model_config(cfg, lm_config)

    resume_data: dict[str, Any] = {
        "model": _to_cpu_state_dict(model, cast_floating_to_fp16=False),
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "swanlab_id": _get_swanlab_id(swanlab_),
        "rng_state": _capture_rng_state(),
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


def get_model_params(model: MiniDeepSeekForCausalLM, config: MiniDeepSeekConfig):
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


def sync_lm_config_with_tokenizer(lm_config: MiniDeepSeekConfig, tokenizer: AutoTokenizer) -> MiniDeepSeekConfig:
    lm_config.vocab_size = len(tokenizer)
    lm_config.bos_token_id = tokenizer.bos_token_id
    lm_config.eos_token_id = tokenizer.eos_token_id
    lm_config.pad_token_id = tokenizer.pad_token_id
    return lm_config


def resolve_lm_config_and_tokenizer(cfg: BaseConfig) -> tuple[MiniDeepSeekConfig, AutoTokenizer]:
    """Build the LM config from training/infer config and sync tokenizer metadata."""
    lm_config_kwargs: dict = cfg.to_lm_config_kwargs()
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    lm_config = MiniDeepSeekConfig(**lm_config_kwargs)
    sync_lm_config_with_tokenizer(lm_config, tokenizer)
    return lm_config, tokenizer


def init_model(
    cfg: TrainConfig,
    lm_config: MiniDeepSeekConfig,
    tokenizer: AutoTokenizer | None = None,
    *,
    from_weight: str | None = None,
) -> tuple[MiniDeepSeekForCausalLM, AutoTokenizer]:
    """
    Initialize a trainable model and optionally load raw `.pth` weights.

    `from_weight` allows training scripts such as DPO/GRPO to override the
    source checkpoint name without mutating `cfg.save_weight`.
    """
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(cfg.tokenizer_path)
    model = MiniDeepSeekForCausalLM(lm_config)

    source_weight = cfg.from_weight if from_weight is None else from_weight
    if source_weight != "none":
        weight_path = resolve_model_weight_path(
            cfg,
            weight=source_weight,
            include_debug=False,
        )
        weights: dict = torch.load(weight_path, map_location=cfg.device)

        ignore_keys = {
            "model.position_embeddings.cos_phi",
            "model.position_embeddings.sin_phi",
        }

        for k in list(weights.keys()):
            if k in ignore_keys:
                weights.pop(k)

        incompatible = model.load_state_dict(weights, strict=False)
        # Fine-tuning may intentionally add heads/layers, but silently ignoring
        # a large architecture mismatch makes it look as if pretrained weights
        # were loaded when most of the model is random.
        if incompatible.missing_keys:
            logger.warning(
                f"加载 {weight_path} 时缺少 {len(incompatible.missing_keys)} 个参数键: "
                f"{incompatible.missing_keys[:8]}"
            )
        if incompatible.unexpected_keys:
            logger.warning(
                f"加载 {weight_path} 时发现 {len(incompatible.unexpected_keys)} 个多余参数键: "
                f"{incompatible.unexpected_keys[:8]}"
            )

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
