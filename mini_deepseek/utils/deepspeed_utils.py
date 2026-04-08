from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.distributed as dist

from mini_deepseek.config.schema import TrainConfig
from mini_deepseek.utils.logger import logger
from mini_deepseek.utils.train_utils import get_universal_name


def _load_json_or_yaml(path: str | Path) -> dict[str, Any]:
    """Load a DeepSpeed config file from json/yaml."""
    config_path = Path(path)
    text = config_path.read_text()
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("读取 YAML 格式的 DeepSpeed 配置需要先安装 pyyaml。") from exc
        return yaml.safe_load(text) or {}
    raise ValueError(f"不支持的 DeepSpeed 配置文件格式: {config_path}")


def get_deepspeed_checkpoint_dir(cfg: TrainConfig) -> Path:
    """
    Return the root directory used by `engine.save_checkpoint(...)`.

    We derive the path only from the training config so callers do not need to
    manually pass `save_dir`, `save_weight`, `hidden_size`, `attention_type` and
    other naming pieces around.
    """
    checkpoint_name = f"{get_universal_name(cfg)}_deepspeed"
    return Path(cfg.save_dir) / checkpoint_name


def get_deepspeed_runtime_config_path(cfg: TrainConfig) -> Path:
    """
    Save the resolved DeepSpeed runtime config next to checkpoints.

    This file is only for inspection/debugging; the real training state is still
    owned by DeepSpeed checkpoints.
    """
    config_name = f"{get_universal_name(cfg)}_deepspeed_config.json"
    return Path(cfg.save_dir) / config_name


def get_deepspeed_resume_metadata_path(cfg: TrainConfig) -> Path:
    """
    Path of a tiny sidecar json used for quick resume metadata lookup.

    DeepSpeed itself stores the true checkpoint data. This file only caches
    fields such as `latest_tag`, `epoch`, `step`, and `swanlab_id`.
    """
    return get_deepspeed_checkpoint_dir(cfg) / "mini_deepseek_resume.json"


def build_deepspeed_config(cfg: TrainConfig) -> dict[str, Any]:
    """
    Build a minimal DeepSpeed config from training arguments.

    This path is used when the user does not provide an external DeepSpeed
    config file.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": cfg.batch_size,
        "gradient_accumulation_steps": cfg.accumulation_steps,
        "train_batch_size": cfg.batch_size * cfg.accumulation_steps * world_size,
        "gradient_clipping": cfg.grad_clip,
        "zero_optimization": {
            "stage": cfg.deepspeed_zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
        },
        "bf16": {"enabled": cfg.dtype == "bfloat16"},
        "fp16": {"enabled": cfg.dtype == "float16"},
    }
    if cfg.deepspeed_offload_optimizer:
        ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if cfg.deepspeed_tensor_parallel_size > 1:
        ds_config["tensor_parallel"] = {"autotp_size": cfg.deepspeed_tensor_parallel_size}
    return ds_config


def resolve_deepspeed_config(cfg: TrainConfig) -> dict[str, Any]:
    """
    Resolve the final DeepSpeed config used by the current run.

    Priority:
    1. User-provided config file
    2. Auto-generated config from current training args
    """
    if cfg.deepspeed_config:
        ds_config = _load_json_or_yaml(cfg.deepspeed_config)
        logger.info(f"使用外部 DeepSpeed 配置文件: {cfg.deepspeed_config}")
    else:
        ds_config = build_deepspeed_config(cfg)
        logger.info(
            "未提供 DeepSpeed 配置文件，已按当前训练参数自动生成配置"
            f"（ZeRO-{cfg.deepspeed_zero_stage}, TP={cfg.deepspeed_tensor_parallel_size}）"
        )

    zero_stage = ds_config.get("zero_optimization", {}).get("stage", cfg.deepspeed_zero_stage)
    if zero_stage not in (0, 1, 2):
        raise ValueError(f"当前预训练入口只支持 DeepSpeed ZeRO stage 0/1/2，收到: {zero_stage}")
    return ds_config


def save_resolved_deepspeed_config(ds_config: dict[str, Any], cfg: TrainConfig) -> None:
    """
    Save the resolved runtime DeepSpeed config for debugging and reproducibility.

    Only rank 0 writes this file to avoid multiple processes racing on the same
    output path.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    path = get_deepspeed_runtime_config_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ds_config, ensure_ascii=False, indent=2) + "\n")


def load_deepspeed_resume_metadata(cfg: TrainConfig) -> dict[str, Any] | None:
    """
    Load the lightweight resume metadata json if it exists.

    This is intentionally cheap: callers can inspect the last saved tag/step
    before the DeepSpeed engine is even initialized.
    """
    meta_path = get_deepspeed_resume_metadata_path(cfg)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def load_deepspeed_engine_checkpoint(engine: Any, cfg: TrainConfig) -> dict[str, Any] | None:
    """
    Ask the DeepSpeed engine to restore the latest training checkpoint.

    The returned `client_state` is the extra metadata we attached when saving,
    for example epoch/step/swanlab_id.
    """
    checkpoint_dir = get_deepspeed_checkpoint_dir(cfg)
    if not checkpoint_dir.exists():
        return None

    load_path, client_state = engine.load_checkpoint(str(checkpoint_dir), tag=None)
    if load_path is None:
        return None
    return client_state or {}


def _get_swanlab_id(swanlab_: Any | None) -> str | None:
    """Extract the current swanlab run id when available."""
    if swanlab_ is None:
        return None
    if hasattr(swanlab_, "get_run"):
        run = swanlab_.get_run()
        return getattr(run, "id", None) if run else None
    return getattr(swanlab_, "id", None)


def save_deepspeed_checkpoint(
    engine: Any,
    epoch: int,
    step: int,
    cfg: TrainConfig,
    swanlab_: Any | None = None,
) -> None:
    """
    Save a resumable DeepSpeed checkpoint plus a small json sidecar.

    Notes:
    - The real training state is saved by DeepSpeed itself.
    - The json sidecar only helps this project quickly find the latest tag and
      recover high-level metadata before the engine is restored.
    """
    checkpoint_dir = get_deepspeed_checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tag = f"epoch{epoch:04d}_step{step:08d}"
    client_state = {
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "swanlab_id": _get_swanlab_id(swanlab_),
    }

    # Let DeepSpeed handle the real checkpoint content: model shards, optimizer
    # state, ZeRO partition info, and the attached `client_state`.
    engine.save_checkpoint(str(checkpoint_dir), tag=tag, client_state=client_state)

    # Only rank 0 writes the human-readable sidecar file.
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    meta_path = get_deepspeed_resume_metadata_path(cfg)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"latest_tag": tag, **client_state}, ensure_ascii=False, indent=2) + "\n"
    )
