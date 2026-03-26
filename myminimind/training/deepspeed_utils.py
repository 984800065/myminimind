from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.distributed as dist

from myminimind.config.schema import PretrainConfig
from myminimind.model.configuration_myminimind import MyMiniMindConfig
from myminimind.utils.logger import logger
from myminimind.utils.train_utils import get_model_variant_suffix


def require_deepspeed():
    try:
        import deepspeed
    except ImportError as exc:
        raise ImportError(
            "当前环境未安装 deepspeed。请先安装后再使用 `--use-deepspeed`，例如："
            " `uv pip install --python .venv/bin/python deepspeed`。"
        ) from exc
    return deepspeed


def _load_json_or_yaml(path: str | Path) -> dict[str, Any]:
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


def build_pretrain_deepspeed_config(cfg: PretrainConfig) -> dict[str, Any]:
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


def resolve_pretrain_deepspeed_config(cfg: PretrainConfig) -> dict[str, Any]:
    if cfg.deepspeed_config:
        ds_config = _load_json_or_yaml(cfg.deepspeed_config)
        logger.info(f"使用外部 DeepSpeed 配置文件: {cfg.deepspeed_config}")
    else:
        ds_config = build_pretrain_deepspeed_config(cfg)
        logger.info(
            "未提供 DeepSpeed 配置文件，已按当前训练参数自动生成配置"
            f"（ZeRO-{cfg.deepspeed_zero_stage}, TP={cfg.deepspeed_tensor_parallel_size}）"
        )

    zero_stage = ds_config.get("zero_optimization", {}).get("stage", cfg.deepspeed_zero_stage)
    if zero_stage not in (0, 1, 2):
        raise ValueError(f"当前预训练入口只支持 DeepSpeed ZeRO stage 0/1/2，收到: {zero_stage}")
    return ds_config


def get_deepspeed_checkpoint_dir(lm_config: MyMiniMindConfig, weight: str, save_dir: str) -> str:
    variant_suffix = get_model_variant_suffix(
        hidden_size=lm_config.hidden_size,
        use_moe=lm_config.use_moe,
        attention_type=getattr(lm_config, "attention_type", "gqa"),
    )
    return f"{save_dir}/{weight}_{variant_suffix}_deepspeed"


def get_deepspeed_runtime_config_path(lm_config: MyMiniMindConfig, weight: str, save_dir: str) -> str:
    variant_suffix = get_model_variant_suffix(
        hidden_size=lm_config.hidden_size,
        use_moe=lm_config.use_moe,
        attention_type=getattr(lm_config, "attention_type", "gqa"),
    )
    return f"{save_dir}/{weight}_{variant_suffix}_deepspeed_config.json"


def save_resolved_deepspeed_config(ds_config: dict[str, Any], lm_config: MyMiniMindConfig, weight: str, save_dir: str) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    path = Path(get_deepspeed_runtime_config_path(lm_config, weight, save_dir))
    path.write_text(json.dumps(ds_config, ensure_ascii=False, indent=2) + "\n")


def get_deepspeed_resume_metadata_path(lm_config: MyMiniMindConfig, weight: str, save_dir: str) -> Path:
    return Path(get_deepspeed_checkpoint_dir(lm_config, weight, save_dir)) / "myminimind_resume.json"


def load_deepspeed_resume_metadata(lm_config: MyMiniMindConfig, weight: str, save_dir: str) -> dict[str, Any] | None:
    meta_path = get_deepspeed_resume_metadata_path(lm_config, weight, save_dir)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def load_deepspeed_engine_checkpoint(engine: Any, lm_config: MyMiniMindConfig, weight: str, save_dir: str) -> dict[str, Any] | None:
    checkpoint_dir = get_deepspeed_checkpoint_dir(lm_config, weight, save_dir)
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    load_path, client_state = engine.load_checkpoint(checkpoint_dir, tag=None)
    if load_path is None:
        return None
    return client_state or {}


def _get_swanlab_id(swanlab_: Any | None) -> str | None:
    if swanlab_ is None:
        return None
    if hasattr(swanlab_, "get_run"):
        run = swanlab_.get_run()
        return getattr(run, "id", None) if run else None
    return getattr(swanlab_, "id", None)


def save_deepspeed_checkpoint(
    engine: Any,
    lm_config: MyMiniMindConfig,
    weight: str,
    save_dir: str,
    epoch: int,
    step: int,
    swanlab_: Any | None = None,
) -> None:
    checkpoint_dir = get_deepspeed_checkpoint_dir(lm_config, weight, save_dir)
    tag = f"epoch{epoch:04d}_step{step:08d}"
    client_state = {
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "swanlab_id": _get_swanlab_id(swanlab_),
    }
    engine.save_checkpoint(checkpoint_dir, tag=tag, client_state=client_state)

    if dist.is_initialized() and dist.get_rank() != 0:
        return
    meta_path = get_deepspeed_resume_metadata_path(lm_config, weight, save_dir)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"latest_tag": tag, **client_state}, ensure_ascii=False, indent=2) + "\n"
    )
