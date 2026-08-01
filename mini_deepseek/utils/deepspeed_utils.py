from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.distributed as dist

from mini_deepseek.config.schema import TrainConfig
from mini_deepseek.utils.logger import logger
from mini_deepseek.utils.train_utils import get_universal_name


def _load_json_or_yaml(path: str | Path) -> dict[str, Any]:
    """
    从 JSON 或 YAML 文件加载 DeepSpeed 配置。

    Args:
        path: DeepSpeed 配置文件路径。

    Returns:
        解析后的 DeepSpeed 配置字典；空 YAML 文件返回空字典。

    Raises:
        ImportError: 读取 YAML 配置但当前环境未安装 `pyyaml`。
        ValueError: 配置文件扩展名不是 `.json`、`.yaml` 或 `.yml`。
    """
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
    构造 DeepSpeed 分片 checkpoint 的根目录。

    目录名称由权重名称和模型结构统一生成，供 `engine.save_checkpoint()` 和
    `engine.load_checkpoint()` 共用。

    Args:
        cfg: 训练配置，提供保存目录、权重名称和模型结构信息。

    Returns:
        DeepSpeed checkpoint 根目录。
    """
    checkpoint_name = f"{get_universal_name(cfg)}_deepspeed"
    return Path(cfg.save_dir) / checkpoint_name


def get_deepspeed_runtime_config_path(cfg: TrainConfig) -> Path:
    """
    构造最终 DeepSpeed 运行配置的保存路径。

    该 JSON 文件只用于检查实际生效的配置和复现实验，不包含模型、优化器或
    ZeRO 分片等训练状态。

    Args:
        cfg: 训练配置，提供保存目录和统一模型名称。

    Returns:
        最终 DeepSpeed 运行配置的 JSON 文件路径。
    """
    config_name = f"{get_universal_name(cfg)}_deepspeed_config.json"
    return Path(cfg.save_dir) / config_name


def get_deepspeed_resume_metadata_path(cfg: TrainConfig) -> Path:
    """
    构造 DeepSpeed 续训元数据 sidecar 的路径。

    sidecar 仅缓存 `latest_tag`、`epoch`、`step` 和 `swanlab_id` 等轻量信息；
    模型、优化器和 ZeRO 分片等真实训练状态仍由 DeepSpeed checkpoint 保存。

    Args:
        cfg: 训练配置，提供 DeepSpeed checkpoint 根目录信息。

    Returns:
        续训元数据 JSON 文件路径。
    """
    return get_deepspeed_checkpoint_dir(cfg) / "mini_deepseek_resume.json"


def build_deepspeed_config(cfg: TrainConfig) -> dict[str, Any]:
    """
    根据训练参数生成最小可用的 DeepSpeed 配置。

    配置包含 batch size、梯度累积、梯度裁剪、ZeRO、混合精度，以及可选的
    optimizer offload 和 tensor parallel。全局 batch size 会乘以当前
    `world_size`。

    Args:
        cfg: 训练配置，提供 DeepSpeed 和训练超参数。

    Returns:
        可传给 `deepspeed.initialize()` 的配置字典。
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


def _complete_and_validate_shared_config(ds_config: dict[str, Any], cfg: TrainConfig) -> None:
    """
    补齐并校验项目配置与外部 DeepSpeed 配置共用的训练字段。

    外部配置缺少公共字段时使用项目配置补齐；显式提供但值不一致时集中报错，
    防止 DataLoader、scheduler 和 DeepSpeed engine 使用不同的训练语义。
    ZeRO、offload 和通信参数等 DeepSpeed 专属字段不在此处修改。

    Args:
        ds_config: 从外部 JSON 或 YAML 加载的 DeepSpeed 配置。
        cfg: 项目训练配置，作为公共训练字段的唯一基准。

    Returns:
        None。

    Raises:
        ValueError: 外部配置中的公共训练字段与项目配置不一致。
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    expected_values = {
        "train_micro_batch_size_per_gpu": cfg.batch_size,
        "gradient_accumulation_steps": cfg.accumulation_steps,
        "train_batch_size": cfg.batch_size * cfg.accumulation_steps * world_size,
        "gradient_clipping": cfg.grad_clip,
    }
    conflicts: list[str] = []
    for key, expected in expected_values.items():
        if key not in ds_config:
            ds_config[key] = expected
        elif ds_config[key] != expected:
            conflicts.append(f"{key}: 外部配置={ds_config[key]!r}, 项目配置={expected!r}")

    expected_precision = {
        "bf16": cfg.dtype == "bfloat16",
        "fp16": cfg.dtype == "float16",
    }
    for section_name, expected_enabled in expected_precision.items():
        section = ds_config.setdefault(section_name, {})
        if not isinstance(section, dict):
            conflicts.append(f"{section_name}: 外部配置必须是对象，实际为 {section!r}")
            continue
        if "enabled" not in section:
            section["enabled"] = expected_enabled
        elif section["enabled"] != expected_enabled:
            conflicts.append(
                f"{section_name}.enabled: 外部配置={section['enabled']!r}, "
                f"项目配置={expected_enabled!r}"
            )

    if conflicts:
        conflict_text = "\n  - ".join(conflicts)
        raise ValueError(
            "外部 DeepSpeed 配置与项目训练配置存在冲突。公共训练字段必须由项目配置统一控制："
            f"\n  - {conflict_text}"
        )


def resolve_deepspeed_config(cfg: TrainConfig) -> dict[str, Any]:
    """
    解析当前训练实际使用的 DeepSpeed 配置。

    优先读取 `cfg.deepspeed_config` 指定的外部文件；未指定时根据训练参数自动
    生成。外部配置缺少公共训练字段时从项目配置补齐，显式冲突时直接报错。
    当前训练入口只允许 ZeRO stage 0、1 或 2。

    Args:
        cfg: 训练配置，提供外部配置路径和自动生成配置所需参数。

    Returns:
        当前训练最终使用的 DeepSpeed 配置字典。

    Raises:
        ImportError: 外部配置为 YAML，但环境中未安装 `pyyaml`。
        ValueError: 外部配置格式不受支持、公共训练字段冲突，或 ZeRO stage
            不是 0、1、2。
    """
    if cfg.deepspeed_config:
        ds_config = _load_json_or_yaml(cfg.deepspeed_config)
        if not isinstance(ds_config, dict):
            raise ValueError("DeepSpeed 配置文件的根节点必须是 JSON/YAML 对象。")
        _complete_and_validate_shared_config(ds_config, cfg)
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
    保存当前训练最终生效的 DeepSpeed 配置。

    分布式训练中仅由全局 rank 0 写入文件，避免多个进程竞争同一路径。

    Args:
        ds_config: 已解析并实际用于训练的 DeepSpeed 配置。
        cfg: 训练配置，提供配置文件保存路径。

    Returns:
        None。
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    path = get_deepspeed_runtime_config_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ds_config, ensure_ascii=False, indent=2) + "\n")


def load_deepspeed_resume_metadata(cfg: TrainConfig) -> dict[str, Any] | None:
    """
    加载 DeepSpeed checkpoint 的轻量续训元数据。

    该函数不依赖 DeepSpeed engine，可在引擎初始化前读取最后保存的 tag、epoch、
    step 和 SwanLab run ID。

    Args:
        cfg: 训练配置，提供续训元数据文件路径。

    Returns:
        解析后的续训元数据；文件不存在时返回 `None`。
    """
    meta_path = get_deepspeed_resume_metadata_path(cfg)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def load_deepspeed_engine_checkpoint(engine: Any, cfg: TrainConfig) -> dict[str, Any] | None:
    """
    使用 DeepSpeed engine 恢复最新训练 checkpoint。

    `tag=None` 时由 DeepSpeed 根据 checkpoint 目录中的 latest 信息选择最近一次
    保存。模型、优化器和 ZeRO 状态由 engine 恢复，函数只返回保存时附加的
    `client_state`。

    Args:
        engine: 已初始化的 DeepSpeed engine。
        cfg: 训练配置，提供 checkpoint 根目录。

    Returns:
        包含 epoch、step、SwanLab run ID 等信息的 `client_state`；checkpoint
        目录不存在或没有可加载状态时返回 `None`。
    """
    checkpoint_dir = get_deepspeed_checkpoint_dir(cfg)
    if not checkpoint_dir.exists():
        return None

    load_path, client_state = engine.load_checkpoint(str(checkpoint_dir), tag=None)
    if load_path is None:
        return None
    return client_state or {}


def _get_swanlab_id(swanlab_: Any | None) -> str | None:
    """
    获取当前 SwanLab 实验的 run ID。

    Args:
        swanlab_: SwanLab 模块、run 对象或 `None`。

    Returns:
        当前实验的 run ID；未启用 SwanLab 或无法获取时返回 `None`。
    """
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
    保存可续训的 DeepSpeed checkpoint 和轻量元数据 sidecar。

    所有 rank 都必须调用 `engine.save_checkpoint()`，由 DeepSpeed 协同保存模型
    分片、优化器状态、ZeRO 信息和 `client_state`。保存完成后仅由全局 rank 0
    写入便于引擎初始化前查询的 JSON sidecar。

    Args:
        engine: 已初始化的 DeepSpeed engine。
        epoch: 当前 epoch 的零基索引。
        step: 当前 batch 在 epoch 内的零基索引。
        cfg: 训练配置，提供 checkpoint 保存目录。
        swanlab_: SwanLab 实验对象；未启用实验跟踪时为 `None`。

    Returns:
        None。
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

    # DeepSpeed 协同保存模型分片、优化器状态、ZeRO 分区信息和附加的 client_state。
    engine.save_checkpoint(str(checkpoint_dir), tag=tag, client_state=client_state)

    # 仅由全局 rank 0 写入便于直接查看的续训元数据 sidecar。
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    meta_path = get_deepspeed_resume_metadata_path(cfg)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"latest_tag": tag, **client_state}, ensure_ascii=False, indent=2) + "\n"
    )
