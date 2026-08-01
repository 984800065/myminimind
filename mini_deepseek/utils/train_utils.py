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
    构造不包含扩展名的统一模型文件名。

    文件名格式为
    `<weight>_<hidden_size>_moe-<moe_type>_attn-<attention_type>`，用于让权重、
    配置和续训状态共享同一命名前缀。

    Args:
        cfg: 训练或推理配置，提供权重名称和模型结构信息。
        weight: 显式指定的权重名称；未指定时从 `cfg` 读取。

    Returns:
        不包含目录和扩展名的模型文件名。
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
    将统一文件名拼接到配置指定的保存目录。

    Args:
        cfg: 训练或推理配置，提供 `save_dir`。
        universal_file_name: 需要保存的文件名，可以包含扩展名。

    Returns:
        格式为 `<save_dir>/<universal_file_name>` 的完整路径。
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
    """
    使用统一规则构造 SwanLab 实验名称。

    Args:
        project_model_name: 项目或模型名称。
        universal_name: 由模型结构和权重名称组成的统一名称。
        epochs: 训练总 epoch 数。
        batch_size: 每个 micro-batch 的样本数。
        learning_rate: 初始学习率。

    Returns:
        包含模型、训练轮数、batch size 和学习率的实验名称。
    """

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
    """
    根据训练配置构造 SwanLab 实验名称。

    Args:
        cfg: 训练配置，提供模型结构和训练超参数。
        project_model_name: 项目或模型名称。
        weight: 显式指定的权重名称；未指定时使用 `cfg.save_weight`。

    Returns:
        遵循项目统一命名规则的 SwanLab 实验名称。
    """

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
    构造训练、推理和导出共用的原始模型权重路径。

    路径格式为 `<save_dir>/<universal_name>[_debug].pth`。训练配置默认根据
    `cfg.debug` 决定是否添加 `_debug`，推理配置默认不添加。

    Args:
        cfg: 训练或推理配置。
        weight: 显式指定的权重名称；未指定时从 `cfg` 读取。
        include_debug: 是否添加 `_debug` 后缀；`None` 表示按配置自动判断。

    Returns:
        原始模型权重文件的完整路径。
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
    定位与权重名称匹配的原始模型 checkpoint。

    优先返回根据当前配置生成的精确路径。精确文件不存在时，按权重名称前缀
    搜索保存目录；仅有一个候选文件时自动采用，多个候选文件时要求调用方明确
    模型结构，避免加载错误 checkpoint。

    Args:
        cfg: 训练或推理配置。
        weight: 显式指定的权重名称；未指定时从 `cfg` 读取。
        include_debug: 是否在精确路径中添加 `_debug` 后缀。

    Returns:
        已定位的 checkpoint 路径；没有匹配文件时返回按配置生成的预期路径。

    Raises:
        ValueError: 权重名称前缀匹配到多个 checkpoint，无法唯一确定目标文件。
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
    构造原生 PyTorch 训练使用的续训 checkpoint 路径。

    原始推理权重保存为 `*.pth`，优化器、调度器和训练位置等续训状态单独保存为
    `*_resume.pth`，避免推理加载不必要的训练状态。

    Args:
        cfg: 训练配置，提供保存目录、权重名称和 debug 标记。
        weight: 显式指定的权重名称；未指定时使用 `cfg.save_weight`。

    Returns:
        续训 checkpoint 的完整路径。
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
    构造与原始模型权重同名的结构配置路径。

    PyTorch state dict 不包含模型超参数，因此使用同名 `.config.json` 文件保存
    完整模型结构，防止推理或导出时因默认配置变化而错误重建 MLA、MoE 或 MTP
    等模块。

    Args:
        cfg: 训练或推理配置。
        weight: 显式指定的权重名称；未指定时从 `cfg` 读取。
        include_debug: 是否在权重文件名中添加 `_debug` 后缀。

    Returns:
        模型结构 sidecar 文件的完整路径。
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
    """
    将模型结构配置原子写入原始权重旁的 sidecar 文件。

    先写入临时文件，再通过 `os.replace` 替换目标文件，避免进程中断后留下
    不完整的 JSON 配置。

    Args:
        cfg: 训练或推理配置。
        lm_config: 需要持久化的完整模型结构配置。
        weight: 显式指定的权重名称；未指定时从 `cfg` 读取。
        include_debug: 是否在权重文件名中添加 `_debug` 后缀。

    Returns:
        已保存的模型结构配置路径。
    """
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
    """
    根据启动器注入的环境变量初始化分布式训练，并返回当前进程的本地 GPU 编号。

    `torchrun` 或 DeepSpeed launcher 会为每个训练进程设置以下环境变量：
    - `RANK`：当前进程在整个分布式任务中的全局编号。
    - `LOCAL_RANK`：当前进程在本机上的编号，用于选择对应 GPU。
    - `WORLD_SIZE`：参与训练的总进程数。

    如果不存在 `RANK`，说明当前是普通单进程启动，此时不创建进程组并返回
    `0`。分布式启动时统一使用 NCCL 后端：DeepSpeed 模式由 DeepSpeed
    初始化通信环境，原生 DDP 模式则由 PyTorch 从环境变量完成 rendezvous。

    Args:
        use_deepspeed: 是否由 DeepSpeed 初始化分布式通信环境。

    Returns:
        当前进程的 `LOCAL_RANK`；非分布式模式返回 `0`。
    """
    # 普通 `python -m ...` 启动不会设置 RANK，不需要初始化分布式进程组。
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    # 两种模式都使用 NCCL 进行 GPU 间通信，但初始化入口由训练引擎决定。
    if use_deepspeed:
        deepspeed.init_distributed(dist_backend="nccl")
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    # 每个进程只操作 LOCAL_RANK 对应的 GPU，避免多个进程误用同一设备。
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

    Args:
        seed: 用于初始化各随机数生成器的随机种子。

    Returns:
        None。
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
    移除 DDP 或 `torch.compile` 包装，返回实际持有状态的对象。

    Args:
        value: 原始对象、DDP 包装对象或包含 `_orig_mod` 的编译对象。

    Returns:
        可直接读取 `state_dict()` 的底层对象。
    """
    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
    return getattr(raw_value, "_orig_mod", raw_value)


def _to_cpu_state_dict(
    value: Any,
    *,
    cast_floating_to_fp16: bool,
) -> dict[str, Any]:
    """
    将模型 state dict 转换为存放在 CPU 上的字典。

    推理权重可以转为 FP16 以减小文件体积；续训权重保留原始精度，避免恢复训练
    后因权重精度变化影响结果。

    Args:
        value: 模型本身或其 DDP、`torch.compile` 包装对象。
        cast_floating_to_fp16: 是否将浮点 tensor 转换为 FP16。

    Returns:
        所有 tensor 均已脱离计算图并移动到 CPU 的 state dict。
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
    将 checkpoint 字段转换为可保存的值。

    Args:
        value: 优化器、调度器等含 `state_dict()` 的对象，或普通 Python 值。

    Returns:
        对象的 state dict；没有 `state_dict()` 时原样返回输入值。
    """
    if hasattr(value, "state_dict"):
        return _unwrap_stateful_object(value).state_dict()
    return value


def _atomic_torch_save(payload: Any, path: str) -> None:
    """
    使用临时文件原子保存 PyTorch 数据。

    Args:
        payload: 需要交给 `torch.save` 序列化的数据。
        path: 最终保存路径。

    Returns:
        None。
    """
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


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


def _capture_rng_state() -> dict[str, Any]:
    """
    保存精确续训需要的随机数生成器状态。

    Returns:
        包含 Python、NumPy、PyTorch CPU 以及可用 CUDA RNG 状态的字典。
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint_data: dict[str, Any]) -> None:
    """
    从续训 checkpoint 恢复随机数生成器状态。

    checkpoint 不含 `rng_state` 时直接返回，以兼容旧版本保存的训练状态。

    Args:
        checkpoint_data: 已加载的续训 checkpoint。

    Returns:
        None。
    """
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
    保存或加载原生 PyTorch 的可续训 checkpoint。

    `model` 不为 `None` 时进入保存模式，同时写入便于推理的 FP16 原始权重和
    保留原精度的续训状态。`model` 为 `None` 时进入加载模式，读取续训状态；
    若 GPU 数量发生变化，会按新旧 world size 换算当前 epoch 的 step。

    Args:
        cfg: 训练配置，提供 checkpoint 路径和运行参数。
        model: 待保存的模型；为 `None` 时表示加载 checkpoint。
        optimizer: 优化器；保存模式下将其 state dict 写入续训状态。
        lm_config: 模型结构配置；提供时保存到权重旁的 sidecar 文件。
        epoch: 保存时所在的 epoch 索引。
        step: 保存时所在的 batch 索引。
        swanlab_: SwanLab 实验对象，用于持久化 run ID。
        **extra_state: 需要额外保存的调度器、GradScaler 等训练状态。

    Returns:
        加载模式下返回 checkpoint 字典，文件不存在时返回 `None`；保存模式下
        返回 `None`。
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
    将训练指标和进度信息记录到 SwanLab。

    指标名称直接包含秒、分钟、百分比等单位，并使用跨 epoch 单调递增的
    `global_step` 作为横轴。`swanlab_run` 为 `None` 时直接返回。

    Args:
        swanlab_run: SwanLab 实验对象；未启用实验跟踪时为 `None`。
        epoch: 当前 epoch 的零基索引。
        step: 当前 batch 在 epoch 内的零基索引。
        steps_per_epoch: 每个 epoch 的 batch 总数。
        total_epochs: 训练总 epoch 数。
        learning_rate: 当前优化器学习率。
        elapsed_seconds: 当前 epoch 已用时间，单位为秒。
        eta_minutes: 当前 epoch 预计剩余时间，单位为分钟。
        train_metrics: 需要记录到 `train` 分组的训练指标。

    Returns:
        None。

    Raises:
        ValueError: `steps_per_epoch` 或 `total_epochs` 不为正数。
    """
    if swanlab_run is None:
        return

    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}.")
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be > 0, got {total_epochs}.")

    # 使用从 1 开始的全局 step，便于阅读，并避免从 epoch 中间续训时重复使用 step 0。
    global_step = epoch * steps_per_epoch + step + 1
    epoch_step = step + 1

    # 百分比指标按 [0, 100] 保存，并在指标名中标注单位，方便直接阅读 SwanLab 图表。
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


def is_main_process() -> bool:
    """
    判断当前进程是否为负责日志和文件写入的主进程。

    Returns:
        单进程环境或分布式全局 rank 为 `0` 时返回 `True`，否则返回 `False`。
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def get_model_params(model: MiniDeepSeekForCausalLM, config: MiniDeepSeekConfig) -> None:
    """
    统计并记录模型总参数量及 MoE 单 token 激活参数量。

    Dense 模型只记录总参数量；MoE 模型根据路由专家数、每 token 激活专家数和
    共享专家数，估算一次 token 前向实际参与计算的参数量。日志单位为百万参数。

    Args:
        model: 需要统计参数量的 MiniDeepSeek 模型。
        config: 模型结构配置，用于读取注意力类型和专家数量。

    Returns:
        None。
    """
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
    """
    从指定路径加载 tokenizer，并确保其具有 padding token。

    tokenizer 缺少 `pad_token` 时使用 `eos_token` 作为 padding token；两者都
    不存在时无法构造训练 batch。

    Args:
        tokenizer_path: Hugging Face tokenizer 的本地路径或模型标识。

    Returns:
        已配置 padding token 的 tokenizer。

    Raises:
        ValueError: tokenizer 同时缺少 `pad_token` 和 `eos_token`。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer {tokenizer_path!r} 缺少 pad_token 和 eos_token，当前训练入口无法自动补齐。")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def sync_lm_config_with_tokenizer(lm_config: MiniDeepSeekConfig, tokenizer: AutoTokenizer) -> MiniDeepSeekConfig:
    """
    将 tokenizer 的词表和特殊 token 信息同步到模型配置。

    Args:
        lm_config: 待更新的模型结构配置。
        tokenizer: 已加载并完成特殊 token 配置的 tokenizer。

    Returns:
        原地更新后的 `lm_config`。
    """
    lm_config.vocab_size = len(tokenizer)
    lm_config.bos_token_id = tokenizer.bos_token_id
    lm_config.eos_token_id = tokenizer.eos_token_id
    lm_config.pad_token_id = tokenizer.pad_token_id
    return lm_config


def resolve_lm_config_and_tokenizer(cfg: BaseConfig) -> tuple[MiniDeepSeekConfig, AutoTokenizer]:
    """
    根据运行配置创建模型配置和 tokenizer，并同步两者的词表信息。

    Args:
        cfg: 训练或推理基础配置。

    Returns:
        `(lm_config, tokenizer)`，其中模型配置的词表大小和特殊 token ID 已与
        tokenizer 对齐。
    """
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
    初始化可训练模型，并按需加载原始 `.pth` 权重。

    `from_weight` 可让 DPO、GRPO 等训练入口覆盖来源 checkpoint，同时不修改
    当前任务的 `cfg.save_weight`。权重使用非严格模式加载，并记录缺失或多余
    参数，兼容微调阶段新增模块。

    Args:
        cfg: 训练配置，提供设备、tokenizer 路径和来源权重名称。
        lm_config: 用于构建模型的完整结构配置。
        tokenizer: 已加载的 tokenizer；未提供时根据 `cfg.tokenizer_path` 加载。
        from_weight: 显式指定的来源权重名称；未指定时使用 `cfg.from_weight`。

    Returns:
        `(model, tokenizer)`，其中模型已加载可用权重并移动到 `cfg.device`。
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
        # 微调可以新增层，但必须报告结构差异，避免在大部分参数未加载时误以为预训练权重已生效。
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
    """
    将样本索引组成 batch，并跳过开头指定数量的 batch。

    该采样器用于从 epoch 中间续训：上游 sampler 重新生成与中断前一致的样本
    顺序，本采样器丢弃已经完成的 batch，只向 DataLoader 提供剩余索引。

    Args:
        sampler: 提供单个样本索引的基础采样器或索引序列。
        batch_size: 每个 batch 的样本数。
        skip_batches: 从开头跳过的完整 batch 数。
    """

    def __init__(self, sampler: Sampler, batch_size: int, skip_batches: int = 0):
        """
        初始化批采样器。

        Args:
            sampler: 提供单个样本索引的基础采样器或索引序列。
            batch_size: 每个 batch 的样本数。
            skip_batches: 从开头跳过的完整 batch 数。

        Returns:
            None。
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """
        依次生成未被跳过的 batch 索引。

        Yields:
            一个 batch 的样本索引列表；最后一个 batch 可以少于 `batch_size`。
        """
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
        """
        返回跳批后剩余的 batch 数。

        Returns:
            可由当前采样器生成的 batch 数，不小于 `0`。
        """
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
