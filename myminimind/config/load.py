"""
配置加载：按「默认 → 配置文件 → 命令行」三层覆盖，得到最终的配置对象。

为什么要分层：
  - 默认值：代码里写死一套合理默认。
  - 配置文件：不同实验用不同 json/yaml，不改代码。
  - 命令行：临时覆盖某几项（如 --batch-size 64），不用改文件。

按场景分别使用 get_pretrain_config() / get_infer_config() / get_sft_config() 等入口。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings

from .schema import DistillationConfig, DPOConfig, GRPOConfig, InferConfig, PretrainConfig, SFTConfig


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    """根据后缀把配置文件读成字典，只支持 .json / .yaml / .yml。"""
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml

            return yaml.safe_load(text) or {}
        except ImportError:
            raise ImportError("YAML 支持需要: pip install pyyaml") from None
    raise ValueError(f"不支持的配置文件格式: {suffix}")


def _bool_opt(s: str | None) -> bool | None:
    """把命令行传来的 0/1、true/false、yes 等转成 bool，None 保持 None（表示未传）。"""
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    return str(s).lower() in ("1", "true", "yes")


def _new_parser(description: str) -> argparse.ArgumentParser:
    """创建 parser，并自动挂上配置文件参数和分布式 launcher 兼容参数。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, dest="local_rank", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径 (json/yaml)")
    return parser


def _add_save_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册保存目录、权重名和日志/保存间隔参数。"""
    parser.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    parser.add_argument("--save-weight", type=str, default=None, dest="save_weight")
    parser.add_argument("--save-interval", type=int, default=None, dest="save_interval")
    parser.add_argument("--log-interval", type=int, default=None, dest="log_interval")


def _add_train_hparam_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册训练轮数、batch size、学习率等基础训练超参。"""
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    parser.add_argument("--accumulation-steps", type=int, default=None, dest="accumulation_steps")
    parser.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")


def _add_device_dtype_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册 device 和混合精度 dtype 参数。"""
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"])


def _add_basic_data_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册数据路径、DataLoader 线程数和训练数据截断长度参数。"""
    parser.add_argument("--data-path", type=str, default=None, dest="data_path")
    parser.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    parser.add_argument("--data-max-seq-len", "--max-seq-len", type=int, default=None, dest="data_max_seq_len")


def _add_tokenizer_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册 tokenizer 路径参数。"""
    parser.add_argument("--tokenizer-path", type=str, default=None, dest="tokenizer_path")


def _add_attention_parser_args(parser: argparse.ArgumentParser) -> None:
    """给命令行解析器补充 attention_type 和 MLA 相关参数。"""
    parser.add_argument("--attention-type", type=str, default=None, dest="attention_type", choices=["gqa", "mla"])
    parser.add_argument("--mla-q-lora-rank", type=int, default=None, dest="mla_q_lora_rank")
    parser.add_argument("--mla-kv-lora-rank", type=int, default=None, dest="mla_kv_lora_rank")
    parser.add_argument("--mla-qk-nope-head-dim", type=int, default=None, dest="mla_qk_nope_head_dim")
    parser.add_argument("--mla-qk-rope-head-dim", type=int, default=None, dest="mla_qk_rope_head_dim")
    parser.add_argument("--mla-v-head-dim", type=int, default=None, dest="mla_v_head_dim")


def _add_norm_parser_args(parser: argparse.ArgumentParser) -> None:
    """给命令行解析器补充 norm 实现选择参数。"""
    parser.add_argument("--norm-implementation", type=str, default=None, dest="norm_implementation")


def _add_linear_cross_entropy_parser_args(parser: argparse.ArgumentParser) -> None:
    """给命令行解析器补充 LM head loss 实现选择参数。"""
    parser.add_argument(
        "--linear-cross-entropy-implementation",
        type=str,
        default=None,
        dest="linear_cross_entropy_implementation",
    )


def _add_rope_parser_args(parser: argparse.ArgumentParser) -> None:
    """给命令行解析器补充扁平的 RoPE scaling 参数，随后由 schema 组装为 `rope_scaling` 字典。"""
    parser.add_argument("--rope-implementation", type=str, default=None, dest="rope_implementation")
    parser.add_argument("--rope-type", type=str, default=None, dest="rope_type")
    parser.add_argument("--rope-factor", type=float, default=None, dest="rope_factor")
    parser.add_argument("--rope-beta-fast", type=float, default=None, dest="rope_beta_fast")
    parser.add_argument("--rope-beta-slow", type=float, default=None, dest="rope_beta_slow")
    parser.add_argument(
        "--rope-original-max-position-embeddings",
        type=int,
        default=None,
        dest="rope_original_max_position_embeddings",
    )
    parser.add_argument("--rope-attention-factor", type=float, default=None, dest="rope_attention_factor")


def _add_multi_token_prediction_args(parser: argparse.ArgumentParser) -> None:
    """给命令行解析器补充多 token 预测相关参数。"""
    parser.add_argument(
        "--mtp-level",
        type=int,
        default=None,
        dest="mtp_level",
        choices=[0, 1, 2],
        help="多token预测数量（0=无多token预测，1=多预测1个token，2=多预测2个token，......）",
    )
    parser.add_argument("--mtp-lambda", type=float, default=None, dest="mtp_lambda", help="多token预测损失权重")


def _add_train_model_parser_args(parser: argparse.ArgumentParser, *, include_mtp: bool = False) -> None:
    """注册训练侧共享的模型结构参数，并按需补充 MTP 参数。"""
    parser.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    parser.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    parser.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")
    _add_attention_parser_args(parser)
    _add_norm_parser_args(parser)
    _add_linear_cross_entropy_parser_args(parser)
    _add_rope_parser_args(parser)
    if include_mtp:
        _add_multi_token_prediction_args(parser)


def _add_resume_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册初始化权重和断点续训相关参数。"""
    parser.add_argument("--from-weight", type=str, default=None, dest="from_weight")
    parser.add_argument("--from-resume", nargs="?", const="1", default=None, dest="from_resume", help="0/1 或省略即 1")


def _add_experiment_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册实验记录和编译开关参数。"""
    parser.add_argument("--use-swanlab", nargs="?", const="1", default=None, dest="use_swanlab", help="启用 swanlab")
    parser.add_argument("--swanlab-project", type=str, default=None, dest="swanlab_project")
    parser.add_argument("--use-compile", nargs="?", const="1", default=None, dest="use_compile", help="0/1 或省略即 1")


def _add_deepspeed_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册 DeepSpeed 训练引擎相关参数。"""
    parser.add_argument("--use-deepspeed", nargs="?", const="1", default=None, dest="use_deepspeed", help="启用 DeepSpeed")
    parser.add_argument("--deepspeed-config", type=str, default=None, dest="deepspeed_config", help="DeepSpeed 配置文件路径（json/yaml）")
    parser.add_argument("--deepspeed-zero-stage", type=int, default=None, dest="deepspeed_zero_stage", choices=[0, 1, 2])
    parser.add_argument(
        "--deepspeed-offload-optimizer",
        nargs="?",
        const="1",
        default=None,
        dest="deepspeed_offload_optimizer",
        help="启用 DeepSpeed CPU optimizer offload",
    )
    parser.add_argument("--deepspeed-tensor-parallel-size", type=int, default=None, dest="deepspeed_tensor_parallel_size")


def _add_infer_load_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册推理时的权重、配置和 tokenizer 加载参数。"""
    parser.add_argument("--tokenizer-path", type=str, default=None, dest="tokenizer_path")
    parser.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--lora-weight", type=str, default=None, dest="lora_weight")
    parser.add_argument("--model-config-path", type=str, default=None, dest="model_config_path")
    parser.add_argument("--hf-model-dir", type=str, default=None, dest="hf_model_dir")


def _add_infer_model_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册推理时需要的模型结构、MoE 和 attention 参数。"""
    parser.add_argument("--hidden-act", type=str, default=None, dest="hidden_act")
    parser.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    parser.add_argument("--intermediate-size", type=int, default=None, dest="intermediate_size")
    parser.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")
    parser.add_argument("--num-attention-heads", type=int, default=None, dest="num_attention_heads")
    parser.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    parser.add_argument("--group-num", type=int, default=None, dest="group_num")
    parser.add_argument("--vocab-size", type=int, default=None, dest="vocab_size")
    parser.add_argument("--rms-norm-eps", type=float, default=None, dest="rms_norm_eps")
    parser.add_argument("--rope-theta", type=int, default=None, dest="rope_theta")
    parser.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")
    parser.add_argument("--flash-attention", nargs="?", const="1", default=None, dest="flash_attention", help="0/1 或省略即 1")
    parser.add_argument("--num-experts-per-token", type=int, default=None, dest="num_experts_per_token")
    parser.add_argument("--num-routed-experts", type=int, default=None, dest="num_routed_experts")
    parser.add_argument("--num-shared-experts", type=int, default=None, dest="num_shared_experts")
    parser.add_argument("--scoring-function", type=str, default=None, dest="scoring_function")
    parser.add_argument("--aux-loss-alpha", type=float, default=None, dest="aux_loss_alpha")
    parser.add_argument("--seq-aux", nargs="?", const="1", default=None, dest="seq_aux", help="0/1 或省略即 1")
    parser.add_argument("--norm-topk-prob", nargs="?", const="1", default=None, dest="norm_topk_prob", help="0/1 或省略即 1")
    parser.add_argument("--capacity-factor", type=float, default=None, dest="capacity_factor")
    parser.add_argument("--mtp-level", type=int, default=None, dest="mtp_level", choices=[0, 1, 2])
    _add_attention_parser_args(parser)
    _add_norm_parser_args(parser)
    _add_linear_cross_entropy_parser_args(parser)
    _add_rope_parser_args(parser)


def _add_infer_generation_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册生成策略和 vLLM 调度相关参数。"""
    parser.add_argument("--inference-rope-scaling", nargs="?", const="1", default=None, dest="inference_rope_scaling", help="启用RoPE外推")
    parser.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None, dest="top_p")
    parser.add_argument("--vllm-runner", type=str, default=None, dest="vllm_runner")
    parser.add_argument("--vllm-model-impl", type=str, default=None, dest="vllm_model_impl")
    parser.add_argument("--vllm-dtype", type=str, default=None, dest="vllm_dtype")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=None, dest="vllm_tensor_parallel_size")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None, dest="vllm_gpu_memory_utilization")
    parser.add_argument("--vllm-max-model-len", type=int, default=None, dest="vllm_max_model_len")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=None, dest="vllm_max_num_seqs")
    parser.add_argument("--vllm-enforce-eager", nargs="?", const="1", default=None, dest="vllm_enforce_eager", help="0/1 或省略即 1")
    parser.add_argument("--vllm-trust-remote-code", nargs="?", const="1", default=None, dest="vllm_trust_remote_code", help="0/1 或省略即 1")


def _add_infer_display_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册对话展示和推理设备相关参数。"""
    parser.add_argument("--historys", type=int, default=None)
    parser.add_argument("--show-speed", type=int, default=None, dest="show_speed")
    parser.add_argument("--device", type=str, default=None)


def _add_debug_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册 debug 模式开关。"""
    parser.add_argument("--debug", nargs="?", const="1", default=None, dest="debug", help="启用 debug 模式（小数据集+多日志）")


def _add_dpo_extra_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册 DPO 独有的 beta 参数。"""
    parser.add_argument("--beta", type=float, default=None)


def _add_grpo_data_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册 GRPO 数据采样和生成长度参数。"""
    parser.add_argument("--max-gen-len", type=int, default=None, dest="max_gen_len")
    parser.add_argument("--num-generations", type=int, default=None, dest="num_generations")


def _add_grpo_extra_parser_args(parser: argparse.ArgumentParser) -> None:
    """注册 GRPO 独有的奖励模型和算法参数。"""
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--reasoning", type=int, default=None)
    parser.add_argument("--reward-model-name", type=str, default=None, dest="reward_model_name")
    parser.add_argument("--reward-model-tokenizer-name", type=str, default=None, dest="reward_model_tokenizer_name")


def _overlay_file_config(config_dict: dict[str, Any], config_path: Path | None) -> None:
    """用配置文件覆盖默认配置，只覆盖当前配置对象里已有的键。"""
    if config_path is None or not config_path.exists():
        return

    file_dict = _load_json_or_yaml(config_path)

    # Training configs now use `data_max_seq_len` to mean dataset truncation
    # length. Keep old config files using `max_seq_len` working during the
    # migration period.
    if "data_max_seq_len" in config_dict and "data_max_seq_len" not in file_dict and "max_seq_len" in file_dict:
        file_dict["data_max_seq_len"] = file_dict["max_seq_len"]

    for key, value in file_dict.items():
        if key in config_dict and value is not None:
            config_dict[key] = value


def _overlay_cli_args(config_dict: dict[str, Any], parsed: argparse.Namespace, bool_keys: tuple[str, ...]) -> None:
    """用命令行覆盖当前配置，只处理用户真正传了的参数。"""
    bool_key_set = set(bool_keys)
    for key in list(config_dict.keys()):
        value = getattr(parsed, key, None)
        if value is None:
            continue
        if key in bool_key_set:
            value = _bool_opt(value)
        config_dict[key] = value


def _dump_base_config(config_cls: type[BaseSettings]) -> dict[str, Any]:
    """
    先构造一份“默认值 + 环境变量”合并后的基础配置字典。
    """
    return config_cls().model_dump()


def _build_config[ConfigT: BaseSettings](
    config_cls: type[ConfigT],
    parser: argparse.ArgumentParser,
    *,
    bool_keys: tuple[str, ...],
    args: list[str] | None = None,
) -> ConfigT:
    """
    通用三层配置加载：
      1. 默认值 + 对应前缀环境变量
      2. 配置文件
      3. 命令行
    """
    parsed = parser.parse_args(args)
    config_dict = _dump_base_config(config_cls)
    _overlay_file_config(config_dict, parsed.config)
    _overlay_cli_args(config_dict, parsed, bool_keys)
    return config_cls(**config_dict)


def _build_pretrain_parser() -> argparse.ArgumentParser:
    """构建预训练命令行解析器。"""
    parser = _new_parser("MiniMind 预训练")
    _add_save_parser_args(parser)
    _add_train_hparam_parser_args(parser)
    _add_device_dtype_parser_args(parser)
    _add_basic_data_parser_args(parser)
    _add_tokenizer_parser_args(parser)
    _add_train_model_parser_args(parser, include_mtp=True)
    _add_resume_parser_args(parser)
    _add_experiment_parser_args(parser)
    _add_deepspeed_parser_args(parser)
    _add_debug_parser_args(parser)
    return parser


def _build_infer_parser() -> argparse.ArgumentParser:
    """构建推理/对话命令行解析器。"""
    parser = _new_parser("MiniMind模型推理与对话")
    _add_infer_load_parser_args(parser)
    _add_infer_model_parser_args(parser)
    _add_infer_generation_parser_args(parser)
    _add_infer_display_parser_args(parser)
    return parser


def _build_sft_parser() -> argparse.ArgumentParser:
    """构建 Full SFT 命令行解析器。"""
    parser = _new_parser("MiniMind Full SFT")
    _add_save_parser_args(parser)
    _add_train_hparam_parser_args(parser)
    _add_device_dtype_parser_args(parser)
    _add_basic_data_parser_args(parser)
    _add_tokenizer_parser_args(parser)
    _add_train_model_parser_args(parser)
    _add_resume_parser_args(parser)
    _add_experiment_parser_args(parser)
    return parser


def _build_dpo_parser() -> argparse.ArgumentParser:
    """构建 DPO 命令行解析器。"""
    parser = _new_parser("MiniMind DPO (Direct Preference Optimization)")
    _add_save_parser_args(parser)
    _add_train_hparam_parser_args(parser)
    _add_device_dtype_parser_args(parser)
    _add_basic_data_parser_args(parser)
    _add_tokenizer_parser_args(parser)
    _add_train_model_parser_args(parser)
    _add_resume_parser_args(parser)
    _add_dpo_extra_parser_args(parser)
    _add_experiment_parser_args(parser)
    return parser


def _build_grpo_parser() -> argparse.ArgumentParser:
    """构建 GRPO 命令行解析器。"""
    parser = _new_parser("MiniMind GRPO (Group Relative Policy Optimization)")
    _add_save_parser_args(parser)
    _add_train_hparam_parser_args(parser)
    _add_device_dtype_parser_args(parser)
    _add_basic_data_parser_args(parser)
    _add_grpo_data_parser_args(parser)
    _add_tokenizer_parser_args(parser)
    _add_train_model_parser_args(parser)
    _add_resume_parser_args(parser)
    _add_grpo_extra_parser_args(parser)
    _add_experiment_parser_args(parser)
    return parser


def _build_distillation_parser() -> argparse.ArgumentParser:
    """构建 On-policy 蒸馏命令行解析器。"""
    parser = _new_parser("MiniMind On-policy Distillation (白盒蒸馏)")
    _add_save_parser_args(parser)
    _add_train_hparam_parser_args(parser)
    _add_device_dtype_parser_args(parser)
    _add_basic_data_parser_args(parser)
    _add_tokenizer_parser_args(parser)
    _add_train_model_parser_args(parser)
    _add_resume_parser_args(parser)
    _add_experiment_parser_args(parser)
    return parser


PRETRAIN_BOOL_KEYS = (
    "use_moe",
    "from_resume",
    "use_swanlab",
    "use_compile",
    "use_deepspeed",
    "deepspeed_offload_optimizer",
    "debug",
)

INFER_BOOL_KEYS = (
    "use_moe",
    "flash_attention",
    "seq_aux",
    "norm_topk_prob",
    "inference_rope_scaling",
    "vllm_enforce_eager",
    "vllm_trust_remote_code",
)

TRAIN_BOOL_KEYS = ("use_moe", "from_resume", "use_swanlab", "use_compile")


def get_pretrain_config(args: list[str] | None = None) -> PretrainConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 PretrainConfig 实例。

    参数：
      args：通常不传，表示用 sys.argv（即当前进程的命令行）；传了则用这份列表解析，便于测试或二次封装。
    """
    return _build_config(PretrainConfig, _build_pretrain_parser(), bool_keys=PRETRAIN_BOOL_KEYS, args=args)


def get_infer_config(args: list[str] | None = None) -> InferConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 InferConfig 实例。

    用法：cfg = get_infer_config()，然后 cfg.tokenizer_path、cfg.weight、cfg.max_new_tokens 等。
    """
    return _build_config(InferConfig, _build_infer_parser(), bool_keys=INFER_BOOL_KEYS, args=args)


def get_sft_config(args: list[str] | None = None) -> SFTConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 SFTConfig 实例。

    用法：cfg = get_sft_config()，然后 cfg.batch_size、cfg.to_lm_config_kwargs() 等。
    """
    return _build_config(SFTConfig, _build_sft_parser(), bool_keys=TRAIN_BOOL_KEYS, args=args)


def get_dpo_config(args: list[str] | None = None) -> DPOConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 DPOConfig 实例。

    用法：cfg = get_dpo_config()，然后 cfg.batch_size、cfg.beta、cfg.to_lm_config_kwargs() 等。
    """
    return _build_config(DPOConfig, _build_dpo_parser(), bool_keys=TRAIN_BOOL_KEYS, args=args)


def get_grpo_config(args: list[str] | None = None) -> GRPOConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 GRPOConfig 实例。

    用法：cfg = get_grpo_config()，然后 cfg.batch_size、cfg.beta、cfg.to_lm_config_kwargs() 等。
    """
    return _build_config(GRPOConfig, _build_grpo_parser(), bool_keys=TRAIN_BOOL_KEYS, args=args)


def get_distillation_config(args: list[str] | None = None) -> DistillationConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 DistillationConfig 实例。

    用法：cfg = get_distillation_config()，然后 cfg.batch_size、cfg.to_lm_config_kwargs() 等。
    """
    return _build_config(DistillationConfig, _build_distillation_parser(), bool_keys=TRAIN_BOOL_KEYS, args=args)
