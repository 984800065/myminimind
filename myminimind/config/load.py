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


def _add_attention_parser_args(parser: argparse.ArgumentParser) -> None:
    """给命令行解析器补充 attention_type 和 MLA 相关参数。"""
    parser.add_argument("--attention-type", type=str, default=None, dest="attention_type", choices=["gqa", "mla"])
    parser.add_argument("--mla-q-lora-rank", type=int, default=None, dest="mla_q_lora_rank")
    parser.add_argument("--mla-kv-lora-rank", type=int, default=None, dest="mla_kv_lora_rank")
    parser.add_argument("--mla-qk-nope-head-dim", type=int, default=None, dest="mla_qk_nope_head_dim")
    parser.add_argument("--mla-qk-rope-head-dim", type=int, default=None, dest="mla_qk_rope_head_dim")
    parser.add_argument("--mla-v-head-dim", type=int, default=None, dest="mla_v_head_dim")


def _build_pretrain_parser() -> argparse.ArgumentParser:
    """
    构建命令行解析器。所有参数 default=None，表示「没传就不覆盖」：
    这样在 get_pretrain_config() 里只把「用户真正传了」的项覆盖到配置上，没传的用默认或配置文件里的值。
    """
    p = argparse.ArgumentParser(description="MiniMind 预训练")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径 (json/yaml)")

    # 保存与输出
    p.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    p.add_argument("--save-weight", type=str, default=None, dest="save_weight")
    p.add_argument("--save-interval", type=int, default=None, dest="save_interval")
    p.add_argument("--log-interval", type=int, default=None, dest="log_interval")

    # 训练超参
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    p.add_argument("--accumulation-steps", type=int, default=None, dest="accumulation_steps")
    p.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")

    # 设备与精度
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"])

    # 数据
    p.add_argument("--data-path", type=str, default=None, dest="data_path")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")

    # 分词器
    p.add_argument("--tokenizer-path", type=str, default=None, dest="tokenizer_path")

    # 模型结构
    p.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    p.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    # nargs="?" + const="1"：只写 --use-moe 时当作 "1"（True），写 --use-moe 0 时为 "0"（False）
    p.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")
    _add_attention_parser_args(p)

    # 恢复与续训
    p.add_argument("--from-weight", type=str, default=None, dest="from_weight")
    p.add_argument("--from-resume", nargs="?", const="1", default=None, dest="from_resume", help="0/1 或省略即 1")

    # 实验与工具
    p.add_argument("--use-swanlab", nargs="?", const="1", default=None, dest="use_swanlab", help="启用 swanlab")
    p.add_argument("--swanlab-project", type=str, default=None, dest="swanlab_project")
    p.add_argument("--use-compile", nargs="?", const="1", default=None, dest="use_compile", help="0/1 或省略即 1")

    # DeepSpeed
    p.add_argument("--use-deepspeed", nargs="?", const="1", default=None, dest="use_deepspeed", help="启用 DeepSpeed")
    p.add_argument("--deepspeed-config", type=str, default=None, dest="deepspeed_config", help="DeepSpeed 配置文件路径（json/yaml）")
    p.add_argument("--deepspeed-zero-stage", type=int, default=None, dest="deepspeed_zero_stage", choices=[0, 1, 2])
    p.add_argument("--deepspeed-offload-optimizer", nargs="?", const="1", default=None, dest="deepspeed_offload_optimizer", help="启用 DeepSpeed CPU optimizer offload")
    p.add_argument("--deepspeed-tensor-parallel-size", type=int, default=None, dest="deepspeed_tensor_parallel_size")

    # 调试相关（不覆盖配置文件，专门用来临时开启 debug 模式）
    p.add_argument("--debug", nargs="?", const="1", default=None, dest="debug", help="启用 debug 模式（小数据集+多日志）")

    return p


def _build_infer_parser() -> argparse.ArgumentParser:
    """构建推理/对话命令行解析器。所有参数 default=None，没传则不覆盖配置。"""
    p = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径 (json/yaml)")

    # 模型加载
    p.add_argument("--tokenizer-path", type=str, default=None, dest="tokenizer_path")
    p.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    p.add_argument("--weight", type=str, default=None)
    p.add_argument("--lora-weight", type=str, default=None, dest="lora_weight")
    p.add_argument("--model-config-path", type=str, default=None, dest="model_config_path")
    p.add_argument("--hf-model-dir", type=str, default=None, dest="hf_model_dir")

    # 模型结构
    p.add_argument("--hidden-act", type=str, default=None, dest="hidden_act")
    p.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    p.add_argument("--intermediate-size", type=int, default=None, dest="intermediate_size")
    p.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")
    p.add_argument("--num-attention-heads", type=int, default=None, dest="num_attention_heads")
    p.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    p.add_argument("--group-num", type=int, default=None, dest="group_num")
    _add_attention_parser_args(p)
    p.add_argument("--vocab-size", type=int, default=None, dest="vocab_size")
    p.add_argument("--rms-norm-eps", type=float, default=None, dest="rms_norm_eps")
    p.add_argument("--rope-base", type=int, default=None, dest="rope_base")
    p.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")
    p.add_argument("--flash-attention", nargs="?", const="1", default=None, dest="flash_attention", help="0/1 或省略即 1")
    p.add_argument("--num-experts-per-token", type=int, default=None, dest="num_experts_per_token")
    p.add_argument("--num-routed-experts", type=int, default=None, dest="num_routed_experts")
    p.add_argument("--num-shared-experts", type=int, default=None, dest="num_shared_experts")
    p.add_argument("--scoring-function", type=str, default=None, dest="scoring_function")
    p.add_argument("--aux-loss-alpha", type=float, default=None, dest="aux_loss_alpha")
    p.add_argument("--seq-aux", nargs="?", const="1", default=None, dest="seq_aux", help="0/1 或省略即 1")
    p.add_argument("--norm-topk-prob", nargs="?", const="1", default=None, dest="norm_topk_prob", help="0/1 或省略即 1")
    p.add_argument("--capacity-factor", type=float, default=None, dest="capacity_factor")

    # 推理与生成
    p.add_argument("--inference-rope-scaling", nargs="?", const="1", default=None, dest="inference_rope_scaling", help="启用RoPE外推")
    p.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None, dest="top_p")
    p.add_argument("--vllm-runner", type=str, default=None, dest="vllm_runner")
    p.add_argument("--vllm-model-impl", type=str, default=None, dest="vllm_model_impl")
    p.add_argument("--vllm-dtype", type=str, default=None, dest="vllm_dtype")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=None, dest="vllm_tensor_parallel_size")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None, dest="vllm_gpu_memory_utilization")
    p.add_argument("--vllm-max-model-len", type=int, default=None, dest="vllm_max_model_len")
    p.add_argument("--vllm-max-num-seqs", type=int, default=None, dest="vllm_max_num_seqs")
    p.add_argument("--vllm-enforce-eager", nargs="?", const="1", default=None, dest="vllm_enforce_eager", help="0/1 或省略即 1")
    p.add_argument("--vllm-trust-remote-code", nargs="?", const="1", default=None, dest="vllm_trust_remote_code", help="0/1 或省略即 1")

    # 对话与展示
    p.add_argument("--historys", type=int, default=None)
    p.add_argument("--show-speed", type=int, default=None, dest="show_speed")

    # 设备
    p.add_argument("--device", type=str, default=None)

    return p


def _build_sft_parser() -> argparse.ArgumentParser:
    """构建 Full SFT 命令行解析器。所有参数 default=None，没传则不覆盖配置。"""
    p = argparse.ArgumentParser(description="MiniMind Full SFT")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径 (json/yaml)")

    # 保存与输出
    p.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    p.add_argument("--save-weight", type=str, default=None, dest="save_weight")
    p.add_argument("--save-interval", type=int, default=None, dest="save_interval")
    p.add_argument("--log-interval", type=int, default=None, dest="log_interval")

    # 训练超参
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    p.add_argument("--accumulation-steps", type=int, default=None, dest="accumulation_steps")
    p.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")

    # 设备与精度
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"])

    # 数据
    p.add_argument("--data-path", type=str, default=None, dest="data_path")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")

    # 分词器
    p.add_argument("--tokenizer-path", type=str, default=None, dest="tokenizer_path")

    # 模型结构
    p.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    p.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    p.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")
    _add_attention_parser_args(p)

    # 恢复与续训
    p.add_argument("--from-weight", type=str, default=None, dest="from_weight")
    p.add_argument("--from-resume", nargs="?", const="1", default=None, dest="from_resume", help="0/1 或省略即 1")

    # 实验与工具
    p.add_argument("--use-swanlab", nargs="?", const="1", default=None, dest="use_swanlab", help="启用 swanlab")
    p.add_argument("--swanlab-project", type=str, default=None, dest="swanlab_project")
    p.add_argument("--use-compile", nargs="?", const="1", default=None, dest="use_compile", help="0/1 或省略即 1")

    return p


def _build_dpo_parser() -> argparse.ArgumentParser:
    """构建 DPO 命令行解析器。所有参数 default=None，没传则不覆盖配置。"""
    p = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径 (json/yaml)")

    # 保存与输出
    p.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    p.add_argument("--save-weight", type=str, default=None, dest="save_weight")
    p.add_argument("--save-interval", type=int, default=None, dest="save_interval")
    p.add_argument("--log-interval", type=int, default=None, dest="log_interval")

    # 训练超参
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    p.add_argument("--accumulation-steps", type=int, default=None, dest="accumulation_steps")
    p.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")

    # 设备与精度
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"])

    # 数据
    p.add_argument("--data-path", type=str, default=None, dest="data_path")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")

    # 分词器
    p.add_argument("--tokenizer-path", type=str, default=None, dest="tokenizer_path")

    # 模型结构
    p.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    p.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    p.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")
    _add_attention_parser_args(p)

    # 恢复与续训
    p.add_argument("--from-weight", type=str, default=None, dest="from_weight")
    p.add_argument("--from-resume", nargs="?", const="1", default=None, dest="from_resume", help="0/1 或省略即 1")

    # DPO 专用
    p.add_argument("--beta", type=float, default=None)

    # 实验与工具
    p.add_argument("--use-swanlab", nargs="?", const="1", default=None, dest="use_swanlab", help="启用 swanlab")
    p.add_argument("--swanlab-project", type=str, default=None, dest="swanlab_project")
    p.add_argument("--use-compile", nargs="?", const="1", default=None, dest="use_compile", help="0/1 或省略即 1")

    return p


def _build_grpo_parser() -> argparse.ArgumentParser:
    """构建 GRPO 命令行解析器。所有参数 default=None，没传则不覆盖配置。"""
    p = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径 (json/yaml)")

    # 保存与输出
    p.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    p.add_argument("--save-weight", type=str, default=None, dest="save_weight")
    p.add_argument("--save-interval", type=int, default=None, dest="save_interval")
    p.add_argument("--log-interval", type=int, default=None, dest="log_interval")

    # 训练超参
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    p.add_argument("--accumulation-steps", type=int, default=None, dest="accumulation_steps")
    p.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")

    # 设备与精度
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"])

    # 数据
    p.add_argument("--data-path", type=str, default=None, dest="data_path")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")
    p.add_argument("--max-gen-len", type=int, default=None, dest="max_gen_len")
    p.add_argument("--num-generations", type=int, default=None, dest="num_generations")

    # 分词器
    p.add_argument("--tokenizer-path", type=str, default=None, dest="tokenizer_path")

    # 模型结构
    p.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    p.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    p.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")
    _add_attention_parser_args(p)

    # 恢复与续训
    p.add_argument("--from-resume", nargs="?", const="1", default=None, dest="from_resume", help="0/1 或省略即 1")

    # GRPO 专用
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--reasoning", type=int, default=None)
    # p.add_argument("--reward-model-path", type=str, default=None, dest="reward_model_path")
    p.add_argument("--reward-model-name", type=str, default=None, dest="reward_model_name")
    p.add_argument("--reward-model-tokenizer-name", type=str, default=None, dest="reward_model_tokenizer_name")

    # 实验与工具
    p.add_argument("--use-swanlab", nargs="?", const="1", default=None, dest="use_swanlab", help="启用 swanlab")
    p.add_argument("--swanlab-project", type=str, default=None, dest="swanlab_project")
    p.add_argument("--use-compile", nargs="?", const="1", default=None, dest="use_compile", help="0/1 或省略即 1")

    return p


def _build_distillation_parser() -> argparse.ArgumentParser:
    """构建 On-policy 蒸馏命令行解析器。所有参数 default=None，没传则不覆盖配置。"""
    p = argparse.ArgumentParser(description="MiniMind On-policy Distillation (白盒蒸馏)")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径 (json/yaml)")

    # 保存与输出
    p.add_argument("--save-dir", type=str, default=None, dest="save_dir")
    p.add_argument("--save-weight", type=str, default=None, dest="save_weight")
    p.add_argument("--save-interval", type=int, default=None, dest="save_interval")
    p.add_argument("--log-interval", type=int, default=None, dest="log_interval")

    # 训练超参
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    p.add_argument("--accumulation-steps", type=int, default=None, dest="accumulation_steps")
    p.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")

    # 设备与精度
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"])

    # 数据
    p.add_argument("--data-path", type=str, default=None, dest="data_path")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    p.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")

    # 分词器
    p.add_argument("--tokenizer-path", type=str, default=None, dest="tokenizer_path")

    # 模型结构
    p.add_argument("--hidden-size", type=int, default=None, dest="hidden_size")
    p.add_argument("--num-hidden-layers", type=int, default=None, dest="num_hidden_layers")
    p.add_argument("--use-moe", nargs="?", const="1", default=None, dest="use_moe", help="0/1 或省略即 1")
    _add_attention_parser_args(p)

    # 恢复与续训
    p.add_argument("--from-weight", type=str, default=None, dest="from_weight")
    p.add_argument("--from-resume", nargs="?", const="1", default=None, dest="from_resume", help="0/1 或省略即 1")

    # 实验与工具
    p.add_argument("--use-swanlab", nargs="?", const="1", default=None, dest="use_swanlab", help="启用 swanlab")
    p.add_argument("--swanlab-project", type=str, default=None, dest="swanlab_project")
    p.add_argument("--use-compile", nargs="?", const="1", default=None, dest="use_compile", help="0/1 或省略即 1")

    return p


def get_pretrain_config(args: list[str] | None = None) -> PretrainConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 PretrainConfig 实例。

    参数：
      args：通常不传，表示用 sys.argv（即当前进程的命令行）；传了则用这份列表解析，便于测试或二次封装。

    步骤简述：
      1. 用 PretrainConfig() 无参构造：会用到 schema 里的默认值，并自动读 .env 和 TRAIN_* 环境变量，得到第一版字典。
      2. 若命令行带了 --config 且文件存在：用该文件内容覆盖字典里同名字段（只覆盖文件中出现的、且值非 None 的）。
      3. 若命令行带了其它参数（如 --batch-size 64）：再覆盖字典里对应字段；没带的保持上一步的值。
      4. 用最终字典构造 PretrainConfig 并返回。
    """
    parser = _build_pretrain_parser()
    parsed = parser.parse_args(args)

    # 第一层：默认值 + .env + 环境变量（TRAIN_*）。无参构造时 BaseSettings 会自动读 env
    config_dict = PretrainConfig().model_dump()

    # 第二层：配置文件。只覆盖 config_dict 里已有的 key，且文件里值非 None 才覆盖
    if parsed.config is not None and parsed.config.exists():
        file_dict = _load_json_or_yaml(parsed.config)
        for k, v in file_dict.items():
            if k in config_dict and v is not None:
                config_dict[k] = v

    # 第三层：命令行。只处理用户真的传了的参数（getattr 得到 None 表示没传，不覆盖）
    for key in list(config_dict.keys()):
        val = getattr(parsed, key, None)
        if val is None:
            continue
        # 命令行里布尔类参数可能是字符串 "0"/"1"，转成 bool
        if key in ("use_moe", "from_resume", "use_swanlab", "use_compile", "use_deepspeed", "deepspeed_offload_optimizer"):
            val = _bool_opt(val)
        config_dict[key] = val

    return PretrainConfig(**config_dict)


def get_infer_config(args: list[str] | None = None) -> InferConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 InferConfig 实例。

    用法：cfg = get_infer_config()，然后 cfg.tokenizer_path、cfg.weight、cfg.max_new_tokens 等。
    """
    parser = _build_infer_parser()
    parsed = parser.parse_args(args)

    config_dict = InferConfig().model_dump()

    if parsed.config is not None and parsed.config.exists():
        file_dict = _load_json_or_yaml(parsed.config)
        for k, v in file_dict.items():
            if k in config_dict and v is not None:
                config_dict[k] = v

    infer_bool_keys = (
        "use_moe",
        "flash_attention",
        "seq_aux",
        "norm_topk_prob",
        "inference_rope_scaling",
        "vllm_enforce_eager",
        "vllm_trust_remote_code",
    )
    for key in list(config_dict.keys()):
        val = getattr(parsed, key, None)
        if val is None:
            continue
        if key in infer_bool_keys:
            val = _bool_opt(val)
        config_dict[key] = val

    return InferConfig(**config_dict)


def get_sft_config(args: list[str] | None = None) -> SFTConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 SFTConfig 实例。

    用法：cfg = get_sft_config()，然后 cfg.batch_size、cfg.to_lm_config_kwargs() 等。
    """
    parser = _build_sft_parser()
    parsed = parser.parse_args(args)

    config_dict = SFTConfig().model_dump()

    if parsed.config is not None and parsed.config.exists():
        file_dict = _load_json_or_yaml(parsed.config)
        for k, v in file_dict.items():
            if k in config_dict and v is not None:
                config_dict[k] = v

    sft_bool_keys = ("use_moe", "from_resume", "use_swanlab", "use_compile")
    for key in list(config_dict.keys()):
        val = getattr(parsed, key, None)
        if val is None:
            continue
        if key in sft_bool_keys:
            val = _bool_opt(val)
        config_dict[key] = val

    return SFTConfig(**config_dict)


def get_dpo_config(args: list[str] | None = None) -> DPOConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 DPOConfig 实例。

    用法：cfg = get_dpo_config()，然后 cfg.batch_size、cfg.beta、cfg.to_lm_config_kwargs() 等。
    """
    parser = _build_dpo_parser()
    parsed = parser.parse_args(args)

    config_dict = DPOConfig().model_dump()

    if parsed.config is not None and parsed.config.exists():
        file_dict = _load_json_or_yaml(parsed.config)
        for k, v in file_dict.items():
            if k in config_dict and v is not None:
                config_dict[k] = v

    dpo_bool_keys = ("use_moe", "from_resume", "use_swanlab", "use_compile")
    for key in list(config_dict.keys()):
        val = getattr(parsed, key, None)
        if val is None:
            continue
        if key in dpo_bool_keys:
            val = _bool_opt(val)
        config_dict[key] = val

    return DPOConfig(**config_dict)


def get_grpo_config(args: list[str] | None = None) -> GRPOConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 GRPOConfig 实例。

    用法：cfg = get_grpo_config()，然后 cfg.batch_size、cfg.beta、cfg.to_lm_config_kwargs() 等。
    """
    parser = _build_grpo_parser()
    parsed = parser.parse_args(args)

    config_dict = GRPOConfig().model_dump()

    if parsed.config is not None and parsed.config.exists():
        file_dict = _load_json_or_yaml(parsed.config)
        for k, v in file_dict.items():
            if k in config_dict and v is not None:
                config_dict[k] = v

    grpo_bool_keys = ("use_moe", "from_resume", "use_swanlab", "use_compile")
    for key in list(config_dict.keys()):
        val = getattr(parsed, key, None)
        if val is None:
            continue
        if key in grpo_bool_keys:
            val = _bool_opt(val)
        config_dict[key] = val

    return GRPOConfig(**config_dict)


def get_distillation_config(args: list[str] | None = None) -> DistillationConfig:
    """
    按「默认 → 配置文件 → 命令行」三层叠加，返回一个 DistillationConfig 实例。

    用法：cfg = get_distillation_config()，然后 cfg.batch_size、cfg.to_lm_config_kwargs() 等。
    """
    parser = _build_distillation_parser()
    parsed = parser.parse_args(args)

    config_dict = DistillationConfig().model_dump()

    if parsed.config is not None and parsed.config.exists():
        file_dict = _load_json_or_yaml(parsed.config)
        for k, v in file_dict.items():
            if k in config_dict and v is not None:
                config_dict[k] = v

    distill_bool_keys = ("use_moe", "from_resume", "use_swanlab", "use_compile")
    for key in list(config_dict.keys()):
        val = getattr(parsed, key, None)
        if val is None:
            continue
        if key in distill_bool_keys:
            val = _bool_opt(val)
        config_dict[key] = val

    return DistillationConfig(**config_dict)
