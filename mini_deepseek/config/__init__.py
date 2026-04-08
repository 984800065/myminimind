"""
配置包：训练与推理参数的定义与加载。

  - PretrainConfig / get_pretrain_config()：预训练配置。
  - SFTConfig / get_sft_config()：Full SFT 配置。
  - DPOConfig / get_dpo_config()：DPO 配置。
  - DistillationConfig / get_distillation_config()：On-policy 白盒蒸馏配置。
  - GRPOConfig / get_grpo_config()：GRPO 配置。
  - InferConfig / get_infer_config()：推理/对话配置。

用法：
  from mini_deepseek.config import get_pretrain_config, get_sft_config, get_dpo_config, get_distillation_config, get_infer_config
  cfg = get_pretrain_config()       # 预训练
  cfg = get_sft_config()           # Full SFT
  cfg = get_dpo_config()           # DPO
  cfg = get_distillation_config()  # 蒸馏
  infer_cfg = get_infer_config()   # 推理
"""

from .load import get_distillation_config, get_dpo_config, get_grpo_config, get_infer_config, get_pretrain_config, get_sft_config
from .schema import DistillationConfig, DPOConfig, GRPOConfig, InferConfig, PretrainConfig, SFTConfig

__all__ = ["DPOConfig", "DistillationConfig", "GRPOConfig", "InferConfig", "PretrainConfig", "SFTConfig", "get_dpo_config", "get_distillation_config", "get_grpo_config", "get_pretrain_config", "get_sft_config", "get_infer_config"]
