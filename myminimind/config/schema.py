"""
预训练配置的 Pydantic 模型：类型、校验、文档。

作用：
  - 把所有训练相关参数写在一个类里，带类型和校验（如 batch_size > 0）。
  - 配合 pydantic-settings：无参构造时自动从 .env 和环境变量（TRAIN_*）读值。
  - 和 load.get_*_config() 一起用：对应入口会按「默认 → 配置文件 → 命令行」逐层覆盖，最后得到这个类的实例。

对应 train_pretrain 里原来的 argparse 参数，字段名和含义一一对应。
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_device() -> str:
    """根据是否有 GPU 返回默认设备字符串，供 device 字段的 default_factory 使用。"""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class PretrainConfig(BaseSettings):
    """
    预训练配置：可从 .env、环境变量（TRAIN_*）、配置文件、命令行加载，后者覆盖前者。

    使用方式：不要手写 PretrainConfig(xxx)，而是用 get_pretrain_config() 得到实例，例如：
      cfg = get_pretrain_config()
      cfg.batch_size
      lm_config = MiniMindConfig(**cfg.to_lm_config_kwargs())
    """

    # ----- 下面这一块是 pydantic-settings 的配置，控制「从环境变量怎么读」 -----
    model_config = SettingsConfigDict(
        env_prefix="TRAIN_",  # 环境变量前缀：TRAIN_BATCH_SIZE、TRAIN_LEARNING_RATE 等会映射到对应字段
        env_nested_delimiter="__",  # 嵌套字段用双下划线，如 TRAIN_OPTIM__LR（当前 schema 无嵌套，可忽略）
        extra="ignore",  # 环境变量里多出来的 key 不报错，直接忽略
        str_strip_whitespace=True,  # 字符串自动去首尾空格
    )

    # ----- 保存与输出 -----
    save_dir: str = Field("out", description="模型/checkpoint 保存目录")
    save_weight: str = Field("pretrain", description="保存权重文件名前缀")
    save_interval: int = Field(1000, gt=0, description="每 N step 保存一次")
    log_interval: int = Field(100, gt=0, description="每 N step 打一次日志")

    # ----- 训练超参 -----
    epochs: int = Field(1, ge=1, description="训练轮数")
    batch_size: int = Field(8, gt=0, description="batch size")
    learning_rate: float = Field(5e-4, gt=0.0, description="初始学习率")
    accumulation_steps: int = Field(8, ge=1, description="梯度累积步数")
    grad_clip: float = Field(1.0, ge=0.0, description="梯度裁剪阈值")

    # ----- 设备与精度 -----
    # default_factory：用函数在「每次创建实例」时算默认值，这里用来根据有没有 GPU 选 cuda:0 或 cpu
    device: str = Field(default_factory=_default_device, description="训练设备，如 cuda:0 / cpu")
    # Literal 表示只能是这两个字符串之一，写错会校验报错
    dtype: Literal["bfloat16", "float16"] = Field("bfloat16", description="混合精度类型")

    # ----- 数据 -----
    data_path: str = Field("./dataset/pretrain_hq.jsonl", description="预训练数据路径（jsonl）")
    num_workers: int = Field(8, ge=0, description="DataLoader 线程数")
    max_seq_len: int = Field(1024, gt=0, description="训练时最大截断长度（token）")

    # ----- 分词器 -----
    tokenizer_path: str = Field("./myminimind/config/tokenizer", description="分词器路径")

    # ----- 模型结构（与 MiniMindConfig 对齐） -----
    hidden_size: int = Field(1024, gt=0, description="隐藏层维度")
    num_hidden_layers: int = Field(8, gt=0, description="隐藏层数量")
    use_moe: bool = Field(True, description="是否使用 MoE 架构")

    # ----- 恢复与续训 -----
    from_weight: str = Field("none", description="从哪个权重继续训，none 表示从头")
    from_resume: bool = Field(False, description="是否自动检测 checkpoint 并续训")

    # ----- 实验与工具 -----
    use_swanlab: bool = Field(True, description="是否使用 swanlab 记录")
    swanlab_project: str = Field("MiniMind-Pretrain", description="swanlab 项目名")
    use_compile: bool = Field(False, description="是否使用 torch.compile 加速")

    # ------ debug -----
    debug: bool = Field(False, description="是否开启 debug 模式（小数据、频繁保存、详细日志）")

    def to_lm_config_kwargs(self) -> dict:
        """
        只抽出「模型结构」相关字段，方便直接传给 MiniMindConfig。

        用法：lm_config = MiniMindConfig(**cfg.to_lm_config_kwargs())
        这样训练配置和模型配置解耦，PretrainConfig 管训练，MiniMindConfig 管模型结构。
        """
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "use_moe": self.use_moe,
        }


class InferConfig(BaseSettings):
    """
    推理/对话配置：可从 .env、环境变量（INFER_*）、配置文件、命令行加载，后者覆盖前者。

    对应推理脚本里的 argparse 参数，字段名和含义一一对应。
    使用方式：用 get_infer_config() 得到实例。
    """

    model_config = SettingsConfigDict(
        env_prefix="INFER_",
        env_nested_delimiter="__",
        extra="ignore",
        str_strip_whitespace=True,
    )

    # ----- 模型加载 -----
    tokenizer_path: str = Field("./myminimind/config/tokenizer", description="分词器路径")
    save_dir: str = Field("out", description="模型权重目录")
    weight: str = Field("pretrain", description="权重名称前缀（pretrain, full_sft, dpo, rlhf, reason, ppo_actor, grpo, spo）")
    lora_weight: str = Field("None", description="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")

    # ----- 模型结构（与 MiniMindConfig 对齐） -----
    hidden_size: int = Field(1024, gt=0, description="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    num_hidden_layers: int = Field(8, gt=0, description="隐藏层数量（Small/MoE=8, Base=16）")
    use_moe: bool = Field(True, description="是否使用MoE架构")

    # ----- 推理与生成 -----
    inference_rope_scaling: bool = Field(False, description="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    max_new_tokens: int = Field(1024, gt=0, description="最大生成长度（注意：并非模型实际长文本能力）")
    temperature: float = Field(0.85, ge=0.0, le=2.0, description="生成温度，控制随机性（0-1，越大越随机）")
    top_p: float = Field(0.85, ge=0.0, le=1.0, description="nucleus采样阈值（0-1）")

    # ----- 对话与展示 -----
    historys: int = Field(0, ge=0, description="携带历史对话轮数（需为偶数，0表示不携带历史）")
    show_speed: int = Field(1, ge=0, description="显示decode速度（tokens/s）")

    # ----- 设备 -----
    device: str = Field(default_factory=_default_device, description="运行设备")

    def to_lm_config_kwargs(self) -> dict:
        """抽出模型结构相关字段，传给 MiniMindConfig。"""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "use_moe": self.use_moe,
            "inference_rope_scaling": self.inference_rope_scaling,
        }


class SFTConfig(BaseSettings):
    """
    Full SFT 配置：可从 .env、环境变量（SFT_*）、配置文件、命令行加载，后者覆盖前者。

    使用方式：用 get_sft_config() 得到实例，例如：
      cfg = get_sft_config()
      lm_config = MiniMindConfig(**cfg.to_lm_config_kwargs())
    """

    model_config = SettingsConfigDict(
        env_prefix="SFT_",
        env_nested_delimiter="__",
        extra="ignore",
        str_strip_whitespace=True,
    )

    # ----- 保存与输出 -----
    save_dir: str = Field("./out", description="模型/checkpoint 保存目录")
    save_weight: str = Field("full_sft", description="保存权重文件名前缀")
    save_interval: int = Field(1000, gt=0, description="每 N step 保存一次")
    log_interval: int = Field(100, gt=0, description="每 N step 打一次日志")

    # ----- 训练超参 -----
    epochs: int = Field(1, ge=1, description="训练轮数")
    batch_size: int = Field(8, gt=0, description="batch size")
    learning_rate: float = Field(1e-6, gt=0.0, description="初始学习率")
    accumulation_steps: int = Field(1, ge=1, description="梯度累积步数")
    grad_clip: float = Field(1.0, ge=0.0, description="梯度裁剪阈值")

    # ----- 设备与精度 -----
    device: str = Field(default_factory=_default_device, description="训练设备，如 cuda:0 / cpu")
    dtype: Literal["bfloat16", "float16"] = Field("bfloat16", description="混合精度类型")

    # ----- 数据 -----
    data_path: str = Field("./dataset/sft_512.jsonl", description="SFT 训练数据路径（jsonl）")
    num_workers: int = Field(8, ge=0, description="DataLoader 线程数")
    max_seq_len: int = Field(1024, gt=0, description="训练时最大截断长度（token）")

    # ----- 分词器 -----
    tokenizer_path: str = Field("./myminimind/config/tokenizer", description="分词器路径")

    # ----- 模型结构（与 MiniMindConfig 对齐） -----
    hidden_size: int = Field(1024, gt=0, description="隐藏层维度")
    num_hidden_layers: int = Field(8, gt=0, description="隐藏层数量")
    use_moe: bool = Field(True, description="是否使用 MoE 架构")

    # ----- 恢复与续训 -----
    from_weight: str = Field("pretrain", description="从哪个权重继续训，none 表示从头")
    from_resume: bool = Field(False, description="是否自动检测 checkpoint 并续训")

    # ----- 实验与工具 -----
    use_swanlab: bool = Field(True, description="是否使用 swanlab 记录")
    swanlab_project: str = Field("MiniMind-Full-SFT", description="swanlab 项目名")
    use_compile: bool = Field(False, description="是否使用 torch.compile 加速")

    def to_lm_config_kwargs(self) -> dict:
        """抽出模型结构相关字段，传给 MiniMindConfig。"""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "use_moe": self.use_moe,
        }


class DPOConfig(BaseSettings):
    """
    DPO (Direct Preference Optimization) 配置：可从 .env、环境变量（DPO_*）、配置文件、命令行加载，后者覆盖前者。

    使用方式：用 get_dpo_config() 得到实例，例如：
      cfg = get_dpo_config()
      lm_config = MiniMindConfig(**cfg.to_lm_config_kwargs())
    """

    model_config = SettingsConfigDict(
        env_prefix="DPO_",
        env_nested_delimiter="__",
        extra="ignore",
        str_strip_whitespace=True,
    )

    # ----- 保存与输出 -----
    save_dir: str = Field("./out", description="模型/checkpoint 保存目录")
    save_weight: str = Field("dpo", description="保存权重文件名前缀")
    save_interval: int = Field(100, gt=0, description="每 N step 保存一次")
    log_interval: int = Field(100, gt=0, description="每 N step 打一次日志")

    # ----- 训练超参 -----
    epochs: int = Field(1, ge=1, description="训练轮数")
    batch_size: int = Field(4, gt=0, description="batch size")
    learning_rate: float = Field(4e-8, gt=0.0, description="初始学习率（建议<=5e-8 避免遗忘）")
    accumulation_steps: int = Field(1, ge=1, description="梯度累积步数")
    grad_clip: float = Field(1.0, ge=0.0, description="梯度裁剪阈值")

    # ----- 设备与精度 -----
    device: str = Field(default_factory=_default_device, description="训练设备，如 cuda:0 / cpu")
    dtype: Literal["bfloat16", "float16"] = Field("bfloat16", description="混合精度类型")

    # ----- 数据 -----
    data_path: str = Field("./dataset/dpo.jsonl", description="DPO 训练数据路径（jsonl）")
    num_workers: int = Field(8, ge=0, description="DataLoader 线程数")
    max_seq_len: int = Field(1024, gt=0, description="训练时最大截断长度（token）")

    # ----- 分词器 -----
    tokenizer_path: str = Field("./myminimind/config/tokenizer", description="分词器路径")

    # ----- 模型结构（与 MiniMindConfig 对齐） -----
    hidden_size: int = Field(512, gt=0, description="隐藏层维度")
    num_hidden_layers: int = Field(8, gt=0, description="隐藏层数量")
    use_moe: bool = Field(False, description="是否使用 MoE 架构")

    # ----- 恢复与续训 -----
    from_weight: str = Field("full_sft", description="基于哪个权重训练")
    from_resume: bool = Field(False, description="是否自动检测 checkpoint 并续训")

    # ----- DPO 专用 -----
    beta: float = Field(0.1, gt=0.0, description="DPO 中的 beta 参数")

    # ----- 实验与工具 -----
    use_swanlab: bool = Field(False, description="是否使用 swanlab 记录")
    swanlab_project: str = Field("MiniMind-DPO", description="swanlab 项目名")
    use_compile: bool = Field(False, description="是否使用 torch.compile 加速")

    def to_lm_config_kwargs(self) -> dict:
        """抽出模型结构相关字段，传给 MiniMindConfig。"""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "use_moe": self.use_moe,
        }


class GRPOConfig(BaseSettings):
    """
    GRPO (Group Relative Policy Optimization) 配置：可从 .env、环境变量（GRPO_*）、配置文件、命令行加载，后者覆盖前者。

    使用方式：用 get_grpo_config() 得到实例，例如：
      cfg = get_grpo_config()
      lm_config = MiniMindConfig(**cfg.to_lm_config_kwargs())
    """

    model_config = SettingsConfigDict(
        env_prefix="GRPO_",
        env_nested_delimiter="__",
        extra="ignore",
        str_strip_whitespace=True,
    )

    # ----- 保存与输出 -----
    save_dir: str = Field("./out", description="模型/checkpoint 保存目录")
    save_weight: str = Field("grpo", description="保存权重文件名前缀")
    save_interval: int = Field(10, gt=0, description="每 N step 保存一次")
    log_interval: int = Field(1, gt=0, description="每 N step 打一次日志")

    # ----- 训练超参 -----
    epochs: int = Field(1, ge=1, description="训练轮数")
    batch_size: int = Field(2, gt=0, description="batch size")
    learning_rate: float = Field(8e-8, gt=0.0, description="初始学习率")
    accumulation_steps: int = Field(1, ge=1, description="梯度累积步数")
    grad_clip: float = Field(1.0, ge=0.0, description="梯度裁剪阈值")

    # ----- 设备与精度 -----
    device: str = Field(default_factory=_default_device, description="训练设备，如 cuda:0 / cpu")
    dtype: Literal["bfloat16", "float16"] = Field("bfloat16", description="混合精度类型")

    # ----- 数据 -----
    data_path: str = Field("./dataset/rlaif-mini.jsonl", description="RLAIF 训练数据路径（jsonl）")
    num_workers: int = Field(8, ge=0, description="DataLoader 线程数")
    max_seq_len: int = Field(66, gt=0, description="Prompt 最大长度")
    max_gen_len: int = Field(1536, gt=0, description="生成的最大长度")
    num_generations: int = Field(8, gt=0, description="每个 prompt 生成的样本数")

    # ----- 分词器 -----
    tokenizer_path: str = Field("./myminimind/config/tokenizer", description="分词器路径")

    # ----- 模型结构（与 MiniMindConfig 对齐） -----
    hidden_size: int = Field(512, gt=0, description="隐藏层维度")
    num_hidden_layers: int = Field(8, gt=0, description="隐藏层数量")
    use_moe: bool = Field(False, description="是否使用 MoE 架构")

    # ----- 恢复与续训 -----
    from_resume: bool = Field(False, description="是否自动检测 checkpoint 并续训")

    # ----- GRPO 专用 -----
    beta: float = Field(0.02, gt=0.0, description="KL 惩罚系数")
    reasoning: int = Field(1, ge=0, le=1, description="推理模型类型（0=普通模型，1=推理模型）")
    # reward_model_path: str = Field(
    #     "../../internlm2-1_8b-reward",
    #     description="Reward 模型路径",
    # )
    reward_model_name: str = Field("internlm/internlm2-1_8b-reward", description="Reward 模型名称")
    reward_model_tokenizer_name: str = Field("internlm/internlm2-1_8b-reward", description="Reward 模型分词器名称")

    # ----- 实验与工具 -----
    use_swanlab: bool = Field(False, description="是否使用 swanlab 记录")
    swanlab_project: str = Field("MiniMind-GRPO", description="swanlab 项目名")
    use_compile: bool = Field(False, description="是否使用 torch.compile 加速")

    def to_lm_config_kwargs(self) -> dict:
        """抽出模型结构相关字段，传给 MiniMindConfig（供 policy / reference 模型使用）。"""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "use_moe": self.use_moe,
            # GRPO 里 max_seq_len 通常为 prompt+生成的总长度
            "max_seq_len": self.max_seq_len + self.max_gen_len,
        }


class DistillationConfig(BaseSettings):
    """
    On-policy 白盒蒸馏配置：可从 .env、环境变量（DISTILL_*）、配置文件、命令行加载，后者覆盖前者。

    使用方式：用 get_distillation_config() 得到实例，例如：
      cfg = get_distillation_config()
      lm_config = MiniMindConfig(**cfg.to_lm_config_kwargs())
    """

    model_config = SettingsConfigDict(
        env_prefix="DISTILL_",
        env_nested_delimiter="__",
        extra="ignore",
        str_strip_whitespace=True,
    )

    # ----- 保存与输出 -----
    save_dir: str = Field("./out", description="模型/checkpoint 保存目录")
    save_weight: str = Field("distill", description="保存权重文件名前缀")
    save_interval: int = Field(1000, gt=0, description="每 N step 保存一次")
    log_interval: int = Field(100, gt=0, description="每 N step 打一次日志")

    # ----- 训练超参 -----
    epochs: int = Field(2, ge=1, description="训练轮数")
    batch_size: int = Field(16, gt=0, description="batch size")
    learning_rate: float = Field(1e-6, gt=0.0, description="初始学习率")
    accumulation_steps: int = Field(1, ge=1, description="梯度累积步数")
    grad_clip: float = Field(1.0, ge=0.0, description="梯度裁剪阈值")

    # ----- 设备与精度 -----
    device: str = Field(default_factory=_default_device, description="训练设备，如 cuda:0 / cpu")
    dtype: Literal["bfloat16", "float16"] = Field("bfloat16", description="混合精度类型")

    # ----- 数据 -----
    data_path: str = Field("./dataset/sft_mini_512.jsonl", description="蒸馏训练数据路径（jsonl）")
    num_workers: int = Field(8, ge=0, description="DataLoader 线程数")
    max_seq_len: int = Field(340, gt=0, description="训练时最大截断长度（token）")

    # ----- 分词器 -----
    tokenizer_path: str = Field("./myminimind/config/tokenizer", description="分词器路径")

    # ----- 模型结构（与 MiniMindConfig 对齐） -----
    hidden_size: int = Field(512, gt=0, description="隐藏层维度")
    num_hidden_layers: int = Field(8, gt=0, description="隐藏层数量")
    use_moe: bool = Field(False, description="是否使用 MoE 架构")

    # ----- 恢复与续训 -----
    from_weight: str = Field("pretrain", description="基于哪个权重训练，none 表示从头")
    from_resume: bool = Field(False, description="是否自动检测 checkpoint 并续训")

    # ----- 实验与工具 -----
    use_swanlab: bool = Field(False, description="是否使用 swanlab 记录")
    swanlab_project: str = Field("MiniMind-Distillation", description="swanlab 项目名")
    use_compile: bool = Field(False, description="是否使用 torch.compile 加速")

    def to_lm_config_kwargs(self) -> dict:
        """抽出模型结构相关字段，传给 MiniMindConfig。"""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "use_moe": self.use_moe,
        }
