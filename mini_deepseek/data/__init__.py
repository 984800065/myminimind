"""
本项目内部使用的数据集模块。

说明：
  - 训练脚本统一从 `mini_deepseek.data` 导入 Dataset 实现。
  - 顶层 `dataset/` 目录作为数据文件入口使用，可继续来自外部挂载目录。
"""

from .lm_dataset import DPODataset, PretrainDataset, RLAIFDataset, SFTDataset

__all__ = ["DPODataset", "PretrainDataset", "RLAIFDataset", "SFTDataset"]
