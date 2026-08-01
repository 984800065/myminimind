# MiniDeepSeek

MiniDeepSeek 是一个用于学习和实验小型语言模型训练流程的项目，支持从预训练、监督微调到偏好对齐，并提供 Hugging Face 模型导出和评测能力。

本项目基于 [MiniMind](https://github.com/jingyaogong/minimind) 二次开发。整体训练流程、数据处理方式和工程结构继承自 MiniMind；在此基础上，项目重新组织了 Python 包结构，并扩展了 MLA、MoE、MTP、DeepSpeed、Liger Kernel 和 Hugging Face 兼容等能力。本项目不是 MiniMind 官方仓库。

## 主要能力

- Decoder-only Causal Language Model
- GQA（Grouped Query Attention）与 MLA（Multi-head Latent Attention）
- Dense FFN 与 MoE（Mixture of Experts）
- MTP（Multi-Token Prediction）
- RMSNorm、RoPE 和 LM Head + Cross Entropy 的 eager / Liger 实现
- 单卡、PyTorch DDP 和 DeepSpeed 预训练
- Full SFT、DPO 和 teacher-logit 白盒蒸馏
- Pydantic 配置，以及配置文件、环境变量和命令行分层覆盖
- 原始 checkpoint 导出为 Hugging Face 模型
- Transformers、vLLM 和 lm-evaluation-harness 推理评测
- SwanLab 实验记录

## 环境要求

- Python 3.12 或更高版本
- Linux
- NVIDIA GPU 与可用的 CUDA 环境
- 推荐使用 `uv` 管理环境和依赖

项目默认从 PyTorch CUDA 12.6 软件源安装 `torch` 和 `torchvision`。如果本机 CUDA 环境不同，请修改 `pyproject.toml` 中的 `tool.uv.index` 和 `tool.uv.sources`。

环境安装方式：

```bash
git clone <repository-url>
cd mini-deepseek

uv python install 3.12
uv sync
```

DeepSpeed 也可以单独安装：

```bash
bash scripts/install_deepspeed.sh
```

## 项目结构

```text
mini_deepseek/
├── config/       # 训练、推理配置及命令行解析
├── data/         # 预训练、SFT、DPO 等数据集
├── eval/         # Transformers 与 vLLM 推理
├── model/        # 模型配置、注意力、MoE、MTP、RoPE 等
├── training/     # 预训练、SFT、DPO、蒸馏训练入口
└── utils/        # checkpoint、分布式训练和 DeepSpeed 工具
scripts/          # 训练、导出和 lm-eval 脚本
tests/            # 模型及数据辅助测试
```

## 数据来源于格式 TODO：完善来源部分

### 预训练

支持单个 `.json`、`.jsonl`、`.parquet` 文件，也支持包含这些文件的目录。每条数据必须包含 `text` 字段：

```json
{"text": "这是一条预训练文本。"}
```

### SFT

每条数据包含 `conversations`，消息格式与 Hugging Face chat template 一致：

```json
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好，有什么可以帮助你？"}]}
```

### DPO

每条数据包含一组 `chosen` 和 `rejected` 对话：

```json
{"chosen": [{"role": "user", "content": "解释一下机器学习"}, {"role": "assistant", "content": "机器学习是……"}], "rejected": [{"role": "user", "content": "解释一下机器学习"}, {"role": "assistant", "content": "不知道。"}]}
```

## 配置方式

训练参数按以下优先级合并，后者覆盖前者：

1. 代码默认值和环境变量
2. JSON / YAML 配置文件
3. 命令行参数

例如：

```bash
uv run python -m mini_deepseek.training.train_pretrain \
  --config configs/pretrain.yaml \
  --batch-size 8 \
  --learning-rate 5e-4
```

不同入口使用独立的环境变量前缀，例如预训练使用 `TRAIN_`，SFT 使用 `SFT_`，DPO 使用 `DPO_`。

## 预训练

单卡：

```bash
uv run python -m mini_deepseek.training.train_pretrain \
  --device cuda:0 \
  --data-path ./dataset/pretrain.jsonl \
  --save-dir ./out \
  --save-weight pretrain \
  --use-swanlab false
```

多卡 DDP：

```bash
NGPUS=4 bash scripts/train_pretrain.sh \
  --data-path ./dataset/pretrain.jsonl \
  --batch-size 4
```

多卡 DeepSpeed：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run deepspeed \
  --module mini_deepseek.training.train_pretrain \
  --use-deepspeed true \
  --data-path ./dataset/pretrain.jsonl \
  --save-weight pretrain \
  --batch-size 4
```

可通过 `--deepspeed-zero-stage 0|1|2`、`--deepspeed-offload-optimizer` 和 `--deepspeed-tensor-parallel-size` 调整 DeepSpeed。当前预训练入口仅支持 ZeRO Stage 0、1、2，不能启用 ZeRO Stage 3。

当前预训练入口尚未完成 DeepSpeed Tensor Parallel 的数据切分、全局 batch size 计算和权重合并适配，因此 `deepspeed_tensor_parallel_size` 必须保持为 `1`。

使用外部 DeepSpeed JSON/YAML 时，micro batch、梯度累积、全局 batch、梯度裁剪和 FP16/BF16 精度由项目训练配置统一控制。外部文件缺少这些字段时会自动补齐，显式配置不一致时会在启动阶段报错；外部文件主要用于配置 ZeRO、offload 和通信参数。

`save_interval` 按 micro-batch step 计数，必须大于等于 `accumulation_steps`，并且是 `accumulation_steps` 的整数倍，确保 checkpoint 只在参数更新边界保存。

## SFT、DPO 与蒸馏

```bash
# Full SFT
uv run python -m mini_deepseek.training.train_full_sft \
  --data-path ./dataset/sft.jsonl \
  --from-weight pretrain

# Full SFT with DeepSpeed
CUDA_VISIBLE_DEVICES=0,1 uv run deepspeed \
  --module mini_deepseek.training.train_full_sft \
  --use-deepspeed true \
  --data-path ./dataset/sft.jsonl \
  --from-weight pretrain

# DPO
uv run python -m mini_deepseek.training.train_dpo \
  --data-path ./dataset/dpo.jsonl \
  --from-weight full_sft

# 白盒 logit 蒸馏
uv run python -m mini_deepseek.training.train_distillation \
  --data-path ./dataset/sft.jsonl \
  --from-weight pretrain \
  --teacher-weight full_sft \
  --distill-alpha 0.5 \
  --distill-temperature 2.0
```

Full SFT 的 DeepSpeed、梯度累积、checkpoint 和外部配置约束与预训练一致，当前同样仅支持 ZeRO Stage 0、1、2，且尚未适配 Tensor Parallel。

本项目是用于学习和验证训练链路的 toy project。当前 `SFTDataset` 假设 tokenizer 的 chat template 使用固定的 assistant/EOS 角色边界，并通过手工 token 匹配生成监督标签；复杂多轮对话、用户正文包含角色标记及 tool call 模板尚未系统覆盖，后续会统一重构数据读取和 assistant mask。

Full SFT 会强制读取基础权重或 resume checkpoint 旁的 `.config.json`，并严格校验模型结构、token、RoPE、MoE 和 MTP 配置；配置缺失或不一致时直接报错，不会静默覆盖。SFT 可以单独调整 `data_max_seq_len`，因此上下文长度不参与兼容性比较。

权重名称会自动包含 hidden size、MoE 类型和 attention 类型。`--from-resume true` 用于恢复模型、优化器、学习率调度器和训练进度；`--from-weight` 只用于指定初始化模型权重。

蒸馏入口会冻结 `--teacher-weight` 指定的 teacher，在 SFT assistant token 上混合 hard-label CE 与 temperature-scaled KL。`--distill-alpha` 为 KL 权重，取值范围为 0 到 1。

## 推理 TODO：写的更加清晰一些

使用项目原始 `.pth` 权重：

```bash
uv run python -m mini_deepseek.eval.eval_llm \
  --save-dir ./out \
  --weight full_sft \
  --device cuda:0
```

训练保存原始权重时会同时写入同名 `.config.json`，推理和导出会自动读取其中的模型结构。对于没有 sidecar 的旧 checkpoint，请显式传入 `--model-config-path`，避免用推理默认值重建错误的 GQA/MLA、MoE 或 MTP 结构。

## 导出 Hugging Face 模型

```bash
uv run python scripts/export_hf.py \
  --checkpoint ./out/model.pth \
  --output-dir ./out/hf_model \
  --tokenizer-path mini_deepseek/config/tokenizer
```

导出后可以通过 Transformers 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./out/hf_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
```

也可以使用 vLLM：

```bash
uv run python -m mini_deepseek.eval.eval_vllm \
  --hf-model-dir ./out/hf_model
```

## 模型评测

安装 lm-evaluation-harness 并运行评测：

```bash
bash scripts/install_lm_eval_hf.sh
bash scripts/run_lm_eval_hf.sh ./out/hf_model
```

可以通过环境变量指定任务和运行参数：

```bash
TASKS=hellaswag,piqa,arc_easy \
DEVICE=cuda:0 \
BATCH_SIZE=auto \
bash scripts/run_lm_eval_hf.sh ./out/hf_model
```

## 致谢与上游项目

本项目基于以下开源项目开发：

- [MiniMind: Train a Tiny LLM from Scratch](https://github.com/jingyaogong/minimind)
- 作者：Jingyao Gong
- 许可证：Apache License 2.0

如果本项目对你的研究或开发有帮助，请同时引用 MiniMind：

```bibtex
@misc{gong2024minimind,
  title        = {MiniMind: Train a Tiny LLM from Scratch},
  author       = {Gong, Jingyao},
  year         = {2024},
  howpublished = {\url{https://github.com/jingyaogong/minimind}},
  note         = {GitHub repository}
}
```

感谢 MiniMind 作者及相关开源项目贡献者。
