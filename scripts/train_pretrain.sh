#!/usr/bin/env bash
set -euo pipefail

# 位置：项目根目录下的 scripts/train_pretrain.sh
# 用法示例：
#   单卡：
#     bash scripts/train_pretrain.sh
#   单卡 + 自定义参数：
#     bash scripts/train_pretrain.sh --epochs 3 --batch-size 64
#   4 卡：
#     NGPUS=4 bash scripts/train_pretrain.sh --use-moe true --from-resume 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# 默认 1 卡；设置环境变量 NGPUS=4 可改为 4 卡
NGPUS="${NGPUS:-1}"

MODULE="mini_deepseek.training.train_pretrain"

COMMON_ARGS=(
  --save-dir out
  --save-weight pretrain
  --data-path ./dataset/pretrain_hq.jsonl
)

if [[ "$NGPUS" -gt 1 ]]; then
  torchrun \
    --standalone \
    --nproc_per_node="$NGPUS" \
    -m "$MODULE" \
    "${COMMON_ARGS[@]}" \
    "$@"
else
  python -m "$MODULE" \
    --device cuda:0 \
    "${COMMON_ARGS[@]}" \
    "$@"
fi

