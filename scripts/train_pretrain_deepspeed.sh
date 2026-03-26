#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

NGPUS="${NGPUS:-1}"
MODULE="myminimind.training.train_pretrain"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-$ROOT_DIR/.venv/bin/deepspeed}"

COMMON_ARGS=(
  --save-dir out
  --save-weight pretrain
  --data-path ./dataset/pretrain_hq.jsonl
  --use-deepspeed 1
)

if [[ ! -x "$DEEPSPEED_BIN" ]]; then
  if command -v deepspeed >/dev/null 2>&1; then
    DEEPSPEED_BIN="$(command -v deepspeed)"
  else
    echo "未找到 deepspeed 可执行文件。"
    echo "请先安装 deepspeed，或通过 DEEPSPEED_BIN 指定路径。"
    exit 1
  fi
fi

"$DEEPSPEED_BIN" \
  --num_gpus="$NGPUS" \
  -m "$MODULE" \
  "${COMMON_ARGS[@]}" \
  "$@"
