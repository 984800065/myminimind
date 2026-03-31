#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MODULE="myminimind.training.train_pretrain"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-$ROOT_DIR/.venv/bin/deepspeed}"
SAVE_DIR="${SAVE_DIR:-out}"
SAVE_WEIGHT="${SAVE_WEIGHT:-pretrain}"
DATA_PATH="${DATA_PATH:-./dataset/pretrain_hq.jsonl}"
ZERO_STAGE="${ZERO_STAGE:-2}"

if [[ -n "${NGPUS:-}" ]]; then
  NGPUS="$NGPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  NGPUS="${#VISIBLE_GPUS[@]}"
else
  NGPUS=1
fi

COMMON_ARGS=(
  --save-dir "$SAVE_DIR"
  --save-weight "$SAVE_WEIGHT"
  --data-path "$DATA_PATH"
  --use-deepspeed 1
  --deepspeed-zero-stage "$ZERO_STAGE"
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
