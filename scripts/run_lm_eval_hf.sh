#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  bash scripts/run_lm_eval_hf.sh /path/to/exported_hf_model [额外 lm-eval 参数]

常用环境变量：
  TASKS=hellaswag,piqa,winogrande,arc_easy,arc_challenge
  DEVICE=cuda:0
  DTYPE=bfloat16
  BATCH_SIZE=auto
  MAX_BATCH_SIZE=16
  OUTPUT_PATH=artifacts/lm_eval/run_name
  LIMIT=10
  NUM_FEWSHOT=5
  APPLY_CHAT_TEMPLATE=1
  LOG_SAMPLES=1
  SHOW_CONFIG=1
  USE_CACHE=artifacts/lm_eval/cache/sqlite.db
  MODEL_ARGS_EXTRA=attn_implementation=flash_attention_2
  GEN_KWARGS=temperature=0.0,top_p=1.0
  SYSTEM_INSTRUCTION='You are a helpful assistant.'

示例：
  bash scripts/run_lm_eval_hf.sh out/hf/pretrain_1024_moe
  APPLY_CHAT_TEMPLATE=1 TASKS=gsm8k bash scripts/run_lm_eval_hf.sh out/hf/full_sft_1024_moe
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "缺少 HF 模型目录。"
  echo
  usage
  exit 1
fi

HF_MODEL_DIR="$1"
shift

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "未找到项目虚拟环境 Python: $VENV_PYTHON"
  echo "请先运行 bash scripts/install_lm_eval_hf.sh"
  exit 1
fi

if [[ ! -d "$HF_MODEL_DIR" ]]; then
  echo "HF 模型目录不存在: $HF_MODEL_DIR"
  exit 1
fi

if [[ ! -f "$HF_MODEL_DIR/config.json" ]]; then
  echo "目录中未找到 config.json，看起来不是导出的 Hugging Face 模型目录: $HF_MODEL_DIR"
  echo "请先运行 scripts/export_hf.py 导出，再把导出目录传给本脚本。"
  exit 1
fi

if ! "$VENV_PYTHON" -m lm_eval --help >/dev/null 2>&1; then
  echo "当前 .venv 中还没有可用的 lm_eval。"
  echo "请先运行 bash scripts/install_lm_eval_hf.sh"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-$(basename "$HF_MODEL_DIR")}"
TASKS="${TASKS:-hellaswag,piqa,winogrande,arc_easy,arc_challenge}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/artifacts/lm_eval/$MODEL_NAME}"

# Accelerate may require these on some RTX 40-series setups.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

mkdir -p "$OUTPUT_PATH"

cmd=(
  "$VENV_PYTHON" -m lm_eval run
  --model hf
  --model_args
  "pretrained=$HF_MODEL_DIR"
  "dtype=$DTYPE"
  --tasks "$TASKS"
  --batch_size "$BATCH_SIZE"
  --device "$DEVICE"
  --output_path "$OUTPUT_PATH"
  --trust_remote_code
  --confirm_run_unsafe_code
)

if [[ -n "${MODEL_ARGS_EXTRA:-}" ]]; then
  cmd+=("$MODEL_ARGS_EXTRA")
fi

if [[ -n "${MAX_BATCH_SIZE:-}" ]]; then
  cmd+=(--max_batch_size "$MAX_BATCH_SIZE")
fi

if [[ -n "${NUM_FEWSHOT:-}" ]]; then
  cmd+=(--num_fewshot "$NUM_FEWSHOT")
fi

if [[ -n "${LIMIT:-}" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ -n "${USE_CACHE:-}" ]]; then
  cmd+=(--use_cache "$USE_CACHE")
fi

if [[ -n "${GEN_KWARGS:-}" ]]; then
  cmd+=(--gen_kwargs "$GEN_KWARGS")
fi

if [[ -n "${SYSTEM_INSTRUCTION:-}" ]]; then
  cmd+=(--system_instruction "$SYSTEM_INSTRUCTION")
fi

if [[ "${APPLY_CHAT_TEMPLATE:-0}" != "0" ]]; then
  cmd+=(--apply_chat_template)
fi

if [[ "${LOG_SAMPLES:-0}" != "0" ]]; then
  cmd+=(--log_samples)
fi

if [[ "${SHOW_CONFIG:-0}" != "0" ]]; then
  cmd+=(--show_config)
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

echo "HF model dir : $HF_MODEL_DIR"
echo "Tasks        : $TASKS"
echo "Device       : $DEVICE"
echo "DType        : $DTYPE"
echo "Batch size   : $BATCH_SIZE"
echo "Output path  : $OUTPUT_PATH"
echo "NCCL_P2P     : $NCCL_P2P_DISABLE"
echo "NCCL_IB      : $NCCL_IB_DISABLE"
echo
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

exec "${cmd[@]}"
