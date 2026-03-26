#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
HARNESS_DIR="${HARNESS_DIR:-/home/dkr/codes/lm-evaluation-harness}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "未找到项目虚拟环境 Python: $VENV_PYTHON"
  echo "请先创建 .venv，或通过 VENV_PYTHON 指定解释器。"
  exit 1
fi

if [[ ! -d "$HARNESS_DIR" ]]; then
  echo "未找到 lm-evaluation-harness 仓库目录: $HARNESS_DIR"
  echo "请先 clone，或通过 HARNESS_DIR 指定目录。"
  exit 1
fi

if [[ ! -f "$HARNESS_DIR/pyproject.toml" ]]; then
  echo "目录存在但不是有效的 lm-evaluation-harness 仓库: $HARNESS_DIR"
  exit 1
fi

if [[ ! -x "$UV_BIN" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    echo "未找到 uv。当前默认路径: $HOME/.local/bin/uv"
    echo "请先安装 uv，或通过 UV_BIN 指定可执行文件路径。"
    exit 1
  fi
fi

echo "Using Python: $VENV_PYTHON"
echo "Using harness: $HARNESS_DIR"
echo "Using uv: $UV_BIN"

"$UV_BIN" pip install --python "$VENV_PYTHON" -e "${HARNESS_DIR}[hf]"
"$VENV_PYTHON" -m lm_eval --help >/dev/null

echo
echo "lm-evaluation-harness[hf] 已安装到项目 .venv。"
echo "下一步可运行：bash scripts/run_lm_eval_hf.sh /path/to/exported_hf_model"
