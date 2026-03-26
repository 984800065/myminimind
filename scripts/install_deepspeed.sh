#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "未找到项目虚拟环境 Python: $VENV_PYTHON"
  echo "请先创建 .venv，或通过 VENV_PYTHON 指定解释器。"
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
echo "Using uv: $UV_BIN"

"$UV_BIN" pip install --python "$VENV_PYTHON" deepspeed

echo
echo "DeepSpeed 已安装到项目 .venv。"
echo "下一步可运行：NGPUS=4 bash scripts/train_pretrain_deepspeed.sh"
