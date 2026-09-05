#!/usr/bin/env bash
# 一键配置 rag-agent 环境：创建虚拟环境、安装 CLI（含可选依赖）、
# 生成 .env（如缺失）并运行环境自检。用法：bash scripts/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "未找到 $PYTHON，请先安装 Python 3.11+。" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "已生成 .env，请填入 LLM_API_KEY；或直接运行 rag-agent init 交互式配置。"
fi

rag-agent doctor || true
echo
echo "环境就绪。使用方式："
echo "  source .venv/bin/activate"
echo "  rag-agent --help"
