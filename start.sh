#!/usr/bin/env bash
# Start the OpenAI-compatible exllamav3 server (tools/serve_openai.py).
# Configuration lives in .env — created from .env.example on first run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "No .env found — created one from .env.example."
    echo "Edit it (model paths, context, GPU memory) and run ./start.sh again."
    exit 1
fi
# shellcheck disable=SC1091
source .env

# Prefer the repo's venv if present
PYTHON="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
fi

MODEL_DIR="${MODEL_DIR:?MODEL_DIR must be set in .env}"
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
CONTEXT_SIZE="${CONTEXT_SIZE:-65536}"
GPU_MEM_GB="${GPU_MEM_GB:-110}"
CACHE_QUANT="${CACHE_QUANT:-none}"
DRAFT_DIR="${DRAFT_DIR:-none}"
CPU_CACHE_GB="${CPU_CACHE_GB:-0}"

cmd=("$PYTHON" -u tools/serve_openai.py
     --model "$MODEL_DIR"
     --host "$HOST"
     --port "$PORT"
     --cache_size "$CONTEXT_SIZE"
     --grid_size "$GPU_MEM_GB")

if [ "$CACHE_QUANT" != "none" ]; then
    cmd+=(--cache_quant "$CACHE_QUANT")
fi
if [ "$DRAFT_DIR" != "none" ]; then
    cmd+=(--draft_model "$DRAFT_DIR")
fi
if [ "$CPU_CACHE_GB" != "0" ]; then
    cmd+=(--cpu_cache_size "$CPU_CACHE_GB")
fi

echo "Starting: ${cmd[*]}"
exec "${cmd[@]}"
