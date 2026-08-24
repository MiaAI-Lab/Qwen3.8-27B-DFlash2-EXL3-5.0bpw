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

# --- bootstrap: build the venv + install the engine on first run ----------
# New users have no .venv; this creates it and installs everything.
if [ ! -x .venv/bin/python ]; then
    echo "First run: creating .venv and installing the exllamav3 engine …"
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    # GPU torch straight from the PyTorch index (PyPI's torch is CPU-only).
    .venv/bin/pip install --quiet --no-deps torch \
        --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    # The engine itself; its requirements.txt supplies the rest of the deps.
    # Default = upstream exllamav3 (x86 CUDA). The DFlash2/aarch64 fork goes
    # here — set EXL3_REPO in .env (git+https://… or a local path).
    # --no-build-isolation + the env vars below compile the native ext at
    # install time (aarch64/GB10 recipe; override via .env as needed).
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;12.1}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export MAX_JOBS="${MAX_JOBS:-4}"
    .venv/bin/pip install --quiet --no-build-isolation \
        "${EXL3_REPO:-git+https://github.com/turboderp-org/exllamav3}"
    .venv/bin/pip install --quiet aiohttp huggingface_hub
    echo "Engine installed."
fi

PYTHON=.venv/bin/python
# venv tools (ninja, …) must stay findable for the engine's JIT fallback.
export PATH="$(pwd)/.venv/bin:$PATH"

MODEL_DIR="${MODEL_DIR:?MODEL_DIR must be set in .env}"
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
CONTEXT_SIZE="${CONTEXT_SIZE:-65536}"
GPU_MEM_GB="${GPU_MEM_GB:-110}"
CACHE_QUANT="${CACHE_QUANT:-none}"
DRAFT_DIR="${DRAFT_DIR:-none}"
CPU_CACHE_GB="${CPU_CACHE_GB:-0}"

# --- auto-download from the Hub if missing --------------------
# Repos are private: put HF_TOKEN=<token> in .env, or `hf auth login`.
HF_TARGET_REPO="${HF_TARGET_REPO:-Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw}"
HF_DRAFT_REPO="${HF_DRAFT_REPO:-Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw}"

dl_model() {   # dl_model <repo_id> <dir> <label>
    local repo="$1" dir="$2" label="$3"
    if [ -f "$dir/config.json" ] && compgen -G "$dir/*.safetensors" > /dev/null; then
        echo "$label: $dir already present — skipping download."
        return 0
    fi
    echo "$label: not found at $dir — downloading from huggingface.co/$repo …"
    mkdir -p "$dir"
    "$PYTHON" - "$repo" "$dir" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id = sys.argv[1], local_dir = sys.argv[2])
print(f"  downloaded -> {path}")
PYEOF
}

dl_model "$HF_TARGET_REPO" "$MODEL_DIR" "target model"
if [ "$DRAFT_DIR" != "none" ]; then
    dl_model "$HF_DRAFT_REPO" "$DRAFT_DIR" "DFlash2 draft"
fi

# Context beyond the native 262144 needs the YaRN config variant.
if [ "$CONTEXT_SIZE" -gt 262144 ] \
   && [ -f "$MODEL_DIR/config.yarn-1m.json" ] \
   && ! grep -q rope_scaling "$MODEL_DIR/config.json"; then
    cp "$MODEL_DIR/config.yarn-1m.json" "$MODEL_DIR/config.json"
    echo "CONTEXT_SIZE > 262k: switched $MODEL_DIR/config.json to the YaRN 1M variant."
fi

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
