#!/usr/bin/env bash
# Start the OpenAI-compatible exllamav3 server (tools/serve_openai.py).
# Configuration lives in .env — created from .env.example on first run.
#
# Works from the deployment kit or from the engine repo itself. First run
# builds .venv, installs torch + the engine (compiling the CUDA kernels) +
# server deps, then downloads the model weights from Hugging Face and serves.
# Later runs start the server directly.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f tools/serve_openai.py ]; then
    echo "start.sh must run from the deployment kit or the engine repo" >&2
    echo "(tools/serve_openai.py not found next to it)." >&2
    exit 1
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "No .env found — created one from .env.example."
    echo "Edit it (model paths, context, GPU memory) and run ./start.sh again."
    exit 1
fi
# shellcheck disable=SC1091
source .env

# .env is sourced as shell vars; the model-download subprocess needs the HF
# token in its environment, so export it if set.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi

# --- bootstrap: build the venv + install the engine on first run ----------
# Re-enters if the venv is missing OR the install is incomplete (e.g. a
# Ctrl-C during the first run left a half-installed venv) — pip is idempotent.
if [ ! -x .venv/bin/python ] \
   || ! .venv/bin/python -c "import torch, exllamav3, aiohttp, huggingface_hub" 2>/dev/null; then
    echo "First-run setup — one time only (later runs skip straight to the model):"
    BOOT_LOG="$(mktemp /tmp/exl3_setup.XXXXXX.log)"

    _elapsed() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }

    # Step whose own output is useful (pip download bars): run in foreground.
    _step() {   # _step "label" cmd [args…]
        local label="$1" t0=$SECONDS; shift
        echo "  [ .. ] $label"
        if "$@"; then
            echo "  [ ok ] $label ($(_elapsed $((SECONDS - t0))))"
        else
            echo "  [FAIL] $label — after $(_elapsed $((SECONDS - t0)))"
            return 1
        fi
    }

    # Long silent step (CUDA compile): spinner + live timer on a terminal,
    # plain lines when piped; output captured, tail shown on failure.
    _quiet_step() {   # _quiet_step "label" cmd [args…]
        local label="$1" t0=$SECONDS; shift
        : > "$BOOT_LOG"
        if [ -t 1 ]; then
            "$@" >>"$BOOT_LOG" 2>&1 &
            local pid=$! i=0 spin='-\|/'
            while kill -0 "$pid" 2>/dev/null; do
                prog=$(grep -oE '^\[[0-9]+/[0-9]+\]' "$BOOT_LOG" 2>/dev/null | tail -1 || true)
                printf '\r  [%s] %s … %s %s   ' "${spin:$((i % 4)):1}" "$label" \
                    "${prog:+$prog }" "$(_elapsed $((SECONDS - t0)))"
                i=$((i + 1)); sleep 0.25
            done
            if wait "$pid"; then
                printf '\r\033[K  [ ok ] %s (%s)\n' "$label" "$(_elapsed $((SECONDS - t0)))"
                return 0
            fi
        else
            echo "  [ .. ] $label"
            if "$@" >>"$BOOT_LOG" 2>&1; then
                echo "  [ ok ] $label ($(_elapsed $((SECONDS - t0))))"
                return 0
            fi
        fi
        printf '\r\033[K  [FAIL] %s — after %s\n' "$label" "$(_elapsed $((SECONDS - t0)))"
        echo "  ---- last output (full log: $BOOT_LOG) ----"
        tail -n 20 "$BOOT_LOG" | sed 's/^/  | /'
        return 1
    }

    _step "1/5 creating Python virtualenv" python3 -m venv .venv
    _quiet_step "2/5 build tools (pip, setuptools, wheel)" \
        .venv/bin/pip install --quiet --upgrade pip setuptools wheel typing_extensions packaging
    # GPU torch + its NVIDIA runtime deps; PyPI stays primary so the
    # nvidia-* runtime wheels resolve too (cu130 local-version wheel wins).
    # Output NOT hidden: pip's own download progress bars show here.
    _step "3/5 PyTorch (~2–3 GB download the first time)" \
        .venv/bin/pip install torch \
            --extra-index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    # The engine itself; its setup.py pulls in the rest of the deps.
    # Inside the engine repo: build from the local checkout (EXL3_REPO is
    # ignored there). Elsewhere (deployment kit): install from EXL3_REPO —
    # default is the MiaAI-Lab fork (DFlash2/MTP drafting, NVFP4/FP8 KV,
    # aarch64 GB10 + x86 CUDA); override in .env for a token URL (while
    # the repos are private) or a local path.
    # --no-build-isolation + the env vars below compile the native ext at
    # install time (override via .env as needed).
    if [ -f exllamav3/__init__.py ]; then
        _engine_src="."
        _engine_note="local engine repo — compiling CUDA kernels"
    else
        _engine_src="${EXL3_REPO:-git+https://github.com/MiaAI-Lab/exllamav3}"
        _engine_note="exllamav3 engine — clone + compile CUDA kernels"
    fi
    if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        export TORCH_CUDA_ARCH_LIST
    elif [ "$(uname -m)" = "aarch64" ]; then
        # GB10/Spark needs the arch list spelled out; x86 auto-detects.
        export TORCH_CUDA_ARCH_LIST="12.0;12.1"
    fi
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    # 8 parallel nvcc jobs when the machine can take it (halves wall time on
    # many-core boxes); 4 otherwise. Override in .env if needed.
    export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) >= 8 && $(free -g | awk '/^Mem:/{print $2}') >= 32 ? 8 : 4 ))}"
    # Fail fast (clear error) instead of hanging if the engine repo needs
    # auth (GIT_ASKPASS: proven on git 2.43 where GIT_TERMINAL_PROMPTS
    # alone does not suppress the credential prompt).
    export GIT_TERMINAL_PROMPTS=0
    export GIT_ASKPASS=/bin/true
    _quiet_step "4/5 ${_engine_note} (5–20 min depending on machine)" \
        .venv/bin/pip install --no-build-isolation \
            "${_engine_src}"
    _quiet_step "5/5 server dependencies (aiohttp, huggingface_hub, transformers)" \
        .venv/bin/pip install --quiet aiohttp huggingface_hub transformers
    echo "Setup complete."
fi

PYTHON=.venv/bin/python
# venv tools (ninja, …) must stay findable for the engine's JIT fallback.
export PATH="$(pwd)/.venv/bin:$PATH"

MODEL_DIR="${MODEL_DIR:?MODEL_DIR must be set in .env}"
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
CONTEXT_SIZE="${CONTEXT_SIZE:-65536}"
if [ -n "${GPU_MEM_GB:-}" ]; then
    echo "GPU memory budget: ${GPU_MEM_GB} GB (from .env)"
else
    _vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) || true
    if [ "${_vram:-0}" -gt 0 ] 2>/dev/null; then
        # discrete GPU: VRAM minus a little headroom (e.g. 24 GB card -> 22)
        GPU_MEM_GB=$(( _vram / 1024 - 2 ))
    else
        # GB10/unified memory (nvidia-smi reports no total): available system
        # RAM minus a reserve for the OS and anything else on the box
        GPU_MEM_GB=$(( $(free -g | awk '/^Mem:/{print $7}') - 16 ))
    fi
    if [ "$GPU_MEM_GB" -lt 8 ]; then GPU_MEM_GB=8; fi
    echo "GPU_MEM_GB not set — auto-detected budget: ${GPU_MEM_GB} GB (override in .env)"
fi
CACHE_QUANT="${CACHE_QUANT:-none}"
CPU_CACHE_GB="${CPU_CACHE_GB:-0}"

# --- speculative decoding method ---------------------------------------------
# DRAFT = mtp | dflash2 | none (see .env.example for the trade-offs).
# Legacy: DRAFT_DIR set to a path keeps working (= dflash2 with that path);
# DRAFT_DIR=none (= explicit no-draft) also keeps working.
DRAFT="${DRAFT:-}"
if [ -z "$DRAFT" ]; then
    if [ "${DRAFT_DIR:-}" = "none" ]; then DRAFT=none
    elif [ -n "${DRAFT_DIR:-}" ]; then DRAFT=dflash2    # legacy: explicit draft path
    else DRAFT=mtp                                       # default: MTP head, best context/GB
    fi
fi
DRAFT="$(echo "$DRAFT" | tr '[:upper:]' '[:lower:]')"

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

case "$DRAFT" in
    mtp)
        # MTP head lives inside the target checkpoint — nothing to download.
        ;;
    dflash2)
        if [ -z "${DRAFT_DIR:-}" ] || [ "$DRAFT_DIR" = "none" ]; then
            DRAFT_DIR="models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw"
        fi
        dl_model "$HF_DRAFT_REPO" "$DRAFT_DIR" "DFlash2 draft"
        ;;
    none)
        ;;
    *)
        echo "DRAFT must be mtp, dflash2, or none (got: $DRAFT)" >&2
        exit 1
        ;;
esac

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
case "$DRAFT" in
    mtp)      cmd+=(--draft_model mtp) ;;
    dflash2)  cmd+=(--draft_model "$DRAFT_DIR") ;;
    none)     cmd+=(--draft_model none) ;;   # skip the server's built-in default draft path
esac
if [ "$CPU_CACHE_GB" != "0" ]; then
    cmd+=(--cpu_cache_size "$CPU_CACHE_GB")
fi

echo "Starting: ${cmd[*]}"
exec "${cmd[@]}"
