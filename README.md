# Qwen3.8-27B · EXL3 · DFlash2 — deployment kit

Private deployment kit for serving **Qwen3.8-27B** quantized to EXL3 3.5bpw
with **DFlash2 speculative decoding** (EXL3 5.0bpw draft), including an
NVFP4 KV cache lane for maximum context per GB.

This repo contains the launcher and configuration. The model weights live on
Hugging Face (see [Model cards](#model-cards)) and the inference engine is
the [exllamav3](https://github.com/turboderp-org/exllamav3) fork with
DFlash2/aarch64 support.

## What's in the box

| file | what |
|---|---|
| `start.sh` | env-driven launcher for the OpenAI-compatible server |
| `.env.example` | documented configuration knobs |
| `model-cards/` | model cards for the two HF weight sets |

## Quick start

```bash
# 1. install the exllamav3 fork (DFlash2 + aarch64 port) in a venv
# 2. download the weight sets from Hugging Face into models/
# 3. configure and launch:
cp .env.example .env      # edit: model paths, context, GPU memory
./start.sh                # serves http://localhost:8888/v1
```

`start.sh` auto-creates `.env` from the example on first run and prefers
`.venv/bin/python` when present.

## Configuration highlights (`.env`)

- `MODEL_DIR` / `DRAFT_DIR` — target and speculative draft (`none` disables drafting)
- `CONTEXT_SIZE` — KV cache size in tokens (native limit 262,144; 1M via the
  YaRN config variant, see model card)
- `CACHE_QUANT` — KV format: `none` (fp16) / `8` / `8,4` / `fp8` / `nvfp4`
  (~4.5 bits/elem, measured lossless at generation level)
- `GPU_MEM_GB` — memory budget for weights + caches (110 on DGX Spark, ~22 on
  a 24 GB GPU)
- `CPU_CACHE_GB` — optional CPU second-tier cache; cold pages spill from GPU

Concurrency note: the server generates one request at a time (batch-1
speculative decoding); concurrent requests queue automatically.

## Model cards

- [`model-cards/Qwen3.8-27B-EXL3-3.5bpw.md`](model-cards/Qwen3.8-27B-EXL3-3.5bpw.md) —
  target: EXL3 3.5bpw, workload-calibrated, 14.2 GB
- [`model-cards/Qwen3.8-27B-DFlash2-EXL3-5.0bpw.md`](model-cards/Qwen3.8-27B-DFlash2-EXL3-5.0bpw.md) —
  draft: DFlash2 EXL3 5.0bpw, 1.4 GB, +33% decode throughput at acceptance parity

Measured on DGX Spark (GB10): HumanEval-class decode **47.5 tok/s** at
T=0.6 with acceptance 4.43 tokens/step; tool-eval-bench hardmode @ T=1.0:
**87–88 / 100**.

## License

Code in this repo: provided as-is for internal deployment. Model weights are
Apache-2.0 derivatives (see model cards); exllamav3 is MIT.
