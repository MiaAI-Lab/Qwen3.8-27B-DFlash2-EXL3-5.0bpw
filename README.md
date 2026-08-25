# Qwen3.8-27B · EXL3 · Speculative Decoding

Private deployment kit for serving **Qwen3.8-27B** quantized to EXL3 3.5bpw
with speculative decoding — **MTP by default** (draft head inside the
checkpoint, best context per GB) or **DFlash2** (EXL3 5.0bpw draft model,
~15% faster) — plus an NVFP4 KV cache lane for maximum context per GB.

This repo contains the launcher and configuration. The model weights live on
Hugging Face (see [Model cards](#model-cards)) and the inference engine is
the [exllamav3](https://github.com/turboderp-org/exllamav3) fork with
DFlash2/aarch64 support.

## What's in the box

| file | what |
|---|---|
| `start.sh` | env-driven launcher for the OpenAI-compatible server |
| `stop.sh` | stop the server (graceful shutdown, safe to re-run) |
| `.env.example` | documented configuration knobs |
| `model-cards/` | model cards for the two HF weight sets |

## Quick start

```bash
# 1. authenticate to Hugging Face (the weight repos are private):
#    hf auth login     (or put HF_TOKEN=... in .env — see below)
# 2. configure and launch:
cp .env.example .env      # edit: context, GPU memory, HF_TOKEN
./start.sh                # first run: builds .venv + installs the engine,
                          # downloads weights, serves http://localhost:8888/v1
```

`start.sh` is self-bootstrapping: on first run it creates `.venv` and
pip-installs the exllamav3 engine (GPU torch from the PyTorch index, engine
and server deps). Set `EXL3_REPO` in `.env` to install a fork instead of
upstream — required on DGX Spark/aarch64, and for NVFP4 KV / DFlash2
drafting.

Running on a 24 GB GPU (RTX 3090/4090)? Jump to
[24 GB config](#24-gb-gpus-rtx-3090--4090).

`start.sh` auto-creates `.env` from the example on first run, prefers
`.venv/bin/python` when present, and **downloads the target and draft weight
sets from Hugging Face automatically** on first launch (and resumes partial
downloads) — no manual fetching. Set `HF_TARGET_REPO` / `HF_DRAFT_REPO` in
`.env` to pull from a mirror instead.

## Configuration highlights (`.env`)

- `DRAFT` — speculative decoding method: `mtp` (default; the draft head
  lives inside the target checkpoint — no download, ~50% more context on
  small GPUs), `dflash2` (dedicated draft model, fastest tokens/s), or
  `none`. `MODEL_DIR` / `DRAFT_DIR` set the target and (for dflash2) draft
  locations; missing dirs are fetched from the Hub automatically
  (`HF_TARGET_REPO` / `HF_DRAFT_REPO` override the repo ids, `HF_TOKEN`
  authenticates — repos are private)
- `CONTEXT_SIZE` — KV cache size in tokens (native limit 262,144; 1M works and
  the launcher swaps in the YaRN config variant automatically)
- `CACHE_QUANT` — KV format: `none` (fp16) / `8` / `8,4` / `fp8` / `nvfp4`
  (~4.5 bits/elem, measured lossless at generation level)
- `GPU_MEM_GB` — memory budget for weights + caches (110 on DGX Spark, ~22 on
  a 24 GB GPU)
- `CPU_CACHE_GB` — optional CPU second-tier cache; cold pages spill from GPU

Concurrency note: the server generates one request at a time (batch-1
speculative decoding); concurrent requests queue automatically.

## 24 GB GPUs (RTX 3090 / 4090)

The kit targets DGX Spark (121 GB) by default. On a 24 GB card the recipe is
the same weights, a tighter KV budget, and NVFP4 KV — the model card calls it
"14.2 GB + 1.4 GB + NVFP4 ≈ 220k tokens fully resident".

```bash
# .env — 24 GB recipe
GPU_MEM_GB=22            # weights + caches budget; 24 GB card, keep ~2 GB headroom
CONTEXT_SIZE=220000      # see math below
CACHE_QUANT=nvfp4        # required at this size; lossless at generation level (measured)
DRAFT_DIR=models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw
# CPU_CACHE_GB=16        # CPU spill tier: NOT YET ACTIVE (planned; see notes)
```

Memory math: weights are 15.6 GB total (14.2 target + 1.4 draft), so a 22 GB
budget leaves ~6 GB of KV. NVFP4 KV costs ~18 KB/token (only the 16
full-attention layers hold a KV cache; fp16 is ~64 KB/token) → **~220k tokens
to ~262k (the native limit) fully resident**. Past that the cache does not
hold today; the CPU spill tier is planned but not yet implemented.

RTX-class notes (vs the DGX Spark the numbers above were measured on):

- **Perf is memory-bandwidth-bound at batch 1, and should be *higher* on an
  RTX card**: GB10 reads weights from ~273 GB/s-class LPDDR5x unified memory;
  a 3090/4090 reads the same weights from ~1 TB/s-class VRAM. Acceptance and
  quality are unchanged — only tok/s moves. Not yet benchmarked on RTX, treat
  the table in the model cards as the lower bound.
- **CPU spill (planned, `CPU_CACHE_GB`)**: once implemented (T6 follow-up),
  a desktop's 32–64 GB of system RAM backs cold KV pages over PCIe — slow to
  touch, but it converts the hard ~220k resident limit into a graceful
  decline. Today the knob is accepted but inert.
- **The 1M YaRN config does not fit**: 1M tokens of NVFP4 KV ≈ 19 GB on top of
  15.6 GB of weights. Only reachable with CPU spill, and 262k is the
  quality-faithful limit anyway — don't expect the 1M headroom to be useful
  here.
- **Use the fork on x86 too**: stock exllamav3 serves the model on CUDA, but
  NVFP4 KV and DFlash2 drafting are fork features; install the fork's x86 CUDA
  build.
- Need more context than ~262k? `DRAFT_DIR=none` frees 1.4 GB (~+75k tokens),
  at the cost of speculative decoding.

## Model cards

- [`model-cards/Qwen3.8-27B-EXL3-3.5bpw.md`](model-cards/Qwen3.8-27B-EXL3-3.5bpw.md) —
  target: EXL3 3.5bpw, workload-calibrated, 14.2 GB
- [`model-cards/Qwen3.8-27B-DFlash2-EXL3-5.0bpw.md`](model-cards/Qwen3.8-27B-DFlash2-EXL3-5.0bpw.md) —
  draft: DFlash2 EXL3 5.0bpw, 1.4 GB, +33% decode throughput at acceptance parity

Measured on DGX Spark (GB10): HumanEval-class decode **47.5 tok/s** at
T=0.6 with acceptance 4.43 tokens/step; tool-eval-bench hardmode @ T=1.0:
**87–88 / 100**.

## License

Code in this repo: [MIT](LICENSE). Model weights are Apache-2.0 derivatives
(see model cards); exllamav3 is MIT.
