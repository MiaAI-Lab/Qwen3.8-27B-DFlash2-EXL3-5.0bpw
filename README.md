<h1 align="center">Qwen3.8-27B · EXL3 · Speculative Decoding</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Deployment kit for serving **Qwen3.8-27B** quantized to EXL3 3.5bpw
with speculative decoding — **MTP by default** (draft head inside the
checkpoint, best context per GB) or **DFlash2** (EXL3 5.0bpw draft model,
~15% faster) — plus an NVFP4 KV cache lane for maximum context per GB.

This repo contains the launcher, server and configuration. The model weights
live on Hugging Face (see [Model cards](#model-cards)) and the inference
engine is our [exllamav3 fork](https://github.com/MiaAI-Lab/exllamav3)
(DFlash2/MTP drafting, NVFP4/FP8 KV, aarch64 GB10 + x86 CUDA).

## What's in the box

| file | what |
|---|---|
| `start.sh` | env-driven launcher for the OpenAI-compatible server |
| `stop.sh` | stop the server (graceful shutdown, safe to re-run) |
| `tools/serve_openai.py` | the server itself (chat completions, streaming, tool calling) |
| `.env.example` | documented configuration knobs |
| `model-cards/` | model cards for the two HF weight sets |

## Quick start

```bash
# 1. configure and launch:
cp .env.example .env      # edit: context, GPU memory
./start.sh                # first run: builds .venv + installs the engine,
                          # downloads weights, serves http://localhost:8888/v1
```

`start.sh` is self-bootstrapping: on first run it creates `.venv`, installs
GPU torch and pip-installs the exllamav3 engine from this fork (all of
DFlash2/MTP drafting, NVFP4/FP8 KV and the aarch64 GB10 port are fork
features). `EXL3_REPO` in `.env` overrides where the engine comes from — any
git URL or a local path.

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
  (`HF_TARGET_REPO` / `HF_DRAFT_REPO` override the repo ids)
- `CONTEXT_SIZE` — KV cache size in tokens (native limit 262,144; 1M works and
  the launcher swaps in the YaRN config variant automatically)
- `CACHE_QUANT` — KV format: `none` (fp16) / `8` / `8,4` / `fp8` / `nvfp4`
  (~4.5 bits/elem, measured lossless at generation level)
- `GPU_MEM_GB` — memory budget for weights + caches; auto-detected when unset
  (discrete GPU: VRAM − 2 GB, unified/GB10: available RAM − 16 GB)
- `CPU_CACHE_GB` — CPU second-tier cache knob (accepted but not yet active;
  see the 24 GB notes)

Concurrency note: the server generates one request at a time (batch-1
speculative decoding); concurrent requests queue automatically.

## 24 GB GPUs (RTX 3090 / 4090)

The kit targets DGX Spark (121 GB) by default. On a 24 GB card the recipe is
the same weights, a tighter KV budget, and NVFP4 KV. The default drafter
(MTP) is the right choice here:

```bash
# .env — 24 GB recipe (MTP, the default)
GPU_MEM_GB=22            # auto-detected when unset; explicit for clarity
CONTEXT_SIZE=262144      # full native context fits (see math below)
CACHE_QUANT=nvfp4        # required at this size; lossless at generation level (measured)
# DRAFT=mtp is the default — no extra config, no draft download
```

- **`DRAFT=mtp` (default)**: no draft weights, ~20% less KV per token → the
  full native 262k context fits fully resident, at ~30 tok/s (GB10; an RTX
  card should be faster — decode is memory-bandwidth-bound).
- **`DRAFT=dflash2`**: ~34.5 tok/s but 1.4 GB of draft weights + draft KV →
  ~220k–262k tokens fully resident.

```bash
# .env — alternative: DFlash2 at 24 GB
GPU_MEM_GB=22
CONTEXT_SIZE=220000
CACHE_QUANT=nvfp4
DRAFT=dflash2
DRAFT_DIR=models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw   # auto-downloaded
# CPU_CACHE_GB=16        # CPU spill tier: NOT YET ACTIVE (planned; see notes)
```

Memory math (GiB): target weights 14.2 + MTP head ~0.05 → ~6.7 GB left for
KV at a 22 GB budget; NVFP4 KV costs ~18 KB/token for the target plus ~1.2
KB/token for the MTP head (only the 16 full-attention layers hold KV; fp16
is ~64 KB/token) → the full native 262k fits with ~1 GB to spare. With the
DFlash2 draft instead: 15.6 GiB of weights + ~24 KB/token → ~220k–262k.
Past that the cache does not hold today; the CPU spill tier is planned but
not yet implemented.

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
- Need more context than ~262k? That requires the YaRN 1M config and more
  memory than a 24 GB card has. `DRAFT=none` frees the draft memory too, but
  gives up speculative decoding — `DRAFT=mtp` already dominates it on both
  memory and speed.

## Model cards

- [`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw) —
  target: EXL3 3.5bpw, workload-calibrated, 14.2 GB
- [`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) —
  draft: DFlash2 EXL3 5.0bpw, 1.4 GB, +33% decode throughput at acceptance parity

Measured on DGX Spark (GB10): HumanEval-class decode **47.5 tok/s** at
T=0.6 with acceptance 4.43 tokens/step; tool-eval-bench hardmode @ T=1.0:
**87–88 / 100**.

## License

Code in this repo: [MIT](LICENSE). Model weights are Apache-2.0 derivatives
(see model cards); exllamav3 is MIT.
