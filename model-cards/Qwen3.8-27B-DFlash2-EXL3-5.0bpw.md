---
license: apache-2.0
base_model:
- incoai/Qwen3.8-27B-DFlash2
- Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: exllamav3
inference: false
tags:
- exl3
- exllamav3
- draft-model
- speculative-decoding
- dflash2
- block-diffusion
---

# Qwen3.8-27B-DFlash2 · EXL3 · 5.0bpw

EXL3 quantization of the **DFlash2 draft model** for
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — a 1.93B
block-diffusion speculative-decoding drafter (original bf16 release:
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
[DFlash2 blog](https://inco.ai/blog/dflash2/),
[GitHub](https://github.com/z-lab/dflash)).

This is **not a standalone language model**. It drafts token blocks for the
Qwen3.8-27B target to verify, inside a speculative-decoding engine:

- drafts a whole block (up to 7 speculative tokens, `block_size: 8`) in one pass
- keeps top candidates at every position; a lightweight selector traces one
  coherent path through them
- two-tap dynamic convolutions in the backbone keep the draft from decaying
  over long block horizons

The EXL3 quant cuts the draft from **3.85 GB (bf16) to 1.4 GB**, which matters
because at decode batch sizes the drafter's weight reads are a first-order
cost: on DGX Spark (GB10) this raised end-to-end decode throughput **+33%**
over the bf16 draft with acceptance at parity.

## Quantization details

| setting | value |
|---|---|
| format | EXL3 `v1.4.2` |
| bits | 5.0 bpw (module-adaptive, includes the candidate selector) |
| calibration | default calibration data |
| module rmse | ~0.00085 (per-projection) |
| size | 1.4 GB (from 3.85 GB bf16) |

## Measured impact (with the 3.5bpw EXL3 target, DGX Spark / GB10)

| metric | bf16 draft | this quant |
|---|---|---|
| greedy acceptance (code prose) | 2.68–2.80 | 2.68–2.74 (one run bit-identical output) |
| HumanEval-style accept (T = 0.6) | 4.27 | **4.43** |
| decode tok/s (HumanEval, T = 0.6) | 35.7 | **47.5** |

## Usage

Requires the **exllamav3 fork with DFlash2 support** (architecture, selector
walk, and speculative generation loop are fork additions; stock exllamav3 does
not implement DFlash2). With the fork's OpenAI-compatible server:

```bash
python tools/serve_openai.py --port 8000 \
  -m Qwen3.8-27B-EXL3-3.5bpw \
  -dm Qwen3.8-27B-DFlash2-EXL3-5.0bpw \
  -cq nvfp4 -cs 262144
```

`-dm` loads the draft (bf16 or this EXL3 quant both work); `-cq nvfp4` is the
fork's NVFP4 KV cache. The draft's own KV cache stays fp16 (parity-pinned).

Pair with target:
[`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw).

## License

Apache-2.0 (inherited from the DFlash2 draft release). This repository is a
quantized derivative for exllamav3 speculative decoding; credit for the DFlash2
architecture and training goes to z-lab / inco.ai.
