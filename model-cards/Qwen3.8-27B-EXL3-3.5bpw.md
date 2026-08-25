---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: exllamav3
tags:
- exl3
- exllamav3
- quantization
- long-context
- speculative-decoding
---

# Qwen3.8-27B · EXL3 · 3.5bpw (workload-calibrated)

EXL3 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
the hybrid linear-attention / full-attention (gated-deltanet + full-attn) dense
27B model. **14.2 GB** of weights — sized to leave room for large KV caches on
24 GB GPUs: with the built-in MTP draft head the full native 262k context fits
fully resident (see [Memory & context](#memory--context)).

The conversion was calibrated on a **workload-matched, self-generated trace**
(622k tokens of coding + math reasoning sampled from this model family) instead
of generic corpus data — measurably better downstream acceptance and task
behavior than the generic-calibration build of the same bpw.

## Quantization details

| setting | value |
|---|---|
| format | EXL3 `v1.4.2` (`quantization_config.json`) |
| average bits | 3.5 bpw (module-adaptive) |
| head bits | 6 |
| codebook | `mul1` |
| high-quality mode | yes (`-hq`) |
| calibration | 250 rows × 2048 cols, workload-matched trace (622,515 tokens) |
| embeddings / lm_head | stored unquantized (`embed_tokens` bf16, `lm_head` fp16) |
| contents | text model + tokenizer; no vision tower |

## Requirements

- [exllamav3](https://github.com/turboderp-org/exllamav3) — the `qwen3_5`
  architecture used here is supported upstream, including MTP speculative
  decoding; no fork required for basic inference on x86 CUDA GPUs.
- On **DGX Spark / GB10 (aarch64)** you need the aarch64 port of exllamav3
  (upstream is x86-only at time of writing).
- FP8/NVFP4 KV cache and DFlash2/DSpark speculative decoding (below) require
  the extended exllamav3 fork this model was developed against.

## Quick start

```python
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler.presets import ComboSampler

config = Config.from_directory("Qwen3.8-27B-EXL3-3.5bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 32768)
model.load()
tokenizer = Tokenizer.from_config(config)

generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
job = Job(
    input_ids = tokenizer.encode("Hello!"),
    max_new_tokens = 200,
    sampler = ComboSampler(temperature = 0.6, top_p = 0.95, top_k = 20),
)
generator.enqueue(job)
while generator.num_remaining_jobs():
    for result in generator.iterate():
        print(result.get("text") or "", end = "")
```

Recommended sampling (Qwen3.8 thinking mode): temperature 0.6, top_p 0.95,
top_k 20. Thinking can be disabled per-request through the chat template
(`enable_thinking = False`).

## Memory & context

Only the 16 full-attention layers hold a KV cache (the 48 gated-deltanet
layers use a small fixed-size recurrent state), so long context is cheap:

| KV cache | bits/elem | @ 128k | @ 262k (native) | @ 1M (YaRN) |
|---|---|---|---|---|
| fp16 (default) | 16 | 8.4 GB | 17.2 GB | 68.7 GB |
| int8 / int4 (Hadamard) | ~8.5 / ~4.5 | 4.5 / 2.4 GB | 9.1 / 4.8 GB | 36.4 / 19.2 GB |
| fp8 E4M3 * | 8 | 4.2 GB | 8.6 GB | 34.4 GB |
| NVFP4 (E2M1 + E4M3/16) * | 4.5 | 2.4 GB | 4.8 GB | 19.2 GB |

\* fp8 and NVFP4 KV formats and the fused quantize-into-cache kernels are
features of the exllamav3 fork this model was benchmarked with. NVFP4 KV was
verified lossless-in-the-noise at generation level (acceptance and quality
gates unchanged vs fp16 KV; E2M1 cos-sim vs fp16 KV 0.99995).

**24 GB GPU recipe:** these weights (14.2 GB) + the built-in MTP head + NVFP4
KV ≈ the **full native 262k context fully resident**; with the DFlash2 draft
(1.4 GB) instead ≈ 220k tokens. A 3.0bpw build of the same recipe reaches
~290k.

## Long context (YaRN 1M)

Native context is 262,144 tokens. A ready-made YaRN extension to 1,048,576
tokens is included as **`config.yarn-1m.json`** — replace `config.json` with it
to enable (rope_scaling: yarn, factor 4.0). exllamav3 supports YaRN natively.

Measured behavior at 300k tokens (needle-in-haystack, 294.5k wikitext filler):
needles at 10% and 50% depth **pass**; needles in the last ~30k tokens (past
native 262k) fail. This is position-level YaRN behavior — identical with fp16
and NVFP4 KV caches — so treat 262k as the quality-faithful limit and 1M as
best-effort headroom.

## Speculative decoding

Two drafting options, both verified with this exact weight set:

### MTP — built into this checkpoint (best context)

The checkpoint ships Qwen3.8's multi-token-prediction head (~50 MB, 4-bit
quantized). Drafting with it needs **no download and no extra weights in
VRAM**, and skips a separate draft model's KV cache (~20% less KV per token
than an external drafter) — on a 24 GB card the full native 262k context fits
fully resident. Support is upstream exllamav3 (`--mtp` in `model_init`;
`DRAFT=mtp` in the serving kit).

Measured on DGX Spark (GB10): ~2.2 accepted tokens/step, ~30 tok/s
(HumanEval-class, T = 0.6).

### DFlash2 — companion draft model (fastest)

A companion EXL3 quant of the DFlash2 block-diffusion draft model exists:
[`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
(~+15% tokens/s vs MTP, at the cost of 1.4 GB of draft weights and its KV
cache). With it, measured on DGX Spark (GB10, 273 GB/s-class LPDDR5x):

| workload (T = 0.6) | accepted tokens/step | tok/s |
|---|---|---|
| HumanEval-style code | 4.43 | 47.5 |
| code prose (greedy) | 2.7–2.8 | 40–43 |
| code prose (T = 0.6) | 2.4–2.5 | 38–40 |

Acceptance is at parity with the unquantized bf16 draft; the quantized draft
added **+33%** decode throughput over the bf16 draft on the same hardware
(draft weight reads at 5bpw instead of 16). DFlash2 drafting in exllamav3 is a
fork feature.

## Quality gates (measured, this exact weight set)

- tool-eval-bench 2.5.1, hardmode, temperature 1.0, seed 42: **87–88 / 100**
  (84 scenarios, agentic tool calling through an OpenAI-compatible server;
  remaining failures triaged as model-judgment, not protocol)
- DFlash2 speculative acceptance: see table above; greedy output of the
  drafting lane verified token-identical to the non-speculative lane in spot
  runs
- Long-context: passkey gates as described above

## License

Apache-2.0, inherited from the base model. Quantization of an Apache-2.0 model
with EXL3 tooling; no additional restrictions.

## Credits

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Alibaba Qwen team)
- Quantization format & tooling: [exllamav3](https://github.com/turboderp-org/exllamav3) (turboderp)
