# Race C — EXL3 low-bit baseline PRESERVED (not touched by Race 001)

The EXL3 `.tfold` checkpoint `/home/chris/models/tfold/Qwen3.5-2B-exl3-mcg.tfold`
(sha256 `027ee2964e3868fa…`, 3,439,095,240 bytes, created Nov 14 22:04 UTC)
is **unchanged** by Race 001.

## Its previously measured scenario numbers (from the `original-parity` milestone)

| metric | value | source receipt |
|---|---|---|
| scenario compute | median 2630 ms (3 runs: 2596, 2630, 2664) | docs/tfold/parity-receipts/summary.json |
| peak VRAM | 3.24 GiB | docs/tfold/parity-receipts/summary.json |
| logit dump sha256 | `c22df21edf589eceaa8405b6cd75883d9b97bd493fbc042c4cf5811238f5340f` | docs/tfold/parity-receipts/summary.json |
| greedy token sequence | `197 197 13 197 197 13 197 197 13 197 197 13 197 197 13 197 197 13 197 13 13 197` | docs/tfold/parity-receipts/summary.json |

## Why Race 001 does NOT mix these numbers with Race A/B

The brief for Race 001 pins the model to the **unquantized BF16 upstream
Qwen3.5-2B**. Everything in `receipts/race_a.jsonl`, `receipts/vllm_sweep.json`,
and `receipts/vllm_dtype_audit.json` loads **unquantized** weights.

EXL3 is an 8.00-bpw quantized representation with its own tile layout,
`cuda_exl3_weights.zig` loader, and `exl3_mcg_tile` dispatch. It is a
different numerical path (`.tfold`'s quant block has `scheme = "exl3_mcg"`
there and `scheme = "none"` for Race 001's checkpoint), and its throughput /
VRAM live on different Pareto axes.

The EXL3 argmax tokens happen to match BF16's argmax tokens for this
particular set of test prompts — which is interesting as a "no obvious
quant-induced token flip" check, but is NOT evidence of throughput or
latency equivalence.

## What the next race ("002") will exercise

A **low-bit race** compares:
- EXL3 8.00-bpw `.tfold` (Qwen3.5-2B-exl3-mcg.tfold) through TFOLD,
- potentially EXL3 lower bit widths if produced,
- whatever vLLM supports for the same base model (e.g., quantization=`gptq`,
  `awq` or `bitsandbytes` variants, but ONLY those backing the same base
  model identity),
- with the same prompt freeze and the same CONFIG.md discipline.

That race is out of scope here and will get its own directory
(`docs/dyno/race-002/`).
