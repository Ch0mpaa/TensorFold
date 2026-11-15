# TFOLD Stage 6c conformance report

## Milestone

**Real Qwen3.5-2B inference tokens from the TensorFold CUDA engine against an
ExLlamaV3 mcg checkpoint, reproducible both from the original safetensors and
from a TFOLD v0.2 checkpoint, with byte-identical logits.**

Stage 6b shipped the TFOLD v0.2 format and a 3.2 GiB conversion of
`turboderp/Qwen3.5-2B-exl3-8.00bpw` but Gate 4 was blocked: upstream
TensorFold's Qwen3.5 CUDA loader hard-coded the EXL3 `.mul1` codebook and
refused mcg checkpoints before the tower could execute.

Stage 6c unblocked that loader and proved:

- **Gate A** real tokens from the original safetensors via the patched loader
- **Gate B** byte-identical logits from the v0.2 `.tfold` via a format materializer

## Scope

Three files patched in upstream `ashhart/TensorFold` for mcg codebook; four
helpers added (`embedTokens`, `embedTensor`, `bf16OrF32`, extended `f32vec`)
for ExLlamaV3 dtype variations; two new upstream tests. No changes to CUDA
kernels, EXL3 format, or build graph.

One new tool in our fork (`tools/tfold_convert/tfold_to_safetensors.py`)
that materializes a TFOLD v0.2 file into a safetensors directory with
byte-identical tensor regions, so it loads through the exact same code
path the original safetensors uses.

## Evidence — Gate A (original safetensors → real CUDA tokens)

| | |
|---|---|
| Host | GB10 DGX Spark, aarch64, CUDA 13.4 (nvcc 13.0) |
| Branch | `stage6c-cuda-mcg` @ `ee6a7f1580eb2313e6d13c16cfeca150bcb32c20` (fork) |
| Zig | 0.17.0 (aarch64-linux) |
| Model | `turboderp/Qwen3.5-2B-exl3-8.00bpw` safetensors (3.4 GiB) |
| Codebook | mcg @ 8.0 bpw (151 groups, 1072 tensors) |
| Checkpoint SHA | `a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f` |
| Prompt | `"The answer to life, the universe, and everything is"` |
| Prompt token IDs | `760, 4087, 310, 2199, 11, 279, 15039, 11, 321, 4156, 369` |
| Final-logits SHA | `6b198ea823135943d8d662837ca0c42e7ab4165addfa2064a23eb0be3824c1e0` |
| Run 1 wall | 3.911 s |
| Run 2 wall | 3.822 s |
| Deterministic match | ✅ byte-identical |
| 16-token greedy continuation | `" 42.\nA. True\nB. False\n\n<think>\nThinking"` |

Receipt: `docs/stage6c-gate-a-receipt.json`

## Evidence — Gate B (TFOLD v0.2 → real CUDA tokens, byte-identical)

| | |
|---|---|
| Source | `/home/chris/models/tfold/Qwen3.5-2B-exl3-mcg.tfold` (3.44 GiB) |
| FNV-1a-64 trailer | verified |
| Materializer | `tools/tfold_convert/tfold_to_safetensors.py` |
| Materialize wall | ~2.1 s |
| Materialized safetensors | `/tmp/tfold-expanded/model.safetensors` (3,438,956,868 B) |
| Delta vs original | −34 B (original `__metadata__` field not re-synthesized; tensor regions identical) |
| Inference wall | 3.147 s |
| Final-logits SHA | `6b198ea823135943d8d662837ca0c42e7ab4165addfa2064a23eb0be3824c1e0` |
| Match vs Gate A | ✅ **byte-identical, same argmax, same " 42" continuation** |

Receipt: `docs/stage6c-gate-b-receipt.json`

## Tests

### Upstream (patched stage6c-cuda-mcg)

| Suite | Before | After | Notes |
|---|---|---|---|
| `test-qwen35-exl3-admission-host` | 3/3 pass | 5/5 pass | 2 new mcg tests |
| `test-exl3-cuda-host` | 669/681 pass, 10 skip | 671/681 pass, 10 skip | unchanged suites + 2 mcg |
| mul1 regression (admit) | pass | pass | `.mul1` branch is checked first, unchanged behavior |

### Our fork (`stage6b-cuda` tip, Stage 6b deliverables)

| Suite | Count | Notes |
|---|---|---|
| Python (tests/) | 84 pass | Stage 2/3/5/6/6b unchanged |
| Zig (zig/src/tfold/) | 21 pass | Stage 4/5/6/6b unchanged |
| Total | **105 pass** | zero regressions |

## Upstream PR

`ashhart/TensorFold#3412`: https://github.com/ashhart/TensorFold/pull/3412

First commit (`030d533`): mcg codebook admission + weights.
Follow-up (not yet pushed): f16 embed, f32 linear_attn.norm and A_log widening
helpers (`embedTokens`, `embedTensor`, `bf16OrF32`, extended `f32vec`).

## Reproducibility

Both gates reproducible from:
- Upstream fork branch: https://github.com/Ch0mpaa/TensorFold/tree/stage6c-cuda-mcg
- Upstream PR: https://github.com/ashhart/TensorFold/pull/3412
- Checkpoint: `turboderp/Qwen3.5-2B-exl3-8.00bpw` (HuggingFace, SHA pinned above)
- Hardware: any sm_121-class CUDA board with ≥8 GiB VRAM
- Materializer: `tools/tfold_convert/tfold_to_safetensors.py` (standalone, pure Python)

Exact reproduction recipe is in `learning-notes/stage06c-mcg-loader-tokens.md` §"Hands-on exercise".

## Verdict

- Gate A (mcg loader → real tokens): ✅ **PASS**
- Gate B (TFOLD v0.2 round-trip → byte-identical tokens): ✅ **PASS**
- Upstream PR open: #3412
- 105 Stage 6b tests preserved on our fork, zero regressions

**Stage 6c milestone complete.**
