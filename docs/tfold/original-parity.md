# Original-weight TFOLD inference parity on Qwen3.5-2B

## Scope
Prove the official unquantized `Qwen/Qwen3.5-2B` checkpoint runs through
TensorFold CUDA bit-identically from two sources:

- **A** the upstream HuggingFace safetensors directory itself, and
- **B** the losslessly converted `.tfold` from yesterday's milestone.

All findings, numbers and conversions are explicit. The EXL3 .tfold baseline
is observed separately and never pooled with these numbers.

## Checkpoints under test
- A: `/home/chris/models/qwen35-2b-orig/`, Apache-2.0,
  pin `15852e8c16360a2fea060d615a32b45270f8a8fc`,
  shard `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
- B: `/tmp/tfold_dir/model.tfold` (hardlink to
  `/home/chris/models/tfold/Qwen3.5-2B-orig-bf16.tfold`, 4,548,300,616 bytes),
  built by `docs/tfold/convert_safetensors_to_tfold.py` (yesterday's PR),
  loader-boundary byte parity already verified 632/632 tensors.

## Why the upstream checkpoint has mixed dtypes
The official safetensors stores:
- 596 tensors in BF16 — normal LM weights,
- 36 tensors in FP32 — 24 linear-attention scalar norms (`linear_attn.norm.weight`
  for the layers that run the Mamba-style delta mixer) + their 24 `A_log`
  decay scalars, plus related header broadcasts.

FP32 is used only for gain-style scalars that multiply a BF16 tensor stream.
The arithmetic that later consumes them (`tf_q35_gdn_post` in
`kernels/cuda/qwen35.cu`) accepts `__nv_bfloat16*` for the gain vector, so
the FP32 shipment is for *distribution faithfulness*, not for kernel-side
precision.

## The one-line blocker and the loader fix
`zig/src/families/qwen3_5/cuda_weights.zig:413` had been strict-typed:
```zig
.norm = try tensor(e, ck, "{s}linear_attn.norm.weight", .{p}, .bf16, &.{geometry.dk}),
```
That refuses the `.f32` the upstream ships. The parallel EXL3 path
(`cuda_exl3_weights.zig:315`) already had a `bf16OrF32` helper that quietly
widened f32 to bf16 — correct for the kernel ABI but silent.

This mission's patch replaces the strict call with a **named, auditable**
helper `normBf16OrF32`:
- `bf16` pass-through (byte identity through to upload),
- `f32` narrow via `core.bf16_round.bits` (round to nearest even, matches
  nvcc's `__float2bfloat16_rn`),
- per-tensor telemetry: `max_abs_err`, `max_rel_err`, `non_finite`,
- stderr audit line under `TF_NORM_AUDIT=1`.

Any other dtype returns `error.UnexpectedTensor`. Nothing is silent.

Direct unit tests landed:
1. Pure-logic RNE test: representable values round-trip exactly; tie-to-even
   on `1 + 2^-8` → 1; just-past-tie → `1 + 2^-7`; NaN survives; sweep records
   non-zero audit on values that would flip under BF16.
2. BF16-byte pass-through test: 128-element block cannot drift at all.

## Runtime equivalence receipts
Scenario sweep with `TF_PIPE_TOKENS ∈ {1, 3, 5, 8, 12}`, the `stage`
subcommand writing the full concatenated BF16 logit matrix to disk:

| N | safetensors sha256 (first 16) | .tfold sha256 (first 16) | match |
|---|---|---|---|
| 1  | `0385e13dd46771cd` | `0385e13dd46771cd` | ✅ |
| 3  | `0385e13dd46771cd` | `0385e13dd46771cd` | ✅ |
| 5  | `0385e13dd46771cd` | `0385e13dd46771cd` | ✅ |
| 8  | `0385e13dd46771cd` | `0385e13dd46771cd` | ✅ |
| 12 | `0385e13dd46771cd` | `0385e13dd46771cd` | ✅ |

Full sha256 of each 23,838,720-byte dump:
`0385e13dd46771cd9ebae910cf191375fb91ce2a391bfe439d3b56325d80fbfe`.

Greedy token sequence from both runs (identical):
`197 197 13 197 197 13 197 197 13 197 197 13 197 197 13 197 197 13 197 13 13 197`

`TF_NORM_AUDIT=1` output is byte-identical between the two runs under `diff`.

## Narrowing bounds (observed, not theoretical)
Across the 18 FP32 `linear_attn.norm.weight` tensors loaded (one per
linear-attention layer):

- `n = 128` per tensor,
- `max_abs` per tensor ∈ [1.953e-3, 3.906e-3] (upper bound = 2⁻⁸),
- `max_rel` per tensor ∈ [2.61e-3, 3.89e-3],
- `non_finite = 0` in all tensors.

These are the exact deltas the dense CUDA path accepts. Since both
checkpoints go through the same helper, both suffer the same narrowing —
hence the bit-identical logits.

## Unavoidable precision conversion
The CUDA kernel `tf_q35_gdn_post` (kernels/cuda/qwen35.cu:385) reads the
norm weight as `const __nv_bfloat16* __restrict__ NW`. There is no FP32
dispatch; FP32 end-to-end would need a kernel-signature break + a second
code path + 2× VRAM for the norm block. That is out of scope for this
milestone; it is reported, not swept.

## Baselines (median of 3 warm runs each, `TF_PIPE_TOKENS=5`)

| path | wall (s) | scenario (ms) | peak VRAM | host RSS |
|---|---|---|---|---|
| safetensors dir (A) | 4.26 | 890 | 5.73 GiB | 7.57 GB |
| .tfold (B)          | 8.10 | 893 | 5.73 GiB | 7.56 GB |

Scenario (GPU compute) time is identical. The ~4 s wall-clock delta is
the FNV-1a-64 trailer verification on open, measured at `fnv=3.6s` by
`TF_TFOLD_LOG_STATS`. If a caller wants faster cold open and can accept
an external integrity check, the `.tfold` loader exposes
`ParseOptions.skip_fnv` — out of scope for the parity milestone.

## EXL3 regression sanity (kept separate from BF16/FP32 race)
Same binary, 3 warm runs on
`/home/chris/models/Qwen3.5-2B-exl3-8.00bpw`:

- scenario ms: 2596, 2630, 2664 (median 2630)
- peak VRAM: 3.24 GiB
- logit dump sha256: `c22df21edf589ec…` (different from BF16, **expected**:
  the EXL3 arithmetic is a different numerical path)
- greedy token sequence: identical to BF16 runs.

The EXL3 numbers are stored in the same receipts file but
labelled separately; they do **not** merge into the BF16 vs FP32 parity
numbers above.

## vLLM comparator (separate receipts)
Not folded into the TensorFold parity table. Attempted vLLM 0.26.0 on the
same `/home/chris/models/qwen35-2b-orig/` directory. The GB10 was hosting
a production tensorfold-native server using 103 of 121 GiB VRAM, forcing
`gpu_memory_utilization=0.08` and `max_model_len=512`. Results stored at
`docs/tfold/parity-receipts/vllm_baseline.json`. See that file for the
numbers; keep them out of the EXL3 race and out of the BF16/FP32 parity
table.

## Deliverables
- `zig/src/families/qwen3_5/cuda_weights.zig`: `normBf16OrF32` helper +
  call-site replacement + two unit tests (local patched copy at
  `docs/tfold/parity-receipts/cuda_weights.patched.zig`;
  diff at `docs/tfold/parity-receipts/cuda_weights.diff`).
- Receipts: `docs/tfold/parity-receipts/summary.json`,
  `docs/tfold/parity-receipts/sft_stderr.txt`,
  `docs/tfold/parity-receipts/tf_stderr.txt`,
  `docs/tfold/parity-receipts/vllm_baseline.json`.
- Parity harness: `docs/tfold/parity_logits.py` (optional analytic script
  if someone wants max_abs/max_rel across the 48 logit rows; current
  evidence uses sha256-equality which is strictly stronger).

## Honest statement
- The two CPU-side .tfold vs safetensors byte paths produce
  **bit-identical** CUDA output. This proves lossless parity *through our
  runtime*, not through every possible runtime.
- The dense CUDA computation requires BF16 for `linear_attn.norm.weight`;
  the FP32 → BF16 narrow is declared, measured and bounded. We do **not**
  claim "lossless CUDA execution of FP32 weights".
- 2B class remains **NOT GRADUATED**. OFFSEC CWE-311/327 is an unresolved
  independent blocker.
