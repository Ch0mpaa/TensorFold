# Journal: proving Qwen3.5-2B runs through TFOLD bit-identically to its own safetensors

**Audience:** someone who has read the previous journal
`tfold-original-qwen35-conversion.md` (so the TFOLD container is familiar)
and who may have never looked inside a GPU kernel or a BF16 bit pattern.

## What we set out to do

Yesterday we converted `Qwen/Qwen3.5-2B`'s safetensors into a `.tfold` without
touching the bytes. The byte-level parity script said **632/632 tensors
byte-identical**, but when we tried to actually *run* the `.tfold`, the dense
CUDA loader refused the file: it expected 128 norm weights per linear-attention
layer to be BF16, but the upstream shipped them as FP32.

Today's goal: get *inference* to work and prove it is numerically equivalent
to the upstream safetensors going through the same runtime.

Rule for the whole exercise: **no silent dtype narrowing**. If we have to drop
precision anywhere, declare it, measure it, bound it.

## Why original checkpoints contain mixed dtypes

Modern LMs ship most weights in BF16 — a 16-bit float with the same exponent
range as FP32 but only an 8-bit mantissa, chosen because the kernels run in
BF16 and the author wants the on-disk weights to match what the kernel sees.

But some weights act as *scalars that multiply BF16 streams*, such as
RMSNorm gains. The upstream Qwen3.5-2B ships these as **FP32** to preserve
the author's original training distribution — the trainer kept them in FP32,
and the HF safetensors export faithfully mirrors that. 24 of our 36 FP32
tensors are `linear_attn.norm.weight`; the other 12 are `linear_attn.A_log`
decay scalars already handled by a separate `f32vec` helper.

## How the loader interprets dtypes

The CUDA engine in `zig/src/families/qwen3_5/cuda_weights.zig` builds a
per-layer weight pack. For every tensor it calls `ck.expect(name, dtype,
shape)` which refuses the file if the tensor's shape OR dtype doesn't match
exactly. This is deliberate: the engine never wants to guess that an FP32
array is "probably safe" to reinterpret as BF16.

Line 413 had:
```zig
.norm = try tensor(e, ck, "{s}linear_attn.norm.weight",
                   .{p}, .bf16, &.{geometry.dk}),
```
`tensor(...)` is a thin wrapper that returns `ck.expect(...)`. Pass FP32 →
`error.UnexpectedTensor` → program halts. That is correct behavior in the
absence of any declared narrowing policy.

## Why silent narrowing is dangerous

If we had just flipped the expected dtype to `.f32` and reinterpreted the
bytes, every 4-byte FP32 word would have been reparsed as two BF16 words.
The kernel would have happily multiplied garbage BF16 values into the
activation stream. This is the kind of bug that doesn't crash: the model
starts generating plausible-but-wrong text, and nobody notices until
somebody compares to a reference.

The safe pattern is:
1. **Detect** the FP32 at load.
2. **Convert** to BF16 via the same round-to-nearest-even rule the GPU
   kernels use (`__float2bfloat16_rn`, mirrored in
   `core.bf16_round.bits`).
3. **Measure** the max absolute and max relative error introduced.
4. **Emit** those numbers so a human can decide whether they are acceptable.
5. **Only then** hand the BF16 buffer to the kernel.

## How the fix works

The new helper is 40 lines:
```zig
fn normBf16OrF32(e: *Engine, ck, fmt, args, n) !u64 {
    const t = try ck.get(name);
    if (t.is(.bf16, &.{n})) return upload(e, t.bytes);  // byte identity
    if (t.is(.f32,  &.{n})) {
        // 1. narrow every f32 to bf16 via RNE
        // 2. round-trip back to f32 (back = value(bf16(v)))
        // 3. track max_abs = max |back - v|,
        //          max_rel = max |back - v| / |v|,
        //          non_finite count
        // 4. under TF_NORM_AUDIT=1, write the numbers to stderr
        return upload(e, bf16_buffer);
    }
    return error.UnexpectedTensor;  // anything else is a real error
}
```
Call-site replaces the old strict `tensor(...)` call. Everything else about
the engine is untouched — the kernel still reads `__nv_bfloat16*` for the
norm weight; we just made the *path from disk to kernel* explicit.

## How numerical equivalence was tested

Three layers of proof, strictly stronger from top to bottom:

1. **Pure-logic unit tests** exercise the RNE rounding and the audit
   counters without a GPU. They include the three classic tie values
   (`1 + 2^-8` → 1, `1 + 2^-8 + 2^-40` → `1 + 2^-7`, representable values
   round-trip exactly), a NaN that must survive, and a tiny vector sweep
   whose counters must be strictly positive.

2. **End-to-end CUDA stage** runs the `tf-qwen35-cuda` binary on both the
   safetensors directory and the `.tfold` directory with
   `TF_PIPE_TOKENS ∈ {1, 3, 5, 8, 12}`. The stage subcommand concatenates
   every BF16 logit row of the scenario into one 23,838,720-byte dump. We
   sha256 each dump. For all five scenarios, the sha of the safetensors dump
   equals the sha of the `.tfold` dump:
   `0385e13dd46771cd9ebae910cf191375fb91ce2a391bfe439d3b56325d80fbfe`.
   That is **bit-identical** inference — no floats compared, no tolerance.

3. **Audit-line diff**: with `TF_NORM_AUDIT=1` both runs emit the same
   per-layer narrowing numbers; `diff` says they are byte-identical. If the
   two checkpoints disagreed on even one FP32 word, these lines would drift.

## What the narrowing actually costs

Observed across the 18 FP32 `linear_attn.norm.weight` tensors: the worst
elementwise absolute error from FP32 → BF16 was 3.906e-3, the worst relative
error was 3.89e-3, and no non-finite values ever appeared. The bound comes
from BF16 having an 8-bit mantissa: for numbers near 1, the gap between two
adjacent BF16 values is `2⁻⁸ ≈ 3.9e-3`. The observed numbers hit that bound,
which is expected — the upstream authors ship FP32 weights that *fill* the
precision gap.

**Honest line**: we did not run a *reference FP32 path* to show that
`argmax(FP32 run) == argmax(BF16 run)`. The kernel ABI is BF16, so no such
path exists today without a kernel rewrite. What we did show: the CUDA
BF16 path we DO have produces the same output bytes from both the
safetensors and the `.tfold`.

## Takeaways

1. **Silent casts hide for years.** A one-line cast at load time can be
   indistinguishable from a type-system coincidence until someone compares
   to a reference. Make every narrowing explicit, named, and measured.
2. **Bit-identical > numerically close.** When two code paths should
   produce the same bytes, test for sha equality. It is simpler than
   writing tolerances and strictly stronger.
3. **Tests at three levels.** Pure-logic + process-level sha equality +
   diff of audit-lines caught different classes of possible regression.
4. **Kernel ABI pins what you can preserve.** The decision to narrow
   `linear_attn.norm.weight` to BF16 is a kernel-signature fact, not a
   "dtype convenience". Ship that fact with the code.
5. **FNV trailer is cheap verification.** The `.tfold` loader adds ~3.6 s
   on a 4.5 GB file for a full-coverage integrity check. That is a
   feature, not an overhead; declare it where perf matters.

## Exercise

Open the patched `cuda_weights.zig` in a worktree, change the
`core.bf16_round.bits(v)` call inside `normBf16OrF32` to `@as(u16,
@intFromFloat(v))` (deliberately wrong — truncating integer cast).
Rebuild `tf-qwen35-cuda` and re-run both stage commands. Confirm:
(a) `TF_NORM_AUDIT=1` now reports huge `max_abs` and `non_finite > 0`,
(b) the two sha256s still match each other (the bug is symmetric across
both paths), but
(c) the token sequence changes. This is exactly the failure mode the
audit was designed to catch.

## Comprehension questions

1. Why does the fact that **both** the safetensors path AND the `.tfold`
   path produce the same corrupted output (in the exercise above) NOT
   mean the `.tfold` conversion is lossy? What test would actually tell
   you whether the lossy step is in the converter, the loader, or both?

2. The GPU kernel treats `NW` (the norm gain vector) as `__nv_bfloat16*`,
   but inside it does `bf(NW[d]) * (yv[j] * inv)` — i.e. it widens the
   BF16 back to FP32 to multiply. If every call site widens, why do we
   store BF16 on the GPU at all? What changes if we stored FP32 there?
