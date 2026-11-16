# Candidate Optimization — Fuse `tf_exl3_rot_in` into `tf_exl3_linear_*` prologue

> Picked ONE candidate per the brief. Implementation lives on a separate
> branch, never touches baseline receipts.

## Evidence (from `receipts/kernel_bottleneck.md`)

On the EXL3-8bpw `.tfold` path the top two kernels together account for
**97.0%** of GPU time:

- `tf_exl3_linear_16_1_4` — 66.9% (trellis dequant + matmul fused)
- `tf_exl3_rot_in`      — 30.1% (pre-matmul Hadamard-like rotation on input)

They are launched in a near-1:1 pair (6,400 vs 6,404 instances across a
48-logit-row stage). Average per-pair cost ≈ 1.55 ms. The rotation is a
small, bandwidth-bound op dispatched independently for every projection.

## The hypothesis

The rotation output feeds into the first MMA of the linear kernel's
prologue. If we fold the rotation into the linear kernel's shared-memory
load, the rotation never materialises in global memory and the separate
kernel disappears.

Upper bound on win:
- Eliminate the 30.1% rotation kernel.
- Eliminate 6,400 kernel launches per stage (scheduler/driver overhead
  invisible to `cuda_gpu_kern_sum` but measurable with nsys timeline).
- Keep the trellis decode + MMA cost unchanged.

Realistic win (conservative guess, to be measured):
- 15-25% end-to-end stage speedup on EXL3-8bpw.
- No change to BF16 or GGUF paths.

## Success criteria (pre-committed)

1. **Correctness**: logit dump sha256 of fused EXL3 run **equals**
   current `c22df21edf589ec…`. Bit-identical. If anything differs, the
   fusion is wrong.
2. **Throughput**: scenario_ms median across 10 warm runs drops by **≥
   10%** vs current 5,819 ms median (goal 4,400-5,000 ms).
3. **No regression** elsewhere: BF16 scenario_ms stays within ±2% of
   884 ms.
4. **Launch count**: nsys reports `tf_exl3_rot_in` instance count = 0 for
   the fused path.

## Non-goals

- NO fine-tuning.
- NO new quantizer, no new container format, no format-version bump.
- NO Android, no new UI, no agent work.
- NO claim that this makes TFOLD beat EXL3-standalone, GGUF or vLLM —
  it is a within-TFOLD EXL3-path improvement only.

## Scope / implementation branch

- File: `zig/kernels/cuda/qwen35.cu` (prologue of
  `tf_exl3_linear_*` family).
- Separate branch: `race-002/candidate/exl3-fuse-rot-in`.
- Will NOT be merged into `stage6b-cuda` until all four success criteria
  pass. Baseline receipts in this directory are frozen; they are the
  reference the candidate measures against.

## Honest disclosures

- The 30.1% figure is a **TFOLD-stage-only** measurement on a 48-row
  scenario. Different sequence lengths, different batch sizes, and the
  decode phase (post-prefill) may show a different ratio. The candidate
  must be re-profiled at the decode steady state before any absolute
  claim.
- We have not controlled for driver cache state across runs; the
  launch-overhead portion of the win may be smaller after `nvcuda`
  warms up its graph-launch path.
- We do NOT claim this fusion outperforms EXL3's own reference kernels
  until the same comparison runs on both.
