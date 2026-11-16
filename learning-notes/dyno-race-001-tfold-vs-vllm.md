# Journal: the first Dyno Race — TFOLD vs vLLM, fairly measured

**Audience:** someone who has read the previous two journals (`tfold-original-qwen35-conversion.md`
and `original-tfold-inference-parity.md`) so the TFOLD container, loader fix,
and numerical narrowing are familiar.

## The job

Build the **first reproducible performance comparison** between the TensorFold
CUDA runtime and vLLM, on the same unquantized Qwen3.5-2B upstream model.
Record every input hash. Keep the EXL3 low-bit baseline strictly out of the
numbers. Decide a winner per metric; declare *no winner* when the comparison
isn't apples-to-apples.

Three sub-races:

- **Race A** compares two *containers* feeding the same TFOLD runtime:
  the official safetensors directory vs the losslessly converted `.tfold`.
  Correctness and startup, not model quality.
- **Race B** compares two *runtimes* on the same model: TFOLD CUDA vs vLLM.
  Throughput, latency, memory.
- **Race C** reminds us that the existing EXL3 `.tfold` baseline is a
  separate race ("002") and never gets pooled into these numbers.

## What each metric actually measures

- **Logit SHA**: the full output tensor every greedy step emits, hashed.
  Two runs with identical SHA are bit-identical; that's strictly stronger
  than "max absolute difference < epsilon".
- **Greedy token sequence**: the IDs chosen at each step. A difference here
  means model behavior diverged, not just internal arithmetic.
- **Scenario compute (ms)**: how long the GPU forward passes take. Does not
  include loading or integrity checks.
- **Wall-clock (s)**: everything from process start to exit. Open + load +
  integrity + compute + teardown. A user watching their terminal sees this.
- **FNV verify (ms)**: the `.tfold` loader's full-file FNV-1a-64 integrity
  check. Pure byte read with unrolled hashing. Scales with file size.
- **Peak VRAM (GiB)**: highest resident device-memory allocation. On GB10
  this is unified with host memory; we still track it.
- **TTFT (s) — vLLM only**: time to first *generated* token. Dominated by
  prefill + scheduler + kernel dispatch latency.
- **Decode tok/s — vLLM only**: how fast new tokens stream once the KV
  cache is warm. The "steady-state" number everyone quotes.
- **Total throughput (tok/s)**: for batched runs, sum of tokens produced
  across all concurrent requests divided by batch wall time.
- **P50 / P95 wall (s)**: request-latency percentiles over a batch. P50 is
  typical UX; P95 is the tail everyone forgets.

## Why the comparison *can* be fair

Mixed-dtype checkpoints are usually where "fair" benchmarks go to die. The
upstream Qwen3.5-2B ships 596 BF16 tensors and 36 FP32 tensors. The FP32
set is the `linear_attn.norm.weight` and `linear_attn.A_log` scalars that
RMSNorm uses.

If TFOLD narrowed FP32→BF16 silently and vLLM kept FP32, the comparison
would be bogus — vLLM would be doing more work per step. We settled this
*before* measuring throughput:

- TFOLD's loader uses the new `normBf16OrF32` helper: explicit RNE narrow,
  per-tensor max_abs/max_rel/non_finite audit under `TF_NORM_AUDIT=1`,
  bounded by `≤ 2⁻⁸ ≈ 3.9e-3` for unit-scale weights.
- vLLM 0.26 with `dtype='bfloat16'` runs the default weight loader at
  `vllm/model_executor/model_loader/weight_utils.py`, which does
  `param_data.copy_(loaded_weight.to(param_data.dtype))`. That's PyTorch's
  RNE cast — the same rounding rule nvcc uses in `__float2bfloat16_rn`
  and that `core.bf16_round.bits` implements.

Both runtimes keep BF16 on device after narrowing the FP32 scalars with the
same rule. **The throughput comparison is apples-to-apples.** The only
policy difference is that TFOLD prints the narrowing error; vLLM does not.

## Why Race A's winner is a tie (and a feature declaration)

Race A loads the *same model* through TFOLD twice: once from the upstream
safetensors directory, once from the `.tfold` we built yesterday. Same
binary, same inputs, same prompts. Measured over 22 timed runs:

- Logit dump sha across *all 22 runs*: a singleton `{0385e13dd46771cd…}`.
- Token sequence across all 22 runs: a singleton.
- `TF_NORM_AUDIT` lines across both paths: byte-identical under `diff`.
- Scenario compute: 875 ms median (safetensors) vs 884 ms (.tfold) — 9 ms,
  well below the 27 ms single-path variance.

But wall-clock: safetensors 3.58 s, `.tfold` 7.48 s. That's a 3.9 s gap, and
it is **entirely** the FNV-1a-64 integrity trailer verification. We can see
it in the `fnv_us` phase stat (median 4529 ms). The safetensors path has no
equivalent full-file check.

So the Race A verdict is not "safetensors is faster". It is **two separate
truths**:

- *Compute:* tie.
- *Startup:* safetensors wins by one integrity check's worth of time,
  because it does less.

The `.tfold` loader exposes `ParseOptions.skip_fnv` for callers that have
already verified the file externally. We don't flip it, because for a race
that frames integrity-in-the-loader as part of the product, that would be
cheating.

## Why Race B's winner is more nuanced

vLLM 0.26 can batch. The `tf-qwen35-cuda` binary we built is a
single-scenario stage runner; it does not expose a batched decode CLI.
Honest conclusions:

- At concurrency = 1 we can report both and compare like-for-like.
- At concurrency > 1 the only number is vLLM's; TFOLD gets a `no-winner`
  mark on those rows and the REPORT states that the CLI, not the runtime,
  is the blocker.

The brief asked for this explicitly: "Do not mix low-bit race with
original-weight race. Declare no-winner where comparison isn't fair."
Those rules are what let the table be readable and trusted.

## What this establishes

- **Byte-identical inference across safetensors and .tfold through our
  runtime.** That is not a weak "numerically close"; it's a sha equality.
- **FP32 narrowing policy is now documented AND identical** between TFOLD
  (explicit) and vLLM (silent, via dtype arg).
- **A reproducible benchmark harness**: `race_a_bench.sh`, `vllm_sweep.py`,
  `vllm_dtype_audit2.py`, `analyze_race_a.py` with every input hash pinned
  in `CONFIG.md`. Someone can rerun the race in a day.
- **A shape for future races**: once a batched TFOLD CLI exists, filling
  the concurrency rows is a one-line protocol change.

## Exercise

Open `race_a_bench.sh` and change `TOKENS` from `5` to `1`. Re-run. You
should observe:

- The scenario budget drops (fewer tokens per window, fewer compute rows).
- The `fnv_us` phase stat barely changes (it is a function of file size,
  not prompt length).
- The sha of the logit dump is **different** from the `5`-token sha (you
  measured a different workload) but it is still **a singleton** across
  both paths (parity still holds).

The exercise teaches two things: that the integrity cost is a fixed
overhead, and that parity is a per-workload invariant.

## Comprehension questions

1. Race A's `wall_s` winner is safetensors; Race A's `scenario_ms` winner
   is a tie. Explain in one sentence why those two answers are **not** in
   conflict. (Hint: what happens between process start and the first
   kernel launch?)

2. The FP32 → BF16 narrowing error for the 128-element
   `linear_attn.norm.weight` tensors is bounded at `≤ 2⁻⁸`. If a future
   version of Qwen shipped the same tensors in FP64 instead, would the
   TFOLD audit's `max_abs_err` go up, down, or stay the same, and why?
