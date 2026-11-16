# Race 001 — Correction: cold-load winner claim was unsupported

**What the original REPORT said**
> **Cold load** — s — **TFOLD (both paths)** — 3.58 s (A) / 7.48 s (B) — vLLM 412.58 s — "TFOLD wins by 55× vs vLLM (FlashInfer/FlashAttention JIT first-touch dominates vLLM)".

**Why the 55× victory claim is unsupported**

The two wall-clock numbers measure different workloads:

- The TFOLD 3.58/7.48 s wall-clock is a *scenario run* through a prebuilt,
  statically-linked Zig binary (`tf-qwen35-cuda`). CUDA kernels are
  pre-compiled into the fatbin (via `-Dnvcc=…` at build time). Launch cost
  is near-zero.

- The vLLM 412.58 s wall-clock includes **first-touch compilation** of
  FlashAttention and FlashInfer kernels, graph-break analysis, the custom
  fusion pass (`norm_quant`, `act_quant`), eager-mode CUDA graph capture,
  Triton/FLA GDN prefill kernel JIT, and the tokenizer initialization.
  **A second** vLLM load — with the Triton/Flash caches warmed — is
  considerably faster.

Those are not the same thing. Calling it a 55× TFOLD win elides the
"prebuilt-vs-JIT" asymmetry. The honest statement is:

> **Prebuilt binary launch (TFOLD) vs JIT-compile engine (vLLM, cold
> cache):** 3.58 s vs 412.58 s. This gap is dominated by vLLM's
> first-touch JIT path, not by anything TFOLD does better at runtime.

**What's preserved and what's annotated**

- `docs/dyno/race-001/receipts/vllm_sweep.json` — unchanged. The raw
  `cold_load_s: 412.58` number stays as recorded.
- `docs/dyno/race-001/REPORT.md` — gets an inline annotation right next
  to the cold-load row that points at this file; the receipt is NOT
  rewritten.
- `docs/dyno/race-001/REPORT.md`'s Race A numbers (sha-identical logits,
  scenario compute tie, startup 3.9 s FNV gap) are unaffected.

**What a fair cold-load comparison would look like**

1. Measure vLLM with a warmed Triton/Flash cache (`TORCHINDUCTOR_CACHE_DIR`
   persisted across runs). Repeat 3×, take median.
2. Measure TFOLD in the same shell, same filesystem, same number of
   FNV passes.
3. Report both cold-cache and warm-cache numbers side by side.

That measurement is not part of Race 002's quantization scope. It is
recorded here as follow-up ("Race 001b"), not retroactively papered over.

**Explicit statement**

Race 002 does **not** reuse Race 001's cold-load number as evidence.
Only the Race A `scenario_ms` tie, the FP32-norm RNE narrowing
equivalence, and the EXL3 preservation carry forward.
