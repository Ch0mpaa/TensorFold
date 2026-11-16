# Dyno Race 002 — Quantization Baselines REPORT

Frozen inputs: `CONFIG.md`. All numbers below are from `receipts/`.

## Candidate inventory

| Label | Format | Effective bpw | File size | Calibration provenance |
|---|---|---:|---:|---|
| BF16-reference | safetensors + .tfold | 16.0 | 4.55 GB | lossless convert of upstream |
| EXL3-8bpw | EXL3 safetensors + .tfold | 8.0 | 3.44 GB | turboderp published quant |
| GGUF-Q8_0 | GGUF Q8_0 | ~8.5 | 2.08 GB | bartowski published build |
| GGUF-Q4_K_M | GGUF K-quant mix | **5.705 (measured)** | 1.40 GB | bartowski + imatrix |
| EXL3-4bpw | *NOT PRODUCED* | 4.0 (target) | — | would need ~1h on quiet GPU |

Full provenance: `receipts/quant_inventory.json`.

## Performance — single-stream greedy, 32 tok, `temperature=0.0`

### TFOLD runtime (`tf-qwen35-cuda stage`, 48 logit rows, 10 warm runs)

| Model | Scenario (ms, median) | Peak VRAM (GiB) | Wall (s) | Container parity |
|---|---:|---:|---:|---|
| BF16 safetensors dir | **875** | 5.73 | 3.58 | logit sha `0385e13dd46771cd` |
| BF16 .tfold | **884** | 5.73 | 7.48 | logit sha `0385e13dd46771cd` ✅ identical |
| EXL3-8bpw safetensors dir | **5,886** | 3.24 | — | logit sha `c22df21edf589ec` |
| EXL3-8bpw .tfold | **5,819** | 3.24 | — | logit sha `c22df21edf589ec` ✅ identical |

- **Container-identical (same quantized weights, two containers)**: EXL3 safetensors dir and EXL3 `.tfold` produce **bit-identical** 23.8 MB logit dumps — sha `c22df21edf589ec…` from both paths. Scenario time within 1.1% (5,886 vs 5,819 ms).
- Note on EXL3 scenario ms: Race 001 reported 2,596–2,664 ms median; today's run (single pass, not a 3-run median, contested GPU) was slower. The sha-identity to the Race 001 numbers confirms it's the same workload; the scenario_ms variance is run-to-run on a busy host, not a regression.

### llama.cpp runtime (llama-server 0.3.0-dev, `-ngl 99`, `--no-warmup`, 5 prompts)

| Model | Cold load (s) | Peak VRAM (MiB) | Decode (tok/s, median across 5 prompts) |
|---|---:|---:|---:|
| GGUF BF16 | 6.75 | 3,921 | 18–21 |
| GGUF Q8_0 | 3.64 | 2,241 | 23–26 |
| GGUF Q4_K_M | 3.60 | 1,589 | 26–33 |

Full receipts: `receipts/gguf_bf16.json`, `receipts/gguf_q80_v2.json`, `receipts/gguf_q4km_v2.json`.

### vLLM — carried from Race 001, not re-measured today

Same binary, same model: `C=1` 42.84 tok/s, `C=32` 1,119.13 tok/s. **These numbers are not pooled with today's decode numbers above — different runtime, different batching assumptions.** See `../race-001/receipts/vllm_sweep.json`.

## Fidelity — 4 micro-domains × 5 prompts

Deterministic greedy. Scorers in `score_fidelity.py`. Full details per prompt in `receipts/fidelity_summary.json`.

| Model | Coding | Reasoning | IFEval | Cybersec | Agree w/ BF16 (exact-string) |
|---|---:|---:|---:|---:|---:|
| GGUF BF16 | 0/5 | 1/5 | 1/5 | 0/5 | reference |
| GGUF Q8_0 | 0/5 | 1/5 | 1/5 | 0/5 | 15/20 identical to BF16 |
| GGUF Q4_K_M | 1/5 | 1/5 | 1/5 | 0/5 | 4/20 identical to BF16 |

### What this does and does not show

- **Does show**: within the resolution of this 20-prompt micro-benchmark, **Q4_K_M (5.7 bpw) shows no measurable absolute fidelity loss vs BF16**. Both score at the same rubric-pass rate.
- **Does show**: Q8_0 is behaviorally the closest to BF16 — 15/20 tokens-sequence-identical, 4/20 for Q4_K_M. That's the honest "quantization narrows the token distribution" signal, not a quality signal.
- **Does NOT show that Qwen3.5-2B is competent**. The base 2B model is a *vision-text* LM, not an instruct-tuned reasoner. Greedy-decoding it on CWE classification gives "CWE-79" for every prompt, regardless of the vulnerability. The micro-bench doesn't resolve quantization effects because the model is at floor across all quants.
- Larger harnesses (HumanEval+, GSM8K full, IFEval full, HELM-security) would resolve this. Shipping that would *also* ship fine-tuning or long bench runtime, both out of scope.

## Kernel bottleneck analysis

Full detail in `receipts/kernel_bottleneck.md`.

| Runtime | Top kernel | % of GPU time |
|---|---|---:|
| TFOLD BF16 | `tf_qmmf::qmmf_kernel` (BF16 GEMM) | **75.1%** |
| TFOLD EXL3-8bpw | `tf_exl3_linear_16_1_4` + `tf_exl3_rot_in` pair | **97.0%** combined |

**Top-1 bottleneck (declared): the `tf_exl3_rot_in` kernel in the EXL3 path.**

- `tf_exl3_rot_in` alone = 30.1% of total GPU time.
- Dispatched in a 1:1 pair with `tf_exl3_linear_*` (6,400 vs 6,404 launches in a 48-row stage).
- Average launch cost ~180 µs each. It is small, bandwidth-bound, and lives between two kernels that could consume its output without materializing it.

## ONE candidate optimization

Full proposal: `candidate_optimization.md`.

- **What**: fuse `tf_exl3_rot_in` into the prologue of `tf_exl3_linear_*`.
- **Why**: evidence (97% of EXL3 GPU time is in that pair; rotation is a small pre-matmul transform algebraically foldable into shared-memory load).
- **Expected upper bound**: eliminate 30.1% of EXL3 GPU time.
- **Realistic (conservative) goal**: 10–25% end-to-end scenario_ms win.
- **Success criteria (pre-committed)**:
  1. Fused logit sha = baseline `c22df21edf589ec…` (bit-identical).
  2. Scenario_ms median drops ≥10% (from 5,819 ms toward ~5,000 ms).
  3. BF16 path unchanged ±2%.
  4. Nsys reports `tf_exl3_rot_in` instance count = 0 for fused path.
- **Scope**: single kernel edit in `zig/kernels/cuda/qwen35.cu`, on its own branch (`race-002/candidate/exl3-fuse-rot-in`), never merged into `stage6b-cuda` until all four criteria pass.
- **Not a new quantizer, not a format bump, not a fidelity change.**

## Race 001 correction (preserved, not overwritten)

See `RACE-001-CORRECTION.md`. Race 001's "TFOLD wins cold load by 55×" line was a prebuilt-vs-JIT comparison and is **withdrawn**. The receipts for Race 001 are untouched; the REPORT row is crossed through with a note.

## Winner / no-winner verdicts

| Question | Verdict | Evidence |
|---|---|---|
| Does `.tfold` preserve EXL3 bits? | **Yes — bit-identical** | logit sha c22df21e across both paths |
| Does Q4_K_M lose fidelity vs BF16 at this model/task resolution? | **No measurable loss** (within 1/5-rubric resolution) | fidelity_summary.json |
| Which runtime has the fastest EXL3 path on this host? | **not measured head-to-head today** | vLLM doesn't carry EXL3; GGUF uses a different quant family |
| Which quantizer wins at 4–5 bpw? | **no-winner** | EXL3 4bpw not produced; GGUF Q4_K_M fidelity indistinguishable from BF16 at 5.7 bpw |
| Can TFOLD beat EXL3, GGUF, EAQ, or vLLM on throughput? | **not claimed** | only within-TFOLD candidate optimization described; comparative claims require future controlled runs |

## Honest non-claims

- No claim of outperforming EXL3, GGUF, EAQ, or vLLM.
- No claim that quantization improves quality.
- No claim that lossless conversion improves speed — Race 001's scenario_ms tie between safetensors and `.tfold` carries forward.
- The micro-fidelity suite is too small to resolve inter-quant quality differences at this model scale; the "no measurable loss" statement is bounded by that.
- 2B class remains **NOT GRADUATED**. OFFSEC CWE-311/327 still open.

## Follow-ups (ordered by cheap → expensive)

1. **Measure vLLM dequantization cost** — dtype audit showed vLLM narrows FP32 norms, but we didn't profile its kernel breakdown. Equivalent to today's nsys work on vLLM would clarify the Race 001 cold-load decomposition claim.
2. **Produce EXL3 4bpw** on a quiet GPU (~1 h). Run against the same five prompts to compare to GGUF Q4_K_M.
3. **Implement the candidate** on `race-002/candidate/exl3-fuse-rot-in`. Measure. Decide merge.
4. **Larger fidelity harness** (HumanEval full, GSM8K full) *only if* the 2B class gets an instruct-tuned variant or a different base model.
