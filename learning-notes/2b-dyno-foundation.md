# 2B Chop Shop Dyno — Foundation journal

> First-ever reproducible benchmark + evaluation pass for Qwen3.5-2B on
> DGX Spark. Covers what we measured, why it was measured that way, what
> we did NOT measure (and why), and the honest shape of the gap between
> "a baseline exists" and "the championship is decided".

## 1. Why a dyno at all?

A dyno is the test rig a tuner runs a car on before touching the engine.
You don't trust a lap time if the dyno's baseline isn't recorded first.
For a model family we need the same:

- One reference checkpoint with its hash pinned.
- One runtime whose version is frozen.
- One hardware target whose GPU, driver and compile capability are
  recorded.
- One set of knobs (prompt, sampler, max tokens, context, seeds) that
  everything else is held constant against.

Without those four, a "speedup" is just a sentence. With them, it's a
verifiable delta.

## 2. What we measured — EXL3 via exllamav3 1.6.7

- Model: `UnstableLlama/Qwen3.5-2B-exl3-8.00bpw`
- SHA-256: `a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f`
- Hardware: DGX Spark, `NVIDIA GB10` sm_121, torch 2.11.0+cu130
- Runtime: `exllamav3==1.6.7` from PyPI
- Prompt: 15 tokens ("You are a helpful assistant. List three security best practices for password hashing.")
- Context: 4096, max_new_tokens=128, GreedySampler, seed=1234

One cold run plus three warm runs on the same prompt. The warm runs
converge to a stable ~110 tok/s decode and ~17 ms TTFT, which is the
baseline we'll quote against future runs.

| Phase | Measurement |
|---|---|
| Model VRAM after load | **2208 MB** |
| Peak VRAM (with KV cache) | **2247 MB** |
| Cold load (model + cache + alloc) | 1.94 s |
| TTFT, first (cold) generation | 77.9 ms |
| TTFT, warm (steady) | 17.4 – 17.5 ms |
| Decode tokens/s, cold | 107.95 |
| Decode tokens/s, warm (median) | 109.82 |
| Warm Cache reinit | 0.55 ms |

The decode-tps variance across three warm runs is 0.9 % — small enough
to detect meaningful kernel-level improvements without needing a huge
trial count.

## 3. What we did NOT measure, and why

**Native `.tfold` v0.2 end-to-end inference.** The TFOLD v0.2 reader
exists and is exercised by `test-checkpoint-io-host` + `test-exl3-cuda-host`
(1006 tests passing on Spark). But there is no user-facing inference
CLI that takes a `.tfold` file and generates tokens — the current
`tf-qwen35-cuda` still refuses with `error: NoSafetensors` when the
model directory contains only `model.tfold`. Wiring the Checkpoint
auto-detection through the inference CLI is out of PR #3412 scope
and is on the roadmap as its own PR.

For comparison we measured the raw file I/O throughput of the
`.tfold` file (~15.2 GB/s on page-cached read; the file is 138,338
bytes larger than the safetensors, accounted for by the TFOLD v0.2
header). That's load-side overhead, NOT inference throughput. It will
set an upper bound on cold-load wins, not a decoder-side speedup.

**ExLlamaV2 or llama.cpp.** The brief lists them as "where identical
weights and comparable settings are available". We don't have a
matching mcg-codebook EXL3 reader in llama.cpp (none exists at the
time of writing), and ExLlamaV2 doesn't read EXL3 format. The honest
answer is "no shared-weight, shared-sampling comparison point today".

**Unquantized reference.** We have no unquantized bf16 Qwen3.5-2B
checkpoint locally. A full ~5 GB download would be needed; outside
this session's budget. Quality comparisons against unquantized stay
a Dyno v1 target.

## 4. Why "deterministic decoding" doesn't measure reliability

The operator brief warned explicitly: `Do not claim that repeated
deterministic decoding with unchanged settings measures reasoning
reliability.` It's an important trap.

If you fix seed, sampler, and prompt, then run a generation 50 times,
you get the same tokens 50 times. The number "50/50 passed" tells you
exactly nothing except that the GPU's floating-point reduction is
deterministic. It is NOT a reliability signal.

Reliability requires VARIATION on at least one axis:

- varied seeds under temperature > 0, OR
- paraphrased prompts for the same question, OR
- independently generated input variants, OR
- different sampling policies.

For the OffSec Dyno v0 we measured two conditions for scenario
`vuln_classify_01_cwe`:

- **Temperature = 0 greedy**: 50 trials, all seeds different, all
  outputs IDENTICAL by construction. This number is the "best case
  under this sampler" — a correctness floor.
- **Temperature = 0.7 DefaultSampler**: 50 trials, each with a
  different seed, outputs genuinely vary. This number is a
  decoding-variance signal under a realistic sampler.

Only the second one is a reliability measurement. Both appear in the
receipt so a future reader can see the split.

## 5. The OffSec Dyno scaffold

Ten scenarios drafted across the five categories the brief named.
See `docs/offsec-dyno-v0-scenarios.json` for the full list; summary:

| # | Scenario id | Category |
|---|---|---|
| 1 | `ad_attack_path_01_easy` | AD attack-path reasoning |
| 2 | `ad_attack_path_02_medium` | AD attack-path reasoning |
| 3 | `vuln_classify_01_cwe` | Vulnerability classification (IMPLEMENTED) |
| 4 | `vuln_classify_02_severity_dispute` | Vulnerability classification |
| 5 | `authz_01_rbac_bypass` | Web/API authorization |
| 6 | `authz_02_idor_stacked` | Web/API authorization |
| 7 | `codegen_01_sql_param` | Security codegen |
| 8 | `codegen_02_jwt_verify` | Security codegen |
| 9 | `toolcall_01_single_shell` | Tool-call correctness |
| 10 | `toolcall_02_evidence_report` | Evidence-based reporting |

**Scenarios 2-10 have no fixture or rubric script yet.** Writing
them + the scoring rubrics is Dyno v1 scope. Reporting 10 × 50 trials
of empty prompts into the model would be fabrication.

**Scenario 3 is fully executed.** 10 CWE-25 snippets with pinned
ground truth, exact-match rubric on `CWE-NNN` regex extraction, 50
trials each at T=0 and T=0.7. Results live in
`docs/dyno/offsec/vuln_classify_01_t0.json` and
`docs/dyno/offsec/vuln_classify_01_t07.json`.

## 6. Comparing different quantizations

The PR #3412 review (Synthena) hammered a principle that transfers
straight to this dyno: **never compare different quantizations' outputs
as if they should match**. EXL3-mul1 and EXL3-mcg use different
trellis codebooks; their decoded weights differ by codebook
construction, not by bug. Every quantization gets compared only to
its own reference run with the same sampler + seed, never against a
sibling quant's run.

For the EXL3 2B above this means: our 110 tok/s decode is the
EXL3-mcg baseline. A future TFOLD-native inference baseline on the
same checkpoint IS a defensible comparison, because the quantization
is identical and only the loader + kernel dispatch change. A future
llama.cpp baseline on an entirely different quantization (Q4_K_M,
Q5_0, etc) is NOT, because the weights differ.

## 7. What changed, why

| Change | Why |
|---|---|
| Added `docs/dyno/perf-exl3.json` | Pin baseline measurements before touching the stack |
| Added `docs/dyno/offsec/vuln_classify_01_*.json` | One scenario executed to prove the pipeline, not ten faked |
| Added `docs/offsec-dyno-v0-scenarios.json` | Scenario catalog with explicit `deferred_items` list |
| Created Spark worktree `~/worktrees/pr3412` | Validate PR head independently from main checkout |
| Installed `~/opt/exllamav3-venv` | Reference runtime; pinned version 1.6.7 |

## 8. Alternatives considered

| Option | Reason rejected |
|---|---|
| Run all 10 × 50 trials with the DESCRIPTIONS as prompts | Would be fabricating scores against prompts the rubric has no answer key for |
| Skip the scenario work entirely; just report perf numbers | Brief explicitly required the OffSec scaffold |
| Run all 10 scenarios with only 5 trials each | Fewer trials, still invalid without rubrics |
| Measure llama.cpp at the same decode length | No mcg-codebook llama.cpp support exists; would be measuring a different model |
| Compare TFOLD-native tokens/s against exllamav3 directly | No TFOLD inference CLI exists yet; would be apples-to-oranges |

## 9. Hands-on exercise

Reproduce the EXL3 baseline on your own hardware:

```bash
# 1. install
python3 -m venv ~/opt/exlv3 --system-site-packages
~/opt/exlv3/bin/pip install exllamav3

# 2. verify
~/opt/exlv3/bin/python -c "import exllamav3, torch; print(torch.cuda.is_available())"
# True

# 3. get the pinned checkpoint
hf download UnstableLlama/Qwen3.5-2B-exl3-8.00bpw --local-dir ./q35-2b-exl3

# 4. verify SHA
sha256sum q35-2b-exl3/model.safetensors
# a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f  ...

# 5. clone + run the perf harness
scp spark:/tmp/perf_bench.py .
~/opt/exlv3/bin/python perf_bench.py exl3_qwen35_2b > my-baseline.json
```

**Expected shape**:

- `decode_tps` steady-state within ~15 % of 110 depending on your GPU
  (Spark GB10 is sm_121; a 3090 Ti or 4090 should go faster, a laptop
  RTX up to several × slower).
- `t_cold_load_s` roughly proportional to disk bandwidth and PCIe
  width.
- `gpu_mem_after_load_mb` always 2200 ± 20 MB — the model VRAM is
  determined by the checkpoint, not your GPU.

If your decode_tps is >10 × off from 110 either way, first suspect a
runtime version mismatch, THEN kernel compile capability
(`torch.cuda.get_device_capability()`), THEN check that the GPU
isn't being shared with something else.

## 10. Comprehension questions (left open on purpose)

**Q1**. Our cold-run decode tps is 107.95, warm-run median is 109.82,
a +1.7 % gap. Why does the first generation decode slower, and what
single CUDA call would you expect to dominate the delta? Hint: it's
NOT the GPU clocks warming up.

**Q2**. We measured both `temperature=0` (greedy) and `temperature=0.7`
(sampled) over 50 trials with different seeds for the vuln classify
scenario. Imagine a future scenario where the temperature=0 condition
reports 42/50 trials perfect and the temperature=0.7 condition
reports 38/50 trials perfect. What does the gap (42 − 38 = 4) tell
you about the MODEL vs the gap (50 − 42 = 8) between perfect and
greedy? Which gap is a safer regression gate and why? Hint: think
about what each number varies.
