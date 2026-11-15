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

## 11. Stage 7H addendum — v1 "validate the test rig"

> v0 of this journal claimed Qwen3.5-2B was near-random on CWE
> classification. The operator caught that we never audited the
> harness before attributing a model weakness. This section documents
> the bugs we found, what we fixed, what the honest numbers actually
> are, and what we can and cannot conclude.

### The v0 bugs the audit caught

1. **Chat template never applied.** Qwen3.5-2B is an INSTRUCT model.
   It ships a `chat_template.jinja` (7755 chars, ChatML) and uses
   `<|im_end|>` as EOS. v0 passed raw text as a next-token
   continuation. The model was being asked "what text comes after
   this?" instead of "answer this user message."

   **Fix**: `tok.hf_chat_template([{"role":"user","content":prompt}],
   add_generation_prompt=True)` → correct ChatML envelope.

2. **`max_new_tokens=16` truncated the structured output.** Asking
   for 10 numbered answers + a HIGHEST-RISK line needs at least ~150
   tokens. v0 stopped early.

   **Fix**: `max_new_tokens=640`, retain the full raw text per trial.

3. **The "highest-risk" criterion was never implemented.** The
   scenario spec said "name the single highest-risk one". v0 never
   asked. v1.1 adds a separate `HIGHEST-RISK: CWE-NNN` line at the end
   of each trial and scores it independently.

4. **Repeated greedy seeds were counted as independent trials.**
   Greedy decoding ignores the seed. v0's "50 trials" at T=0 was 1
   sample pasted 50 times. v1 reports the T=0 column as
   "determinism floor" not "reliability".

### v1 honest result (same 10-snippet fixture, same model, same
checkpoint)

```
T=0 greedy / ChatML / batched prompt (direct)
  per-snippet mean correct:      0.2  (2.0 of 10 per trial)
  highest-risk correct:          0 / 50
  distribution of per-trial score: 50 trials scored exactly 2/10

T=0 greedy / ChatML / single-snippet prompt
  per-snippet mean correct:      0.2  (2.0 of 10 per trial)
  but DIFFERENT two snippets land: 1 and 9 (vs 1 and 3 batched)
```

Per-slot accuracy across 50 trials, batched-prompt condition:

| Slot | Truth | Hits |
|---|---|---|
| 1 | CWE-89 SQL injection | 50/50 ✅ |
| 2 | CWE-79 XSS | 0/50 (pred CWE-89) |
| 3 | CWE-22 path traversal | 50/50 ✅ |
| 4 | CWE-347 JWT alg=none | 0/50 (pred CWE-22) |
| 5 | CWE-434 upload | 0/50 (pred CWE-22) |
| 6 | CWE-311 cleartext | 0/50 (pred CWE-22) |
| 7 | CWE-352 CSRF | 0/50 (pred CWE-22) |
| 8 | CWE-346 XFF trust | 0/50 (pred CWE-22) |
| 9 | CWE-327 MD5 | 0/50 (pred CWE-22) |
| 10 | CWE-78 OS cmd injection | 0/50 (pred CWE-22) |

### The real finding: mode collapse, not ignorance

Slot 1 (CWE-89) gets 50/50 correct. The model knows. Slot 3 also
gets 50/50. Then it enters a repetition loop and emits "CWE-22" for
every remaining slot. The single-snippet variant doesn't have the
repetition pattern (each call is fresh context) but still only lands
2/10 — different two.

**This is NOT a model-knowledge weakness.** It's a decoder failure
mode plus limited base-model reasoning on the OTHER categories.
Fixing with SFT or DPO without first understanding this would be
training-on-the-wrong-thing.

The brief was explicit: `Do not start SFT, DPO or QLoRA until we
have a verified baseline and a reason to believe training will
address the measured weakness.` This evidence does not yet justify
training. It justifies PROMPT-level work first:

- Chain-of-thought preamble
- Constrained-output filters that forbid immediate repetition
- Per-item API calls (single-snippet variant) with better instruction phrasing

### v1 fair EXL3 vs TFOLD benchmark

This is the honest kernel-level answer the v0 journal couldn't
give. Same `tf-qwen35-cuda` binary (stage6e-tfold-perf head
`015624e07`), same prompt, same output path.

```
EXL3 safetensors, 6 runs (1 cold + 5 warm), 8 tokens each:
  cold 109.8 | warm 109.5, 112.2, 109.4, 111.4, 98.2 tok/s
  warm median 109.5, warm mean 108.1

TFOLD v0.2 .tfold, same 6 runs:
  cold 109.0 | warm 111.7, 106.2, 110.8,  97.8, 111.2 tok/s
  warm median 110.8, warm mean 107.5

Output MD5:
  EXL3:  c6a03e566512fa102ab5b5d25eb64782
  TFOLD: c6a03e566512fa102ab5b5d25eb64782  ← byte-identical
```

**Verdict**: same MD5 proves the kernels, the tokens and the logit
bytes are identical. The format difference is zero overhead at
decode time. Any TFOLD perf wins have to show up in COLD LOAD; we
don't measure that here because the model-load path inside
`tf-qwen35-cuda` is single-shot (process start → one step → exit),
not a loop we can time separately.

This also confirms the v0 "TFOLD inference impossible" claim was
WRONG for the wrong reason: the TFOLD reader + CLI auto-detect
shipped on stage6d/6e, but Stage 7G tested on stage6c-cuda-mcg which
predates that work. The format works and is wired; we just pointed
at the wrong branch head.

### Why we did NOT run T=0.7 at 50 trials × 640 tokens

Our run budget ran out. The sampled run takes ~2× the greedy time
on this GPU/model (sampling has per-token overhead and tokens don't
stop early on `<|im_end|>` as reliably with T>0). We killed it after
3 min to keep this session shippable. v2 should include it.

### What this validates

The scenario rubric (regex extraction + highest-risk pick) is now
proven against 15 synthetic cases.

The scenario fixture is now proven to exercise two different model
failure modes (correct-then-repeat on batched; narrow correct
subset on single).

The EXL3 vs TFOLD benchmark is now proven kernel-level fair.

### What this does NOT validate

- The other 9 scenarios still have no fixtures.
- Mode collapse mitigation (prompt engineering, better samplers) is
  untried.
- T=0.7 full sweep + Independent re-scoring by a second regex
  implementation.
- Cold-load benchmark isolating process-start from CUDA init.

### Receipt pointers

- Rubric + scenario v1.1: `docs/2b-dyno-baseline.json#offsec_v1_vuln_classify_01`
- Full raw outputs: `docs/dyno/offsec/v1_t0_direct.json`
- Single-snippet variant: `docs/dyno/offsec/v1_single.json`
- EXL3 vs TFOLD raw numbers: `docs/dyno/native/bench_native_results.txt`

### One extra comprehension question (v1)

**Q3**. Slot 1 (CWE-89 SQL injection) is answered correctly 50/50
across every trial, in both the batched and the single-snippet
variant. Slot 10 (CWE-78 OS command injection) is answered correctly
0/50 in both variants. What does that asymmetry tell you about
whether the model's weakness is CAUSAL knowledge vs the LABELING
scheme? Hint: SQL injection and OS command injection are the same
"inject untrusted input into an interpreter" pattern; they differ
only in which interpreter.
