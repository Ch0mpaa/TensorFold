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

## 12. Stage 7I — Dyno v2 "finish one quality benchmark"

> v1 reported 2/10 per trial and called it "mode collapse". v2 proved
> that wasn't true. The actual pattern was EOS not being honored, the
> Qwen3 `<think></think>` thinking tokens eating generation budget,
> and a small base model that is strong on some categories and
> genuinely blind to others. Once we built a reliable trial runner
> and fixed the stop condition, the honest accuracy for the best
> condition rose to **26.2 % (95 % CI 22.5–30.2)** and the per-category
> picture became actionable.

### What was wrong with v1 that v2 fixed

1. **`stop_conditions` were never passed to `Job`.** Every raw output
   in v1 contained `<|im_end|><|endoftext|><|im_start|>user...` —
   the model emitted end-of-turn but the generator just kept going,
   because exllamav3's `Job` does not auto-populate stop conditions
   from the tokenizer's eos token. The "stop_reason" field was always
   `null`, which should have been the first clue. v2 passes
   `stop_conditions=[tok.single_id("<|im_end|>"), tok.single_id("<|endoftext|>")]`.

2. **Qwen3's `<think></think>` tokens ate generation budget.** Qwen3
   adds a "think" block before answering. v0 budgeted 16 new tokens
   total; v1 bumped to 640 but still mixed thinking tokens with the
   answer. v2 budgets 300 new tokens and respects EOS, so the
   generation ends naturally at `<|im_end|>` ~30-150 tokens in.

3. **Shared Generator / Shared KV state hypothesis** — DISPROVEN.
   Running a FRESH `Generator + Cache` per trial gave byte-identical
   output to the shared path. The PagedKVCache correctly isolates
   per-job prompt pages. Writing this off conclusively saves us from
   spending future debug cycles there.

### The reliable trial runner (G2)

```
docs/dyno/dyno_v2_runner.py
```

- **JSONL receipts**, one line per completed trial, flushed on every
  write. Crash-safe.
- **Resume** reads the existing JSONL and skips `(condition, seed)`
  tuples whose status is `completed`.
- **Per-trial 30 s timeout + 300-new-token budget**, with distinct
  statuses: `completed`, `timeout`, `truncation`, `failure`.
  Interrupted trials never claim `completed`.
- **Full retention**: `prompt_text`, `output_text`, `seed`,
  `condition`, `truths`, `elapsed_s`, `stop_reason`.
- Four conditions covered: `zero_batched`, `few_batched`,
  `zero_single`, `few_single`.

### CWE evaluation — final numbers

Dev/held-out split: dev = {SQLi, XSS, path traversal} (prompts were
only tuned against these); held-out = the other 7 snippets. We
report overall accuracy over the full 10-set.

50 trials × 4 conditions × (10 for single or 1 for batched) = 1 100
actual model invocations. Every invocation `completed`. All
outputs are in `docs/dyno/offsec/v2/*.jsonl`.

| Condition | Correct / Total | Accuracy | 95 % Wilson CI |
|---|---|---|---|
| zero_batched | 53 / 500 | 10.6 % | 8.2 – 13.6 |
| few_batched  | 97 / 500 | 19.4 % | 16.2 – 23.1 |
| zero_single  | 72 / 500 | 14.4 % | 11.6 – 17.7 |
| **few_single** | **131 / 500** | **26.2 %** | **22.5 – 30.2** |

Highest-risk CWE-78 selection (batched only):

| Condition | Correct | 95 % CI |
|---|---|---|
| zero_batched | 1 / 50 (2 %) | 0.4 – 10.5 |
| few_batched  | 0 / 50 (0 %) | 0.0 – 7.1 |

### Per-category accuracy (few_single, the winner)

| CWE | Category | Accuracy | What it means |
|---|---|---|---|
| **CWE-89**  | SQL injection | **94 %** | Model knows SQL interpolation is dangerous |
| **CWE-352** | Shared CSRF token | **72 %** | Few-shot preamble unlocks CSRF reasoning |
| **CWE-327** | MD5 password hash | **56 %** | Few-shot unlocks weak-crypto class |
| CWE-79  | XSS reflection | 20 % | Partial — model confuses with other injection types |
| CWE-78  | OS command injection | 18 % | Model tends to mis-classify as SQLi |
| CWE-22  | Path traversal | 2 % (↓ from 16 % zero-shot) | Few-shot's `eval()` example appears to distract |
| CWE-311 | Password cleartext email | 0 % | Model doesn't flag disclosure patterns |
| CWE-347 | JWT alg=none | 0 % | Zero coverage of auth-protocol attacks |
| CWE-346 | Trust X-Forwarded-For | 0 % | Zero coverage of origin-spoofing attacks |
| CWE-434 | Unrestricted file upload | 0 % | Zero coverage of file-write attacks |

### What the surgical pattern means

The base model's knowledge of CWE-25 is **concentrated**, not
uniform. SQL injection, path traversal (zero-shot), CSRF and weak
hashing (with few-shot preamble) are well-represented. Auth-protocol
flaws, server-origin trust and file-upload semantics are flat zero.

That's the shape that would guide a future SFT program if we ever
approved one: the training signal would target the 0 % categories,
not the 94 % ones. **We do NOT approve SFT today; the brief forbade
it, and the honest reading of this evidence is that prompt
engineering has not yet been exhausted.** A retrieval-augmented
approach (pull the CWE-25 definition into the prompt) could lift
the zero rows without any training at all.

### Performance baseline — G4 clarification

The MD5 match between EXL3 safetensors and .tfold v0.2 output bin
files was called "byte-identical logit bytes" in Stage 7H. Gate 4
asked us to verify. The audit:

- `zig/tests/qwen35_cuda_main.zig` `step` handler writes
  `std.mem.sliceAsBytes(cache.logits)` per step.
- `cache.logits` is `[]u16` (bf16 logits over vocab) per
  `zig/src/families/qwen3_5/cuda_cache.zig:139`.
- Each write is raw bf16 logit bytes from the engine's final-norm
  row.

**Conclusion**: MD5 equality IS logit equality at the bf16 bit
level. The Stage 7H claim stands. No correction to the preserved
EXL3-vs-TFOLD numbers. Per-brief: no new optimization work in v2.

### Hands-on exercise (reproduce the fix)

```python
# BAD (v1): generation runs past <|im_end|> into fake turns
job = Job(input_ids=templated, max_new_tokens=640,
          sampler=GreedySampler(), seed=1234)

# GOOD (v2): respect the ChatML end-of-turn token
eos_ids = [tok.single_id("<|im_end|>"), tok.single_id("<|endoftext|>")]
job = Job(input_ids=templated, max_new_tokens=300,
          sampler=GreedySampler(), seed=1234,
          stop_conditions=eos_ids)
```

Run both forms against Qwen3.5-2B with an instruction prompt. v1
output will contain `<|endoftext|><|im_start|>user` repeated; v2
will end cleanly at the first `<|im_end|>`. Compare `stop_reason`
in the `finished` event — v1 is `null`, v2 is a token id.

### Comprehension questions

**Q4** (v2). The few-shot preamble (3 examples: `eval()` → CWE-95,
cookie-missing-HttpOnly → CWE-1004, divide-by-zero → CWE-369) raises
CSRF accuracy from 12 % to 72 % and MD5 accuracy from 6 % to 56 %,
but drops path-traversal accuracy from 16 % to 2 %. Why does the
SAME preamble help some categories and hurt one? Hint: look at
what's between `eval()` and path traversal in terms of what the
model "sees" as the mental model. Also hint: the preamble never
mentioned path traversal.

**Q5** (v2). Our highest-risk accuracy is 0-2 %. The model picks the
same WRONG highest-risk CWE across every trial. Separately, our
per-category accuracy for CWE-78 (OS command injection, the true
highest risk) is 18 % in few_single. What relationship do you expect
between "correctly classifies the OS cmd injection snippet AT ALL"
and "correctly picks it as highest-risk"? Write it as a conditional
probability inequality, then propose ONE prompt change that would
let the two probabilities agree with each other. Hint: a selection
task is not the same as a classification task.

## 13. Stage 7J — Dyno v2.1 "generalization audit"

> v2 declared `few_single` the winner at **26.2 % (CI 22.5-30.2)**.
> v2.1 audits that number. Short version: the 26.2 was dev-inflated,
> real held-out accuracy is 20.9 %, and three of the four conditions
> are statistically at chance (p > 0.3 against the 4 % Top-25
> baseline). The operator was right to ask.

### What v2 got right vs what v2.1 corrects

| Claim | v2 said | v2.1 says |
|---|---|---|
| Headline accuracy | 26.2 % (CI 22.5-30.2) | **held-out 20.9 %** (CI 16.9-25.4 Wilson, but 2.6-44.0 BOOTSTRAP over 7 items) |
| Scorer reliability | one regex scorer, unit-tested | TWO scorers, **100 % agreement** on 2,100 extractions |
| Three other conditions | "beats chance" implied | NONE beats chance on held-out (binomial p > 0.3) |
| Winner significance | implied by CI non-overlap | McNemar p < 10^-11 on paired (seed, snippet) data |

### G1 — Dev vs held-out split

DEV snippets (used while picking prompts):
`CWE-89` SQL injection · `CWE-79` XSS · `CWE-22` path traversal

HELD-OUT snippets (scored once, prompts never saw them):
`CWE-347` JWT alg=none · `CWE-434` upload · `CWE-311` cleartext · `CWE-352` CSRF · `CWE-346` XFF trust · `CWE-327` MD5 · `CWE-78` OS cmd injection

Trial-level Wilson 95 % CIs (hides item-level uncertainty — see G3):

| Condition | DEV k/n (acc) | HELD-OUT k/n (acc) |
|---|---|---|
| zero_batched | 42/150 (**28.0 %**) | 11/350 (**3.1 %**) |
| few_batched  | 88/150 (**58.7 %**) |  9/350 (**2.6 %**) |
| zero_single  | 56/150 (**37.3 %**) | 16/350 (**4.6 %**) |
| few_single   | 58/150 (**38.7 %**) | 73/350 (**20.9 %**) |

The dev-vs-held-out gap for `few_batched` is **56 percentage points**
(58.7 → 2.6). That's the shape of textbook prompt overfitting. The
few-shot preamble taught the model to recognize the three dev
categories; it generalized nowhere else when the questions were
batched.

### G2 — Independent scorer

`docs/dyno/scorers/scorer_v2b.py` is a pure str-find + manual digit
scan with no regex. Its 20-case unit-test battery was designed
independently of scorer_v2's 15 regex-focused cases. Running both
scorers over all 1,100 trials:

```
batched per-slot agreement:     1000/1000 = 100.00 %
batched highest-risk agreement:  100/100 = 100.00 %
single pred agreement:          1000/1000 = 100.00 %
total disagreements:            0
```

The regex scorer and the no-regex scorer extract the same identifier
from every trial. v2's metrics are not scorer artifacts; they're
accurate counts over a confounded split. The fix was never
"different scorer" — it was "report the split".

### G3 — Statistical correctness

**Random-chance check on held-out** (vs 4 % CWE-25 uniform prior,
binomial one-sided):

| Condition | Held-out acc | Binomial p vs 4 % chance | Beats chance? |
|---|---|---|---|
| zero_batched |  3.1 % | 0.83 | **NO** |
| few_batched  |  2.6 % | 0.93 | **NO** |
| zero_single  |  4.6 % | 0.34 | **NO** |
| few_single   | 20.9 % | **< 10^-6** | YES |

Three of the four conditions cannot be distinguished from a model
that picks uniformly from the 25 top CWEs. Only `few_single` beats
that bar.

**Paired McNemar on matched (seed, snippet) pairs, held-out:**

```
zero_batched vs few_batched : chi2=0.06   p=0.80     NO DIFFERENCE
zero_batched vs zero_single : chi2=0.64   p=0.42     NO DIFFERENCE
zero_batched vs few_single  : chi2=50.28  p=1.3e-12  significant
few_batched  vs zero_single : chi2=1.44   p=0.23     NO DIFFERENCE
few_batched  vs few_single  : chi2=55.12  p=1.1e-13  significant
zero_single  vs few_single  : chi2=46.81  p=7.8e-12  significant
```

Only `few_single` is significantly better than the rest. The three
non-winners are pairwise indistinguishable.

**Item-level bootstrap** (n=10,000 resamples of the 7 held-out
snippet accuracies):

| Condition | Held-out snippet-mean | Bootstrap 95 % CI |
|---|---|---|
| zero_batched |  3.1 % | 0.3 – 7.1 |
| few_batched  |  2.6 % | 0.9 – 4.6 |
| zero_single  |  4.6 % | 1.1 – 8.6 |
| few_single   | 20.9 % | **2.6 – 44.0** |

The trial-level Wilson CI said few_single was 16.9-25.4; the
snippet-level bootstrap says 2.6-44.0. **The difference matters
and we must report both.** The Wilson CI treats 50 responses to the
same snippet as 50 independent events, which they are not. A single
snippet whose answer shifts up or down swings the mean dramatically.

### G4 — Zero-category probes (dev-only)

We tried three alternative prompt strategies on the four lowest
categories using NEW dev-only paraphrases — the seven held-out
snippets stayed untouched. Prompts:

1. **Minimal**: bare "classify this" instruction.
2. **Textbook preamble**: four CWE definitions pasted in before the
   question.
3. **Wiki examples**: five "known vulnerability → CWE" pairs.

Results on **1,800 dev-only trials**:

| True CWE | minimal | textbook | wiki |
|---|---|---|---|
| CWE-346 (XFF trust) |  0 % | **100 %** |  0 % |
| CWE-434 (upload)    |  0 % | **94 %**  |  0 % |
| CWE-311 (cleartext) |  0 % | **90 %**  |  0 % |
| CWE-347 (JWT sig)   |  2 % |   22 %    |  2 % |

This is the big correction on v2's story. In v2 I wrote "model
lacks knowledge of these CWEs" because zero-shot said so. The
textbook condition — which just pastes the four CWE definitions
into the prompt before the question — shows the 2B model has the
knowledge. It just doesn't retrieve the correct CWE identifier
without being told what the candidate set is.

Three readings matter:
1. 0 % on CWE-346/434/311 in v2 was a **prompt failure**, not a
   model failure. SFT or RAG is a legitimate next step but not
   the ONLY next step; a longer system prompt would also work.
2. CWE-347 still only reaches 22 % even WITH the definition. The
   model recognizes "JWT" but often picks CWE-20 or CWE-287. This
   is the one true capability gap, and it's specifically around
   cryptographic verification reasoning.
3. The "wiki" style (giving five unrelated CWE examples) gave
   0 % everywhere — a model given a list of 5 examples will not
   pick the 6th CWE unless that CWE is in the list. This is
   measurable prompt recency / copying bias.

None of these dev-probe numbers change the held-out verdict
above. v2.1 held-out `few_single` stays at 20.9 %. We did NOT run
textbook-style prompts on held-out (that would require generating
new held-out snippets — out of scope today).

### Hands-on exercise

Recompute both the Wilson CI and the snippet-level bootstrap CI for
`few_single` held-out, from scratch, using the checked-in JSONL:

```bash
python3 docs/dyno/scorers/analyze_v2_1.py
python3 docs/dyno/scorers/paired_mcnemar.py
```

Change the `heldout_snippets` list in `scorer_v2b.py` to any 7 of
the 10 (not the current held-out set). Observe how much the
snippet-level bootstrap CI shifts. That shift is how fragile a
7-snippet benchmark really is.

### Comprehension questions

**Q6** (v2.1). The Wilson 95 % CI for `few_single` on held-out is
16.9 – 25.4 (width ≈ 8 pp). The bootstrap 95 % CI over snippets is
2.6 – 44.0 (width ≈ 41 pp). Both are "95 % confidence intervals".
Why is the second one 5× wider, and which one is what you report to
a stakeholder who asks "will this work on a new vulnerability we
haven't tested?" Hint: write down what each CI is sampling over.

**Q7** (v2.1). We ran McNemar on paired trials and McNemar on
bootstrap items; one gave p < 10^-11, the other gave CI width 41 pp.
An outside reviewer argues we are "p-hacking" by reporting the
significant result. Describe in 2-3 sentences the one honest way to
report BOTH results in the same table, and why NOT doing so would
inflate the claim of the few_single win. Hint: this is why
pre-registration exists.

## 14. Stage 7K — Fresh held-out generalization test

> Stage 7J left us with a brittle estimate: 20.9% held-out accuracy
> but a bootstrap CI (2.6-44.0) so wide it told us almost nothing
> about a real population of vulnerability examples. Stage 7K runs
> the same frozen prompt on **30 genuinely new** snippets to see
> whether the number survives.

### What we froze BEFORE running the model

`docs/dyno/offsec/stage7k/freeze.json` lists:
- Model repo + SHA-256 of the safetensors file.
- The `few_single` prompt (the only v2.1 condition that beat chance).
- Sampler constructors (`GreedySampler()` for deterministic;
  `DefaultSampler()` for sampled — EXACTLY matching the v2 runner).
- Seeds: `range(5000, 5050)`, non-overlapping with every prior stage.
- 30 s timeout, 100 new-token cap, `scorer_v2b` as the primary scorer.

The freeze was committed to disk before `stage7k_runner.py` ran. No
model output for any Stage 7K snippet was inspected during prompt
design. That's the point of freezing.

### Headline numbers

**Sampled (1,500 trials, 50 seeds × 30 snippets)**

| Metric | Value |
|---|---|
| Trial-level | 380 / 1,500 = **25.3%** (Wilson 23.2 - 27.6) |
| **Item-level (bootstrap over 30)** | **25.3% (CI 13.1 - 38.9)** |
| Macro over 10 CWEs | 25.3% (bootstrap 7.9 - 46.9) |
| Malformed outputs | 0 (frozen prompt always emits `CWE-NNN`) |
| Wrong-CWE | 1,120 / 1,500 (74.7%) |

**Deterministic (greedy, 30 trials)**: 9 / 30 = 30.0% (Wilson 16.7-47.9).
Greedy and sampled agree closely — the frozen few-shot prompt
produces a very peaked distribution.

### Per-CWE breakdown (sampled)

| CWE | correct / 150 | acc |
|---|---|---|
| CWE-89  (SQL)         | 148 / 150 | **98.7%** |
| CWE-79  (XSS)         |  77 / 150 | 51.3% |
| CWE-22  (path)        |  76 / 150 | 50.7% |
| CWE-352 (CSRF)        |  41 / 150 | 27.3% |
| CWE-78  (OS cmd)      |  34 / 150 | 22.7% |
| CWE-327 (weak crypto) |   3 / 150 |  2.0% |
| CWE-347 (JWT sig)     |   1 / 150 |  0.7% |
| CWE-311 (cleartext)   |   0 / 150 | **0.0%** |
| CWE-346 (origin)      |   0 / 150 | **0.0%** |
| CWE-434 (upload)      |   0 / 150 | **0.0%** |

Per-example results are **bimodal**: 12 snippets at ≥40/50, 13 at
0/50. The overall mean is driven by a few examples that the model
gets almost ALWAYS right, not by partial competence everywhere.

### Did Stage 7K beat Stage 7J?

Stage 7J: 20.9% [bootstrap 2.6 - 44.0]
Stage 7K: 25.3% [bootstrap 13.1 - 38.9]

**No.** Stage 7J's point estimate sits well inside Stage 7K's bootstrap
interval, and the two intervals overlap by 26 percentage points. The
4.4-point jump is NOT statistically distinguishable from no change.
This is exactly the discipline the brief demanded: "Do not claim
statistical improvement merely because a point estimate increased."

What Stage 7K actually gives us:
- The 20-25% range **reproduces** on a bigger, deliberately-different
  pool of examples. Stage 7J's number was not a 7-snippet fluke.
- Item-level CI **tightens** from 41pp to 26pp — better evidence,
  same score.
- The same four categories (311, 346, 434, 347) stay broken.

### Why the frozen prompt did NOT fix the zero categories

g4 (Stage 7J) showed that pasting the four CWE definitions into the
prompt lifts these categories from 0% to 90-100% on dev snippets.
The Stage 7K frozen prompt is the `few_single` prompt (examples for
CWE-89, 79, 22 only) — it does NOT contain the catalog of
definitions for 311 / 346 / 434. So the model has no way to
retrieve those CWE IDs. The 0% outcome in Stage 7K was predictable
from the g4 finding and is a direct consequence of the frozen
prompt choice.

Takeaway: the model has the SEMANTIC knowledge (g4 proved it) but
not the ID-recall capability under a non-catalog prompt. This is a
retrieval gap, not a model capability gap.

### Hands-on exercise

Open `docs/dyno/offsec/stage7k/sampled.jsonl` and read any 20 lines
for `snippet_id: s7k16` (`CWE-311`, "cookie without Secure or
HttpOnly"). What CWE does the model pick INSTEAD? Does it change
across seeds? Do the model's wrong answers cluster around related
categories (CWE-614, CWE-1004, CWE-384) or are they scattered?
Write a one-paragraph summary.

### Comprehension questions

**Q8** (v2.1 → 7K). The Stage 7J Wilson CI for few_single held-out
was 16.9-25.4 (width 8pp). The Stage 7K Wilson CI is 23.2-27.6
(width 4pp). The Wilson CI got TIGHTER — great news, right? Why is
this comparison misleading, and which TWO CIs would you actually
compare to show whether anything changed? Hint: think about what
each Wilson interval is sampling over.

**Q9** (7K). We found that 13 of the 30 fresh snippets score 0/50
and 12 score ≥40/50. Call the overall accuracy `a` and the number of
ALWAYS-correct snippets `k`. If the model were "mediocre" (same `a`
but partial competence spread evenly), what would the per-snippet
accuracy distribution look like? Which one of the two patterns would
SFT most easily move — and why? Hint: look at logistic regression
against class prevalence.

## 15. Stage 7L — CWE catalog A/B/control experiment

> Stage 7K named "catalog-in-prompt" as the single next graduation
> blocker but did not test it. Stage 7L runs the controlled
> experiment: baseline (A) vs catalog in prompt (B) vs
> length-matched neutral text in prompt (C). The control is the
> whole point — it distinguishes "the CWE definitions helped" from
> "a longer prompt helped".

### Why we needed a control

A common reviewer trap: paste more text in the prompt, see accuracy
go up, declare the extra text helped. The hidden alternative
explanation is that **any** longer prompt helps (via register,
calibration, or attention-pattern reasons unrelated to content). A
length-matched neutral text controls for that.

- **A**: Stage 7K's `few_single` prompt (CWE-89 / 79 / 22 examples).
- **B**: A + 435 words of MITRE CWE definitions for all 10 evaluated
  CWEs (including CWE-346 and CWE-311, which are NOT in the Top-25).
- **C**: A + 441 words of CPython asyncio reference text,
  length-matched to B, deliberately containing zero security
  vocabulary.

### Headline results

Sampled (1,500 trials / condition):

| Cond | k/n | Trial Wilson | Item bootstrap |
|---|---|---|---|
| **A** baseline | 357/1,500 (**23.8%**) | 21.7-26.0 | 11.8-36.9 |
| **B** +catalog | **1,373/1,500 (91.5%)** | 90.0-92.8 | **83.8-97.4** |
| **C** +control | 370/1,500 (24.7%) | 22.6-26.9 | 12.6-37.9 |

Deterministic greedy: A=8/30, **B=29/30**, C=6/30.

Paired McNemar on matched (snippet, seed):
```
A vs B  chi2=868.66  p = 6.4e-191   (catalog massively helps)
B vs C  chi2=879.93  p = 2.3e-193   (catalog massively beats control)
A vs C  chi2=  1.12  p = 0.29       (control = baseline)
```

Per-CWE highlights (sampled, 150 trials each):
- Zero → full: CWE-311 (0.7 → 100), CWE-346 (0 → 100), CWE-434 (0
  → 100), CWE-347 (0 → 96.7), CWE-352 (0.7 → 98), CWE-327 (6 → 100).
- Mild gain: CWE-78 (16 → 100), CWE-79 (56 → 98), CWE-22 (60.7 → 63.3).
- **Regression**: CWE-89 (SQL) **98 → 59.3** (-38.7pp). The one
  category A got right now gets confused with CWE-78.

Per-example reliability bands (of 30 snippets):
- A: 2 × 50/50, 14 mid, 14 × 0/50.
- B: **20 × 50/50**, 10 mid, **0 × 0/50**. Nothing is a hard zero anymore.
- C: 2 × 50/50, 14 mid, 14 × 0/50 — identical shape to A.

Prompt text lengths: A ≈ 500 chars (~130 tokens), B ≈ 3,500 chars
(~920), C ≈ 3,600 chars (~940). Cost: tokens-per-correct-classification
drops from A=26.4 to **B=7.3** — 3.6× cheaper per correct answer
despite 7× longer prompt.

### What the comparison tells us

The brief's two hypotheses:
1. "Catalog content helps because the model needed the definitions."
2. "Any long prompt helps (register / calibration / pattern)."

Stage 7L rules out (2). Condition C is statistically
indistinguishable from A (p=0.29). B overwhelmingly beats both.
The signal is in the CWE definitions themselves, not in the extra
length. That's what a length-matched control is FOR.

Stage 7L does NOT tell us whether a shorter catalog would suffice
(we tested one length, not a dose-response curve), nor whether the
CWE-89 regression could be fixed by rewording to reduce the
"constructs all or part of ... command" overlap with CWE-78. Both
are in-scope for a future stage.
### Hands-on exercise

Pick any one zero-category snippet from Stage 7L sampled receipts.
Read all 50 seeds under each of A, B, C. Record:
1. The five most common wrong CWEs the model picks in each condition.
2. Whether the wrong CWEs under B are closer to the truth category
   semantically (CWE-311 near CWE-319 / CWE-1004 / CWE-614) than
   the wrong CWEs under A.
3. Whether C's wrong answers look more like A's or more like B's.

Write 3-5 sentences on what this tells you about where the catalog's
signal is coming from.

### Comprehension questions

**Q10** (7L). The brief required a length-matched control. Suppose
C's accuracy came out identical to B's on the held-out 30 snippets.
What follow-up experiment would you design, and what hypothesis
would it test? Hint: there are at least two ways to vary the control.

**Q11** (7L). Prompt-token overhead grew from ~X in A to ~Y in B. For
a production CWE classifier that must stay under a fixed per-request
latency SLO, derive a rough rule for when adding the catalog is
still worth it. Hint: think about tokens-per-correct-answer and
prefill vs decode time on a specific hardware class.


## 16. Stage 7M — CWE disambiguation (two words changed, one blocker closed)

> Stage 7L's 91.5% catalog-condition number included a brutal
> regression: CWE-89 went from 98% to 59.3%. Stage 7M tests the
> hypothesis that the catalog's CWE-78 and CWE-89 entries confused
> the model because of literal stem overlap, and fixes it by
> rewriting exactly those two entries.

### Confirming the hypothesis (free — no new run)

Re-scoring Stage 7L's 150 CWE-89 trials under condition B showed
**44 of the 61 errors went to CWE-78** (72%). Under condition A (no
catalog) and condition C (length-matched neutral text) the model
picked CWE-78 just 1 time out of 150. The confusion is not
background noise — it's a specific interaction with the catalog's
shared "constructs all or part of ... command using externally-
influenced input" phrasing.

### The catalog revision

Two entries rewritten, eight byte-identical. Diff is intentionally
small:

- CWE-78 now LEADS with "operating-system shell or subprocess API"
  and names concrete sinks: `os.system`, `subprocess shell=True`,
  `Runtime.exec /bin/sh -c`, `spawn`, backticks, `$(...)`.
- CWE-89 now LEADS with "database driver or ORM" and names
  concrete sinks: `cursor.execute`, `knex.raw`, `sa.text`, `PDO`,
  `gorm.Raw`, `Jdbc.executeQuery`.
- Each entry carries a short negation of the other
  category ("SQL databases are not involved" / "Operating-system
  shells are not involved").

Nothing else changed in the catalog.

### Headline (sampled, 2,000 trials per condition)

| Cond | k/n | Trial Wilson | Item bootstrap over 40 | Det |
|---|---|---|---|---|
| A (Stage 7L catalog) | 1,867/2,000 (**93.3%**) | 92.2-94.4 | 88.4-97.3 | 38/40 |
| B (revised catalog)  | 1,900/2,000 (**95.0%**) | 94.0-95.9 | 91.3-97.9 | **40/40** |

Paired McNemar p = 1.16e-3; B significantly better than A, though
item-level CIs overlap.

Per-CWE deltas (B-A):
- CWE-89: **+18.5** (62.0 → 80.5)  ← target recovered
- CWE-22: +6.0
- CWE-79: +2.5
- CWE-346: +2.0
- CWE-78: +1.5 (preserved at 100%)
- CWE-347 / 434 / 352 / 327: 0 (all preserved)
- **CWE-311: -14.0** (100 → 86)  ← NEW regression; 28/200 went to CWE-327

Cost: TTFT 25.1 → 48.3 ms (longer catalog, 821 → 980 prompt tokens).
Tokens/correct essentially unchanged (7.1 → 7.0).

### What this tells us

The targeted fix worked: CWE-89 → CWE-78 confusion collapsed from
55/200 to 10/200 trials. But the catalog is now ~160 tokens longer
and the model's attention mass redistributed. The next-most-
collapsible pair in the 2B's representation — CWE-311 ("no
encryption at all") vs CWE-327 ("broken algorithm") — became the
new confusion site, with 28/200 CWE-311 trials misclassified as
CWE-327.

The lesson: at 2B scale, fixing one pairwise confusion exposes the
next-weakest pair. Per-pair prompt engineering is a conveyor belt;
it did close the Stage 7L blocker but it is probably not the right
long-term structure. Stage 7N could try chain-of-selection (ask
for a top-3 shortlist, then disambiguate) which attacks the
bandwidth limit structurally rather than one pair at a time.

### Hands-on exercise

Open `docs/dyno/offsec/stage7m/sampled.jsonl`. For snippet `s7m01`
(a `cur.execute` SQL injection in Python), read all 50 seeds under
conditions A and B. Note the pred CWE on each.
1. Under A (Stage 7L catalog) what fraction went to CWE-78? Under B
   (revised)?
2. Repeat for `s7m05` (a `subprocess.Popen shell=True`). In which
   direction does the confusion run before and after the fix?

Write 3-4 sentences linking what you see to the catalog revision.

### Comprehension questions

**Q12** (7M). The catalog revision is TWO ENTRIES out of ten. If
I told you the revised catalog's OVERALL held-out accuracy was 93%,
what would you insist on seeing before accepting the claim? List the
two specific measurements the brief REQUIRES to prevent a hidden
regression in one of the eight untouched categories.

**Q13** (7M). The confusion between CWE-78 and CWE-89 was a
"catalog-induced" error — evidence: it does not happen under A or
C. Under what conditions would you expect catalog revisions to
still fail, even with disambiguating wording? Hint: think about how
many categories you have and how distinct their concrete sinks are.

## 17. Stage P1 — Coding agent pilot (SWE-bench Verified `pallets__flask-5014`)

> First real test of the CODING AGENT track: can the 2B model, shown one
> failing test and the file that needs changing, produce a correct
> unified diff that `git apply` accepts and `pytest` turns green?
> **No. 0 of 4 attempts resolved.** But the harness works, the oracle
> works, and the failure mode is specific and documented.

### The scaffold

Picked one SWE-bench Verified instance (`pallets__flask-5014`) for the
pilot: 390-byte gold patch, one FAIL_TO_PASS test, 59 PASS_TO_PASS
tests to protect. Model is shown problem statement + failing test +
full target file; it must emit a unified diff inside a ```diff``` fence.
The harness clones the repo at the pinned base commit, applies the SWE
test_patch, applies the model's diff with `git apply --index`, then
runs pytest for both the failing test and the first 60 passing tests.
Resolved = ALL pass. No LLM judgment anywhere.

### What actually happened (1 greedy + 3 sampled, seeds 8000-8003)

| metric | value |
|---|---|
| extracted a unified diff | **4 / 4** |
| `git apply` succeeded | **0 / 4** |
| **RESOLVED** | **0 / 4** |
| mean tokens per attempt | 636 |

Every attempt FAILED at `git apply` with exit 128. The model produced
plausible patches with the right conceptual edit (adding `if not name:
raise ValueError(...)`), but with HALLUCINATED hunk line numbers.
Greedy emitted `@@ -100,6 +100,10 @@` for a function that actually lives
at line 190. Correct semantics, wrong offsets, git refuses.

### What this teaches

Two things:
1. **The harness is honest.** Dropped `diff --git` headers, pinned
   `werkzeug`, pinned `pytest`, added `-W ignore::DeprecationWarning`
   — four harness bugs fixed before the recorded run. The 0/4 verdict
   is on the MODEL, not the setup.
2. **2B's failure mode has a name.** `hallucinated hunk line numbers`.
   This is a well-known 2B-class limitation (small models do not
   reliably track line positions in long files). A fuzzy-apply
   wrapper (`patch -F 3`) would recover these — but then the oracle
   is less strict than SWE-bench Verified's own grader. We did NOT
   relax.

### Hands-on exercise

Open `docs/dyno/coding/p1/attempts.jsonl`. For each of the 4 attempts
read `raw_output_head`. Count how many produced a diff whose BODY is
semantically equivalent to the gold patch (`if not name: raise
ValueError(...)` inserted anywhere inside `Blueprint.__init__` AFTER
the `super().__init__` call). Also count how many diffs would succeed
if you replaced `git apply` with `git apply --fuzz 3` or `patch -F 5`
(which allow offset drift). Write 3-5 sentences on the trade-off.

### Comprehension questions

**Q14** (P1). SWE-bench Verified's own evaluation harness uses `git
apply` strictly — no fuzz, no retry. Under what conditions would it
be defensible to relax that to `git apply --fuzz 3` for the 2B's
coding track, and what would you have to document to keep the number
comparable to the published SWE-bench leaderboard?

**Q15** (P1). The oracle here runs 60 of 59 pass_to_pass tests
(we cap for latency). If a cherry-picked subset of 60 is "friendly"
(all tests the model typically keeps green), item-level accuracy
could inflate. Describe one adversarial subset-selection the brief's
"do not inflate" rule would explicitly rule out.

## 18. Stage P2 — Agentic API-key replacement pilot

> The agentic track's protocol test: give the 2B a tiny Python repo
> with a hardcoded API key, a tool catalog, and 25 turns to replace
> the key with `os.environ["API_KEY"]`. **0 of 11 attempts resolved**,
> for an interesting reason: the model cannot reliably emit the
> required JSON tool-call format.

### The protocol

System prompt lists five tools and requires:

```
TOOL: {"tool": "write_file", "args": {"path": "src/x.py", "content": "..."}}
```

Harness parses the single-line JSON object after `TOOL:`, executes
the tool, feeds back `OBSERVATION: {...}`. Up to 25 turns.

### Oracle (environment, no LLM judgment)

1. `grep '<old key>' -r .` returns 0 hits.
2. `os.environ["API_KEY"]` idiom present.
3. `python -m compileall -q .` exits 0.
4. `pytest -q` with `API_KEY=dummy-value-for-tests` exits 0.

Resolved = all four.

### What happened (1 greedy + 10 sampled, seeds 9000-9010)

| metric | value |
|---|---|
| Resolved | **0 / 11** |
| Model emitted `finish` | 2 / 11 |
| Mean turns used | 24.1 / 25 (cap) |
| Parseable TOOL calls | 36 / 275 turn-slots (13%) |
| `write_file` calls that parsed | **0** |
| Mean tokens / wall time | 753 / 8.6 s per attempt |

### What the model did instead

It emitted `TOOL: read_file args {"path": "src/x.py"}` — the
category word (`read_file`) outside the JSON object. The parser
rejected it (required `{"tool": "read_file", "args": {...}}`). Two
attempts eventually produced valid `TOOL: {"tool": "finish", ...}`
and gave up without having written anything. The environment-ran
`pytest` kept passing because the seed files were never touched.

### Hands-on exercise

Open `docs/dyno/agentic/p2/attempts.jsonl`. For attempt `sampled/9010`
(highest tokens) read all 25 turns in `turns_head`. Identify:
1. Which turn first produces a conceptually-correct file replacement
   as the `content` argument?
2. Why does the parser reject it anyway?
3. What is the smallest harness-side change that would let the model
   succeed WITHOUT relaxing the oracle? (hint: there's a one-line
   change that keeps the oracle strict but makes the protocol forgiving).

### Comprehension questions

**Q16** (P2). The pytest oracle passed on all 11 attempts (the
existing tests don't reference the key) even though the task was
NOT solved. Why is this pytest-passes-but-resolved=false pattern a
feature, not a bug, of the four-rule oracle?

**Q17** (P2). We measured 13% TOOL-call parse rate. If we relaxed the
protocol to accept the model's dominant format (`TOOL: <tool> args
{json}`) the parse rate would jump but the task-completion rate might
not. Propose a MEASUREMENT that distinguishes "model can plan and
write" from "model can also format tool calls cleanly". Which of the
two is the real graduation gate?

## 19. Stage P1b + P2b — Action reliability: separating 4 capabilities

> P1 and P2 both reported 0 / N resolved. The question this experiment
> forced us to answer: **were those failures task-reasoning failures,
> or output-format failures?** With cleaner harnesses we can split them.

### Four distinct capabilities, often conflated

| Capability | What it means | How to measure |
|---|---|---|
| **Task reasoning** | Does the model know the right ANSWER? | semantically correct edit regardless of format |
| **Output formatting** | Can the model encode the answer in the required protocol? | parser accepts output / `git apply --check` passes |
| **Tool execution** | Can the harness actually EXECUTE the model's action? | tool actually runs, mutates state |
| **Verified completion** | Does the environment oracle confirm success? | pytest exit 0, grep count 0, etc. |

The raw P1/P2 numbers (0 / N) collapsed all four axes. The P1b + P2b
two-lane experiments let us separate them.

### P1 — coding (unified-diff vs JSON edit-spec lane)

Raw lane: model emits unified diff directly. Assisted lane: model
emits a structural edit-spec `{file, action, anchor, payload}`;
harness applies the spec literally, then generates the diff via
`git diff HEAD` and validates with `git apply --check`.

On the original SWE-bench Verified instance `pallets__flask-5014`:

| Metric | Raw | Assisted |
|---|---|---|
| Response parsed | 4 / 4 | 4 / 4 |
| `git apply --check` passes | **0 / 4** | **3 / 4** |
| RESOLVED (f2p + p2p) | 0 / 4 | 0 / 4 |

Formatting unlocked (0 → 3), but task reasoning still wrong on
real-repo code.

On a fresh held-out synthetic task (`area_of_circle` off-by-factor bug):

| Metric | Raw | Assisted |
|---|---|---|
| `git apply --check` passes | 0 / 4 | **4 / 4** |
| **RESOLVED (pytest exit 0)** | 0 / 4 | **3 / 4** |

🎯 First end-to-end resolved coding task for 2B. The lane effect
generalized across two different tasks (one real, one synthetic).
Where capability existed (simple math fix), formatting had been the
bottleneck; where it didn't (real repo code), assisted lane still
failed at the reasoning step.

### P2 — agentic (free-form TOOL: vs JSON-schema constrained)

Raw lane: model emits `TOOL: {json}` free-form; regex parses.
Constrained lane: exllamav3 `LLGuidanceFilter(json_schema=...)`
constrains decoding so the entire assistant turn is a schema-valid
tool call.

On the original API-key replacement task:

| Metric | Raw | Constrained |
|---|---|---|
| Parse-failure rate per turn | **99.3 %** | **0.0 %** |
| Tool actions actually executed (sum over 11 attempts) | 2 | **112** |
| RESOLVED (all 4 oracle checks) | 0 / 11 | 0 / 11 |

On a fresh "remove DEBUG prints" task: parse-failure 89 % → 1 %;
tool actions 28 → 87; RESOLVED 0 → 0.

Formatting + tool-execution both unlocked across BOTH tasks. Task
completion still at zero: the model picks a `write_file` action,
writes content that is syntactically valid, but the content often
drops the `import os` line or preserves the original key string.
The four-rule oracle catches all of these.

### What this teaches

1. **Two distinct 2B failure modes**: formatting-limited AND
   capability-limited. The same model that can't emit a unified
   diff CAN emit a JSON edit-spec AND can drive a 25-turn tool loop
   when the decoder enforces schema. Prior P1/P2 reports confounded
   these; P1b/P2b separate them.
2. **Where capability exists, tooling removes the bottleneck**.
   Fresh-held-out P1 went from 0 / 4 to 3 / 4 purely by changing
   the output protocol — no new model, no new training.
3. **Where capability does not exist, tooling cannot save it**.
   P2 tool-execution rose 56×, verified completion rose 0×. 2B
   does not multi-step-plan file edits that must keep ONE file
   working while modifying another.
4. **Do NOT silently repair raw output**. The brief required it;
   we honored it. Raw numbers stayed at 0 / 4 and 0 / 11.

### Hands-on exercise

Open `docs/dyno/coding/p1b_fresh/attempts.jsonl` and `_generated_patch.diff`
under each `attempts/attempt_assisted_*/` directory. For the one
assisted FAILURE (seed 10003) read the generated patch. The
`git apply --check` passed; `pytest` failed. In one paragraph
describe what the model did wrong at the TASK-REASONING step, and
explain why no amount of output-formatting improvement would have
saved this attempt.

### Comprehension questions

**Q18** (P1b). We generate the patch from `git diff HEAD` on a
committed base. Why do we commit the test_patch BEFORE the model
edits (vs leaving it in the index)? What specific git-apply failure
mode did we see before this fix?

**Q19** (P2b). The constrained lane's tool-execution rate is 56×
the raw lane's, but the verified-completion rate is unchanged. If
you had to pick ONE of these two metrics as your official
graduation signal for the AGENTIC track, which would you pick and
why? The brief's "the environment confirms success" rule constrains
your answer; so does operator trust in a tool-using agent.

## 20. Stage P2c — Agentic failure diagnosis + minimal intervention

> P2b showed constrained decoding unlocked 100% tool-parse rate but
> 0/22 task completion. WHY? Diagnostic re-run with full trace
> capture, per-attempt classification, and ONE minimal harness
> intervention lifted completion 1→5 on DEBUG-prints and 0→2 on a
> fresh rename task without touching the oracle.

### Reconciliation of prior denominator

§19 wrote "3/26 verified completion" — arithmetic error. The actual
P1b + P2b totals:
- 60 total attempts (raw + treatment lanes combined).
- 30 treatment-lane attempts (assisted P1b + constrained P2b).
- 3 resolved, all in P1b-fresh-assisted.
- Correct ratio: **3 / 60** overall, **3 / 30** treatment only,
  **3 / 15** fresh-only treatment.

Not a different conclusion (2B still NOT GRADUATED on any track),
just honest math.

### The four failure modes we found

Per-attempt classification across 22 baseline constrained traces:

| Label | Count | Earliest mistake |
|---|---|---|
| repeated_actions | 10 / 22 | 3+ identical consecutive tool calls (usually write_file looping) |
| incorrect_edit | 3 / 22 | finish called with the old pattern still in the repo |
| missing_validation | 4 / 22 | finish called without ever running grep |
| other / resolved | 5 / 22 | hit cap without a diagnosable path; 1 actual pass |

Earliest-irreversible-mistake turn was recorded per attempt. For
incorrect_edit + missing_validation, the mistake is the `finish` call
itself (model declared done too early). For repeated_actions, the
mistake is the first of the three identical consecutive tool calls.

### The intervention (one if-branch)

Keep a `saw_zero_grep` flag per attempt. Set it True the moment the
model's grep for the task's `old_key` returns rc=1. When the model
calls `finish`, if the flag is False, the harness returns a REJECTED
observation saying "grep the pattern first; it must return rc=1".
The loop continues; the 25-turn cap is unchanged; the oracle is
unchanged.

No prompt engineering beyond ONE extra sentence about the gate. We
do NOT silently repair raw output; both lanes are preserved.

### Measurement (3 tasks × 2 lanes × 11 seeds = 66 attempts)

| Task | Baseline | Intervened | Δ | Finish rejections |
|---|---|---|---|---|
| API-key | 0 / 11 | 0 / 11 | 0 | 9 |
| DEBUG-prints | 1 / 11 | **5 / 11** | **+4** | 1 |
| Rename (fresh held-out) | 0 / 11 | **2 / 11** | **+2** | 0 |

Paired on seed × lane: DEBUG improved on 5 seeds that baseline
failed; rename improved on 2 seeds. API-key stayed at 0 because the
oracle requires THREE independent conditions and the gate only
gates on ONE of them.

### Why the API-key task does not improve

The API-key oracle is a conjunction: grep=0 AND
`os.environ["API_KEY"]` present AND `import os` present AND pytest
passes. The grep-before-finish gate addresses the first condition
only. The 2B model, after satisfying grep=0, consistently forgets
`import os` or writes a replacement that drops the function-call
arg. pytest catches this, the oracle correctly rejects, and
resolution stays at 0.

Honest reading: the intervention improves tasks whose oracle is
grep-dominated. It is NOT a universal planning fix.

### Hands-on exercise

Open `docs/dyno/agentic/p2c/attempts_i_debug.jsonl`. Find the 5
seeds that RESOLVED under the intervention (verdict.resolved=True).
For each, examine the `full_turns` trace and identify:
1. Which turn the model FIRST emitted a grep for `DEBUG:`.
2. How many finish rejections the model received before the final
   successful finish.
3. Whether the model's first finish attempt was before or after the
   grep.

Write 3-5 sentences on what the gate is teaching the model within
the loop vs what pre-training would have to teach ahead of time.

### Comprehension questions

**Q20** (P2c). The intervention improved DEBUG and Rename, but not
API-key. Would increasing the intervention's scope (e.g. ALSO gate
finish on pytest returning 0) generalize to API-key? What would go
wrong, and what does this teach about "oracle-shaped interventions"?

**Q21** (P2c). Baseline DEBUG resolved 1/11 in this run but 0/11 in
the prior P2b report. Both used the same seeds (9100-9110), same
model, same task file. Propose the single most likely cause of the
discrepancy and name the ONE exhibit in the receipts you would
examine to confirm or refute it.
