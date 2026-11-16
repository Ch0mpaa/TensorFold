# 2B Dyno — Complete Specification and Roadmap

**Scope**: perfect Qwen3.5-2B across five independent evaluation tracks
before advancing to 4B or larger. No training, no new quantizers, no
new model sizes until all five tracks pass their own gates on this one
model.

**Status at time of writing**: the OFFSEC-classification sub-track is
mature (Stages 7H → 7M). The other three accuracy tracks and the
performance/fidelity track exist only as this specification. Nothing
is implemented from them yet. The brief is explicit: deliver a
**prioritized plan**, not a full implementation.

## The five tracks

| Track | What it measures | Current state |
|---|---|---|
| GENERAL | reasoning, math, comprehension, knowledge, instruction following | NOT STARTED |
| AGENTIC | planning, tool use, state management, recovery, verified task completion | NOT STARTED |
| CODING AGENT | repo edits, debugging, compiling, regression tests, patch correctness | NOT STARTED |
| OFFSEC | CWE classification (Stages 7H-7M), web / API sec, AD attack paths, lab tasks, pentest reporting | CWE sub-track mature; other sub-tracks NOT STARTED |
| PERFORMANCE / FIDELITY | decoding, prefill, TTFT, memory, energy, numerical correctness | Partially existing (EXL3 / TFOLD benchmarks preserved since Stage 7H) |

The four accuracy tracks and the fifth fidelity track run
INDEPENDENTLY — a model that perfects OFFSEC classification but
fails CODING AGENT patch-correctness does NOT graduate.

## Non-negotiable rules across every track

1. **Dev / held-out split.** Every track MUST publish which examples
   are used for prompt / system-design iteration (dev) and which are
   touched exactly once at evaluation time (held-out). No track may
   quote a held-out number from a pool used during iteration.
2. **Independent scoring.** Every track MUST have at least TWO scoring
   implementations that AGREE on a dry-run before the primary becomes
   the reference. CWE sub-track did this with `scorer_v2a`
   (regex) vs `scorer_v2b` (no-regex); 100% agreement on 1,100 trials
   is the acceptance bar.
3. **Frozen BEFORE run.** Prompt, model revision, sampler, EOS IDs,
   seeds, scorer, and all reference assets are committed to git before
   the model is invoked. SHA-256 recorded in a `freeze.json`.
4. **Item-level AND trial-level CIs.** Trial-level Wilson CI
   reported alongside bootstrap CI over distinct items. A track that
   only reports the trial-level number is not passing.
5. **Paired comparisons.** When two conditions are compared the run
   MUST share seeds per item so paired McNemar is valid.
6. **Preserve prior receipts.** No stage may modify or delete the
   receipts of a prior stage. Every `jsonl` is append-only.
7. **No cross-track leakage.** The CODING track's held-out
   repositories may NOT appear in the GENERAL track's training-set
   stand-ins and vice versa.

## Per-track specification

### GENERAL track

| Field | Value |
|---|---|
| Datasets (held-out) | GPQA-diamond (≤ 198 items, open-book) · MATH-500 (held-out 100-item subset) · MMLU-Pro (random 500-item sample, fixed seed) · IFEval (instructions-with-constraints subset, 150 prompts) |
| Dev slice | separate 20-item slice from each dataset's publicly reserved val/dev split |
| Scorer primary | exact-match or regex for numeric / final-answer tokens |
| Scorer secondary | LLM-judge: a 3B-or-smaller judge voting majority-of-3 using FROZEN 3B weights; its own agreement vs exact-match must be ≥ 90% on the dev slice before it joins the primary scorer |
| Trial N | 20 sampled seeds per prompt (GPQA / MATH / IFEval), 1 greedy per prompt (MMLU-Pro) |
| CI methods | Wilson + per-item bootstrap over distinct prompts |
| Reproducibility rule | seeds `range(8000, 8020)`; new run uses non-overlapping seed window |

### AGENTIC track

| Field | Value |
|---|---|
| Harness | a single-process ReAct-style loop with a FIXED tool catalog (shell, http, file-read, file-write, grep, note); scripted per task |
| Tasks | 50 bounded workflow tasks across 5 families: repo investigation, bug-reproduction, log triage, file-system refactor, package-install dependency resolution |
| Verifier | per-task Oracle: automated check script that confirms the end-state. **No LLM judgment on success/fail**; the environment confirms or denies. |
| Dev slice | 10 tasks |
| Held-out | 40 tasks, scored once |
| Trial N | 10 sampled seeds per task (temperature=0.4 to allow some exploration) + 1 greedy |
| Capture | full tool-call trace with hashes; recoveries counted; failed-and-recovered counted separately from first-try success |
| CI | bootstrap over 40 tasks |
| Max agentic-loop turns | 25 |

### CODING AGENT track

| Field | Value |
|---|---|
| Test harness | SWE-bench Verified: subset of 50 instances filtered by: (a) single-file edits only, (b) test-suite runs in < 60 s, (c) container cache fits on Spark node |
| Scoring | authoritative: the project's own test runner before/after. "Correct" means the fail→pass test transitioned from red to green AND the existing green tests stayed green. |
| Dev slice | 10 instances |
| Held-out | 40 instances |
| Trial N | 3 sampled seeds per instance (temperature=0.3) + 1 greedy baseline |
| CI | item-level bootstrap over 40 instances |
| Non-goal | we are NOT trying to compete with Claude / GPT-4-class scores. We want a reproducible number for 2B on single-file SWE-bench and a honest honest-gap report. |

### OFFSEC track

Current (CWE classification sub-track):
- Stages 7H → 7M, 30-40 held-out snippets per stage, paired McNemar,
  item-level bootstrap, three independent scorers in use.

Expansion for the full track:
| Sub-track | Status | Plan |
|---|---|---|
| CWE classification | mature | continue with ≥ 40 held-out per revision |
| Web / API sec | NOT STARTED | 20-task benchmark: parameter tampering, IDOR, open redirect, SSRF guard, rate-limit bypass. Oracle: scripted request + expected status / content. |
| AD attack-path | NOT STARTED | out of scope for a 2B run on Spark without a lab; deferred to 4B+ |
| Authorized lab | NOT STARTED | depends on an isolated lab network; deferred |
| Pentest reporting | NOT STARTED | LLM-judge on rubric by a second model; defer until we trust the judge |

### PERFORMANCE / FIDELITY track

Preserved through Stages 7H-7M and NOT revisited casually.

| Metric | Current measurement | Acceptance (unchanged) |
|---|---|---|
| EXL3 safetensors warm decoding | 109.5 tok/s median, 108.1 mean | ≥ 100 tok/s |
| TFOLD v0.2 warm decoding | 110.8 tok/s median, 107.5 mean | ≥ 100 tok/s |
| Prefill cost | not benchmarked yet | add during Stage 7P |
| TTFT | 25-46 ms in Stage 7L | document band per prompt length |
| VRAM | 2208 MB | ≤ 2500 MB |
| Numerical correctness | MD5 `c6a03e566512fa102ab5b5d25eb64782` (both formats, byte-identical bf16 logits) | MUST match |
| Energy | not measured | optional; defer |

## Implementation plan — WHAT TO BUILD NEXT

Per the brief, DELIVER the plan — do NOT implement all tracks now.

The next two implementation priorities, in order:

### Priority 1 — one small CODING AGENT task (dev)

Pick ONE SWE-bench Verified instance from the 10-task dev slice that:
- Changes a single file.
- Has the fail→pass test in the top-level `tests/` directory.
- Container build cache is already warm on Spark.

Build:
- `dyno/coding/runner.py` — invokes the agentic loop (reuse the
  AGENTIC harness if possible), applies the proposed patch, runs the
  project's test command, records before/after output.
- `dyno/coding/snippets/<instance>.json` — task description + test
  command + file list.
- Freeze JSON for this single task.
- Run 3 sampled seeds + greedy. Produce a receipt per seed.
- Score with the project's native test runner.

The entire deliverable is **one task, five receipts, one analysis
record**. The purpose is to prove the harness works end-to-end on 2B
before scaling to 10 then 40 tasks.

### Priority 2 — one small AGENTIC task (dev)

Pick ONE bounded workflow task. Suggested: "Given a repo path,
find all references to a hardcoded API key and replace them with
`os.environ['API_KEY']`." This is bounded, verifiable (grep for the
old key = 0 hits, new idiom present ≥ 1), and easy to seed.

Build:
- `dyno/agentic/harness.py` — ReAct loop with shell/file/grep tools.
- `dyno/agentic/tasks/replace_api_key.json` — the task spec + oracle check.
- Freeze JSON.
- Run 10 sampled seeds + greedy.
- Oracle checks grep / idiom presence.

Again: **one task, 11 receipts, one analysis record**.

### Explicit non-goals for now

- Do NOT implement GENERAL's four datasets. They are the biggest track
  but also the most well-understood; we can build the harness and copy
  the dataset handling later.
- Do NOT start AD attack-path, authorized lab, pentest reporting.
- Do NOT add performance / fidelity metrics beyond what Stages 7H-7M
  already measure.

## 2B graduation gate

The 2B does NOT advance to 4B until EVERY ONE of the following holds:

1. OFFSEC CWE classification: held-out ≥ 90%, CWE-89 regression resolved,
   independently scored, reproducible receipts committed.
2. OFFSEC web / API sec: 40 held-out tasks scored, item-level CI
   computed, no CI-lower-bound below 40%.
3. GENERAL track: at least ONE of GPQA / MATH / MMLU-Pro / IFEval
   has a held-out score and documented CI.
4. AGENTIC: 40 held-out tasks scored by environment oracle; first-try
   success rate reported.
5. CODING AGENT: 40 held-out SWE-bench Verified instances scored by the
   project's own test runner; CI reported.
6. PERFORMANCE / FIDELITY: EXL3 and TFOLD decoding ≥ 100 tok/s
   preserved; MD5 `c6a03e5665...` bf16 identity preserved.

Short form: five numbers plus two preserved baselines. Nothing softer.

## Reproducibility rules shared by every track

- Seeds published in `freeze.json`; never randomised at runtime.
- No retries hidden from the receipt.
- No prompt change based on a failed held-out trial.
- Appeal path: if a scorer disagrees with itself, open a new file;
  never rewrite the receipt.

## Journal expectations

Each track MUST update `learning-notes/2b-dyno-foundation.md` with a
new section on build, failure modes, independent-scoring evidence, and
one hands-on exercise + two comprehension questions.
