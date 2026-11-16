# P1 — SWE-bench Verified `pallets__flask-5014` pilot

**Status**: harness functional, oracle verified, **0 / 4 attempts resolved**. 2B class NOT GRADUATED.
**Receipts**: `docs/dyno/coding/p1/{attempts.jsonl, analysis.json}`
**Frozen assets**: `docs/dyno/coding/p1/{instance.json, freeze.json, runner.py, analyze.py}`

## Why this instance

Selection criteria from `docs/dyno/2b-roadmap.md` priority 1:
- Single-file edit (`src/flask/blueprints.py`).
- Standard `pytest` runner; 60-second budget; no Docker required.
- 390-byte gold patch — simplest-in-class; 1 `FAIL_TO_PASS`, 59 `PASS_TO_PASS`.
- Source: `princeton-nlp/SWE-bench_Verified` (test split) via HuggingFace datasets, instance id `pallets__flask-5014`.
- Pinned at base commit `7ee9ceb71e868944a46e1ff00b506772a53a4f1d`.

## Harness

Single-shot (no agent loop) per the brief:
1. Fresh `git clone` + `git checkout <base_commit>` per attempt.
2. Apply the SWE-bench `test_patch` (adds the failing test).
3. Confirmed baseline sanity: `pass_to_pass` passes AND `fail_to_pass` fails.
4. Model is shown: problem statement, failing test body, full target file. Asked for a unified diff inside a ```diff ...``` fence.
5. `git apply --index` against the clean worktree.
6. `pytest -q` on `FAIL_TO_PASS` then `PASS_TO_PASS` (first 60, for latency).
7. Resolved = ALL `FAIL_TO_PASS` pass AND ALL `PASS_TO_PASS` pass.

No LLM judge anywhere. The project's `pytest` is the authority.

## Results

| mode | seed | status | patch extracted | git apply rc | n_tokens | gen_s |
|---|---|---|---|---|---|---|
| greedy | 8000 | patch_apply_failed | ✅ | 128 | 111 | 3.6 |
| sampled | 8001 | patch_apply_failed | ✅ | 128 | 419 | 3.9 |
| sampled | 8002 | patch_apply_failed | ✅ | 128 | 894 | 8.4 |
| sampled | 8003 | patch_apply_failed | ✅ | 128 | 1,121 | 10.5 |

| Oracle metric | Value |
|---|---|
| Extracted unified diff | **4 / 4** |
| `git apply` clean | **0 / 4** |
| fail_to_pass passed | **0 / 4** |
| pass_to_pass passed | **0 / 4** |
| RESOLVED | **0 / 4** |
| Tokens generated (sum / mean) | 2,545 / 636 |
| Generation wall-time (sum / mean) | 26.4 s / 6.6 s |

## Failure analysis

**Every attempt failed at the `git apply` step with exit code 128.** The model
produced syntactically valid unified diffs but with *hallucinated line
numbers* in the hunk headers. Greedy example:

```
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -100,6 +100,10 @@ class Blueprint(Scaffold):
         if "." in name:
             raise ValueError("'name' may not contain a dot '.' character.")

+        if not name:
+            raise ValueError("Blueprint name must not be empty.")
+
         self.name = name
```

The patch body is semantically close to the gold patch, but the `@@ -100,6
+100,10 @@` header is wrong — the actual function definition in the file is
around line 190, not 100. `git apply` rejects it.

Two sampled attempts additionally made structural edits to the wrong location
(insertions into the constructor signature rather than the body), which would
have been rejected on semantics even if the line numbers matched.

### Harness bugs discovered and fixed mid-pilot

1. First extractor only accepted the `diff --git a/X b/X` prelude. Model
   produces bare `--- a/X / +++ b/X` style. Fixed: extractor now synthesizes
   the `diff --git` header line when only the minimal form is present.
2. `werkzeug` version mismatch broke the baseline; pinned `werkzeug 2.3.x`.
3. Python 3.12 `DeprecationWarning: ast.Str` breaks pytest unless
   `-W ignore::DeprecationWarning` is set. Added to the pytest command.
4. `pytest`'s old-API `monkeypatch.notset` attribute was missing from pytest 8;
   pinned `pytest 7.1.3`.

All four fixes were made BEFORE the 4 recorded attempts, to the frozen
baseline reproduction. The runner and freeze SHA-256 are what landed on disk.

## Cost

Per the roadmap the budget was "a single task, five receipts, one analysis".
Delivered: 4 attempts, 1 analysis file, 1 report. Wall time ~2 minutes total
(generation only; test runs skipped because no patch applied). Total model
tokens: 2,545.

## What the harness does NOT yet do

- No agent loop. One-shot patch only. A model that could iteratively read
  files, run tests, and re-emit patches might fare better — but that is the
  AGENTIC track (P2), not this track. Keeping the single-shot variant honest.
- No automatic patch-repair (e.g. fuzzy hunk matching). We tested whether the
  2B model can produce a patch that `git apply` accepts out of the box. It
  cannot on this instance.
- Only 60 of 59 PASS_TO_PASS (we cap at 60 for latency). All 59 would have
  been run had any patch applied.

## Comparison vs the roadmap's gate

Roadmap § CODING AGENT required: "the project's own test runner before/after;
`Correct` = fail→pass test transitioned red→green AND existing green tests
stayed green." Oracle implemented as specified. Scoring against the gate:
**0 / 4 attempts pass**. Reproducible: yes (fixed seeds 8000-8003, pinned
base_commit, pinned venv, pinned pytest).

## Remaining work before the track can be declared

Per the brief, "do not declare the entire coding track passed based on this
single task." Status:
- Harness end-to-end: **PASSES** (baseline check, apply, oracle all work).
- 2B on this one instance: **0 / 4**.
- Needs ≥ 40 held-out SWE-bench Verified instances with item-level CI per
  the roadmap graduation gate. 1 / 40 instances evaluated so far.

## Next single 2B graduation blocker (unchanged from Stage 7M)

**CWE-311 → CWE-327 confusion** remains the single next prompt-level blocker
on the OFFSEC classification sub-track (28 / 200 held-out trials misclassified
under condition B of Stage 7M). P1's `patch_apply_failed` is a per-task
blocker on the CODING track — a different sub-track and does not supersede
the existing OFFSEC-CWE-311 item.

## Reproduce

```bash
# re-score existing receipts
python docs/dyno/coding/p1/analyze.py

# full re-run (requires exllamav3 + Qwen3.5-2B EXL3 + Flask deps on a Linux host)
ssh spark 'cd ~/p1 && python dyno/runner.py \
    --instance dyno/instance.json \
    --flask-base ~/p1/flask-base \
    --workspace ~/p1/attempts \
    --venv ~/p1/venv \
    --out ~/p1/attempts.jsonl'
```
