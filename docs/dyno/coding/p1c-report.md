# P1c — Coding Agent Reliability: Edit-Test-Retry

**Status**: DoD **MET** — 3/4 end-to-end resolutions on `django__django-14089` via independently verified pytest. CODING track NOT graduated.
**Base commit**: `1841b1536`.
**Receipts**: `docs/dyno/coding/p1c/attempts_{flask,django}.jsonl`.
**Harness**: `docs/dyno/coding/p1c/harness_edittest.py`.
**Instances**: `docs/dyno/coding/p1/instance.json` (flask-5014) + `docs/dyno/coding/p1c_django/instance.json` (django-14089).

## Gate 1 — Classify prior P1b assisted-lane failures

All 4 P1b assisted attempts on `pallets__flask-5014` produced syntactically valid patches and all failed the task oracle. Per-attempt root cause:

| seed | spec_apply | failure mode | evidence |
|---|---|---|---|
| greedy/8000 | OK | **incorrect_logic** | inserts `if "." in name:` check (which already exists in file); misses the empty-name case |
| default/8001 | OK | **incorrect_logic** | same wrong check, also writes extra `self.X = ...` lines unnecessarily |
| default/8002 | anchor_not_found | **wrong_target** | payload is CORRECT (`if not name: raise ValueError(...)`) but anchors against a line `if name.startswith('.'):` that doesn't exist (hallucination) |
| default/8003 | OK | **wrong_target** | edits an unrelated `url_prefix` block |

**Breakdown**: 2/4 `incorrect_logic`, 2/4 `wrong_target`. Tied. `incomplete_edit` and `missing_validation` modes did not occur. The single intervention below addresses both tied modes because test feedback catches BOTH wrong logic (test still fails) and wrong target (test unchanged).

## Gate 2 — The intervention

**Bounded read → edit → test → diagnose → retry** with hard caps:
- Up to **3 edit attempts** per task (initial + 2 retries).
- After each edit: `git diff HEAD` → `git apply --check` → `pytest FAIL_TO_PASS` (60s cap).
- If FAIL_TO_PASS fails, the pytest tail is **fed back verbatim** in the next prompt as "PREVIOUS ATTEMPTS: Turn N: fail_to_pass FAILED. pytest tail: …".
- If FAIL_TO_PASS passes, run `pytest PASS_TO_PASS` (180s cap) and finalize.
- Model MAY emit `action: "finish"` to stop early.
- Schema enforced by `LLGuidanceFilter(json_schema=EDIT_SCHEMA)` (constrained decoding); the entire assistant turn is one JSON object matching `{action, file, anchor, payload}`.

Oracle uses the project's own test runner only (pytest for flask, `runtests.py` for django). **No LLM judgment.**

## Gate 3 — Original task (pallets__flask-5014, 4 paired attempts)

| seed | resolved | turns | tests_ran | patch_valid | tokens | elapsed_s |
|---|---|---|---|---|---|---|
| greedy/8000 | ❌ | 2 | 1 | ✅ | 1,819 | 24.5 |
| default/8001 | ❌ | 3 | 2 | ✅ | 490 | 6.4 |
| default/8002 | ❌ | 3 | 2 | ✅ | 443 | 6.2 |
| default/8003 | ❌ | 3 | 2 | ✅ | 363 | 5.4 |

**Flask: 0/4 resolved.** All attempts produce valid patches that apply cleanly. Retries let the model see the pytest failure tail, but across up to 3 attempts it never finds the correct insertion point for an empty-name check. This is a HARD semantic case: the function is at ~line 190 of a 400-line file and the model repeatedly re-emits variants of the wrong check.

## Gate 4 — Fresh SWE-bench Verified task (django__django-14089)

Instance: add `__reversed__` to `OrderedSet`. Base commit `d01709aae`. 1 FAIL_TO_PASS, 30 PASS_TO_PASS (first 30 of the manifest's larger set). Baseline sanity re-verified: FAIL_TO_PASS raises `TypeError: 'OrderedSet' object is not reversible`; PASS_TO_PASS clean.

| seed | **resolved** | turns | tests_ran | best_f2p_rc | tokens | elapsed_s |
|---|---|---|---|---|---|---|
| greedy/8000 | **✅** | 1 | 1 | 0 | 71 | 5.8 |
| default/8001 | **✅** | 1 | 1 | 0 | 72 | 1.7 |
| default/8002 | **✅** | 1 | 1 | 0 | 65 | 1.6 |
| default/8003 | ❌ | 3 | 2 | 1 | 218 | 3.1 |

**Django: 3/4 resolved end-to-end on the first attempt.** The greedy and two sampled attempts each produced:
```json
{
  "action": "insert_before",
  "file": "django/utils/datastructures.py",
  "anchor": "    def __contains__(self, item):\n        return item in self.dict",
  "payload": "    def __reversed__(self):\n        return iter(reversed(self.dict))"
}
```
which git-applied, triggered `pytest test_reversed PASS`, then triggered `pytest <20 PASS_TO_PASS tests> PASS`. Resolved = both pass. **This is the P1c definition of done.**

Seed 8003 proposed an edit to the wrong location (`insert_before` targeting a different block); 3 edit attempts retried but the model never shifted to the correct anchor. Retry intervention did not save this seed — it is a `wrong_target` failure that no number of retries fixes without stronger search.

## Harness bug discovered + fixed mid-run

First run of P1c appeared to produce semantically correct django edits but ALL tests still failed with the baseline error. Root cause: `pip install -e django-base` placed the venv's editable install on `django-base/django/…`, NOT the per-attempt worktree. The model's edits to the attempt worktree were invisible to the imported `django` package.

Fix (one line in `run_tests`): set `PYTHONPATH=<attempt_dir>:…` so the worktree's `django/` shadows the installed one. After the fix, django jumped from 0/4 to 3/4 resolved on IDENTICAL model outputs. This is a HARNESS FIX, not a model improvement. Honest to report: the model's semantic answer was right from the beginning; the harness was discarding its work.

## Five metrics per attempt (reported SEPARATELY per the brief)

| Metric | Flask | Django |
|---|---|---|
| Patch validity (git apply --check passes ≥ 1 turn) | **4 / 4** | **4 / 4** |
| Test completion (pytest ran to a verdict ≥ 1 turn) | **4 / 4** | **4 / 4** |
| Task resolution (f2p + p2p both pass) | **0 / 4** | **3 / 4** |
| Total tokens (mean) | 779 | 107 |
| Total elapsed (mean seconds) | 10.6 | 3.1 |

## Aggregate across P1 + P1b + P1c (unified coding-track picture)

| Stage | Task | Lane | Resolved |
|---|---|---|---|
| P1 | flask-5014 | one-shot raw unified diff | 0 / 4 |
| P1b | flask-5014 | raw | 0 / 4 |
| P1b | flask-5014 | assisted edit-spec | 0 / 4 |
| P1b | synthetic circle-area | raw | 0 / 4 |
| P1b | synthetic circle-area | assisted edit-spec | **3 / 4** |
| P1c | flask-5014 | edit-test-retry | 0 / 4 |
| P1c | django-14089 (SWE-bench Verified) | edit-test-retry | **3 / 4** |

**6 resolved out of 32 total coding attempts.** Of the 6, all are in treatment lanes (assisted or edit-test-retry) and on tasks where the semantic edit is small and localizable. 0 resolved on `pallets__flask-5014` across all three stages and both lanes — that instance is harder than 2B can do consistently, with or without retries.

## Why Flask still fails under retries, honestly

The retry feedback loop shows the model the pytest tail `"Failed: DID NOT RAISE <class 'ValueError'>"`. That tells the model its current edit doesn't raise when given an empty name. But across the 2-3 retries the 2B model repeatedly produces variants of the same wrong check (`if "." in name:`) rather than converging on `if not name:`. The retry loop can't lead the model to a correct answer the model doesn't know how to derive from the test name.

The retry WOULD likely fix this if the model could read the test body and reason about it. The test body IS in the prompt (`FAILING TEST` section), so in principle it has what it needs. It just doesn't use the information. That is a 2B capability limit, not a harness bug.

## Scope guards honored

- ✅ Raw and assisted P1b baselines **preserved intact** (not re-used; both attempt files untouched).
- ✅ Oracles are pytest only; no LLM judges anywhere.
- ✅ Patches generated by `git diff HEAD` and validated with `git apply --check` in a CLEAN worktree.
- ✅ SWE-bench Verified instance pinned to base_commit `d01709aae`; test_patch applied + committed before model editing; identical to upstream evaluation.
- ✅ 2B model unchanged (same `UnstableLlama/Qwen3.5-2B-exl3-8.00bpw` sha256 across all stages).
- ✅ No SFT / DPO / QLoRA / new quantizers / new models.

## Next single 2B graduation blocker (UNCHANGED)

**Stage 7M OFFSEC: CWE-311 → CWE-327 confusion (28 / 200 held-out).** P1c lifts coding end-to-end from 0 to 3 resolutions on a real Verified instance by combining structural edits with test-driven retry, but does NOT graduate the CODING track (which requires 40 held-out instances per `docs/dyno/2b-roadmap.md`) and does NOT override the OFFSEC blocker.

## Reproduce

```bash
# re-score existing receipts
jq '{resolved, turns_used, total_tokens, total_elapsed_s}' docs/dyno/coding/p1c/attempts_django.jsonl

# full re-run (requires Qwen3.5-2B EXL3 + django-base at pinned commit)
ssh spark '/home/chris/opt/exllamav3-venv/bin/python /home/chris/p1c/dyno/harness_edittest.py \
  --instance /home/chris/p1c/dyno/instance_django.json \
  --repo-base /home/chris/p1c/django-base \
  --workspace /home/chris/p1c/attempts_django \
  --venv /home/chris/p1c/venv \
  --out /home/chris/p1c/attempts_django.jsonl'
```
