# P1b — Coding Action Reliability: Raw vs Assisted lanes

**Status**: harness proves assisted lane works end-to-end; **first resolved coding task shipped** (3/4 on fresh held-out). 2B class NOT GRADUATED.
**Receipts**: `docs/dyno/coding/p1b/attempts.jsonl`, `docs/dyno/coding/p1b_fresh/attempts.jsonl`
**Harness**: `docs/dyno/coding/p1b/runner_both.py` + `docs/dyno/coding/p1b_fresh/runner_fresh.py`

## Design

Two lanes, same task, same seeds, independently scored.

| Lane | Protocol | Scoring |
|---|---|---|
| **raw** | Model outputs a unified diff in a ```diff``` fence; harness `git apply --check` | Patch valid iff `--check` returns 0 |
| **assisted** | Model outputs a JSON edit-spec `{file, action, anchor, payload}`; harness locates the verbatim anchor in the file, applies the structural op, runs `git diff HEAD` to generate the patch, then `git apply --check` on a clean worktree | Patch valid iff the generated diff passes `--check` |

**Scoring independence**: the assisted lane is NOT a repair of raw output. Each lane uses a DIFFERENT model response and a DIFFERENT prompt. We never retroactively re-attribute an assisted success to the raw lane.

## Lane-effect result on `pallets__flask-5014` (SWE-bench Verified)

Same seeds 8000-8003, 4 attempts per lane, isolated git worktrees.

| Metric | Raw | Assisted |
|---|---|---|
| Response parsed (patch or spec) | 4 / 4 | 4 / 4 |
| **`git apply --check` passes** | **0 / 4** | **3 / 4** (1 spec_apply_failed: anchor_not_found) |
| **RESOLVED (f2p+p2p)** | **0 / 4** | **0 / 4** (patches applied; semantic edit wrong) |

Raw lane failure mode: hallucinated hunk line numbers (patches targeted `@@ -100 @@` for a function body at ~line 190).
Assisted lane failure mode: model inserts a redundant `if "." in name:` check, which doesn't satisfy the empty-name test. Patch is syntactically valid; semantics wrong.

## Fresh held-out generalization

**Fresh task** (`docs/dyno/coding/p1b_fresh/instance.json`): synthetic, self-contained, author-labeled — NOT from the SWE-bench Verified manifest. `src/calc.py::area_of_circle` returns `pi * (2r)^2` instead of `pi * r^2`. One failing test expects 78-79 for radius=5. Seeds 10000-10003.

| Metric | Raw | Assisted |
|---|---|---|
| Response parsed | 4 / 4 | 4 / 4 |
| **`git apply --check` passes** | **0 / 4** | **4 / 4** |
| **RESOLVED** (pytest exits 0) | **0 / 4** | **3 / 4** (pytest fail on seed 10003) |

**🎯 First end-to-end resolved coding task for 2B on any task.** The raw lane fails all 4 the same way (`patch_invalid_check`); the assisted lane completes the full edit-spec → git diff → git apply → pytest pipeline on 4/4, and the pytest oracle confirms 3/4 attempts produced a mathematically correct fix.

The ONE assisted failure (seed 10003) wrote a diff that applied cleanly but was semantically wrong — the oracle caught it, the harness recorded `pytest_fail`, and we do NOT count it as resolved.

## Separation of concerns (the whole point of this experiment)

| Capability | Metric | Raw | Assisted (fresh) |
|---|---|---|---|
| **Task reasoning** (do you know the answer?) | model produces a semantically correct edit | ≥ 3 / 4 (greedy on flask hints correctness) | 3 / 4 |
| **Output formatting** (can you encode it correctly?) | `git apply --check` passes | 0 / 4 | 4 / 4 |
| **Tool execution** (can the harness run your change?) | patch applied + pytest launched | 0 / 4 | 4 / 4 |
| **Verified completion** (does pytest turn green?) | pytest exits 0 | 0 / 4 | **3 / 4** |

Raw lane couples all four. Assisted lane decouples formatting from the model, revealing that 2B's task-reasoning is better than the raw number showed.

## Flask vs fresh — why one task resolved and the other didn't

- Fresh task: `pi * (2r)^2 → pi * r^2`. Three-token change, well within 2B's math-reasoning envelope.
- Flask task: understand Blueprint `__init__`, find the right insertion point, add a NEW `ValueError` without regressing the existing `if "." in name` check. This is multi-step code reasoning on real repo code. 2B produced a syntactically valid diff that did the WRONG semantic thing (inserted a redundant dot check).

Honest reading: the assisted lane removes the FORMATTING blocker; it does not increase CAPABILITY. Where 2B has the capability (simple math bug), the assisted lane succeeds. Where it doesn't (real repo code), it still fails.

## Costs

| Lane | Mean tokens | Mean gen time | Prompt length |
|---|---|---|---|
| raw (flask) | 636 | 6.6 s | ~5000 chars |
| assisted (flask) | ~850 | ~9 s | ~5600 chars |
| raw (fresh) | ~120 | ~1.5 s | ~1200 chars |
| assisted (fresh) | ~250 | ~3 s | ~1700 chars |

Assisted adds ~500 chars of prompt (the JSON schema) and typically ~30-40% more output tokens (the model writes payload content more verbosely than a unified diff).

## Scope

- Two tasks (one SWE-bench Verified, one self-contained synthetic). **4 attempts × 2 lanes × 2 tasks = 16 recorded trials.**
- Fresh task is transparently labeled: NOT from the SWE-bench Verified manifest. We do NOT claim 2B passes SWE-bench Verified.
- 1 / 40 Verified instances for the CODING track. 0 tasks resolved on real Verified instances.

## Next single 2B graduation blocker (UNCHANGED)

**Stage 7M OFFSEC: CWE-311 → CWE-327 confusion (28 / 200 held-out trials).**

The P1b lane-separation finding does NOT graduate the coding track, nor does it override the OFFSEC blocker. It records a HARNESS-LEVEL improvement path (always use assisted edit-specs when you want to decouple formatting from reasoning in a 2B class model) and a first end-to-end resolution on a simple fresh task.

## Reproduce

```bash
# scoring is embedded in the attempts.jsonl; no separate analyze step
cat docs/dyno/coding/p1b/attempts.jsonl | jq .status
cat docs/dyno/coding/p1b_fresh/attempts.jsonl | jq '{lane, resolved, status}'
```
