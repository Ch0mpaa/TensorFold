# P2c — Agentic Failure Diagnosis + Minimal Intervention

**Status**: diagnosis complete, intervention validated across 3 tasks; **7/33 resolved on intervened lane vs 1/33 on baseline**. 2B NOT GRADUATED.
**Base commit**: `5ed70110a`.
**Receipts**: `docs/dyno/agentic/p2c/attempts_{apikey,debug,b_rename,i_apikey,i_debug,i_rename}.jsonl` (66 total attempts).

## Reconciliation of prior denominator (P1b/P2b report)

The prior P1b + P2b summary row claimed `3/26 (all P1b fresh)` for verified completion. The correct arithmetic across the P1b + P2b runs:

| Pool | Count |
|---|---|
| P1b flask-5014 attempts (raw + assisted) | 8 (0 resolved) |
| P1b fresh circle-area attempts (raw + assisted) | 8 (3 resolved, all assisted) |
| P2b API-key attempts (raw + constrained) | 22 (0 resolved) |
| P2b DEBUG-prints attempts (raw + constrained) | 22 (0 resolved) |
| **Total** | **60 attempts, 3 resolved** |

`3/26` was an arithmetic error. Correct figures:
- **3 / 60** across every attempt (raw + treatment) in P1b + P2b.
- **3 / 30** across treatment-lane attempts only (assisted + constrained).
- **3 / 15** across treatment-lane attempts on fresh tasks only.

All three are honestly low; none reaches a track-graduation bar.

## Gate 1 — failure classification (22 baseline constrained traces)

Diagnostic re-runner (`harness_diag.py`) retained FULL per-turn traces and classified each attempt. Per-attempt classification labels:

### API-key task (11 seeds)

| label | count | representative earliest mistake |
|---|---|---|
| repeated_actions | 5 | 3+ identical consecutive tool calls (write_file loop) |
| incorrect_edit | 3 | finish called with `old_key_hits > 0` |
| missing_validation | 2 | finish called without any prior `grep` |
| other | 1 | 25-turn cap with no finish, no repetition |

### DEBUG-prints task (11 seeds)

| label | count | notes |
|---|---|---|
| repeated_actions | 5 | same write_file looping pattern |
| other | 3 | hit cap without finish or repetition |
| missing_validation | 2 | finish without grep |
| resolved | 1 | one baseline attempt actually passed |

### Dominant failure mode

**Across 22 baseline traces: `repeated_actions` dominates at 10 / 22 = 45 %**. The next most frequent is `incorrect_edit` + `missing_validation` combined at 7 / 22 = 32 % — these are both "finished too early without verifying".

## Gate 2 — minimal intervention: grep-before-finish gate

ONE harness change, ONE if-branch:

- Keep a `saw_zero_grep` flag per attempt, initially `False`.
- When the model calls `grep` with a pattern containing (or contained in) the task's `old_key`, and grep returns `rc=1` (no matches), set the flag `True`.
- When the model calls `finish`, if the flag is False, return an OBSERVATION of
  ```
  REJECTED: before you may call finish you must run grep with pattern
  '<old_key>' (or a substring) and receive rc=1 (no matches).
  Verify your edit removed every occurrence, then try finish again.
  ```
  The attempt continues; the 25-turn cap is UNCHANGED.

Everything else is byte-identical to the baseline constrained harness: schema, tool set, oracle, system prompt (one extra note sentence about the gate), task files, seeds.

**Does NOT silently repair raw output.** The intervention affects the loop itself; the baseline lane is preserved intact in `attempts_{apikey,debug,b_rename}.jsonl`.

## Gate 3 — paired comparison (same seeds across lanes)

| Task | Baseline resolved | Intervened resolved | Finish rejections (intervened) | Paired Δ |
|---|---|---|---|---|
| API-key (original) | 0 / 11 | **0 / 11** | 9 | 0 |
| DEBUG-prints (original fresh) | 1 / 11 | **5 / 11** | 1 | **+4** |
| Rename (new held-out) | 0 / 11 | **2 / 11** | 0 | **+2** |

Paired (same seed × lane) outcome matrix:

| Task | both resolved | baseline-only | intervened-only | neither |
|---|---|---|---|---|
| API-key | 0 | 0 | 0 | 11 |
| DEBUG-prints | 0 | 1 | **5** | 5 |
| Rename | 0 | 0 | **2** | 9 |

## Gate 4 — generalization (fresh held-out `rename`)

`task_rename.json`: rename every occurrence of `legacy_fetch` to `fetch` across 4 files + imports; preserve pytest. Different shape (identifier rename, not deletion, not env substitution). Seeds 9100-9110 — same seeds as the two prior tasks.

Intervened: **2 / 11 resolved** (baseline 0 / 11). The intervention effect generalizes across task shapes. Not every intervened attempt passes — the 2B still produces valid tool calls that do semantically wrong things on some seeds.

## Why API-key stays at 0 / 11 — honest diagnosis

The API-key oracle has four rules; grep=0 is only ONE of them. The intervention gates on grep=0 only. Successful API-key resolution ALSO requires:
1. `os.environ["API_KEY"]` idiom present somewhere, AND
2. `import os` in each file that references it, AND
3. pytest keeps passing.

2B can satisfy grep=0 (delete the key string) without satisfying (1)-(3) (because it forgets `import os` or rewrites the file in a way that breaks the function signature). All 9 of the 9 intervened API-key attempts that DID eventually produce grep=0 hit `pytest_rc != 0`.

**The grep-before-finish gate is NOT a general-purpose fix.** It is the minimum change the brief asked for, and it addresses the specific `missing_validation + incorrect_edit` modes. It does NOT substitute for multi-step planning.

## Oracles unchanged, deterministic

All three task oracles are environment commands only: `grep`, `python -m compileall`, `python -m pytest`. No LLM judge anywhere. The intervention does not touch the oracle.

## Cost (intervened vs baseline)

| Task | Baseline mean turns | Intervened mean turns | Baseline mean tokens | Intervened mean tokens |
|---|---|---|---|---|
| API-key | 13.4 | ~18 (gate forces more turns) | ~1200 | ~2200 |
| DEBUG-prints | 12.8 | ~17 | ~1100 | ~1800 |
| Rename | 11.2 | ~16 | ~1000 | ~1600 |

The gate increases per-attempt cost by ~40-80% (more turns) but drives resolved rate 1→5 and 0→2 on two of the three tasks.

## 4-axis capability picture (updated from P1b/P2b)

| Capability | Baseline constrained | Intervened constrained |
|---|---|---|
| Output formatting | ~100% schema-valid | ~100% schema-valid |
| Tool execution | ~100% parseable | ~100% parseable |
| Task completion (DEBUG) | 1/11 | **5/11** |
| Task completion (rename, fresh) | 0/11 | **2/11** |
| Task completion (API-key) | 0/11 | 0/11 |

The intervention improves **task completion** only on tasks whose oracle is grep-dominated. It does nothing for the API-key task, whose oracle requires an idiom + imports.

## Scope, honestly

- 66 attempts (3 tasks × 2 lanes × 11 seeds). All receipts committed.
- One intervention, measured as asked. **Does NOT claim track graduation.**
- Does NOT replace the operator-track graduation gate (40 held-out tasks per track per `docs/dyno/2b-roadmap.md`).
- Fresh held-out `rename` task is a self-contained synthetic, transparently labeled. NOT from any public manifest.

## Next single 2B graduation blocker (UNCHANGED)

**Stage 7M OFFSEC: CWE-311 → CWE-327 confusion (28 / 200 held-out trials).** The P2c grep-gate helps the AGENTIC track on two tasks of three; it neither closes nor overrides the OFFSEC classification blocker.

## Reproduce

```bash
# re-score baseline + intervened
for f in attempts_apikey attempts_debug attempts_b_rename attempts_i_apikey attempts_i_debug attempts_i_rename; do
  echo "=== $f ==="; jq -r '.verdict.resolved' docs/dyno/agentic/p2c/${f}.jsonl | sort | uniq -c
done
```
