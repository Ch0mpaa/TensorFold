# P2b — Agentic Action Reliability: Raw vs Constrained lanes

**Status**: harness proves constrained lane works end-to-end; **tool execution unlocked (0% parse-fail vs 89-99%)** on both tasks; task completion still 0/22. 2B class NOT GRADUATED.
**Receipts**: `docs/dyno/agentic/p2b/attempts.jsonl`, `docs/dyno/agentic/p2b_fresh/attempts.jsonl`
**Harness**: `docs/dyno/agentic/p2b/harness_both.py`

## Design

Two lanes, same task, same seeds, independently scored.

| Lane | Protocol | Scoring |
|---|---|---|
| **raw** | Model emits `TOOL: {json}` free-form; `^TOOL:\s*({...})$` regex parse; invalid → `PARSE_ERROR` fed back to model | We do NOT silently repair malformed calls. |
| **constrained** | exllamav3 `LLGuidanceFilter(json_schema=...)` enforces a tool-call schema at decode time; entire assistant turn MUST be a valid JSON object matching `{tool, args}` where `tool ∈ {shell, read_file, write_file, grep, finish}` | |

**Scoring independence**: lanes use the same seeds but different system prompts (raw vs constrained) and different decode paths. We record `parse_failures` per attempt separately from `tool_counts` and `verdict.resolved`.

## Lane-effect on original API-key task

Seeds 9100-9110, 11 attempts per lane (we did not re-use the 11 attempts from P2's prior session — those used seeds 9000-9010 under harness.py).

| Metric | Raw | Constrained |
|---|---|---|
| **Turns with PARSE_ERROR** | 99.3 % | **0.0 %** |
| Model emitted `finish` tool | 0 / 11 | **9 / 11** |
| Tool actions executed (total across 11 attempts) | 2 (2 × read_file) | **112** (54 shell, 32 write_file, 20 read_file, 9 finish, 6 grep → edge-triggered) |
| **Resolved (all 4 oracle checks)** | **0 / 11** | **0 / 11** |
| Mean tokens per attempt | ~130 (parse errors keep turns short) | ~2500 (fuller attempts because tools actually work) |

Dramatic difference in TOOL-EXECUTION rate. Still 0 / 11 resolved on the oracle. Constrained lane proves the model CAN drive a tool loop; it does NOT prove the model can SOLVE the task.

## Fresh held-out task

**Fresh task** (`docs/dyno/agentic/p2b_fresh/task.json`): remove four `print('DEBUG: ...')` statements from a mini-repo across two files, without breaking two existing pytest tests. Different task shape from the API-key one (deletion + preservation vs substitution). Seeds 9100-9110.

| Metric | Raw | Constrained |
|---|---|---|
| **Turns with PARSE_ERROR** | 89.0 % | **1.1 %** |
| Model emitted `finish` tool | 2 / 11 | **10 / 11** |
| Tool actions executed | 28 (14 read, 7 shell, 4 grep, 2 finish, 1 write) | **87** (41 shell, 31 write_file, 10 finish, 5 read_file) |
| **Resolved** | **0 / 11** | **0 / 11** |

The lane effect on **parse rate and tool-execution** generalizes to the fresh task. The lane effect on **task completion** does not — neither lane completes either task end-to-end.

## Separation of concerns (what this experiment measures)

| Capability | Metric | Raw | Constrained |
|---|---|---|---|
| **Output formatting** (can you produce a parseable action?) | 1 - parse-failure-rate | 0.7-11 % | **98.9-100 %** |
| **Tool execution** (can the harness actually run your action?) | tool_counts > 0 | 2-28 total | **87-112 total** |
| **Task reasoning** (do your chosen actions converge on the goal?) | oracle-measurable progress | zero direct state change | state changes in many attempts but never oracle-complete |
| **Verified completion** (does the environment confirm success?) | verdict.resolved | 0 / 22 | **0 / 22** |

Constraint-enforced decoding fixes two of the four capability axes (formatting, tool execution). The remaining two (reasoning, completion) are not addressed by constrained decoding and remain 2B-size-limited.

## Why constrained still fails to complete

Representative pattern from constrained lane: model (a) reads `src/client.py`, (b) emits a `write_file` with a NEW content that includes `os.environ["API_KEY"]`, (c) immediately emits `finish`. But the written file often:
- preserves the OLD `API_KEY = 'sk-prod-...'` line somewhere (model rewrote the top but kept constants);
- or forgets `import os` at the top → `compileall` passes but `pytest` fails at `NameError: os`;
- or writes to one file but forgets the second occurrence in a different file.

The oracle (four strict checks) catches all three. Constrained decoding does not plan multi-file edits; it just produces well-formed single tool calls.

## Costs

| Lane | Mean tokens / attempt | Mean wall time | Notes |
|---|---|---|---|
| raw (original) | ~130 | ~2 s | short because parse errors loop quickly |
| constrained (original) | ~2500 | ~25 s | full 25-turn loops because tools execute |
| raw (fresh) | ~150 | ~2 s | same pattern |
| constrained (fresh) | ~2300 | ~22 s | same pattern |

**Cost warning**: constrained decoding + a real tool loop is ~15× more expensive than raw per attempt in both tokens and time. On a compliance-gated or latency-sensitive production system this ratio matters.

## Scope

- **Two tasks × two lanes × 11 seeds = 44 recorded attempts.** All receipts committed.
- Fresh task is a self-contained mini-repo; transparently labeled.
- 1 / 40 agentic tasks resolved on the AGENTIC track (none, actually). The experiment does NOT graduate the track.

## Next single 2B graduation blocker (UNCHANGED)

**Stage 7M OFFSEC: CWE-311 → CWE-327 confusion.** The P2b separation
between tool-execution and verified-completion refines our understanding of
the AGENTIC track but does not override the OFFSEC blocker.

## Reproduce

```bash
cat docs/dyno/agentic/p2b/attempts.jsonl       | jq '{lane, parse_failures, "resolved": .verdict.resolved}'
cat docs/dyno/agentic/p2b_fresh/attempts.jsonl | jq '{lane, parse_failures, "resolved": .verdict.resolved}'
```
