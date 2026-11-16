# P2 — Agentic API-key replacement pilot

**Status**: harness functional, oracle verified, **0 / 11 attempts resolved**. 2B class NOT GRADUATED.
**Receipts**: `docs/dyno/agentic/p2/{attempts.jsonl, analysis.json}`
**Frozen assets**: `docs/dyno/agentic/p2/{task.json, freeze.json, harness.py, analyze.py}`

## Task

Replace a hardcoded API key (`sk-prod-ABCDEF12345`) in a seeded 5-file mini
repository with `os.environ["API_KEY"]` (add `import os` where needed).
Repository must still import, syntax-check, and `pytest -q` must exit 0.

## Harness

ReAct loop:
- Fresh workspace, seeded from `task.json#seed_repo_files`, per attempt.
- System prompt lists five tools: `shell`, `read_file`, `write_file`, `grep`, `finish`.
- Each turn: model emits one `TOOL: {"tool": ..., "args": ...}` line; harness
  parses, executes, feeds back `OBSERVATION: {...}`.
- 25-turn cap. 30s per-turn generation budget. 400 max_new_tokens per turn.
- `pip`, `git`, `curl`, `wget`, `ssh`, `nc` are REFUSED (no network).

### Oracle (environment-run, no LLM judgment)

1. `grep 'sk-prod-ABCDEF12345' -r .` returns 0 hits.
2. `os.environ["API_KEY"]` or `os.environ['API_KEY']` appears at least once.
3. `python -m compileall -q .` exits 0.
4. `pytest -q` with `API_KEY=dummy-value-for-tests` exits 0.

Resolved = ALL FOUR pass.

## Results

| Metric | Value |
|---|---|
| Attempts | 11 (1 greedy + 10 sampled) |
| **Resolved** | **0 / 11** |
| Model emitted `finish` tool | 2 / 11 |
| Mean turns used | 24.1 / 25 (nearly always hit the cap) |
| Mean tokens generated | 752.9 |
| Mean wall time per attempt | 8.6 s |

### Per-attempt oracle breakdown

| mode | seed | turns | finish? | old_key_hits | idiom | compile | pytest | resolved |
|---|---|---|---|---|---|---|---|---|
| greedy | 9000 | 25 | ❌ | 2 | ❌ | ✅ | ✅ | ❌ |
| sampled | 9001-9010 | 16-25 | 2 x ✅ | 2 | ❌ | ✅ | ✅ | ❌ |

`pytest` passed on EVERY attempt because the existing tests don't reference
the key. `compileall` passed because no file was mutated. The failing checks
are `old_key_hits > 0` (model never successfully wrote a replacement file)
and `idiom_present = False` (ditto).

### Tool-call distribution (all 11 attempts combined)

| Tool | calls |
|---|---|
| shell | 28 |
| read_file | 4 |
| grep | 2 |
| finish | 2 |
| **write_file** | **0** |

Across 275 total turn-slots (11 × 25), only **36 turns produced a parseable
TOOL call** (13 %). The other 239 turns were `PARSE_ERROR: no TOOL: line`.

## Failure analysis

The **root cause** is model protocol-adherence failure, not harness bugs.
Looking at an actual high-token attempt (seed 9010):

```
turn 0  MODEL: TOOL: read_file args {"path": "src/x.py"}       OBS: PARSE_ERROR
turn 1  MODEL: TOOL: read_file args {"path": "src/x.py"}       OBS: PARSE_ERROR
turn 2  MODEL: I see the issue. The previous attempt to "read" the file...
turn 3  MODEL: The system is treating my responses as tool calls...
turn 4  MODEL: TOOL: write_file args {"path": "src/x.py", "content": "..."
turn 5  MODEL: TOOL: write_file args {"path": "src/x.py", "content": "..."
... (all turns 4-24 repeat the same malformed line)
```

The specified protocol was `TOOL: {"tool": "read_file", "args": {"path": "..."}}`
(JSON object directly after the colon). The 2B model consistently emitted
`TOOL: read_file args {...}` (non-JSON). The parser correctly rejected these.
Two attempts (seeds 9004, 9008) eventually emitted `TOOL: {"tool": "finish",
"args": {}}` — but by then, no `write_file` had succeeded.

### What did make it through the parser

- **28 shell calls**: mostly `ls -la`, `cat src/client.py` — harmless inspection.
- **4 read_file**: once the model adapted.
- **2 grep**: searching for the old key.
- **0 write_file** across 11 attempts of 25 turns each. The model never
  produced a correctly-formed `write_file` TOOL call.

## Cost

| Metric | Value |
|---|---|
| Total model generation time | 92.0 s |
| Total tokens generated | 8,282 |
| Total wall time | ~95 s |

## Harness verification (no false negatives from harness bugs)

- Baseline check: seeded workspace reproduces the exact file bytes from
  `task.json` every attempt; verified by `grep old_key` returning 2 hits.
- Oracle sanity: manually ran `pytest` + `grep` + `compileall` against the
  seeded workspace outside the harness; all behave as documented.
- Tool parse: tested known-good `TOOL: {"tool":"grep","args":{"pattern":"x"}}`
  outside the model loop → parses and runs correctly.
- Workspace isolation: each attempt uses its own directory (`attempt_<mode>_<seed>/`);
  state does not leak between attempts.

## What the harness does NOT yet do

- No tool-call retry prompt. The brief did not specify one; we recorded
  honestly instead. A Stage-7N-equivalent experiment could add a stricter
  wrapper prompt that shows one correct example per turn.
- No partial-credit score. Resolved is binary. A richer metric (eg
  parse-rate, correct-grep-rate) is in `analysis.json` but not in the
  headline number, per the brief.

## Comparison to roadmap's AGENTIC gate

Roadmap required: "verified task completion; **No LLM judgment on success/fail**;
the environment confirms or denies." Oracle uses `grep`, `compileall`, `pytest`
— all three are environment verifications. **0 / 11 attempts pass the oracle**;
the single-task result is honestly reported.

## Remaining work before the track can be declared

- Harness end-to-end: **PASSES** (seed, loop, parse, oracle all work).
- 2B on this one task: **0 / 11**.
- Needs 40 held-out tasks per the roadmap graduation gate. 1 / 40 tasks
  evaluated so far.
- The 2B model needs either a more permissive tool-call protocol, or a
  one-shot example-in-prompt that demonstrates the exact JSON format.
  That is tooling / prompt work and is explicitly OUT OF SCOPE for P2.

## Next single 2B graduation blocker (still unchanged)

**Stage 7M OFFSEC blocker remains the #1 item: CWE-311 → CWE-327 confusion
(28 / 200 trials, Stage 7M condition B).**

The P1 `patch_apply_failed` and P2 `PARSE_ERROR` pathologies are
*track-specific* pilot-scaffold findings. They demonstrate the respective
harnesses work and name concrete model pathologies to attack in later stages,
but they do not graduate or override the pre-existing OFFSEC blocker.

## Reproduce

```bash
python docs/dyno/agentic/p2/analyze.py

ssh spark 'cd ~/p2 && python dyno/harness.py \
    --task dyno/task.json \
    --workspace ~/p2/attempts \
    --out ~/p2/attempts.jsonl'
```
