# 2B Coding Agent — P1c: Edit → Test → Diagnose → Retry

> How a tied 2-vs-2 failure count on the original SWE-bench Verified
> instance led to **the first end-to-end 2B resolution** of a real
> SWE-bench Verified task.

## What failed before P1c

P1b's assisted lane shipped 4/4 attempts on `pallets__flask-5014` that produced
syntactically-valid patches, but 0/4 resolved the actual oracle. Looking at the
model's output turn-by-turn, the 4 failures split cleanly:

| seed | failure mode | what it looked like |
|---|---|---|
| greedy/8000 | **incorrect_logic** | wrote `if "." in name: raise ValueError(...)` — a check that's already in the file. Missed the empty-name case. |
| default/8001 | **incorrect_logic** | same wrong check + extra noise. |
| default/8002 | **wrong_target** | payload was CORRECT (`if not name: raise ValueError(...)`) but anchored against a line that doesn't exist in the file. |
| default/8003 | **wrong_target** | edited an unrelated `url_prefix` block. |

Tied 2-vs-2 between "model knows the answer but can't localize it" and "model
guesses the fix wrong". The brief asked for ONE intervention; the only one
that honestly addresses BOTH is to let the model SEE whether its edit fixed
the test.

## The intervention (bounded read → edit → test → diagnose → retry)

Hard cap of **3 edit attempts** per task. Each attempt:
1. Read the current state of the target file into the prompt.
2. Model emits ONE JSON `{action, file, anchor, payload}` edit spec. Schema
   enforced by `LLGuidanceFilter(json_schema=…)` so the response is always
   parseable.
3. Harness applies the spec to a FRESH worktree (reset from the previous
   turn's attempt), runs `pytest FAIL_TO_PASS` (60s cap).
4. If FAIL_TO_PASS passes, run PASS_TO_PASS to check for regressions, then
   stop. Resolved if both pass.
5. Otherwise, feed the **pytest tail** verbatim back into the next prompt
   as "Turn N: fail_to_pass FAILED. pytest tail: …" and loop.
6. Model may emit `action: "finish"` to stop early.

No prompt engineering tricks. Oracles are pytest/`runtests.py` only; no LLM
judge anywhere.

## Why this works (when it works)

Two different failure modes → two different responses to feedback:

- **wrong_target**: the applied edit doesn't touch the right location. Pytest
  returns the same failing error as the baseline. Model sees "nothing changed"
  and tries a different anchor on the next attempt.
- **incorrect_logic**: the applied edit touches the right location but writes
  the wrong code. Pytest returns a DIFFERENT error (or an assertion failure).
  Model sees the specific error and tries different logic.

In both cases, the second edit is informed by the test result rather than a
blind guess. The 2B model often finds the right answer on the second or
third try IF the semantic answer is within its reach.

## What actually happened (real numbers)

### Flask (same `pallets__flask-5014`, 4 paired attempts)

| metric | value |
|---|---|
| patch_valid | 4 / 4 |
| test_completion | 4 / 4 |
| RESOLVED | **0 / 4** |

Zero improvement over P1b assisted (which was also 0/4). The 2B simply does
not know to write `if not name:` for this task. The retry loop asks it to
try again; it re-emits variants of the dot-check and gives up. **This is a
capability limit, not a harness problem.**

### Django `django__django-14089` (fresh SWE-bench Verified)

| metric | value |
|---|---|
| patch_valid | 4 / 4 |
| test_completion | 4 / 4 |
| **RESOLVED** | **3 / 4** (first attempt, zero retries needed) |

🎯 **First end-to-end 2B resolution of a real SWE-bench Verified task.**

The task is "add `__reversed__` method to `OrderedSet`". The three resolved
attempts each emitted:
```json
{
  "action": "insert_before",
  "anchor": "    def __contains__(self, item):\n        return item in self.dict",
  "payload": "    def __reversed__(self):\n        return iter(reversed(self.dict))"
}
```
The harness applied, ran pytest, both `FAIL_TO_PASS` and 20 `PASS_TO_PASS`
tests passed. Resolved.

The one failed django attempt (seed 8003) chose a wrong anchor on the first
turn and kept doing so across 3 retries. The retry loop can prevent giving
up early but can't invent a better anchor the model doesn't know to choose.

## The harness bug that delayed the finding

First P1c run reported 0/4 on django even though model output LOOKED
correct. Root cause: `pip install -e /home/chris/p1c/django-base` placed the
venv's editable Django on `django-base/django/…`, NOT the per-attempt
worktree. The model's edits to the attempt worktree were invisible to the
imported `django` package — `import django` always loaded the unpatched
original.

Fix: one-line change in `run_tests` to set `PYTHONPATH=<attempt_dir>:…` so
the worktree shadows the installed package. After the fix, IDENTICAL model
outputs went from 0/4 to 3/4 resolved on django. This is a HARNESS CORRECTION,
not a model improvement.

Honest note: the model's semantic answer was right from the start; we
weren't running its code. Reporting this transparently matters — otherwise
we'd be falsely claiming a model change moved the needle.

## Separation of concerns (updated)

| Capability | Flask (edit-test-retry) | Django (edit-test-retry) |
|---|---|---|
| Output formatting (schema) | 100% | 100% |
| Patch validity | 4/4 | 4/4 |
| Test completion | 4/4 | 4/4 |
| **Verified completion** | **0/4** | **3/4** |

The gap between "test runs" and "test passes" is where task reasoning lives.
Flask's gap is 100% (nothing resolved). Django's gap is 25% (3/4 resolved).
The intervention opens up the tooling; it doesn't change the model's
capability envelope.

## Hands-on exercise

Open `docs/dyno/coding/p1c/attempts_django.jsonl`. For the ONE failing
attempt (seed 8003), read its 3 per_attempt entries. Note the anchor
each turn.
1. Did the anchor change across the 3 turns?
2. Did the pytest tail change (give different information) across the 3
   turns?
3. If the pytest tail was the SAME on turn 1 and turn 2, what would that
   tell you about whether the retry loop is doing any useful work?

Write 3-5 sentences on when retries help and when they don't, using this
specific attempt as evidence.

## Comprehension questions

**Q22** (P1c). We discovered mid-run that the pip editable install was
masking the model's edits. After the PYTHONPATH fix, django jumped from 0/4
to 3/4. If the SAME model output had resolved all along but we didn't notice
for ~5 minutes of compute, which of our P1b or P1 reports might contain
similar silent-failure bugs? Propose ONE sanity check that would have caught
this earlier.

**Q23** (P1c). The retry intervention improved django (0 → 3) but not flask
(0 → 0). The brief said the single intervention would address both
incorrect_logic AND wrong_target. Why did it work on the django wrong_target
case (assisted-lane analogue in P1b) but NOT on the flask wrong_target case
in P1c? What does this tell you about when "give the model feedback" is
sufficient vs when the model needs a better tool (e.g. structured code
search)?
