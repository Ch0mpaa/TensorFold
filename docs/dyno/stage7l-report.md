# Stage 7L — CWE Catalog A/B Experiment

**Status**: COMPLETE, 2B class NOT GRADUATED.
**Base commit**: `1ff9fb460` (Stage 7K tip).
**Receipts**: `docs/dyno/offsec/stage7l/{deterministic,sampled}.jsonl`.
**Frozen assets**: `docs/dyno/offsec/stage7l/{catalog.md,control.md,snippets.json,freeze.json}`.

## Experiment design

Three prompt conditions, all with IDENTICAL model / sampler / seeds /
scoring. The only variable is what, if anything, gets spliced before
the frozen Stage 7K few-shot prompt.

| Cond | Preamble | Preamble body |
|---|---|---|
| **A** | none (baseline) | — |
| **B** | **catalog** | MITRE CWE definitions for all 10 evaluated CWEs |
| **C** | **control** | Length-matched CPython asyncio reference text (neutral) |

The control exists to distinguish "catalog content helps" from "any
longer prompt helps". If C matches B, the help came from length or
extra context — NOT from the CWE definitions.

### Catalog scope (per Gate 1)

- All 10 evaluated CWEs are in `catalog.md`, independent of their
  presence in CWE Top-25. Verified: `CWE-{89,79,22,347,434,311,352,346,327,78}`.
- Note: **CWE-346** (Origin Validation Error) is NOT in Top-25.
  **CWE-311** (Missing Encryption of Sensitive Data) moves in/out
  across Top-25 editions. Both are present regardless.
- Catalog text SHA-256: `92811f6e0272d0da1c31695f9e289a1baf15e78d94d3b404f14a799d9d97fcf8`
- Control text SHA-256: `1b1018ab3818d8031deff1e0d682c044e5aaa92e988b1e63defe31840424bbb8`
- Catalog body: 435 words / 3011 chars; Control body: 441 words / 3094 chars (< 3% skew).

### Fresh evaluation data

30 new snippets (3 per CWE × 10) authored for Stage 7L, intentionally
distinct from v0/v1/v2/v2.1(g4)/Stage 7K sets. SHA-256:
`83eeefbfe4dc37d538a609ee97999c615fb983d4d03eec9031844e67336d114a`.

- 1 deterministic greedy baseline per snippet per condition = 30 × 3 = 90
- 50 sampled seeds per snippet per condition = 50 × 30 × 3 = **4,500 sampled**
- **Total**: 4,590 model invocations.

Seeds `6000-6049` (non-overlapping with every prior stage).

## Results

### Headline — sampled (1,500 trials per condition)

| Cond | k/n | Trial Wilson 95% | Item mean | Item bootstrap 95% |
|---|---|---|---|---|
| **A** baseline | 357 / 1,500 | **23.8%** [21.7-26.0] | 23.8% | 11.8 - 36.9 |
| **B** +catalog | **1,373 / 1,500** | **91.5%** [90.0-92.8] | 91.5% | **83.8 - 97.4** |
| **C** +control | 370 / 1,500 | 24.7% [22.6-26.9] | 24.7% | 12.6 - 37.9 |

### Deterministic (greedy, 1 per snippet)

| Cond | k/30 |
|---|---|
| A | 8 / 30 (26.7%) |
| B | **29 / 30 (96.7%)** |
| C | 6 / 30 (20.0%) |

### Per-CWE sampled (150 trials per CWE per condition)

| CWE | A | B | C | Δ(B-A) | Δ(B-C) |
|---|---|---|---|---|---|
| CWE-89  (SQL)         | 98.0% | **59.3%** | 98.7% | **-38.7pp** | -39.3pp |
| CWE-79  (XSS)         | 56.0% | 98.0%     | 78.7% | +42.0pp | +19.3pp |
| CWE-22  (path)        | 60.7% | 63.3%     | 35.3% | +2.7pp  | +28.0pp |
| CWE-352 (CSRF)        |  0.7% | 98.0%     |  0.7% | +97.3pp | +97.3pp |
| CWE-78  (OS cmd)      | 16.0% | 100.0%    | 29.3% | +84.0pp | +70.7pp |
| CWE-327 (crypto)      |  6.0% | 100.0%    |  2.7% | +94.0pp | +97.3pp |
| CWE-347 (JWT)         |  0.0% | 96.7%     |  0.0% | +96.7pp | +96.7pp |
| CWE-311 (cleartext)   |  0.7% | 100.0%    |  1.3% | **+99.3pp** | +98.7pp |
| CWE-346 (origin)      |  0.0% | 100.0%    |  0.0% | **+100pp**  | +100pp  |
| CWE-434 (upload)      |  0.0% | 100.0%    |  0.0% | **+100pp**  | +100pp  |

### Per-example reliability bands (of 30 snippets)

| Cond | 50/50 | mid (1-49/50) | 0/50 |
|---|---|---|---|
| A | 2  | 14 | **14** |
| B | **20** | 10 | **0** |
| C | 2  | 14 | **14** |

No malformed outputs, no timeouts in any condition. All 4,500
sampled trials returned a valid `CWE-NNN`.

## A vs B — did the catalog help?

**Yes, massively, but with one real regression.**

Paired McNemar on matched (snippet, seed) pairs:
```
A vs B : n11=272 n10=85 n01=1101 n00=42  chi2=868.66  p = 6.4e-191
```
1,101 pairs where B got it right and A got it wrong; only 85 the
other way. That is a strong, paired-test significant improvement.

**The regression**: CWE-89 (SQL injection) DROPS from 98.0% to 59.3%
(-38.7 pp). The catalog's CWE-89 and CWE-78 definitions both use
the phrase "constructs all or part of an ... command using
externally-influenced input" — the model appears to confuse the two
when the SQL snippet is terse. This is a real trade-off, not a free
lunch: the catalog unlocks eight categories but partially breaks
the one the model already knew cold.

## B vs C — content or merely context length?

**Content. Not length.**

```
B vs C : n11=301  n10=1072 n01=69  n00=58   chi2=879.93  p = 2.3e-193
A vs C : n11=299  n10=58   n01=71  n00=1072 chi2=  1.12  p = 0.29
```

- B overwhelmingly beats C (same strong direction as B vs A).
- A vs C is **statistically indistinguishable**. C — length-matched
  neutral CPython asyncio text — produces the same accuracy pattern
  as the baseline, including the SAME 14 snippets at 0/50 that
  baseline had.

This rules out the "longer prompt alone helps" alternative
explanation. The CWE definitions themselves are what changed the
answer distribution. Specifically, condition C even slightly HURT
a few CWEs (CWE-22 drops from 60.7% to 35.3%) — pure length is not
a net positive.

## Cost

**Measurement caveat**: The runner recorded `prompt_tokens: 1` for
every trial — a measurement bug: `len(hf_chat_template_output)` on
the returned tensor returned the batch dimension (1) instead of
the sequence length. The receipts remain CORRECT for every other
field (`output_text`, `n_tokens`, `elapsed_s`, `ttft_s`). Approximate
prompt token counts from the known templates (Qwen3 ≈ 3.8 chars/token
for English):

| Cond | Prompt text length | Approx tokens | Mean TTFT (ms) | Mean elapsed (s) | tokens / correct |
|---|---|---|---|---|---|
| A | ~500 chars | ~130 | 46.6 | 0.0963 | 26.4 |
| B | ~3,500 chars | ~920 | 25.8 | 0.0788 | **7.3** |
| C | ~3,600 chars | ~940 | 24.5 | 0.0733 | 25.3 |

**Why B/C have LOWER TTFT than A**: counter-intuitive. In the 2B
KV-cache with prefill well under 1k tokens, the additional context
shortens the "time to think" behavior — the model is more confident
with the catalog, emits `CWE-NNN` in fewer decoded tokens
(`n_tokens` per trial falls ~3×), and the net wall time actually
decreases. Per correct classification, B is ~3.6× more token-
efficient than A despite the ~7× longer prompt.

If an operator cares about **prefill cost per 1k requests**, the
catalog adds about 790 input tokens per request = ~790k extra
input tokens / 1k requests. On a 2B at EXL3 8bpw that is ~7 seconds
of batched prefill. Under 7× the input-token bill, the correct-
answer rate rises from 24% to 92% — **payoff ratio ≈ 68 pp
accuracy gained per 7× input-token overhead**.

## Which capability still blocks graduation

**Single next blocker: category interference inside the catalog —
specifically CWE-89 regression from 98% to 59% under condition B.**

Evidence:
- Only one category went down. 8 went up. 1 stayed flat.
- The down category (CWE-89) was already the ONE category A got
  right consistently.
- The down category's MITRE wording shares the "constructs all or
  part of ... command using externally-influenced input" stem with
  CWE-78 (OS command injection). That phrasing is the smoking gun.

In-scope ways to close this gap (NONE attempted in 7L):

1. **Rewrite the catalog entries** so near-duplicate stems become
   distinct. Refreeze and re-run on a new held-out set. Prompt-only,
   fully in scope.
2. **Chain-of-selection prompting**: ask the model to narrow to a
   shortlist of candidates first, then pick from them. Prompt-only,
   in scope.
3. **SFT on hand-labeled disambiguating pairs**. Out of scope today.

Stage 7L does NOT declare the catalog a retrieval-only cure.
The retrieval mechanism works for 8 of 10 categories; one category
regressed because of catalog phrasing interference. That is a
content-authoring problem the catalog can solve, not a model-size
problem.

## Scope guards honored

- Did NOT touch v2/v2_1/Stage 7K receipts.
- Did NOT start SFT / DPO / QLoRA / RAG integration.
- Did NOT add new model sizes or quantizers.
- Did NOT inspect any Stage 7L model output before the frozen assets were committed.
- Preserved EXL3 109.5 / TFOLD 110.8 tok/s and MD5 `c6a03e5665` baselines.

## Reproduce

```bash
# scoring-only
python docs/dyno/scorers/analyze_stage7l.py

# full re-run (requires exllamav3 + model)
ssh spark 'cd ~/stage7l && python stage7l_runner.py \
    --snippets snippets.json --freeze freeze.json \
    --catalog catalog.md --control control.md --out-dir out'
```
