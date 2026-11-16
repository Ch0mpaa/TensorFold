# Stage 7K — Fresh Held-Out Generalization Benchmark

**Status**: COMPLETE, 2B class NOT GRADUATED.
**Base commit**: `ab008039e` (Stage 7J tip).
**Receipts**: `docs/dyno/offsec/stage7k/{deterministic,sampled}.jsonl`.
**Frozen config**: `docs/dyno/offsec/stage7k/freeze.json`.
**Snippets**: `docs/dyno/offsec/stage7k/snippets.json` (30 fresh, 3 per CWE × 10).

## Why Stage 7K

Stage 7J's held-out set had only 7 distinct examples. The winning
condition (`few_single`, 20.9% held-out) had a 41-percentage-point
bootstrap CI (2.6-44.0). Before extending the Dyno or starting training,
we need to see whether the point estimate reflects anything real.

Stage 7K adds **30 genuinely fresh examples** — 3 per CWE across the
same 10 categories, deliberately varying language stack, framework
surface, variable identifiers, wording, and framing from v0/v1/v2/v2.1
fixtures. The prompt template, model revision, sampler, EOS IDs, and
scoring rules were ALL frozen before any of these snippets were
evaluated. We ran:

1. **One deterministic baseline** per snippet (greedy, argmax at each step).
2. **50 recorded sampling seeds** per snippet (same `DefaultSampler()` as Stage 7J).

Total: **1,530 model invocations** across one frozen condition. No
prompt tuning on Stage 7K data.

## What stayed frozen

| Thing | Value |
|---|---|
| Model repo | `UnstableLlama/Qwen3.5-2B-exl3-8.00bpw` |
| Revision SHA-256 | `a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f` |
| Runtime | `exllamav3 1.6.7` |
| Context | 4096 tokens |
| EOS | `<|im_end|>=248046`, `<|endoftext|>=248044` |
| Prompt | `few_single` 3-shot (ex: CWE-89, 79, 22 only — no label leak) |
| Deterministic sampler | `GreedySampler()` |
| Sampling sampler | `DefaultSampler()` — identical ctor to v2 |
| Seeds | `range(5000, 5050)` — non-overlapping w/ v0/v1/v2/v2.1/g4 |
| Timeout | 30 s per trial |
| Max new tokens | 100 per trial |
| Scorer | `scorer_v2b` (no regex; 100% agreement w/ scorer_v2a on 1,100 v2.1 trials) |

## Results

### Headline (sampled, 1,500 trials)

| Metric | Value |
|---|---|
| Trial-level accuracy | 380 / 1,500 = **25.3%** (Wilson 95% CI 23.2 - 27.6) |
| **Item-level (30 examples, bootstrap)** | **25.3%** (CI **13.1 - 38.9**) |
| Macro over 10 CWEs (bootstrap) | 25.3% (CI 7.9 - 46.9) |
| Malformed outputs | 0 of 1,500 (frozen prompt always emits `CWE-NNN`) |
| Timeouts | 0 of 1,500 |
| Wrong-CWE | 1,120 / 1,500 (74.7%) |
| Runtime | 144.8 s total / 9,529 tokens / 6.4 tok mean per trial |

### Deterministic baseline (greedy, 30 trials)

9 / 30 = **30.0%** (Wilson 16.7 - 47.9). Comparable to the sampled mean;
the frozen few-shot prompt produces a very peaked distribution per
snippet, so sampling and greedy agree on most inputs.

### Per-CWE sampled (150 trials each = 3 snippets × 50 seeds)

| CWE | correct / 150 | acc | Wilson 95% CI |
|---|---|---|---|
| CWE-89  (SQL injection)       | 148 / 150 | **98.7%** | 95.3 - 99.6 |
| CWE-79  (XSS)                 |  77 / 150 | 51.3%    | 43.4 - 59.2 |
| CWE-22  (Path traversal)      |  76 / 150 | 50.7%    | 42.7 - 58.6 |
| CWE-352 (CSRF)                |  41 / 150 | 27.3%    | 20.8 - 35.0 |
| CWE-78  (OS command injection)|  34 / 150 | 22.7%    | 16.7 - 30.0 |
| CWE-327 (Weak crypto)         |   3 / 150 |  2.0%    |  0.7 - 5.7 |
| CWE-347 (JWT signature)       |   1 / 150 |  0.7%    |  0.1 - 3.7 |
| CWE-311 (Cleartext)           |   0 / 150 | **0.0%** |  0.0 - 2.5 |
| CWE-346 (Origin validation)   |   0 / 150 | **0.0%** |  0.0 - 2.5 |
| CWE-434 (Upload)              |   0 / 150 | **0.0%** |  0.0 - 2.5 |

### Per-example success rate (sampled)

Highly bimodal. Among the 30 fresh snippets:

- **12 snippets at >= 40/50** (most of CWE-89, two CWE-79, two CWE-22, one CWE-352, one CWE-78)
- **13 snippets at 0/50** (all CWE-311, CWE-346, CWE-434; two CWE-327; two CWE-347; one CWE-352)
- **5 snippets in the middle** (8-24 of 50)

The overall 25.3% is driven by a handful of snippets that are almost
ALWAYS right, not by partial competence across the pool. The model is
confidently correct on its strong CWEs and confidently wrong on its
weak ones — sampling does not rescue the weak ones.

## Comparison to Stage 7J few_single held-out

Stage 7J header: **20.9%** (Wilson 16.9-25.4, bootstrap over 7 snippets 2.6-44.0).
Stage 7K header: **25.3%** (Wilson 23.2-27.6, bootstrap over 30 snippets 13.1-38.9).

**Honest interpretation**: Stage 7K's item-level CI (13.1-38.9) FULLY
CONTAINS Stage 7J's point estimate (20.9%), and the two bootstrap CIs
overlap by ~26 percentage points. The 4.4-point jump in point estimate
is NOT statistically distinguishable from no change. We refuse to claim
improvement.

**What the comparison DOES tell us, with higher confidence**:

1. The ~20-25% held-out range reproduces on 30 fresh examples. The
   Stage 7J estimate was not an artifact of its 7-snippet pool.
2. The item-level CI tightened from 41.4pp (7 snippets) to 25.8pp
   (30 snippets) — an evidence upgrade, not a score upgrade.
3. The categorical structure is stable across Stage 7J and 7K:
   CWE-89 wins by a wide margin, CWE-311 / CWE-346 / CWE-434 /
   CWE-347 stay at zero, the rest in the middle.

## Why evaluation design matters

1. **Dev/held-out separation.** Stage 7J (v2) averaged dev and held-out
   together and reported 26.2%. The split showed the real held-out was
   20.9% and three other conditions were at chance.
2. **Item-level vs trial-level CIs.** 1,500 trials on 30 snippets is
   NOT 1,500 independent experiments. The honest CI is over the 30
   example accuracies, not over the 1,500 token streams.
3. **Freezing BEFORE evaluation.** The classical p-hacking failure
   mode is to tweak the prompt after the first bad result. Stage 7K
   commits the freeze JSON before the model is invoked.

## Which capability still blocks graduation

**Single next blocker: four CWE categories at or near 0% held-out**
(`CWE-311`, `CWE-346`, `CWE-434`, `CWE-347`).

Evidence:
- Every Stage 7K snippet in CWE-311 / CWE-346 / CWE-434 scored 0 / 50.
- CWE-347 scored 1 / 150.
- Deterministic greedy also hit 0 / 3 on each.
- These are the SAME categories that g4 (Stage 7J) showed the model
  COULD answer at 90-100% WHEN a textbook preamble listed the CWE
  definitions in context. So the model has the semantic knowledge of
  the vulnerabilities; it cannot retrieve the correct identifier under
  the few_single prompt the rest of the benchmark uses.

This is a **retrieval / ID-recall gap**, not a semantic gap. The
two in-scope ways to close it, in order of complexity:

1. **Change the in-prompt CWE catalog.** Expand the frozen prompt to
   include a 25-item CWE-Top-25 definition block (a RAG-lite style:
   paste the catalog in system text). This is a prompt change and
   would require a new frozen evaluation on a new held-out set.
2. **SFT on CWE definitions.** If prompt-level retrieval is not an
   acceptable long-term answer, supervise fine-tuning on the CWE-25
   definitions themselves. Out of scope today.

Neither path is attempted in Stage 7K. The blocker is *named*, not
*fixed*.

## Reproduce

```bash
# scoring-only: uses committed receipts
python docs/dyno/scorers/analyze_stage7k.py

# full re-run (requires exllamav3 + model):
ssh spark 'cd ~/stage7k && python stage7k_runner.py \
    --snippets snippets.json --freeze freeze.json --out-dir out'
```

Identical seeds → identical receipts (deterministic sampler is
seed-free; sampled sampler is seed-driven through exllamav3's
`Job(seed=)`).
