# Stage 7M — CWE Disambiguation (Mission 1 report)

**Status**: COMPLETE, 2B class NOT GRADUATED.
**Base commit**: `2350367c2` (Stage 7L tip).
**Receipts**: `docs/dyno/offsec/stage7m/{deterministic,sampled}.jsonl`.
**Frozen assets**: `docs/dyno/offsec/stage7m/{catalog_v2.md, snippets.json, freeze.json}`.
**Companion**: `docs/dyno/2b-roadmap.md` (Mission 2).

## Why Stage 7M

Stage 7L lifted held-out accuracy from 23.8% (no catalog) to 91.5%
with a MITRE CWE catalog spliced into the prompt — but CWE-89 (SQL
injection) regressed from 98% to 59.3%. The single next 2B blocker
named at the end of Stage 7L was:

> The catalog's CWE-89 and CWE-78 definitions share the "constructs
> all or part of ... command using externally-influenced input" stem;
> the model conflates them when the SQL snippet is terse.

Stage 7M tests that hypothesis and the fix.

## Confusion evidence (Stage 7L receipts, no new run)

Re-scoring Stage 7L condition B's 150 CWE-89 trials
(`docs/dyno/offsec/stage7l/cwe89_confusion.json`):

| Prediction | n / 150 | % |
|---|---|---|
| CWE-89 (truth) | 89 | 59.3% |
| **CWE-78 (OS cmd)** | **44** | **29.3%** |
| CWE-79 (XSS) | 14 | 9.3% |
| CWE-327 / 347 / 311 | 3 | 2.0% |

**72% of CWE-89 errors go to CWE-78.** Hypothesis confirmed. Under
condition A (no catalog) only 1 of 150 CWE-89 trials picked CWE-78;
under condition C (control, same length) just 1 of 150. The
confusion is catalog-induced, specifically induced by the CWE-78
and CWE-89 entries of that catalog.

## What changed in the catalog (condition B of Stage 7M)

`catalog_v2.md` (SHA-256 `d705df48...`) differs from the Stage 7L
catalog (SHA-256 `92811f6e...`) in EXACTLY two entries:

- **CWE-78** now LEADS with "operating-system shell or subprocess
  API" and names concrete sinks (`os.system`, `subprocess shell=True`,
  `Runtime.getRuntime().exec /bin/sh -c`, `exec`, `execSync`, `spawn`,
  backtick / `$(...)` templates). Dangerous characters named: `; & | $()`.
- **CWE-89** now LEADS with "database driver or ORM" and names
  concrete sinks (`cursor.execute`, `db.query`, `knex.raw`, `sa.text`,
  `PDO prepare`, `gorm.Raw`, `Jdbc.executeQuery`). Dangerous characters
  named: single quote, `--`, semicolon, `UNION`, `OR 1=1`.
- Each entry now contains a one-sentence negation of the other
  category ("SQL databases are not involved" for CWE-78; "Operating-
  system shells are not involved" for CWE-89).

The other **eight entries are byte-identical** to the Stage 7L
catalog — grep-verified in the commit. Only the two interfering
entries changed.

## Experiment design

| Field | Value |
|---|---|
| Conditions | A = Stage 7L catalog (unchanged); B = revised catalog_v2 |
| Fresh snippets | 40, 4 per CWE × 10 categories; SHA-256 `4a023a9e...` |
| Seeds | `range(7000, 7050)` (non-overlapping with every prior stage) |
| Trials | 2 × 40 greedy + 2 × 40 × 50 sampled = **4,080** |
| Scorer | `scorer_v2b` (no regex; 100% agreement w/ v2a on 1,100 v2.1 trials) |
| Model | Qwen3.5-2B EXL3 8bpw, sha256 `a106b1c9...` (UNCHANGED since Stage 7I) |
| Sampler | `GreedySampler()` / `DefaultSampler()` (UNCHANGED) |
| Freeze | `freeze.json` committed before model invoked |

Snippets distinctness: CWE-89 snippets all name a database driver
(`psycopg2`, `mysql2`, `sqlx`, `ActiveRecord`) with SQL keywords
plainly visible. CWE-78 snippets all name an unambiguous shell sink
(`subprocess.Popen shell=True`, `spawn 'sh','-c'`, `exec.Command
"bash","-c"`, `Deno.Command 'sh'`). This gives the catalog an actual
retrieval surface to use.

## Results

### Headline (sampled, 2,000 trials per condition, 40 fresh snippets × 50 seeds)

| Cond | k/n | Trial Wilson 95% | Item mean | Item bootstrap 95% | Deterministic |
|---|---|---|---|---|---|
| A (Stage 7L catalog) | 1,867 / 2,000 | **93.3%** [92.2-94.4] | 93.3% | 88.4 - 97.3 | 38 / 40 |
| B (revised catalog)  | 1,900 / 2,000 | **95.0%** [94.0-95.9] | 95.0% | 91.3 - 97.9 | **40 / 40** |

**Paired McNemar on matched (snippet, seed) pairs**:
```
A vs B : n11=1835  n10=32  n01=65  n00=68   chi2=10.56   p = 1.16e-3
```
B significantly better than A, BUT the two item-level bootstrap CIs
overlap substantially. The point-estimate improvement is 1.7 pp —
about two snippets' worth out of 40.

### Per-CWE change

| CWE | A | B | Δ(B-A) | Verdict |
|---|---|---|---|---|
| CWE-89 (SQL) | 62.0% | **80.5%** | **+18.5pp** | **TARGET RECOVERED** |
| CWE-22 (path) | 92.5% | 98.5% | +6.0pp | preserved + gain |
| CWE-79 (XSS) | 88.0% | 90.5% | +2.5pp | preserved |
| CWE-346 (origin) | 97.0% | 99.0% | +2.0pp | preserved |
| CWE-78 (OS cmd) | 98.5% | 100.0% | +1.5pp | preserved |
| CWE-347 (JWT) | 95.5% | 95.5% | 0 | preserved |
| CWE-434 (upload) | 100.0% | 100.0% | 0 | preserved |
| CWE-352 (CSRF) | 100.0% | 100.0% | 0 | preserved |
| CWE-327 (crypto) | 100.0% | 100.0% | 0 | preserved |
| **CWE-311 (cleartext)** | **100.0%** | **86.0%** | **-14.0pp** | **NEW REGRESSION** |

Three key points:
1. CWE-89 recovered from 62.0% to 80.5%. Target met.
2. CWE-78 preserved at 100% — the counterpart was not harmed.
3. **CWE-311 fell 14 pp**, with all 28 lost trials going to CWE-327.

### Per-example reliability

| Cond | 50/50 | mid | 0/50 |
|---|---|---|---|
| A | 25 | 15 | 0 |
| B | 26 | 14 | 0 |

Both much healthier than Stage 7L's A (14 at 0/50). Revised catalog preserves this.

## Confusion matrix comparison

Only cells where |Δ| ≥ 3; full 10×10 matrices in `analysis.json`.

| Truth → Pred | A n | B n | Δ |
|---|---|---|---|
| CWE-89 → CWE-89 (correct) | 124 | 161 | **+37** |
| CWE-89 → CWE-78 | 55 | 10 | **-45** ← the Stage 7L blocker is gone |
| CWE-89 → CWE-79 | 12 | 24 | +12 ← residual |
| CWE-311 → CWE-311 (correct) | 200 | 172 | **-28** |
| CWE-311 → CWE-327 | 0 | 28 | **+28** ← NEW confusion |
| CWE-22 → CWE-22 (correct) | 185 | 197 | +12 |
| CWE-22 → CWE-434 | 7 | 3 | -4 |

CWE-89 → CWE-78 collapsed from 55 to 10. The targeted fix worked
exactly as designed. In exchange, a brand-new CWE-311 → CWE-327
confusion appeared that did not exist in Stage 7L.

Hypothesis for the new regression: the revised CWE-78 and CWE-89
entries are ~160 tokens longer total; the model's attention mass
redistributed and CWE-311 (plain-text) vs CWE-327 (broken
algorithm) is the next-most-collapsible pair at the 2B model's
representational capacity.

## Inference cost

| Cond | Mean prompt tokens | Mean TTFT (ms) | Mean elapsed (ms) | tokens / correct |
|---|---|---|---|---|
| A | 821.3 | 25.1 | 77.2 | 7.1 |
| B | 980.3 | **48.3** | 100.9 | 7.0 |

Revised catalog adds ~160 prompt tokens and nearly doubles TTFT
(25.1 → 48.3 ms). Tokens-per-correct is unchanged (7.1 vs 7.0) —
same output length despite higher prompt cost.

Note: this run's `prompt_tokens` are correctly measured (via tensor
shape[-1], not the Stage 7L `len()` bug).

## Which capability still blocks graduation

**Single next blocker: CWE-311 → CWE-327 confusion (new, Stage 7M-induced).**
28 of 200 CWE-311 trials under condition B misclassified as CWE-327.
The pattern is predictable: longer CWE-78 / CWE-89 entries shifted
attention mass, and the next-most-confusable pair in 2B's
representation (plain-text vs broken-algorithm data protection)
surfaced.

In-scope fixes (Stage 7N would do ONE, not all):
1. Rewrite CWE-311 and CWE-327 with disambiguating leads
   ("no encryption at all" vs "using a weak or deprecated algorithm")
   + negations of each other — same recipe that worked for
   CWE-78 / CWE-89.
2. Chain-of-selection prompting: top-3 shortlist then disambiguate.
3. Shorten the whole catalog to 10 crisp one-sentence entries and
   measure if less context reduces attention-mass redistribution.

Zero of these are attempted in Stage 7M; naming is not fixing.

**Second-order observation** (NOT a graduation blocker, logged):
each catalog revision fixes one pair and surfaces another. At 2B
the model has limited bandwidth for pairwise disambiguation under
ANY 10-item catalog. The right structural answer is probably
"top-3 then disambiguate" (in scope for Stage 7N) or SFT on
labeled disambiguating pairs (out of scope per standing rule).

## Scope guards honored

- Prior Stage 7J/7K/7L/v2/v2_1/g4 receipts: untouched.
- No new models, quantizers, or benchmark categories.
- Did NOT start SFT / DPO / QLoRA / RAG.
- Preserved EXL3 109.5 / TFOLD 110.8 tok/s and MD5 `c6a03e5665...` bf16 identity.
- Catalog revision: minimal (2 entries); 8 untouched byte-identical.
- Frozen BEFORE model invoked; sha256 recorded.

## Reproduce

```bash
# scoring only (receipts committed)
python docs/dyno/scorers/analyze_stage7m.py

# full re-run (requires exllamav3 + model)
ssh spark 'cd ~/stage7m && python stage7m_runner.py \
    --snippets snippets.json --freeze freeze.json \
    --catalog-a catalog.md --catalog-b catalog_v2.md \
    --out-dir out'
```
