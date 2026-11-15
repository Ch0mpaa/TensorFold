"""Paired within-seed comparisons on held-out set.

For each pair of conditions, build contingency:
  n11 = both correct
  n10 = cond_a correct, cond_b wrong
  n01 = cond_a wrong, cond_b correct
  n00 = both wrong
McNemar chi-squared with continuity correction = (|n10 - n01| - 1)^2 / (n10 + n01)
"""
import json, sys, re
from pathlib import Path
sys.path.insert(0, '/tmp')
from scorer_v2b import parse_batched, parse_single

HELDOUT_IDX = [3, 4, 5, 6, 7, 8, 9]  # 0-indexed slot numbers for the 7 held-out
ALL_TRUTHS = ["CWE-89", "CWE-79", "CWE-22", "CWE-347", "CWE-434", "CWE-311", "CWE-352", "CWE-346", "CWE-327", "CWE-78"]
HELDOUT_TRUTHS = [ALL_TRUTHS[i] for i in HELDOUT_IDX]

ROOT = Path("D:/op13-root/src/tensorfold/docs/dyno/offsec/v2")

def load(cond):
    return [json.loads(l) for l in (ROOT / f"{cond}.jsonl").open()]

def heldout_scores(cond):
    """Return list of (seed, snippet_idx, correct) for the 7*50 = 350 held-out trials."""
    rows = load(cond)
    is_batched = cond.endswith("batched")
    results = []
    if is_batched:
        for r in rows:
            preds = parse_batched(r["output_text"])
            for i in HELDOUT_IDX:
                results.append((r["seed"], i, 1 if preds[i] == ALL_TRUTHS[i] else 0))
    else:
        for r in rows:
            truth = r["truths"][0]
            if truth not in HELDOUT_TRUTHS:
                continue
            # single-snippet: condition key has _snipN suffix; r["seed"] + truth are unique
            pred = parse_single(r["output_text"])
            # find snippet idx
            idx = ALL_TRUTHS.index(truth)
            results.append((r["seed"], idx, 1 if pred == truth else 0))
    return {(s, i): c for s, i, c in results}

CONDS = ["zero_batched", "few_batched", "zero_single", "few_single"]
scores = {c: heldout_scores(c) for c in CONDS}

def mcnemar(d_a, d_b):
    keys = set(d_a) & set(d_b)
    n11 = n10 = n01 = n00 = 0
    for k in keys:
        a, b = d_a[k], d_b[k]
        if a and b: n11 += 1
        elif a and not b: n10 += 1
        elif not a and b: n01 += 1
        else: n00 += 1
    disc = n10 + n01
    if disc == 0:
        chi2 = 0
    else:
        chi2 = ((abs(n10 - n01) - 1) ** 2) / disc
    # 1-dof chi-squared p-value approximation via survival function
    # Using erfc approximation: p = erfc(sqrt(chi2)/sqrt(2))
    import math
    p = math.erfc((chi2 ** 0.5) / (2 ** 0.5)) if chi2 > 0 else 1.0
    return {"n11": n11, "n10": n10, "n01": n01, "n00": n00, "chi2": chi2, "p": p, "n": len(keys)}

print(f"Held-out set: 7 snippets x 50 seeds = 350 pairs per condition")
print()
print(f"{'pair':30s} {'n11':>5s} {'n10':>5s} {'n01':>5s} {'n00':>5s} {'chi2':>6s} {'p':>8s}")
print("-" * 70)
for i, a in enumerate(CONDS):
    for b in CONDS[i+1:]:
        res = mcnemar(scores[a], scores[b])
        print(f"{a:12s} vs {b:12s}  {res['n11']:5d} {res['n10']:5d} {res['n01']:5d} {res['n00']:5d} {res['chi2']:6.2f} {res['p']:8.4f}")

print()
print("## Random-chance check")
# Random guessing on CWE-25 = 4%. What proportion of held-out accuracies beat that?
for c in CONDS:
    d = scores[c]
    k = sum(d.values())
    n = len(d)
    chance = 0.04
    # Binomial one-sided test: P(X >= k | n, chance)
    import math
    def binom_sf(k, n, p):
        # survival: 1 - CDF(k-1). Use normal approx for large n.
        mean = n * p
        sd = (n * p * (1-p)) ** 0.5
        z = (k - 0.5 - mean) / sd  # continuity correction
        return 0.5 * math.erfc(z / (2 ** 0.5))
    p_val = binom_sf(k, n, chance)
    print(f"  {c:15s} {k}/{n} = {k/n*100:.1f}% ; binomial vs 4% chance: p = {p_val:.6f}")
