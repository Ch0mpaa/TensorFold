"""Stage 7K analysis: score receipts with scorer_v2b, compute per-CWE and per-example
accuracy, item-level bootstrap CI over 30 examples, failure breakdown, and compare
to Stage 7J few_single held-out. Reads from <repo>/docs/dyno/offsec/stage7k/."""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import random

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from scorer_v2b import parse_single

_REPO = _HERE.parent.parent.parent
STAGE_DIR = _REPO / "docs" / "dyno" / "offsec" / "stage7k"

def load_jsonl(p):
    return [json.loads(l) for l in p.open()]

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5) / d
    return (max(0,c-hw), min(1,c+hw))

def bootstrap(values, n_boot=10000, alpha=0.05, seed=0):
    if not values: return (0.0, 0.0)
    rng = random.Random(seed); n = len(values)
    boots = [sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(n_boot)]
    boots.sort()
    return boots[int(n_boot*alpha/2)], boots[int(n_boot*(1-alpha/2))]

snippets = json.loads((STAGE_DIR/"snippets.json").read_text())["snippets"]
truth_by_id = {s["id"]: s["cwe"] for s in snippets}
cwe_by_id   = {s["id"]: s["cwe"] for s in snippets}

det_rows = load_jsonl(STAGE_DIR/"deterministic.jsonl")
smp_rows = load_jsonl(STAGE_DIR/"sampled.jsonl")

print(f"# Stage 7K analysis")
print(f"deterministic trials: {len(det_rows)}  sampled trials: {len(smp_rows)}")

# Score each row
def score(row):
    pred = parse_single(row["output_text"])
    truth = row["cwe_truth"]
    correct = (pred == truth)
    if pred is None:
        kind = "malformed"
    elif row["status"] != "completed":
        kind = row["status"]
    elif correct:
        kind = "correct"
    else:
        kind = "wrong_cwe"
    return pred, correct, kind

det_scored = [(r, *score(r)) for r in det_rows]
smp_scored = [(r, *score(r)) for r in smp_rows]

# ---- Deterministic baseline
print(f"\n## Deterministic (greedy, 30 trials)")
det_correct = sum(1 for r,p,c,k in det_scored if c)
dlo, dhi = wilson(det_correct, 30)
print(f"  overall: {det_correct}/30 = {det_correct/30*100:.1f}% [{dlo*100:.1f}-{dhi*100:.1f}]")
per_cwe_det = defaultdict(lambda: [0,0])
for r,p,c,k in det_scored:
    per_cwe_det[r["cwe_truth"]][1] += 1
    if c: per_cwe_det[r["cwe_truth"]][0] += 1
print(f"  {'CWE':10s} {'correct/3':>12s}")
for cwe in sorted(per_cwe_det):
    k, n = per_cwe_det[cwe]
    print(f"  {cwe:10s} {k:3d}/{n}")

# ---- Sampled sweep per snippet/CWE
print(f"\n## Sampled (50 seeds × 30 snippets = 1500 trials)")
per_snip = defaultdict(lambda: [0,0])
per_cwe  = defaultdict(lambda: [0,0])
kind_cnt = Counter()
tok_total = 0
elapsed_total = 0.0
for r,p,c,k in smp_scored:
    per_snip[r["snippet_id"]][1] += 1
    per_cwe[r["cwe_truth"]][1] += 1
    kind_cnt[k] += 1
    tok_total += r.get("n_tokens", 0)
    elapsed_total += r.get("elapsed_s", 0.0)
    if c:
        per_snip[r["snippet_id"]][0] += 1
        per_cwe[r["cwe_truth"]][0] += 1

total_correct = sum(k for k,n in per_snip.values())
total_n = sum(n for k,n in per_snip.values())
mlo, mhi = wilson(total_correct, total_n)
print(f"  trial-level: {total_correct}/{total_n} = {total_correct/total_n*100:.1f}% [Wilson {mlo*100:.1f}-{mhi*100:.1f}]")

# Item-level (per-example) bootstrap
item_accs = [k/n for k,n in per_snip.values() if n>0]
item_mean = sum(item_accs)/len(item_accs) if item_accs else 0
blo, bhi = bootstrap(item_accs)
print(f"  item-level: mean of 30 example accuracies = {item_mean*100:.1f}% [bootstrap {blo*100:.1f}-{bhi*100:.1f}]")

# Macro-averaged accuracy (per-CWE)
cwe_accs = [k/n for cwe,(k,n) in per_cwe.items()]
macro = sum(cwe_accs)/len(cwe_accs)
cwe_lo, cwe_hi = bootstrap(cwe_accs)
print(f"  macro over 10 CWEs: {macro*100:.1f}% [bootstrap {cwe_lo*100:.1f}-{cwe_hi*100:.1f}]")

print(f"\n## Per-CWE sampled accuracy (3 snippets × 50 seeds = 150 per CWE)")
print(f"  {'CWE':10s} {'correct/150':>14s}  {'acc':>7s}  {'Wilson':>12s}")
for cwe in sorted(per_cwe):
    k, n = per_cwe[cwe]
    lo, hi = wilson(k, n)
    print(f"  {cwe:10s} {k:4d}/{n:3d}      {k/n*100:5.1f}%  [{lo*100:4.1f}-{hi*100:4.1f}]")

print(f"\n## Per-example success rate (sampled)")
print(f"  {'snippet':10s} {'CWE':10s} {'correct/50':>13s}  {'sampled det':>12s}")
for s in snippets:
    sid = s["id"]
    k, n = per_snip.get(sid, (0,0))
    det_match = next((c for r,p,c,k2 in det_scored if r["snippet_id"]==sid), False)
    print(f"  {sid:10s} {s['cwe']:10s}  {k:3d}/{n:2d}   {str(det_match):>12s}")

print(f"\n## Failure breakdown (sampled)")
for k, c in kind_cnt.most_common():
    print(f"  {k:15s} {c:5d}  ({c*100/len(smp_scored):.1f}%)")

print(f"\n## Runtime / tokens")
print(f"  sampled sum elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
print(f"  sampled total tokens generated: {tok_total}")
print(f"  mean tokens/trial: {tok_total/len(smp_scored):.1f}")

# ---- Compare vs Stage 7J
print(f"\n## Compare vs Stage 7J few_single held-out (73/350 = 20.9%, Wilson 16.9-25.4)")
print(f"  Stage 7K trial-level: {total_correct/total_n*100:.1f}%  (point estimate)")
print(f"  Stage 7K item-level: {item_mean*100:.1f}% [{blo*100:.1f}-{bhi*100:.1f}]  (what to report)")
print(f"  Sample size note: 30 examples vs Stage 7J's 7. CI will be tighter but still not a population.")

# Machine-readable summary
out = {
  "set": "stage7k_fresh_heldout",
  "trials": {"deterministic": len(det_rows), "sampled": len(smp_rows)},
  "deterministic": {
      "overall": {"correct": det_correct, "total": 30, "acc": det_correct/30, "wilson_ci": [dlo, dhi]},
      "per_cwe": {cwe: {"correct": k, "total": n, "acc": k/n} for cwe,(k,n) in per_cwe_det.items()},
  },
  "sampled": {
      "trial_level":  {"correct": total_correct, "total": total_n, "acc": total_correct/total_n, "wilson_ci": [mlo, mhi]},
      "item_level":   {"mean": item_mean, "bootstrap_ci": [blo, bhi], "n_examples": len(item_accs)},
      "macro_cwe":    {"mean": macro,     "bootstrap_ci": [cwe_lo, cwe_hi], "n_categories": len(cwe_accs)},
      "per_cwe":      {cwe: {"correct": k, "total": n, "acc": k/n, "wilson_ci": list(wilson(k,n))} for cwe,(k,n) in per_cwe.items()},
      "per_snippet":  {sid: {"correct": k, "total": n, "acc": k/n if n else None} for sid,(k,n) in per_snip.items()},
      "failure_breakdown": dict(kind_cnt),
      "runtime_s": elapsed_total,
      "tokens_total": tok_total,
      "tokens_mean": tok_total/len(smp_scored) if smp_scored else 0,
  },
  "comparison_stage7j": {
      "stage7j_few_single_heldout": {"k": 73, "n": 350, "acc": 0.209, "wilson_ci": [0.169, 0.254], "bootstrap_ci": [0.026, 0.440]},
  },
}
(STAGE_DIR/"analysis.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {STAGE_DIR/'analysis.json'}")
