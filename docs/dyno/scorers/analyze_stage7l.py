"""Stage 7L analysis: A/B/C comparison + paired McNemar + prompt-token / TTFT cost."""
import json, sys, math
from pathlib import Path
from collections import Counter, defaultdict
import random

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from scorer_v2b import parse_single

_REPO = _HERE.parent.parent.parent
STAGE_DIR = _REPO / "docs" / "dyno" / "offsec" / "stage7l"

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5) / d
    return (max(0,c-hw), min(1,c+hw))

def bootstrap(values, n_boot=10000, seed=0):
    if not values: return (0.0, 0.0)
    rng = random.Random(seed); n = len(values)
    boots = [sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(n_boot)]
    boots.sort()
    return boots[int(n_boot*0.025)], boots[int(n_boot*0.975)]

def mcnemar(d_a, d_b):
    """d_a/d_b map (snippet_id, seed) -> {0,1}. Chi-sq w/ continuity correction."""
    keys = set(d_a) & set(d_b)
    n11=n10=n01=n00=0
    for k in keys:
        a,b = d_a[k], d_b[k]
        if a and b: n11+=1
        elif a and not b: n10+=1
        elif not a and b: n01+=1
        else: n00+=1
    disc = n10+n01
    chi2 = ((abs(n10-n01)-1)**2)/disc if disc else 0.0
    p = math.erfc((chi2**0.5)/(2**0.5)) if chi2>0 else 1.0
    return dict(n11=n11, n10=n10, n01=n01, n00=n00, chi2=chi2, p=p, n=len(keys))

def load_jsonl(p):
    return [json.loads(l) for l in p.open()]

det_rows = load_jsonl(STAGE_DIR/"deterministic.jsonl")
smp_rows = load_jsonl(STAGE_DIR/"sampled.jsonl")
print(f"deterministic trials: {len(det_rows)}  sampled trials: {len(smp_rows)}")

def score(row):
    pred = parse_single(row["output_text"])
    correct = (pred == row["cwe_truth"])
    return pred, correct

CONDS = ("A","B","C")
# Per-condition accuracy dictionaries keyed by (snippet, seed)
smp_by_cond = {c: {} for c in CONDS}
det_by_cond = {c: {} for c in CONDS}
per_snip = {c: defaultdict(lambda: [0,0]) for c in CONDS}
per_cwe  = {c: defaultdict(lambda: [0,0]) for c in CONDS}
ttft = {c: [] for c in CONDS}
elapsed = {c: [] for c in CONDS}
prompt_tokens = {c: [] for c in CONDS}
n_tokens = {c: [] for c in CONDS}

for r in smp_rows:
    c = r["condition"]
    _,corr = score(r)
    smp_by_cond[c][(r["snippet_id"], r["seed"])] = 1 if corr else 0
    per_snip[c][r["snippet_id"]][1] += 1
    per_cwe[c][r["cwe_truth"]][1]   += 1
    if corr:
        per_snip[c][r["snippet_id"]][0] += 1
        per_cwe[c][r["cwe_truth"]][0]   += 1
    if r.get("ttft_s") is not None: ttft[c].append(r["ttft_s"])
    elapsed[c].append(r.get("elapsed_s", 0.0))
    prompt_tokens[c].append(r.get("prompt_tokens", 0))
    n_tokens[c].append(r.get("n_tokens", 0))
for r in det_rows:
    c = r["condition"]
    _,corr = score(r)
    det_by_cond[c][r["snippet_id"]] = 1 if corr else 0

print(f"\n## Overall accuracy (sampled)")
print(f"  {'cond':4s} {'k/n':>14s} {'trial Wilson':>22s} {'item mean':>12s} {'item boot CI':>18s}")
for c in CONDS:
    kn = sum(smp_by_cond[c].values()); n = len(smp_by_cond[c])
    lo,hi = wilson(kn,n)
    accs = [k/t for k,t in per_snip[c].values() if t>0]
    mu = sum(accs)/len(accs) if accs else 0
    blo,bhi = bootstrap(accs)
    print(f"  {c:4s} {kn:4d}/{n:4d}   {kn/n*100:5.1f}% [{lo*100:4.1f}-{hi*100:4.1f}]  {mu*100:5.1f}%  [{blo*100:4.1f}-{bhi*100:4.1f}]")

print(f"\n## Deterministic accuracy (greedy, 1 per snippet)")
for c in CONDS:
    hits = sum(det_by_cond[c].values())
    print(f"  {c:4s} {hits:2d}/30 = {hits/30*100:.1f}%")

print(f"\n## Per-CWE sampled (correct / 150 = 3 snips * 50 seeds)")
cwes = sorted({cwe for c in CONDS for cwe in per_cwe[c]})
print(f"  {'CWE':10s} {'A':>12s} {'B':>12s} {'C':>12s}  (delta B-A, B-C)")
for cwe in cwes:
    row = []
    for c in CONDS:
        k,n = per_cwe[c][cwe]
        row.append((k,n,k/n if n else 0))
    da = (row[1][2]-row[0][2])*100
    dc = (row[1][2]-row[2][2])*100
    print(f"  {cwe:10s} {row[0][0]:3d}/{row[0][1]:3d} ({row[0][2]*100:4.1f}%) {row[1][0]:3d}/{row[1][1]:3d} ({row[1][2]*100:4.1f}%) {row[2][0]:3d}/{row[2][1]:3d} ({row[2][2]*100:4.1f}%)   +{da:5.1f}pp / +{dc:5.1f}pp")

print(f"\n## Paired McNemar on matched (snippet,seed) pairs")
pairs = [("A","B"), ("B","C"), ("A","C")]
for a,b in pairs:
    m = mcnemar(smp_by_cond[a], smp_by_cond[b])
    print(f"  {a} vs {b}: n11={m['n11']:4d} n10={m['n10']:4d} n01={m['n01']:4d} n00={m['n00']:4d} chi2={m['chi2']:7.2f} p={m['p']:.3e}")

print(f"\n## Prompt / latency / token cost")
print(f"  {'cond':4s} {'mean prompt tok':>16s} {'mean ttft (ms)':>14s} {'mean elapsed (s)':>16s} {'tok / correct':>14s}")
for c in CONDS:
    mpt = sum(prompt_tokens[c])/len(prompt_tokens[c]) if prompt_tokens[c] else 0
    mtt = (sum(ttft[c])/len(ttft[c]))*1000 if ttft[c] else 0
    mel = sum(elapsed[c])/len(elapsed[c]) if elapsed[c] else 0
    nt = sum(n_tokens[c])
    corr = sum(smp_by_cond[c].values())
    tpc = nt/corr if corr else float('inf')
    print(f"  {c:4s} {mpt:16.1f} {mtt:14.1f} {mel:16.4f} {tpc:14.1f}")

# Per-example reliability table (sampled)
rel_bands = Counter()
for c in CONDS:
    for sid,(k,n) in per_snip[c].items():
        band = "50/50" if k==50 else ("0/50" if k==0 else "mid")
        rel_bands[(c, band)] += 1
print(f"\n## Per-example reliability bands (of 30 snippets per condition)")
for c in CONDS:
    print(f"  {c}: 50/50={rel_bands[(c,'50/50')]:2d}   mid={rel_bands[(c,'mid')]:2d}   0/50={rel_bands[(c,'0/50')]:2d}")

# Dump machine-readable
out = {
    "trials": {"deterministic": len(det_rows), "sampled": len(smp_rows)},
    "sampled_overall": {
        c: {
            "k": sum(smp_by_cond[c].values()),
            "n": len(smp_by_cond[c]),
            "acc_trial": sum(smp_by_cond[c].values())/max(1,len(smp_by_cond[c])),
            "wilson_ci": list(wilson(sum(smp_by_cond[c].values()), len(smp_by_cond[c]))),
            "item_mean": sum(k/t for k,t in per_snip[c].values() if t>0)/max(1,len(per_snip[c])),
            "item_bootstrap_ci": list(bootstrap([k/t for k,t in per_snip[c].values() if t>0])),
        }
        for c in CONDS
    },
    "deterministic_overall": {c: {"k": sum(det_by_cond[c].values()), "n": len(det_by_cond[c])} for c in CONDS},
    "per_cwe_sampled": {
        c: {cwe: {"correct": k, "total": n, "acc": k/n if n else 0} for cwe,(k,n) in per_cwe[c].items()}
        for c in CONDS
    },
    "per_snippet_sampled": {
        c: {sid: {"correct": k, "total": n} for sid,(k,n) in per_snip[c].items()}
        for c in CONDS
    },
    "paired_mcnemar": {f"{a}_vs_{b}": mcnemar(smp_by_cond[a], smp_by_cond[b]) for a,b in pairs},
    "latency_tokens": {
        c: {
            "mean_prompt_tokens": sum(prompt_tokens[c])/len(prompt_tokens[c]) if prompt_tokens[c] else 0,
            "mean_ttft_s": sum(ttft[c])/len(ttft[c]) if ttft[c] else 0,
            "mean_elapsed_s": sum(elapsed[c])/len(elapsed[c]) if elapsed[c] else 0,
            "total_output_tokens": sum(n_tokens[c]),
            "tokens_per_correct": (sum(n_tokens[c])/sum(smp_by_cond[c].values())) if sum(smp_by_cond[c].values()) else None,
        }
        for c in CONDS
    },
}
(STAGE_DIR/"analysis.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {STAGE_DIR/'analysis.json'}")
