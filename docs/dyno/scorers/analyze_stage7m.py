"""Stage 7M analysis: A (Stage 7L catalog) vs B (revised catalog) on 40 fresh snippets.
Adds a 10x10 confusion matrix per condition + paired McNemar."""
import json, sys, math, random
from pathlib import Path
from collections import Counter, defaultdict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from scorer_v2b import parse_single

_REPO = _HERE.parent.parent.parent
STAGE_DIR = _REPO / "docs" / "dyno" / "offsec" / "stage7m"

ALL_CWES = ["CWE-89","CWE-79","CWE-22","CWE-347","CWE-434","CWE-311","CWE-352","CWE-346","CWE-327","CWE-78"]

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

CONDS = ("A","B")
smp_by_cond = {c: {} for c in CONDS}
det_by_cond = {c: {} for c in CONDS}
per_snip = {c: defaultdict(lambda: [0,0]) for c in CONDS}
per_cwe  = {c: defaultdict(lambda: [0,0]) for c in CONDS}
confusion = {c: defaultdict(Counter) for c in CONDS}  # [truth][pred]
elapsed = {c: [] for c in CONDS}; ttft={c: [] for c in CONDS}; prompt_tokens={c: [] for c in CONDS}; n_tokens={c: [] for c in CONDS}

for r in smp_rows:
    c = r["condition"]
    pred = parse_single(r["output_text"]) or "malformed"
    truth = r["cwe_truth"]
    corr = (pred == truth)
    smp_by_cond[c][(r["snippet_id"], r["seed"])] = 1 if corr else 0
    per_snip[c][r["snippet_id"]][1] += 1
    per_cwe[c][truth][1] += 1
    if corr:
        per_snip[c][r["snippet_id"]][0] += 1
        per_cwe[c][truth][0] += 1
    confusion[c][truth][pred] += 1
    if r.get("ttft_s") is not None: ttft[c].append(r["ttft_s"])
    elapsed[c].append(r.get("elapsed_s", 0.0))
    prompt_tokens[c].append(r.get("prompt_tokens", 0))
    n_tokens[c].append(r.get("n_tokens", 0))
for r in det_rows:
    c = r["condition"]
    pred = parse_single(r["output_text"])
    det_by_cond[c][r["snippet_id"]] = 1 if pred == r["cwe_truth"] else 0

print(f"\n## Overall accuracy")
for c in CONDS:
    kn = sum(smp_by_cond[c].values()); n = len(smp_by_cond[c])
    wlo,whi = wilson(kn, n)
    accs = [k/t for k,t in per_snip[c].values() if t>0]
    mu = sum(accs)/len(accs)
    blo,bhi = bootstrap(accs)
    print(f"  {c}  sampled {kn}/{n}  trial Wilson {kn/n*100:.1f}% [{wlo*100:.1f}-{whi*100:.1f}]  item mean {mu*100:.1f}% [bootstrap {blo*100:.1f}-{bhi*100:.1f}]")
    print(f"     deterministic {sum(det_by_cond[c].values())}/40")

print(f"\n## Per-CWE sampled (correct / 200 = 4 snips * 50 seeds)")
print(f"  {'CWE':10s} {'A':>12s} {'B':>12s}  delta(B-A)")
for cwe in ALL_CWES:
    ka,na = per_cwe['A'][cwe]; kb,nb = per_cwe['B'][cwe]
    da = ((kb/nb)-(ka/na))*100 if na and nb else 0
    marker = ""
    if cwe == "CWE-89" and da > 10: marker = " RECOVERED"
    elif cwe == "CWE-78" and abs(da) > 2: marker = " SIDE EFFECT"
    print(f"  {cwe:10s} {ka:3d}/{na:3d} ({ka/na*100:5.1f}%) {kb:3d}/{nb:3d} ({kb/nb*100:5.1f}%)  {da:+6.1f}pp{marker}")

print(f"\n## Confusion matrix (sampled)")
for c in CONDS:
    print(f"  == cond={c} ==")
    print("    truth      | " + " ".join(f"{cwe:>7s}" for cwe in ALL_CWES) + " |   total   acc")
    for truth in ALL_CWES:
        row = confusion[c][truth]
        total = sum(row.values())
        cells = " ".join(f"{row.get(cwe,0):>7d}" for cwe in ALL_CWES)
        correct = row.get(truth, 0)
        print(f"    {truth:10s} | {cells} | {total:5d}   {correct*100/total:.1f}%")
    other = sum(sum(row.values()) for truth, row in confusion[c].items() if truth not in ALL_CWES)
    if other:
        print(f"    (other truth rows: {other} trials)")

print(f"\n## Paired McNemar on matched (snippet, seed)")
m = mcnemar(smp_by_cond['A'], smp_by_cond['B'])
print(f"  A vs B: n11={m['n11']} n10={m['n10']} n01={m['n01']} n00={m['n00']} chi2={m['chi2']:.2f} p={m['p']:.3e}")

print(f"\n## Prompt / latency / token cost")
for c in CONDS:
    mpt = sum(prompt_tokens[c])/len(prompt_tokens[c]) if prompt_tokens[c] else 0
    mtt = (sum(ttft[c])/len(ttft[c]))*1000 if ttft[c] else 0
    mel = sum(elapsed[c])/len(elapsed[c]) if elapsed[c] else 0
    corr = sum(smp_by_cond[c].values())
    tpc = sum(n_tokens[c])/corr if corr else float('inf')
    print(f"  {c}  prompt_tok {mpt:.1f}  ttft {mtt:.1f} ms  elapsed {mel*1000:.1f} ms  tokens/correct {tpc:.1f}")

print(f"\n## Per-example reliability (bands)")
for c in CONDS:
    bands = Counter()
    for sid,(k,n) in per_snip[c].items():
        bands["50/50" if k==50 else ("0/50" if k==0 else "mid")] += 1
    print(f"  {c}  50/50 = {bands['50/50']:2d}   mid = {bands['mid']:2d}   0/50 = {bands['0/50']:2d}")

# Machine-readable
out = {
    "trials": {"deterministic": len(det_rows), "sampled": len(smp_rows)},
    "sampled_overall": {c: {
        "k": sum(smp_by_cond[c].values()), "n": len(smp_by_cond[c]),
        "trial_wilson_ci": list(wilson(sum(smp_by_cond[c].values()), len(smp_by_cond[c]))),
        "item_mean": sum(k/t for k,t in per_snip[c].values() if t>0)/max(1,len(per_snip[c])),
        "item_bootstrap_ci": list(bootstrap([k/t for k,t in per_snip[c].values() if t>0])),
    } for c in CONDS},
    "deterministic_overall": {c: {"k": sum(det_by_cond[c].values()), "n": len(det_by_cond[c])} for c in CONDS},
    "per_cwe": {c: {cwe: {"k": per_cwe[c][cwe][0], "n": per_cwe[c][cwe][1]} for cwe in ALL_CWES} for c in CONDS},
    "confusion_matrix": {c: {truth: dict(confusion[c][truth]) for truth in ALL_CWES} for c in CONDS},
    "paired_mcnemar": {"A_vs_B": m},
    "latency_tokens": {c: {
        "mean_prompt_tokens": sum(prompt_tokens[c])/len(prompt_tokens[c]) if prompt_tokens[c] else 0,
        "mean_ttft_s": sum(ttft[c])/len(ttft[c]) if ttft[c] else 0,
        "mean_elapsed_s": sum(elapsed[c])/len(elapsed[c]) if elapsed[c] else 0,
        "tokens_per_correct": sum(n_tokens[c])/sum(smp_by_cond[c].values()) if sum(smp_by_cond[c].values()) else None,
    } for c in CONDS},
}
(STAGE_DIR/"analysis.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {STAGE_DIR/'analysis.json'}")
