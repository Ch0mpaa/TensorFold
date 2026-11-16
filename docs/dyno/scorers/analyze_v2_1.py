"""v2.1 analysis: dev/held-out split, dual-scorer agreement, item-level uncertainty."""
import json, sys, re
from pathlib import Path
from collections import Counter, defaultdict
import random

# Make scorer_v2b importable regardless of cwd: it sits beside this file.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from scorer_v2b import parse_batched as p2b_batched, parse_single as p2b_single, parse_highest_risk as p2b_hr

DEV_SNIPPETS_CWE = {"CWE-89", "CWE-79", "CWE-22"}
HELDOUT_SNIPPETS_CWE = {"CWE-347", "CWE-434", "CWE-311", "CWE-352", "CWE-346", "CWE-327", "CWE-78"}
ALL_TRUTHS = ["CWE-89", "CWE-79", "CWE-22", "CWE-347", "CWE-434", "CWE-311", "CWE-352", "CWE-346", "CWE-327", "CWE-78"]
HIGHEST_RISK_TRUTH = "CWE-78"

# scorer_v2a (the regex one used in v2)
def v2a_extract_list(text):
    out = [None] * 10
    for line in text.splitlines():
        m = re.match(r"\s*(\d{1,2})[\.\):]\s*.*?(CWE-?\d{1,4})", line, re.I)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < 10 and out[idx] is None:
                n = re.search(r"\d{1,4}", m.group(2)).group()
                out[idx] = f"CWE-{n}"
    return out

def v2a_extract_cwe(text):
    m = re.search(r"CWE-?(\d{1,4})", text)
    return f"CWE-{m.group(1)}" if m else None

def v2a_extract_hr(text):
    m = re.search(r"HIGHEST[- ]?RISK[:\s]+CWE-?(\d{1,4})", text, re.I)
    return f"CWE-{m.group(1)}" if m else None

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    hw = z * ((p * (1-p)/n + z*z/(4*n*n))**0.5) / denom
    return (max(0, center - hw), min(1, center + hw))

def bootstrap_ci_mean(values, n_boot=10000, alpha=0.05, seed=0):
    if not values: return (0.0, 0.0)
    rng = random.Random(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        samp = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(samp) / n)
    boots.sort()
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)

# Repo-relative: this file lives at <repo>/docs/dyno/scorers/analyze_v2_1.py.
# Receipts live at <repo>/docs/dyno/offsec/v2 and the summary lands beside them.
_REPO = _HERE.parent.parent.parent
ROOT = _REPO / "docs" / "dyno" / "offsec" / "v2"
OUT_PATH = _REPO / "docs" / "dyno" / "v2_1_analysis.json"

def load(cond):
    path = ROOT / f"{cond}.jsonl"
    return [json.loads(l) for l in path.open()]

print("=" * 90)
print("# Dyno v2.1 Generalization Audit")
print("=" * 90)

# ---------------------------------------------------------------- G2: dual-scorer agreement
print("\n## G2 — Independent scorer agreement across 1,100 trials")
agreements = {"batched": [0, 0, 0, 0], "single": [0, 0]}  # per-slot + hr; for single: pred, nothing else
total_slots = 0
disagreements = []
for cond in ["zero_batched", "few_batched", "zero_single", "few_single"]:
    rows = load(cond)
    is_batched = cond.endswith("batched")
    for r in rows:
        txt = r["output_text"]
        if is_batched:
            a = v2a_extract_list(txt)
            b = p2b_batched(txt)
            for i in range(10):
                if a[i] == b[i]:
                    agreements["batched"][0] += 1
                else:
                    disagreements.append((cond, r["seed"], i, a[i], b[i], txt[:200]))
                agreements["batched"][1] += 1
            a_hr = v2a_extract_hr(txt)
            b_hr = p2b_hr(txt)
            if a_hr == b_hr:
                agreements["batched"][2] += 1
            else:
                disagreements.append((cond, r["seed"], "HR", a_hr, b_hr, txt[:200]))
            agreements["batched"][3] += 1
        else:
            a = v2a_extract_cwe(txt)
            b = p2b_single(txt)
            if a == b:
                agreements["single"][0] += 1
            else:
                disagreements.append((cond, r["seed"], "single", a, b, txt[:200]))
            agreements["single"][1] += 1

bat_agree_slot = agreements["batched"][0] / max(1, agreements["batched"][1])
bat_agree_hr   = agreements["batched"][2] / max(1, agreements["batched"][3])
sin_agree      = agreements["single"][0] / max(1, agreements["single"][1])
print(f"  batched per-slot agreement: {agreements['batched'][0]}/{agreements['batched'][1]} = {bat_agree_slot*100:.2f}%")
print(f"  batched highest-risk agreement: {agreements['batched'][2]}/{agreements['batched'][3]} = {bat_agree_hr*100:.2f}%")
print(f"  single pred agreement: {agreements['single'][0]}/{agreements['single'][1]} = {sin_agree*100:.2f}%")
print(f"  total disagreements: {len(disagreements)}")
for d in disagreements[:5]:
    print(f"    ex: cond={d[0]} seed={d[1]} slot={d[2]} v2a={d[3]} v2b={d[4]}")
# Agreement for picking downstream scorer: use v2b (stricter)
# We'll use v2b for the rest

# ---------------------------------------------------------------- G1 & G3: dev vs held-out, per-snippet CIs
print("\n## G1 — Accuracy split DEV (3 snippets) vs HELD-OUT (7 snippets)")
print()
print(f"{'condition':15s} {'DEV correct/150':>18s} {'DEV acc[CI]':>22s} {'HELDOUT correct/350':>22s} {'HELDOUT acc[CI]':>22s}")
print("-" * 105)
results_table = {}

for cond in ["zero_batched", "few_batched", "zero_single", "few_single"]:
    rows = load(cond)
    is_batched = cond.endswith("batched")
    dev_hits, dev_n = 0, 0
    heldout_hits, heldout_n = 0, 0
    # Per-snippet accuracy for bootstrap
    per_snip_acc_dev = defaultdict(list)
    per_snip_acc_heldout = defaultdict(list)
    for r in rows:
        txt = r["output_text"]
        if is_batched:
            preds = p2b_batched(txt)
            for i, (p, t) in enumerate(zip(preds, ALL_TRUTHS)):
                correct = 1 if p == t else 0
                if t in DEV_SNIPPETS_CWE:
                    dev_hits += correct; dev_n += 1
                    per_snip_acc_dev[t].append(correct)
                else:
                    heldout_hits += correct; heldout_n += 1
                    per_snip_acc_heldout[t].append(correct)
        else:
            truth = r["truths"][0]
            pred = p2b_single(txt)
            correct = 1 if pred == truth else 0
            if truth in DEV_SNIPPETS_CWE:
                dev_hits += correct; dev_n += 1
                per_snip_acc_dev[truth].append(correct)
            else:
                heldout_hits += correct; heldout_n += 1
                per_snip_acc_heldout[truth].append(correct)
    # Wilson CIs at TRIAL level
    dev_p = dev_hits/dev_n if dev_n else 0
    heldout_p = heldout_hits/heldout_n if heldout_n else 0
    dlo, dhi = wilson_ci(dev_hits, dev_n)
    hlo, hhi = wilson_ci(heldout_hits, heldout_n)
    print(f"{cond:15s}  {dev_hits:4d}/{dev_n:3d}  {dev_p*100:5.1f}% [{dlo*100:5.1f}-{dhi*100:5.1f}]  {heldout_hits:4d}/{heldout_n:3d}  {heldout_p*100:5.1f}% [{hlo*100:5.1f}-{hhi*100:5.1f}]")
    results_table[cond] = {
        "dev": {"correct": dev_hits, "total": dev_n, "acc": dev_p, "wilson_ci": [dlo, dhi]},
        "heldout": {"correct": heldout_hits, "total": heldout_n, "acc": heldout_p, "wilson_ci": [hlo, hhi]},
        "per_snippet_dev": {t: sum(a)/len(a) for t, a in per_snip_acc_dev.items()},
        "per_snippet_heldout": {t: sum(a)/len(a) for t, a in per_snip_acc_heldout.items()},
    }

print("\n## G3 — Item-level uncertainty (ONLY 10 distinct examples)")
print("Report: bootstrap CI over the 10 per-snippet accuracies (treats each snippet as a cluster)")
print()
print(f"{'condition':15s} {'DEV snip-mean':>18s} {'DEV snip CI':>22s} {'HELDOUT snip-mean':>22s} {'HELDOUT snip CI':>22s}")
print("-" * 105)
for cond, res in results_table.items():
    dev_snip_accs = list(res["per_snippet_dev"].values())
    hld_snip_accs = list(res["per_snippet_heldout"].values())
    dev_mean = sum(dev_snip_accs) / len(dev_snip_accs) if dev_snip_accs else 0
    hld_mean = sum(hld_snip_accs) / len(hld_snip_accs) if hld_snip_accs else 0
    dlo, dhi = bootstrap_ci_mean(dev_snip_accs, seed=42)
    hlo, hhi = bootstrap_ci_mean(hld_snip_accs, seed=42)
    print(f"{cond:15s}  n=3 snips  {dev_mean*100:5.1f}% [{dlo*100:5.1f}-{dhi*100:5.1f}]  n=7 snips  {hld_mean*100:5.1f}% [{hlo*100:5.1f}-{hhi*100:5.1f}]")
    res["dev_snip_mean"] = dev_mean
    res["dev_snip_bootstrap_ci"] = [dlo, dhi]
    res["heldout_snip_mean"] = hld_mean
    res["heldout_snip_bootstrap_ci"] = [hlo, hhi]

# Dump results for the receipt JSON
with OUT_PATH.open("w") as f:
    json.dump({
        "agreement": {
            "batched_per_slot": {"match": agreements['batched'][0], "total": agreements['batched'][1], "pct": bat_agree_slot},
            "batched_highest_risk": {"match": agreements['batched'][2], "total": agreements['batched'][3], "pct": bat_agree_hr},
            "single_pred": {"match": agreements['single'][0], "total": agreements['single'][1], "pct": sin_agree},
            "total_disagreements": len(disagreements),
        },
        "per_condition": results_table,
    }, f, indent=2)
print("\nWrote docs/dyno/v2_1_analysis.json")
