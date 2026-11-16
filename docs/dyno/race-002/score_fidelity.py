"""Score the 4-domain micro-fidelity outputs.

Takes any number of fidelity_*.json receipts. For each, runs small deterministic
rubrics per domain and emits a summary with per-prompt pass/fail + per-domain
pass rate. Also emits a cross-model agreement table (how often each model's
response equals BF16 ref's response verbatim).
"""
import argparse, json, re, sys
from pathlib import Path

# Expected answers / rubrics
CODING_SIG = [
    # Just check for any valid-looking python code (contains return + indent OR def)
    lambda s: ("return" in s) and (("    " in s) or ("\t" in s)),
    lambda s: "return" in s and ("==" in s or ".lower" in s or "alnum" in s or "[::-1]" in s),
    lambda s: "return" in s and ("%" in s or "str(" in s or "divmod" in s or "sum(" in s),
    lambda s: "return" in s and ("for" in s or "while" in s or "*=" in s),
    lambda s: ("Fizz" in s and "Buzz" in s) and "return" in s,
]
REASONING_EXPECTED = ["22", "9:00", "12", "13 and 7", "4.5"]
# accept "9 PM" variants as well; 9:00 PM works
REASONING_ACCEPT = [
    lambda s: "22" in s,
    lambda s: any(k in s for k in ("9:00", "9 PM", "9pm", "21:00", "nine")),
    lambda s: "12" in s,
    lambda s: ("13" in s and "7" in s) or ("length" in s.lower() and "width" in s.lower()),
    lambda s: any(k in s for k in ("4.5", "4 hours and 30", "4.5 hours", "4 1/2")),
]
IFEVAL_RUBRIC = [
    lambda s: s.lower().count("azure") == 2,
    lambda s: len([x for x in re.split(r"[,\n]", s.strip()) if x.strip()]) == 3 and "." not in s[:80],
    lambda s: s.strip().upper() in ("YES", "YES.", "YES!") or s.strip().startswith("YES"),
    lambda s: len([ln for ln in s.strip().split("\n") if ln.strip()]) == 3,  # 3-line haiku
    lambda s: "hola" in s.lower() and len(s.strip().split()) <= 3,
]
CWE_EXPECTED = [
    ("CWE-95", "CWE-94"),  # code injection via eval
    ("CWE-89",),           # SQL injection
    ("CWE-79",),           # XSS
    ("CWE-208", "CWE-203", "CWE-200", "CWE-326"),  # timing side channel
    ("CWE-22", "CWE-23"),  # path traversal
]
def _text(entry):
    r = entry.get("response", "")
    if isinstance(r, dict): return r.get("content", "")
    if isinstance(r, str):
        try: return json.loads(r).get("content", "")
        except Exception: return r
    return str(r)

def score_coding(domain_entries):
    out = []
    for i, e in enumerate(domain_entries):
        t = _text(e)
        ok = CODING_SIG[i](t) if i < len(CODING_SIG) else False
        out.append({"idx": i, "pass": bool(ok), "snippet": t[:120]})
    return out
def score_reasoning(domain_entries):
    return [{"idx": i, "pass": bool(REASONING_ACCEPT[i](_text(e))), "snippet": _text(e)[:160]}
            for i, e in enumerate(domain_entries) if i < len(REASONING_ACCEPT)]
def score_ifeval(domain_entries):
    return [{"idx": i, "pass": bool(IFEVAL_RUBRIC[i](_text(e))), "snippet": _text(e)[:160]}
            for i, e in enumerate(domain_entries) if i < len(IFEVAL_RUBRIC)]
def score_cyber(domain_entries):
    out = []
    for i, e in enumerate(domain_entries):
        t = _text(e)
        m = re.search(r"CWE[- ]?\d+", t)
        answer = m.group(0).upper().replace(" ", "-") if m else ""
        ok = answer in CWE_EXPECTED[i] if i < len(CWE_EXPECTED) else False
        out.append({"idx": i, "pass": bool(ok), "model_answer": answer, "expected": CWE_EXPECTED[i]})
    return out

def summarize(scored):
    n = len(scored); p = sum(1 for x in scored if x["pass"])
    return {"n": n, "pass": p, "rate": round(p/n, 3) if n else 0.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = {"per_model": {}, "cross_agreement_bf16": {}}
    bf16_responses = {}
    for p in args.inputs:
        d = json.load(open(p))
        label = d["label"]; doms = d["domains"]
        coding = score_coding(doms["coding"])
        reasoning = score_reasoning(doms["reasoning"])
        ifeval = score_ifeval(doms["ifeval"])
        cyber = score_cyber(doms["cyber"])
        rep["per_model"][label] = {
            "coding": {"details": coding, **summarize(coding)},
            "reasoning": {"details": reasoning, **summarize(reasoning)},
            "ifeval": {"details": ifeval, **summarize(ifeval)},
            "cyber": {"details": cyber, **summarize(cyber)},
        }
        if label == "bf16":
            for dom, items in doms.items():
                bf16_responses[dom] = [_text(e).strip() for e in items]
    # cross-agreement: exact-string match vs bf16 per domain
    for label, m in rep["per_model"].items():
        if label == "bf16": continue
        agree = {}
        d2 = json.load(open([p for p in args.inputs if json.load(open(p))["label"] == label][0]))
        for dom, items in d2["domains"].items():
            vals = [_text(e).strip() for e in items]
            ref = bf16_responses.get(dom, [])
            agree[dom] = {"n": len(vals), "exact_match_vs_bf16":
                         sum(1 for a, b in zip(vals, ref) if a == b)}
        rep["cross_agreement_bf16"][label] = agree
    Path(args.out).write_text(json.dumps(rep, indent=2))
    # terse table
    print("\n===== FIDELITY TABLE =====")
    print(f"{'model':>10} {'coding':>8} {'reason':>8} {'ifeval':>8} {'cyber':>8}")
    for label, m in rep["per_model"].items():
        print(f"{label:>10} {m['coding']['pass']}/{m['coding']['n']:<4} "
              f"{m['reasoning']['pass']}/{m['reasoning']['n']:<4} "
              f"{m['ifeval']['pass']}/{m['ifeval']['n']:<4} "
              f"{m['cyber']['pass']}/{m['cyber']['n']:<4}")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
