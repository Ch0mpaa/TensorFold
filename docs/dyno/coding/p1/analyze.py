"""Stage P1 analyzer: summarize attempts.jsonl."""
import json, sys
from pathlib import Path
from collections import Counter

_HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (_HERE / "attempts.jsonl").open()]
print(f"attempts: {len(rows)}")

status = Counter(r.get("status", "?") for r in rows)
print("\n## status distribution")
for s, n in status.most_common():
    print(f"  {s:30s} {n}")

print("\n## per-attempt detail")
print(f"  {'mode':8s} {'seed':5s} {'status':22s} {'patch?':>7s} {'apply?':>7s} {'f2p?':>5s} {'p2p?':>5s} {'tok':>4s} {'gen_s':>7s} {'f2p_s':>6s} {'p2p_s':>6s}")
for r in rows:
    print(f"  {r.get('mode',''):8s} {str(r.get('seed','')):5s} {r.get('status',''):22s} "
          f"{str(r.get('patch_extracted',False)):>7s} {str(r.get('patch_apply_rc','')):>7s} "
          f"{('OK' if r.get('f2p_rc')==0 else str(r.get('f2p_rc',''))):>5s} "
          f"{('OK' if r.get('p2p_rc')==0 else str(r.get('p2p_rc',''))):>5s} "
          f"{r.get('n_tokens','-'):>4} {r.get('gen_elapsed_s',''):>7} "
          f"{r.get('f2p_elapsed_s',''):>6} {r.get('p2p_elapsed_s',''):>6}")

print("\n## oracle summary")
valid_patch = sum(1 for r in rows if r.get("patch_extracted"))
applied = sum(1 for r in rows if r.get("patch_apply_rc") == 0)
f2p_ok = sum(1 for r in rows if r.get("f2p_rc") == 0)
p2p_ok = sum(1 for r in rows if r.get("p2p_rc") == 0)
resolved = sum(1 for r in rows if r.get("resolved") is True)
print(f"  extracted unified diff: {valid_patch} / {len(rows)}")
print(f"  patch applied cleanly:  {applied} / {len(rows)}")
print(f"  fail_to_pass passed:    {f2p_ok} / {len(rows)}")
print(f"  pass_to_pass passed:    {p2p_ok} / {len(rows)}")
print(f"  RESOLVED (both):        {resolved} / {len(rows)}")

tok = [r.get("n_tokens",0) for r in rows if r.get("n_tokens")]
if tok:
    print(f"  tokens generated: sum={sum(tok)}  mean={sum(tok)/len(tok):.1f}")
gen = [r.get("gen_elapsed_s",0) for r in rows if r.get("gen_elapsed_s")]
if gen:
    print(f"  gen elapsed sum={sum(gen):.1f}s  mean={sum(gen)/len(gen):.1f}s")

# Dump machine-readable
out = {
    "attempts": len(rows),
    "status_counts": dict(status),
    "oracle": {
        "valid_patch": valid_patch,
        "patch_applied": applied,
        "fail_to_pass_passed": f2p_ok,
        "pass_to_pass_passed": p2p_ok,
        "resolved": resolved,
    },
    "tokens_total": sum(tok),
    "tokens_mean": sum(tok)/len(tok) if tok else 0,
    "gen_elapsed_sum_s": sum(gen),
    "gen_elapsed_mean_s": sum(gen)/len(gen) if gen else 0,
}
(_HERE / "analysis.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {_HERE/'analysis.json'}")
