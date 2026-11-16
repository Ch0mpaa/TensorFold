"""Stage P2 analyzer: summarize agentic attempts.jsonl."""
import json
from pathlib import Path
from collections import Counter

_HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (_HERE / "attempts.jsonl").open()]
print(f"attempts: {len(rows)}")

resolved = sum(1 for r in rows if r["verdict"]["resolved"])
finished = sum(1 for r in rows if r.get("model_finished"))
print(f"\n## oracle verdict")
print(f"  resolved (all 4 checks pass): {resolved} / {len(rows)}")
print(f"  model emitted finish:         {finished} / {len(rows)}")

print(f"\n## per-attempt")
print(f"  {'mode':8s} {'seed':5s} {'turns':>5s} {'fin?':>5s} {'oldkey':>7s} {'idiom':>6s} {'compile':>8s} {'pytest':>7s} {'resolved':>9s} {'tok':>5s} {'s':>6s}")
for r in rows:
    v = r["verdict"]
    print(f"  {r['mode']:8s} {str(r['seed']):5s} {r['turns_used']:>5d} {str(r.get('model_finished','?')):>5s} "
          f"{v['old_key_hits']:>7d} {str(v['idiom_present']):>6s} {str(v['compiles']):>8s} "
          f"{v['pytest_rc']:>7d} {str(v['resolved']):>9s} {r['total_tokens']:>5d} {r['total_elapsed_s']:>6.1f}")

# Tool-call distribution
tool_counter = Counter()
for r in rows:
    for t, n in (r.get("tool_counts") or {}).items():
        tool_counter[t] += n
print(f"\n## tool-call distribution across all attempts")
for t, n in tool_counter.most_common():
    print(f"  {t:12s} {n}")

# Latency / tokens summary
tok = [r["total_tokens"] for r in rows]
gen = [r["total_gen_s"] for r in rows]
ela = [r["total_elapsed_s"] for r in rows]
turns_used = [r["turns_used"] for r in rows]
print(f"\n## latency/tokens")
print(f"  tokens  mean={sum(tok)/len(tok):.1f}  sum={sum(tok)}")
print(f"  gen_s   mean={sum(gen)/len(gen):.2f}  sum={sum(gen):.1f}")
print(f"  elapsed mean={sum(ela)/len(ela):.2f}")
print(f"  turns   mean={sum(turns_used)/len(turns_used):.1f}  max={max(turns_used)}")

out = {
    "attempts": len(rows),
    "resolved": resolved,
    "model_finished": finished,
    "first_try_success": sum(1 for r in rows if r["verdict"]["resolved"] and r["turns_used"] <= 5),
    "with_recovery_success": sum(1 for r in rows if r["verdict"]["resolved"] and r["turns_used"] > 5),
    "tool_counts": dict(tool_counter),
    "tokens_mean": sum(tok)/len(tok),
    "tokens_sum": sum(tok),
    "elapsed_s_mean": sum(ela)/len(ela),
    "turns_mean": sum(turns_used)/len(turns_used),
    "turns_max": max(turns_used),
    "per_attempt": [
        {"mode": r["mode"], "seed": r["seed"], "turns": r["turns_used"],
         "finished": r.get("model_finished", False), "verdict": r["verdict"]}
        for r in rows
    ],
}
(_HERE / "analysis.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {_HERE/'analysis.json'}")
