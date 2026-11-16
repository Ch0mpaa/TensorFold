"""Reduce race_a.jsonl to summary stats per label + a per-phase table."""
import argparse, json, statistics
from pathlib import Path

def pct(xs, p):
    s = sorted(xs); k = max(0, min(len(s)-1, int(round((p/100)*(len(s)-1)))))
    return s[k]

def summarize(rows):
    def _s(key, mul=1.0, cast=float):
        xs = [cast(r[key]) * mul for r in rows if r.get(key) not in (None, "", "0", 0)]
        if not xs:
            # include zero-valued keys for sft (fnv_us = "0" means N/A)
            xs2 = [cast(r[key]) * mul for r in rows]
            return {"min": min(xs2), "med": statistics.median(xs2), "max": max(xs2),
                    "mean": round(statistics.mean(xs2),3), "p95": pct(xs2, 95)}
        return {"min": round(min(xs),3), "med": round(statistics.median(xs),3),
                "max": round(max(xs),3), "mean": round(statistics.mean(xs),3),
                "p95": round(pct(xs, 95),3)}
    return {
        "n": len(rows),
        "wall_s":      _s("wall_s"),
        "user_s":      _s("user_s"),
        "sys_s":       _s("sys_s"),
        "scenario_ms": _s("scenario_ms", cast=lambda x: int(x) if x else 0),
        "fnv_ms":      _s("fnv_us", 1e-3),
        "header_parse_ms": _s("header_parse_us", 1e-3),
        "preamble_us": _s("preamble_us", cast=lambda x: int(x) if x else 0),
        "mmap_us":     _s("mmap_us", cast=lambda x: int(x) if x else 0),
        "peak_gib":    _s("peak_gib"),
        "rss_kb":      _s("rss_kb"),
        "unique_dump_sha16": sorted({r["dump_sha16"] for r in rows}),
        "unique_tokens":     sorted({r["tokens"] for r in rows}),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out",   required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    # drop the warmup (run=0)
    rows_timed = [r for r in rows if r["run"] != 0 and r["rc"] == 0]
    by_label = {}
    for r in rows_timed: by_label.setdefault(r["label"], []).append(r)
    rep = {
        "jsonl": args.jsonl,
        "total_rows": len(rows),
        "timed_rows": len(rows_timed),
        "by_label": {k: summarize(v) for k, v in by_label.items()},
        "winner_per_metric": {},
        "parity": {
            "unique_logit_sha_across_all_labels": sorted({r["dump_sha16"] for r in rows_timed}),
            "unique_tokens_across_all_labels":    sorted({r["tokens"] for r in rows_timed}),
            "all_runs_identical": len({r["dump_sha16"] for r in rows_timed}) == 1,
        },
    }
    # Per-metric winner comparison (lower is better)
    def winner(a, b, metric, path_a="sft", path_b="tfold"):
        va = rep["by_label"][path_a][metric]["med"]
        vb = rep["by_label"][path_b][metric]["med"]
        if abs(va - vb) < max(0.001, 0.005 * max(abs(va), abs(vb))):
            return f"TIE ({path_a}={va} {path_b}={vb})"
        return f"{path_a} wins ({va} vs {vb})" if va < vb else f"{path_b} wins ({va} vs {vb})"
    for m in ("wall_s", "scenario_ms", "peak_gib"):
        rep["winner_per_metric"][m] = winner(rep["by_label"], rep["by_label"], m)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
