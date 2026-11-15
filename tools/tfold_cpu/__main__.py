"""CLI: python -m tfold_cpu [--path PATH] [--pin SHA] [--repeat N] [--json]

Default path is examples/demo.tfold relative to the repo root.
Prints:
    - validated file identity
    - cold load+run time (one-shot)
    - warm run median across --repeat runs (default 10)
    - peak RSS (Linux/macOS; "unavailable" on others)
    - the resulting y vector
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource as _resource_mod  # may be unavailable on Windows
import sys
import time

from .executor import load_and_run


def _peak_rss_kb() -> int | None:
    try:
        ru = _resource_mod.getrusage(_resource_mod.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS; approximate — label the output
        return int(ru.ru_maxrss)
    except Exception:
        return None


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tfold_cpu")
    ap.add_argument("--path", type=pathlib.Path, default=None,
                    help="Path to a .tfold file (default: examples/demo.tfold)")
    ap.add_argument("--pin", type=str, default=None,
                    help="Required source_sha256 (32-byte hex); reader rejects on mismatch")
    ap.add_argument("--repeat", type=int, default=10,
                    help="Warm-run count for timing median (default 10)")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of text")
    args = ap.parse_args(argv)

    path = args.path
    if path is None:
        root = pathlib.Path(__file__).resolve().parents[2]
        path = root / "examples" / "demo.tfold"

    if not path.exists():
        print(f"ERROR: no such file: {path}", file=sys.stderr)
        return 2

    # cold load+run
    t0 = time.perf_counter()
    y = load_and_run(path, pin_source_sha256=args.pin)
    t1 = time.perf_counter()
    cold_s = t1 - t0

    # warm repeats
    warm_times: list[float] = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        y = load_and_run(path, pin_source_sha256=args.pin)
        t1 = time.perf_counter()
        warm_times.append(t1 - t0)

    peak_kb = _peak_rss_kb()
    report = {
        "file": str(path),
        "dim": len(y),
        "cold_load_run_s": cold_s,
        "warm_runs": args.repeat,
        "warm_median_s": _median(warm_times),
        "warm_min_s": min(warm_times) if warm_times else None,
        "warm_max_s": max(warm_times) if warm_times else None,
        "peak_rss_kb_label": "Linux: KiB, macOS: bytes, else: n/a",
        "peak_rss_raw": peak_kb,
        "output": y,
        "note": "Python CPU-reference performance only; not optimized native inference.",
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"file           : {report['file']}")
        print(f"output dim     : {report['dim']}")
        print(f"cold load+run  : {report['cold_load_run_s']:.6f} s")
        print(f"warm median    : {report['warm_median_s']:.6f} s  (n={args.repeat})")
        print(f"warm min/max   : {report['warm_min_s']:.6f} / {report['warm_max_s']:.6f} s")
        print(f"peak rss raw   : {report['peak_rss_raw']}  ({report['peak_rss_kb_label']})")
        print(f"output         : {y}")
        print(f"note           : {report['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
