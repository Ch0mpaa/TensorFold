"""Open the converted 3.2 GiB Qwen3.5-2B .tfold v0.2 file and verify
full integrity + per-tensor SHA. Timing included.

Run:
    PYTHONPATH=tools python -m tfold_stage6b.verify_converted \\
        /tmp/Qwen3.5-2B-exl3-mcg.tfold
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import traceback

import tfold_py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument("--skip-sha", action="store_true", help="skip per-tensor SHA check")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"missing: {args.path}", file=sys.stderr)
        return 2

    size = args.path.stat().st_size
    print(f"file   : {args.path}")
    print(f"size   : {size:,} bytes ({size/(1<<30):.3f} GiB)")

    try:
        t0 = time.perf_counter()
        hdr, tensors = tfold_py.read_v02(args.path, verify_tensor_hashes=False)
        t1 = time.perf_counter()
    except Exception as e:
        traceback.print_exc()
        return 1

    print(f"parse  : {(t1-t0)*1000:.0f} ms (header + layout + FNV trailer; no per-tensor SHA)")
    print(f"version: {hdr['tfold']['version']}")
    print(f"scheme : {hdr['quant']['scheme']} @ {hdr['quant']['bits_per_weight']} bpw")
    print(f"tensors: {len(hdr['tensors'])}")
    print(f"groups : {len(hdr['quant'].get('groups', []))}")
    print(f"overrs : {len(hdr['quant'].get('overrides', []))}")
    print(f"source : {hdr['model']['source_sha256']}")
    print(f"vocab  : {hdr['model']['vocab_size']}")
    print(f"params : {hdr['model']['parameters']:,}")

    if not args.skip_sha:
        try:
            t0 = time.perf_counter()
            _hdr2, _ = tfold_py.read_v02(args.path, verify_tensor_hashes=True)
            t1 = time.perf_counter()
            print(f"full SHA-256 of all {len(hdr['tensors'])} tensors in {t1-t0:.1f} s")
        except Exception:
            traceback.print_exc()
            return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
