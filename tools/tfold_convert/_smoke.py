"""Smoke: load oracle, convert, roundtrip, byte-exact compare."""
from __future__ import annotations

import pathlib

import tfold_convert as tc


def main() -> int:
    g = tc.load_oracle_mcg_block()
    print(f"codebook={g.codebook} halves={g.halves} bits={g.bits_per_weight} K={g.K} N={g.N}")
    print(f"trellis bytes={len(g.trellis_i16)} suh={len(g.suh_fp16)} svh={len(g.svh_fp16)} dq={len(g.dequant_reference_fp64)}")

    out = pathlib.Path("/tmp/exl3_oracle_mcg.tfold")
    hdr = tc.convert_group_to_tfold(g, out)
    print(f"wrote {out} size={out.stat().st_size}")
    for t in hdr["tensors"]:
        name = t["name"]
        print(
            f"  {name:8s}  dtype={t['dtype']:14s}  "
            f"shape={t['shape']}  offset={t['offset']:6d}  "
            f"nbytes={t['nbytes']:6d}  sha={t['sha256'][:16]}..."
        )

    bytes_back = tc.read_converted_group_bytes(out)
    import struct
    orig_map = {
        "__codebook__": struct.pack("<I", 0xCBAC1FED),
        "trellis": g.trellis_i16,
        "suh": g.suh_fp16,
        "svh": g.svh_fp16,
    }
    all_ok = True
    for name, raw in bytes_back.items():
        orig = orig_map[name]
        ok = raw == orig
        all_ok = all_ok and ok
        print(f"  {name:12s}: byte-exact={ok} len={len(raw)}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
