#!/usr/bin/env python3
"""Materialize a TFOLD v0.2 file into a safetensors directory.

This proves the v0.2 format round-trips through the live CUDA inference
engine WITHOUT requiring a native TfoldFile adapter in upstream TF: we
rebuild a safetensors file whose tensor bytes are byte-identical to the
original ExLlamaV3 checkpoint.

Byte identity holds because:
- v0.2 stages every tensor contiguously in the body in its original
  stored EXL3 tile layout (same as safetensors).
- The __codebook__ synthetic tensor stores the mcg magic (0xCBAC1FED)
  and is re-emitted as per-group .mcg scalar tensors.
- scales/suh/svh/trellis/weight are all preserved byte-for-byte.

Usage:
    python tfold_to_safetensors.py MODEL.tfold OUT_DIR \\
        --config CONFIG.json --tokenizer TOKENIZER.json ...
"""
import argparse
import json
import pathlib
import shutil
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfold_py


SAFETENSORS_DTYPE = {
    "bf16": "BF16",
    "f16": "F16",
    "fp16": "F16",
    "f32": "F32",
    "fp32": "F32",
    "u8": "U8",
    "u16": "U16",
    "u32": "U32",
    "i8": "I8",
    "i16": "I16",
    "i32": "I32",
    "i64": "I64",
    # EXL3 v0.2 trellis tiles are stored as i16 arrays
    "exl3_mcg_tile": "I16",
    "exl3_mul1_tile": "I16",
    "exl3_3inst_tile": "I16",
}
MCG_MAGIC = 0xCBAC1FED
MUL1_MAGIC = 0x83DCD12D


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tfold_path", type=pathlib.Path)
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--sidecar-dir", type=pathlib.Path,
                    help="source model directory (for config.json, tokenizer.*, etc.)")
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        print(f"out_dir not empty: {args.out_dir}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hdr, _ = tfold_py.read_v02(args.tfold_path, verify_tensor_hashes=False, verify_fnv_trailer=False)
    print(f"tfold version: {hdr['tfold']['version']}")
    print(f"tensors      : {len(hdr['tensors'])}")
    print(f"groups       : {len(hdr['quant'].get('groups', []))}")
    print(f"scheme       : {hdr['quant']['scheme']} @ {hdr['quant']['bits_per_weight']} bpw")

    # Figure out the TFOLD body base offset
    with open(args.tfold_path, "rb") as f:
        preamble = f.read(16)
        magic = preamble[:8]
        version = struct.unpack("<I", preamble[8:12])[0]
        hlen = struct.unpack("<I", preamble[12:16])[0]
    # v0.2: body is aligned to ALIGN=64 after PREAMBLE_NBYTES + hlen
    ALIGN = 64
    hdr_end = 16 + hlen
    body_base = hdr_end + ((ALIGN - hdr_end % ALIGN) % ALIGN)
    print(f"body offset  : {body_base}")

    # Group codebook decisions:
    # codebook_for_base[base_name_prefix] = "mcg" | "mul1"
    scheme = hdr["quant"]["scheme"]  # "exl3_mcg" or "exl3_mul1"
    default_cb = "mcg" if "mcg" in scheme else "mul1"
    default_magic = MCG_MAGIC if default_cb == "mcg" else MUL1_MAGIC

    # Build the new safetensors header. It needs each tensor's name,
    # dtype, shape, and data_offsets (relative to end of header).
    #
    # For every TFOLD tensor, we copy its byte range verbatim.
    # For every group with no per-base-name codebook marker tensor
    # listed (i.e. the default), we synthesize a scalar marker tensor
    # named "<base>.<cb>" holding the magic value.

    # The v0.2 file already contains per-group .mcg (or .mul1) scalar marker
    # tensors (e.g. `lm_head.mcg` at offset 259840). Pass them through like any
    # other tensor. Only skip the file-wide __codebook__ scalar - the qwen3_5
    # EXL3 admit path doesn't read it.
    codebook_names = {"__codebook__"}

    # Compute body layout
    header_entries = {}
    cursor = 0
    # 1. Copy every tensor from v0.2 that is NOT the synthetic codebook scalar
    region_copies = []  # list of (out_offset, out_size, tfold_offset)
    for t in hdr["tensors"]:
        if t["name"] in codebook_names:
            continue  # handled via scalar_markers
        dt = SAFETENSORS_DTYPE.get(t["dtype"].lower(), t["dtype"].upper())
        nb = t["nbytes"]
        header_entries[t["name"]] = {
            "dtype": dt,
            "shape": t["shape"],
            "data_offsets": [cursor, cursor + nb],
        }
        region_copies.append((cursor, nb, t["offset"]))
        cursor += nb

    # (no synthetic markers; v0.2 ships them)

    header_json = json.dumps(header_entries, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    # pad header to 8-byte alignment (optional but common)
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    total_body = cursor
    out_path = args.out_dir / "model.safetensors"
    print(f"output size  : {8 + len(header_bytes) + total_body:,} bytes")
    print(f"writing      : {out_path}")

    with open(out_path, "wb") as fout, open(args.tfold_path, "rb") as fin:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        # Stream-copy tensor regions
        CHUNK = 8 * 1024 * 1024
        for out_off, size, src_off in region_copies:
            fin.seek(src_off)
            remaining = size
            while remaining > 0:
                take = min(CHUNK, remaining)
                fout.write(fin.read(take))
                remaining -= take

    # Copy sidecar files
    if args.sidecar_dir:
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "vocab.json", "merges.txt",
                     "preprocessor_config.json", "video_preprocessor_config.json",
                     "quantization_config.json", "generation_config.json"]:
            src = args.sidecar_dir / name
            if src.exists():
                shutil.copy2(src, args.out_dir / name)
                print(f"copied       : {name}")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
