"""Lossless Safetensors -> TFOLD v0.2 converter.

Design:
  - Read original safetensors header (JSON) + payload bytes.
  - Preserve every tensor's name, shape, dtype and raw byte payload.
  - Emit a .tfold file with the proven v0.2 container:
      magic "TFOLD" (5 bytes)
      version  u8       (0x02)
      reserved 6 bytes  (0)
      hdr_len  u32 LE
      <json header>
      <binary payload bytes, tensor-by-tensor in the order of the header>
  - quant.scheme = "none" (identity). No silent dtype casts anywhere.
  - For every tensor, hash the payload bytes (sha256) and record it in the header.
  - Also write a sidecar manifest file with per-tensor hashes for independent checks.

Supported dtypes (TFOLD v0.2 native): bf16, fp16, fp32, i32, u32, i8, u8, i16, u16, i64, u64, bool.
If the safetensors contains a dtype outside this allow-list the converter ABORTS with a clear
error message rather than performing any cast. See README section "Format extensions".
"""
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys, time
from pathlib import Path

ST_DTYPE_TO_TFOLD = {
    "BF16":  "bf16",
    "F16":   "fp16",
    "F32":   "fp32",
    "F64":   "fp64",
    "I8":    "i8",  "U8":   "u8",
    "I16":   "i16", "U16":  "u16",
    "I32":   "i32", "U32":  "u32",
    "I64":   "i64", "U64":  "u64",
    "BOOL":  "bool",
}

CONVERTER_NAME = "tfold-convert-lossless"
CONVERTER_VERSION = "0.1.0"
FORMAT_VERSION = "0.2"  # same container as existing exl3-mcg files; no bump needed for unquantized

def read_safetensors_header(path: Path):
    """Return (header_dict, header_byte_length, data_offset)."""
    with path.open("rb") as f:
        hdr_len = int.from_bytes(f.read(8), "little")
        hdr_bytes = f.read(hdr_len)
    header = json.loads(hdr_bytes.decode("utf-8"))
    # metadata is stored at the "__metadata__" key per the spec
    return header, hdr_len, 8 + hdr_len

def shard_sha256(path: Path, chunk=1<<20):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True, help="path to the original safetensors shard")
    ap.add_argument("--config", required=True, help="path to the original config.json")
    ap.add_argument("--tokenizer", required=False, help="path to the original tokenizer.json (for sha)")
    ap.add_argument("--license", required=False, help="path to LICENSE file (for sha)")
    ap.add_argument("--out", required=True, help="path to write the .tfold")
    ap.add_argument("--manifest", required=True, help="path to write the sidecar manifest JSON")
    ap.add_argument("--model-name", default="Qwen3.5-2B-orig-bf16")
    args = ap.parse_args()

    st_path = Path(args.safetensors)
    out_path = Path(args.out)
    mani_path = Path(args.manifest)

    print(f"[1/5] reading safetensors header: {st_path}", file=sys.stderr)
    header, hdr_byte_len, data_offset = read_safetensors_header(st_path)
    source_meta = header.pop("__metadata__", {}) if "__metadata__" in header else {}
    tensor_count = len(header)
    print(f"  tensors: {tensor_count}", file=sys.stderr)

    print(f"[2/5] hashing shard bytes (streaming)", file=sys.stderr)
    t0 = time.time()
    source_sha = shard_sha256(st_path)
    print(f"  source_sha256: {source_sha}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    print(f"[3/5] validating dtypes + computing per-tensor sha256 (no casts)", file=sys.stderr)
    cfg = json.loads(Path(args.config).read_text())
    tok_sha = shard_sha256(Path(args.tokenizer)) if args.tokenizer else None
    lic_sha = shard_sha256(Path(args.license)) if args.license else None

    # Build TFOLD tensor entries. Preserve original name/shape; translate dtype 1:1.
    tfold_tensors = []
    running_offset = 0  # computed after we know the header size; we'll rewrite below
    with st_path.open("rb") as f:
        for name, meta in header.items():
            st_dtype = meta["dtype"]
            if st_dtype not in ST_DTYPE_TO_TFOLD:
                raise SystemExit(f"ABORT: dtype {st_dtype!r} for tensor {name!r} is not in the TFOLD v0.2 allow-list "
                                 f"({sorted(ST_DTYPE_TO_TFOLD)}). No silent cast; declare a format extension or retry.")
            shape = meta["shape"]
            start = data_offset + meta["data_offsets"][0]
            end   = data_offset + meta["data_offsets"][1]
            nbytes = end - start
            # stream + hash the payload
            h = hashlib.sha256()
            left = nbytes; f.seek(start)
            while left > 0:
                chunk = f.read(min(left, 1<<20))
                h.update(chunk); left -= len(chunk)
            tfold_tensors.append({
                "name": name,
                "dtype": ST_DTYPE_TO_TFOLD[st_dtype],
                "shape": shape,
                "offset": 0,            # will be rewritten after header size is known
                "nbytes": nbytes,
                "sha256": h.hexdigest(),
                # pin the original source byte range for audit
                "source_byte_range": [start, end],
            })

    print(f"[4/5] building TFOLD header JSON", file=sys.stderr)
    model_block = {
        "name": args.model_name,
        "arch": "qwen3_5",   # from the official config (not silently renamed to qwen3)
        "parameters": sum((eval_shape(t["shape"]) for t in tfold_tensors), 0),
        "vocab_size": cfg.get("text_config", cfg).get("vocab_size", cfg.get("vocab_size", -1)),
        "context_length": cfg.get("text_config", cfg).get("max_position_embeddings", -1),
        "source_sha256": source_sha,
        "source_url": "Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc/model.safetensors-00001-of-00001.safetensors",
        "upstream_license": "apache-2.0",
    }
    quant_block = {"scheme": "none"}
    tokenizer_block = {"type": "qwen3_5-tokenizer", "sha256": tok_sha} if tok_sha else {}
    hdr_obj = {
        "tfold":  {"version": FORMAT_VERSION, "writer": f"{CONVERTER_NAME}@{CONVERTER_VERSION}"},
        "model":  model_block,
        "tokenizer": tokenizer_block,
        "quant":  quant_block,
        "tensors": tfold_tensors,
    }
    header_json = json.dumps(hdr_obj, separators=(",",":"))
    # Iterate until stable: setting offsets may grow the JSON, requiring new offsets.
    # Container layout per Zig spec src/core/tfold_file.zig:
    #   bytes 0..8  : MAGIC "TFOLD\x02\x00\x00"
    #   bytes 8..12 : version u32 LE = 0x00020000
    #   bytes 12..16: header length u32 LE
    #   16..16+hdr_len: header JSON
    #   padded to ALIGN (64) boundary
    #   body
    #   trailer: last 8 bytes = FNV-1a-64 LE over every byte before it
    ALIGN = 64
    for _ in range(8):
        preamble_and_header = 16 + len(header_json.encode())
        data_start = (preamble_and_header + ALIGN - 1) // ALIGN * ALIGN
        off = data_start
        for t in tfold_tensors:
            off = (off + ALIGN - 1) // ALIGN * ALIGN
            t["offset"] = off
            off += t["nbytes"]
        new_json = json.dumps(hdr_obj, separators=(",",":"))
        if new_json == header_json: break
        header_json = new_json
    else:
        raise SystemExit("ABORT: header length did not stabilize after 8 iterations")
    header_bytes = header_json.encode()
    preamble_and_header = 16 + len(header_bytes)
    data_start = (preamble_and_header + ALIGN - 1) // ALIGN * ALIGN
    pad_nbytes = data_start - preamble_and_header

    print(f"[5/5] writing {out_path}", file=sys.stderr)
    t0 = time.time()
    with out_path.open("wb") as fo, st_path.open("rb") as fi:
        # preamble per spec
        fo.write(b"TFOLD\x02\x00\x00")                      # 8 bytes magic
        fo.write((0x00020000).to_bytes(4, "little"))        # 4 bytes version u32 LE
        fo.write(len(header_bytes).to_bytes(4, "little"))   # 4 bytes hdr_len u32 LE
        fo.write(header_bytes)                              # JSON header
        if pad_nbytes > 0:
            fo.write(b"\x00" * pad_nbytes)                  # 64-byte alignment pad
        cur = data_start
        for t in tfold_tensors:
            pad = t["offset"] - cur
            if pad > 0: fo.write(b"\x00" * pad); cur += pad
            s, e = t["source_byte_range"]
            fi.seek(s)
            left = t["nbytes"]
            while left > 0:
                chunk = fi.read(min(left, 1<<20))
                fo.write(chunk); left -= len(chunk); cur += len(chunk)
    # Append FNV-1a-64 trailer via compiled C helper (Python byte-loop is 50x slower).
    # The helper is a tiny, auditable O(file-size) program with the standard
    # FNV-1a-64 constants (offset 0xcbf29ce484222325, prime 0x100000001b3).
    import subprocess
    helper = os.environ.get("TFOLD_FNV_HELPER", "/tmp/fnv_append")
    if not Path(helper).exists():
        raise SystemExit(f"ABORT: FNV helper {helper} missing; compile with `gcc -O3 -o {helper} fnv_append.c`")
    rc = subprocess.run([helper, str(out_path)], check=True)
    elapsed = time.time() - t0
    out_size = out_path.stat().st_size
    print(f"  wrote {out_size} bytes in {elapsed:.1f}s", file=sys.stderr)

    # Manifest sidecar: all tensor hashes + container-level meta
    manifest = {
        "converter": {"name": CONVERTER_NAME, "version": CONVERTER_VERSION},
        "source": {
            "safetensors_path": str(st_path),
            "safetensors_sha256": source_sha,
            "config_path": str(args.config),
            "tokenizer_path": str(args.tokenizer) if args.tokenizer else None,
            "tokenizer_sha256": tok_sha,
            "license_path": str(args.license) if args.license else None,
            "license_sha256": lic_sha,
            "upstream_repo": "Qwen/Qwen3.5-2B",
            "upstream_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "upstream_license_id": "apache-2.0",
        },
        "tfold": {
            "format_version": FORMAT_VERSION,
            "output_path": str(out_path),
            "output_size_bytes": out_size,
            "quant_scheme": "none",
            "tensor_count": len(tfold_tensors),
        },
        "tensors": {t["name"]: {"dtype": t["dtype"], "shape": t["shape"], "nbytes": t["nbytes"], "sha256": t["sha256"]}
                    for t in tfold_tensors},
    }
    mani_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest at {mani_path}", file=sys.stderr)

def eval_shape(shape):
    n = 1
    for d in shape: n *= d
    return n

if __name__ == "__main__":
    main()
