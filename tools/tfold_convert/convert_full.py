"""Convert the full Qwen3.5-2B-exl3-8.00bpw safetensors into a single
.tfold v0.2 container. Preserves every tensor byte-for-byte.

Usage:
    python -m tfold_convert.convert_full \\
        --model-dir D:/op13-root/models/Qwen3.5-2B-exl3-8.00bpw \\
        --out       examples/Qwen3.5-2B-exl3-mcg.tfold

Reads:
    model.safetensors    the actual weights
    quantization_config.json  per-tensor quant info

Writes:
    one .tfold v0.2 file with every stored tensor as its own tfold tensor,
    quant.overrides for anything whose bits differ from the file default,
    and quant.groups listing each EXL3 group's (trellis, suh, svh, [bias])
    with K, N dimensions.

The resulting file is ~2.77 GB. We stream to avoid loading everything
into Python memory at once. All SHA-256 digests are computed during the
single pass over each tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tfold_py._hash import sha256_hex, fnv1a_64_chunks
from tfold_py._v02 import MAGIC_V02, SPEC_VERSION_U32_V02
from tfold_py._version import ALIGN, PREAMBLE_NBYTES, TRAILER_NBYTES
from tfold_py._validate import load_schema  # not used directly; keeps the import tested
from tfold_py._v02 import validate_header_v02


# Qwen3.5-2B EXL3 provenance (same pins as Stage 5 oracle.py)
MODEL_SOURCE_SHA = "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f"
MODEL_URL = "turboderp/Qwen3.5-2B-exl3@8.00bpw/model.safetensors"


# ------------------------- safetensors header reader -------------------------


def read_safetensors_header(path: pathlib.Path) -> tuple[int, dict]:
    """Return (data_region_offset, header_dict)."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(n)
    return 8 + n, json.loads(header_bytes)


_SAFETENSORS_DTYPE_MAP = {
    "F16":  ("fp16", 2),
    "BF16": ("bf16", 2),
    "F32":  ("fp32", 4),
    "F64":  (None, 8),       # skipped; v0.1/v0.2 don't register it
    "I16":  ("i16",  2),
    "I32":  ("i32",  4),
    "I64":  (None, 8),       # dropped — no v0.2 i64 dtype; no Qwen3.5 tensor uses it
    "U8":   ("u8",   1),
    "I8":   ("i8",   1),
    "U16":  ("u16",  2),
    "U32":  ("u32",  4),
}


def plan_tensors(st_header: dict, quant_cfg: dict) -> list[dict]:
    """Produce the tfold tensor list from the safetensors header + quantization_config.
    Tensor dtypes: safetensors native for plain weights; `exl3_mcg_tile`
    for trellis entries (recognized by name suffix `.trellis`)."""
    tensors: list[dict] = []
    tensor_names = [n for n in st_header if n != "__metadata__"]
    # Enforce deterministic order
    tensor_names.sort()

    # Build a mapping of tensor name → mcg marker presence by scanning the quant config
    mcg_names = set()
    for group_name, group_data in (quant_cfg.get("tensor_storage") or {}).items():
        for n in group_data.get("stored_tensors", {}):
            if n.endswith(".mcg"):
                mcg_names.add(n)

    for n in tensor_names:
        entry = st_header[n]
        st_dtype = entry["dtype"]
        if st_dtype not in _SAFETENSORS_DTYPE_MAP:
            raise ValueError(f"tensor {n!r} uses unsupported safetensors dtype {st_dtype}")
        mapped, esize = _SAFETENSORS_DTYPE_MAP[st_dtype]
        shape = list(entry["shape"])
        off0, off1 = entry["data_offsets"]
        nbytes = off1 - off0

        if nbytes == 0:
            # EXL3 marker tensor (`*.mcg`/`*.mul1`/`*.3inst`) — zero bytes
            # v0.2 schema requires nbytes >= 1. Encode markers as 1-byte u8
            # placeholders so they remain visible to downstream readers.
            # The actual codebook identity lives in quant.scheme.
            tensors.append({
                "name": n,
                "dtype": "u8",
                "shape": [1],
                "nbytes": 1,
                "sha256": None,
                "source_offset": None,
                "marker_name": True,
            })
            continue

        if n.endswith(".trellis"):
            dtype = "exl3_mcg_tile"
        elif mapped is None:
            raise ValueError(f"tensor {n!r} dtype {st_dtype} not expressible in v0.2")
        else:
            dtype = mapped
        tensors.append({
            "name": n,
            "dtype": dtype,
            "shape": shape,
            "nbytes": nbytes,
            "sha256": None,
            "source_offset": off0,
            "marker_name": False,
        })
    return tensors


def build_groups(tensor_names_set: set[str]) -> list[dict]:
    """Group EXL3 triples by their common dotted prefix."""
    groups = []
    for n in sorted(tensor_names_set):
        if not n.endswith(".trellis"):
            continue
        prefix = n[:-len(".trellis")]
        suh = f"{prefix}.suh"
        svh = f"{prefix}.svh"
        bias = f"{prefix}.bias"
        if suh in tensor_names_set and svh in tensor_names_set:
            groups.append({
                "name": prefix,
                "trellis": n,
                "suh": suh,
                "svh": svh,
                "bias": bias if bias in tensor_names_set else None,
                "K": 0,  # filled in later from shapes
                "N": 0,
            })
    return groups


def _pad_to(n: int, align: int) -> int:
    r = n % align
    return 0 if r == 0 else align - r


def write_v02_streaming(
    out_path: pathlib.Path,
    source_safetensors: pathlib.Path,
    source_data_offset: int,
    tensors: list[dict],
    header_fixed: dict,
) -> dict:
    """Pass 1: compute per-tensor SHA-256 while copying bytes to a temp
    layout file; Pass 2: emit preamble + header + aligned tensor region.

    Memory bound: one 8 MB chunk at a time."""
    # First compute SHAs and total layout
    chunk = 8 * 1024 * 1024
    layout_tmp = out_path.with_suffix(".tmp-tensors")
    print(f"pass 1: hash + stage {len(tensors)} tensors → {layout_tmp}")
    t_start = time.perf_counter()

    cur_offset = 0  # offset within the tensor region (relative)
    offsets_within_region: list[int] = []
    with open(source_safetensors, "rb") as src, open(layout_tmp, "wb") as dst:
        for i, spec in enumerate(tensors):
            # pad to ALIGN
            pad = _pad_to(cur_offset, ALIGN)
            if pad:
                dst.write(b"\x00" * pad)
                cur_offset += pad
            offsets_within_region.append(cur_offset)

            if spec["marker_name"]:
                # 1-byte placeholder; content = 0
                data = b"\x00"
                dst.write(data)
                cur_offset += 1
                spec["sha256"] = sha256_hex(data)
                continue

            src.seek(source_data_offset + spec["source_offset"])
            h = hashlib.sha256()
            remaining = spec["nbytes"]
            while remaining:
                to_read = min(chunk, remaining)
                b = src.read(to_read)
                if len(b) != to_read:
                    raise RuntimeError("short read from safetensors source")
                h.update(b)
                dst.write(b)
                remaining -= to_read
            spec["sha256"] = h.hexdigest()
            cur_offset += spec["nbytes"]

            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(tensors)} tensors staged "
                      f"({cur_offset/(1<<30):.2f} GiB)")
    tensor_region_bytes = cur_offset
    print(f"pass 1 done in {time.perf_counter()-t_start:.1f}s, "
          f"tensor region = {tensor_region_bytes/(1<<30):.3f} GiB")

    # Fill in header tensor list with placeholder offsets, iterate until stable
    hdr = dict(header_fixed)
    hdr["tensors"] = []
    for spec, roff in zip(tensors, offsets_within_region):
        hdr["tensors"].append({
            "name": spec["name"],
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "offset": 0,  # patched
            "nbytes": 1 if spec["marker_name"] else spec["nbytes"],
            "sha256": spec["sha256"],
        })

    # Compute final header_length by serializing twice if it changes.
    for _ in range(4):
        hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        header_end = PREAMBLE_NBYTES + len(hj)
        region_start = header_end + _pad_to(header_end, ALIGN)
        changed = False
        for spec, roff in zip(hdr["tensors"], offsets_within_region):
            true_off = region_start + roff
            if spec["offset"] != true_off:
                spec["offset"] = true_off
                changed = True
        if not changed:
            break

    # fill codebook pointers for the exl3_mcg scheme (we store a 4-byte __codebook__)
    # per-tensor for Qwen3.5 means we'd have 151 codebooks which is silly; we instead
    # pin the file-wide scheme to exl3_mcg and set a dummy codebook_* trio pointing
    # at the first tensor. The v0.2 schema requires those keys for exl3_* schemes.
    # Pick the smallest: the first trellis's first 4 bytes.

    # Validate the header before writing
    validate_header_v02(hdr)

    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    header_end = PREAMBLE_NBYTES + len(hj)
    pad_after_header = b"\x00" * _pad_to(header_end, ALIGN)

    # Pass 2: write the file
    print(f"pass 2: emit .tfold v0.2 → {out_path}")
    t2 = time.perf_counter()
    with open(out_path, "wb") as dst, open(layout_tmp, "rb") as tmp:
        dst.write(MAGIC_V02)
        dst.write(struct.pack("<I", SPEC_VERSION_U32_V02))
        dst.write(struct.pack("<I", len(hj)))
        dst.write(hj)
        dst.write(pad_after_header)
        # Copy staged tensor region verbatim
        while True:
            b = tmp.read(chunk)
            if not b:
                break
            dst.write(b)

    layout_tmp.unlink()

    # Compute FNV-1a-64 trailer in a second pass over the file
    print(f"pass 3: FNV-1a-64 trailer...")
    h = 0xCBF2_9CE4_8422_2325
    PRIME = 0x100000001B3
    MASK = (1 << 64) - 1
    with open(out_path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            for byte in b:
                h ^= byte
                h = (h * PRIME) & MASK
    with open(out_path, "ab") as f:
        f.write(struct.pack("<Q", h))

    print(f"wrote {out_path} ({out_path.stat().st_size/(1<<30):.3f} GiB) "
          f"in {time.perf_counter()-t2:.1f}s")
    return hdr


def main() -> int:
    ap = argparse.ArgumentParser(prog="convert_full")
    ap.add_argument("--model-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    st_path = args.model_dir / "model.safetensors"
    qc_path = args.model_dir / "quantization_config.json"
    assert st_path.exists(), f"{st_path} not found"
    assert qc_path.exists(), f"{qc_path} not found"

    data_offset, st_header = read_safetensors_header(st_path)
    quant_cfg = json.loads(qc_path.read_text())

    tensors = plan_tensors(st_header, quant_cfg)
    print(f"planned {len(tensors)} tensors")

    tensor_name_set = {t["name"] for t in tensors}
    groups = build_groups(tensor_name_set)
    # Backfill K, N from trellis shape [K/16, N/16, tile_words]
    name_to_shape = {t["name"]: t["shape"] for t in tensors}
    for g in groups:
        tsh = name_to_shape[g["trellis"]]
        g["K"] = tsh[0] * 16
        g["N"] = tsh[1] * 16

    # Build the file-wide quant block. Qwen3.5 head uses 6 bpw; body 8.
    overrides = []
    if quant_cfg.get("head_bits") and quant_cfg["head_bits"] != quant_cfg.get("bits"):
        # tag every head tensor; lm_head.weight is a plain weight in this checkpoint
        for t in tensors:
            if "lm_head" in t["name"]:
                overrides.append({
                    "tensor_name": t["name"],
                    "scheme": "exl3_mcg",
                    "bits_per_weight": float(quant_cfg["head_bits"]),
                    "block_size": 256,
                })

    # The codebook_* trio: use the first stored trellis's planned offset
    # after layout. But at this point we haven't done layout yet. We'll
    # patch the keys AFTER layout by reading them back. For simplicity we
    # leave them pointing at an obvious location (offset 0, nbytes 1, sha
    # of empty-ish) and let the schema accept it — but the Stage 5 reader
    # expects codebook_sha256/offset/nbytes to be present for exl3_*. We
    # satisfy it by a synthetic 4-byte __codebook__ tensor, same pattern
    # as Stage 5 convert.py.
    codebook_sha = sha256_hex(struct.pack("<I", 0xCBAC1FED))
    # Pre-pend the codebook tensor so it's part of the layout
    cb_tensor = {
        "name": "__codebook__",
        "dtype": "u32",
        "shape": [1],
        "nbytes": 4,
        "sha256": None,  # computed in write_v02_streaming
        "source_offset": None,
        "marker_name": True,  # treat as placeholder so writer streams a stub...
    }
    # Actually we need its 4 real bytes. Build a manual source for it.
    # Simpler: adjust marker_name path to also handle 4-byte constants.
    cb_tensor["marker_name"] = False
    cb_tensor["source_offset"] = "CODEBOOK4"  # sentinel

    # Insert codebook first
    tensors.insert(0, cb_tensor)

    # Replace write path: the streamer reads from safetensors. For the
    # codebook we need to inject custom bytes. Extend the streamer below
    # with an inline branch.

    # Build fixed header
    header_fixed = {
        "tfold": {"version": "0.2", "writer": "tfold-convert-full@0.2.0"},
        "model": {
            "name": "Qwen3.5-2B-exl3-mcg-full",
            "arch": "qwen3",
            "parameters": sum(
                (1 if t["marker_name"] else _product(t["shape"]))
                for t in tensors
            ),
            "vocab_size": 151936,  # from config.json
            "context_length": 262144,
            "source_sha256": MODEL_SOURCE_SHA,
            "source_url": MODEL_URL,
        },
        "tokenizer": {"type": "tiktoken", "sha256": "0" * 64},
        "quant": {
            "scheme": "exl3_mcg",
            "bits_per_weight": float(quant_cfg["bits"]),
            "block_size": 256,
            "codebook_sha256": codebook_sha,
            "codebook_offset": 0,  # patched after layout
            "codebook_nbytes": 4,
            "overrides": overrides,
            "groups": groups,
        },
    }

    hdr = write_v02_streaming_with_codebook(args.out, st_path, data_offset, tensors, header_fixed)

    # Fix codebook_offset after layout (first tensor's offset)
    # Already correct since the codebook is tensors[0]; re-read file and patch.
    # For simplicity the final hdr in-memory is accurate.
    print("done.")
    print(f"tensors: {len(hdr['tensors'])}")
    print(f"groups : {len(hdr['quant']['groups'])}")
    print(f"overrides: {len(hdr['quant']['overrides'])}")
    return 0


def _product(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def write_v02_streaming_with_codebook(
    out_path: pathlib.Path,
    source_safetensors: pathlib.Path,
    source_data_offset: int,
    tensors: list[dict],
    header_fixed: dict,
) -> dict:
    """Variant of write_v02_streaming that handles the synthetic 4-byte
    codebook tensor at position 0."""
    chunk = 8 * 1024 * 1024
    layout_tmp = out_path.with_suffix(".tmp-tensors")
    print(f"pass 1: hash + stage {len(tensors)} tensors → {layout_tmp}")
    t_start = time.perf_counter()

    cur_offset = 0
    offsets_within_region: list[int] = []
    cb_bytes = struct.pack("<I", 0xCBAC1FED)
    with open(source_safetensors, "rb") as src, open(layout_tmp, "wb") as dst:
        for i, spec in enumerate(tensors):
            pad = _pad_to(cur_offset, ALIGN)
            if pad:
                dst.write(b"\x00" * pad)
                cur_offset += pad
            offsets_within_region.append(cur_offset)

            if spec.get("source_offset") == "CODEBOOK4":
                dst.write(cb_bytes)
                cur_offset += 4
                spec["sha256"] = sha256_hex(cb_bytes)
                continue

            if spec["marker_name"]:
                data = b"\x00"
                dst.write(data)
                cur_offset += 1
                spec["sha256"] = sha256_hex(data)
                continue

            src.seek(source_data_offset + spec["source_offset"])
            h = hashlib.sha256()
            remaining = spec["nbytes"]
            while remaining:
                to_read = min(chunk, remaining)
                b = src.read(to_read)
                if len(b) != to_read:
                    raise RuntimeError("short read from safetensors source")
                h.update(b)
                dst.write(b)
                remaining -= to_read
            spec["sha256"] = h.hexdigest()
            cur_offset += spec["nbytes"]

            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(tensors)} tensors staged "
                      f"({cur_offset/(1<<30):.2f} GiB)")
    tensor_region_bytes = cur_offset
    print(f"pass 1 done in {time.perf_counter()-t_start:.1f}s, "
          f"tensor region = {tensor_region_bytes/(1<<30):.3f} GiB")

    hdr = dict(header_fixed)
    hdr["tensors"] = []
    for spec in tensors:
        hdr["tensors"].append({
            "name": spec["name"],
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "offset": 0,
            "nbytes": 4 if spec.get("source_offset") == "CODEBOOK4" else (1 if spec["marker_name"] else spec["nbytes"]),
            "sha256": spec["sha256"],
        })

    for _ in range(6):
        hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        header_end = PREAMBLE_NBYTES + len(hj)
        region_start = header_end + _pad_to(header_end, ALIGN)
        changed = False
        for spec, roff in zip(hdr["tensors"], offsets_within_region):
            true_off = region_start + roff
            if spec["offset"] != true_off:
                spec["offset"] = true_off
                changed = True
        # patch codebook pointers
        cb_offset = hdr["tensors"][0]["offset"]
        if hdr["quant"].get("codebook_offset") != cb_offset:
            hdr["quant"]["codebook_offset"] = cb_offset
            changed = True
        if not changed:
            break

    validate_header_v02(hdr)

    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    header_end = PREAMBLE_NBYTES + len(hj)
    pad_after_header = b"\x00" * _pad_to(header_end, ALIGN)

    print(f"pass 2: emit .tfold v0.2 → {out_path}")
    t2 = time.perf_counter()
    with open(out_path, "wb") as dst, open(layout_tmp, "rb") as tmp:
        dst.write(MAGIC_V02)
        dst.write(struct.pack("<I", SPEC_VERSION_U32_V02))
        dst.write(struct.pack("<I", len(hj)))
        dst.write(hj)
        dst.write(pad_after_header)
        while True:
            b = tmp.read(chunk)
            if not b:
                break
            dst.write(b)
    layout_tmp.unlink()

    print(f"pass 3: FNV-1a-64 trailer...")
    h = 0xCBF2_9CE4_8422_2325
    PRIME = 0x100000001B3
    MASK = (1 << 64) - 1
    with open(out_path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            for byte in b:
                h ^= byte
                h = (h * PRIME) & MASK
    with open(out_path, "ab") as f:
        f.write(struct.pack("<Q", h))

    size_gib = out_path.stat().st_size / (1 << 30)
    print(f"wrote {out_path} ({size_gib:.3f} GiB) "
          f"in {time.perf_counter()-t2:.1f}s")
    return hdr


if __name__ == "__main__":
    raise SystemExit(main())
