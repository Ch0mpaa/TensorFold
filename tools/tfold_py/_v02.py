"""tfold v0.2 writer + reader.

v0.2 is a major version bump. Backward compatibility is **through
coexistence, not interpretation**: v0.1 readers reject v0.2 files
with `TFOLD_VERSION_UNSUPPORTED`; v0.2 readers reject v0.1 files with
the same code (unless opened via tfold_py.read, which still handles v0.1).

v0.2 changes vs v0.1:
    - magic bytes: b"TFOLD\\x02\\x00\\x00"
    - version u32: 0x00020000
    - OPTIONAL quant.overrides: per-tensor scheme/bits overrides
    - OPTIONAL quant.groups: explicit EXL3 group structure
    - OPTIONAL model.tensor_count: redundant length check
    - max tensors raised 65535 → 1_048_575 (3-decimal digit room)
    - two new EXL3 scheme/dtype variants (`exl3_mul1`, `exl3_3inst` +
      matching tile dtypes) so a v0.2 file can name the codebook even
      when the global scheme differs
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import struct
from typing import Any, Mapping

import numpy as np

from . import _errors as E
from . import _dtypes
from ._hash import fnv1a_64_chunks, sha256_hex
from ._version import ALIGN, PREAMBLE_NBYTES, TRAILER_NBYTES

MAGIC_V02: bytes = b"TFOLD\x02\x00\x00"
SPEC_VERSION_U32_V02: int = 0x0002_0000
MAJOR_MASK: int = 0xFFFF_0000

# Dtype registry v0.2 superset: adds two codebook-specific tile dtypes.
_V02_EXTRA_DTYPES = {
    "exl3_mul1_tile":  (0, None, True),
    "exl3_3inst_tile": (0, None, True),
}


def _v02_is_known_dtype(name: str) -> bool:
    return name in _dtypes.DTYPES or name in _V02_EXTRA_DTYPES


def _v02_element_size(name: str) -> int | None:
    if name in _dtypes.DTYPES:
        return _dtypes.expected_nbytes.__wrapped__ if False else None  # handled by _dtypes
    return None  # packed


# ---------------------------------------------------------------------------
# schema validator for v0.2 header
# ---------------------------------------------------------------------------

_SCHEMA_V02_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "schemas" / "tfold-v0.2.json"
_SCHEMA_V02: dict | None = None


def _load_v02_schema():
    global _SCHEMA_V02
    if _SCHEMA_V02 is None:
        import json as _json
        _SCHEMA_V02 = _json.loads(_SCHEMA_V02_PATH.read_text(encoding="utf-8"))
    return _SCHEMA_V02


def validate_header_v02(header: dict) -> None:
    import jsonschema
    schema = _load_v02_schema()
    try:
        jsonschema.validate(header, schema)
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise E.SpecError(E.HEADER_SCHEMA, f"v0.2 at {path}: {e.message}")


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


def _pad_to(n: int, align: int) -> int:
    r = n % align
    return 0 if r == 0 else align - r


def write_v02(
    path: str | pathlib.Path,
    header: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray | bytes | memoryview],
) -> dict[str, Any]:
    """Write a v0.2 file. Header + tensor buffers; same layout rules as
    v0.1 except magic / version / optional overrides / optional groups.
    """
    path = pathlib.Path(path)
    header = copy.deepcopy(dict(header))
    header.setdefault("tfold", {})
    header["tfold"]["version"] = "0.2"
    header["tfold"].setdefault("writer", "tfold-py@0.2.0")

    tensor_specs = header.get("tensors")
    if not isinstance(tensor_specs, list) or not tensor_specs:
        raise E.TfoldError("header['tensors'] must be a non-empty list")

    names = [t["name"] for t in tensor_specs]
    if len(set(names)) != len(names):
        raise E.SpecError(E.TENSOR_DUP_NAME, "duplicate tensor names in header")

    # Materialize buffers
    blobs: dict[str, bytes] = {}
    for spec in tensor_specs:
        name = spec["name"]
        dtype = spec["dtype"]
        shape = list(spec["shape"])
        buf = tensors[name]

        if not _v02_is_known_dtype(dtype):
            raise E.SpecError(E.UNKNOWN_DTYPE, f"tensor {name!r}: {dtype!r}")

        # Known plain dtypes from v0.1 registry
        if dtype in _dtypes.DTYPES:
            np_dtype = _dtypes.numpy_dtype_of(dtype)
            if np_dtype is not None:
                if not isinstance(buf, np.ndarray):
                    raise E.TfoldError(f"tensor {name!r}: numpy array required for dtype {dtype}")
                if list(buf.shape) != shape:
                    raise E.TfoldError(
                        f"tensor {name!r}: declared shape {shape} != actual {list(buf.shape)}"
                    )
                arr = np.ascontiguousarray(buf, dtype=np_dtype)
                blob = arr.tobytes(order="C")
            else:
                blob = bytes(buf) if not isinstance(buf, np.ndarray) else bytes(np.ascontiguousarray(buf).view(np.uint8))
        else:
            # v0.2 EXL3 tile dtypes: always raw bytes
            blob = bytes(buf) if not isinstance(buf, np.ndarray) else bytes(np.ascontiguousarray(buf).view(np.uint8))

        blobs[name] = blob

    # Iterate layout
    def layout(hdr: dict) -> tuple[bytes, list[tuple[int, int]]]:
        hj = json.dumps(hdr, separators=(",", ":"), sort_keys=False).encode("utf-8")
        header_end = PREAMBLE_NBYTES + len(hj)
        tensor_region_start = header_end + _pad_to(header_end, ALIGN)
        offsets: list[tuple[int, int]] = []
        cur = tensor_region_start
        for spec in hdr["tensors"]:
            nb = len(blobs[spec["name"]])
            offsets.append((cur, nb))
            cur += nb + _pad_to(cur + nb, ALIGN)
        return hj, offsets

    for i, spec in enumerate(tensor_specs):
        spec["offset"] = 0
        spec["nbytes"] = len(blobs[spec["name"]])
        spec["sha256"] = sha256_hex(blobs[spec["name"]])

    def patch_codebook_offset(hdr: dict, offsets):
        names = [t["name"] for t in hdr["tensors"]]
        if "__codebook__" in names:
            idx = names.index("__codebook__")
            off, nb = offsets[idx]
            hdr.setdefault("quant", {})
            hdr["quant"]["codebook_offset"] = off
            hdr["quant"]["codebook_nbytes"] = nb
            hdr["quant"]["codebook_sha256"] = hdr["tensors"][idx]["sha256"]
        return hdr

    for _ in range(8):
        hj, offsets = layout(header)
        changed = False
        for i, (off, nb) in enumerate(offsets):
            if tensor_specs[i]["offset"] != off or tensor_specs[i]["nbytes"] != nb:
                tensor_specs[i]["offset"] = off
                tensor_specs[i]["nbytes"] = nb
                changed = True
        header = patch_codebook_offset(header, offsets)
        if not changed:
            break

    # tensor_count sanity if caller asked us to emit it
    if "tensor_count" in header.get("model", {}):
        if header["model"]["tensor_count"] != len(tensor_specs):
            header["model"]["tensor_count"] = len(tensor_specs)

    validate_header_v02(header)

    # Emit bytes
    hj = json.dumps(header, separators=(",", ":"), sort_keys=False).encode("utf-8")
    parts = [
        MAGIC_V02,
        struct.pack("<I", SPEC_VERSION_U32_V02),
        struct.pack("<I", len(hj)),
        hj,
        b"\x00" * _pad_to(PREAMBLE_NBYTES + len(hj), ALIGN),
    ]
    prev_end = PREAMBLE_NBYTES + len(hj) + _pad_to(PREAMBLE_NBYTES + len(hj), ALIGN)
    for spec in header["tensors"]:
        off = spec["offset"]
        if off != prev_end:
            parts.append(b"\x00" * (off - prev_end))
        parts.append(blobs[spec["name"]])
        prev_end = off + spec["nbytes"]
    body = b"".join(parts)
    h = fnv1a_64_chunks([body])
    trailer = struct.pack("<Q", h)
    path.write_bytes(body + trailer)
    return header


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


def read_v02(
    path: str | pathlib.Path,
    pin_source_sha256: str | None = None,
    verify_tensor_hashes: bool = True,
    verify_fnv_trailer: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = pathlib.Path(path)
    file_bytes = path.read_bytes()

    min_size = PREAMBLE_NBYTES + 2 + TRAILER_NBYTES
    if len(file_bytes) < min_size:
        raise E.SpecError(E.TOO_SMALL, f"{len(file_bytes)} < {min_size}")
    if file_bytes[:8] != MAGIC_V02:
        raise E.SpecError(E.BAD_MAGIC, f"got {file_bytes[:8]!r}, want v0.2 magic")
    (version,) = struct.unpack("<I", file_bytes[8:12])
    if (version & MAJOR_MASK) != (SPEC_VERSION_U32_V02 & MAJOR_MASK):
        raise E.SpecError(E.VERSION_UNSUPPORTED, f"major {version:#010x}")
    (hlen,) = struct.unpack("<I", file_bytes[12:16])
    if PREAMBLE_NBYTES + hlen + TRAILER_NBYTES > len(file_bytes):
        raise E.SpecError(E.HEADER_OVERFLOW, f"header_length={hlen}")

    try:
        header = json.loads(file_bytes[PREAMBLE_NBYTES:PREAMBLE_NBYTES + hlen].decode("utf-8"))
    except Exception as exc:
        raise E.SpecError(E.HEADER_PARSE, str(exc))

    validate_header_v02(header)

    if verify_fnv_trailer:
        body = file_bytes[:-TRAILER_NBYTES]
        want = struct.unpack("<Q", file_bytes[-TRAILER_NBYTES:])[0]
        got = fnv1a_64_chunks([body])
        if got != want:
            raise E.SpecError(E.INTEGRITY, f"got {got:#018x} != {want:#018x}")

    file_end = len(file_bytes) - TRAILER_NBYTES
    tensor_specs = header["tensors"]
    names_seen = set()
    sorted_by_off = sorted(tensor_specs, key=lambda t: t["offset"])
    prev_end = 0
    for spec in sorted_by_off:
        n = spec["name"]
        if n in names_seen:
            raise E.SpecError(E.TENSOR_DUP_NAME, n)
        names_seen.add(n)
        off = spec["offset"]; nb = spec["nbytes"]
        if off % ALIGN != 0:
            raise E.SpecError(E.TENSOR_ALIGN, f"{n} offset {off}")
        if off + nb > file_end:
            raise E.SpecError(E.TENSOR_OOB, f"{n} end={off+nb} > {file_end}")
        if off < prev_end:
            raise E.SpecError(E.TENSOR_OVERLAP, f"{n} at {off} < prev_end {prev_end}")
        prev_end = off + nb

    if pin_source_sha256 is not None:
        if header["model"]["source_sha256"] != pin_source_sha256:
            raise E.SpecError(E.CHECKPOINT_MISMATCH, header["model"]["source_sha256"])

    views: dict[str, np.ndarray] = {}
    for spec in tensor_specs:
        name = spec["name"]
        off = spec["offset"]; nb = spec["nbytes"]
        blob = file_bytes[off:off + nb]
        if verify_tensor_hashes:
            if sha256_hex(blob) != spec["sha256"]:
                raise E.SpecError(E.TENSOR_INTEGRITY, name)
        dtype = spec["dtype"]
        if dtype in _dtypes.DTYPES:
            np_dtype = _dtypes.numpy_dtype_of(dtype)
            if np_dtype is not None:
                views[name] = np.frombuffer(blob, dtype=np_dtype).reshape(spec["shape"])
            else:
                views[name] = np.frombuffer(blob, dtype=np.uint8)
        else:
            views[name] = np.frombuffer(blob, dtype=np.uint8)
    return header, views


# ---------------------------------------------------------------------------
# convenience: resolve per-tensor scheme/bits with overrides applied
# ---------------------------------------------------------------------------


def resolve_tensor_scheme(header: dict, tensor_name: str) -> dict:
    """Return the effective {scheme, bits_per_weight, block_size} for the
    named tensor, applying quant.overrides if present."""
    quant = header["quant"]
    out = {
        "scheme": quant["scheme"],
        "bits_per_weight": quant["bits_per_weight"],
        "block_size": quant["block_size"],
    }
    for ov in quant.get("overrides", []):
        if ov["tensor_name"] == tensor_name:
            out.update({k: v for k, v in ov.items() if k != "tensor_name"})
    return out
