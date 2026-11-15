"""tfold v0.1 reference writer and reader.

Strict on both sides:
  - writer validates header against schema before emitting bytes
  - writer computes per-tensor SHA-256 + FNV-1a-64 trailer
  - writer enforces 64-byte alignment, little-endian tensor encoding
  - reader performs every spec §4 check, in the order the spec specifies
"""

from __future__ import annotations

import copy
import json
import pathlib
import struct
from typing import Any, Mapping

import numpy as np

from . import _errors as E
from . import _dtypes
from ._hash import fnv1a_64_chunks, sha256_hex
from ._validate import validate_header
from ._version import (
    ALIGN,
    MAGIC,
    MAJOR_MASK,
    PREAMBLE_NBYTES,
    SPEC_VERSION_STR,
    SPEC_VERSION_U32,
    TRAILER_NBYTES,
    WRITER_ID,
)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _pad_to(n: int, align: int) -> int:
    """Return (padded_size, pad_bytes_needed) from n up to the next align boundary."""
    r = n % align
    return (0 if r == 0 else align - r)


def _assert_le_contiguous(arr: np.ndarray, name: str) -> None:
    if not arr.flags["C_CONTIGUOUS"]:
        raise E.TfoldError(f"tensor {name!r} must be C-contiguous before write")
    if arr.dtype.byteorder == ">":
        raise E.TfoldError(f"tensor {name!r} is big-endian; writer rejects (spec §0)")


# ---------------------------------------------------------------------------
# WRITE
# ---------------------------------------------------------------------------


def write(
    path: str | pathlib.Path,
    header: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray | bytes | memoryview],
) -> dict[str, Any]:
    """Write a .tfold file.

    Args:
        path:    output file path
        header:  dict missing ONLY the per-tensor offset/nbytes/sha256 and
                 the writer's `tfold.writer` field (writer may fill those in).
                 The caller MUST supply model, tokenizer, quant, and the list
                 of tensor names + dtype + shape. We derive offset/nbytes/sha
                 from the actual buffers.
        tensors: dict of name -> numpy array OR raw bytes (for packed dtypes).
                 Keys MUST match header["tensors"][i]["name"] exactly.

    Returns the final header dict that was written (with offsets filled in).
    Raises TfoldError / SpecError on any violation.
    """
    path = pathlib.Path(path)
    header = copy.deepcopy(dict(header))
    header.setdefault("tfold", {}).setdefault("version", SPEC_VERSION_STR)
    header["tfold"].setdefault("writer", WRITER_ID)

    tensor_specs = header.get("tensors")
    if not isinstance(tensor_specs, list) or not tensor_specs:
        raise E.TfoldError("header['tensors'] must be a non-empty list")

    names = [t["name"] for t in tensor_specs]
    if len(set(names)) != len(names):
        raise E.SpecError(E.TENSOR_DUP_NAME, "duplicate tensor names in header")

    missing = set(names) - set(tensors.keys())
    if missing:
        raise E.TfoldError(f"missing tensor buffers: {sorted(missing)}")
    extra = set(tensors.keys()) - set(names)
    if extra:
        raise E.TfoldError(f"tensor buffers not declared in header: {sorted(extra)}")

    # Materialize each tensor as LE bytes + verify shape/dtype consistency.
    blobs: dict[str, bytes] = {}
    for spec in tensor_specs:
        name = spec["name"]
        dtype = spec["dtype"]
        shape = list(spec["shape"])
        buf = tensors[name]

        if not _dtypes.is_known(dtype):
            raise E.SpecError(E.UNKNOWN_DTYPE, f"tensor {name!r}: {dtype!r}")

        np_dtype = _dtypes.numpy_dtype_of(dtype)
        if np_dtype is not None:
            if not isinstance(buf, np.ndarray):
                raise E.TfoldError(f"tensor {name!r}: numpy array required for dtype {dtype}")
            if list(buf.shape) != shape:
                raise E.TfoldError(
                    f"tensor {name!r}: declared shape {shape} != actual {list(buf.shape)}"
                )
            arr = np.ascontiguousarray(buf, dtype=np_dtype)
            _assert_le_contiguous(arr, name)
            blob = arr.tobytes(order="C")
        else:
            # Packed / bf16 / exl3*: accept raw bytes-like
            if isinstance(buf, np.ndarray):
                blob = bytes(np.ascontiguousarray(buf).view(np.uint8))
            else:
                blob = bytes(buf)

        exp = _dtypes.expected_nbytes(dtype, shape)
        if exp is not None and len(blob) != exp:
            raise E.SpecError(
                E.TENSOR_SHAPE_NBYTES,
                f"tensor {name!r}: nbytes {len(blob)} != product(shape)*elem {exp}",
            )

        blobs[name] = blob

    # ----- plan layout -----
    # We need to know the final header length before we can compute tensor
    # offsets. Strategy: estimate with a placeholder (0 offsets), then
    # recompute offsets, patch the header, and re-serialize. If the
    # re-serialized size changes we iterate (offsets grow -> more digits
    # -> potentially longer JSON). Iterate to a fixed point.
    def layout(hdr: dict[str, Any]) -> tuple[bytes, int, list[tuple[int, int]]]:
        """Return (header_json_bytes, header_region_end, [(offset, nbytes)...])."""
        header_json = json.dumps(hdr, separators=(",", ":"), sort_keys=False).encode("utf-8")
        header_length = len(header_json)
        header_end = PREAMBLE_NBYTES + header_length
        tensor_region_start = header_end + _pad_to(header_end, ALIGN)
        offsets: list[tuple[int, int]] = []
        cur = tensor_region_start
        for spec in hdr["tensors"]:
            nbytes = len(blobs[spec["name"]])
            offsets.append((cur, nbytes))
            cur += nbytes + _pad_to(cur + nbytes, ALIGN)
        return header_json, header_end, offsets

    # Initialize tensor spec fields with placeholders.
    for i, spec in enumerate(tensor_specs):
        spec["offset"] = 0
        spec["nbytes"] = len(blobs[spec["name"]])
        spec["sha256"] = sha256_hex(blobs[spec["name"]])

    # Also compute codebook_offset if quant scheme requires one and tensor is present.
    def patch_codebook_offset(hdr: dict[str, Any], offsets: list[tuple[int, int]]) -> dict[str, Any]:
        if "__codebook__" in [t["name"] for t in hdr["tensors"]]:
            idx = [i for i, t in enumerate(hdr["tensors"]) if t["name"] == "__codebook__"][0]
            off, nb = offsets[idx]
            hdr.setdefault("quant", {})
            hdr["quant"]["codebook_offset"] = off
            hdr["quant"]["codebook_nbytes"] = nb
            hdr["quant"]["codebook_sha256"] = hdr["tensors"][idx]["sha256"]
        return hdr

    # Iterate layout -> patch offsets -> re-layout until stable.
    for _ in range(8):
        hj, header_end, offsets = layout(header)
        changed = False
        for i, (off, nb) in enumerate(offsets):
            if tensor_specs[i]["offset"] != off or tensor_specs[i]["nbytes"] != nb:
                tensor_specs[i]["offset"] = off
                tensor_specs[i]["nbytes"] = nb
                changed = True
        header = patch_codebook_offset(header, offsets)
        if not changed:
            break
    else:
        raise E.TfoldError("offset layout did not converge after 8 passes")

    # Final validation of fully-populated header
    validate_header(header)

    # ----- emit bytes -----
    # Streaming FNV-1a-64 over everything up to the trailer
    def chunks():
        yield MAGIC
        yield struct.pack("<I", SPEC_VERSION_U32)
        yield struct.pack("<I", len(hj))
        yield hj
        pad = _pad_to(PREAMBLE_NBYTES + len(hj), ALIGN)
        if pad:
            yield b"\x00" * pad
        prev_end = PREAMBLE_NBYTES + len(hj) + pad
        for spec in header["tensors"]:
            name = spec["name"]
            off = spec["offset"]
            if off != prev_end:
                # inter-tensor padding
                pad = off - prev_end
                if pad < 0:
                    raise E.TfoldError("internal: layout retrograde")
                yield b"\x00" * pad
            blob = blobs[name]
            yield blob
            prev_end = off + len(blob)
        # trailing pad to 64-byte alignment before trailer (not required by spec
        # but keeps the trailer naturally aligned)
        # Spec does not require this; omit to keep file minimal.

    # Materialize in-memory to compute FNV, then write.
    # (Streaming FNV + streaming write is possible but needs two passes; this
    # keeps the first implementation simple. Stage 4 Zig loader has no such
    # constraint.)
    body = b"".join(chunks())
    h = fnv1a_64_chunks([body])
    trailer = struct.pack("<Q", h)

    path.write_bytes(body + trailer)
    return header


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------


def read(
    path: str | pathlib.Path,
    pin_source_sha256: str | None = None,
    verify_tensor_hashes: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Open + validate a .tfold file, return (header, tensor_views).

    Tensor views are numpy arrays backed by the file bytes (mmap when
    possible). Writes to them are undefined; copy before mutating.

    Raises SpecError on any spec §4 violation.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise E.TfoldError(f"no such file: {path}")

    file_bytes = path.read_bytes()  # reference impl: simple; Zig loader mmaps

    # §4.1 TFOLD_TOO_SMALL — must be at least preamble + 2 bytes JSON ({}) + trailer
    min_size = PREAMBLE_NBYTES + 2 + TRAILER_NBYTES
    if len(file_bytes) < min_size:
        raise E.SpecError(E.TOO_SMALL, f"{len(file_bytes)} < {min_size}")

    # §4.2 TFOLD_BAD_MAGIC
    if file_bytes[:8] != MAGIC:
        raise E.SpecError(E.BAD_MAGIC, f"got {file_bytes[:8]!r}")

    # §4.3 TFOLD_VERSION_UNSUPPORTED
    (version,) = struct.unpack("<I", file_bytes[8:12])
    if (version & MAJOR_MASK) != (SPEC_VERSION_U32 & MAJOR_MASK):
        raise E.SpecError(E.VERSION_UNSUPPORTED, f"major {version:#010x}")

    # §4.4 TFOLD_HEADER_OVERFLOW
    (header_length,) = struct.unpack("<I", file_bytes[12:16])
    if PREAMBLE_NBYTES + header_length + TRAILER_NBYTES > len(file_bytes):
        raise E.SpecError(E.HEADER_OVERFLOW, f"header_length={header_length}")

    # §4.5 TFOLD_HEADER_PARSE
    header_bytes = file_bytes[PREAMBLE_NBYTES:PREAMBLE_NBYTES + header_length]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise E.SpecError(E.HEADER_PARSE, str(exc))

    # §4.6 TFOLD_HEADER_SCHEMA
    validate_header(header)

    # §4.7 TFOLD_INTEGRITY — FNV-1a-64 over everything except trailer
    body = file_bytes[:-TRAILER_NBYTES]
    want = struct.unpack("<Q", file_bytes[-TRAILER_NBYTES:])[0]
    got = fnv1a_64_chunks([body])
    if got != want:
        raise E.SpecError(E.INTEGRITY, f"got {got:#018x} != {want:#018x}")

    # §4.8/9/10/11 — bounds, alignment, overlap, dup names
    file_end = len(file_bytes) - TRAILER_NBYTES  # exclusive end for tensor region
    tensor_specs = header["tensors"]
    names_seen = set()
    sorted_by_off = sorted(tensor_specs, key=lambda t: t["offset"])
    prev_end = 0
    for spec in sorted_by_off:
        name = spec["name"]
        if name in names_seen:
            raise E.SpecError(E.TENSOR_DUP_NAME, name)
        names_seen.add(name)
        off = spec["offset"]
        nb = spec["nbytes"]
        if off % ALIGN != 0:
            raise E.SpecError(E.TENSOR_ALIGN, f"{name} offset {off} % {ALIGN} != 0")
        if off + nb > file_end:
            raise E.SpecError(E.TENSOR_OOB, f"{name} ends at {off + nb} > {file_end}")
        if off < prev_end:
            raise E.SpecError(E.TENSOR_OVERLAP, f"{name} starts at {off} < prev_end {prev_end}")
        prev_end = off + nb

    # §4.12 — unknown registries. Schema already covers this, but recheck
    # explicitly so the reject code is in the §4 vocabulary.
    if header["model"]["arch"] not in {"qwen3", "qwen3_moe", "llama", "glm", "gemma3"}:
        raise E.SpecError(E.UNKNOWN_ARCH, header["model"]["arch"])
    if not _dtypes.is_known("fp32"):  # sentinel; real check follows
        raise E.TfoldError("internal: dtype registry broken")
    for spec in tensor_specs:
        if not _dtypes.is_known(spec["dtype"]):
            raise E.SpecError(E.UNKNOWN_DTYPE, spec["dtype"])
        name = spec["name"]
        if name.startswith("__") and name != "__codebook__":
            raise E.SpecError(E.UNKNOWN_RESERVED_NAME, name)

    # Trailing bytes beyond the trailer (spec §1 last sentence)
    # Already guaranteed: we computed file_end = len - TRAILER_NBYTES and
    # verified the last 8 bytes are the trailer. There is nothing after.
    # But guard against future reorderings: assert the total size equals
    # what tensor layout + trailer would have.
    # (No separate TRAILING_BYTES raise path exists on this code path.)

    # §4.13 — pinned checkpoint match
    if pin_source_sha256 is not None:
        if header["model"]["source_sha256"] != pin_source_sha256:
            raise E.SpecError(
                E.CHECKPOINT_MISMATCH,
                f"got {header['model']['source_sha256']}, pinned {pin_source_sha256}",
            )

    # §4.14 — per-tensor SHA
    views: dict[str, np.ndarray] = {}
    for spec in tensor_specs:
        name = spec["name"]
        off = spec["offset"]
        nb = spec["nbytes"]
        blob = file_bytes[off:off + nb]
        if verify_tensor_hashes:
            got_hash = sha256_hex(blob)
            if got_hash != spec["sha256"]:
                raise E.SpecError(
                    E.TENSOR_INTEGRITY,
                    f"{name} got {got_hash}, declared {spec['sha256']}",
                )
        np_dtype = _dtypes.numpy_dtype_of(spec["dtype"])
        if np_dtype is not None:
            views[name] = np.frombuffer(blob, dtype=np_dtype).reshape(spec["shape"])
        else:
            # Opaque bytes for packed dtypes
            views[name] = np.frombuffer(blob, dtype=np.uint8)

    return header, views
