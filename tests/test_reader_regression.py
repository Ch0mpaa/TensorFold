"""Regression tests for the Stage 2 schema-negative cases that were
either skipped or could only be verified indirectly.

Approach: build a syntactically-complete .tfold file from raw bytes
with bad header content, recompute the FNV-1a-64 trailer so the
integrity check passes, then verify the reader rejects at the
specific §4 step we want to test.

This exercises the reader path end-to-end, not just the schema. No
spec change was needed.
"""

from __future__ import annotations

import json
import pathlib
import struct

import pytest

import tfold_py
from tfold_py import _errors as E
from tfold_py._hash import fnv1a_64_chunks
from tfold_py._version import ALIGN, MAGIC, PREAMBLE_NBYTES, SPEC_VERSION_U32, TRAILER_NBYTES


PIN = "a" * 64


def _pad_to(n: int, align: int) -> int:
    r = n % align
    return 0 if r == 0 else align - r


def _craft(header: dict, tensor_blobs: dict[str, bytes], tmp_path: pathlib.Path) -> pathlib.Path:
    """Manually craft a .tfold file with a specified header, computing
    all offsets + per-tensor SHAs + FNV trailer so the file is
    structurally valid even when the header content is semantically wrong.
    """
    from tfold_py._hash import sha256_hex

    # First pass: assume placeholder offsets so we can measure header JSON length,
    # then compute real offsets, then re-serialize.
    for spec in header["tensors"]:
        spec["nbytes"] = len(tensor_blobs[spec["name"]])
        spec["sha256"] = sha256_hex(tensor_blobs[spec["name"]])
        spec["offset"] = 0

    # Iterate to a fixed point like the writer does
    for _ in range(8):
        hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
        hdr_end = PREAMBLE_NBYTES + len(hj)
        tensor_start = hdr_end + _pad_to(hdr_end, ALIGN)
        changed = False
        cur = tensor_start
        for spec in header["tensors"]:
            nb = spec["nbytes"]
            if spec["offset"] != cur:
                spec["offset"] = cur
                changed = True
            cur += nb + _pad_to(cur + nb, ALIGN)
        if not changed:
            break

    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = bytearray()
    body += MAGIC
    body += struct.pack("<I", SPEC_VERSION_U32)
    body += struct.pack("<I", len(hj))
    body += hj
    body += b"\x00" * _pad_to(PREAMBLE_NBYTES + len(hj), ALIGN)
    prev = PREAMBLE_NBYTES + len(hj) + _pad_to(PREAMBLE_NBYTES + len(hj), ALIGN)
    for spec in header["tensors"]:
        off = spec["offset"]
        if off != prev:
            body += b"\x00" * (off - prev)
        blob = tensor_blobs[spec["name"]]
        body += blob
        prev = off + len(blob)
    trailer = struct.pack("<Q", fnv1a_64_chunks([bytes(body)]))
    out = tmp_path / "crafted.tfold"
    out.write_bytes(bytes(body) + trailer)
    return out


def _baseline_header(arch: str = "qwen3") -> dict:
    return {
        "tfold": {"version": "0.1", "writer": "regression"},
        "model": {
            "name": "t", "arch": arch, "parameters": 1,
            "vocab_size": 1, "context_length": 1,
            "source_sha256": PIN, "source_url": "x",
        },
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "w", "dtype": "fp32", "shape": [1],
             "offset": 0, "nbytes": 4, "sha256": "0" * 64},
        ],
    }


# ---------- previously SKIPPED case: unknown arch at read ----------


def test_rejects_unknown_arch_at_read(tmp_path):
    """Previously skipped because in-place mutation grew the JSON length.
    Now constructed from scratch instead."""
    hdr = _baseline_header(arch="unknown_arch")
    blobs = {"w": struct.pack("<f", 1.0)}
    path = _craft(hdr, blobs, tmp_path)
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(path)
    # Schema catches first; UNKNOWN_ARCH in the explicit reader path is also
    # acceptable depending on check ordering.
    assert ei.value.code in (E.HEADER_SCHEMA, E.UNKNOWN_ARCH), (
        f"expected HEADER_SCHEMA or UNKNOWN_ARCH, got {ei.value.code}"
    )


# ---------- bonus: unknown scheme at read ----------


def test_rejects_unknown_scheme_at_read(tmp_path):
    hdr = _baseline_header()
    hdr["quant"]["scheme"] = "unknown_q4"
    hdr["quant"].pop("block_size", None)
    hdr["quant"]["block_size"] = 0
    blobs = {"w": struct.pack("<f", 1.0)}
    path = _craft(hdr, blobs, tmp_path)
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(path)
    assert ei.value.code in (E.HEADER_SCHEMA, E.UNKNOWN_SCHEME)


# ---------- bonus: unknown tokenizer type ----------


def test_rejects_unknown_tokenizer_type_at_read(tmp_path):
    hdr = _baseline_header()
    hdr["tokenizer"]["type"] = "weird_bpe"
    blobs = {"w": struct.pack("<f", 1.0)}
    path = _craft(hdr, blobs, tmp_path)
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(path)
    assert ei.value.code in (E.HEADER_SCHEMA, E.UNKNOWN_TOKENIZER_TYPE)


# ---------- bonus: unknown reserved __name__ at read ----------


def test_rejects_unknown_reserved_name_at_read(tmp_path):
    hdr = _baseline_header()
    hdr["tensors"][0]["name"] = "__secret__"
    blobs = {"__secret__": struct.pack("<f", 1.0)}
    path = _craft(hdr, blobs, tmp_path)
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(path)
    assert ei.value.code in (E.HEADER_SCHEMA, E.UNKNOWN_RESERVED_NAME)


# ---------- sanity: good baseline still opens ----------


def test_crafted_baseline_opens_cleanly(tmp_path):
    """Make sure the test harness itself produces valid files."""
    hdr = _baseline_header()
    blobs = {"w": struct.pack("<f", 1.5)}
    path = _craft(hdr, blobs, tmp_path)
    header, tensors = tfold_py.read(path)
    assert header["model"]["arch"] == "qwen3"
    assert list(tensors["w"]) == pytest.approx([1.5])
