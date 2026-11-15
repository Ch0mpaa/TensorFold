"""Corruption + fail-closed tests. Each case targets one reject path
from docs/tfold-format-v0.md §4.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import struct

import numpy as np
import pytest

import tfold_py
from tfold_py import _errors as E
from tfold_py._version import MAGIC, PREAMBLE_NBYTES, TRAILER_NBYTES


PIN = hashlib.sha256(b"tfold-v0.1-demo-fp32-fixture").hexdigest()


def _good_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    rng = np.random.default_rng(seed=0x7F01D)
    w = rng.standard_normal((8, 4), dtype=np.float32)
    b = rng.standard_normal((4,), dtype=np.float32)
    x = rng.standard_normal((8,), dtype=np.float32)
    header = {
        "tfold": {"version": "0.1", "writer": "tfold-py@0.1.0"},
        "model": {
            "name": "tfold-demo-fp32", "arch": "qwen3",
            "parameters": 36, "vocab_size": 1, "context_length": 1,
            "source_sha256": PIN,
            "source_url": "builtin://tfold-v0.1-demo-fp32-fixture",
        },
        "tokenizer": {"type": "tiktoken",
                      "sha256": hashlib.sha256(b"tfold-v0.1-demo-no-tokenizer").hexdigest()},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "weight.linear_0", "dtype": "fp32", "shape": [8, 4], "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "bias.linear_0",   "dtype": "fp32", "shape": [4],    "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "input.x",         "dtype": "fp32", "shape": [8],    "offset": 0, "nbytes": 0, "sha256": "0" * 64},
        ],
    }
    out = tmp_path / "f.tfold"
    tfold_py.write(out, header, {"weight.linear_0": w, "bias.linear_0": b, "input.x": x})
    return out


def _mutate(src: pathlib.Path, dst: pathlib.Path, offset: int, replacement: bytes) -> None:
    data = bytearray(src.read_bytes())
    data[offset:offset + len(replacement)] = replacement
    dst.write_bytes(bytes(data))


# ---------- §4.1 TOO_SMALL ----------
def test_rejects_empty_file(tmp_path):
    p = tmp_path / "e.tfold"
    p.write_bytes(b"")
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(p)
    assert ei.value.code == E.TOO_SMALL


def test_rejects_tiny_file(tmp_path):
    p = tmp_path / "t.tfold"
    p.write_bytes(MAGIC + b"\x00" * 4)  # magic + 4 bytes, still too small
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(p)
    assert ei.value.code == E.TOO_SMALL


# ---------- §4.2 BAD_MAGIC ----------
def test_rejects_wrong_magic(tmp_path):
    good = _good_fixture(tmp_path)
    bad = tmp_path / "bm.tfold"
    _mutate(good, bad, 0, b"NOPE\x00\x00\x00\x00")
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(bad)
    assert ei.value.code == E.BAD_MAGIC


# ---------- §4.3 VERSION_UNSUPPORTED ----------
def test_rejects_wrong_major_version(tmp_path):
    good = _good_fixture(tmp_path)
    bad = tmp_path / "ver.tfold"
    # Writer wrote 0x00010000 (major 1); replace with 0x00020000 (major 2)
    _mutate(good, bad, 8, struct.pack("<I", 0x00020000))
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(bad)
    assert ei.value.code == E.VERSION_UNSUPPORTED


# ---------- §4.4 HEADER_OVERFLOW ----------
def test_rejects_declared_header_length_overflows_file(tmp_path):
    good = _good_fixture(tmp_path)
    bad = tmp_path / "ho.tfold"
    # Set header_length to a huge value
    _mutate(good, bad, 12, struct.pack("<I", 0xFFFFFFFF))
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(bad)
    assert ei.value.code == E.HEADER_OVERFLOW


# ---------- §4.5 HEADER_PARSE ----------
def test_rejects_malformed_json(tmp_path):
    good = _good_fixture(tmp_path)
    bad = tmp_path / "jp.tfold"
    raw = bytearray(good.read_bytes())
    # Replace the first byte of the header with '}' to break JSON
    raw[PREAMBLE_NBYTES] = ord("}")
    bad.write_bytes(bytes(raw))
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(bad)
    # Header parse fails first; integrity would also trip downstream
    assert ei.value.code in (E.HEADER_PARSE, E.INTEGRITY)


# ---------- §4.6 HEADER_SCHEMA ----------
# The unknown-arch-at-read case lives in tests/test_reader_regression.py
# (built fresh from raw bytes so JSON length is free to change). Keeping
# only a pointer here to avoid confusion.


# ---------- §4.7 INTEGRITY ----------
def test_rejects_corrupted_trailer(tmp_path):
    good = _good_fixture(tmp_path)
    bad = tmp_path / "it.tfold"
    raw = bytearray(good.read_bytes())
    # Flip one bit in the trailer
    raw[-1] ^= 0x80
    bad.write_bytes(bytes(raw))
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(bad)
    assert ei.value.code == E.INTEGRITY


def test_rejects_corrupted_tensor_bytes_via_trailer(tmp_path):
    """Flipping a byte in the tensor region changes the FNV trailer check."""
    good = _good_fixture(tmp_path)
    bad = tmp_path / "ctb.tfold"
    raw = bytearray(good.read_bytes())
    # Flip a byte deep inside the tensor region
    raw[-16] ^= 0xFF
    bad.write_bytes(bytes(raw))
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(bad)
    assert ei.value.code == E.INTEGRITY


# ---------- §4.8 TENSOR_OOB (requires valid trailer after mutation) ----------
# Hardest to craft without recomputing the trailer; skip in favor of a
# full programmatic test in test_reader_bounds below.
def test_rejects_tensor_offset_past_eof():
    """Programmatic: synthesize a header whose tensor offset exceeds file size.
    This goes through the read path post-schema, post-integrity, into §4.8."""
    import tfold_py._io as _io
    from tfold_py._version import MAGIC, PREAMBLE_NBYTES, SPEC_VERSION_U32, TRAILER_NBYTES
    from tfold_py._hash import fnv1a_64_chunks

    # Build a syntactically valid file whose tensor claims offset beyond file end.
    header = {
        "tfold": {"version": "0.1", "writer": "test"},
        "model": {"name": "t", "arch": "qwen3", "parameters": 1,
                  "vocab_size": 1, "context_length": 1,
                  "source_sha256": PIN, "source_url": "x"},
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "w", "dtype": "fp32", "shape": [1], "offset": 1_000_000,
             "nbytes": 4, "sha256": "0" * 64},
        ],
    }
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = MAGIC + struct.pack("<I", SPEC_VERSION_U32) + struct.pack("<I", len(hb)) + hb
    # pad to alignment then add 4-byte "tensor" that is NOT at the declared offset
    pad = (64 - (len(body) % 64)) % 64
    body += b"\x00" * pad + struct.pack("<f", 1.0)
    trailer = struct.pack("<Q", fnv1a_64_chunks([body]))
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".tfold", delete=False) as f:
        f.write(body + trailer)
        tmp = pathlib.Path(f.name)
    try:
        with pytest.raises(E.SpecError) as ei:
            tfold_py.read(tmp)
        assert ei.value.code == E.TENSOR_OOB
    finally:
        tmp.unlink()


# ---------- §4.13 CHECKPOINT_MISMATCH ----------
def test_rejects_pinned_checkpoint_mismatch(tmp_path):
    good = _good_fixture(tmp_path)
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(good, pin_source_sha256="b" * 64)
    assert ei.value.code == E.CHECKPOINT_MISMATCH

# ---------- §4.14 TENSOR_INTEGRITY ----------
def test_rejects_tensor_sha_mismatch_programmatically(tmp_path):
    """Flip a byte in the tensor region AND recompute the FNV trailer.
    The per-tensor SHA no longer matches but the integrity trailer does;
    reader must still reject at step 14.
    """
    from tfold_py._hash import fnv1a_64_chunks
    good = _good_fixture(tmp_path)
    bad = tmp_path / "ts.tfold"
    raw = bytearray(good.read_bytes())
    # Flip a byte near the end of the first tensor region (safely inside it)
    # Tensor region starts after preamble + header + padding; the first tensor
    # region definitely has a byte at least a few in.
    hlen = struct.unpack("<I", bytes(raw[12:16]))[0]
    tensor_region_start = PREAMBLE_NBYTES + hlen
    tensor_region_start += (64 - (tensor_region_start % 64)) % 64
    raw[tensor_region_start + 7] ^= 0xFF
    # Recompute trailer so integrity passes
    body = bytes(raw[:-TRAILER_NBYTES])
    new_trailer = struct.pack("<Q", fnv1a_64_chunks([body]))
    raw[-TRAILER_NBYTES:] = new_trailer
    bad.write_bytes(bytes(raw))
    with pytest.raises(E.SpecError) as ei:
        tfold_py.read(bad)
    assert ei.value.code == E.TENSOR_INTEGRITY
def test_writer_rejects_duplicate_tensor_names(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4,), dtype=np.float32)
    header = {
        "tfold": {"version": "0.1", "writer": "test"},
        "model": {"name": "t", "arch": "qwen3", "parameters": 4,
                  "vocab_size": 1, "context_length": 1,
                  "source_sha256": "0" * 64, "source_url": "x"},
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "w", "dtype": "fp32", "shape": [4], "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "w", "dtype": "fp32", "shape": [4], "offset": 0, "nbytes": 0, "sha256": "0" * 64},
        ],
    }
    with pytest.raises(E.SpecError) as ei:
        tfold_py.write(tmp_path / "dup.tfold", header, {"w": w})
    assert ei.value.code == E.TENSOR_DUP_NAME


def test_writer_rejects_shape_vs_buffer_mismatch(tmp_path):
    w_small = np.zeros((2,), dtype=np.float32)
    header = {
        "tfold": {"version": "0.1", "writer": "test"},
        "model": {"name": "t", "arch": "qwen3", "parameters": 4,
                  "vocab_size": 1, "context_length": 1,
                  "source_sha256": "0" * 64, "source_url": "x"},
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "w", "dtype": "fp32", "shape": [4], "offset": 0, "nbytes": 0, "sha256": "0" * 64},
        ],
    }
    with pytest.raises(tfold_py.TfoldError):
        tfold_py.write(tmp_path / "s.tfold", header, {"w": w_small})


def test_writer_rejects_unknown_dtype(tmp_path):
    w = np.zeros((4,), dtype=np.float32)
    header = {
        "tfold": {"version": "0.1", "writer": "test"},
        "model": {"name": "t", "arch": "qwen3", "parameters": 4,
                  "vocab_size": 1, "context_length": 1,
                  "source_sha256": "0" * 64, "source_url": "x"},
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "w", "dtype": "fp8_e4m3", "shape": [4], "offset": 0, "nbytes": 0, "sha256": "0" * 64},
        ],
    }
    with pytest.raises(E.SpecError) as ei:
        tfold_py.write(tmp_path / "u.tfold", header, {"w": w})
    assert ei.value.code == E.UNKNOWN_DTYPE
