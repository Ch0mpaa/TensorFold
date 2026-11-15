"""Stage 5: convert the oracle EXL3 mcg block into .tfold v0.1 and
verify byte-for-byte that the packed trellis survives the trip.

Also exercises negative paths: bad metadata, wrong shapes, unknown
codebooks all fail closed.
"""

from __future__ import annotations

import hashlib
import pathlib
import struct

import numpy as np
import pytest

import tfold_convert as tc
import tfold_py


ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "zig/tests/fixtures/exl3_oracle.bin"


def _require_oracle():
    if not FIXTURE.exists():
        pytest.skip("oracle fixture not present")


def test_oracle_block_structure():
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    assert g.codebook == "mcg"
    assert g.halves == 16
    assert g.bits_per_weight == 8.0
    assert g.K == 128
    assert g.N == 128
    assert g.kt == 8
    assert g.nt == 8
    assert len(g.trellis_i16) == 16384
    assert len(g.suh_fp16) == 256
    assert len(g.svh_fp16) == 256
    assert len(g.dequant_reference_fp64) == 131072


def test_oracle_fixture_hash_is_pinned():
    _require_oracle()
    data = FIXTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == tc.oracle.FIXTURE_SHA256


def test_convert_roundtrip_byte_exact(tmp_path):
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    out = tmp_path / "exl3.tfold"
    tc.convert_group_to_tfold(g, out)

    bytes_back = tc.read_converted_group_bytes(out)
    # Codebook marker
    assert bytes_back["__codebook__"] == struct.pack("<I", 0xCBAC1FED)
    # Original EXL3 payloads survive verbatim
    assert bytes_back["trellis"] == g.trellis_i16
    assert bytes_back["suh"] == g.suh_fp16
    assert bytes_back["svh"] == g.svh_fp16


def test_convert_pinned_checkpoint_mismatch_rejects(tmp_path):
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    out = tmp_path / "exl3.tfold"
    tc.convert_group_to_tfold(g, out)
    with pytest.raises(tfold_py.SpecError) as ei:
        tfold_py.read(out, pin_source_sha256="b" * 64)
    assert ei.value.code == "TFOLD_CHECKPOINT_MISMATCH"


def test_convert_header_quant_block(tmp_path):
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    out = tmp_path / "exl3.tfold"
    hdr = tc.convert_group_to_tfold(g, out)
    assert hdr["quant"]["scheme"] == "exl3_mcg"
    assert hdr["quant"]["bits_per_weight"] == 8.0
    assert hdr["quant"]["block_size"] == 256
    # Codebook fields must be populated (required by schema for exl3_*)
    assert "codebook_sha256" in hdr["quant"]
    assert "codebook_offset" in hdr["quant"]
    assert "codebook_nbytes" in hdr["quant"]
    assert hdr["quant"]["codebook_nbytes"] == 4


def test_convert_header_source_identity(tmp_path):
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    out = tmp_path / "exl3.tfold"
    hdr = tc.convert_group_to_tfold(g, out)
    assert hdr["model"]["source_sha256"] == tc.oracle.MODEL_SHA256
    assert tc.oracle.MODEL_URL in hdr["model"]["source_url"]


# ---------- negative paths ----------


def test_convert_rejects_unknown_codebook(tmp_path):
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    # Build a shallow copy with a bogus codebook
    from dataclasses import replace
    bad = replace(g, codebook="inst3")
    out = tmp_path / "bad.tfold"
    with pytest.raises(ValueError, match="Stage 5 supports codebook 'mcg' only"):
        tc.convert_group_to_tfold(bad, out)


def test_convert_rejects_bias_shape_mismatch(tmp_path):
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    out = tmp_path / "bad.tfold"
    with pytest.raises(ValueError, match="bias_fp16 must be"):
        tc.convert_group_to_tfold(g, out, bias_fp16=b"\x00" * 10)


def test_oracle_loader_rejects_corrupted_fixture(tmp_path):
    _require_oracle()
    bad = tmp_path / "corrupt.bin"
    bad.write_bytes(b"NOTEXL3\x00" + b"\x00" * 100)
    with pytest.raises(ValueError, match="magic|SHA-256"):
        tc.load_oracle_mcg_block(bad)


def test_oracle_loader_rejects_sha_mismatch(tmp_path):
    _require_oracle()
    # Take the real fixture, flip a byte, feed to loader → SHA-256 reject
    data = bytearray(FIXTURE.read_bytes())
    data[100] ^= 0xFF
    bad = tmp_path / "flipped.bin"
    bad.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        tc.load_oracle_mcg_block(bad)


def test_converted_file_passes_full_stage2_reader(tmp_path):
    """The converted file must satisfy every spec §4 check, not just trivial ones."""
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    out = tmp_path / "exl3.tfold"
    tc.convert_group_to_tfold(g, out)
    # No pin → reader still verifies integrity + per-tensor SHAs
    header, tensors = tfold_py.read(out)
    assert "trellis" in tensors
    assert "__codebook__" in tensors
    assert header["quant"]["scheme"] == "exl3_mcg"


def test_converted_file_rejects_integrity_tamper(tmp_path):
    _require_oracle()
    g = tc.load_oracle_mcg_block()
    out = tmp_path / "exl3.tfold"
    tc.convert_group_to_tfold(g, out)
    raw = bytearray(out.read_bytes())
    # Flip the last byte of the trailer → integrity reject
    raw[-1] ^= 0x80
    out.write_bytes(bytes(raw))
    with pytest.raises(tfold_py.SpecError) as ei:
        tfold_py.read(out)
    assert ei.value.code == "TFOLD_INTEGRITY"
