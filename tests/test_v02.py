"""Stage 6b Gate 3 — v0.2 writer/reader + v0.1 backward compatibility."""

from __future__ import annotations

import pathlib
import struct

import numpy as np
import pytest

import tfold_py


EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "v02-mixed-widths-example.tfold"


def _v02_header_with_overrides():
    return {
        "tfold": {"version": "0.2", "writer": "test"},
        "model": {"name": "t", "arch": "qwen3", "parameters": 4,
                  "vocab_size": 1, "context_length": 1,
                  "source_sha256": "a" * 64, "source_url": "x"},
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {
            "scheme": "exl3_mcg",
            "bits_per_weight": 8.0,
            "block_size": 256,
            "overrides": [
                {"tensor_name": "head.weight", "bits_per_weight": 6.0, "scheme": "exl3_mcg"},
            ],
        },
        "tensors": [
            {"name": "__codebook__", "dtype": "u32", "shape": [1], "offset": 0, "nbytes": 4, "sha256": "0" * 64},
            {"name": "body.weight", "dtype": "exl3_mcg_tile", "shape": [2, 2, 32], "offset": 0, "nbytes": 128, "sha256": "0" * 64},
            {"name": "head.weight", "dtype": "exl3_mcg_tile", "shape": [1, 1, 24], "offset": 0, "nbytes": 24, "sha256": "0" * 64},
        ],
    }


def test_v02_write_read_roundtrip(tmp_path):
    header = _v02_header_with_overrides()
    buffers = {
        "__codebook__": np.array([0xCBAC1FED], dtype=np.uint32),
        "body.weight": bytes(128),
        "head.weight": bytes(24),
    }
    out = tmp_path / "rt.tfold"
    tfold_py.write_v02(out, header, buffers)
    hdr, tensors = tfold_py.read_v02(out)
    assert hdr["tfold"]["version"] == "0.2"
    assert "overrides" in hdr["quant"]
    assert len(hdr["quant"]["overrides"]) == 1
    assert tensors["body.weight"].nbytes == 128
    assert tensors["head.weight"].nbytes == 24


def test_v02_overrides_resolve_correctly(tmp_path):
    header = _v02_header_with_overrides()
    buffers = {
        "__codebook__": np.array([0xCBAC1FED], dtype=np.uint32),
        "body.weight": bytes(128),
        "head.weight": bytes(24),
    }
    out = tmp_path / "rt.tfold"
    tfold_py.write_v02(out, header, buffers)
    hdr, _ = tfold_py.read_v02(out)
    body = tfold_py.resolve_tensor_scheme(hdr, "body.weight")
    head = tfold_py.resolve_tensor_scheme(hdr, "head.weight")
    assert body["bits_per_weight"] == 8.0
    assert head["bits_per_weight"] == 6.0  # override applied
    assert body["scheme"] == "exl3_mcg"
    assert head["scheme"] == "exl3_mcg"


def test_v01_reader_rejects_v02_file(tmp_path):
    """v0.1 reader must reject v0.2 file (different magic)."""
    header = _v02_header_with_overrides()
    buffers = {
        "__codebook__": np.array([0xCBAC1FED], dtype=np.uint32),
        "body.weight": bytes(128),
        "head.weight": bytes(24),
    }
    out = tmp_path / "rt.tfold"
    tfold_py.write_v02(out, header, buffers)
    with pytest.raises(tfold_py.SpecError) as ei:
        tfold_py.read(out)
    assert ei.value.code == "TFOLD_BAD_MAGIC"


def test_v02_reader_rejects_v01_file():
    """v0.2 reader must reject v0.1 file (different magic)."""
    v01 = pathlib.Path(__file__).resolve().parents[1] / "examples" / "demo.tfold"
    if not v01.exists():
        pytest.skip("v0.1 demo fixture not present")
    with pytest.raises(tfold_py.SpecError) as ei:
        tfold_py.read_v02(v01)
    assert ei.value.code == "TFOLD_BAD_MAGIC"


def test_v02_schema_rejects_bad_override_format(tmp_path):
    header = _v02_header_with_overrides()
    header["quant"]["overrides"] = [{"tensor_name": "x", "bogus_key": 1}]  # fails additionalProperties:false
    with pytest.raises(tfold_py.SpecError) as ei:
        tfold_py.write_v02(tmp_path / "x.tfold", header, {
            "__codebook__": np.array([0xCBAC1FED], dtype=np.uint32),
            "body.weight": bytes(128),
            "head.weight": bytes(24),
        })
    assert ei.value.code == "TFOLD_HEADER_SCHEMA"


def test_v02_without_overrides_also_works(tmp_path):
    """A v0.2 file with no overrides should still open fine."""
    header = _v02_header_with_overrides()
    del header["quant"]["overrides"]
    out = tmp_path / "noover.tfold"
    tfold_py.write_v02(out, header, {
        "__codebook__": np.array([0xCBAC1FED], dtype=np.uint32),
        "body.weight": bytes(128),
        "head.weight": bytes(24),
    })
    hdr, _ = tfold_py.read_v02(out)
    assert "overrides" not in hdr["quant"]
    assert tfold_py.resolve_tensor_scheme(hdr, "head.weight")["bits_per_weight"] == 8.0


def test_committed_v02_example_opens():
    if not EXAMPLE.exists():
        pytest.skip("examples/v02-mixed-widths-example.tfold not built yet")
    hdr, tensors = tfold_py.read_v02(EXAMPLE)
    assert hdr["tfold"]["version"] == "0.2"
    assert "__codebook__" in tensors
    assert hdr["quant"]["overrides"][0]["tensor_name"] == "head.weight"


def test_v02_rejects_schema_violation_in_quant_scheme(tmp_path):
    header = _v02_header_with_overrides()
    header["quant"]["scheme"] = "bogus_scheme"
    with pytest.raises(tfold_py.SpecError) as ei:
        tfold_py.write_v02(tmp_path / "bad.tfold", header, {
            "__codebook__": np.array([0xCBAC1FED], dtype=np.uint32),
            "body.weight": bytes(128),
            "head.weight": bytes(24),
        })
    assert ei.value.code == "TFOLD_HEADER_SCHEMA"
