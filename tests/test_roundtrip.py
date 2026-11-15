"""Round-trip + verified matrix-vector execution against a numpy reference."""

from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np
import pytest

import tfold_py


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _fixture_payload():
    rng = np.random.default_rng(seed=0x7F01D)
    w = rng.standard_normal((8, 4), dtype=np.float32)
    b = rng.standard_normal((4,), dtype=np.float32)
    x = rng.standard_normal((8,), dtype=np.float32)
    return w, b, x


def _fixture_header(pin_sha: str) -> dict:
    return {
        "tfold": {"version": "0.1", "writer": "tfold-py@0.1.0"},
        "model": {
            "name": "tfold-demo-fp32",
            "arch": "qwen3",
            "parameters": 8 * 4 + 4,
            "vocab_size": 1,
            "context_length": 1,
            "source_sha256": pin_sha,
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


PIN = hashlib.sha256(b"tfold-v0.1-demo-fp32-fixture").hexdigest()


def test_round_trip_bytes_match(tmp_path):
    w, b, x = _fixture_payload()
    out = tmp_path / "rt.tfold"
    tfold_py.write(out, _fixture_header(PIN), {"weight.linear_0": w, "bias.linear_0": b, "input.x": x})
    header, tensors = tfold_py.read(out, pin_source_sha256=PIN)

    assert header["model"]["source_sha256"] == PIN
    np.testing.assert_array_equal(tensors["weight.linear_0"], w)
    np.testing.assert_array_equal(tensors["bias.linear_0"], b)
    np.testing.assert_array_equal(tensors["input.x"], x)


def test_round_trip_alignment(tmp_path):
    w, b, x = _fixture_payload()
    out = tmp_path / "rt.tfold"
    final = tfold_py.write(out, _fixture_header(PIN),
                            {"weight.linear_0": w, "bias.linear_0": b, "input.x": x})
    for t in final["tensors"]:
        assert t["offset"] % 64 == 0, f"tensor {t['name']} offset {t['offset']} not 64-aligned"


def test_round_trip_little_endian(tmp_path):
    """FP32 bytes on disk must be little-endian regardless of host."""
    w, b, x = _fixture_payload()
    out = tmp_path / "rt.tfold"
    final = tfold_py.write(out, _fixture_header(PIN),
                            {"weight.linear_0": w, "bias.linear_0": b, "input.x": x})
    raw = out.read_bytes()
    # Grab the first weight element directly from disk and compare to little-endian encoding
    w_spec = next(t for t in final["tensors"] if t["name"] == "weight.linear_0")
    w_bytes = raw[w_spec["offset"]:w_spec["offset"] + 4]
    import struct
    assert struct.unpack("<f", w_bytes)[0] == pytest.approx(float(w.flatten()[0]))


def test_round_trip_matmul_matches_numpy_reference(tmp_path):
    w, b, x = _fixture_payload()
    out = tmp_path / "rt.tfold"
    tfold_py.write(out, _fixture_header(PIN), {"weight.linear_0": w, "bias.linear_0": b, "input.x": x})
    _, tensors = tfold_py.read(out, pin_source_sha256=PIN)

    y_from_tfold = tensors["input.x"] @ tensors["weight.linear_0"] + tensors["bias.linear_0"]
    y_reference = x @ w + b

    # Bit-exact because inputs are identical and FP32 is deterministic
    np.testing.assert_array_equal(y_from_tfold, y_reference)
    assert y_from_tfold.shape == (4,)
    assert y_from_tfold.dtype == np.float32


def test_existing_examples_demo_tfold_is_valid():
    """The checked-in demo fixture must open and match its reference."""
    demo = ROOT / "examples" / "demo.tfold"
    ref  = ROOT / "examples" / "demo.reference.npz"
    if not demo.exists() or not ref.exists():
        pytest.skip("examples/demo.tfold not built yet; run examples/build_demo.py")
    header, tensors = tfold_py.read(demo, pin_source_sha256=PIN)
    r = np.load(ref)
    np.testing.assert_array_equal(tensors["weight.linear_0"], r["weight"])
    np.testing.assert_array_equal(tensors["bias.linear_0"], r["bias"])
    np.testing.assert_array_equal(tensors["input.x"], r["input"])
    y = tensors["input.x"] @ tensors["weight.linear_0"] + tensors["bias.linear_0"]
    np.testing.assert_array_equal(y, r["output"])
