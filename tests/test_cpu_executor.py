"""End-to-end: load examples/demo.tfold, run y=x@W+b via tfold_cpu ops,
compare against the numpy reference saved beside it.

The arithmetic path is 100% tfold_cpu.ops. numpy appears only as the
oracle to compare against; tfold_cpu.executor converts numpy views to
Python lists before any arithmetic runs.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pytest

import tfold_cpu
import tfold_py
from tfold_cpu import CpuError, FP32_TOL, FiniteError, ShapeError, load_and_run


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo.tfold"
REF = ROOT / "examples" / "demo.reference.npz"
PIN = hashlib.sha256(b"tfold-v0.1-demo-fp32-fixture").hexdigest()


def _require_fixture():
    if not DEMO.exists() or not REF.exists():
        pytest.skip("examples/demo.tfold not built; run examples/build_demo.py")


def test_executor_matches_numpy_reference_within_tolerance():
    _require_fixture()
    y = load_and_run(DEMO, pin_source_sha256=PIN)
    ref = np.load(REF)["output"]
    assert len(y) == len(ref) == 4
    assert np.allclose(np.asarray(y, dtype=np.float32), ref,
                       atol=FP32_TOL["atol"], rtol=FP32_TOL["rtol"]), (
        f"tfold_cpu output {y} differs from numpy reference {ref.tolist()} "
        f"beyond FP32 tolerance {FP32_TOL}"
    )


def test_executor_output_is_fp32_rounded():
    """Every output value must round-trip through fp32 exactly."""
    import struct
    _require_fixture()
    y = load_and_run(DEMO, pin_source_sha256=PIN)
    for i, v in enumerate(y):
        again = struct.unpack("<f", struct.pack("<f", v))[0]
        assert v == again, f"y[{i}]={v!r} is not fp32-representable"


def test_executor_rejects_wrong_pinned_checkpoint():
    _require_fixture()
    with pytest.raises(tfold_py.SpecError) as ei:
        load_and_run(DEMO, pin_source_sha256="b" * 64)
    assert ei.value.code == "TFOLD_CHECKPOINT_MISMATCH"


def test_executor_rejects_missing_file(tmp_path):
    with pytest.raises(tfold_py.TfoldError):
        load_and_run(tmp_path / "does-not-exist.tfold")


def test_executor_rejects_file_without_required_tensors(tmp_path):
    """A .tfold file that lacks the demo's required tensors must fail cleanly."""
    import numpy as _np
    bogus_arr = _np.zeros((4,), dtype=_np.float32)
    header = {
        "tfold": {"version": "0.1", "writer": "test"},
        "model": {
            "name": "bogus", "arch": "qwen3", "parameters": 4,
            "vocab_size": 1, "context_length": 1,
            "source_sha256": "a" * 64, "source_url": "x",
        },
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "unrelated", "dtype": "fp32", "shape": [4],
             "offset": 0, "nbytes": 0, "sha256": "0" * 64},
        ],
    }
    out = tmp_path / "bogus.tfold"
    tfold_py.write(out, header, {"unrelated": bogus_arr})
    with pytest.raises(CpuError) as ei:
        load_and_run(out)
    assert "missing required tensor" in str(ei.value)


def test_executor_rejects_wrong_dtype_in_required_tensor(tmp_path):
    """If weight.linear_0 exists but is not fp32, executor refuses."""
    import numpy as _np
    w_u8 = _np.zeros((8, 4), dtype=_np.uint8)
    b = _np.zeros((4,), dtype=_np.float32)
    x = _np.zeros((8,), dtype=_np.float32)
    header = {
        "tfold": {"version": "0.1", "writer": "test"},
        "model": {
            "name": "mixed", "arch": "qwen3", "parameters": 36,
            "vocab_size": 1, "context_length": 1,
            "source_sha256": "a" * 64, "source_url": "x",
        },
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "weight.linear_0", "dtype": "u8",  "shape": [8, 4], "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "bias.linear_0",   "dtype": "fp32","shape": [4],    "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "input.x",         "dtype": "fp32","shape": [8],    "offset": 0, "nbytes": 0, "sha256": "0" * 64},
        ],
    }
    out = tmp_path / "mixed.tfold"
    tfold_py.write(out, header, {"weight.linear_0": w_u8, "bias.linear_0": b, "input.x": x})
    with pytest.raises(CpuError) as ei:
        load_and_run(out)
    assert "expected fp32" in str(ei.value)


def test_executor_rejects_non_finite_weight(tmp_path):
    """If a weight tensor has a NaN, the executor's finite check rejects."""
    import numpy as _np
    w = _np.zeros((8, 4), dtype=_np.float32)
    w[3, 1] = _np.nan
    b = _np.zeros((4,), dtype=_np.float32)
    x = _np.ones((8,), dtype=_np.float32)
    header = {
        "tfold": {"version": "0.1", "writer": "test"},
        "model": {
            "name": "nan", "arch": "qwen3", "parameters": 36,
            "vocab_size": 1, "context_length": 1,
            "source_sha256": "a" * 64, "source_url": "x",
        },
        "tokenizer": {"type": "tiktoken", "sha256": "a" * 64},
        "quant": {"scheme": "fp32", "bits_per_weight": 32.0, "block_size": 0},
        "tensors": [
            {"name": "weight.linear_0", "dtype": "fp32", "shape": [8, 4], "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "bias.linear_0",   "dtype": "fp32", "shape": [4],    "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "input.x",         "dtype": "fp32", "shape": [8],    "offset": 0, "nbytes": 0, "sha256": "0" * 64},
        ],
    }
    out = tmp_path / "nan.tfold"
    tfold_py.write(out, header, {"weight.linear_0": w, "bias.linear_0": b, "input.x": x})
    with pytest.raises(FiniteError):
        load_and_run(out)
