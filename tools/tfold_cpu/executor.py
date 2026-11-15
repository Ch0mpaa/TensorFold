"""Load a .tfold file via the Stage 2 reader and run the demo linear calc
using our own Python arithmetic.

Expected tensors (from examples/build_demo.py):

    weight.linear_0  fp32 [K, N]
    bias.linear_0    fp32 [N]
    input.x          fp32 [K]

Result:

    y = x @ W + b    fp32 [N]
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

# NOTE: numpy is used only to decode tensor bytes into Python lists. No
# numpy operation performs arithmetic in this module. The Stage 2 reader
# returns np.ndarray views; we call .tolist() at the first opportunity to
# force the data into pure Python types before any ops run.
import numpy as np

import tfold_py

from .ops import bias_add, matvec
from .policy import CpuError, FiniteError, ShapeError, _assert_finite_1d, _assert_finite_2d


DEMO_TENSOR_WEIGHT = "weight.linear_0"
DEMO_TENSOR_BIAS = "bias.linear_0"
DEMO_TENSOR_INPUT = "input.x"


def _require_fp32(spec: dict, label: str) -> None:
    if spec["dtype"] != "fp32":
        raise CpuError(f"{label}: expected fp32, got {spec['dtype']}")


def _tolist_1d(a: np.ndarray, label: str) -> list[float]:
    if a.ndim != 1:
        raise ShapeError(f"{label}: expected 1D, got {a.ndim}D with shape {a.shape}")
    out = [float(v) for v in a.tolist()]
    _assert_finite_1d(out, label)
    return out


def _tolist_2d(a: np.ndarray, label: str) -> list[list[float]]:
    if a.ndim != 2:
        raise ShapeError(f"{label}: expected 2D, got {a.ndim}D with shape {a.shape}")
    out = [[float(v) for v in row] for row in a.tolist()]
    _assert_finite_2d(out, label)
    return out


def load_and_run(
    path: str | pathlib.Path,
    *,
    pin_source_sha256: str | None = None,
) -> list[float]:
    """Open + validate a .tfold file, perform y = x @ W + b, return y.

    The arithmetic is done entirely in tfold_cpu.ops (Python loops, no
    numpy). numpy is used ONLY to decode tensor bytes into Python lists
    (via .tolist()).
    """
    header, tensors = tfold_py.read(path, pin_source_sha256=pin_source_sha256)

    tspec = {t["name"]: t for t in header["tensors"]}
    for name in (DEMO_TENSOR_WEIGHT, DEMO_TENSOR_BIAS, DEMO_TENSOR_INPUT):
        if name not in tspec:
            raise CpuError(f"missing required tensor: {name}")
        _require_fp32(tspec[name], name)

    w_np = tensors[DEMO_TENSOR_WEIGHT]
    b_np = tensors[DEMO_TENSOR_BIAS]
    x_np = tensors[DEMO_TENSOR_INPUT]

    W = _tolist_2d(w_np, "weight.linear_0")
    b = _tolist_1d(b_np, "bias.linear_0")
    x = _tolist_1d(x_np, "input.x")

    # Shape sanity: W is [K, N], x is [K], b is [N]
    K, N = len(W), len(W[0])
    if len(x) != K:
        raise ShapeError(f"input.x K={len(x)} != weight K={K}")
    if len(b) != N:
        raise ShapeError(f"bias N={len(b)} != weight N={N}")

    y = matvec(x, W, transpose_weight=False)
    y = bias_add(y, b)
    return y
