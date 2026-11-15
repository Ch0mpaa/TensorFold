"""Unit tests for tfold_cpu.ops (pure-Python arithmetic).

Numerical oracle: numpy. But numpy is only used to compute the
*expected* value; tfold_cpu.ops never touches numpy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tfold_cpu import FP32_TOL, FiniteError, ShapeError, bias_add, dot, matmul, matvec


ATOL = FP32_TOL["atol"]
RTOL = FP32_TOL["rtol"]


def _close(a, b):
    return np.allclose(a, b, atol=ATOL, rtol=RTOL)


# ---------------- dot ----------------


def test_dot_basic():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.5, -1.0, 2.5, 0.0]
    got = dot(a, b)
    exp = float(np.array(a, dtype=np.float32) @ np.array(b, dtype=np.float32))
    assert _close(got, exp)


def test_dot_length_mismatch():
    with pytest.raises(ShapeError):
        dot([1.0, 2.0], [1.0, 2.0, 3.0])


def test_dot_empty():
    with pytest.raises(ShapeError):
        dot([], [])


def test_dot_rejects_nan():
    with pytest.raises(FiniteError):
        dot([1.0, float("nan")], [1.0, 2.0])


def test_dot_rejects_inf():
    with pytest.raises(FiniteError):
        dot([1.0, float("inf")], [1.0, 2.0])


# ---------------- matvec ----------------


def test_matvec_basic():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((8,), dtype=np.float32)
    W_np = rng.standard_normal((8, 4), dtype=np.float32)
    exp = (x_np @ W_np).tolist()

    x = x_np.tolist()
    W = W_np.tolist()
    got = matvec(x, W, transpose_weight=False)
    assert len(got) == 4
    assert _close(got, exp)


def test_matvec_transpose_equivalent():
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((5,), dtype=np.float32)
    # W_T shape [N, K] = [3, 5]; then matvec with transpose_weight=True computes
    # y[n] = sum_k x[k] * W_T[n][k]
    W_T_np = rng.standard_normal((3, 5), dtype=np.float32)
    W_np = W_T_np.T  # [5, 3]
    exp = (x_np @ W_np).tolist()

    x = x_np.tolist()
    W_T = W_T_np.tolist()
    got = matvec(x, W_T, transpose_weight=True)
    assert len(got) == 3
    assert _close(got, exp)


def test_matvec_shape_mismatch():
    with pytest.raises(ShapeError):
        matvec([1.0, 2.0, 3.0], [[1.0, 2.0], [3.0, 4.0]])  # K=3 vs W rows=2


def test_matvec_empty_x():
    with pytest.raises(ShapeError):
        matvec([], [[1.0]])


def test_matvec_empty_W():
    with pytest.raises(ShapeError):
        matvec([1.0], [])


def test_matvec_rejects_jagged_W():
    with pytest.raises(ShapeError):
        matvec([1.0, 2.0], [[1.0, 2.0], [3.0]])


def test_matvec_rejects_nan_in_W():
    with pytest.raises(FiniteError):
        matvec([1.0, 2.0], [[1.0, 2.0], [float("nan"), 4.0]])


# ---------------- matmul ----------------


def test_matmul_basic():
    rng = np.random.default_rng(2)
    A_np = rng.standard_normal((3, 5), dtype=np.float32)
    B_np = rng.standard_normal((5, 2), dtype=np.float32)
    exp = (A_np @ B_np).tolist()

    got = matmul(A_np.tolist(), B_np.tolist())
    assert len(got) == 3
    assert len(got[0]) == 2
    for i in range(3):
        assert _close(got[i], exp[i])


def test_matmul_shape_mismatch():
    with pytest.raises(ShapeError):
        matmul([[1.0, 2.0]], [[1.0, 2.0]])  # K=2 vs K=1


def test_matmul_empty():
    with pytest.raises(ShapeError):
        matmul([], [[1.0]])


# ---------------- bias_add ----------------


def test_bias_add_basic():
    y = [1.0, 2.0, 3.0]
    b = [0.5, -1.0, 10.0]
    got = bias_add(y, b)
    assert _close(got, [1.5, 1.0, 13.0])


def test_bias_add_length_mismatch():
    with pytest.raises(ShapeError):
        bias_add([1.0, 2.0], [1.0, 2.0, 3.0])


def test_bias_add_rejects_nan():
    with pytest.raises(FiniteError):
        bias_add([1.0, float("nan")], [1.0, 2.0])


def test_bias_add_empty():
    with pytest.raises(ShapeError):
        bias_add([], [])


# ---------------- numerical policy documentation sanity ----------------


def test_fp32_tolerance_documented():
    assert "atol" in FP32_TOL and "rtol" in FP32_TOL
    assert FP32_TOL["atol"] > 0
    assert FP32_TOL["rtol"] > 0
