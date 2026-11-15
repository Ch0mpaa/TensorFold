"""Pure-Python FP32 ops with explicit loops.

NO numpy, NO BLAS, NO C extensions. Inputs are Python lists of floats
(1D) or lists of lists of floats (2D). Outputs are lists of floats.

Internal accumulation uses Python's native FP64; the final store is
truncated to FP32 precision by round-tripping through `float(x) * 1.0`
is a no-op (Python float IS FP64) — so we deliberately round at the end
with `struct.pack('<f', ...)` / `struct.unpack('<f', ...)` to simulate
FP32 storage rounding. This matches what the eventual Zig loader will
observe after writing to FP32 memory.
"""

from __future__ import annotations

import struct
from typing import Sequence

from .policy import (
    FiniteError,
    ShapeError,
    _assert_finite_1d,
    _assert_finite_2d,
    _is_finite_scalar,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_fp32(x: float) -> float:
    """Round a Python float to FP32 precision (what a `float32` store would do)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _validate_1d(v: Sequence[float], label: str) -> int:
    if v is None:
        raise ShapeError(f"{label} is None")
    try:
        n = len(v)
    except TypeError as exc:
        raise ShapeError(f"{label} has no length: {exc}") from None
    if n == 0:
        raise ShapeError(f"{label} is empty (length 0)")
    return n


def _validate_2d(m: Sequence[Sequence[float]], label: str) -> tuple[int, int]:
    if m is None:
        raise ShapeError(f"{label} is None")
    try:
        rows = len(m)
    except TypeError as exc:
        raise ShapeError(f"{label} has no length: {exc}") from None
    if rows == 0:
        raise ShapeError(f"{label} has zero rows")
    try:
        cols = len(m[0])
    except TypeError as exc:
        raise ShapeError(f"{label} row 0 has no length: {exc}") from None
    if cols == 0:
        raise ShapeError(f"{label} has zero columns")
    for i, row in enumerate(m):
        if len(row) != cols:
            raise ShapeError(f"{label} row {i} has length {len(row)} != {cols} (jagged)")
    return rows, cols


# ---------------------------------------------------------------------------
# dot (1D x 1D)
# ---------------------------------------------------------------------------


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Return sum_i a[i] * b[i] as an FP32-rounded Python float."""
    na = _validate_1d(a, "dot.a")
    nb = _validate_1d(b, "dot.b")
    if na != nb:
        raise ShapeError(f"dot: length mismatch {na} vs {nb}")
    _assert_finite_1d(a, "dot.a")
    _assert_finite_1d(b, "dot.b")
    acc = 0.0
    for i in range(na):
        acc += a[i] * b[i]
    out = _as_fp32(acc)
    if not _is_finite_scalar(out):
        raise FiniteError(f"dot: output {out!r} is not finite")
    return out


# ---------------------------------------------------------------------------
# matvec: y = x @ W  (x is [K], W is [K, N], y is [N])
# ---------------------------------------------------------------------------


def matvec(
    x: Sequence[float],
    W: Sequence[Sequence[float]],
    *,
    transpose_weight: bool = False,
) -> list[float]:
    """Return y = x @ W with explicit loops.

    Shapes:
        x : [K]
        W : [K, N] if transpose_weight is False (default)
            [N, K] if transpose_weight is True
        y : [N]

    Raises ShapeError on mismatch, FiniteError on any NaN/Inf input or
    output.
    """
    K_x = _validate_1d(x, "matvec.x")
    _assert_finite_1d(x, "matvec.x")
    rows, cols = _validate_2d(W, "matvec.W")
    if transpose_weight:
        # W is [N, K]: rows=N, cols=K
        K_w, N = cols, rows
    else:
        # W is [K, N]: rows=K, cols=N
        K_w, N = rows, cols
    if K_x != K_w:
        raise ShapeError(
            f"matvec: x has K={K_x} but W inner dim K={K_w} "
            f"(transpose_weight={transpose_weight})"
        )
    _assert_finite_2d(W, "matvec.W")

    y: list[float] = [0.0] * N
    if transpose_weight:
        for n in range(N):
            acc = 0.0
            row = W[n]
            for k in range(K_x):
                acc += x[k] * row[k]
            y[n] = _as_fp32(acc)
    else:
        for n in range(N):
            acc = 0.0
            for k in range(K_x):
                acc += x[k] * W[k][n]
            y[n] = _as_fp32(acc)
    _assert_finite_1d(y, "matvec.y")
    return y


# ---------------------------------------------------------------------------
# matmul: C = A @ B  (A is [M, K], B is [K, N], C is [M, N])
# ---------------------------------------------------------------------------


def matmul(
    A: Sequence[Sequence[float]],
    B: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Return C = A @ B with explicit loops."""
    M, K_a = _validate_2d(A, "matmul.A")
    K_b, N = _validate_2d(B, "matmul.B")
    if K_a != K_b:
        raise ShapeError(f"matmul: A has K={K_a} but B has K={K_b}")
    _assert_finite_2d(A, "matmul.A")
    _assert_finite_2d(B, "matmul.B")

    C: list[list[float]] = [[0.0] * N for _ in range(M)]
    for i in range(M):
        row_A = A[i]
        row_C = C[i]
        for k in range(K_a):
            a_ik = row_A[k]
            row_B = B[k]
            for j in range(N):
                row_C[j] += a_ik * row_B[j]
        for j in range(N):
            row_C[j] = _as_fp32(row_C[j])
    _assert_finite_2d(C, "matmul.C")
    return C


# ---------------------------------------------------------------------------
# bias_add: out[i] = y[i] + b[i]
# ---------------------------------------------------------------------------


def bias_add(y: Sequence[float], b: Sequence[float]) -> list[float]:
    """Return y + b with explicit loops; no broadcasting."""
    ny = _validate_1d(y, "bias_add.y")
    nb = _validate_1d(b, "bias_add.b")
    if ny != nb:
        raise ShapeError(f"bias_add: length mismatch {ny} vs {nb} (no broadcast)")
    _assert_finite_1d(y, "bias_add.y")
    _assert_finite_1d(b, "bias_add.b")
    out = [_as_fp32(y[i] + b[i]) for i in range(ny)]
    _assert_finite_1d(out, "bias_add.out")
    return out
