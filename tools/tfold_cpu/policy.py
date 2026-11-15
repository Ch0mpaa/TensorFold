"""Numerical policy for the tfold_cpu reference executor.

Documented tolerances
---------------------
FP32 matmul / matvec / dot products do not have a single canonical
summation order. Different implementations (naive left-to-right,
pairwise, Kahan-compensated, hardware-horizontal) produce values that
can differ by a few ULPs. Our reference implementation uses explicit
left-to-right accumulation in Python floats (IEEE-754 FP64 internal
precision collapsed down to FP32 only when storing the result). numpy's
reference uses its own pairwise/BLAS pathway.

Therefore the documented tolerance for Stage 3 FP32 outputs is:

    atol = 1e-5
    rtol = 1e-5

over the small (K <= 8) inner-dimension fixture bundled in this
repository. Larger K requires either a tighter summation algorithm
(Kahan) or a looser documented tolerance.

Non-finite handling
-------------------
NaN and +/-Inf are rejected at every boundary:

  - load_and_run rejects a weight tensor that contains any non-finite
    element with FiniteError before executing anything
  - every op checks its inputs on entry and its output on exit
  - callers see a FiniteError, never a silently-propagated NaN

Rationale: "fast and wrong is not useful" — a NaN that flows through
the pipeline is exactly the silent-wrongness the project principle
forbids.

No silent shape/dtype coercion
------------------------------
Every op requires its exact operand shapes. Mismatches raise
ShapeError; the ops do not reshape, broadcast, transpose, cast, or
promote. Callers that need a transpose pass transpose_weight=True
explicitly.
"""

from __future__ import annotations

import math
from typing import Final


FP32_TOL_ATOL: Final[float] = 1.0e-5
FP32_TOL_RTOL: Final[float] = 1.0e-5
FP32_TOL: Final[dict[str, float]] = {"atol": FP32_TOL_ATOL, "rtol": FP32_TOL_RTOL}


class CpuError(Exception):
    """Base class for tfold_cpu numerical / validation rejections."""


class ShapeError(CpuError):
    """Operand shapes are incompatible; op does not silently reshape."""


class FiniteError(CpuError):
    """Input or output contains a NaN or infinity; pipeline refuses to propagate."""


def _is_finite_scalar(x: float) -> bool:
    return math.isfinite(x)


def _assert_finite_1d(seq, label: str) -> None:
    for i, v in enumerate(seq):
        if not _is_finite_scalar(v):
            raise FiniteError(f"{label}[{i}] is not finite: {v!r}")


def _assert_finite_2d(mat, label: str) -> None:
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            if not _is_finite_scalar(v):
                raise FiniteError(f"{label}[{i}][{j}] is not finite: {v!r}")
