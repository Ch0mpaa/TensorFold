"""tfold_cpu — Python CPU-reference executor for .tfold v0.1 weights.

Purpose
-------
A deliberately-slow Python implementation of the arithmetic a native
backend would perform, used to verify that:

  1. the Stage 2 file reader actually surfaces loadable weights
  2. a toy linear-projection pipeline (y = x @ W + b) can be built end
     to end from real .tfold bytes
  3. the result reproduces an independent numpy reference within a
     documented FP32 tolerance

This is **not** a serving backend. No BLAS, no numpy arithmetic, no
GPU, no NPU. It exists so that Stage 4's native Zig loader has a
Python oracle to diff against.

Public API:

    load_and_run(path, *, pin_source_sha256=None)
        Open a .tfold file, run the demo linear calc, return a flat list
        of FP32 output values.

    dot(a, b), matvec(W, x, transpose_weight=False), matmul(A, B),
    bias_add(y, b)
        Explicit-loop implementations. See tools/tfold_cpu/ops.py.

    FP32_TOL
        Documented numerical tolerance. See tools/tfold_cpu/policy.py.

    CpuError, ShapeError, FiniteError
        Rejection taxonomy. See tools/tfold_cpu/policy.py.
"""

from .policy import CpuError, FiniteError, FP32_TOL, ShapeError
from .ops import bias_add, dot, matmul, matvec
from .executor import load_and_run

__all__ = [
    "load_and_run",
    "dot",
    "matvec",
    "matmul",
    "bias_add",
    "FP32_TOL",
    "CpuError",
    "ShapeError",
    "FiniteError",
]
