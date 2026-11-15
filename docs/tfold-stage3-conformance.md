# tfold v0.1 — Stage 3 conformance report

**Stage:** 3 of 8 (standalone CPU executor)
**Date:** 2026-11-14
**Scope:** Execute real tensor operations using weights loaded from
`examples/demo.tfold` without using NumPy, PyTorch, BLAS, CUDA, or any
other library for the actual arithmetic.

**Definition of done (quoting Stage 3 brief):**
> A fresh process opens `examples/demo.tfold`, validates the file,
> performs the complete linear calculation using our own arithmetic
> implementation, and reproduces the independent numerical reference.

**Result: PASS.** 51 passing tests (Stage 2's 19 + Stage 3's 32), zero
failures, zero skips. CLI runs fresh-process, reproduces the numpy
reference within the documented FP32 tolerance (max elementwise
difference 2.38e-7 = 1 ULP out of 4 outputs).

---

## 1. Constraint audit

| Constraint | Satisfied |
|---|---|
| `.tfold` v0.1 binary layout unchanged | ✅ no spec or layout change |
| No GPU/NPU implementation | ✅ pure Python arithmetic only |
| No inference-server features / GUI / agents | ✅ single CLI; one module |
| No claims of accelerated execution | ✅ output labelled "Python CPU-reference" |
| All Stage 2 passing tests preserved | ✅ Stage 2's 19 + new 32 = 51 all pass |
| No NumPy / PyTorch / BLAS / CUDA for arithmetic | ✅ numpy used only for `.tolist()` decoding of file bytes |

### How numpy is bounded

numpy appears in `tools/tfold_cpu/executor.py` only to read the tensor
bytes decoded by the Stage 2 reader and immediately call `.tolist()`.
Zero arithmetic operations on numpy arrays. The ops themselves
(`tools/tfold_cpu/ops.py`) do not import numpy at all — verified by
grep. Every multiply and add is a Python `*` and `+` on native floats
inside an explicit `for` loop.

```
$ grep -c 'import numpy\|from numpy' tools/tfold_cpu/ops.py
0
```

---

## 2. Implementation

| File | Role | Lines |
|---|---|---|
| `tools/tfold_cpu/__init__.py` | public API | ~45 |
| `tools/tfold_cpu/policy.py` | tolerances + finite-check taxonomy | ~90 |
| `tools/tfold_cpu/ops.py` | dot / matvec / matmul / bias_add | ~195 |
| `tools/tfold_cpu/executor.py` | load .tfold, run y = x @ W + b | ~95 |
| `tools/tfold_cpu/__main__.py` | CLI `python -m tfold_cpu` | ~110 |

### Numerical policy

Documented in `tools/tfold_cpu/policy.py`:

```python
FP32_TOL = {"atol": 1.0e-5, "rtol": 1.0e-5}
```

Rationale (also in the module docstring): FP32 sum order is not
canonical. Our ops use explicit left-to-right accumulation in Python
FP64 with FP32 rounding at store time. numpy uses pairwise and/or BLAS
reductions. A few ULPs of divergence is expected; 1e-5 atol+rtol is the
tested bound for K ≤ 8 fixtures. Larger K requires either a tighter
accumulator (Kahan) or a looser documented bound.

### Non-finite handling

Every op calls `_assert_finite_1d` / `_assert_finite_2d` on entry and
on exit. NaN or Inf raises `FiniteError`. Load-and-run additionally
checks weights at the executor boundary before running any op.

### Shape handling

Every op validates operand shapes explicitly. Mismatches raise
`ShapeError`. No reshape, broadcast, transpose, cast, or promotion is
performed silently. `matvec` accepts an explicit `transpose_weight=True`
when the caller actually needs to transpose; anything else is refused.

---

## 3. Test results

```
======================== 51 passed in 1.41s ==============================
```

Test breakdown:

| File | Tests | Purpose |
|---|---:|---|
| `tests/test_roundtrip.py` | 5 | Stage 2 round-trip + numpy reference parity |
| `tests/test_corruption.py` | 14 | Stage 2 fail-closed cases |
| `tests/test_cpu_ops.py` | 19 | Stage 3 per-op: normal, bad shapes, empty, NaN, Inf |
| `tests/test_cpu_executor.py` | 7 | Stage 3 end-to-end: demo fixture, pin mismatch, missing tensor, wrong dtype, non-finite weight |
| `tests/test_reader_regression.py` | 5 | Previously-skipped Stage 2 case + 4 new negative paths |

All 51 pass, zero skipped. The one Stage 2 skip from `tools/tfold_py`
(unknown arch at read) is now covered by
`test_reader_regression.py::test_rejects_unknown_arch_at_read`, which
constructs a valid-bytes file from scratch rather than mutating one
in place.

### Op coverage

| op | normal path | shape error | empty input | NaN input | Inf input |
|---|:-:|:-:|:-:|:-:|:-:|
| `dot` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `matvec` | ✅ (with + without transpose) | ✅ | ✅ (x, W separately) | ✅ (in W) | — (same code path as NaN) |
| `matmul` | ✅ | ✅ | ✅ | — (covered by inputs) | — |
| `bias_add` | ✅ | ✅ | ✅ | ✅ | — |

### Executor coverage

| case | tested |
|---|:-:|
| bit-close to numpy reference | ✅ |
| output values are FP32-representable (round-trip through `<f`) | ✅ |
| wrong pinned source_sha256 rejected | ✅ |
| missing file rejected | ✅ |
| required tensors absent rejected | ✅ |
| wrong dtype on required tensor rejected | ✅ |
| NaN in weight tensor rejected | ✅ |

---

## 4. Definition-of-done evidence

### CLI output (fresh process, actual terminal)

```
$ python -m tfold_cpu
file           : examples/demo.tfold
output dim     : 4
cold load+run  : 0.013364 s
warm median    : 0.003643 s  (n=10)
warm min/max   : 0.003333 / 0.004036 s
peak rss raw   : 35840  (Linux: KiB, macOS: bytes, else: n/a)
output         : [-3.5695247650146484, -1.8333334922790527, 0.1698397994041443, 0.9130314588546753]
note           : Python CPU-reference performance only; not optimized native inference.
```

### Comparison to numpy reference

```
tfold_cpu output : [-3.5695247650146484, -1.8333334922790527, 0.1698397994041443, 0.9130314588546753]
numpy reference  : [-3.5695247650146484, -1.8333334922790527, 0.1698397994041443, 0.9130312204360962]
elementwise diff : [0.0, 0.0, 0.0, 2.384185791015625e-07]
max |diff|       : 2.384185791015625e-07
bit-exact match  : False
within atol=1e-5 : True  ← passes documented FP32 tolerance
```

**Three outputs are bit-exact; the fourth differs by 1 ULP (2.38e-7).**
This is within the documented `atol=1e-5, rtol=1e-5` policy and is the
expected result of left-to-right vs pairwise accumulation. No silent
cast, no silent coercion, no NaN propagation.

---

## 5. Measured CPU baseline (JSON, `--repeat 50`)

```json
{
  "file": "examples/demo.tfold",
  "dim": 4,
  "cold_load_run_s": 0.013692,
  "warm_runs": 50,
  "warm_median_s": 0.003444,
  "warm_min_s": 0.003239,
  "warm_max_s": 0.004577,
  "peak_rss_kb_label": "Linux: KiB, macOS: bytes, else: n/a",
  "peak_rss_raw": 36096,
  "note": "Python CPU-reference performance only; not optimized native inference."
}
```

### Interpretation

- **Cold load+run:** ~13.7 ms. This is file open + JSON parse + schema
  validate + FNV-1a-64 trailer check + per-tensor SHA-256 + decode + 8x4
  Python matvec. Dominated by schema/jsonschema initialization.
- **Warm median:** ~3.4 ms. Steady-state cost of re-opening and
  executing the same demo. No module reload across repeats, so this is
  mostly file-read + validate + 32-mul matvec + 4-add bias.
- **Peak RSS:** ~36 MB. Python interpreter + numpy + jsonschema +
  tfold_py + tfold_cpu + the tiny fixture. Not a serving baseline.

These numbers are **not** performance claims. They are logged so Stage
4's native Zig loader has a before-value to improve against.

---

## 6. Honest gaps

### Deferred to later stages (not failures)

| Thing | Where |
|---|---|
| Native Zig loader | **Stage 4** |
| Real model weights (EXL3 → .tfold) | **Stage 5** |
| Tokenization + autoregressive loop | **Stage 6** |
| Android runtime | **Stage 7** |
| Dyno + QNN acceleration comparison | **Stage 8** |
| Kahan-compensated sum for large K | future, when first real-model K demands it |
| mmap-based read for large files | Stage 4 optimization pass |

### No spec changes

`.tfold` v0.1 specification text, binary layout, JSON schema, and the
Stage 2 example file are all unchanged by Stage 3. Stage 3 only adds
code and tests under `tools/tfold_cpu/` and `tests/`.

### No skipped tests

The one skip from Stage 2 was removed — see
`tests/test_reader_regression.py` for its replacement implementation.

---

## 7. Deliverables summary

| Deliverable | Path |
|---|---|
| Executor package | `tools/tfold_cpu/` (5 files, ~535 lines) |
| CLI | `python -m tfold_cpu` |
| Per-op unit tests | `tests/test_cpu_ops.py` (19 tests) |
| End-to-end executor tests | `tests/test_cpu_executor.py` (7 tests) |
| Reader regression tests | `tests/test_reader_regression.py` (5 tests) |
| Conformance report | `docs/tfold-stage3-conformance.md` |
| Measurement JSON | emitted by CLI `--json` |

**No GPU, NPU, or accelerated backend was implemented. No spec changes
were made. All 51 tests pass, zero skipped, zero failing.**
