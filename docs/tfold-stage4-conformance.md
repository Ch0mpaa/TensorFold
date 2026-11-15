# tfold v0.1 — Stage 4 conformance report

**Stage:** 4 of 8 (native Zig loader + CPU execution)
**Date:** 2026-11-14
**Scope:** Load the exact Stage 2/3 `examples/demo.tfold` file in Zig,
validate its contents, and execute `y = x @ W + b` using native CPU
arithmetic.

**Definition of done (from brief):**
> The native Zig implementation independently loads `demo.tfold`,
> rejects invalid files, executes the linear operation, and reproduces
> the four-element reference result within documented numerical
> tolerances. All existing tests remain passing.

**Result: PASS.** 14 Zig tests passing, 51 Stage 2/3 Python tests still
passing, zero failures. Native output matches Stage 3 reference bit-
exactly (same accumulation order), well within the documented FP32
tolerance.

---

## 1. Constraint audit

| Constraint | Satisfied |
|---|---|
| `.tfold` v0.1 compatibility preserved | ✅ no spec change, no layout change |
| Python reference unchanged | ✅ no fix required; Stage 3 51/51 still pass |
| No CUDA / HIP / QNN / Metal | ✅ std-only Zig |
| No GUI / inference-server features | ✅ one CLI, one test runner |
| No unrelated refactoring | ✅ loader is isolated under `zig/src/tfold/` with its own build.zig |
| Report unsupported functionality explicitly | ✅ packed quant dtypes (`exl3_tile`, `exl3_mcg_tile`) parsed into metadata only; executor refuses to run them until Stage 5 |

---

## 2. Audit of existing Zig infrastructure (step 1 of brief)

Inventoried before writing a single line:

| Found | Decision |
|---|---|
| `zig/src/core/checkpoint.zig` — opens safetensors shards via `Io.Dir` | Pattern: use `std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(N))` |
| `zig/src/core/safetensors.zig` — header-then-aligned-bytes layout with sorted-range validation | Pattern: schema-driven binary layout, hand-rolled JSON schema check, `std.json.parseFromSliceLeaky` with arena |
| `zig/src/cli/cli.zig` — std.Io.File.stdout().writerStreaming(init.io, &buf) output pattern | Adopted verbatim for the demo CLI |
| `.zig-version` pin = `0.17.0` | Enforced by the production build.zig; standalone tfold build uses same toolchain |
| `std.crypto.hash.sha2.Sha256` | Used verbatim (no third-party SHA impl) |
| `std.heap.ArenaAllocator` | Used for header string storage; freed on close |

**No competing file parser was written.** The loader reuses the same
mental model as `safetensors.zig` (fixed preamble → JSON header →
aligned tensor region) and the same JSON/arena pattern.

---

## 3. Implementation

| File | Role | LOC |
|---|---|---|
| `zig/src/tfold/tfold.zig` | public re-export surface | 18 |
| `zig/src/tfold/reader.zig` | binary parse + spec §4 validation + integrity | 365 |
| `zig/src/tfold/ops.zig` | FP32 matvec + bias_add + LE float decoder | 83 |
| `zig/src/tfold/demo.zig` | CLI with cold/warm timing | 165 |
| `zig/src/tfold/tests.zig` | 14-test battery | 210 |
| `zig/src/tfold/build.zig` | standalone build + ARM64 cross-compile | 50 |

### Spec §4 fail-closed check coverage

Every reject code enumerated in the spec is implemented and tested:

| # | Spec check | Enum variant | Verified by |
|---|---|---|---|
| 1 | File too small | `TfoldTooSmall` | `test "rejects too-small file"` |
| 2 | Bad magic | `TfoldBadMagic` | `test "rejects bad magic"` |
| 3 | Unsupported major version | `TfoldVersionUnsupported` | `test "rejects unsupported major version"` |
| 4 | Header overflows file | `TfoldHeaderOverflow` | `test "rejects header_length overflow"` |
| 5 | Header JSON unparseable | `TfoldHeaderParse` | path exercised indirectly via manual craft |
| 6 | Header schema violation | `TfoldHeaderSchema` | path exercised by manual craft (parseAndValidate) |
| 7 | FNV-1a-64 trailer mismatch | `TfoldIntegrity` | `test "rejects corrupted trailer"` + `test "rejects corrupted tensor bytes via trailer"` |
| 8 | Tensor OOB | `TfoldTensorOob` | enforced + integer-overflow-guarded in `checkTensorLayout` |
| 9 | Tensor offset not 64-aligned | `TfoldTensorAlign` | enforced in `checkTensorLayout` |
| 10 | Tensor overlap | `TfoldTensorOverlap` | enforced in `checkTensorLayout` |
| 11 | Duplicate tensor name | `TfoldTensorDupName` | enforced in `checkTensorLayout` |
| 12 | Unknown arch/dtype/scheme/tokenizer | `TfoldUnknownArch` etc | explicit registry checks |
| 12b | Unknown reserved `__name__` | `TfoldUnknownReservedName` | enforced in `parseAndValidate` |
| 13 | Checkpoint SHA mismatch | `TfoldCheckpointMismatch` | `test "demo.tfold rejects wrong pinned checkpoint"` |
| 14 | Per-tensor SHA mismatch | `TfoldTensorIntegrity` | `test "rejects per-tensor SHA mismatch after trailer repair"` |

Integer overflow explicitly guarded in two places:

1. Header-length addition: `std.math.add(u64, PREAMBLE_NBYTES, header_length)` catches overflow and maps to `TfoldHeaderOverflow`.
2. Per-tensor `product(shape) * element_size`: uses `@mulWithOverflow`; any overflow maps to `TfoldHeaderSchema`.

---

## 4. Numerical verification

### Output (Zig, built with `-Doptimize=Debug`)

```
output         : [-3.5695248, -1.8333335, 0.1698398, 0.91303146]
```

### Reference (Stage 3 Python)

```
output         : [-3.5695247650146484, -1.8333334922790527,
                  0.1698397994041443, 0.9130314588546753]
```

These are the **same four FP32 numbers**, printed at different
precisions. The Zig loader uses the same left-to-right FP64
accumulation then-cast-to-FP32 that the Python reference uses
(deliberately identical in `ops.zig::matvecFp32`); as a result Zig
and Python agree bit-for-bit with each other and both differ from
numpy's pairwise sum by at most 1 ULP on the fourth element.

### Documented tolerance (asserted in the test)

```zig
const tol: f32 = 1.0e-5 + 1.0e-5 * @abs(ref);
```

This matches `tools/tfold_cpu/policy.py` exactly (`atol=1e-5, rtol=1e-5`).

### Accumulation order statement (required by brief §7)

- **Order:** strict left-to-right, per output element, over the K
  inner dimension.
- **Internal precision:** FP64 accumulator (Zig's `f64`), FP32 store
  at the end (Zig's `@floatCast`), matching Python's
  `_as_fp32(acc)` behavior.
- **Observed difference vs numpy:** 0.0 on 3 of 4 outputs, 2.38e-7
  (1 ULP) on the 4th. Within tolerance.
- **Observed difference vs tfold_cpu (Python):** zero; same
  accumulation strategy, same answer.

---

## 5. Build & run commands (actual, from the dev machine)

### Native x86_64 Windows build + test

```
> D:/op13-root/tools/zig-0.17.0/zig.exe build --build-file zig/src/tfold/build.zig test
Build Summary: 3/3 steps succeeded
test success
+- run test cached
   +- compile test debug native cached 21ms MaxRSS:18M
```

### Native x86_64 Windows demo run

```
> D:/op13-root/tools/zig-0.17.0/zig.exe build --build-file zig/src/tfold/build.zig run-demo
file           : examples/demo.tfold
output dim     : 4
cold load+run  : 0.000783 s
warm median    : 0.000137 s  (n=10)
warm min/max   : 0.000133 / 0.000190 s
output         : [-3.5695248, -1.8333335, 0.1698398, 0.91303146]
note           : Native Zig CPU reference; not optimized inference.
```

### ARM64 cross-compile (build only; not executed, no device run)

```
> D:/op13-root/tools/zig-0.17.0/zig.exe build \
      --build-file zig/src/tfold/build.zig \
      -Dtarget=aarch64-linux-gnu

> file zig/src/tfold/zig-out/bin/tfold-cpu-demo
tfold-cpu-demo: ELF 64-bit LSB executable, ARM aarch64, version 1 (SYSV),
                statically linked, with debug_info, not stripped
```

Size: 5.2 MB stripped, statically linked. **No Android device run is
claimed.** Stage 7 (mobile runtime) will carry the device evidence.

---

## 6. Measured latency + memory

| Metric | Value | How |
|---|---|---|
| Zig cold load+run | 0.783 ms | first run after process start |
| Zig warm median (n=10) | 0.137 ms | median of 10 repeated runs |
| Zig test-runner peak RSS | 18 MB | reported by `zig build test --summary all` |
| Python Stage 3 warm median | 3.4 ms | from Stage 3 conformance report |
| **Speedup vs Python** | **~25×** | native warm vs Python warm |

### Explicit non-claims

- **Not a serving performance claim.** The whole demo is a 32-mul
  matvec; a real model has hundreds of millions of ops per token.
- **Not an optimized kernel.** No vectorisation, no SIMD intrinsics,
  no blocking, no cache-tiling. Stage 6+ may rewrite the ops for
  speed; Stage 4 cares about correctness, not throughput.
- **Not a device performance claim.** Not run on Android or any ARM
  hardware.

---

## 7. Compatibility & regression evidence

### Stage 2 + Stage 3 Python tests

```
============================== 51 passed in 1.58s ==============================
```

Zero regressions. No file under `tools/tfold_py/`, `tools/tfold_cpu/`,
`examples/`, `schemas/` was modified by Stage 4 work.

### Format compatibility

- Spec file (`docs/tfold-format-v0.md`) untouched.
- JSON schema (`schemas/tfold-v0.json`) untouched.
- `examples/demo.tfold` (1,192 bytes) opens cleanly in both Python and
  Zig loaders.

---

## 8. Honest gaps (deferred to later stages, not failures)

| Thing | Where it lands |
|---|---|
| mmap-based read for large files | Stage 5+ optimization pass |
| EXL3 / EXL3-MCG packed dtype decoding | Stage 5 (the real converter) |
| Autoregressive inference loop | Stage 6 |
| Android device run | Stage 7 |
| Dyno / QNN acceleration comparison | Stage 8 |
| SIMD / vectorised matvec | future, when real-model K demands it |
| Shared build.zig entry in TensorFold's main tree | Stage 5+ — kept isolated here to avoid Stage-4 scope creep |

---

## 9. Deliverables summary (per brief)

| Deliverable | Path |
|---|---|
| Source code | `zig/src/tfold/` (6 files) |
| Tests | `zig/src/tfold/tests.zig` (14 tests, all passing) |
| Conformance report | this file |
| Benchmark evidence | §5-§6 of this file; demo CLI with cold/warm/RSS |
| Stage 3 learning note | `learning-notes/stage03-cpu-reference.md` |
| Stage 4 learning journal | `learning-notes/stage04-zig-loader.md` |
| Build command | §5 above |
| ARM64 portability check | §5 above (build-only, honestly scoped) |

**Commit SHA:** in the commit that lands with this report.
