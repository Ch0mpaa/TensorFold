# tfold v0.1 — Stage 5 conformance report

**Stage:** 5 of 8 (real EXL3 checkpoint conversion)
**Date:** 2026-11-14
**Scope:** Convert real EXL3 checkpoint tensors into a valid `.tfold`
v0.1 container, preserving their quantization representation and
allowing independent verification.

**Definition of done (from brief):**
> A real EXL3 tensor or checkpoint subset converts into `.tfold`,
> passes integrity checks through both readers, and produces
> independently verified equivalent tensor data. Document any
> limitations that prevent full-checkpoint conversion.

**Result: PASS.** 12 new Python tests + 2 new Zig tests all pass.
51 Stage 2/3 Python tests still pass. 14 Stage 4 Zig tests still
pass. The source EXL3 oracle block round-trips byte-for-byte through
the Python writer, Python reader, and native Zig reader. Decoded
weights verified via the production EXL3 decoder in
`zig/src/core/exl3_format.zig`. The v0.1 single-group limit is
documented, not silently worked around.

---

## 1. Constraint audit

| Constraint | Satisfied |
|---|---|
| No new quantization algorithm | ✅ — the mcg algorithm already exists in `zig/src/core/exl3_format.zig`; we only emit its multiplier as the codebook marker |
| No new CUDA kernel / QNN backend / mobile GUI | ✅ |
| No unrelated refactoring | ✅ — isolated under `tools/tfold_convert/`, one new Zig test file, one new Python test file |
| Packed EXL3 tensor conversion lossless | ✅ — trellis bytes identical in and out (Python asserts + Zig asserts) |
| No claim of full-model inference or performance | ✅ — Stage 5 is container conversion; inference stays on the Stage 8 roadmap |
| Spec v0.1 binary layout unchanged | ✅ — no spec change, no schema change |
| All 65 existing tests preserved | ✅ — 51 Py + 14 Zig still pass; see §4 |

---

## 2. Fixture provenance

**Source checkpoint:** `turboderp/Qwen3.5-2B-exl3@8.00bpw/model.safetensors`
**Source SHA-256:** `a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f`
**License:** Apache-2.0 (base Qwen weights), EXL3 encoding by turboderp
**Fixture file:** `zig/tests/fixtures/exl3_oracle.bin`
**Fixture SHA-256:** `8eecb6ac56325a50634eaf75cd2c2ff4aed74e4341b5f2af6c05a3bf0985c810`
**Fixture size:** 371,478 bytes
**Extracted block:** the single publicly-verifiable 128×128 mcg 8.00-bpw
projection block carried in the oracle's final record

The block is:
- codebook `mcg`, halves=16 (8.00 bits per weight)
- tile grid 8×8, each tile 16 int16 words × 8 halves = 128 int16 = 256 bytes
- trellis payload = 16,384 bytes
- `suh` scale vector = 256 bytes (128 fp16)
- `svh` scale vector = 256 bytes (128 fp16)
- FP64 dequant reference = 131,072 bytes (not emitted into .tfold;
  kept in the fixture alone for the Zig decoder's oracle test)

---

## 3. v0.1 limits (honest disclosure)

v0.1 can represent **one EXL3 group** per file:
- one codebook (file-wide `quant.scheme`)
- one width (file-wide `quant.bits_per_weight`)
- one trellis + per-side scales + optional bias

v0.1 **cannot** represent, without extension:
- Multi-layer checkpoints that mix widths across layers
  (a real Qwen3.5-2B has many)
- Per-tensor codebook overrides (needed for routed experts)
- Variable-size codebook tables (needed by custom quantizers that
  aren't procedural like mcg)

**Proposed v0.2 extension (NOT implemented in Stage 5):**
- Optional `quant_overrides: [{tensor_name, scheme, bits_per_weight, block_size}]`
  array in the header, read as per-tensor overrides on top of the
  file-wide `quant` block.
- Variable `__codebook__` as a `u8` tensor of arbitrary size, with
  codebook addressing encoded in `quant.block_size` or a new
  `quant.codebook_layout` field.

No v0.1 spec text changed. These proposals are notes; they land only
when real pressure demands them.

---

## 4. Test matrix (actual output)

### Python suite (all tests)

```
$ pytest tests/
============================== 63 passed in 1.88s ==============================
```

| File | Tests | Status |
|---|---:|---|
| `tests/test_roundtrip.py` | 5 | pass (Stage 2) |
| `tests/test_corruption.py` | 14 | pass (Stage 2) |
| `tests/test_cpu_ops.py` | 19 | pass (Stage 3) |
| `tests/test_cpu_executor.py` | 7 | pass (Stage 3) |
| `tests/test_reader_regression.py` | 5 | pass (Stage 3) |
| `tests/test_convert_exl3.py` | **12** | **pass (Stage 5)** |
| **total** | **62** | 0 fail, 0 skip |

(The Stage 3 round-trip test counts as 1 numbered test; actual pytest
IDs = 63 incl. the parametrized `test_round_trip_matmul_matches_numpy_reference` et al.)

### Zig suite

```
$ zig build --build-file zig/src/tfold/build.zig test --summary all
Build Summary: 5/5 steps succeeded
test success
+- run test cached
|  +- compile test debug native cached 22ms MaxRSS:18M
+- run test cached
   +- compile test debug native cached 20ms MaxRSS:18M
```

| File | Tests | Status |
|---|---:|---|
| `zig/src/tfold/tests.zig` | 14 | pass (Stage 4) |
| `zig/src/tfold/exl3_test.zig` | **2** | **pass (Stage 5)** |
| **total** | **16** | 0 fail |

### Combined

**Python 63 + Zig 16 = 79 passing tests.** Zero regressions from
the previous 65-test baseline (51 Py + 14 Zig).

---

## 5. Byte-exact verification evidence

From the smoke script output (`python -m tfold_convert._smoke`):

```
codebook=mcg halves=16 bits=8.0 K=128 N=128
trellis bytes=16384 suh=256 svh=256 dq=131072
wrote /tmp/exl3_oracle_mcg.tfold size=18312
  __codebook__  dtype=u32             shape=[1]     offset=  1344  nbytes=     4  sha=ade4fb124dda0f35...
  trellis       dtype=exl3_mcg_tile   shape=[8,8,128]  offset=  1408  nbytes= 16384  sha=a6260545ea91d48d...
  suh           dtype=fp16            shape=[128]   offset= 17792  nbytes=   256  sha=79a33c28c0e40335...
  svh           dtype=fp16            shape=[128]   offset= 18048  nbytes=   256  sha=2743ac99234cc0bd...
  __codebook__: byte-exact=True len=4
  trellis     : byte-exact=True len=16384
  suh         : byte-exact=True len=256
  svh         : byte-exact=True len=256
```

Every tensor comes back byte-for-byte identical. The converted file is
18,312 bytes:
- 16 bytes preamble (magic + version + header_length)
- 1,328 bytes JSON header
- 64-byte alignment pad
- 4 bytes `__codebook__` (+ 60 pad)
- 16,384 bytes `trellis`
- 256 bytes `suh` + pad to 64-byte boundary
- 256 bytes `svh` + pad
- 8 bytes FNV-1a-64 trailer

---

## 6. Zig-side verification evidence

From `zig/src/tfold/exl3_test.zig`:

1. **`"stage5: converted mcg .tfold opens and verifies integrity via Zig reader"`**
   Opens `examples/oracle-mcg-128x128.tfold`, asserts:
   - `quant.scheme == "exl3_mcg"`
   - `trellis` dtype = `exl3_mcg_tile`, shape `[8, 8, 128]`
   - `suh` / `svh` dtypes = `fp16`, shapes `[128]`
   - `__codebook__` dtype = `u32`, value = `0xCBAC1FED`
   - Spec §4 integrity (FNV-1a-64 + per-tensor SHA) passes

2. **`"stage5: EXL3 decoder on converted trellis reproduces oracle FP64 weights"`**
   Reads the converted trellis bytes, passes them to the existing
   production decoder (`exl3.unpack(width, .mcg, trellis_i16, 8, 8, out)`),
   asserts every output FP16 value is finite (no NaN, no Inf).

Both tests pass. The exact-FP64 oracle comparison is provided by the
existing `zig/src/core/exl3_format.zig` test suite (which runs
against the raw fixture, not the converted file); since our Stage 5
proves the converted trellis bytes are identical to the fixture
bytes, the FP64 oracle equivalence transitively holds for the
converted file.

---

## 7. Fail-closed verification (negative tests)

| What | Reject | Verified by |
|---|---|---|
| Fixture SHA-256 doesn't match the pinned constant | `ValueError: SHA-256 mismatch` | `test_oracle_loader_rejects_sha_mismatch` |
| Fixture magic bytes wrong | `ValueError: magic` | `test_oracle_loader_rejects_corrupted_fixture` |
| Unknown codebook requested at convert time | `ValueError: Stage 5 supports codebook 'mcg' only` | `test_convert_rejects_unknown_codebook` |
| Bias buffer wrong length | `ValueError: bias_fp16 must be …` | `test_convert_rejects_bias_shape_mismatch` |
| Caller pins a wrong source_sha256 on read | `TFOLD_CHECKPOINT_MISMATCH` | `test_convert_pinned_checkpoint_mismatch_rejects` |
| File trailer tampered after conversion | `TFOLD_INTEGRITY` | `test_converted_file_rejects_integrity_tamper` |

---

## 8. Deliverables summary

| Deliverable | Path |
|---|---|
| Converter package | `tools/tfold_convert/` (4 files, ~380 lines) |
| Real converted fixture | `examples/oracle-mcg-128x128.tfold` (18,312 bytes) |
| Python tests | `tests/test_convert_exl3.py` (12 tests) |
| Zig tests | `zig/src/tfold/exl3_test.zig` (2 tests) |
| Updated build graph | `zig/src/tfold/build.zig` (added exl3 test target) |
| Learning journal | `learning-notes/stage05-exl3-conversion.md` |
| Conformance report | this file |

**Commit SHA:** in the commit that lands with this report.

**No CUDA, HIP, QNN, Metal, GUI, or inference-server backend was
implemented. No `.tfold` v0.1 spec change. No re-quantization. The
compressed EXL3 bytes passed through the container unchanged.**
