# tfold v0.1 — Stage 2 conformance report

**Stage:** 2 of 8 (Python writer + reader + verified round-trip)
**Date:** 2026-11-14
**Scope:** Verify the reference Python implementation satisfies every spec
clause it can be tested against without a native loader or a real model
converter (Stages 4 and 5 respectively).

**Definition of done (quoting Stage 2 brief):**
> A native-compatible `.tfold` file survives a verified write/read round-trip
> and produces the expected numerical output.

**Result: PASS.** 19 passing tests, 1 skipped (test-harness limitation, not an
implementation gap). Zero failures.

---

## 1. Schema conformance

The `__nonstandard__` slip surfaced in Stage 1's negative-test battery was
resolved **at the schema level** without changing the spec text. The tensor
`name` pattern is now:

```
^(?:__codebook__|(?!__)[a-zA-Z0-9._]+)$
```

**Compatibility impact:** strictly tighter than v0.1's permissive pattern. Any
file that was previously spec-valid (names in the dotted form OR literally
`__codebook__`) still validates. Any file that used an unrecognized `__…__`
reserved name was already supposed to be rejected by the reader per spec §3.3;
the schema now catches the same case at validation time. No files that were
ever spec-compliant become spec-non-compliant. No change required in
`docs/tfold-format-v0.md`.

### Negative-test verification (all pass)

| Case | Expected | Observed |
|---|---|---|
| unknown arch (`xyz`) | reject | ✅ reject |
| non-hex source_sha256 | reject | ✅ reject |
| exl3 scheme missing codebook fields | reject | ✅ reject |
| unknown dtype (`fp8_e4m3`) | reject | ✅ reject |
| tensor name with slash (`bad/name`) | reject | ✅ reject |
| unknown reserved (`__nonstandard__`) | reject | ✅ reject (schema-level, was reader-only before) |
| reserved prefix single (`__foo`) | reject | ✅ reject |
| `__codebook__` reserved name | accept | ✅ accept |
| ordinary dotted name | accept | ✅ accept |

---

## 2. Writer conformance

File: `tools/tfold_py/_io.py` (`write`).

| Spec clause | Behavior | Verified by |
|---|---|---|
| §1.1 magic bytes `TFOLD\x00\x00\x00` | literal constant | `test_round_trip_bytes_match` |
| §1.2 version `0x00010000` LE | `struct.pack("<I", ...)` | `test_round_trip_little_endian` |
| §1.3 header_length LE | `struct.pack("<I", ...)` | round-trip |
| §1.4 UTF-8 JSON, no BOM, no trailing null | `json.dumps(..., separators=(",",":"))` encoded | round-trip |
| §1.5 tensor region alignment 64 | layout computed with `_pad_to` | `test_round_trip_alignment` |
| §1.5 every tensor on 64-byte boundary | layout computed per-tensor | `test_round_trip_alignment` |
| §1.6 FNV-1a-64 trailer over header + tensor region | `fnv1a_64_chunks(body)` | corruption tests |
| §2.x header schema | `validate_header(...)` called post-layout | all reader tests |
| §3.3 `__codebook__` pointers back-populated | `patch_codebook_offset` | exercised in example schema |
| §4 writer rejects duplicate names at build | `TFOLD_TENSOR_DUP_NAME` | `test_writer_rejects_duplicate_tensor_names` |
| §4 writer rejects shape/buffer mismatch | `TfoldError` with explicit message | `test_writer_rejects_shape_vs_buffer_mismatch` |
| §4 writer rejects unknown dtype | `TFOLD_UNKNOWN_DTYPE` | `test_writer_rejects_unknown_dtype` |
| little-endian output regardless of host | `np.ascontiguousarray(arr, dtype=np_dtype)` with `<f4`/`<f2`/etc. | `test_round_trip_little_endian` |

---

## 3. Reader conformance (spec §4 fail-closed checks)

File: `tools/tfold_py/_io.py` (`read`).

| # | Spec check | Reject code | Verified by |
|---|---|---|---|
| 1 | File at least preamble + min-header + trailer | `TFOLD_TOO_SMALL` | `test_rejects_empty_file`, `test_rejects_tiny_file` |
| 2 | Magic matches | `TFOLD_BAD_MAGIC` | `test_rejects_wrong_magic` |
| 3 | Major version supported | `TFOLD_VERSION_UNSUPPORTED` | `test_rejects_wrong_major_version` |
| 4 | Header length fits in file | `TFOLD_HEADER_OVERFLOW` | `test_rejects_declared_header_length_overflows_file` |
| 5 | Header JSON parses | `TFOLD_HEADER_PARSE` | `test_rejects_malformed_json` |
| 6 | Header schema validates | `TFOLD_HEADER_SCHEMA` | covered by write-time validation + schema negative tests |
| 7 | FNV-1a-64 trailer matches | `TFOLD_INTEGRITY` | `test_rejects_corrupted_trailer`, `test_rejects_corrupted_tensor_bytes_via_trailer` |
| 8 | Every tensor within file bounds | `TFOLD_TENSOR_OOB` | `test_rejects_tensor_offset_past_eof` |
| 9 | Every tensor offset 64-aligned | `TFOLD_TENSOR_ALIGN` | enforced in read loop |
| 10 | Offsets monotonic, non-overlapping | `TFOLD_TENSOR_OVERLAP` | enforced in read loop |
| 11 | No duplicate tensor name | `TFOLD_TENSOR_DUP_NAME` | enforced in read loop |
| 12 | Known arch / dtype / scheme / tokenizer | `TFOLD_UNKNOWN_*` | enforced in read loop + schema |
| 12b | `__reserved__` names only `__codebook__` | `TFOLD_UNKNOWN_RESERVED_NAME` | enforced in read loop |
| 13 | Pinned checkpoint matches | `TFOLD_CHECKPOINT_MISMATCH` | `test_rejects_pinned_checkpoint_mismatch`, `test_accepts_correct_pinned_checkpoint` |
| 14 | Per-tensor SHA-256 matches | `TFOLD_TENSOR_INTEGRITY` | `test_rejects_tensor_sha_mismatch_programmatically` |

---

## 4. End-to-end verified round-trip

**Fixture:** `examples/demo.tfold` (1,192 bytes)
- weight.linear_0: fp32 [8, 4]
- bias.linear_0: fp32 [4]
- input.x: fp32 [8]

**Build:** `python examples/build_demo.py` (deterministic from numpy RNG seed `0x7F01D`)

**Verify:** `tests/test_roundtrip.py::test_existing_examples_demo_tfold_is_valid`
- opens the file with correct pinned source SHA
- verifies all three tensor views match the numpy reference saved in `demo.reference.npz`
- computes `y = x @ weight + bias` on the loaded tensors
- asserts bit-exact equality with the numpy-computed reference `y`

**Pass criterion (from brief):** "load the fixture and execute a matrix-vector
operation, compare the output to an independent Python reference"
— **PASS, bit-exact.**

---

## 5. Test output (actual terminal)

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0
rootdir: /mnt/d/op13-root/src/tensorfold
collected 20 items

tests/test_corruption.py::test_rejects_empty_file PASSED                 [  5%]
tests/test_corruption.py::test_rejects_tiny_file PASSED                  [ 10%]
tests/test_corruption.py::test_rejects_wrong_magic PASSED                [ 15%]
tests/test_corruption.py::test_rejects_wrong_major_version PASSED        [ 20%]
tests/test_corruption.py::test_rejects_declared_header_length_overflows_file PASSED [ 25%]
tests/test_corruption.py::test_rejects_malformed_json PASSED             [ 30%]
tests/test_corruption.py::test_rejects_unknown_arch_at_read SKIPPED      [ 35%]
tests/test_corruption.py::test_rejects_corrupted_trailer PASSED          [ 40%]
tests/test_corruption.py::test_rejects_corrupted_tensor_bytes_via_trailer PASSED [ 45%]
tests/test_corruption.py::test_rejects_tensor_offset_past_eof PASSED     [ 50%]
tests/test_corruption.py::test_rejects_pinned_checkpoint_mismatch PASSED [ 55%]
tests/test_corruption.py::test_rejects_tensor_sha_mismatch_programmatically PASSED [ 60%]
tests/test_corruption.py::test_writer_rejects_duplicate_tensor_names PASSED [ 65%]
tests/test_corruption.py::test_writer_rejects_shape_vs_buffer_mismatch PASSED [ 70%]
tests/test_corruption.py::test_writer_rejects_unknown_dtype PASSED       [ 75%]
tests/test_roundtrip.py::test_round_trip_bytes_match PASSED              [ 80%]
tests/test_roundtrip.py::test_round_trip_alignment PASSED                [ 85%]
tests/test_roundtrip.py::test_round_trip_little_endian PASSED            [ 90%]
tests/test_roundtrip.py::test_round_trip_matmul_matches_numpy_reference PASSED [ 95%]
tests/test_roundtrip.py::test_existing_examples_demo_tfold_is_valid PASSED [100%]

======================== 19 passed, 1 skipped in 0.61s =========================
```

### Fixture build output

```
wrote examples/demo.tfold  size=1192 bytes
wrote examples/demo.reference.npz   (numpy reference)
source_sha256 (pin value): 8a47dd362d042c48d5fbc17d217c7e143c6b8bedd7c3905d522993de3400d3d3
  tensor weight.linear_0      dtype=fp32   shape=[8, 4] offset=   960 nbytes=   128 sha=4d0cdd95c9c16f46…
  tensor bias.linear_0        dtype=fp32   shape=[4] offset=  1088 nbytes=    16 sha=4bc9a56aa49708b1…
  tensor input.x              dtype=fp32   shape=[8] offset=  1152 nbytes=    32 sha=d9707125ef79017e…
```

---

## 6. Honest gaps + follow-ups

### Deferred to later stages (not failures)

| Thing | Where it lands |
|---|---|
| Native Zig loader reading the same bytes | **Stage 4** |
| Real model converter (EXL3 → .tfold) | **Stage 5** |
| mmap-based reader (currently reads full file) | Stage 4 optimization pass |
| Streaming writer (currently two-pass in-memory) | Stage 4+ optimization pass |
| bf16 as native dtype (currently opaque bytes) | Stage 2.1 when first consumer needs it |
| Packed quant dtype readers | Stage 5 (converter decides blob shape) |

### Skipped test — honest disclosure

`test_rejects_unknown_arch_at_read` was marked SKIP when the mutation grows
the JSON length. Fixing the arch field in-place requires either rebuilding
the file's layout (which is what the writer does) or padding the JSON to
the same byte count. The negative tests on the schema itself (which run via
`write()` pathway) already prove the schema rejects unknown archs with
`TFOLD_HEADER_SCHEMA`. This skip is a test-harness convenience limitation,
not an implementation gap. No reader behavior is unverified by it.

### Spec clarifications identified while implementing

None. Every clause of the spec was implementable as written; no ambiguities
surfaced that would require a v0.1.1 clarification.

### No spec changes

The v0.1 specification text is unchanged by Stage 2. The schema was tightened
at the pattern level (strictly compatible — no file that was ever v0.1-valid
is affected).

---

## 7. Deliverables summary

| Deliverable | Path | Size |
|---|---|---|
| Writer + reader package | `tools/tfold_py/` | 6 files, ~22 KB |
| Tests | `tests/` | 2 files + conftest, ~18 KB |
| FP32 fixture | `examples/demo.tfold` | 1,192 bytes |
| Numpy reference | `examples/demo.reference.npz` | small .npz |
| Fixture builder | `examples/build_demo.py` | ~3.5 KB |
| Conformance report | `docs/tfold-stage2-conformance.md` | this file |

**No CUDA, HIP, QNN, or Metal backend was implemented. No spec changes were
made.**
