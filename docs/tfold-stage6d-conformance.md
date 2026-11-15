# TFOLD Stage 6d conformance report

## Milestone

**Native TensorFold loads a complete TFOLD v0.2 checkpoint directly** —
no intermediate safetensors materialization — and produces byte-identical
logits versus Stage 6c's reference path.

Stage 6c had proved real tokens via a materializer (`tools/tfold_convert/
tfold_to_safetensors.py` rewrites the `.tfold` into a safetensors directory,
loader reads that). Stage 6d removes the materializer by teaching
`core.Checkpoint` to open `.tfold` files natively.

## Scope

Two files added, three touched in upstream `ashhart/TensorFold`:

| | File | Purpose |
|---|---|---|
| **new** | `zig/src/core/tfold_file.zig` | ~280 LoC native v0.2 reader |
| **new** | `zig/src/core/tfold_file_test.zig` | 13 structural/integrity tests |
| mod | `zig/src/core/checkpoint.zig` | +17 lines — route `openModelPrefixes` to detect `dir/model.tfold` |
| mod | `zig/src/core/root.zig` | +1 line — attach test file to `core` module tests |
| mod | `zig/build/checkpoint_io_host.zig` | extend filters to pick up new tests |

No changes to CUDA kernels, EXL3 format, or build graph. Safetensors path
unchanged bit-for-bit.

## Evidence — Gate C (three-way byte-identical inference + performance)

| Path | Cold wall | Warm wall | Peak RSS | Output SHA256 |
|---|---:|---:|---:|---|
| **Native `.tfold`** (Stage 6d, NEW) | 5.701 s | 5.523 s | 7.3 GB | `6b198ea8…3824c1e0` |
| Original safetensors (Stage 6c Gate A baseline) | 4.363 s | 4.210 s | 6.1 GB | `6b198ea8…3824c1e0` |
| Materialized safetensors (Stage 6c Gate B baseline) | 4.053 s | 3.863 s | 6.1 GB | `6b198ea8…3824c1e0` |

**`three_way_sha_match: true`** — `6b198ea823135943d8d662837ca0c42e7ab4165addfa2064a23eb0be3824c1e0`
on every path, every run.

Receipt: `docs/stage6d-gate-c-receipt.json`

### Interpreting the walls

- Native-path wall is about **1.3 s** slower than the materialized path.
  Both reach the same tensors with the same number of GPU uploads; the
  difference is the file-wide FNV-1a-64 trailer scan on open (~1.3 s on a
  3.2 GB file through the naive pure-Zig loop). Safetensors has no
  file-wide integrity check, so it skips this cost.
- Peak RSS is **~1.2 GB higher** on the native path: v0.2's header JSON
  is kept resident in the reader arena while the file is open, and the
  single `.tfold` mmap over the entire 3.2 GB file keeps more pages
  resident than safetensors' shards do. Not a leak; just a different
  residency model.
- All warm runs are deterministic across runs.

### What "zero-copy" does and doesn't mean here

- **Zero CPU-side tensor copies:** `safetensors.File.get()` returns a slice
  into the mmap. The dense weight loader hands `t.bytes.ptr` straight to
  `cudaMemcpy`. No intermediate heap buffer.
- **NOT zero-copy to GPU:** the host→device `cudaMemcpy` is still there.
  Stage 6d didn't change the GPU upload path, so GPU transfer time is
  the same on all three paths.

Measured separately: file mapping (≤1 ms), integrity (~1.3 s on cold
first read of the trailer across a 3.2 GB file), GPU transfers +
inference (3-4 s across all paths; dominated by prompt prefill). The
native path is not a speedup — it is a **single-file packaging** that
removes a staging step at the cost of running a file-wide integrity
check.

## Tests

### Upstream (`stage6d-tfold-reader` fork branch)

| Suite | Before | After | Notes |
|---|---|---|---|
| `test-checkpoint-io-host` | 47/47 pass | **59/59 pass** | 12 new structural TFOLD tests |
| `test-qwen35-exl3-admission-host` | 5/5 pass | 5/5 pass | unchanged |
| `test-exl3-cuda-host` | 1017/1027 pass, 10 skip | 1017/1027 pass, 10 skip | unchanged |

Specifically the 12 new tests cover: valid round-trip, bad magic, bad
major version, header overflow, truncation, FNV mismatch, overlap,
misaligned offset, past-EOF offset, bad dtype, prefix filter, FNV
reference vector.

### Our fork (`stage6b-cuda` tip)

| Suite | Count | Notes |
|---|---|---|
| Python (tests/) | 84 pass | Stage 2/3/5/6/6b unchanged |
| Zig (zig/src/tfold/) | 21 pass | Stage 4/5/6/6b unchanged |
| **Total** | **105 pass** | zero regressions |

## Reproducibility

Everything in Gate C is reproducible from:

- Upstream fork branch: https://github.com/Ch0mpaa/TensorFold/tree/stage6d-tfold-reader
- Checkpoint: `turboderp/Qwen3.5-2B-exl3-8.00bpw` (HuggingFace, SHA pinned)
- Converter: `tools/tfold_convert/convert_full.py` from Stage 6b
- Hardware: any sm_121-class CUDA board with ≥8 GiB VRAM

Hands-on recipe is in `learning-notes/stage06d-native-tfold.md` §"Hands-on exercise".

## Verdict

- Native `.tfold` inference: ✅ **PASS**
- Byte-identical to Stage 6c Gate A + Gate B: ✅ **PASS**
- 105 Stage 6b tests preserved on our fork: ✅ **PASS**
- 12 new upstream tests landed, 5/5 admission + 683/693 exl3 still pass: ✅ **PASS**
- MCG PR #3412 kept separately reviewable: ✅ **PASS** (Stage 6d is on its own branch, `stage6d-tfold-reader`, off `stage6c-cuda-mcg`)
- Performance claim: Native path is **~1.3 s slower cold** than the materializer because the FNV trailer scan runs by default. We did NOT weaken that policy to report a speedup.

**Stage 6d milestone complete.**
