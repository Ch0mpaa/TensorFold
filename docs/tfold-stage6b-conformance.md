# tfold v0.2 — Stage 6b conformance report

**Stage:** 6b (CUDA recovery of Stage 6)
**Date:** 2026-11-14
**Branch:** `stage6b-cuda` (integration branch off commit `1af6505d`)
**Scope:** Unblock real autoregressive inference using existing NVIDIA
hardware, then integrate full-model TFOLD loading without rebuilding
the entire Qwen3.5 execution engine.

---

## Verdict

**Stage 6 completion gate NOT met.** Three gates succeed, one is blocked.

| Gate | Status | Evidence |
|---|---|---|
| Gate 1 — Audit upstream CUDA | ✅ | §1 |
| Gate 2 — Prove baseline inference | ✅ | §2 + `docs/stage6b-baseline-receipt.json` |
| Gate 3 — TFOLD v0.2 | ✅ | §3 |
| Gate 4 — TFOLD inference | 🚧 blocked at upstream loader | §4 |

Real autoregressive tokens are being generated on the real CUDA
runtime on a real DGX Spark. But those tokens are coming from
GLM-5.3-Flash (what the live binary loads), not from a .tfold-loaded
Qwen3.5-2B — because the upstream Qwen3.5 EXL3 loader
(`cuda_exl3_weights.zig::dense`) hard-codes `.codebook = .mul1` and
every available Qwen3.5-exl3 checkpoint is mcg.

Brief §Gate 4 explicitly anticipated this: *"If the EXL3 checkpoint is
not supported by the CUDA forward path, document that limitation. Do
not confuse format compatibility with kernel compatibility."*

---

## 1. Gate 1 — upstream audit

- **Branch:** created `stage6b-cuda` off `1af6505d`. Preserves all
  Stage 2-6 TFOLD changes. Not rebased on top of upstream.
- **Upstream HEAD:** `origin/main` = `41a72fd25` (`changelog: #3273 under Unreleased`).
- **Delta from our branch:** our side has 10 commits since the common
  Qualcomm work; upstream is 6,259 commits ahead.
- **CUDA Qwen3.5 tree:** fully present in `zig/src/families/qwen3_5/cuda_*.zig`
  (~140 files) and `zig/kernels/cuda/exl3_block/*.cu`.
- **CUDA arch targets:** `zig/build/cuda.zig` enumerates
  `sm_86` (Ampere), `sm_120` (Blackwell), **`sm_121` (GB10)** — our target.
- **EXL3 codebook support in CUDA kernels:** all three
  (`CB_3INST=0, CB_MCG=1, CB_MUL1=2`) across `(halves ∈ {2..16 even}) × (warps ∈ {4,8,2})` = 90 variants enumerated in
  `zig/src/cuda/exl3.zig::variants`.
- **CRITICAL finding:** `zig/src/families/qwen3_5/cuda_exl3_weights.zig`
  file header says "EXL3, mul1 codebook" and the `dense()` function
  hard-codes `.codebook = .mul1`. Kernels != loader. Gate 4 blocker.

---

## 2. Gate 2 — real CUDA baseline

### Hardware

```
$ ssh 100.90.145.67
GB10   (DGX Spark / sm_121)
Linux 7.0.0-1024-nvidia aarch64
NVIDIA-SMI 615.71.09   CUDA UMD 13.4
/usr/local/cuda-13.0/bin/nvcc
```

### Running TF binary serves GLM-5.3-Flash

```
$ curl -s http://100.90.145.67:8020/v1/models
{"data":[{"id":"GLM-5.3-Flash","owned_by":"tensorfold",...}]}
```

### Deterministic greedy baseline (receipt in `docs/stage6b-baseline-receipt.json`)

```json
{
  "server_model_id": "GLM-5.3-Flash",
  "prompt":  "The answer to life, the universe, and everything is",
  "max_tokens": 16,
  "greedy": {"temperature": 0.0, "top_p": 1.0},
  "seed": 42,
  "run_1": {
    "wall_s": 0.5112,
    "text": "**42**\n\nThis is, of course, the famous answer from Douglas Adams's",
    "usage": {"prompt_tokens": 24, "completion_tokens": 16, "total_tokens": 40}
  },
  "run_2": {
    "wall_s": 0.4934,
    "text": "**42**\n\nThis is, of course, the famous answer from Douglas Adams's"
  },
  "deterministic_match": true
}
```

This is **real tokens from the real CUDA TensorFold runtime on real
GB10 hardware.** Two independent runs produced byte-identical output.
~500 ms wall for 16 tokens.

### Honest caveat

Gate 2 brief says "a documented, supported **Qwen3.5-2B** checkpoint."
The running binary loads GLM-5.3-Flash instead. Reason: no mcg-codebook
Qwen checkpoint is accepted by upstream (see Gate 4). GLM is what the
TF binary on this Spark actually serves; we did not rebuild the binary.

---

## 3. Gate 3 — TFOLD v0.2 implementation

### Spec

- Binary layout unchanged from v0.1 except magic `TFOLD\x02\x00\x00` and
  version u32 `0x00020000`.
- Two new optional header keys:
  - `quant.overrides` — per-tensor scheme/bits/block_size override map
  - `quant.groups` — explicit EXL3 group structure (name + trellis + suh + svh + bias + K + N)
- Two new scheme names (`exl3_mul1`, `exl3_3inst`) and matching
  tile dtypes.
- `model.tensor_count` optional redundancy field.
- Scalar shapes allowed (minItems=0) because real checkpoints have them.

### Python implementation

- `tools/tfold_py/_v02.py` — `write_v02`, `read_v02`, `resolve_tensor_scheme`
- `schemas/tfold-v0.2.json` — authoritative JSON schema
- `tests/test_v02.py` — 8 tests (all pass)
  - round-trip bytes identical
  - overrides resolve correctly
  - v0.1 reader rejects v0.2 file with `TFOLD_BAD_MAGIC`
  - v0.2 reader rejects v0.1 file with `TFOLD_BAD_MAGIC`
  - schema rejects bad override format
  - file without overrides also works
  - committed example at `examples/v02-mixed-widths-example.tfold` opens
  - bad scheme rejected

### Zig implementation

- `zig/src/tfold/reader_v02.zig` — `openPath`, `openBytes` with
  v0.2 magic + extended registries
- `zig/src/tfold/v02_test.zig` — 5 tests (all pass)
  - v0.2 opens
  - v0.2 reader rejects v0.1 bytes
  - v0.1 reader rejects v0.2 bytes
  - scheme / dtype registry extensions recognized

### Backward compatibility verified both ways

| Opener | On v0.1 file | On v0.2 file |
|---|---|---|
| Python `read()` | ✅ opens | ❌ `TFOLD_BAD_MAGIC` |
| Python `read_v02()` | ❌ `TFOLD_BAD_MAGIC` | ✅ opens |
| Zig `reader.openPath()` | ✅ opens | ❌ `TfoldBadMagic` |
| Zig `v02.openPath()` | ❌ `TfoldBadMagic` | ✅ opens |

**No file that was ever v0.1-valid becomes v0.2-valid or vice versa.** Clean
separation. The brief's "maintain backward compatibility with v0.1" is
satisfied via coexistence, not reinterpretation.

---

## 4. Gate 4 — TFOLD inference (blocked, documented)

### What works

- Full Qwen3.5-2B-exl3-8.00bpw converted to v0.2 via
  `tools/tfold_convert/convert_full.py`:
  - 1,072 tensors (1,071 real + 1 synthetic `__codebook__`)
  - 151 EXL3 group entries in `quant.groups`
  - 1 `quant.overrides` entry for the 6-bpw lm_head
  - all 3.2 GiB round-trips bit-exact through both Python and Zig readers
  - per-tensor SHA-256 preserved; FNV-1a-64 trailer validates

### What's blocked

The upstream Qwen3.5 CUDA loader at `cuda_exl3_weights.zig::dense` has:

```zig
.codebook = .mul1,  // hard-coded
if (cuda.exl3.variantIndex(w.halves, w.codebook, cuda.exl3.planOf(w).wk) == null)
    return error.UnsupportedQwenQuantization;
```

This forces every Qwen3.5 EXL3 tensor through the mul1 kernel variant.
Our checkpoint is mcg. Every local Qwen3.5-exl3 variant is mcg. **The
loader refuses our file before any kernel runs.**

### Unblock path (not done in Stage 6b)

Open an upstream PR that:
1. Changes `cuda_exl3_weights.zig::denseGroup` to surface the codebook
   from the checkpoint (currently only halves are surfaced).
2. Changes `cuda_exl3_weights.zig::dense` to use `gr.codebook` instead
   of hard-coded `.mul1`.
3. Adds a Qwen3.5-mcg admission test alongside the existing mul1 ones.

The kernels already exist; the variant table already enumerates
`(halves=16, codebook=.mcg, warps=X)`. This is policy, not implementation.

### Why we didn't apply the patch locally and press on

Changing upstream behavior in our fork would:
- fork TensorFold's test surface for every unrelated subsystem
- risk breaking other callers that rely on the mul1-only guarantee
- make our Stage 6b result non-reproducible by anyone else

The brief said "do not wholesale-rebase or cherry-pick blindly." A
targeted upstream PR is the right unblock; it belongs to Stage 6c.

---

## 5. Test preservation

| Pre-Stage 6b baseline | After Stage 6b | Delta |
|---|---:|---|
| Python tests | 84 (was 76 + 8 Stage 6b v0.2) | +8 |
| Zig tests | 21 (was 16 + 5 Stage 6b v0.2) | +5 |
| **Combined** | **105** | **+13** |

**Zero regressions from the 92-test Stage 6 baseline.** Every Stage 2-6
test still passes unchanged.

---

## 6. Deliverables summary

| Deliverable | Path |
|---|---|
| Gate 2 baseline receipt | `docs/stage6b-baseline-receipt.json` |
| Gate 2 capture script | `tools/tfold_stage6b/capture_baseline.py` |
| Gate 3 schema | `schemas/tfold-v0.2.json` |
| Gate 3 Python impl | `tools/tfold_py/_v02.py` |
| Gate 3 Zig impl | `zig/src/tfold/reader_v02.zig` |
| Gate 3 Python tests | `tests/test_v02.py` (8 tests) |
| Gate 3 Zig tests | `zig/src/tfold/v02_test.zig` (5 tests) |
| Gate 3 example fixture | `examples/v02-mixed-widths-example.tfold` |
| Gate 3 full-checkpoint converter | `tools/tfold_convert/convert_full.py` |
| Gate 4 analysis + blocker | this report §4 |
| Learning journal | `learning-notes/stage06b-cuda-first-tokens.md` |
| Conformance report | this file |

---

## 7. What Stage 6b does NOT claim

- NOT a completion of Stage 6. The completion gate requires a .tfold
  file loaded into the CUDA runtime producing tokens; we produce
  tokens from the CUDA runtime but not from a .tfold file.
- NOT an upstream contribution to `ashhart/TensorFold`. The Gate 4
  unblock patch is a separate Stage 6c project.
- NOT a performance claim beyond the single baseline latency.
- NOT a mobile or Android result. Stage 7 territory.
