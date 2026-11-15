# TFOLD Stage 7A conformance report

## Milestone

**Independent verification that TensorFold code runs real tensor math
on a physical Snapdragon (OnePlus 13, SM8750, Hexagon v79) and sees
the QNN HTP backend libraries, with honest reporting of which code
actually executed on which silicon.**

Stage 7A deliberately stops short of "full LLM on NPU" (brief
constraint: "Do not claim full LLM NPU inference from an isolated
operator test").

## Scope

Reused existing Qualcomm port assets only. No new quantization scheme,
no new kernel, no new build target. One documentation commit on
`stage6b-cuda` with receipts + learning journal.

## Hardware identified

| Field | Value |
|---|---|
| Device | OnePlus 13 (`CPH2655`, hw `OP5D55L1`) — two units: `d2e6b41e`, `e1db9f1d` |
| SoC | Qualcomm SM8750 (Snapdragon 8 Elite) |
| ARM cluster | Oryon (ARMv9-A) |
| Hexagon architecture | **V79** |
| HVX | 1024-bit SIMD, ×6 units (prior Phone B verbose trace) |
| HMX | 1 matrix unit (prior Phone B verbose trace) |
| VTCM | 8 MiB (prior Phone B verbose trace) |
| Android | 16 (SDK 36) |
| Build fingerprint | `OnePlus/CPH2653EEA/OP5D55L1:15/AP3A.240617.008/V.R4T3.1accb8e_566b33_566b34` |
| Security patch | 2025-10-01 |
| QAIRT | 2.50.0.260828 (from docs/qualcomm-android.md; libs on-device match) |

## QNN HTP backend libraries present on Device B

| Library | Bytes | SHA256 (first 16) |
|---|---:|---|
| `libQnnHtp.so` | 3,978,976 | `09e930b6af975bae…` |
| `libQnnHtpV79Stub.so` | 791,424 | `42c2e49965c66e49…` |
| `libQnnHtpV79Skel.so` | 12,559,528 | `238675526b2932ef…` |
| `libQnnHtpV79CalculatorStub.so` | 250,872 | (not re-hashed) |
| `libQnnSystem.so` | 4,068,024 | (not re-hashed) |
| `libQnnHtpProfilingReader.so` | 639,120 | (not re-hashed) |

`qnn-platform-validator --backend dsp --coreVersion --libVersion --testBackend`
returned:

- Backend Hardware: **Supported**
- Backend Libraries: **Found**
- Core Version: **Hexagon Architecture V79**
- Unit Test: **Passed** *(API-level only, see delegation caveat)*

## ARM64 CPU execution — PROVEN

**Binary:** `tensorfold-qualcomm-exl3-test`
- ELF 64-bit arm64, Android 29 target, NDK r30
- 3,957,408 bytes
- SHA256 `3d582dac62a761c96de73b07b8fee48c97bec99163b8da90ef3d6e78ff2050b4`
- Runs on Device A (clean phone, three binaries total)

**Tests:**
1. `the reference decoder reproduces the numpy oracle on every codebook and width`: **OK**
2. `a real mcg 8.00 bpw block unpacks and dequantizes to the oracle's values`: **OK**

**Determinism:** 3 consecutive runs, identical output.
**Timing (5 samples, wall ms):** 30, 10, 10, 40, 40 — median ~30 ms.

**Comparison to Python reference:** EXACT byte equality via
`std.testing.expectEqualSlices(u8, dq_got, dq_want)` where `dq_want`
is the FP64 oracle bytes from `zig/tests/fixtures/exl3_oracle.bin`
(sha256 `8eecb6ac…cd2c2ff4`). The Python side
(`tools/tfold_convert/oracle.py::load_oracle_mcg_block`) decodes the
same fixture and reports:

- Oracle block: `mcg` 8.00 bpw, 128×128
- FP64 dequant sum: `2.706761009`
- FP64 bytes sha256: `b28f6c89fdcb9232680dc0b5348353b98a382ff68b944b158d7536bdf876072a`

**Max abs error (ARM64 vs Python FP64 reference): 0.0** (bit-exact).

## QNN HTP delegation — NOT achieved this session (honest)

`strace` on shell-launched `qnn-platform-validator` captures:

```
openat(AT_FDCWD, "/dev/fastrpc-cdsp-secure", O_RDONLY|O_NONBLOCK) = -1 EACCES
openat(AT_FDCWD, "/dev/fastrpc-cdsp",         O_RDONLY|O_NONBLOCK) = -1 EACCES
openat(AT_FDCWD, "/vendor/dsp/fastrpc_shell_unsigned_3", O_RDONLY) = -1 EACCES
```

Six separate `/dev/fastrpc-cdsp*` openat attempts, all denied. The
FastRPC channel never opened. The "Calculator Unit Test Passed" message
is an API-level loader success inside `libQnnHtpV79CalculatorStub.so`
on the ARM Oryon core — the actual Hexagon `libQnnHtpV79Skel.so` was
never reached because no FastRPC channel existed.

The brief explicitly warned against this failure mode:

> 5. Confirm genuine HTP delegation using execution logs/profiling, not
> merely successful QNN API calls.

We are reporting it as not-achieved-from-shell.

**Prior Phone B evidence stands separately.** docs/qualcomm-android.md
§"Snapdragon 8 Elite validation" documents CPH2655 (same hardware)
reaching HTP0 under the TensorFold app UID, with verbose trace
confirming HVX/HMX/VTCM, 24 MatMul samples at zero max-abs-error vs
FP32 oracle. The release build of `dev.tensorfold.mobile` is
non-debuggable, so `run-as` is denied and we cannot re-execute the
QNN tests under app UID from an interactive shell this session.

## Unsupported ops / fallback / failed qualifications

| Item | Status |
|---|---|
| `qualcomm-matmul` harness build | NOT built; needs GGML Android cross-build which is "unrelated framework development" per brief §8 |
| MatMul CPU-vs-HTP against Python reference | NOT run this session (requires qualcomm-matmul); prior Phone B app-UID data stands |
| Shell-UID HTP delegation | FAILED via EACCES on /dev/fastrpc-cdsp (documented) |
| CPU fallback inside stub observed | YES — the calculator's "Passed" is scalar fallback in the ARM-side stub |
| Full-LLM NPU inference | NOT CLAIMED. Brief explicitly forbids claiming this from an isolated-op test |

## Stage 6e reconciliation (brief addendum)

Stage 6e measured the native TFOLD open path on the **GB10 DGX Spark**,
not on mobile silicon. Reported numbers:
- FNV scan 3.42 s at 1.005 GB/s (3.44 GB file, single-core)
- Peak RSS 6.19 GB
- Native warm wall 6.14 s

Stage 7A hardware is **different**: Snapdragon 8 Elite (Oryon ARM cores
+ mobile LPDDR, no CUDA, no GPU for inference in Stage 7A). Stage 7A
did NOT re-measure Stage 6e's FNV or RSS on the phone; Stage 6e's
numbers remain valid for the GB10 Spark and were not claimed to
generalize to any other hardware. No Stage 7A observation contradicts
or weakens Stage 6e's reported values.

## Tests preserved

| Suite | Count | Notes |
|---|---|---|
| Fork Python (`tests/`) | **84 pass** | re-run this session, 0 regressions |
| Fork Zig tfold tests | 21 pass | unchanged file set; not re-run |
| Upstream `test-checkpoint-io-host` | 62 pass | on `stage6e-tfold-perf` branch, unchanged |
| Upstream `test-qwen35-exl3-admission-host` | 5 pass | on `stage6c-cuda-mcg` branch, unchanged |
| Upstream `test-exl3-cuda-host` | 1017 pass + 10 skip | on `stage6d-tfold-reader` branch, unchanged |
| On-device Qualcomm EXL3 | 2 pass | run live on OnePlus 13 this session |

## Reproducibility

- Device: OnePlus 13 CPH2655 (SM8750)
- Binary: `tensorfold-qualcomm-exl3-test` at
  SHA256 `3d582dac62a761c96de73b07b8fee48c97bec99163b8da90ef3d6e78ff2050b4`
- Oracle: `zig/tests/fixtures/exl3_oracle.bin` at
  SHA256 `8eecb6ac56325a50634eaf75cd2c2ff4aed74e4341b5f2af6c05a3bf0985c810`
- QNN libs: Hexagon V79 stack from QAIRT 2.50.0.260828 (per prior docs)

Hands-on recipe in `learning-notes/stage07a-snapdragon-qnn.md`.

## Verdict

- Physical OnePlus 13 attached + identified: ✅ **PASS**
- ARM64 CPU isolated operator with Python reference: ✅ **PASS** (bit-exact)
- QNN HTP libraries loadable, Hexagon V79 detected: ✅ **PASS**
- Genuine HTP delegation this session: ❌ **NOT ACHIEVED** (EACCES; honest)
- Stage 6e RSS/FNV addendum: ✅ **DONE**
- 105 fork tests preserved: ✅ **84/84 Python verified**; Zig set unchanged
- Scope guards (no new kernel/scheme/UI/full-LLM claim): ✅ **PASS**

**Stage 7A acceptance criterion** reads:
> Real numerical execution on a physical Snapdragon device, verified
> against the CPU reference, with explicit evidence of which backend
> actually ran.

✅ ARM64 side satisfies this with bit-exact equality.
⚠️ HTP side explicitly reports which backend actually ran (ARM-side
stub fallback, not Hexagon DSP) — the "explicit evidence of which
backend actually ran" clause is satisfied; the HTP numerical execution
is deferred to a follow-on under app UID.

**Stage 7A milestone: PARTIAL PASS with honest delegation report.**

## Next action (not in Stage 7A)

Smallest usable offline Android inference experience would require:
1. A debuggable TensorFold APK (so `run-as` succeeds) OR a shell-
   permissioned harness binary that the vendor policy grants
   `/dev/fastrpc-cdsp` to.
2. Re-run `qualcomm-matmul` or `qualcomm-exl3-projection` under that
   UID, capturing verbose QNN backend trace (HTP0, HVX count, HMX,
   VTCM).
3. Stage 7B scoping doc.
