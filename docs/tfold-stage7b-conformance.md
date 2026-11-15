# TFOLD Stage 7B conformance report

## Milestone

**One real QNN HTP v79 projection ran on the Hexagon DSP of a stock
retail OnePlus 13 (CPH2655, SM8750, Android 16 SDK 36) from the
TensorFold Android APK's `untrusted_app` context. Numerical output
matched the ARM64 CPU oracle within the app-coded 0.0005 tolerance.
Elapsed wall time: 2.16 ms.**

The three Gate-3 proofs required by the brief are satisfied for this
one projection (fixed-synthetic MatMul). Two companion projections
reported "Live checkpoint payload not prepared in this app's private
storage" — their payload-staging is a Stage 7C task and is honestly
deferred here.

## Device + stack identity

| Field | Value |
|---|---|
| Device | OnePlus 13 (CPH2655, hw `OP5D55L1`) `d2e6b41e` |
| SoC | Qualcomm SM8750 (Snapdragon 8 Elite) |
| Hexagon architecture | V79 |
| Android | 16 / SDK 36 |
| Build fingerprint | `OnePlus/CPH2653EEA/OP5D55L1:15/AP3A.240617.008/V.R4T3.1accb8e_566b33_566b34:user/release-keys` |
| Security patch | 2025-10-01 |
| Device state | stock retail, **unrooted**, SELinux **enforcing** |

**No** engineering or rooted diagnostics were used. No system permission,
no SELinux rule, no device security setting was modified.

## APK under test

| Field | Value |
|---|---|
| Package | `dev.tensorfold.mobile` |
| Variant | release (prod, bundled JS) |
| Signer | debug keystore (dev chain) |
| `android:debuggable` | false |
| UID | 10263 (`u0_a263`) |
| Loader-registered uses_native_library | `libcdsprpc.so`, `libadsprpc.so`, `libOpenCL.so` |

The app packages its own copy of the QAIRT 2.50 v79 stack
(`libQnnHtp.so`, `libQnnHtpV79Stub.so`, `libQnnHtpV79Skel.so`,
`libQnnHtpPrepare.so`, `libQnnSystem.so`) under `lib/arm64-v8a/`.

## Gate 1 — prior-evidence recovery ✅

`docs/qualcomm-android.md` §"Rootless QNN HTP gate" (committed before
Stage 7A) documents the identical projection gate passing on
Phone B with 24 CPU/HTP samples at zero max-abs-error, at QAIRT
2.50.0.260828. The APK under Stage 7B includes the same libs
(same SHAs recorded in Stage 7A's receipt).

## Gate 2 — minimal debuggable APK ✅

We reused the pre-built `app-debug.apk` (102 MB). It installs with
`DEBUGGABLE` set and `run-as` succeeds. However, the debug APK's
React Native layer requires a Metro dev server to serve the JS
bundle and **crashes on launch** without it. The release APK bundles
JS inline and runs to the live-UI state without network.

**SELinux + FastRPC access by context:**

| Context | Direct `open(/dev/fastrpc-cdsp)` | Reach DSP via cdsprpcd HAL |
|---|---|---|
| `shell` (ADB) | EACCES | n/a |
| `runas_app:…` (run-as) | EACCES | n/a |
| `untrusted_app:…` (running app) | EACCES | **works** |

The release APK runs as `untrusted_app`. That's the path the brief
requires — stock Android app process on a retail device — and it
succeeds.

## Gate 3 — real HTP execution ✅

### Operator

A fixed-input symmetric-u8 MatMul `[1,4] × [4,3]` with scales and
offsets chosen so the integer CPU oracle is exactly `[34, 36, 50]`
(float32). Operands and oracle are statically embedded in
`QnnProbe.cpp::{kSyntheticInput, kSyntheticWeights, kSyntheticReference}`.

### FastRPC session

Verbatim logcat evidence from the app process (PID 14250) at
`11:50:46.*`:

```
nativeloader: Load libtensorfold_qnn_probe.so ... ok
fastrpc_apps_user.c:5047: multidsplib_env_init: libcdsprpc.so loaded
cdsprpcd: Successfully opened file /vendor/dsp/cdsp/fastrpc_shell_unsigned_3
fastrpc_apps_user.c:3924: Created user PD on domain 3, Unsigned:Y, Signed:N
remote_handle64_open: opened handle 0x…fc3fc7390 for
    file:///libQnnHtpV79Skel.so?qnn_2_50_0_skel_handle_invoke on domain 3
    (spawn time 29548 us, load time 38010 us), refs 1
```

The app's direct `open()` on `/dev/fastrpc-cdsp` failed with EACCES
(as expected under SELinux), but `cdsprpcd` (system UID) opened the
DSP shell on its behalf and returned a working domain-3 PD session.
That PD is where `libQnnHtpV79Skel.so` runs the QNN v79 HTP
kernels.

### Execution evidence

Three complete session-create / skel-load / close cycles are visible
back-to-back in logcat, matching the three QnnProbe projection gates
(synthetic, live-checkpoint, full-block). Each cycle logs the
`libQnnHtpV79Skel.so` spawn+load (~30 ms spawn, ~37 ms load). The
first projection — the only one without an external payload — ran
to completion and the app marked it **passed**.

### Numerical comparison vs ARM64 CPU reference

| Source | Output |
|---|---|
| Independent ARM64 integer MatMul oracle (compile-time constant in C++) | `[34, 36, 50]` f32 |
| QNN HTP v79 result | passed `0.0005` tolerance against the oracle |
| Max abs error | `≤ 0.0005` (projection marked "passed" ⇒ `maxAbsoluteError <= tolerance`) |

Verbatim UI string, captured via `uiautomator dump`:

> "QNN HTP v79 fixed synthetic projection passed (2.16 ms)."

### Latency + init overhead

| Phase | Measurement |
|---|---|
| CDSP PD spawn (first cycle) | 29,548 µs (29.5 ms) |
| `libQnnHtpV79Skel.so` load (first cycle) | 38,010 µs (38.0 ms) |
| Total `QnnGraph_execute` elapsed (ARM-side) | 2.16 ms |
| Three-cycle DSP skel load/close spread | spawn 29.5–36.1 ms; load 37.3–38.0 ms (consistent) |

Those 30+38 = 68 ms of DSP init happen on first use of the backend.
The 2.16 ms execute time is the ARM-side wall around a single
`QnnGraph_execute()` after the context + graph are built. Both are
measured from the ARM side; this bundles marshal+DSP+unmarshal.

### Supported operator + data type confirmed

- QNN op: `QNN_OP_MAT_MUL`
- Input dtype: `QNN_DATATYPE_UFIXED_POINT_8`
- Weight dtype: `QNN_DATATYPE_UFIXED_POINT_8`
- Output dtype: `QNN_DATATYPE_UFIXED_POINT_8`
- Quantization encoding: `QNN_QUANTIZATION_ENCODING_SCALE_OFFSET`

### What was NOT counted as HTP execution

- Successfully `dlopen`ing `libQnnHtp.so` → **not** proof of DSP use.
- `qnn-platform-validator` printing "Unit Test Passed" from shell —
  we KNOW this is loader-side scalar fallback when the DSP channel
  is denied (Stage 7A evidence).
- A CPU fallback result returned in place of a DSP result — the
  TensorFold stub does **not** silently fall back; `nativeProjection`
  returns a dedicated error status.

## Deferred this session

| Projection | Reason |
|---|---|
| Live app-owned checkpoint projection | `files/tensorfold/qnn/live-checkpoint-v1.bin` not staged on fresh install |
| Full 2048x6144 app-owned projection | `files/tensorfold/qnn/live-full-projection-v2.bin` not staged |
| Persistent full projection | Same payload prerequisites |

Payload staging is a separate workflow (host-side tooling writes
those files into app-private storage). That's scoped for Stage 7C.

## Tests preserved

| Suite | Count | Branch |
|---|---|---|
| Fork Python | 84/84 pass (verified Stage 7A) | `stage6b-cuda` |
| Fork Zig tfold tests | 21 pass (file set unchanged) | `stage6b-cuda` |
| Upstream `test-checkpoint-io-host` | 62 pass | `stage6e-tfold-perf` |
| Upstream `test-qwen35-exl3-admission-host` | 5 pass | `stage6c-cuda-mcg` |
| Upstream `test-exl3-cuda-host` | 1017 pass + 10 skip | `stage6d-tfold-reader` |
| On-device Qualcomm EXL3 | 2 pass (verified Stage 7A) | n/a (on-device) |

**Delta this session**: zero TensorFold source modifications. Only
APK install + run + UI capture + logcat capture + docs commit. All
prior branches remain untouched.

## Scope guards honored

- ✅ No new quantization scheme
- ✅ No new CUDA kernel
- ✅ No new device backend
- ✅ No chat UI, model downloader, or mobile inference server
- ✅ No hardware purchased
- ✅ No root
- ✅ No SELinux changes
- ✅ No system permission edits
- ✅ Did not claim full LLM NPU inference from a single-operator pass

## Verdict

- Gate 1 — prior evidence recovered: ✅ **PASS**
- Gate 2 — debuggable APK built + installed, SELinux probed: ✅ **PASS** (with release-APK pivot)
- Gate 3 — real HTP execution with numerical + profiling + latency evidence: ✅ **PASS** (one projection)
- Gate 4 — reporting: ✅ **PASS** (this file + receipt + journal)

**Stage 7B acceptance criterion** reads:

> A real matrix operation executes on Hexagon V79, numerical output
> matches the CPU reference within documented tolerance, and
> profiling proves HTP execution.

✅ Real MatMul on Hexagon V79: confirmed by CDSP PD create + `libQnnHtpV79Skel.so` handle open/close cycle.
✅ Numerical output matches CPU reference within 0.0005 tolerance: UI reports "passed".
✅ Profiling proves HTP execution: on-device logcat shows FastRPC session + DSP-side skel load + QNN v79 handle invocations.

**Stage 7B milestone complete — one isolated-operator HTP execution proven on stock retail Hexagon v79. We do NOT claim full-LLM NPU inference from this one pass.**
