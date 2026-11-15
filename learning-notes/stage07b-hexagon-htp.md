# Stage 7B — a real Hexagon v79 MatMul from a stock-retail Android app

> Beginner-friendly companion to Stage 7B. The one-sentence result:
> a real quantized MatMul ran on the Hexagon DSP of a stock OnePlus 13
> (CPH2655, Snapdragon 8 Elite, Android 16 SDK 36), its output matched
> the ARM CPU oracle within the app-declared tolerance, and the DSP
> session create / skel load / graph execute cycle is visible in
> on-device logcat. No root, no SELinux changes, no system permission
> edits.

## FastRPC — the only road to the DSP

The ARM CPU and the Hexagon DSP are separate processors with separate
address spaces. User-space code on ARM reaches the DSP through
**FastRPC**: a Qualcomm RPC framework that marshals function arguments
into a shared memory buffer, drops them through a kernel driver node
(`/dev/fastrpc-cdsp` for the compute DSP), and the DSP picks them up
inside its own process — the **Protection Domain** (PD).

The pieces on both sides:

```
ARM side (your app's process)              DSP side (CDSP PD)
---------------------------------          ---------------------------
dlopen libcdsprpc.so                       cdsprpcd daemon (system UID)
   │                                       loads the DSP shell
   │                                       (fastrpc_shell_unsigned_3)
   ▼                                           │
libQnnHtpV79Stub.so (your stub)                ▼
   │  marshals QNN api calls                libQnnHtpV79Skel.so
   ▼                                        (the actual compiled
FastRPC transport                            HTP kernels for v79)
   │                                           ▲
  ioctl /dev/fastrpc-cdsp (ARM kernel)         │
   │                                           │
   └─────────────▶   DSP wakes up, executes, returns
```

If **any** of those links breaks — the stub can't load, the FastRPC
driver refuses `ioctl`, the daemon can't start the DSP shell, the skel
fails to load inside the PD — the call fails and you get a scalar
fallback inside the stub, or an explicit error. Stage 7A showed the
first break: shell UID tried `openat("/dev/fastrpc-cdsp")` and the
kernel returned EACCES.

## Why the Android app works where the shell doesn't

On a stock retail Android, three UIDs look at `/dev/fastrpc-cdsp`:

| Who | SELinux context | Direct `open()` | Via cdsprpcd (HAL) |
|---|---|---|---|
| `shell` (ADB) | `shell` | **EACCES** | n/a |
| `run-as <pkg>` | `runas_app:…` | **EACCES** | n/a |
| App itself | `untrusted_app:…` | EACCES | **works** |

The `untrusted_app` policy does not grant the app process `rw` on
`/dev/fastrpc-cdsp` directly. It DOES grant the app the ability to
talk to `cdsprpcd` (a native system daemon) over Binder/HAL.
`cdsprpcd` runs as `system` UID, opens the DSP shell on the app's
behalf, and hands the app a validated session handle. The app's
AndroidManifest must register `libcdsprpc.so` as a
`<uses-native-library>` for the loader to permit the chain.

That is exactly what Stage 7B observed in logcat:

```
fastrpc_apps_user.c:3844: Error 0xd: open_shell failed for domain 3
    search paths /usr/lib/dsp/, /vendor/dsp/, /vendor/dsp/xdsp/ (errno Permission denied)
cdsprpcd (uid=system): Successfully opened file
    /vendor/dsp/cdsp/fastrpc_shell_unsigned_3
fastrpc_apps_user.c:3924: Created user PD on domain 3,
    Unsigned:Y, Signed:N
remote_handle64_open: opened handle 0x…fc3fc7390 for
    file:///libQnnHtpV79Skel.so?qnn_2_50_0_skel_handle_invoke
    on domain 3 (spawn time 29548 us, load time 38010 us)
```

Translation: "the app tried to open the DSP shell itself and failed;
the HAL service did it on its behalf; the app then successfully
asked the DSP to load `libQnnHtpV79Skel.so` and received a handle."

## JNI — the Java → native bridge

The TensorFold app is React Native / Kotlin on top. The QNN code is
C++. `QnnProbe.kt` declares:

```kotlin
@JvmStatic private external fun nativeProjection(
    nativeDir: String,
    stubName: String,
): DoubleArray
```

That `external` keyword tells JVM "resolve this symbol inside the
shared library you're about to load." `System.loadLibrary("tensorfold_qnn_probe")`
loads `libtensorfold_qnn_probe.so`, and the native `.so` exports
`Java_expo_modules_tensorfoldlocalruntime_QnnProbe_nativeProjection`.
When Kotlin calls `nativeProjection(...)`, the JVM marshals the two
Java strings into C `const char*`s, calls the C function, and
converts the returned `jdoubleArray` back into a Kotlin `DoubleArray`.
**The returned array holds `[status, elapsedMs, maxAbsoluteError]`**
— a tiny protocol the app uses to communicate HTP projection
results without JNI-level complexity.

## CPU vs DSP execution — same math, different places

The synthetic projection in `QnnProbe.cpp`:

```cpp
constexpr uint8_t kSyntheticInput[]    = {1, 2, 3, 4};
constexpr uint8_t kSyntheticWeights[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 2};
constexpr float   kSyntheticReference[] = {34.0f, 36.0f, 50.0f};
```

- **ARM CPU reference**: scalar integer MatMul on the Oryon cluster,
  rounded to float32. Oracle = `[34, 36, 50]`.
- **Hexagon HTP**: same operands, but the stub marshals them to the
  DSP, the DSP schedules a quantized MatMul on HVX/HMX, writes the
  result into a shared-memory output buffer, and returns.
- **Numerical compare**: `max_abs_error` against the oracle, versus
  `0.0005` tolerance.

Both paths SHOULD land on the same `[34, 36, 50]` because the
operands and scales are chosen to avoid any ambiguous rounding. Stage
7B's observed result: the UI reports *fixed synthetic projection
passed (2.16 ms).* The projection's `passed` boolean requires
`status == 0` and `maxAbsoluteError <= tolerance`. ✅

## QNN graphs and tensor dtypes

The stub builds a tiny graph:

```
Input [1,4] u8 scale_in offset_in ──▶ MatMul ──▶ Output [1,3] u8 scale_out
Weights [4,3] u8 scale_w offset_w  ──┘
```

Each `Qnn_Tensor_t` records its dtype (`QNN_DATATYPE_UFIXED_POINT_8`
here), rank, dimensions, memory type (`QNN_TENSORMEMTYPE_RAW`, i.e.
flat buffer) and quantization encoding
(`QNN_QUANTIZATION_ENCODING_SCALE_OFFSET`). Finalizing the graph
compiles it; executing it moves the input into DSP memory, runs on
HVX/HMX, and copies the output back.

## Numerical precision and tolerance

Why a tolerance instead of exact equality?

- The operands are quantized u8. The compiler may insert a
  re-quantize node that rounds slightly differently on HTP vs. the
  integer-math oracle on CPU.
- FP conversions at the input/output boundaries use the declared
  scale + offset, which is exact for powers of two but incurs
  rounding otherwise.
- HTP may re-order partial sums for parallelism; integer arithmetic
  is still exact for u8 operands with dot-products up to 2^24, but a
  trailing dequantize to f32 picks up ULP-level error.

For this specific operand set, the output should be *exactly*
`[34, 36, 50]` on both paths; the `0.0005` tolerance is the app's
contract for any u8 projection, not a specific expectation for the
synthetic one.

## Memory transfers

ARM heap → shared ION buffer → CDSP PD memory → HMX tile:

- The app allocates the input buffer in the ARM heap.
- `Qnn_Tensor_t.clientBuf.data` points at that buffer.
- When `QnnGraph_execute` runs, the stub copies the host buffer into
  a FastRPC-visible region (shared via ION / dma_heap) and the DSP
  reads from there.
- DSP side: data is streamed into VTCM scratchpad, HMX tiles consume
  128-byte rows, the result goes back to the shared region.
- Output tensor's `clientBuf.data` is where the stub reads the result
  on the ARM side.

The "2.16 ms" elapsed-ms is wall time around the whole
`QnnGraph_execute` call (ARM-side), which includes the one-way copy
down, the DSP kernel, and the copy back.

## What was DECIDED and why

| Decision | Why |
|---|---|
| Reuse the existing debuggable APK, don't build a fresh JNI harness | The brief forbids unnecessary rework; the app already has a working `QnnProbe` module. |
| Install the **release** APK after the debug APK crashed | Debug APK needs a Metro dev server running to serve the JS bundle. Release has the JS pre-bundled. SELinux context (`untrusted_app`) is the same for both; FastRPC access does not depend on `android:debuggable`. |
| Use the UI as the output channel for `elapsedMs` / status | The `TensorFoldSessionProvider` ContentProvider is signature-protected (`dev.tensorfold.mobile.permission.LOCAL_INFERENCE`). Dumping the live view via `uiautomator dump` is the lowest-overhead readback. |
| Not chase the Live-checkpoint and Full-projection gates | Both require pre-staged payload files in `/data/data/.../files/tensorfold/qnn/*.bin` that aren't produced by a fresh install. Staging them is a separate Stage 7C task; the brief says "one verified QNN operator." |
| Not disable SELinux, not re-sign system libraries | The brief forbids this explicitly, and it would invalidate the whole "stock retail" claim. |

## Hands-on exercise

Prerequisites: a OnePlus 13 (CPH2655) or any Snapdragon 8 Elite /
Hexagon v79 phone, developer options on, USB debugging authorized.

```sh
ADB=/path/to/adb
APK=app/mobile/android/app/build/outputs/apk/release/app-release.apk

$ADB install -r -t $APK
$ADB shell am start -n dev.tensorfold.mobile/.MainActivity
sleep 10   # let React Native + QnnProbe complete
$ADB shell uiautomator dump /sdcard/stage7b-ui.xml
$ADB shell cat /sdcard/stage7b-ui.xml | grep -oE 'text="QNN HTP[^"]*"'
```

Expected output, verbatim prefix:

```
text="QNN HTP v79 fixed synthetic projection passed (<N> ms). ..."
```

For logcat evidence of the FastRPC + skel load cycle:

```sh
$ADB logcat -d | grep -E 'libQnnHtpV79Skel|user PD on domain 3|fastrpc_shell'
```

## Comprehension questions

**Q1.** You measure `elapsedMs = 2.16` on the ARM side across a
`QnnGraph_execute()` call. What **three** operations is that
measurement bundling, and what ADDITIONAL instrument would let you
separately report each of their durations? (Hint: think about ARM
→ DSP marshal, DSP compute, DSP → ARM unmarshal.)

**Q2.** The Android app process and a shell process on the same
device both run as unprivileged UIDs, both have `uses_native_library
libcdsprpc.so` **unavailable** to shell, and both observe
`/dev/fastrpc-cdsp` with permission bits `0664 system system`. Yet
the app can reach the DSP and the shell cannot. Name at least two
distinct mechanisms by which the system blocks or permits each one,
and explain how the app's `AndroidManifest.xml` entries interact
with the vendor policy.

---

*Questions deliberately left unanswered. Write out your own; a
reviewer may compare.*
