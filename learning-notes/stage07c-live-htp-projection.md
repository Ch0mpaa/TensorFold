# Stage 7C learning journal — Live HTP projection (OnePlus 13, Hexagon V79)

> Beginner companion for Stage 7C. Read Stages 7A/7B first so you know
> what an app-private FastRPC session looks like. Here the question
> stops being "can we reach the DSP?" and becomes "can we project a
> real checkpoint tensor on the DSP, in milliseconds, repeatedly, and
> prove the answer is right?"

## 1. What is a "real" model projection?

A model projection is a matrix multiply that turns an activation
vector `x` of size K into an output vector `y` of size M, using the
model's own weight matrix `W` of shape K×M. For today's Qwen3.5-2B
block we use:

- K = 2048 (one hidden state)
- M = 6144 (the 3×2048 concatenated q, k, v projection target of
  layer 0's `linear_attn.in_proj_qkv`)

Doing `y = x @ W` for these shapes means 2048×6144 ≈ 12.6 million
multiply-add operations per token per layer. Transformer models call
this hundreds of times per generated token. If the CPU does it, the
phone gets hot and slow. If the Hexagon HTP does it, the CPU stays
cool and the NPU does the math.

Stage 7C isn't full inference — it's one layer-0 projection. But the
plumbing to carry EXL3-dequantized real weights from the
model.safetensors file all the way through FastRPC into DSP memory
and back is now proven.

## 2. How the EXL3-derived weights reach QNN

EXL3 ("ExLlamaV3 3-bit") is a mixed-precision quantization scheme used
by the oracle. For Stage 7C we did NOT ship EXL3 to the DSP — the DSP
backend doesn't speak EXL3. Instead we:

1. Read `model.safetensors` of the Qwen3.5-2B 8.00bpw EXL3 model
   (SHA `a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f`).
2. Dequantized the layer-0 qkv projection back to high-precision
   floats using the CHECKED-IN Zig EXL3 decoder (same code we
   validated against the numpy oracle in Stage 6d).
3. Re-quantized those floats into QNN-friendly symmetric-u8 with a
   per-tensor scale and a zero-point offset of -128. (QNN HTP's
   `QNN_OP_MAT_MUL` wants `QNN_DATATYPE_UFIXED_POINT_8` under
   `QNN_QUANTIZATION_ENCODING_SCALE_OFFSET`.)
4. Also kept the independently calculated fp32 reference vector so
   the device can compare output to a CPU-side oracle.
5. Packed everything (K bytes input + K·M bytes weights + M f32
   reference + 8-byte FNV64 integrity trailer) behind a versioned
   256-byte header and wrote it to disk as `live-checkpoint-v1.bin`.

Same workflow for the full 2048×6144 tensor — that file is
`live-full-projection-v2.bin`, ~12 MB.

The generator is `tensorfold-qualcomm-exl3-qnn-projection`. Its
`prepare` and `prepare-full` subcommands are deterministic: same
model → same output bytes → same SHA. The headers embed pinned SHAs
of the model, the source block, and the content itself; on the
phone, `qnn_probe.cpp` refuses any payload whose embedded hashes
don't match the ones baked into the C++. This is how Stage 7C
guarantees "the DSP really saw the project-layer-0-qkv weights".

## 3. Why quantization parameters matter

Each QNN tensor carries a `scale` and `offset`. Real value =
`scale × (u8 - 128)`. Our scale is picked so that the maximum
absolute weight in a tensor ends up near the top of the u8 range —
spending bits where values actually live. If we picked a scale 10×
too large, most weights would quantize to the same u8 bucket and the
dot product would lose signal. If 10× too small, they'd saturate.

The tolerance used by the app (`fullBlockTolerance = ~0.01`) is set
to `output_scale × 1.5`. That's one output quantization step plus 50%
slack for rounding in the reduction. The observed max absolute
error on the 2048×6144 block was **0.0093**, below tolerance.

## 4. How app-private storage works on Android

Every app has a private dir at `/data/data/<pkg>/files` (API-level
alias `context.filesDir`). It's mode 0700 and owned by the app's UID.
SELinux labels it `app_data_file:<pkg>_data_file:…`. Nothing except
that app can read or write there. Not even `adb shell` as the `shell`
user — it gets `EACCES`.

So the honest problem Stage 7C had to solve was: how do we get a
12 MB file from the workstation into a sandboxed release app's data
dir on a stock, non-rooted, non-debuggable phone?

Three options we rejected:

- **Root / Magisk** — explicitly banned.
- **A ContentProvider** — the app has a ContentProvider but it's
  signature-permission-guarded, so `adb shell content insert` fails
  from UID 2000 (shell).
- **MediaStore / SAF upload** — the app has no "Import weights…"
  button, we aren't allowed to add a new APK.

What we used instead is a **rebuild-free** workflow that exploits the
fact that **both the debug and the release APK are signed by the
same debug keystore**, so Android will install one over the other
with `install -r` and preserve the app's data dir:

1. `adb install -r app-debug.apk` (replaces release, preserves data,
   gives us a DEBUGGABLE flag → `run-as` now works).
2. `adb exec-in run-as dev.tensorfold.mobile sh -c 'cat >
   files/tensorfold/qnn/live-checkpoint-v1.bin' < payload` — the
   stream pipes workstation bytes straight into app-private storage
   via the app's own UID. No sdcard round trip, no permission
   broker.
3. `adb install -r app-release.apk` — reinstall the "production"
   APK; data dir untouched.
4. Launch and let `QnnProbe.kt` find the staged payload.

The release APK is what actually runs and generates all our
evidence; the debug APK exists only as a short-lived data-staging
proxy.

## 5. How HTP results are verified

The app does **four** independent verifications per launch:

| Projection | What it proves |
|---|---|
| Fixed synthetic (compiled-in) | The FastRPC path, graph build, execute all work with known-oracle 3-element output. Stage 7B gate. |
| Live app-owned checkpoint | The 128×128 block from Qwen layer 0 dequantizes and projects within 0.0005. |
| Full 2048×6144 app-owned | The entire layer-0 qkv projection matches CPU reference within ~0.01. |
| Persistent full | Same as above, but the QNN context is cached via a background service and survives app restarts. |

Every projection compares HTP output to an independently computed
CPU-side fp32 reference (stored inside the payload). Each produces a
`maxAbsoluteError` number the app prints. Pass = error ≤ tolerance.
The verification is not a hash of the output — it's the actual
numerical comparison element by element.

The outer "QNN service … completed 100 executions" sentence proves
the bound AIDL service can re-execute the same graph 100 times with
consistent timing and error. Repeated execution amortizes the graph
build.

## 6. What changed this session (and why)

| Change | Why |
|---|---|
| Added `build/stage7c/build.zig` and `build.zig.zon` as a host-only shim | The main `build.zig` only registers the `qualcomm-exl3-qnn-projection` exe inside `qualcommTargets`, which runs only for Android targets. We needed a host build to emit payloads. The shim mirrors just the two modules the exe needs. |
| Cross-compiled the generator to aarch64-linux-gnu, not Windows x86_64 | The source `@import("std").DynLib` doesn't support Windows in Zig 0.17. We cross-compiled to Linux ARM64 and ran it on the GB10 Spark host where model.safetensors is already present (saves ~3.4 GB over sshfs). |
| Did NOT add an "Import weights" UI to the app | Would have required rebuilding the APK. The install-debug / exec-in-run-as / reinstall-release trick is strictly less code and keeps the bytes we ship identical to Stage 7B. |

## 7. Alternatives considered (and why rejected)

| Option | Reason rejected |
|---|---|
| Add an SAF `Intent.ACTION_OPEN_DOCUMENT` import button to the app and rebuild | New JS + Kotlin code → larger diff → risks a React Native build regression. Brief forbids "building another Android application". |
| Use a `BroadcastReceiver` with an `ACTION_IMPORT_PAYLOAD` custom intent | Same rebuild cost; adds persistent attack surface the brief doesn't need. |
| Push to `/sdcard/Download/` + have an in-app importer scan it | Scoped-storage friction (MediaStore vs direct), and no in-app importer exists. |
| `run-as` the release APK directly | The release APK has `android:debuggable=false`. `run-as` requires DEBUGGABLE in manifest. Rejected on first try. |
| Push via a ContentProvider with a dedicated write permission | TensorFoldSessionProvider has `readPermission` only, and bumping it to `writePermission` is an APK change. |

## 8. Actual measurements (Device A, OnePlus 13 CPH2655, Hexagon V79)

### First-launch run

| Projection | Elapsed (ms) | Max abs error |
|---|---|---|
| Fixed synthetic | 1.99 | ≤ 0.0005 |
| Live checkpoint | 2.16 | ≤ 0.0005 |
| Full 2048×6144 | 2.46 | 0.0093 |
| Persistent full — first exec | 2.58 | 0.0093 |
| Persistent full — warm median | 1.23 | — |
| Persistent full — warm last | 1.22 | — |

### Service-mode 100-execution loop

- First execution: 2.62 ms
- Warm median: 2.32 ms (observed 1.35–2.32 across runs)
- Reports `symmetric-u8-per-tensor`, `QNN error 0.0093`

### Lifecycle overhead (persistent context)

| Phase | Measurement |
|---|---|
| Context create | 81–85 ms (per launch) |
| Graph finalize | 118–120 ms (per launch) |
| First execute | 2.55–2.66 ms |
| Warm median execute | 1.21–1.23 ms |
| **DSP PD spawn (observed per cycle)** | 26–35 ms |
| **`libQnnHtpV79Skel.so` load (observed per cycle)** | 37–43 ms |

Four distinct `remote_handle64_open` calls were logged per launch
(one per projection cycle), each with a fresh 20-hex handle and the
exact string `on domain 3 (spawn time … us, load time … us)`.

### After `am force-stop` + relaunch

| Projection | Elapsed (ms) | Max abs error |
|---|---|---|
| Live checkpoint | 2.13 | ≤ 0.0005 |
| Full 2048×6144 | 2.65 | 0.0093 |
| Persistent full — first exec | 2.55 | 0.0093 |
| Persistent full — warm median | 1.22 | — |

Numbers are consistent across launches. The persistent context does
NOT survive process death literally (it's an in-memory QNN context);
what survives is the payload file and the ability to rebuild the
context deterministically from it in 200 ms.

## 9. Hands-on exercise (do this yourself)

**Goal**: break the integrity check and watch the app correctly
refuse your payload, then restore it and watch it pass again.

Prerequisites: both APKs installed previously, ADB set up, Device A
plugged in.

```bash
ADB=/c/Users/chomp/android-tools/platform-tools/adb.exe
PKG=dev.tensorfold.mobile
DEBUG=app-debug.apk
RELEASE=app-release.apk

# Reinstall debug to get run-as back
$ADB install -r $DEBUG

# Back up the good payload inside the app's private dir
$ADB shell "run-as $PKG cp files/tensorfold/qnn/live-checkpoint-v1.bin files/tensorfold/qnn/live-checkpoint-v1.bin.good"

# Pick ONE mutation to try:
# (a) missing file — triggers status -120
$ADB shell "run-as $PKG rm files/tensorfold/qnn/live-checkpoint-v1.bin"

# (b) bit flip in the magic — triggers status -121
#   $ADB shell "run-as $PKG sh -c 'printf X | dd of=files/tensorfold/qnn/live-checkpoint-v1.bin bs=1 count=1 conv=notrunc'"

# (c) bit flip in the embedded model SHA at offset 52 — triggers status -122
#   $ADB shell "run-as $PKG sh -c 'printf b | dd of=files/tensorfold/qnn/live-checkpoint-v1.bin bs=1 seek=52 count=1 conv=notrunc'"

# Switch back to release and launch
$ADB install -r $RELEASE
$ADB shell am force-stop $PKG
$ADB shell am start -n $PKG/.MainActivity
sleep 40
$ADB shell uiautomator dump /sdcard/x.xml && $ADB shell cat /sdcard/x.xml | grep -o 'text="[^"]*checkpoint[^"]*"'

# Restore
$ADB install -r $DEBUG
$ADB shell "run-as $PKG cp files/tensorfold/qnn/live-checkpoint-v1.bin.good files/tensorfold/qnn/live-checkpoint-v1.bin"
$ADB install -r $RELEASE
```

**Expected UI text for each case**:

- (a) `Live checkpoint payload is not prepared in this app's private storage`
- (b) `Live checkpoint payload is malformed or failed its integrity check`
- (c) `Live checkpoint payload is stale for the pinned model identity`

If you see "passed" after mutating, something is wrong with the
integrity check — stop and investigate the exact byte you touched.

## 10. Comprehension questions (left open on purpose)

**Q1**. Why does the four-byte "counts" tuple in the header at offset
244 repeat values that the loader can also derive from K and M in
the same header? In other words — if `qnn_probe.cpp` already knows
`K=128`, `M=128` from offsets 16–20, what's the point of the
`(expected_k, expected_k * expected_m, expected_m * sizeof(float))`
echo at 244–256? Hint: think about what would happen if we quietly
upgraded the payload schema to a different per-section layout but
somebody still had an old `live-checkpoint-v1.bin` on disk.

**Q2**. The "persistent full projection" lists `warm median 1.23 ms`
versus the first-call 2.58 ms. Where do those extra 1.35 ms per
first-call go? Pick ONE specific hypothesis (first-touch mmap?
first-touch DSP cache fill? deferred FastRPC reservation? first-call
JIT pass inside `libQnnHtpV79Skel.so`?) and describe one experiment
you'd run next to confirm or refute it without rebuilding the app.
Hint: both `logcat -v threadtime` and `atrace` are still available
to you on the stock phone without root.
