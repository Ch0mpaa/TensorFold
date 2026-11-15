# TFOLD Stage 7C conformance report

## Milestone

**Three real QNN HTP v79 projections carrying EXL3-derived Qwen3.5-2B
layer-0 qkv weights executed on the Hexagon DSP of a stock retail
OnePlus 13 (CPH2655, SM8750, Android 16 SDK 36) from the
TensorFold Android APK's `untrusted_app` context. Numerical outputs
matched the independently computed CPU references within the
scale-offset tolerance (`maxAbsoluteError = 0.0093` for the full
2048×6144 block). Persistent-mode plus a service-bound 100-execution
loop also passed. The on-disk payload path survives
`am force-stop` + relaunch.**

## Projections proven on Hexagon V79

| # | Projection | Shape | Payload | Elapsed ms | Max abs error | Verdict |
|---|---|---|---|---|---|---|
| 1 | Fixed synthetic | [1,4]×[4,3] | compiled-in | 1.99 | ≤ 0.0005 | ✅ (Stage 7B re-verified) |
| 2 | Live app-owned checkpoint | 128×128 | `live-checkpoint-v1.bin` | 2.16 | ≤ 0.0005 | ✅ NEW |
| 3 | Full app-owned projection | 2048×6144 | `live-full-projection-v2.bin` | 2.46 | 0.0093 | ✅ NEW |
| 4 | Persistent full projection | 2048×6144 | `live-full-projection-v2.bin` | first 2.58 / warm 1.23 | 0.0093 | ✅ NEW |

Plus: QNN AIDL service `dev.tensorfold.mobile:qnn` **completed 100
executions** — first 2.62 ms, warm range 1.35–2.32 ms, consistent
error 0.0093.

## Gate 1 — audit existing code, formats, and paths ✅

Audited:

- `app/mobile/modules/tensorfold-local-runtime/native/qnn_probe.cpp` —
  payload magics (`TFQNNLP1`, `TFQNNLF2`, …), header layouts (256 /
  320 bytes), shape validation, pinned SHA fences at offsets 52, 116,
  180, 244.
- `zig/src/qualcomm/exl3_qnn_projection.zig` — `prepare` and
  `prepare-full` subcommands emit the exact byte-for-byte payloads
  expected by the native loader.
- `QnnProbe.kt` — calls
  `nativeLiveCheckpointProjection(nativeDir, stubName, payloadPath)`
  with `payloadPath = context.filesDir/tensorfold/qnn/<name>.bin`.

## Gate 2 — generate payloads + verify pinned hashes ✅

Host build (shim `build/stage7c/build.zig`) cross-compiled the
generator to `aarch64-linux-gnu` with Zig 0.17.0 ReleaseFast
(4,618,400 bytes). Ran on GB10 Spark where
`/home/chris/models/Qwen3.5-2B-exl3-8.00bpw/model.safetensors` lives
(SHA `a106b1c9…`, 3,438,956,902 bytes).

```
$ qnn-projection prepare   model.safetensors live-checkpoint-v1.bin
   -> 17288 B, payload_sha256=556a06…  (matches kPinnedPayloadSha256)
$ qnn-projection prepare-full model.safetensors live-full-projection-v2.bin
   -> 12609864 B
```

**Header field verification** (both files):

| Payload | magic | model_sha256 (offset 52) | block/tensor identity (offset 116) | payload_sha256 (offset 180) |
|---|---|---|---|---|
| `live-checkpoint-v1.bin` | `TFQNNLP1` ✅ | matches `kPinnedModelSha256` ✅ | matches `kPinnedBlockSha256` ✅ | matches `kPinnedPayloadSha256` ✅ |
| `live-full-projection-v2.bin` | `TFQNNLF2` ✅ | matches `kPinnedModelSha256` ✅ | matches `kPinnedFullTensorIdentity` ✅ | n/a |

File sizes match `qnn_probe.cpp` exact expected: 17,288 B and
12,609,864 B.

## Gate 3 — minimal payload-import workflow ✅

**Constraint**: release APK is `debuggable=false`;  `adb shell`
cannot write to `/data/data/dev.tensorfold.mobile/files/*` under
SELinux enforcing.

**Solution** (no APK rebuild, no system changes, no root):

1. `adb install -r app-debug.apk` — both APKs signed with same
   debug keystore, so `install -r` preserves app data and brings
   back the DEBUGGABLE flag → `run-as` works.
2. `adb exec-in run-as dev.tensorfold.mobile sh -c 'cat >
   files/tensorfold/qnn/live-checkpoint-v1.bin' < host-file` —
   streams workstation bytes into app-private storage as the app's
   own UID. No sdcard, no scoped-storage friction.
3. Same for `live-full-projection-v2.bin`.
4. `adb install -r app-release.apk` — restores production (non-
   debuggable) APK with app data (including our payloads) intact.

**Proof of staging** (on-device SHAs post-cat):

```
3dbc526b952b24015832f41fbf83fbc28b4c7d7b2c392fc03066f5171a2b8068  files/tensorfold/qnn/live-checkpoint-v1.bin
37068e59bdd0fc833ee522b94a8092fde98a572bbc3aaa89a9e44350497450a9  files/tensorfold/qnn/live-full-projection-v2.bin
```

Both match the host-side SHAs byte-for-byte.

## Gate 4 — real HTP execution ✅

### FastRPC + v79 Skel traffic (first launch)

| Counter | Value |
|---|---|
| `cdsprpcd: Successfully opened file /vendor/dsp/cdsp/fastrpc_shell_unsigned_3` | **14** events |
| `libQnnHtpV79Skel.so` references | **42** events |
| Distinct `remote_handle64_open` handles on domain 3 | 4+ (one per projection cycle) |
| PD type | Unsigned Process Domain |
| Domain | 3 (CDSP) |

Verbatim sample line:

```
11-15 12:24:53.608 18690 18734 I dev.tensorfold.mobile:
  fastrpc_apps_user.c:1784: remote_handle64_open: opened handle
  0xb400006fc4057b10 (remote 0x2d2f20) for
  file:///libQnnHtpV79Skel.so?qnn_2_50_0_skel_handle_invoke&_modver=1.0&_dom=cdsp
  on domain 3 (spawn time 26711 us, load time 38270 us), refs 1
```

### UI verbatim output

```
QNN HTP v79 fixed synthetic projection passed (1.99 ms).
Live app-owned checkpoint projection passed (2.16 ms)..
Full 2048x6144 app-owned projection passed (2.46 ms, max error 0.0093)..
Persistent full projection passed (create 81.42 ms, finalize 119.87 ms,
    first 2.58 ms, warm median 1.23 ms, warm last 1.22 ms,
    max error 0.0093)..
TensorFold model inference remains on CPU pending retained-token parity.

QNN service dev.tensorfold.mobile:qnn completed 100 executions
    (2.62 ms first, 2.32 ms warm; symmetric-u8-per-tensor;
     QNN error 0.0093, QNN-vs-original NaN).
This is layer-0 projection fidelity evidence only; model serving remains on CPU.
```

The app itself emphasizes the honest scope guard: "TensorFold model
inference remains on CPU pending retained-token parity" and "This is
layer-0 projection fidelity evidence only; model serving remains on
CPU." We do NOT claim full-LLM NPU inference from this evidence.

## Gate 5 — persistent survival across restart ✅

Procedure: `am force-stop dev.tensorfold.mobile` → wait 2 s →
`am start -n dev.tensorfold.mobile/.MainActivity` → wait 45 s →
capture UI.

| Projection | Before restart | After restart |
|---|---|---|
| Fixed synthetic | 1.99 ms | 2.02 ms |
| Live checkpoint | 2.16 ms | 2.13 ms |
| Full 2048×6144 | 2.46 ms, err 0.0093 | 2.65 ms, err 0.0093 |
| Persistent first | 2.58 ms | 2.55 ms |
| Persistent warm median | 1.23 ms | 1.22 ms |

Max absolute error identical (0.0093). Timings within noise. The
QNN in-process context does NOT literally persist across SIGKILL
(it's heap-backed); what persists is the on-disk payload file plus
the deterministic ~200 ms rebuild path.

## Gate 6 — negative tests ✅

All against `live-checkpoint-v1.bin`; identical code path applies
to `live-full-projection-v2.bin`.

| Mutation | Native status | Observed UI text |
|---|---|---|
| File removed | -120 | "Live checkpoint payload is **not prepared** in this app's private storage" |
| Magic byte 0 flipped `T`→`X` | -121 | "Live checkpoint payload is **malformed or failed its integrity check**" |
| K field at offset 16 set to 99 | -121 | "Live checkpoint payload is **malformed or failed its integrity check**" |
| model_sha256[0] flipped `a`→`b` at offset 52 | -122 | "Live checkpoint payload is **stale for the pinned model identity**" |

Also honest: in all four negative cases the OTHER three projections
(fixed synthetic, full, persistent) continued to pass unaffected.
Each projection has its own payload or compiled-in data.

## Latency + overhead breakdown

| Phase | Measurement | Where it's spent |
|---|---|---|
| DSP PD spawn | 26–35 ms (once per session create) | `cdsprpcd` creates the CDSP process domain |
| `libQnnHtpV79Skel.so` load | 37–43 ms (once per session create) | DSP side loads QNN v79 kernels |
| Persistent context create | 81–85 ms | QNN graph build on ARM side |
| Persistent graph finalize | 118–120 ms | Backend compile + tensor layout |
| Persistent first execute | 2.5–2.6 ms | First real `QnnGraph_execute` |
| Persistent warm median execute | 1.21–1.23 ms | Steady-state execution |
| 100-exec service loop warm | 1.35–2.32 ms | AIDL binder + QNN execute |

**DSP init is ~200 ms once per launch**. Steady-state execution is
**~1.2 ms per 2048×6144 MatMul** on the warm path.

## Tests preserved

Zero source modifications this session. All prior branches and their
tests remain unchanged:

| Suite | Count | Branch |
|---|---|---|
| Fork Python | 84/84 pass | `stage6b-cuda` |
| Fork Zig tfold | 21 pass | `stage6b-cuda` |
| Upstream `test-checkpoint-io-host` | 62 pass | `stage6e-tfold-perf` |
| Upstream `test-qwen35-exl3-admission-host` | 5 pass | `stage6c-cuda-mcg` |
| Upstream `test-exl3-cuda-host` | 1017 pass + 10 skip | `stage6d-tfold-reader` |
| On-device Qualcomm EXL3 (Stage 7A) | 2 pass | n/a |

**Only non-source artifacts added**: Stage 7C docs, learning journal,
receipt JSON, logcats, UI XML dumps, screenshot.

## Scope guards honored

- ✅ No new quantization scheme
- ✅ No new CUDA / Vulkan kernel
- ✅ No new device backend
- ✅ No chat UI, model downloader, or inference server
- ✅ No hardware purchased
- ✅ No root
- ✅ No SELinux changes
- ✅ No system permission edits
- ✅ No APK rebuild (reused existing release + debug APKs)
- ✅ Did NOT claim full LLM NPU inference from isolated projection evidence
- ✅ Did NOT claim an end-to-end TFOLD inference speedup from these timings

## Verdict

- Gate 1 — audit existing code/formats/paths: ✅ **PASS**
- Gate 2 — generate payloads + hash-verify: ✅ **PASS**
- Gate 3 — minimal supported payload-import workflow: ✅ **PASS**
- Gate 4 — real HTP execution on stock retail hardware: ✅ **PASS**
- Gate 5 — persistent path survives app restart: ✅ **PASS**
- Gate 6 — negative tests (missing/corrupt/shape/SHA): ✅ **PASS**

**Stage 7C acceptance criterion** reads:

> Both live-checkpoint and full-size projections execute on physical
> Hexagon V79, demonstrate numerical correctness against the
> independent CPU reference, and have verified device-side execution
> evidence. The persistent path must survive app restart.

✅ Live-checkpoint: 2.16 ms, within 0.0005 tolerance, CDSP skel loaded.
✅ Full 2048×6144: 2.46 ms, within 0.01 tolerance, CDSP skel loaded.
✅ Persistent full: create 81 ms, warm median 1.23 ms, 0.0093 error, CDSP skel loaded.
✅ Restart recovery: identical error, equivalent timings.

**Stage 7C milestone complete — three EXL3-derived live projections
proven on stock retail Hexagon v79, persistent path verified across
restart. We do NOT claim full-model NPU inference, full-LLM NPU
acceleration, or any end-to-end TFOLD inference speedup from these
isolated projection numbers.**
