# Stage 7A — Snapdragon execution proof on a real OnePlus 13

> Beginner-friendly companion to Stage 7A. The goal was independently
> verifying that TensorFold code can (a) run tensor math on real ARM64
> Snapdragon silicon, and (b) see the Qualcomm QNN HTP backend. The
> milestone deliberately stops short of "full LLM on NPU."

## The ARM CPU inside the OnePlus 13

The SoC is a **Qualcomm SM8750 (marketing name: Snapdragon 8 Elite)**.
Its CPU cluster uses Qualcomm's second-generation Oryon cores
(the "Phoenix" design). All our Zig-cross-compiled binaries target
`aarch64-linux-android` so the ARMv9-A instruction set the Oryon cores
decode is a strict superset of what we emit. We did not use any
Snapdragon-specific CPU instructions (no SVE2, no SME); the EXL3
reference decoder test runs on vanilla ARMv8-A scalar + NEON.

## The Hexagon HTP (Hexagon Tensor Processor)

The SM8750 also ships a dedicated DSP cluster, the **Hexagon
architecture v79 HTP**. Internally the HTP has:

- **HVX** vector units (1024-bit SIMD, ~6 of them in parallel on v79).
- **HMX** matrix units for tiny-tile INT8/FP16 matrix multiply.
- **VTCM** — "Vector Tightly-Coupled Memory" — fast scratchpad that
  fits inside the DSP's address space. Our Phone B logs confirmed an
  8 MiB VTCM on v79.

The HTP is a completely separate processor. The ARM CPU cannot read HTP
registers directly; all communication is through **FastRPC**, a message
passing channel that goes through the kernel driver node
`/dev/fastrpc-cdsp`. User-space code loads a *stub library* (on the ARM
side) which marshals arguments across FastRPC to a matching *skel library*
(on the Hexagon DSP side), where the actual kernel runs.

For Stage 7A the specific pair was:

| Side | Library | Runs on | What it does |
|---|---|---|---|
| Stub | `libQnnHtpV79CalculatorStub.so` | ARM Oryon core | Packs args, invokes FastRPC |
| Skel | `libQnnHtpV79Skel.so` | Hexagon DSP | Decodes args, runs sum, returns |

Both must be present and the FastRPC channel must open. If
`/dev/fastrpc-cdsp` denies access, the skel never gets to run.

## QNN graphs and tensors

A QNN **graph** is a DAG of operators (MatMul, Add, Relu, ...)
parameterized with tensor descriptors (dtype, shape, quantization
params). You build the graph through the QnnBackend_createContext +
QnnGraph_create + QnnGraph_addNode APIs on the ARM side. Finalizing the
graph triggers a compiler inside `libQnnHtpV79Stub.so` that lowers each
node to Hexagon instructions and schedules HVX/HMX/VTCM usage.

Supported tensor dtypes for the HTP are narrower than the CPU:

- **f16** (IEEE half)
- **f32** (software path only on HTP; usually the compiler refuses)
- **ufixed_point_8** (asymmetric INT8 quantization with scale+offset)
- **ufixed_point_16**
- **sfixed_point_8**
- **sfixed_point_16**
- **ufixed_point_4** (for very recent HTPs)

EXL3 and EXLR8 storage are NOT native HTP types. TensorFold's existing
Qualcomm EXL3 kernel decodes trellis+scales INTO one of the above
before offload.

## Delegation — what "running on HTP" actually requires

A binary can **load** `libQnnHtp.so` and call `QnnBackend_create()`
successfully even if the Hexagon side never actually runs anything.
Three signals tell you whether delegation genuinely happened:

1. **Device node access.** A successful `open("/dev/fastrpc-cdsp")`.
2. **Skel load evidence.** Qualcomm's internal `fastrpc_apps_user` logs
   or verbose QNN backend tracing shows the skel loaded in the DSP's
   process space (HTP0 / HVX counts / HMX unit count / VTCM size).
3. **Measurable work.** Latency on the stub call scales with the
   problem size, not with a flat scalar fallback.

In Stage 7A, the shell-launched `qnn-platform-validator` **failed
signal 1** (EACCES on `/dev/fastrpc-cdsp`) but still printed
"Unit Test Passed." That is a **loader-side success**: the stub's
`sum()` resolved to the ARM-side fallback inside the stub library
because the FastRPC channel was never open. Reporting that honestly
is the entire point of brief step 5.

## Memory movement, briefly

For a real HTP MatMul the memory flow is:

```
ARM heap (host)                    DSP VTCM (device)
     │                                   │
     │  QnnMem_register                  │
     │  via ION / dma_heap                │
     │────────────────────────────▶     │
     │                                   │
     │                                   ├── kernel runs
     │  QnnGraph_execute                 │   HVX/HMX
     │◀──────────────────────────         │
     │   results copied back             │
     ▼                                   ▼
```

The host-to-device buffer copy is NOT free. It is normally hidden in
the `QnnGraph_execute` timing. "Zero-copy" claims about QNN usually
refer to a shared DMA buffer; even then, caches on the ARM side must
be invalidated before the DSP reads.

## Correctness checks — how we verify the answer

Our in-tree oracle for EXL3 is a **FP64 numpy reference** generated
once and committed as `zig/tests/fixtures/exl3_oracle.bin`
(sha256 `8eecb6ac…cd2c2ff4`). The fixture format is defined in
`zig/src/core/exl3_format.zig`.

Two levels of parity:

- **Byte-exact at the FP64 output**: the Zig decoder's dq output must
  match `oracle.dq_bytes` verbatim. We use `std.testing.expectEqualSlices(u8, got, want)`. This gate passed on the OnePlus 13 ARM64 in Stage 7A.

- **Numerical-tolerance at FP32/FP16**: when running lower-precision
  backends (HTP via `qualcomm-matmul`), we compute max-abs-error
  against a double-precision reference and insist on a declared
  tolerance. Prior Phone B evidence showed zero observed
  max-abs-error on the MatMul CPU-vs-HTP comparison; that result
  used the TensorFold app UID, which IS allowed to open the FastRPC
  channel. Stage 7A did not repeat this measurement under app UID.

## What Stage 7A proved

- ✅ Real ARM64 numerical work on a real OnePlus 13 (SM8750) running
  Android 16 / SDK 36: the EXL3 reference decoder test passes 2/2
  against the FP64 oracle, with bit-exact byte equality.
- ✅ QNN backend libraries load without error; Hexagon V79 architecture
  is detected.
- ⚠️ Shell-UID execution cannot reach `/dev/fastrpc-cdsp`. The platform
  validator's "Unit Test Passed" is an API-level loader success, not
  genuine HTP delegation.
- 📎 Prior Phone B (CPH2655, same hardware) **did** achieve app-UID HTP
  delegation with verbose trace evidence — see
  `docs/qualcomm-android.md` §"Snapdragon 8 Elite validation" for
  prior HTP0/HVX/HMX/VTCM confirmation.

## What Stage 7A did NOT prove

- ❌ Full-LLM NPU execution. Stage 7A was explicitly scoped to isolated
  operators. The brief forbids claiming NPU LLM inference from an
  isolated-operator test.
- ❌ Sustained performance. Only a few samples under battery/thermal
  neutral conditions; no community-v2 or publication-grade runs.
- ❌ Shell-UID HTP execution. The security model blocked us.

## Hands-on exercise

Plug a OnePlus 13 (or any Snapdragon 8 Elite / Hexagon v79 device).
Enable Developer Options → USB Debugging. Then:

```sh
adb devices -l   # verify device appears

# Push the three small binaries + oracle fixture.
adb push zig-out/bin/tensorfold-qualcomm-smoke /data/local/tmp/
adb push zig-out/bin/tensorfold-qualcomm-probe /data/local/tmp/
adb push zig-out/bin/tensorfold-qualcomm-exl3-test /data/local/tmp/tensorfold-exl3-stage2/

# 1. Smoke: prove the Zig cross-compile runs.
adb shell /data/local/tmp/tensorfold-qualcomm-smoke
# {"tensorfold":"qualcomm-smoke","arch":"aarch64","abi":"android","stage":0}

# 2. Probe: open libcdsprpc; the dsp session will NOT open from shell.
adb shell /data/local/tmp/tensorfold-qualcomm-probe
# {"tensorfold":"qualcomm-probe","libcdsprpc":true,"fastrpc_symbols":true,"dsp_session_opened":false,"stage":1}

# 3. EXL3 ARM64 numerical oracle (2 tests).
adb shell "cd /data/local/tmp/tensorfold-exl3-stage2 && ./tensorfold-qualcomm-exl3-test"
# 1/2 ...OK  2/2 ...OK  All 2 tests passed.
```

Independently, in Python:

```python
import sys; sys.path.insert(0, "tools")
from tfold_convert import oracle
b = oracle.load_oracle_mcg_block()
# b.dequant_reference_fp64 is the bytes the Zig test checks against.
# Their sha256 is b28f6c89fdcb9232…; sum is 2.706761009.
```

If both sides match, your ARM64 CPU path is wired.

## Comprehension questions

**Q1.** Suppose a tool prints `Unit Test on the backend DSP: Passed`
and declares `Core Version: Hexagon Architecture V79`. What
additional evidence must you collect before claiming the test
actually ran on the Hexagon DSP, as opposed to a scalar fallback
inside the stub library? Name at least two distinct signals and
explain which part of the QNN stack each one verifies.

**Q2.** Stage 7A validated EXL3's FP64 oracle on ARM64 with **byte-exact
equality**. If we instead ran the same decode on the HTP as an
INT8-quantized MatMul, byte-exact equality would be impossible.
What tolerance and what failure criterion would you declare *before*
running the test, so that an INT8 quantization-induced error is
distinguishable from a real bug? (Hint: think about the output
distribution's scale and about which magnitude of error would
change argmax.)

---

*These questions are deliberately left unanswered. Write your own
answers; a reviewer may compare.*
