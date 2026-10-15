# Qualcomm Android port

Status: stage 2 scalar EXL3 correctness. TensorFold is not yet executing EXL3
model inference on HTP or opening a DSP session.

The port targets 64-bit Android on Snapdragon, with Qualcomm FastRPC/HTP as the
first accelerator path. It keeps TensorFold's EXL3/EXLR8 formats, model logic,
and lane interface. The existing llama.cpp Hexagon work is a reference for the
Android-to-cDSP transport and deployment boundary only; GGUF kernels and model
execution are not being substituted for TensorFold.

## Stage 0: cross-compile and launch

From the repository root, using Zig 0.17.0:

```text
zig build qualcomm-smoke -Dtarget=aarch64-linux-android -Doptimize=ReleaseSafe
adb push zig-out/bin/tensorfold-qualcomm-smoke /data/local/tmp/
adb shell chmod 755 /data/local/tmp/tensorfold-qualcomm-smoke
adb shell /data/local/tmp/tensorfold-qualcomm-smoke
```

Expected output:

```json
{"tensorfold":"qualcomm-smoke","arch":"aarch64","abi":"android","stage":0}
```

This target deliberately has no libc, Android NDK, or Qualcomm SDK dependency.
It proves that the pinned Zig toolchain can produce a binary which Android's
loader accepts before vendor libraries are introduced.

## Port gates

1. Add a small FastRPC loader/probe and report the exact vendor library and cDSP
   device availability without starting model inference.
2. Run deterministic EXL3 unpack/dequantize/projection fixtures on Android CPU
   and compare them bit-for-bit or within a declared tolerance to the upstream
   reference outputs.
3. Implement one Qualcomm kernel primitive at a time, beginning with the
   projection path. Record layout, alignment, datatype, execution unit, and
   numerical tolerance for every primitive.
4. Connect those primitives behind `core/lanes/backend.zig`; do not fork the
   EXL3 format or family-level model semantics.
5. Bring up a minimal model and context, then measure correctness, peak memory,
   prefill, decode, sustained thermals, and energy independently.

Every gate must retain a CPU reference and a deterministic tiny-tensor test.
Kernel or boot changes are out of scope until userspace profiling demonstrates
that they are necessary.

The `qualcomm-probe` target implements the non-invasive portion of gate 1. It
loads `/vendor/lib64/libcdsprpc.so` and checks the required 64-bit remote-handle
and RPC-memory entry points. It does not create a DSP session or load a skeleton.
Because Zig does not bundle Android's Bionic libc, this target also needs an
Android NDK libc configuration:

```text
zig build qualcomm-probe -Dtarget=aarch64-linux-android -Doptimize=ReleaseSafe -Dandroid-libc=/path/to/android-libc.txt
```

Successful output:

```json
{"tensorfold":"qualcomm-probe","libcdsprpc":true,"fastrpc_symbols":true,"dsp_session_opened":false,"stage":1}
```

The next target packages the existing scalar EXL3 format/oracle tests as an
Android executable. Copy it together with
`zig/tests/fixtures/exl3_oracle.bin`, preserve that relative fixture path in the
device working directory, and execute the test binary there:

```text
zig build qualcomm-exl3-test -Dtarget=aarch64-linux-android -Doptimize=ReleaseSafe -Dandroid-libc=/path/to/android-libc.txt
```

On Snapdragon 8 Elite, the test reproduces all generated oracle cases across
the three codebooks and supported widths, then unpacks and dequantizes the real
8.00-bpw fixture within the upstream tolerance.
