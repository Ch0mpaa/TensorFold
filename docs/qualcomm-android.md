# Qualcomm Android port

Status: stage 3 userspace execution bridge. TensorFold opens a real Qualcomm
HTP v79 session and executes deterministic matmul graphs on HTP, but is not yet
executing a complete EXL3 model on HTP.

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

## CPU/HTP bridge harness

`qualcomm-matmul` is the first execution bridge. TensorFold owns the deterministic
inputs, graph construction, correctness oracle, timing and JSON result, while the
selected GGML CPU or Hexagon backend executes the graph. It is an intermediate
integration target, not the final native EXL3 kernel backend.

```text
zig build qualcomm-matmul -Dtarget=aarch64-linux-android -Doptimize=ReleaseSafe \
  -Dandroid-libc=/path/to/android-libc.txt \
  -Dggml-include=/path/to/ggml/include \
  -Dggml-lib=/path/to/android/ggml/libs

tensorfold-qualcomm-matmul cpu K M N ITERATIONS CPU_THREADS
tensorfold-qualcomm-matmul htp K M N ITERATIONS CPU_THREADS
```

Every result includes the matrix shape, time, throughput and maximum absolute
error against TensorFold's scalar FP32 reference.

### Snapdragon 8 Elite validation

Phone B (`CPH2655`) ran three samples of each workload through each backend.
All 24 samples matched the scalar FP32 oracle with zero observed maximum
absolute error. Times below are the sample medians and cover backend graph
compute after one warmup; host-to-backend upload and result download are not
included.

| Workload | Shape `(K,M,N)` | CPU ms | HTP ms | HTP speedup |
| --- | ---: | ---: | ---: | ---: |
| decode micro | 256, 256, 1 | 0.736781 | 0.417172 | 1.766x |
| decode Qwen3.5-4B width | 2560, 2560, 1 | 2.259547 | 0.691000 | 3.270x |
| prefill micro | 256, 256, 128 | 0.636039 | 0.254089 | 2.503x |
| prefill Qwen3.5-4B width | 2560, 2560, 128 | 14.061302 | 0.742344 | 18.942x |

The verbose backend trace identified `HTP0`, architecture v79, six HVX units,
one HMX unit, and an 8 MiB VTCM. The largest prefill case selected the
`hmx-tiled` kernel. The exact backend source was pinned at
`091b6a7fc561b78da90a0ed2d524ec995458c657`; its Android shared libraries
require an API 36 link surface even though TensorFold's standalone smoke,
FastRPC probe, and scalar EXL3 test remain compatible with API 29.
