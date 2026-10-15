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

## Real EXL3 projection bridge

`qualcomm-exl3-projection` advances the bridge from synthetic weights to the
public-checkpoint `mcg` 8.00-bpw block embedded in TensorFold's checked-in
oracle fixture. TensorFold's Zig implementation parses the EXL3 record,
dequantizes and rotates the 128x128 weight, transposes it into projection
layout, creates deterministic activations, and calculates both the decode and
projection references. A narrow C ABI passes the dequantized FP32 projection
to the selected GGML backend.

```text
zig build qualcomm-exl3-projection -Dtarget=aarch64-linux-android \
  -Doptimize=ReleaseSafe -Dandroid-libc=/path/to/android-libc-api36.txt \
  -Dggml-include=/path/to/ggml/include \
  -Dggml-lib=/path/to/android/ggml/libs

tensorfold-qualcomm-exl3-projection cpu exl3_oracle.bin ITERATIONS BATCH CPU_THREADS
tensorfold-qualcomm-exl3-projection htp exl3_oracle.bin ITERATIONS BATCH CPU_THREADS
```

On Snapdragon 8 Elite, all 12 measured samples passed. The EXL3 decode stayed
within `4.858e-17` of the FP64 oracle. Projection maximum absolute error was
`8.941e-8` at batch 1 and `2.492e-4` at batch 128. The HTP trace selected
`hvx-tiled` for batch 1 and `hmx-tiled` for batch 128. Median compute times:

| Real EXL3 block | CPU ms | HTP ms | HTP speedup |
| --- | ---: | ---: | ---: |
| batch 1 | 0.479311 | 0.412476 | 1.162x |
| batch 128 | 0.887531 | 0.426646 | 2.080x |

This still dequantizes on CPU and uploads FP32 weights. The next backend gate
is to consume the packed EXL3 representation without materializing a complete
FP32 model copy.

## Full-checkpoint projection probe

`qualcomm-exl3-model-probe` uses TensorFold's portable safetensors reader to
memory-map the pinned 3.44 GB `UnstableLlama/Qwen3.5-2B-exl3-8.00bpw`
checkpoint. It validates the 1,071-entry tensor index, reads the real layer-0
`linear_attn.in_proj_qkv` EXL3 tensors, reconstructs either a 128x128 block or
the complete 2048x6144 projection, and runs the projection on CPU or HTP.

The production reconstruction uses an O(n log n) Walsh-Hadamard transform.
The original direct O(n squared) transform remains the independent scalar
oracle, and both implementations pass the real-checkpoint fixture tolerance.

Three full-projection samples per backend produced these medians:

| Full 2048x6144 EXL3 projection | CPU ms | HTP ms | HTP speedup |
| --- | ---: | ---: | ---: |
| batch 1 | 1.765583 | 1.064021 | 1.659x |
| batch 128 | 27.979927 | 1.173563 | 23.842x |

All 12 samples passed the declared mixed absolute/relative FP32 tolerance. The
batch-128 trace selected `hmx-tiled`; its maximum absolute error was
`0.0013365746` against a reference magnitude of `3.4685607` (tolerance
`0.0022342803`). The checkpoint and on-phone copy both hash to
`a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f`.
