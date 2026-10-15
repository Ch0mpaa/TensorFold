# Qualcomm Android port

Status: complete autoregressive correctness POC with a persistent decoded-weight
cache. TensorFold executes Qwen3.5-2B and Qwen3.5-4B prompts across every layer,
carries recurrent and KV state, and routes all EXL3 projections through Qualcomm
HTP v79. It is not yet a serving runtime, a sustained-performance result, or a
native packed-EXL3 HTP backend.

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

## Complete autoregressive EXL3 forward

`qualcomm-exl3-forward` executes one or more token positions through Qwen3.5-2B.
`qualcomm-exl3-forward-4b` builds the same model-profiled source for Qwen3.5-4B,
including its 32 DeltaNet value heads, 16 DeltaNet key heads, 16 attention heads,
and four KV heads. TensorFold owns the architecture math for Gated DeltaNet and
full-attention layers, residuals, RMS norms, MLPs, and the tied output head.
Every quantized projection is reconstructed from the pinned EXL3 tensors and
dispatched through one persistent GGML CPU or Hexagon HTP backend. Across
positions it carries the DeltaNet FP32 recurrent matrices, BF16 convolution
history, and rotated full-attention K/V cache.

```text
zig build qualcomm-exl3-forward -Dtarget=aarch64-linux-android \
  -Doptimize=ReleaseSafe -Dandroid-libc=/path/to/android-libc-api36.txt \
  -Dggml-include=/path/to/ggml/include \
  -Dggml-lib=/path/to/android/ggml/libs

zig build qualcomm-exl3-forward-4b -Dtarget=aarch64-linux-android \
  -Doptimize=ReleaseSafe -Dandroid-libc=/path/to/android-libc-api36.txt \
  -Dggml-include=/path/to/ggml/include \
  -Dggml-lib=/path/to/android/ggml/libs

tensorfold-qualcomm-exl3-forward cpu model.safetensors TOKEN_ID CPU_THREADS
tensorfold-qualcomm-exl3-forward htp model.safetensors TOKEN_ID CPU_THREADS
tensorfold-qualcomm-exl3-forward htp model.safetensors TOKEN_ID CPU_THREADS CACHE_DIR MODEL_SHA256
tensorfold-qualcomm-exl3-forward htp model.safetensors TOKEN_ID CPU_THREADS CACHE_DIR MODEL_SHA256 STEPS PROMPT_SUFFIX_TOKEN_IDS
```

`STEPS` is the number of greedy output tokens. The optional comma-separated
prompt suffix follows the initial `TOKEN_ID`; it makes a teacher-forced prompt
before greedy generation. The executable limits a proof run to 128 prompt and
32 generated tokens. Each position emits a JSON correctness record and the
last line summarizes the complete run.

For input token `248045` (`<|im_start|>`), both Snapdragon 8 Elite runs
exited zero and returned next token `846` with the same ordered top-five token
IDs: `846, 8678, 198, 96746, 2244`. Measured projection compute was 936.409
ms on CPU and 122.526 ms on HTP, a 7.64x speedup. Peak RSS was about 2.24 GiB.

This initial path reconstructs weights on demand. Reconstruction took about
46 seconds and dominated the 64.3-second CPU and 62.2-second HTP wall times.
It is a correctness/bring-up executable, not a generation-speed claim. The
next runtime step is a persistent decoded-weight cache, followed by direct
packed-EXL3 kernels. The tied FP16 embedding evaluates the architectural head;
the pinned upstream ExLlamaV3 implementation, using the checkpoint's
duplicated quantized `lm_head`, independently returned the same ordered
top-five IDs. Exact logit parity for that alternative head and multi-token
state/cache parity remain required gates before generation benchmarking.

### Persistent decoded-weight cache

The optional cache stores each reconstructed projection as immutable row-major
FP32. The safe default content-addresses every entry from its complete EXL3
trellis and side tensors. Supplying the checkpoint's independently verified
SHA-256 enables a faster alias key: on its first use TensorFold verifies the
content-addressed entry and creates a hard link, and subsequent process starts
can avoid rescanning the packed tensor. The cache header validates format,
geometry, and payload length; partial files are never published because writes
are synced and renamed into place.

On Phone B, the pinned Qwen3.5-2B checkpoint produced 150 content entries with
5,486,149,632 payload bytes (5.1 GiB apparent disk usage). Three final warm
runs per backend, after removing the one-shot path's redundant benchmark
warm-up execution, measured:

| Cached complete forward | CPU median | HTP median | HTP speedup |
| --- | ---: | ---: | ---: |
| End-to-end | 2,117.835 ms | 1,685.276 ms | 1.257x |
| Projection-call wall time | 1,630.119 ms | 1,086.248 ms | 1.501x |
| Matrix compute | 426.206 ms | 169.033 ms | 2.521x |

All final runs had 150 trusted hits, zero misses, zero reconstruction/hash
time, and the same backend-specific state hashes and top-five tokens as the
uncached proof. The HTP median is 36.91x faster than the earlier 62,201.270 ms
uncached HTP result. Peak process RSS fell from roughly 2.35 GiB to roughly
1.02 GiB because warm entries are mapped one projection at a time instead of
holding reconstruction scratch and transposed copies together.

Profiling also shows only 169.033 ms of the 1,086.248 ms HTP projection wall
median is matrix compute; repeated graph creation and FP32 weight upload now
dominate. Resident packed-EXL3 weights and an accelerated quantized output
head (the scalar tied head costs about 407 ms) are the next runtime gates. Raw
evidence is in `recordings/20261015-080827-tensorfold-decoded-cache/` in the
mobile lab.

### Multi-token Qualcomm POC

The first autoregressive proof used the exact 15-token Qwen3.5 chat template
for `What is 2+2?` and greedily produced eight tokens. Both Snapdragon CPU and
HTP produced the identical 23-token sequence as pinned upstream ExLlamaV3,
including the generated text prefix `<think>\nThinking Process:\n\n1.`. The
greedy top-one token matched at all 22 evaluated positions; the complete
ordered top five matched at 19 positions. The remaining three differences
were lower-rank swaps from the tied FP16 architectural head versus the
checkpoint's duplicated quantized EXL3 `lm_head`; maximum matched-logit
absolute difference was 0.281.

One controlled POC capture per backend measured:

| 15-token prompt + 8 generated tokens | CPU | HTP | HTP speedup |
| --- | ---: | ---: | ---: |
| End-to-end | 46,893.059 ms | 28,967.271 ms | 1.619x |
| Projection-call wall time | 36,870.362 ms | 18,290.011 ms | 2.016x |
| Matrix compute | 10,015.526 ms | 3,764.632 ms | 2.660x |
| Evaluated positions/s | 0.469 | 0.759 | 1.619x |

Every run made 3,300 projection calls with 3,300 trusted cache hits, zero
misses, and zero dequantization time. This is a correctness POC and a single
exploratory timing capture, not a sustained benchmark or a llama.cpp speed
comparison. The correctness mode evaluates the scalar output head at every
teacher-forced prompt position, so its end-to-end generated-token rate is not
an optimized serving metric. Raw evidence is in
`recordings/20261015-085918-tensorfold-multitoken-poc/` in the mobile lab.

### Qwen3.5-4B simultaneous two-phone POC

The 4B profile was validated with `UnstableLlama/Qwen3.5-4B-exl3-4.00bpw`,
SHA-256 `117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1`.
It has 32 layers and 200 projections per position. A cold pass created 200
content-addressed decoded entries; a warm 15-token prompt plus eight-token
generation made 4,400 trusted cache hits with no miss or reconstruction.
TensorFold generated the exact same eight IDs as upstream ExLlamaV3:
`248068,198,90700,8340,25,271,16,13`.

A simultaneous control on a second matching OnePlus 13 used Qwen3.5-4B Q4_0
with llama.cpp/Hexagon HTP. The two lanes consumed the same 15 token IDs,
started from a clock-offset-compensated barrier with about 6 ms estimated skew,
exited zero, and produced the same eight-token text and IDs.

| One exploratory simultaneous run | llama.cpp Q4_0 HTP | TensorFold EXL3 HTP |
| --- | ---: | ---: |
| End-to-end wall time | 689.604 ms | 140,533.959 ms |
| End-to-end generated tokens/s | 11.601 | 0.056926 |
| Prompt tokens / generated tokens | 15 / 8 | 15 / 8 |

llama.cpp was 203.8x faster in this implementation-level POC. That number is
not a format or kernel superiority claim: TensorFold reread 313,733,939,200
bytes from its 13 GiB decoded FP32 cache, rebuilt projection work, and evaluated
the scalar vocabulary head at every one of 22 positions. llama.cpp used a warm
resident Q4_0 server. The phones also began at different battery state, and the
quantizations differ. The result proves functional 4B EXL3 inference on HTP and
identifies resident packed weights, graph reuse, prompt-head elision, and an
accelerated output head as the next gates. Raw evidence is in
`recordings/20261015-095636-qwen35-4b-simultaneous/` in the mobile lab.

### Qwen3.5-4B tuned serving POC

The first serving-oriented tuning pass retained the same 15 prompt token IDs
and exact eight generated IDs while reducing TensorFold wall time from
140,533.959 ms to 8,628.986 ms, a 16.286x improvement. It batches the complete
teacher-forced prompt layer-by-layer, reuses GGML graphs and backend buffers by
shape, skips the 14 prompt-position vocabulary scans whose next tokens are
known, and uses a threaded ARM NEON tied head. A persistent EXL3-derived FP16
execution cache reduces the staged projection traffic from 313.734 GB to
57.043 GB per request and allows batched prompt matrices to use the F16 HMX
path.

| Matched 15-token prompt + 8 generated tokens | llama.cpp Q4_0 HTP | TensorFold tuned EXL3 HTP |
| --- | ---: | ---: |
| End-to-end wall time | 666.062 ms | 8,628.986 ms |
| End-to-end generated tokens/s | 12.011 | 0.927 |
| Projection compute | not exposed | 1,398.517 ms |
| Projection-call wall time | not exposed | 6,167.799 ms |

The original 203.8x implementation gap is 12.955x after tuning; llama.cpp is
still faster. All eight generated top-one choices matched upstream ExLlamaV3,
and six of eight ordered top-five ID lists matched. FP16 staging is therefore
numerically close but not bit-identical to the FP32 correctness path. It is an
intermediate execution format, not native packed-EXL3 execution. The next
performance gate is a packed-EXL3 HTP/HVX/HMX kernel or another resident
compressed-weight path, because 6.168 seconds of the tuned result remains in
projection-call wall time while matrix compute itself is 1.399 seconds. Raw
evidence is in `recordings/20261015-104834-tensorfold-4b-tuned/` in the mobile
lab.

### Resident-runtime v2

The next pass keeps each FP16 projection resident in the Qualcomm backend,
shares one weight copy between prompt and decode graphs, groups independent
projections with a common input, and fuses each MLP's gate, up, SiLU/multiply,
and down operations into one HTP graph. ARM NEON handles the small gate
matvecs and the Gated DeltaNet recurrent update. Repeated-request mode keeps
the backend alive so cold residency setup and warmed execution are reported
separately. The tied FP16 vocabulary head is also resident on HTP after the
first request.

| Same 15-token prompt + 8 generated tokens | Wall time | Original TensorFold speedup |
| --- | ---: | ---: |
| v2 cold single request | 7,062.013 ms | 19.900x |
| v2 warmed range, four samples | 3,384.374-3,513.147 ms | 40.006-41.525x |
| v2 warmed median | 3,458.323 ms | 40.636x |

Every capture generated `248068,198,90700,8340,25,271,16,13`. Each warmed
request made 1,600 resident projection hits with zero weight uploads, cache
reads, or fallbacks. Against the existing 666.062 ms warmed llama.cpp Q4_0
control, the warmed median gap is 5.192x. This is still an implementation-level
comparison rather than an intrinsic format result: TensorFold executes from a
7.13 GB EXL3-derived FP16 staging representation, while llama.cpp uses resident
Q4_0. Raw evidence is in
`recordings/20261015-115427-tensorfold-4b-resident-v2/` in the mobile lab.

## Matched llama.cpp control

The control lane uses the same Snapdragon 8 Elite phone and the same
`091b6a7` llama.cpp/Hexagon binary for CPU and HTP. Its model is the pinned
Qwen3.5-2B Q4_0 GGUF from `unsloth/Qwen3.5-2B-GGUF`, revision
`f6d5376be1edb4d416d56da11e5397a961aca8ae`, SHA-256
`cd70221bebaee0503e0f6717e174250cd7825aa88438b3aabec9ad55731d9bb1`.

Five-repetition `llama-bench` results with six CPU threads:

| Workload | CPU | HTP | HTP / CPU |
| --- | ---: | ---: | ---: |
| prompt, 128 tokens | 319.53 tok/s | 1580.22 tok/s | 4.95x |
| generation, 32 tokens | 40.14 tok/s | 34.13 tok/s | 0.85x |

These controls show why prefill and decode must remain separate claims. HTP
wins decisively on prompt matrix work, while its dispatch and boundary costs
lose at batch-one generation. They also are not directly comparable to the
current TensorFold wall time: llama.cpp uses a warmed Q4_0 runtime, while the
TensorFold proof reconstructs EXL3 weights on demand for a single complete
position-zero correctness pass.
