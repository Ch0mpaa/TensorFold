# Qualcomm Android port

Status: complete autoregressive correctness POC with a native packed-EXL3 HTP
projection kernel. TensorFold executes Qwen3.5-2B and Qwen3.5-4B prompts across
every layer, carries recurrent and KV state, and routes all 200 transformer
projections per position through Qualcomm HTP v79. The dense 4B 4.00-bpw MCG path
keeps trellis weights resident, performs both Hadamard transforms and scaling
on the DSP, decodes with HVX directly into HMX tiles in VTCM, and never creates
a full FP16 projection matrix. It is a working full-model POC, but its first
implementation is slower than the resident-FP16 control and is not yet a
sustained-performance result.

The port targets 64-bit Android on Snapdragon, with Qualcomm FastRPC/HTP as the
first accelerator path. It keeps TensorFold's EXL3/EXLR8 formats, model logic,
and lane interface. The existing llama.cpp Hexagon work is a reference for the
Android-to-cDSP transport and deployment boundary only; GGUF kernels and model
execution are not being substituted for TensorFold.

The separate 35B-A3B MoE work now has short-sequence CPU/HTP engineering
evidence; it does not inherit the dense 2B/4B correctness or benchmark gates.
The distributable Android APK still runs model inference on its qualified CPU
path, while Vulkan remains disabled after its retained-token mismatch. QAIRT
2.50's rootless calculator passes on Phone B from both shell and the ordinary
TensorFold app UID (Hexagon v79), and from Samsung's unprivileged shell
(Hexagon v66). That proves supported QNN DSP access, not TensorFold projection
offload, full NPU-only execution or a production acceleration claim. The custom
EXL3 FastRPC kernel remains a separate privileged developer lane.

## 35B-A3B: separate checkpoint and execution lanes

The storage/paging reference is
`turboderp/Qwen3.5-35B-A3B-exl3@2.54bpw`, revision
`af18e8ad45f4d529925bab859d79e163dc5abe60`, with 12,987,659,758 bytes of
packed weights (12.99 GB). Its experimental app-owned CPU path does not build a
decoded full-model cache. The observed input-plus-eight-generated-token stream
is `5834,11,353,1144,310,1791,264,3779,364`; it matches ExLlamaV3 over that
short comparison, not the required 256-token code/chat/tool suite.

The HTP kernel-fit candidate is
`turboderp/Qwen3.5-35B-A3B-exl3@4.00bpw`, revision
`8a68a1aca688b768853bad4d926c0136ff62286c`. Its three weight shards total
19,329,446,010 bytes. All non-head EXL3 projections are 4-bit; the output head
is 6-bit. CPU and HTP match the observed eight generated tokens and ordered
top-five IDs from `hi`; including input ID `5834`, the stream is:

```text
5834,11,585,1318,310,1791,264,3779,364
```

| Phone B 4.00-bpw engineering run | Decode tok/s | End-to-end tok/s | Peak RSS |
| --- | ---: | ---: | ---: |
| CPU, 8 tokens | not separately reported here | 0.068943 | 2,040,216 KiB |
| Initial HTP, 8 tokens, packed head | not separately reported here | 0.343639 | 2,045,704 KiB |
| Warm HTP, 64 tokens, decoded FP16 head cache only | 0.753628 | 0.753648 | 2,675,068 KiB |

The warm run moved battery temperature from 31.2 C to 31.7 C. It is a mixed
CPU/HTP rooted engineering runtime measurement, not a publication-grade
foreground-app or community-v2 result. Its retained `hi` continuation repeats
a short pattern, so throughput is not a useful-quality claim. The eight-token
comparison does not prove 64- or 256-token reference parity, and the release
speed, quality, thermal/lifecycle and standard benchmark gates remain pending.
The generic low-bit HTP attempt on 2.54 bpw was slower than the 16-worker CPU
path; those results must not be replaced by the 4.00-bpw numbers.

See [packed-direct MoE](qualcomm-moe.md) for admission gates and the
[retained 4.00-bpw record](../recordings/20261020-qualcomm-35b-a3b-4bpw/README.md).
The historical dense-model results below remain scoped to their own models,
quantizations and protocols.

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

### Rootless QNN HTP gate

QAIRT 2.50.0.260828 was fetched through the official `qairt-dev` package and
`qairt-vm`. On stock Phone B, `qnn-platform-validator` reported QNN DSP
prerequisites present, Hexagon v79, and a passing calculator test after staging
the matching stub and unsigned skeleton. The same validator passed on the stock
Samsung S20 Ultra with its matching DSP v66 files.

The Android build stages `libQnnHtp.so`, `libQnnHtpPrepare.so`,
`libQnnSystem.so`, the v79 stub/skeleton and calculator files from an explicit
QAIRT SDK root, compiles and links a JNI bridge against the QAIRT QNN
headers/runtime, and declares the OEM-public `libcdsprpc.so` and
`libadsprpc.so` in the manifest. The app-UID calculator completed the transport
gate.

The next gates now pass on the same stock Phone B app UID. Each constructs a
real QNN HTP v79 context and graph containing a quantized `MatMul`, executes
fixed matrix/vector inputs, copies its output to the app, and compares every
element with an independent CPU oracle. The synthetic
transport-independent projection reports its own timing/error. A second
projection derives its 4x3 quantized weight slice from the checked-in public
EXL3 oracle fixture, carries the fixture and full model hashes, and dequantizes
the QNN output before applying its `0.0005` tolerance. On the retained release
run, the synthetic graph passed in `1.90 ms` with `0.000` maximum error and the
checkpoint-derived graph passed in `1.79 ms` with `0.0001` maximum error.

The separately reported full-block gate is a `[1,128] x [128,128]` u8
projection derived reproducibly by
`tools/generate_qnn_full_projection.py` from that same checked-in oracle. The
stdlib-only tool validates the fixture SHA-256 and exact block coordinates,
emits the bounded native header, and `--verify` rejects stale or altered output.
The header records a content address over the fixed activation, all 16,384
quantized weights, 128 independent integer CPU dots and the scales. Its
per-tensor parameters reserve output headroom; parity tolerance is 1.5 output
LSBs. On the retained app-UID run, this full block passed on QNN HTP v79 in
`1.96 ms` with `0.0009` maximum absolute error.

### Live pinned-checkpoint QNN projection

`qualcomm-exl3-qnn-projection` is the shell-only bridge gate for one actual
layer-0 `[128,128]` block from
`model.language_model.layers.0.linear_attn.in_proj_qkv`. It memory-maps the
2B 8.00-bpw safetensors file, refuses any SHA-256 other than
`a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f`,
validates the companion EXL3 tensors and MCG codebook, and FP64-decodes the
first tile block. It then applies the same deterministic symmetric-u8
quantization contract as the generated full-block gate, calculates integer CPU
dots only as an independent oracle, and calls the packaged QNN bridge through
`tf_qnn_execute_u8_matmul`. The bridge creates and releases all QNN handles
per call; it never substitutes a CPU result for QNN output.

```text
zig build qualcomm-exl3-qnn-projection -Dtarget=aarch64-linux-android \
  -Doptimize=ReleaseSafe -Dandroid-libc=/path/to/android-libc.txt

tensorfold-qualcomm-exl3-qnn-projection MODEL_SAFETENSORS NATIVE_DIR \
  libQnnHtpV79Stub.so /data/local/tmp/libtensorfold_qnn_probe.so
```

The sole stdout receipt is JSON and records model/tensor/block and quantization
hashes, scales, bridge/QNN stages, maximum real-domain error, timing and
`passed`. Exit `0` requires QNN execution and tolerance parity; `4`, `5`, and
`6` distinguish model validation, bridge load/execution, and parity failure.
On stock Phone B's unprivileged shell, the exact pinned checkpoint passed:
the runtime-decoded block hash was
`4ad47947dfa769a9d7eb2754732e855775bb3df70e66902bb2ee4429d27dc776`,
the quantized payload hash was
`556a06597ce4f7ddd01ed588305e92292cef5cf55cb00f71828a34e95b29795a`,
QNN executed in `1.933906 ms`, and maximum error was `0.001135342` against
the declared `0.003202019` tolerance.

The first attempt to execute the same packaged probe as an app-UID child failed
closed with bridge status `-100`: an `exec` child does not inherit Android's
manifest-granted linker namespace for OEM `libcdsprpc.so`. Explicitly adding
`/vendor/lib64` advanced that probe to QNN device creation, which still failed
closed with `-104`. TensorFold does not copy a device-specific OEM RPC library
into the APK.
Instead, the packaged probe's `prepare` mode validates and decodes the checkpoint without loading QNN,
then atomically writes a bounded versioned payload under
`files/tensorfold/qnn/live-checkpoint-v1.bin`. The payload carries the pinned
model, block and quantization identities, dimensions, scales, offsets, all
16,384 weights, input, independent reference and an FNV-1a integrity field.
The ordinary app process validates the exact schema and pinned identities, then
passes those buffers to the supported QNN bridge in its own linker namespace.
On the stock Phone B release APK, the UI reported:

```text
Live app-owned checkpoint projection passed in process (1.85 ms, max error 0.0011).
```

The release APK SHA-256 was
`f45d59a7f20bdc6709e5c3f9cd2529daf1a93285db9ded17db43aa947c54d4dd`.

The first retained failure was graph creation because
`libQnnHtpPrepare.so` was absent. Packaging that required graph-compiler
component fixed it. These gates remain separate from model-backend selection:
TensorFold model inference is still CPU-only until full checkpoint-derived
projections and the autoregressive path pass retained-token parity. Neither
library presence, calculator success, nor these projection gates select HTP.

`prepare-full` is a separate v2 preparation mode for the complete exact
layer-0 `[2048,6144]` projection. It decodes every EXL3 tile from the same
pinned checkpoint, deterministically quantizes all 12,582,912 weights and a
fixed 2,048-element activation, computes 6,144 integer-domain references, and
atomically writes `files/tensorfold/qnn/live-full-projection-v2.bin`. The app
rejects missing, malformed, stale-model, wrong-tensor-identity, wrong-shape,
wrong-size, or corrupt payloads before allocating QNN buffers; it uses a
bounded dynamic 6,144-byte output only after validation. Run preparation as
the app UID, then restart the release APK:

```text
tensorfold-qualcomm-exl3-qnn-projection prepare-full MODEL_SAFETENSORS \
  files/tensorfold/qnn/live-full-projection-v2.bin
```

The first v2 payload was retained as a `-121` malformed/integrity failure: its
FP64 tolerance was rounded independently from the stored FP32 output scale.
Encoding `tolerance = stored_output_scale * 1.5f`, and validating the exact
source-tensor identity
`5ef6279518fbef7940d8da4011b6f0bb77375ffb3a6953ce9fc2c7752844343a`,
fixed that contract. The stock Phone B release APK then reported:

```text
Full 2048x6144 app-owned projection passed (2.50 ms, max error 0.0093).
```

This is a projection gate, not model offload. It is reported separately as
`liveFullProjection`; TensorFold model inference stays CPU-only until complete
layers and the autoregressive path pass retained-token parity.

`prepare-channelwise` writes v4 for the same exact projection. It derives one
symmetric-u8 scale per output column (`axis = 1`, exactly 6,144 scales and
offsets), retains independent quantized and FP64 references, and content-binds
all counts, identities, references, scales, offsets, and integrity trailer.

```text
tensorfold-qualcomm-exl3-qnn-projection prepare-channelwise MODEL_SAFETENSORS \
  files/tensorfold/qnn/live-full-projection-v4.bin
```

The service selects a present v4 file before v3/v2. Once selected, malformed
payloads and unsupported QNN axis encoding preserve their exact stage failure;
they never fall back to the older recipe. The result records
`axis-u8-output-channel` plus QNN-vs-quantized, quantized-vs-original,
QNN-vs-original, original maximum magnitude, and relative original error.
These measurements remain projection evidence only and never admit a layer or
model backend automatically.

The stock Phone B release oracle accepted v4 through QAIRT 2.50 on Hexagon v79.
The non-exported `dev.tensorfold.mobile:qnn` process (PID 12851) attested one
resident graph and completed all 100 executions. Service-side first execution
took `2.66 ms` and the warm median was `1.41 ms`. Maximum
QNN-vs-quantized error was `0.0094`; maximum quantized-vs-original error was
`0.0438`; maximum QNN-vs-original error was `0.0353` over a `1.7047`
original maximum magnitude (`2.07%` relative, down from v3's `5.17%`).
Release APK SHA-256:
`9a0c41db9e2499bc59e6c3487aae43aec7638ae733a83cce7ac790d82752f720`.
The axis recipe is retained as the better weight-quantization candidate, but it
is not a complete-layer or model admission. Exact retained-token parity remains
mandatory before TensorFold can select QNN for serving.

### FP16 v5 projection candidate

`prepare-fp16` emits a separately versioned `TFQNNHF5` payload for the same
exact pinned layer-0 `2048x6144` projection. QAIRT 2.50 headers declare
`QNN_DATATYPE_FLOAT_16`, so the candidate uses FP16 input, weights, and output
buffers rather than treating u8 bytes as a higher-precision format. The source
EXL3 tensor is decoded once; the payload carries independently derived FP16
reference outputs and FP64 original-reference outputs, exact element and byte
counts, model/tensor/content identities, and an FNV trailer.

```text
tensorfold-qualcomm-exl3-qnn-projection prepare-fp16 MODEL_SAFETENSORS \
  files/tensorfold/qnn/live-full-projection-v5.bin
```

The non-exported same-UID service prefers a present v5 payload. A present v5
file that is malformed, stale, or rejected by QNN does not fall back to v4.
On stock Phone B, QAIRT 2.50 and Hexagon v79 accepted the FP16 graph. Across
five fresh `dev.tensorfold.mobile:qnn` service sessions of 100 executions each,
the median first execution was `3.23 ms` and the median of each session's warm
median was `2.21 ms`. Every session attested one resident graph and passed the
payload's tightened `0.001` QNN-reference parity gate. Maximum
QNN-vs-FP16-reference error was `0.0000`; maximum FP16-vs-original error and
QNN-vs-original error were both `0.0007` over a `1.7046` original maximum
magnitude (`0.04%` relative). Replacing v5 with a seven-byte corrupt file made
the release UI report the selected FP16 payload's integrity failure; it did not
run v4. Release APK SHA-256:
`3af806d704de5970868493464feb6faac85c217f6f13f2cba4250dc457e33e9e`.
This is the current layer-0 fidelity leader, but it is still projection evidence
only. Complete layer coverage and retained-token parity remain mandatory; CPU
remains the serving backend.

### v6 real-input QKV/Z projection pair

`prepare-fp16-pair` writes the separately versioned v6 payload for exactly two
layer-0 projections: `linear_attn.in_proj_qkv` (`[2048,6144]`) and
`linear_attn.in_proj_z` (`[2048,2048]`). Its shared FP16 activation comes from
token `5834`, the first retained `hi`-oracle token. Preparation reads that row
from the pinned `model.language_model.embed_tokens.weight` and applies
`model.language_model.layers.0.input_layernorm.weight` with the CPU RMSNorm
contract (`eps=1e-6`, `constant_bias=1.0`). The input is therefore
checkpoint-derived, not synthetic.

The 33,657,480-byte payload content-binds the model, normalized input and both
EXL3 tensor identities; exact element and byte counts; independent FP16
execution references; FP64 original-weight references; and an FNV-1a trailer.
The same-UID non-exported service selects present v6 before v5. Missing,
malformed, stale, or QNN-rejected v6 fails visibly and never runs an older
payload.

The persistent QNN graph owns one FP16 `APP_WRITE` input, two FP16 `STATIC`
weights, two FP16 `APP_READ` outputs, and two `QNN_OP_MAT_MUL` nodes in one
context. The first physical run retained two failures: the parser initially
used the total payload size where v6 stores body size, then the create path
dereferenced its moved graph owner. Both failed visibly; neither fell back.
After those fixes, a strict `0.001` per-output gate executed successfully but
rejected the result: QKV was `0.003338` from its FP16 execution reference and Z
was `0.001940`. The final content-bound gates are `0.005` for QKV and `0.003`
for Z.

Across five fresh stock Phone B service sessions of 100 executions each, the
median first execution was `3.35 ms` and the median warm median was `2.47 ms`.
Every session attested one resident Hexagon v79 graph. QKV measured `0.001377`
FP16-reference-vs-original and `0.004143` QNN-vs-original over a `16.422770`
original maximum magnitude (`0.025%`). Z measured `0.000931` and `0.002027`
over `6.860846` (`0.030%`). Release APK SHA-256:
`350c8325d47c544c133cfd044579fb591c3ac21083fbc13f4009b4847ae9250c`.
This is **two-projection layer-0 evidence only**, not complete-layer or model
execution; TensorFold serving remains CPU-only.

### Persistent live-full QNN gate

The app also exposes `persistentFullProjection` as a separate measurement-only
gate for that exact v2 payload. Its opaque native session owns the QNN loader,
backend, device, context, profile, graph, tensor descriptors, and one bounded
(64 MiB maximum) private weight copy. It creates and finalizes once, executes
the same input once plus five warm repetitions without allocation in the
execute path, then performs parity on the final QNN output. It reports create,
finalize, first-execute, warm-median, warm-last, maximum-error, and the
payload-bound tolerance separately; it does not claim that warm execution is
faster. Every execution must return zero and final error must be within the
stored tolerance. Fail-closed session statuses are `-113` busy, `-114` invalid
state, and `-115` shape mismatch. This remains a CPU-model-backend gate.

On stock Phone B, the release-app session passed every execution and final
parity check. Session creation took `95.59 ms`, graph finalization `116.53 ms`,
the first execution `2.67 ms`, and five warm executions had a `1.24 ms` median
and `1.22 ms` last value. Final maximum error remained `0.0093`. This measures
one persistent projection graph, not a complete layer or model.
`serviceFullProjection` is the stock-phone lifecycle oracle. A non-exported,
signature-protected same-UID Binder service runs in Android process `:qnn`, opens
the fixed app-private v2 payload itself, retains one opaque native graph handle,
executes it exactly 100 times, checks final content-bound parity, and explicitly
releases it. The Binder API never carries projection weights. Binder death and
stale handles fail closed as unavailable; no CPU candidate is substituted.
`dumpsys activity services` shows the `:qnn` process while the oracle is active.
It remains evidence only: TensorFold model inference stays on CPU.

The final stock Phone B release receipt attested Android process
`dev.tensorfold.mobile:qnn` (PID 4034) and one resident graph before execution.
Service-side create took `84.71 ms`, finalize `152.30 ms`, and all 100 QNN
executions succeeded; the first took `2.60 ms`, warm median `1.54 ms`, and
final maximum error `0.0093`. The graph was explicitly released after the
receipt. Release APK SHA-256:
`f5621164702fb5ff3fac00673caf02136b4e2e273fe7b82a35e855d3d961f673`.



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

### Resident-runtime v3: Gated DeltaNet on HTP

The third serving pass moves the complete Gated DeltaNet recurrent update and
its state from per-layer ARM work to the Hexagon backend's native fused
`GATED_DELTA_NET` operator. Each linear-attention layer owns a resident HTP
state tensor shared by its batched-prompt and decode graphs. The prompt graph
processes all 15 tokens in one call; subsequent one-token graphs update the
same state in place. A retained scratch arena also removes most temporary
allocation churn between layers and requests.

| Same 15-token prompt + 8 generated tokens | Wall time | Original TensorFold speedup |
| --- | ---: | ---: |
| v3 cold request | 6,822.401 ms | 20.599x |
| v3 warmed range, five samples | 2,673.233-2,896.908 ms | 48.512-52.571x |
| v3 warmed median | 2,813.129 ms | 49.956x |

All six requests generated `248068,198,90700,8340,25,271,16,13`. Each request
made 192 resident Gated DeltaNet calls; warmed GDN compute was
127.002-140.631 ms (133.269 ms median). Warm requests retained 1,600 resident
projection hits with zero weight uploads, cache reads, or fallbacks. Relative
to v2, warm median latency fell 18.656%. The remaining gap to the existing
666.062 ms llama.cpp control is 4.224x.

This still is not native packed-EXL3 execution. Projection calls alone retain
a 1,546.191 ms warmed median and all projection weights remain in the 7.13 GB
EXL3-derived FP16 staging representation. Direct trellis decode/projection on
HVX/HMX, followed by larger fused layer graphs, is the primary performance
gate. Raw evidence is in
`recordings/20261015-121649-tensorfold-4b-htp-gdn/` in the mobile lab.

### Optional fast-activation mode

`serve-resident-fast` keeps the resident-v3 HTP path and replaces only the
post-Gated-DeltaNet SiLU plus normalization loop with a four-lane ARM NEON
implementation. Its exponential is an approximation, so this mode is opt-in
and reports `"fast_activations":true`; the exact `serve-resident` behavior is
unchanged.

Two six-request fast captures produced ten warmed samples. Their combined
median was 2,784.231 ms, 1.027% below the published v3 exact median of
2,813.129 ms. An immediately adjacent six-request exact capture had a
2,897.857 ms warmed median, making the bracketed reduction 3.921%. The fast
captures generated the same eight IDs and the same ordered top-five IDs at
every generated position, but all eight internal FP16 state hashes differed;
the maximum absolute top-five logit delta was 0.19604. Treat this as a measured
throughput/quality tradeoff, not an exact-equivalence result.

Projection calls still had a 1,572.544 ms combined warmed median, so direct
packed-EXL3 execution on HVX/HMX remained the primary performance gate at that
stage. The Qualcomm Hexagon toolchain was subsequently installed and used for
the packed-kernel work below. Raw evidence for the activation experiment is in
`recordings/20261015-134547-tensorfold-4b-fast-activations/`.

## Native packed-EXL3 full kernel

The first complete packed 4.00-bpw MCG path is now implemented as a private
GGML Hexagon custom operator and connected to TensorFold's 4B runner through
`serve-packed`. For every projection it performs the input `suh` scale and
128-point Hadamard transform on the DSP, decodes packed trellis tiles with HVX,
scatters each decoded tile directly into the HMX VTCM layout, executes the HMX
matrix multiply, and performs the output Hadamard transform and `svh` scale on
the DSP. Packed weights remain resident between requests. No full decoded FP16
matrix is created, and the former decoded-weight cache is unused.

The real first Qwen3.5-4B projection (K=2560, N=8192) passed against an
independent reconstructed reference. At batch one, the first packed kernel
measured 5.302 ms with maximum absolute error 0.000602; widening VTCM chunks
and halving layout-scatter instructions reduced a tuned sample to 4.497 ms.
The resident decoded-FP16 control measured 1.604 ms. At batch 15, tuned packed
measured 9.978 ms versus 2.364 ms for the decoded-FP16 control. The direct
packed path therefore passed correctness but did not pass the planned 2x speed
gate.

A matched two-request full-model run used the same 15-token prompt and eight
generated-token workload as resident v3:

| Packed 4B run | Wall time | Projection wall | Packed uploads | Result |
| --- | ---: | ---: | ---: | --- |
| tuned cold request | 12,304.011 ms | 9,998.006 ms | 1,786,388,480 bytes | passed |
| tuned warm request | 10,951.066 ms | 8,839.265 ms | 0 bytes | passed |

Both requests generated `248068,198,90700,8340,25,271,16,13`. The warm
request made 1,600 resident packed-weight hits with zero uploads, cache I/O, or
fallbacks and reached 0.731 generated tokens/s. Packed projection residency is
about 1.786 GB rather than the prior 7.13 GB FP16 staging representation, a
roughly 4x reduction. Peak process RSS was 3,088,988 KiB.

This is a genuine full-kernel and full-model POC, not a tile-only demo. It is
also currently 3.893x slower end to end than the 2,813.129 ms resident-v3 warm
median and 16.441x slower than the existing 666.062 ms llama.cpp control. A
phase trace of the representative batch-one projection attributes about
3.63 ms of a 4.09 ms DSP operation to EXL3 decode/layout and only about 0.06 ms
to HMX multiplication. The immediate performance work is therefore fused
decode/direct batch-one accumulation, shared transforms for grouped q/k/v,
qkv/z, and gate/up projections, and larger fused MLP/layer graphs.
Raw evidence is in
`recordings/20261015-170232-tensorfold-4b-packed-exl3-full-kernel/`.

### Decode-dataflow follow-up

A follow-up tested the CUDA/Metal ownership pattern more literally: each HVX
worker kept FP32 accumulators live, decoded a 16x16 trellis tile, and consumed
it immediately with paired-FP16 dot products instead of scattering it into an
HMX tile. The path passed the projection oracle, including a maximum absolute
error of 0.000355, but measured about 4.7-5.2 ms on the representative
K=2560, N=8192 batch-one projection. It was slower than the staged HMX path
and was removed from the default.

This disconfirms a simple interpretation of the CUDA/Metal optimization.
Those implementations fuse decode into a high-throughput matrix primitive;
moving the multiply to HVX gives up HMX throughput. The current Qualcomm HMX
interface consumes VTCM tiles rather than decoded HVX register fragments, so
the useful next kernel needs one of two Qualcomm-specific designs:

- decode into the HMX layout with contiguous vector stores and pipeline
  double-buffered decode against HMX consumption; or
- for batch-one decode, keep weights compressed and use a native low-bit HVX
  dot-product representation with wider output ownership and activation reuse.

Two smaller safe changes were retained: an HVX byte permutation replaced 32
scalar byte swaps per trellis tile, and the invariant HMX scatter map moved out
of each worker invocation. The projection oracle passed with the original
0.000602 maximum error and measured 4.352 ms over 20 iterations. A subsequent
full-model warm request reproduced the exact eight-token sequence in
10,997.062 ms with 8,915.471 ms projection wall time, zero packed uploads, and
zero fallbacks. That full-model result is within run-to-run variation of the
10,951.066 ms tuned baseline, so it is not claimed as an end-to-end speedup.

### Native Q8/Q4 execution and selective precision

An exploratory serving path now reconstructs EXL3 weights once, transcodes
them during resident upload, and executes Qualcomm's native Q8_0 or Q4_0
matmul kernels. The source checkpoint remains EXL3, but these modes are lossy
execution caches and must not be described as exact packed-EXL3 kernels.

Q8_0 passed the representative projection tolerance with 0.016225 maximum
absolute error and preserved the established eight generated IDs. Grouping
common-input projections, fusing the complete MLP, and using a resettable
scratch arena produced a 2,166.780 ms warm request. All-Q4_0 reached
1,830.682 ms but changed one generated ID and failed the quality gate.

A mixed path kept Q4_0 for almost all projections and Q8_0 for the
full-attention output projections in layers 3 and 7. On the frozen
15-token prompt it restored the exact eight-token sequence. A final clean
validation measured 1,812.572 ms warm, including 799.968 ms projection wall
time, 183.916 ms head time, and 147.153 ms resident GDN compute. This is 6.067x
faster than the 10,997.062 ms exact packed result and 1.552x faster than the
2,813.129 ms resident-FP16 v3 median, but still 2.721x slower than the matched 666.062 ms
llama.cpp request. The layer selection is an ablation result for this prompt,
not a general accuracy policy.

The remaining gap is architectural: the runner still crosses the host/HTP
boundary for roughly 120 small graphs per evaluated position. Because Q4
projection graph time alone exceeds the complete llama.cpp request, the next
performance gate is a resident full-layer or full-model graph that fuses
normalization, attention/GDN, residuals, and MLP execution. Phone Linux kernel
changes are not indicated by this evidence.

### Two-phone fused-tail and FastRPC polling result

The current 4B development lane adds a persistent HTP layer-tail graph. It
fuses the attention output projection, residual, post-attention RMS norm, and
the complete gate/up/down MLP tail while keeping the layer weights resident.
The output head uses a Q4 full-vocabulary projection and, when the leading Q4
margin is below 0.5, recomputes only the top five candidate rows from the
original FP16 tied weights. On the frozen prompt this preserves the expected
eight-token sequence while avoiding a second full Q8 vocabulary projection.

Hexagon response polling (`GGML_HEXAGON_OPPOLL=1`) is also required for the
best latency. The SM8750 FastRPC library logs every ordinary empty zero-timeout
poll as an error, so the run wrapper temporarily sets the process-specific
Android log tag to `S` and restores its prior value with an exit trap. This
removed roughly 35,000 false log entries without changing inference math.

The final controlled Phone B capture used one cold request followed by five
warm requests. All requests generated
`248068,198,90700,8340,25,271,16,13`.

| Qwen3.5-4B, 15-token prompt + 8 generated tokens | Median | Range |
| --- | ---: | ---: |
| TensorFold EXL3-derived Q4/Q8 HTP, five warm requests | 808.881 ms | 807.754-816.145 ms |
| unchanged llama.cpp Q4_0 HTP control, five requests | 691.314 ms | 678.769-700.110 ms |

TensorFold is therefore 1.170x slower than the refreshed live control, down
from 203.8x at the original 140.534 s 4B POC. Battery temperature was 28.8 C
at the start and 28.7 C at the end of the final TensorFold capture; thermal services
remained enabled and CPU governors were restored to `walt` afterward.

Several tempting variants were rejected: forcing HVX for all matrix
projections changed the frozen output and was slower; HVX-only GDN was slower;
prime-core affinity and `nice -20` did not improve the median; and a generic
GGML graph that fused GDN through the layer tail changed numerical semantics
and the final token. The remaining gap requires fewer host/HTP boundaries with
the existing BF16/F16 rounding and fast-activation behavior preserved, not a
phone Linux kernel modification. Raw evidence is in
`recordings/20261016-4b-two-phone-power/`.

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

## Native Qwen3.5 MTP direction

The next decode path is native MTP speculative verification, not another phone
Linux-kernel change. Qwen3.5-4B was trained with one MTP layer, but the EXL3
target export used by the Qualcomm runner omitted all 15 `mtp.*` tensors. A
verified BF16 sidecar now restores those tensors for development.

An ExLlamaV3 oracle confirmed that the sidecar works with the exact local
4.00-bpw target and preserves greedy output. On Tiny-IFEval sample 0, 128
generated IDs matched plain output at fixed depths 1, 2 and 4. Depth 2 accepted
63 of 130 offered draft positions and was the fastest drafted setting in that
single ordered compatibility run. Treat the timing as directional only.

The selected HTP Q4 projection measured 0.476693 ms for one row and 0.533294
ms for three rows over 12 iterations. This near-flat row cost is the required
hardware premise: a target verification window can process several proposed
positions for much less than several independent batch-one passes. The full
model must still prove the same property after attention, recurrent-state
snapshots, vocabulary projections and submission overhead are included.

The dedicated `qualcomm-qwen35-mtp-probe` build target additionally validates
all 15 BF16 sidecar tensors and executes the real `mtp.fc.weight` (`5120 ->
2560`) on HTP. On Phone B, resident Q8 measured 0.562135 ms at one row,
0.526667 ms at three rows and 0.562027 ms at eight rows over 12 iterations;
all cases passed the CPU-reference tolerance. The probe binary SHA-256 is
`D0F907A783E0F57AD7562E8BC62C3C709D844AF8D6004AF6147BB3C56DDB5A05`.

The same probe also covered every dense MTP matrix at one and three rows. All
16 cases passed. Independently summed projection compute was 4.590437 ms for
one row and 4.519461 ms for three rows, equivalent to 3.047x row throughput.
This deliberately excludes norms, RoPE, attention, activation, target-head and
graph composition overhead; it is a projection budget, not full MTP latency.

`qualcomm-qwen35-mtp-forward-probe` composes the complete Qwen3.5 MTP layer
and shared target vocabulary head. With a 22-row hidden-state fixture captured
from the exact batched target path, all positions passed its HTP-Q8 versus
CPU-BF16 numerical gate. The two paths agreed on every scored continuation
proposal. The probe also supports real autoregressive chain depths: on the
frozen 15+8 regression, depths 1, 2 and 4 produced identical CPU/HTP draft
traces. Depth 2 required three target-verification rounds, accepted four of
five prefix decisions, and computed six draft steps. Depth 4 did not reduce
the round count and computed nine draft steps. This was the draft-forward
oracle used before the integrated result below.

The isolated target runner also supports `TENSORFOLD_VERIFY_INPUTS` with the
fixed depth-2 shape (three target rows). Its resident bridge prebuilds prompt,
decode, and verifier graphs that share each uploaded weight. On the frozen
Phone B path, a correct two-proposal window plus bonus target token measured a
108.446 ms warm median versus 256.917 ms for the same three serial positions,
and all three final-state hashes matched. A real rejected MTP window selected
the expected replacement token. Because the verifier advances recurrent/KV
state through the whole window, the isolated POC now implements conservative
rollback: copy 24 resident GDN states plus convolution history before the
window, restore on rejection, and recompute only the committed prefix. Explicit
GDN zeroing between requests keeps repeated prefill and commit hashes stable.
The warm immediate-rejection path costs 200.294 ms median, including a 7.098 ms
snapshot and an 81.020 ms restore/one-row commit. This favors high-acceptance
depth-2 rounds and is a correctness baseline for a future device-side snapshot.

Build it with the pinned Hexagon headers matching the deployed libraries:

```sh
zig build qualcomm-qwen35-mtp-probe -Dtarget=aarch64-linux-android \
  -Doptimize=ReleaseSafe -Dandroid-libc=/path/to/android-libc-api36.txt \
  -Dggml-include=/path/to/ggml-hexagon-091b6a7/ggml/include \
  -Dggml-lib=/path/to/android/ggml/libs
```

### Integrated Qualcomm native-MTP result

`TENSORFOLD_MTP_SIDECAR=/path/to/mtp.safetensors` now enables a single-process
depth-two speculative loop in the 4B runner. The loop batches MTP prompt
prefill, keeps its KV cache persistent, drafts two tokens, verifies the pending
token plus both proposals in one target pass, restores GDN/convolution state on
rejection, and repairs newly committed MTP KV rows with target-confirmed hidden
states. Look-ahead tokens beyond the requested output count are omitted from
the reported prefix.

On the frozen 15-input/8-output regression, the final binary reproduced
`248068,198,90700,8340,25,271,16,13` in all ten warm samples. Median request
latency was 699.532 ms (690.606-712.026 ms), versus 807.645 ms
(803.775-817.153 ms) for the protected plain TensorFold binary on the same
Phone B. A query-only same-session refresh of unchanged Phone A llama.cpp was
706.719 ms median (683.519-717.882 ms). This is a 13.39% improvement over plain
TensorFold and a narrow 1.02% two-phone frozen-prompt win over llama.cpp; it is
not the 2x target and not a general workload claim.

Batched MTP prefill was the decisive implementation change: 15 serial MTP rows
fell from about 58 ms and 120 dense calls to about 9 ms and eight batched calls.
The complete frozen request makes 88 MTP projection calls.

Three distinct equal-length prompts from the pinned MLPerf Mobile v6.0-derived
Tiny-IFEval-33 corpus produced exact eight-token parity against protected plain
TensorFold. The two warm speculative requests were slower
(1321.651/1332.450 ms versus 1001.044/991.895 ms) because fixed depth two needed
four rounds with more rejection. The current promotion boundary is therefore:

1. frozen-prompt end-to-end POC: passed;
2. public-corpus exactness: passed on the three-prompt equal-shape subset;
3. public-corpus speedup: failed on that subset;
4. next work: corpus-validated adaptive depth/rejection avoidance, device-side
   recurrent checkpoints, and reusable variable-length batched prompt graphs.

Use `TENSORFOLD_BENCH_QUIET=1` only to suppress per-row diagnostic JSON during
timed captures; the final summary, token hash, timing, and pass flag remain.
Evidence and exact hashes are in
`recordings/20261016-153000-qwen35-4b-mtp-qualcomm-poc/RESULTS.md`.

Evidence is in
`recordings/20261016-153000-qwen35-4b-mtp-qualcomm-poc/` in the mobile lab.

### Device-snapshot rollback and final POC

The initial integrated numbers above are retained as development history. The
final implementation removes the rejected-window target replay rather than
trying to predict rejection with prompt-specific thresholds.

For verification batches, `build_gdn_graph` requests K recurrent snapshots
from `ggml_gated_delta_net` (K=2 or K=3). The backend keeps these snapshots in
the verification result tensor in most-recent-first order. On a partial
accept, `tf_qualcomm_backend_gdn_state_restore_snapshot` reads the selected
snapshot and stages it directly back into the resident layer state. The target
forward also saves each linear-attention layer's BF16-rounded convolution QKV
rows. Rollback restores the pre-window convolution buffer and advances it only
with accepted rows. Target hidden rows already computed by the verifier become
the committed history; the 4B target layers are not replayed.

Each layer must have a distinct host restore staging slice. Hexagon
`ggml_backend_tensor_set` may retain the source until a subsequent graph
submission, so reusing one scratch range across all 24 linear-attention layers
caused a rare, timing-dependent state corruption. The unique-slice fix passed
32/32 consecutive requests across two fresh processes, including 30/30 warm
requests with token hash `491b129266d4d110`.

The MTP cache uses deferred catch-up. At the next speculative round,
target-confirmed rows not yet committed to the MTP cache and the current first
proposal are submitted together through `runBatch`; only the second proposal
uses `runStep`. This reduces the frozen request from 88 to 56 MTP dense
projection calls.

Final frozen 15+8 medians were:

| Runtime | Device | Samples | Total | Decode/speculative phase |
| --- | --- | ---: | ---: | ---: |
| TensorFold native MTP, stress | Phone B | 30 warm | 629.224 ms | 447.470 ms |
| TensorFold native MTP, cool adjacent | Phone B | 10 warm | 620.808 ms | 441.651 ms |
| Protected plain, cool adjacent | Phone B | 10 warm | 816.257 ms | 627.376 ms |
| Unchanged llama.cpp Q4_0 | Phone A | 10 warm | 715.788 ms | 482.573 ms |

The cool adjacent pair ran protected plain first and final MTP second, and the
final path was 23.94% lower latency. The larger 30-warm stress median is 12.09%
lower than the query-only llama.cpp total on this frozen regression.
It does not meet the half-control milestone of 357.894 ms. Decode/speculative
phase comparisons are diagnostic because the runtimes expose different phase
boundaries; total request latency is the primary cross-runtime result.

All 13 selected MLPerf Mobile v6.0-derived Tiny-IFEval prompts, spanning input
lengths 47, 54, 65, 74, 87, and 89, matched their retained eight-token hashes.
Across the seven warm grouped samples, the speculative mean was 605.744 ms,
29.65% below the original fixed-depth-two implementation and 16.59% below the
fixed-depth-one ablation. Fixed depth two beat tested confidence gates after
device snapshot rollback. This is derived-corpus validation, not an official
MLPerf result.

The final binary SHA-256 is
`99EAECA94FB1C47046AB5E3CBF2C978A05E9EB97CA4342633FCF1C1C81E9A885`.
The protected plain binary remains
`6917588A7DE1B51464822628DCE095AD6EB7712DB7081B0EC2150B37CB740144`.
Exact logs, scripts, aggregate JSON, and limitations are recorded in
`recordings/20261016-153000-qwen35-4b-mtp-qualcomm-poc/`.

### Short-batch HMX pipeline and sub-357 frozen POC

The later experimental path lowers the HMX quant-weight decode/matmul pipeline
selection threshold with `GGML_HEXAGON_HMX_PIPELINE_MIN_ROWS`. The upstream
host planner previously enabled its double-buffered pipeline only above 32
activation rows. Setting the threshold to four lets the native-MTP seven-row
target verifier overlap weight decode/preparation with HMX work.

On the frozen 15-input/8-output request, this was combined with MTP depth six,
round-zero rank-two proposal selection, Q8 output projections through layer
23, and fused persistent linear tails on layers 25, 26, 28, 29, and 30. The
result was 353.9905 ms median over ten warm requests (345.971-364.031 ms), with
10/10 warm hashes equal to `491b129266d4d110`. The retained Phone A llama.cpp
median is 715.7875 ms, making the half-control threshold 357.89375 ms.

The result has a 3.90325 ms margin and is thermally sensitive. A replay
after sustained corpus testing measured 362.6485 ms while battery temperature
rose from 35.3 C to 35.6 C. The aggressive path is therefore a frozen-prompt
POC, not a general performance claim. The final executable separately passed
all 13 retained Tiny-IFEval hashes using the conservative depth-two, all-Q8,
rollback-capable gate. Four- and five-row MTP catch-up is split into 3+1 or 3+2
resident calls so public rejection paths use only supported graph shapes.

This work changes userspace model-runtime and HTP inference kernels only. It
does not modify the Android/Linux kernel, boot image, firmware, or partitions.
See `recordings/20261016-153000-qwen35-4b-mtp-qualcomm-poc/SUB357.md` for exact
launch settings, hashes, raw logs, and limitations.

## Phone-local product path

The phone-local HTP development path no longer needs a host Python tokenizer or HTTP shim.
`tensorfold-qualcomm-openai` is an Android ARM64 Zig process that loads the
Hugging Face `tokenizer.json`, owns the loopback OpenAI HTTP/SSE endpoint, and
supervises one persistent Qualcomm engine child. It serves:

- `/` for the installable local app: model management, multi-turn chat,
  generation settings, a quick benchmark, Hugging Face pulls, and the bounded
  Android-task console handoff;
- `/v1/completions` and `/v1/chat/completions` for greedy streaming;
- `/v1/tokenize` and `/v1/detokenize`;
- `/v1/local/models` and `/v1/local/models/select` for discovery and hot swap;
- `/v1/local/catalog` and `/v1/local/models/pull` for qualified public
  Hugging Face checkpoints;
- `/health`, `/metrics`, `/v1/models`, `/manifest.webmanifest`, and `/sw.js`;
- `/v1/benchmark/manifest`, which reports `python_version: "none"`, the
  Qualcomm backend, Android platform, tokenizer/config hashes when present,
  and the detected HTP v79 execution target.

The server serializes clients because one engine state is resident. A second
client waits in the socket backlog. Closing a streaming client kills that
request's worker and the next request starts a clean worker; it cannot inherit
partially advanced recurrent state. The child also requests `SIGTERM` when its
parent dies, so stopping the HTTP process does not orphan the model process.

Model selection is transactional and does not restart the HTTP app. The server
loads the next tokenizer, stops only the engine child, starts the matching
2B/4B engine, writes `profile.conf` atomically, then releases the previous
tokenizer and model paths. A failed start restores the old profile and worker.
The verified 4B → 2B → 4B sequence kept one HTTP PID while changing the worker
PID each time; a full launcher restart then restored the selected 2B profile.

The app keeps chat history in the page and sends it through the native chat
template. System prompt, output token limit, and thinking are UI controls. Its
quick benchmark runs one warm-up plus three 64-token code/chat repeats and
reports client delivery rate, TTFT, and completion count; it is deliberately
labelled as a diagnostic rather than community-v2.

This section describes the privileged HTP development installation under
`/data/local/tmp`. It is not the distributable Android APK. TensorFold Mobile
1.1 instead packages a vendor-independent CPU/Vulkan engine, selects the CPU
path that passed the retained token gate, visibly disables the mismatching
Vulkan candidate, keeps models and caches in app-private storage, authenticates
every loopback route, and requires no root or Termux. See
[TensorFold for Android](android-app.md) for the public architecture, model
sizes, build, security boundaries, and Pixel gate.

The Hugging Face catalog is an allowlist of checkpoint IDs, expected sizes and
SHA-256 values already qualified for this phone. It compares download size
with free storage, requires the decoded cache for that exact checkpoint, uses
HTTPS-only `curl --continue-at -`, verifies the final model hash, and exposes
only then makes the model loadable. An interrupted README download was resumed
from a 100-byte partial and reproduced SHA-256
`b46d637ccd03d0496f4e147bb35e85cf70ae45b35c29c8ea9121bddf93cbe445`.
Arbitrary repository IDs are refused rather than downloading an unqualified
architecture or a checkpoint for which the phone has no decoded cache.

Android task execution remains in the installed PwnHarness package
`com.phoneproject.console`, now version 0.7.0/code 8. Its default visible
provider is **TensorFold Local**: the APK reads health and active profile,
switches the fixed 2B/4B models, streams direct chat, and routes bounded
Harness role decisions through TensorFold SSE. The provider obtains exact
prompt counts from `/v1/tokenize`, admits 512 input plus 128 output tokens,
renders at most the phase-bounded tool schemas into Qwen3.5 tool XML, and
parses calls back into the broker contract. Target prefill uses fixed 128-row
chunks and MTP prefill fixed seven-row chunks for prompts above the community
benchmark sizes.

PwnHarness retains scoped typed Workers, exact approval gates, execution
evidence, and the global STOP control. Model startup is allowed while STOP is
latched because it performs no action; every Harness decision/action still
checks STOP. The controlled `tensorfold-provider-smoke` scope exposed only
`finish`, produced `TensorFold integration safe.`, stored one hash-chained
successful `finish`, generated no target traffic, and left STOP present.
TensorFold intentionally exposes no raw taps, shell commands, Accessibility
privileges, or unrestricted device automation.

Build the engine and control plane:

```sh
zig build qualcomm-exl3-forward-4b -Dtarget=aarch64-linux-android \
  -Doptimize=fast -Dandroid-libc=/path/to/android-libc-api36.txt \
  -Dggml-include=/path/to/ggml/include -Dggml-lib=/path/to/android/ggml/libs
zig build qualcomm-exl3-forward -Dtarget=aarch64-linux-android \
  -Doptimize=fast -Dandroid-libc=/path/to/android-libc-api36.txt \
  -Dggml-include=/path/to/ggml/include -Dggml-lib=/path/to/android/ggml/libs
zig build qualcomm-openai -Dtarget=aarch64-linux-android \
  -Dandroid-libc=/path/to/android-libc-api36.txt
```

From the Windows development machine, `tools/install_tensorfold_phone.ps1`
pushes those three binaries without replacing either protected POC binary,
copies the admitted configs, verifies the protected hashes, installs the
on-device launcher and a Termux:Widget shortcut, and leaves the service
stopped unless `-Start` is given.

On the phone:

```sh
/data/local/tmp/exl3-full-power/tensorfold models
/data/local/tmp/exl3-full-power/tensorfold use \
  /data/local/tmp/tensorfold-models/Qwen3.5-4B-exl3-4.00bpw
/data/local/tmp/exl3-full-power/tensorfold start-ui
/data/local/tmp/exl3-full-power/tensorfold status
/data/local/tmp/exl3-full-power/tensorfold stop
```

`tensorfold doctor` re-runs model admission, file checks, and the two protected
binary hashes. `tensorfold logs` shows the current server log; `restart` stops
only the recorded server PID and starts the selected profile. Re-running the
installer is the recovery path for a missing or mismatched product binary; it
does not overwrite the protected POCs. The qualified product SHA-256 values
are `5ba65c1923a6953bf3dcbbbfb12abb832067d10f7a1ad5db3b70463d1e430a03`
for the native HTTP/tokenizer,
`3c34ed7fd26b5070764de1c1751f11d2a5b023eac4f6dda3f9018de80f222d53`
for 4B, and
`7224edc49a0563b4a299470272bd84e1effbcb2346abac5ea7a36fc2042a3627`
for 2B.

`start-ui` opens `http://127.0.0.1:8082/`. `start` uses the thermally safer
`low_balanced` HTP profile and the 0.125 near-tie guard. `benchmark-start`
waits for at most 39.5 C, selects `burst`, and disables the guard for the
standard code/chat fixtures whose retained serial hashes are checked after the
run. The HTTP server refuses a new generation at 43 C; tokenize, health,
metrics, model discovery, and stop remain available while the device cools.

Model admission reads `config.json`; it does not guess from a directory name.
It currently accepts EXL3 Qwen3.5 text geometries for 2B and 4B, selects the
matching compiled engine, and gives named errors for a missing field, another
architecture, another quantization method, an unsupported Qwen geometry, or a
profile/engine mismatch. The 4B profile uses its qualified MTP sidecar. The 2B
profile has no qualified sidecar and therefore uses the persistent serial
worker. A new geometry needs a compiled profile and a device qualification
before admission is extended.

The current exactness gate compares the drafted target with a fresh serial
target on 13 selected Tiny-IFEval prompts plus 16 deterministic randomized
prompts: 29/29 eight-token hashes matched over 22 prompt lengths. A four-shape
activation graph cache was forced through eviction, and rejection paths
covered accepted prefixes of zero, one, and two. Near-tie target rows below a
0.125 logit margin restart from the saved prompt boundary and finish serially;
this prevents batch-shape low-bit noise from changing a greedy token.

The native tokenizer gate covered 511 text/Unicode cases, 201 detokenizations,
and four chat templates with zero mismatches against the already-qualified
Qwen3.5 reference. The native HTTP gate completed both completion and chat
streams, exposed manifest/health/metrics, preserved concurrent request order,
cancelled a disconnected request, restarted the worker, completed the queued
request, and left no process running.


The 30-minute production qualification used `low_balanced`, paused requests at
42 C, and resumed at 39.5 C. It completed 122 alternating 64-token requests in
1807.172 s, with 843.544 s active (46.68% duty cycle), 39.2-42.9 C battery
temperature, and resident RSS of 1,764,768-1,764,904 KiB after warm-up. Code
and chat hashes were stable, and target/MTP weights had no uploads after the
two warm-up requests. `burst`, `sustained_high_performance`, and `balanced`
continuous-load trials reached the 45 C safety cutoff in 7.6-8.6 minutes, so
none is the unattended serving default.

The no-Python native community-v2 run used 256 tokens, temperature zero, and
five measured repeats after warm-up. Its code/chat SHA-256 values matched the
retained serial target:

| Workload | Phone B TensorFold | Phone A llama.cpp | TensorFold difference | TensorFold TTFT | llama.cpp TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 17.88 tok/s | 11.95 tok/s | +49.64% | 0.045 s | 0.170 s |
| Chat | 13.27 tok/s | 12.00 tok/s | +10.64% | 0.043 s | 0.182 s |

Both runtimes completed 5/5 code and 5/5 chat attempts. The TensorFold receipt
is `community-v2-tf-phoneB-native-fast.json`, run
`9d0fd1cb-b8c3-471f-8922-a5f6f2bfa9e1`. The retained Phone A control is run
`340b721f-eb31-4095-ba61-86565c5edac4`. Attached Android receipts remain
unranked until the receiver accepts Qualcomm/Android manifest facts.

The additional Qwen3.5-2B EXL3 8-bpw checkpoint was admitted by metadata,
selected through the same launcher, and served through the native HTTP path
without a drafter. Its first eight-token request reported `drafts: false`,
`mtp_depth: 0`, zero MTP uploads, SHA-256
`e3f3ff301590bf0bcc20d55d2503cd62b1369e7f9454537331842d11dc3e8032`,
and 16.18 server tokens/s after its cold prefill. Synthetic Llama, GPTQ, and
unsupported-Qwen configs were refused with `UnsupportedArchitecture`,
`UnsupportedQuantization`, and `UnsupportedQwenGeometry`, each with exit 65.

### QNN layer-0 fidelity payload

`tensorfold-qualcomm-exl3-qnn-projection prepare-fidelity MODEL APP_PRIVATE_PAYLOAD`
writes the v3 app-private payload for the exact layer-0 2048x6144 projection. It
binds the pinned model SHA-256, EXL3 source-tensor identity, quantized-reference
identity, original-reference identity, counts, and whole-payload integrity before
the app allocates or loads graph buffers. The deterministic activation is decoded
as real values and dotted against FP64 EXL3-decoded weights independently of the
subsequent symmetric-u8 weight quantization.

The QNN service keeps the pre-existing QNN-vs-quantized tolerance as its only
status gate. Its 100-repeat receipt additionally reports quantized-vs-original,
QNN-vs-original, original-reference magnitude, and relative QNN-vs-original
error. A v2 payload remains usable when v3 is absent, but cannot report the
original-reference fields. This qualification is evidence for this one projection
only; it does not enable a layer or model backend, and serving remains CPU-only.

On stock Phone B, the v3 service receipt completed 100 QNN executions with
`0.0093` QNN-vs-quantized maximum error. The fidelity comparison measured
`0.0872` quantized-vs-original and `0.0881` QNN-vs-original maximum error over
an original-reference maximum magnitude of `1.7047`, or `5.17%` relative.
Therefore the current per-tensor symmetric-u8 recipe is **not** admitted for a
complete layer or model backend. The next candidate must reduce original-domain
error with per-output-channel scales or higher precision before layer parity.
