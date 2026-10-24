# Rootless Qualcomm QNN capability gate

QAIRT: `2.50.0.260828`, fetched through the official `qairt-dev`/`qairt-vm` distribution.

## Phone B — Snapdragon 8 Elite

Device: OnePlus CPH2655, `SM8750`, stock app sandbox for the final probe.

The official `qnn-platform-validator` first passed from the unprivileged shell after staging the matching v79 calculator stub and skeleton:

```text
Backend DSP Prerequisites: Present.
Core Version of the backend DSP: Hexagon Architecture V79
Success in executing the sum function
Unit Test on the backend DSP: Passed.
QNN is supported for backend DSP on the device.
```

TensorFold Mobile then packaged the QAIRT v79 runtime files, declared the OEM-public `libcdsprpc.so` and `libadsprpc.so`, and invoked `Qnn_calculatorTest(16)` through `libtensorfold_qnn_probe.so` from the ordinary app process. The visible app result was:

```text
QNN HTP v79 passed the ordinary-app calculator probe. TensorFold model projections remain on CPU until their QNN graph passes retained-token parity.
```

Verified release APK SHA-256: `17de5269161d276754c2bd34e5e115c79b63c77172e091b2bc3d5daa859b569a`.

### Real QNN projection gate

The next release APK constructed and executed a quantized `[1,4] x [4,3]`
`MatMul` graph through the QNN HTP v79 backend from the same ordinary app UID.
The first retained attempt failed closed at graph creation because
`libQnnHtpPrepare.so` was not packaged. After adding that required QAIRT graph
compiler component, the visible app result was:

```text
QNN HTP v79 projection MatMul passed (2.19 ms, max error 0.000). TensorFold model inference remains on CPU pending retained-token parity.
```

Verified projection APK SHA-256: `c00adcd94af61600af77753ce00125b6097b0fa2f5e53a5fd6a37d7fa97ddd1b`.

### Checkpoint-derived QNN projection

The following release retained the synthetic gate and added a separately
reported quantized projection. Its 4x3 weight slice is deterministically
derived from `zig/tests/fixtures/exl3_oracle.bin`, whose real 128x128 EXL3 block
comes from the public `UnstableLlama/Qwen3.5-2B-exl3-8.00bpw` checkpoint. The
source header records the fixture hash, full model hash, block coordinates,
quantization parameters and independent CPU dot products. I independently
decoded the fixture's FP64 values and reproduced all 12 quantized bytes before
the device run.

```text
QNN HTP v79 synthetic projection MatMul passed (1.90 ms, max error 0.000). Checkpoint projection passed (1.79 ms, max error 0.0001). TensorFold model inference remains on CPU pending retained-token parity.
```

Verified dual-projection APK SHA-256: `b68541a476dd263054d3d4e8c9c8c965c82b5d3debf6b2eb98129994ccf35441`.

### Complete 128x128 EXL3 oracle block

`tools/generate_qnn_full_projection.py` independently parses the checked-in
fixture's complete FP64 oracle block, validates its SHA-256 and coordinates,
quantizes all 16,384 weights, computes 128 integer-domain CPU dot products, and
emits a content-addressed bounded native header. Regeneration and its two
stdlib tests passed before the release build. From the stock app UID:

```text
QNN HTP v79 synthetic projection MatMul passed (1.91 ms, max error 0.000). Checkpoint slice passed (2.01 ms, max error 0.0001). Full 128x128 EXL3 block passed (1.96 ms, max error 0.0009). TensorFold model inference remains on CPU pending retained-token parity.
```

Verified full-block APK SHA-256: `88102fc211b7e4a801bbb95b7b75a32ecbcebcfa71751fa16e57987e74bf697e`.

## Samsung S20 Ultra — Snapdragon 865

Device: `SM-G988U`, `SM8250`, stock Android 12/API 31.

The official unprivileged shell validator selected the legacy DSP path and passed after staging the matching v66 files:

```text
Backend DSP Prerequisites: Present.
Core Version of the backend DSP: Hexagon Architecture V66
Success in executing the sum function
Unit Test on the backend DSP: Passed.
QNN is supported for backend DSP on the device.
```

The current APK packages only the v79 app-UID probe, so Samsung app-UID QNN remains unqualified.

## Claim boundary

The calculator proves rootless QNN DSP/HTP transport; the synthetic, checkpoint slice and complete 128x128 block prove real QNN graphs can execute correctly from the ordinary Phone B app UID. They do not prove full TensorFold model inference, retained-token parity, production throughput, or full NPU-only execution. The selected model backend remains CPU until full model layers and the autoregressive path pass their frozen CPU oracles and backend attestation is included in each receipt.
