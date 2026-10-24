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

The calculator runs prove rootless QNN DSP/HTP transport, and the v79 `MatMul` proves a real QNN graph can execute correctly from the ordinary Phone B app UID. Neither proves TensorFold model inference, retained-token parity, production performance, or full NPU-only execution. The model backend remains CPU until checkpoint-derived projections and the autoregressive path pass their frozen CPU oracles and backend attestation is included in each receipt.
