package expo.modules.tensorfoldlocalruntime;

import android.content.Context;
import android.os.SystemClock;
import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.NpuCompatibilityChecker;
import com.google.ai.edge.litert.TensorBuffer;
import java.util.Arrays;
import java.util.List;

final class TensorG5LiteRtRunner {
  private static final int INPUT_FLOATS = 256 * 256 * 3;

  static final class Result {
    final long nativeStatus;
    final boolean fullyAccelerated;
    final double firstExecuteMs;
    final double warmMedianMs;
    final double warmLastMs;
    final double maxAbsoluteError;

    Result(
        long nativeStatus,
        boolean fullyAccelerated,
        double firstExecuteMs,
        double warmMedianMs,
        double warmLastMs,
        double maxAbsoluteError) {
      this.nativeStatus = nativeStatus;
      this.fullyAccelerated = fullyAccelerated;
      this.firstExecuteMs = firstExecuteMs;
      this.warmMedianMs = warmMedianMs;
      this.warmLastMs = warmLastMs;
      this.maxAbsoluteError = maxAbsoluteError;
    }
  }

  private TensorG5LiteRtRunner() {}

  static boolean isDeviceSupported() {
    return NpuCompatibilityChecker.Companion.getGoogleTensor().isDeviceSupported();
  }

  static Result run(Context context, String modelPath) throws Exception {
    final float[] cpuOutput;
    try (CompiledModel cpu =
        CompiledModel.create(modelPath, new CompiledModel.Options(Accelerator.CPU))) {
      cpuOutput = execute(cpu);
    }

    BuiltinNpuAcceleratorProvider provider = new BuiltinNpuAcceleratorProvider(context);
    try (Environment environment = Environment.create(context, provider);
        CompiledModel npu =
            CompiledModel.create(
                modelPath, new CompiledModel.Options(Accelerator.NPU), environment)) {
      long handle =
          npu.getHandle$third_party_odml_litert_litert_kotlin_litert_api_for_maven();
      long[] attestation = TensorG5Native.nativeAttest(handle);
      if (attestation == null || attestation.length != 2) {
        return unavailable(-4);
      }
      long nativeStatus = attestation[0];
      boolean fullyAccelerated = attestation[1] == 1;
      if (nativeStatus != 0 || !fullyAccelerated) {
        return unavailable(nativeStatus == 0 ? -5 : nativeStatus);
      }

      List<TensorBuffer> input = npu.createInputBuffers();
      List<TensorBuffer> output = npu.createOutputBuffers();
      try {
        fillZeroInput(input);
        npu.run(input, output);
        double[] elapsed = new double[5];
        for (int index = 0; index < elapsed.length; index++) {
          long start = SystemClock.elapsedRealtimeNanos();
          npu.run(input, output);
          elapsed[index] = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0;
        }
        float[] npuOutput = flatten(output);
        double[] warm = Arrays.copyOfRange(elapsed, 1, elapsed.length);
        Arrays.sort(warm);
        return new Result(
            0,
            true,
            elapsed[0],
            (warm[1] + warm[2]) / 2.0,
            elapsed[elapsed.length - 1],
            maxAbsoluteError(cpuOutput, npuOutput));
      } finally {
        closeAll(input);
        closeAll(output);
      }
    }
  }

  private static Result unavailable(long status) {
    return new Result(status, false, 0.0, 0.0, 0.0, Double.NaN);
  }

  private static float[] execute(CompiledModel model) throws Exception {
    List<TensorBuffer> input = model.createInputBuffers();
    List<TensorBuffer> output = model.createOutputBuffers();
    try {
      fillZeroInput(input);
      model.run(input, output);
      return flatten(output);
    } finally {
      closeAll(input);
      closeAll(output);
    }
  }

  private static void fillZeroInput(List<TensorBuffer> input) throws Exception {
    if (input.size() != 1) {
      throw new IllegalStateException("The Tensor G5 probe model must have exactly one input.");
    }
    input.get(0).writeFloat(new float[INPUT_FLOATS]);
  }

  private static float[] flatten(List<TensorBuffer> buffers) throws Exception {
    float[][] chunks = new float[buffers.size()][];
    int length = 0;
    for (int index = 0; index < buffers.size(); index++) {
      chunks[index] = buffers.get(index).readFloat();
      length = Math.addExact(length, chunks[index].length);
    }
    float[] flattened = new float[length];
    int offset = 0;
    for (float[] chunk : chunks) {
      System.arraycopy(chunk, 0, flattened, offset, chunk.length);
      offset += chunk.length;
    }
    return flattened;
  }

  private static void closeAll(List<TensorBuffer> buffers) {
    for (TensorBuffer buffer : buffers) buffer.close();
  }

  private static double maxAbsoluteError(float[] cpu, float[] npu) {
    if (cpu.length == 0 || cpu.length != npu.length) return Double.NaN;
    double maximum = 0.0;
    for (int index = 0; index < cpu.length; index++) {
      if (!Float.isFinite(cpu[index]) || !Float.isFinite(npu[index])) return Double.NaN;
      maximum = Math.max(maximum, Math.abs((double) cpu[index] - npu[index]));
    }
    return maximum;
  }
}
