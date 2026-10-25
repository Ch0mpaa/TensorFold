package expo.modules.tensorfoldlocalruntime;

final class TensorG5Native {
  static {
    System.loadLibrary("tensorfold_tensor_g5_probe");
  }

  private TensorG5Native() {}

  // Returns [status, fullyAccelerated], where zero/one is the admitted pair.
  static native long[] nativeAttest(long compiledModelHandle);
}
