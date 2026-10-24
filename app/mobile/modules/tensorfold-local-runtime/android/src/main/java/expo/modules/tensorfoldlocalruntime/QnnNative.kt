package expo.modules.tensorfoldlocalruntime

internal object QnnNative {
  init { System.loadLibrary("tensorfold_qnn_probe") }

  @JvmStatic external fun create(nativeDir: String, payloadPath: String): DoubleArray
  @JvmStatic external fun execute(handle: Long, repeats: Int): DoubleArray
  @JvmStatic external fun release(handle: Long): Int
  @JvmStatic external fun health(): LongArray
  @JvmStatic external fun releaseAll()
}
