package expo.modules.tensorfoldlocalruntime

import android.content.Context

internal object QnnProbe {
  private var cached: Map<String, Any?>? = null

  @JvmStatic private external fun nativeCalculator(nativeDir: String, stubName: String, count: Int): Int

  @Synchronized
  fun inspect(context: Context, socModel: String): Map<String, Any?> {
    cached?.let { return it }
    val architecture = when {
      socModel.equals("SM8750", ignoreCase = true) -> "v79"
      else -> null
    }
    if (architecture == null) return result(false, null, "No packaged QNN probe matches this SoC.").also { cached = it }
    val stub = "libQnnHtpV79CalculatorStub.so"
    val answer = try {
      System.loadLibrary("tensorfold_qnn_probe")
      nativeCalculator(context.applicationInfo.nativeLibraryDir, stub, 16)
    } catch (failure: Throwable) {
      return result(false, architecture, failure.javaClass.simpleName).also { cached = it }
    }
    return result(answer == 0, architecture, if (answer == 0) null else "QNN calculator returned $answer").also { cached = it }
  }

  private fun result(available: Boolean, architecture: String?, reason: String?) = linkedMapOf<String, Any?>(
    "available" to available,
    "architecture" to architecture,
    "probe" to "qnn-calculator-v1",
    "reason" to reason,
  )
}
