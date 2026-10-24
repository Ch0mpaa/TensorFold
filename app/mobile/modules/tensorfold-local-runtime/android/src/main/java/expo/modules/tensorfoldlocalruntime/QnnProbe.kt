package expo.modules.tensorfoldlocalruntime

import android.content.Context

internal object QnnProbe {
  private var cached: Map<String, Any?>? = null

  @JvmStatic private external fun nativeProjection(nativeDir: String, stubName: String): DoubleArray

  @Synchronized
  fun inspect(context: Context, socModel: String): Map<String, Any?> {
    cached?.let { return it }
    val architecture = when {
      socModel.equals("SM8750", ignoreCase = true) -> "v79"
      else -> null
    }
    if (architecture == null) return result(false, null, QnnProjectionResult(-101, 0.0, Double.NaN), "No packaged QNN projection matches this SoC.").also { cached = it }
    val projection = try {
      System.loadLibrary("tensorfold_qnn_probe")
      QnnProjectionResult.fromNative(nativeProjection(context.applicationInfo.nativeLibraryDir, "libQnnHtpV79Stub.so"))
    } catch (failure: Throwable) {
      QnnProjectionResult(-100, 0.0, Double.NaN)
    }
    return result(projection.passed, architecture, projection, projection.reason()).also { cached = it }
  }

  private fun result(available: Boolean, architecture: String?, projection: QnnProjectionResult, reason: String?) = linkedMapOf<String, Any?>(
    "available" to available,
    "architecture" to architecture,
    "probe" to "qnn-htp-projection-v1",
    "elapsedMs" to projection.elapsedMs,
    "maxAbsoluteError" to projection.maxAbsoluteError,
    "tolerance" to QnnProjectionResult.tolerance,
    "reason" to reason,
  )
}
