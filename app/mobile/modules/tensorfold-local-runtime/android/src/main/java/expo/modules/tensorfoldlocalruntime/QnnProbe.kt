package expo.modules.tensorfoldlocalruntime

import android.content.Context

internal object QnnProbe {
  private var cached: Map<String, Any?>? = null

  @JvmStatic private external fun nativeProjection(nativeDir: String, stubName: String): DoubleArray
  @JvmStatic private external fun nativeCheckpointProjection(nativeDir: String, stubName: String): DoubleArray

  @Synchronized
  fun inspect(context: Context, socModel: String): Map<String, Any?> {
    cached?.let { return it }
    val architecture = when {
      socModel.equals("SM8750", ignoreCase = true) -> "v79"
      else -> null
    }
    if (architecture == null) return result(false, null, unavailable("No packaged QNN projection matches this SoC."), unavailable("No packaged checkpoint projection matches this SoC.")).also { cached = it }
    val nativeDir = context.applicationInfo.nativeLibraryDir
    val (projection, checkpoint) = try {
      System.loadLibrary("tensorfold_qnn_probe")
      Pair(
        QnnProjectionResult.fromNative(
          nativeProjection(nativeDir, "libQnnHtpV79Stub.so"),
          QnnProjectionResult.syntheticTolerance,
        ),
        QnnProjectionResult.fromNative(nativeCheckpointProjection(nativeDir, "libQnnHtpV79Stub.so")),
      )
    } catch (failure: Throwable) {
      Pair(
        QnnProjectionResult(-100, 0.0, Double.NaN, QnnProjectionResult.syntheticTolerance),
        QnnProjectionResult(-100, 0.0, Double.NaN),
      )
    }
    return result(projection.passed, architecture, projection, checkpoint).also { cached = it }
  }

  private fun unavailable(reason: String) = linkedMapOf<String, Any?>(
    "available" to false, "elapsedMs" to 0.0, "maxAbsoluteError" to Double.NaN,
    "tolerance" to QnnProjectionResult.checkpointTolerance, "reason" to reason,
  )

  private fun detail(projection: QnnProjectionResult) = linkedMapOf<String, Any?>(
    "available" to projection.passed,
    "elapsedMs" to projection.elapsedMs,
    "maxAbsoluteError" to projection.maxAbsoluteError,
    "tolerance" to projection.tolerance,
    "reason" to projection.reason(),
  )

  private fun result(available: Boolean, architecture: String?, projection: QnnProjectionResult, checkpoint: QnnProjectionResult) = linkedMapOf<String, Any?>(
    "available" to available,
    "architecture" to architecture,
    "probe" to "qnn-htp-projection-v1",
    "elapsedMs" to projection.elapsedMs,
    "maxAbsoluteError" to projection.maxAbsoluteError,
    "tolerance" to 0.5,
    "reason" to projection.reason(),
    "checkpointProjection" to detail(checkpoint),
  )

  private fun result(available: Boolean, architecture: String?, projection: Map<String, Any?>, checkpoint: Map<String, Any?>) = linkedMapOf<String, Any?>(
    "available" to available, "architecture" to architecture, "probe" to "qnn-htp-projection-v1",
    "elapsedMs" to projection["elapsedMs"], "maxAbsoluteError" to projection["maxAbsoluteError"],
    "tolerance" to 0.5, "reason" to projection["reason"], "checkpointProjection" to checkpoint,
  )
}
