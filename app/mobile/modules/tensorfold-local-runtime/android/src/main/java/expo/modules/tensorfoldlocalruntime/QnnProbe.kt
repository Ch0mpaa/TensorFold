package expo.modules.tensorfoldlocalruntime

import android.content.Context

internal object QnnProbe {
  private data class ProbeSet(
    val projections: List<QnnProjectionResult>,
    val persistent: QnnPersistentProjectionResult,
  )

  private var cached: Map<String, Any?>? = null

  @JvmStatic private external fun nativeProjection(nativeDir: String, stubName: String): DoubleArray
  @JvmStatic private external fun nativeCheckpointProjection(nativeDir: String, stubName: String): DoubleArray
  @JvmStatic private external fun nativeFullProjection(nativeDir: String, stubName: String): DoubleArray
  @JvmStatic private external fun nativeLiveCheckpointProjection(
    nativeDir: String,
    stubName: String,
    payloadPath: String,
  ): DoubleArray
  @JvmStatic private external fun nativeLiveFullProjection(
    nativeDir: String,
    stubName: String,
    payloadPath: String,
  ): DoubleArray

  @JvmStatic private external fun nativePersistentLiveFullProjection(
    nativeDir: String,
    stubName: String,
    payloadPath: String,
  ): DoubleArray

  @Synchronized
  fun inspect(context: Context, socModel: String): Map<String, Any?> {
    cached?.let { return it }
    val architecture = if (socModel.equals("SM8750", ignoreCase = true)) "v79" else null
    if (architecture == null) {
      return result(
        false,
        null,
        unavailable("No packaged QNN projection matches this SoC.", QnnProjectionResult.syntheticTolerance),
        unavailable("No packaged checkpoint projection matches this SoC.", QnnProjectionResult.checkpointTolerance),
        unavailable("No packaged full-block projection matches this SoC.", QnnProjectionResult.fullBlockTolerance),
        unavailable("No packaged live checkpoint projection matches this SoC.", QnnProjectionResult.fullBlockTolerance),
        unavailable("No prepared full projection matches this SoC.", QnnProjectionResult.liveFullTolerance),
        persistentUnavailable("No prepared persistent full projection matches this SoC."),
      ).also { cached = it }
    }
    val nativeDir = context.applicationInfo.nativeLibraryDir
    val payloadPath = context.filesDir.resolve("tensorfold/qnn/live-checkpoint-v1.bin").absolutePath
    val fullPayloadPath = context.filesDir.resolve("tensorfold/qnn/live-full-projection-v2.bin").absolutePath
    val probes = try {
      System.loadLibrary("tensorfold_qnn_probe")
      ProbeSet(
        listOf(
          QnnProjectionResult.fromNative(nativeProjection(nativeDir, "libQnnHtpV79Stub.so"), QnnProjectionResult.syntheticTolerance),
          QnnProjectionResult.fromNative(nativeCheckpointProjection(nativeDir, "libQnnHtpV79Stub.so")),
          QnnProjectionResult.fromNative(nativeFullProjection(nativeDir, "libQnnHtpV79Stub.so"), QnnProjectionResult.fullBlockTolerance),
          QnnProjectionResult.fromNative(nativeLiveCheckpointProjection(nativeDir, "libQnnHtpV79Stub.so", payloadPath), QnnProjectionResult.fullBlockTolerance),
          QnnProjectionResult.fromNative(nativeLiveFullProjection(nativeDir, "libQnnHtpV79Stub.so", fullPayloadPath), QnnProjectionResult.liveFullTolerance),
        ),
        QnnPersistentProjectionResult.fromNative(nativePersistentLiveFullProjection(nativeDir, "libQnnHtpV79Stub.so", fullPayloadPath)),
      )
    } catch (failure: Throwable) {
      ProbeSet(
        listOf(
          QnnProjectionResult(-100, 0.0, Double.NaN, QnnProjectionResult.syntheticTolerance),
          QnnProjectionResult(-100, 0.0, Double.NaN),
          QnnProjectionResult(-100, 0.0, Double.NaN, QnnProjectionResult.fullBlockTolerance),
          QnnProjectionResult(-100, 0.0, Double.NaN, QnnProjectionResult.fullBlockTolerance),
          QnnProjectionResult(-100, 0.0, Double.NaN, QnnProjectionResult.liveFullTolerance),
        ),
        QnnPersistentProjectionResult(-100, 0.0, 0.0, 0.0, 0.0, 0.0, Double.NaN, 0.0),
      )
    }
    return result(probes.projections[0].passed, architecture, probes.projections[0], probes.projections[1], probes.projections[2], probes.projections[3], probes.projections[4], probes.persistent)
      .also { cached = it }
  }

  private fun unavailable(reason: String, tolerance: Double) = linkedMapOf<String, Any?>(
    "available" to false,
    "elapsedMs" to 0.0,
    "maxAbsoluteError" to Double.NaN,
    "tolerance" to tolerance,
    "reason" to reason,
  )

  private fun persistentUnavailable(reason: String) = linkedMapOf<String, Any?>(
    "available" to false,
    "createMs" to 0.0,
    "finalizeMs" to 0.0,
    "firstExecuteMs" to 0.0,
    "warmMedianMs" to 0.0,
    "warmLastMs" to 0.0,
    "maxAbsoluteError" to Double.NaN,
    "tolerance" to QnnProjectionResult.liveFullTolerance,
    "reason" to reason,
  )

  private fun detail(probe: QnnProjectionResult) = linkedMapOf<String, Any?>(
    "available" to probe.passed,
    "elapsedMs" to probe.elapsedMs,
    "maxAbsoluteError" to probe.maxAbsoluteError,
    "tolerance" to probe.tolerance,
    "reason" to probe.reason(),
  )

  private fun persistentDetail(probe: QnnPersistentProjectionResult) = linkedMapOf<String, Any?>(
    "available" to probe.passed,
    "createMs" to probe.createMs,
    "finalizeMs" to probe.finalizeMs,
    "firstExecuteMs" to probe.firstExecuteMs,
    "warmMedianMs" to probe.warmMedianMs,
    "warmLastMs" to probe.warmLastMs,
    "maxAbsoluteError" to probe.maxAbsoluteError,
    "tolerance" to probe.tolerance,
    "reason" to probe.reason(),
  )

  private fun result(
    available: Boolean,
    architecture: String?,
    projection: QnnProjectionResult,
    checkpoint: QnnProjectionResult,
    full: QnnProjectionResult,
    live: QnnProjectionResult,
    liveFull: QnnProjectionResult,
    persistentFull: QnnPersistentProjectionResult,
  ) = linkedMapOf<String, Any?>(
    "available" to available,
    "architecture" to architecture,
    "probe" to "qnn-htp-projection-v1",
    "elapsedMs" to projection.elapsedMs,
    "maxAbsoluteError" to projection.maxAbsoluteError,
    "tolerance" to projection.tolerance,
    "reason" to projection.reason(),
    "checkpointProjection" to detail(checkpoint),
    "fullBlockProjection" to detail(full),
    "liveCheckpointProjection" to detail(live),
    "liveFullProjection" to detail(liveFull),
    "persistentFullProjection" to persistentDetail(persistentFull),
  )

  private fun result(
    available: Boolean,
    architecture: String?,
    projection: Map<String, Any?>,
    checkpoint: Map<String, Any?>,
    full: Map<String, Any?>,
    live: Map<String, Any?>,
    liveFull: Map<String, Any?>,
    persistentFull: Map<String, Any?>,
  ) = linkedMapOf<String, Any?>(
    "available" to available,
    "architecture" to architecture,
    "probe" to "qnn-htp-projection-v1",
    "elapsedMs" to projection["elapsedMs"],
    "maxAbsoluteError" to projection["maxAbsoluteError"],
    "tolerance" to projection["tolerance"],
    "reason" to projection["reason"],
    "checkpointProjection" to checkpoint,
    "fullBlockProjection" to full,
    "liveCheckpointProjection" to live,
    "liveFullProjection" to liveFull,
    "persistentFullProjection" to persistentFull,
  )
}
