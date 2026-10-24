package expo.modules.tensorfoldlocalruntime

import android.content.Context
import android.os.Bundle

internal object QnnProbe {
  private const val stubName = "libQnnHtpV79Stub.so"

  private data class ProbeSet(
    val projections: List<QnnProjectionResult>,
    val persistent: QnnPersistentProjectionResult,
    val service: Map<String, Any?>,
  )

  private var cached: Map<String, Any?>? = null

  @JvmStatic private external fun nativeProjection(nativeDir: String, stubName: String): DoubleArray
  @JvmStatic private external fun nativeCheckpointProjection(
    nativeDir: String,
    stubName: String,
  ): DoubleArray
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
        unavailable("No prepared checkpoint projection matches this SoC.", QnnProjectionResult.fullBlockTolerance),
        unavailable("No prepared full projection matches this SoC.", QnnProjectionResult.liveFullTolerance),
        persistentUnavailable("No prepared persistent full projection matches this SoC."),
        serviceUnavailable("No QNN service projection matches this SoC."),
      ).also { cached = it }
    }

    val nativeDir = context.applicationInfo.nativeLibraryDir
    val checkpoint = context.filesDir.resolve("tensorfold/qnn/live-checkpoint-v1.bin").absolutePath
    val full = context.filesDir.resolve("tensorfold/qnn/live-full-projection-v2.bin").absolutePath
    val probes = try {
      System.loadLibrary("tensorfold_qnn_probe")
      val projections = listOf(
        QnnProjectionResult.fromNative(
          nativeProjection(nativeDir, stubName),
          QnnProjectionResult.syntheticTolerance,
        ),
        QnnProjectionResult.fromNative(nativeCheckpointProjection(nativeDir, stubName)),
        QnnProjectionResult.fromNative(
          nativeFullProjection(nativeDir, stubName),
          QnnProjectionResult.fullBlockTolerance,
        ),
        QnnProjectionResult.fromNative(
          nativeLiveCheckpointProjection(nativeDir, stubName, checkpoint),
          QnnProjectionResult.fullBlockTolerance,
        ),
        QnnProjectionResult.fromNative(
          nativeLiveFullProjection(nativeDir, stubName, full),
          QnnProjectionResult.liveFullTolerance,
        ),
      )
      ProbeSet(
        projections,
        QnnPersistentProjectionResult.fromNative(
          nativePersistentLiveFullProjection(nativeDir, stubName, full),
        ),
        serviceDetail(QnnExecutorClient.runOracle(context)),
      )
    } catch (_: Throwable) {
      val failed = QnnProjectionResult(
        -100,
        0.0,
        Double.NaN,
        QnnProjectionResult.liveFullTolerance,
      )
      ProbeSet(
        listOf(
          QnnProjectionResult(-100, 0.0, Double.NaN, QnnProjectionResult.syntheticTolerance),
          QnnProjectionResult(-100, 0.0, Double.NaN),
          QnnProjectionResult(-100, 0.0, Double.NaN, QnnProjectionResult.fullBlockTolerance),
          QnnProjectionResult(-100, 0.0, Double.NaN, QnnProjectionResult.fullBlockTolerance),
          failed,
        ),
        QnnPersistentProjectionResult(
          -100,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          Double.NaN,
          0.0,
        ),
        serviceUnavailable("QNN service or native runtime could not load."),
      )
    }
    return result(
      probes.projections[0].passed,
      architecture,
      detail(probes.projections[0]),
      detail(probes.projections[1]),
      detail(probes.projections[2]),
      detail(probes.projections[3]),
      detail(probes.projections[4]),
      persistentDetail(probes.persistent),
      probes.service,
    ).also { cached = it }
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

  private fun serviceUnavailable(reason: String) = linkedMapOf<String, Any?>(
    "available" to false,
    "serviceProcess" to ":qnn",
    "backend" to "unavailable",
    "servicePid" to 0,
    "graphCount" to 0.0,
    "repeats" to 100,
    "createMs" to 0.0,
    "finalizeMs" to 0.0,
    "firstExecuteMs" to 0.0,
    "warmMedianMs" to 0.0,
    "warmLastMs" to 0.0,
    "qnnQuantizedMaxAbsoluteError" to Double.NaN,
    "quantizedOriginalMaxAbsoluteError" to Double.NaN,
    "qnnOriginalMaxAbsoluteError" to Double.NaN,
    "originalReferenceMaxMagnitude" to Double.NaN,
    "relativeQnnOriginalError" to Double.NaN,
    "tolerance" to 0.0,
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

  private fun serviceDetail(bundle: Bundle) = linkedMapOf<String, Any?>(
    "available" to bundle.getBoolean("available"),
    "serviceProcess" to bundle.getString("serviceProcess", ":qnn"),
    "backend" to bundle.getString("backend", "unavailable"),
    "servicePid" to bundle.getInt("servicePid"),
    "graphCount" to bundle.getLong("graphCount").toDouble(),
    "repeats" to bundle.getInt("repeats", 100),
    "createMs" to bundle.getDouble("createMs"),
    "finalizeMs" to bundle.getDouble("finalizeMs"),
    "firstExecuteMs" to bundle.getDouble("firstExecuteMs"),
    "warmMedianMs" to bundle.getDouble("warmMedianMs"),
    "warmLastMs" to bundle.getDouble("warmLastMs"),
    "qnnQuantizedMaxAbsoluteError" to bundle.getDouble("qnnQuantizedMaxAbsoluteError"),
    "quantizedOriginalMaxAbsoluteError" to bundle.getDouble("quantizedOriginalMaxAbsoluteError"),
    "qnnOriginalMaxAbsoluteError" to bundle.getDouble("qnnOriginalMaxAbsoluteError"),
    "originalReferenceMaxMagnitude" to bundle.getDouble("originalReferenceMaxMagnitude"),
    "relativeQnnOriginalError" to bundle.getDouble("relativeQnnOriginalError"),
    "tolerance" to bundle.getDouble("tolerance"),
    "status" to bundle.getInt("status", -114),
    "reason" to bundle.getString("reason"),
  )

  private fun result(
    available: Boolean,
    architecture: String?,
    projection: Map<String, Any?>,
    checkpoint: Map<String, Any?>,
    full: Map<String, Any?>,
    live: Map<String, Any?>,
    liveFull: Map<String, Any?>,
    persistent: Map<String, Any?>,
    service: Map<String, Any?>,
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
    "persistentFullProjection" to persistent,
    "serviceFullProjection" to service,
  )
}
