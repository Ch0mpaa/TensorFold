package expo.modules.tensorfoldlocalruntime

import android.content.Context
import android.os.Build
import java.io.File

internal data class TensorG5ProjectionResult(
  val status: Int,
  val compatibility: Boolean,
  val fullyAccelerated: Boolean,
  val firstExecuteMs: Double,
  val warmMedianMs: Double,
  val warmLastMs: Double,
  val maxAbsoluteError: Double,
  val tolerance: Double = parityTolerance,
) {
  val available: Boolean get() = status == 0 && compatibility && fullyAccelerated &&
    maxAbsoluteError.isFinite() && maxAbsoluteError <= tolerance

  fun reason(): String? = when {
    status == -10 -> "This probe requires Google Tensor G5."
    status == -11 -> "Google Tensor LiteRT NPU compatibility check failed."
    status == -12 -> "The required selfie_multiclass.tflite probe model is missing or has the wrong content."
    status == -13 -> "LiteRT did not expose a native compiled-model handle."
    status == -1 -> "LiteRT TPU attestation could not open libLiteRt.so."
    status == -2 -> "LiteRT TPU attestation symbol is unavailable."
    status == -3 -> "LiteRT TPU attestation returned an error."
    status == -4 -> "LiteRT reports that this model was not fully accelerated outside CPU."
    status != 0 -> "LiteRT Tensor G5 TPU probe failed (status $status)."
    !compatibility -> "Google Tensor LiteRT NPU compatibility check failed."
    !fullyAccelerated -> "LiteRT reports that this model was not fully accelerated outside CPU."
    !maxAbsoluteError.isFinite() -> "LiteRT TPU output could not be compared with the CPU oracle."
    maxAbsoluteError > tolerance -> "LiteRT TPU output differs from the CPU oracle by $maxAbsoluteError (limit $tolerance)."
    else -> null
  }

  companion object {
    const val parityTolerance = 0.02
    const val probeBytes = 16_371_837L
    const val probeSha256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
    const val nativeStatusMissingHandle = -1
    const val nativeStatusLibraryUnavailable = -2
    const val nativeStatusSymbolUnavailable = -3
    const val nativeStatusAttestationFailed = -4
    const val nativeStatusNotFullyAccelerated = -5

    fun unavailable(status: Int, compatibility: Boolean = false) = TensorG5ProjectionResult(
      status, compatibility, false, 0.0, 0.0, 0.0, Double.NaN,
    )
  }
}

internal object TensorG5Probe {
  private val probeArtifact = ModelArtifact(
    "selfie_multiclass.tflite",
    TensorG5ProjectionResult.probeBytes,
    TensorG5ProjectionResult.probeSha256,
  )
  private var cached: Map<String, Any?>? = null

  @Synchronized
  fun inspect(context: Context): Map<String, Any?> {
    cached?.let { return it }
    if (!Build.SOC_MODEL.equals("Tensor G5", ignoreCase = true)) {
      return detail(TensorG5ProjectionResult.unavailable(-10))
    }
    val compatible = try {
      TensorG5LiteRtRunner.isDeviceSupported()
    } catch (_: Throwable) {
      false
    }
    if (!compatible) return detail(TensorG5ProjectionResult.unavailable(-11))
    val model = context.getExternalFilesDir(null)?.resolve("tensorfold/tpu/selfie_multiclass.tflite")
    if (model == null || !ModelIntegrity.matches(model, probeArtifact)) {
      return detail(TensorG5ProjectionResult.unavailable(-12, true))
    }
    return detail(run(context, model)).also { cached = it }
  }

  private fun run(context: Context, model: File): TensorG5ProjectionResult = try {
    val measured = TensorG5LiteRtRunner.run(context, model.absolutePath)
    val nativeStatus = measured.nativeStatus.toInt()
    if (nativeStatus != 0 || !measured.fullyAccelerated) {
      val status = if (nativeStatus == 0) -4 else nativeStatusToProbeStatus(nativeStatus)
      TensorG5ProjectionResult.unavailable(status, true)
    } else {
      TensorG5ProjectionResult(
        0,
        true,
        true,
        measured.firstExecuteMs,
        measured.warmMedianMs,
        measured.warmLastMs,
        measured.maxAbsoluteError,
      )
    }
  } catch (_: Throwable) {
    TensorG5ProjectionResult.unavailable(-20, true)
  }

  private fun nativeStatusToProbeStatus(status: Int) = when (status) {
    TensorG5ProjectionResult.nativeStatusMissingHandle -> -13
    TensorG5ProjectionResult.nativeStatusLibraryUnavailable -> -1
    TensorG5ProjectionResult.nativeStatusSymbolUnavailable -> -2
    TensorG5ProjectionResult.nativeStatusAttestationFailed -> -3
    TensorG5ProjectionResult.nativeStatusNotFullyAccelerated -> -4
    else -> -3
  }

  private fun detail(result: TensorG5ProjectionResult) = linkedMapOf<String, Any?>(
    "available" to result.available,
    "compatibility" to result.compatibility,
    "fullyAccelerated" to result.fullyAccelerated,
    "firstExecuteMs" to result.firstExecuteMs,
    "warmMedianMs" to result.warmMedianMs,
    "warmLastMs" to result.warmLastMs,
    "maxAbsoluteError" to result.maxAbsoluteError,
    "tolerance" to result.tolerance,
    "reason" to result.reason(),
  )
}
