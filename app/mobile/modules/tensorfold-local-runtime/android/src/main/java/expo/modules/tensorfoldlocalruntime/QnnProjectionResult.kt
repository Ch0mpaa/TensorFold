package expo.modules.tensorfoldlocalruntime

internal data class QnnProjectionResult(
  val status: Int,
  val elapsedMs: Double,
  val maxAbsoluteError: Double,
  val tolerance: Double = checkpointTolerance,
) {
  val passed: Boolean get() = status == 0 && maxAbsoluteError.isFinite() && maxAbsoluteError <= tolerance

  fun reason(): String? = when {
    passed -> null
    status == -100 -> "QNN HTP transport or v79 stub could not load."
    status == -101 -> "QNN HTP interface provider was unavailable."
    status == -102 -> "QNN HTP interface is missing a required graph API."
    status == -103 -> "QNN HTP backend creation failed."
    status == -104 -> "QNN HTP device creation failed."
    status == -105 -> "QNN HTP context creation failed."
    status == -106 -> "QNN HTP profile creation failed."
    status == -107 -> "QNN HTP graph creation failed."
    status == -108 -> "QNN HTP tensor creation failed."
    status == -109 -> "QNN HTP MatMul node creation failed."
    status == -110 -> "QNN HTP graph finalization failed."
    status == -111 -> "QNN HTP graph execution failed."
    status == -112 -> "QNN HTP output exceeded the projection tolerance."
    status == -120 -> "Live checkpoint payload is not prepared in this app's private storage."
    status == -121 -> "Live checkpoint payload is malformed or failed its integrity check."
    status == -113 -> "QNN HTP persistent session is busy."
    status == -114 -> "QNN HTP persistent session is in an invalid state."
    status == -115 -> "QNN HTP persistent session received an invalid tensor shape."
    status == -122 -> "Live checkpoint payload is stale for the pinned model identity."
    status == -123 -> "QNN HTP RMSNorm square node creation failed."
    status == -124 -> "QNN HTP RMSNorm reduce-mean node creation failed."
    status == -125 -> "QNN HTP RMSNorm epsilon-add node creation failed."
    status == -126 -> "QNN HTP RMSNorm reciprocal-square-root node creation failed."
    status == -127 -> "QNN HTP RMSNorm activation-scale node creation failed."
    status == -128 -> "QNN HTP RMSNorm gamma-scale node creation failed."
    else -> "QNN HTP projection failed with status $status."
  }

  companion object {
    const val syntheticTolerance = 0.5
    const val checkpointTolerance = 0.0005
    const val fullBlockTolerance = 0.00263044941
    // Successful live payloads return their content-bound native tolerance as
    // the fourth result value. Zero fails closed for missing/malformed payloads.
    const val liveFullTolerance = 0.0

    fun fromNative(values: DoubleArray, tolerance: Double = checkpointTolerance): QnnProjectionResult = when {
      values.size == 3 -> QnnProjectionResult(values[0].toInt(), values[1], values[2], tolerance)
      values.size == 4 && values[3].isFinite() && values[3] >= 0.0 ->
        QnnProjectionResult(values[0].toInt(), values[1], values[2], values[3])
      else -> QnnProjectionResult(-102, 0.0, Double.NaN, tolerance)
    }
  }
}

internal data class QnnPersistentProjectionResult(
  val status: Int,
  val createMs: Double,
  val finalizeMs: Double,
  val firstExecuteMs: Double,
  val warmMedianMs: Double,
  val warmLastMs: Double,
  val maxAbsoluteError: Double,
  val tolerance: Double,
) {
  val passed: Boolean get() =
    status == 0 && maxAbsoluteError.isFinite() && tolerance.isFinite() && tolerance >= 0.0 &&
      maxAbsoluteError <= tolerance

  fun reason(): String? = if (passed) null else QnnProjectionResult(
    status, firstExecuteMs, maxAbsoluteError, tolerance,
  ).reason()

  companion object {
    fun fromNative(values: DoubleArray): QnnPersistentProjectionResult = if (
      values.size == 8 && values[7].isFinite() && values[7] >= 0.0
    ) {
      QnnPersistentProjectionResult(
        values[0].toInt(), values[1], values[2], values[3], values[4], values[5], values[6], values[7],
      )
    } else {
      QnnPersistentProjectionResult(-114, 0.0, 0.0, 0.0, 0.0, 0.0, Double.NaN, 0.0)
    }
  }
}

internal data class QnnServiceProjectionResult(
  val status: Int,
  val firstExecuteMs: Double,
  val warmMedianMs: Double,
  val warmLastMs: Double,
  val qnnQuantizedMaxAbsoluteError: Double,
  val quantizedOriginalMaxAbsoluteError: Double,
  val qnnOriginalMaxAbsoluteError: Double,
  val originalReferenceMaxMagnitude: Double,
  val relativeQnnOriginalError: Double,
  val tolerance: Double,
) {
  val passed: Boolean get() = status == 0 && qnnQuantizedMaxAbsoluteError.isFinite() &&
    tolerance.isFinite() && tolerance >= 0.0 && qnnQuantizedMaxAbsoluteError <= tolerance
  fun reason(): String? = if (passed) null else QnnProjectionResult(status, firstExecuteMs, qnnQuantizedMaxAbsoluteError, tolerance).reason()

  companion object {
    fun fromNative(values: DoubleArray): QnnServiceProjectionResult = if (
      values.size == 10 && values[9].isFinite() && values[9] >= 0.0
    ) {
      QnnServiceProjectionResult(
        values[0].toInt(), values[1], values[2], values[3], values[4], values[5], values[6],
        values[7], values[8], values[9],
      )
    } else {
      QnnServiceProjectionResult(-114, 0.0, 0.0, 0.0, Double.NaN, Double.NaN, Double.NaN,
        Double.NaN, Double.NaN, 0.0)
    }
  }
}

internal data class QnnPairServiceProjectionResult(
  val status: Int,
  val qkvError: Double,
  val qkvTolerance: Double,
  val zError: Double,
  val zTolerance: Double,
) {
  val passed: Boolean get() = status == 0 && qkvError.isFinite() && zError.isFinite() &&
    qkvTolerance.isFinite() && zTolerance.isFinite() && qkvTolerance >= 0.0 && zTolerance >= 0.0 &&
    qkvError <= qkvTolerance && zError <= zTolerance

  companion object {
    fun fromNative(values: DoubleArray): QnnPairServiceProjectionResult = if (
      values.size == 14 && values[5].isFinite() && values[7].isFinite()
    ) QnnPairServiceProjectionResult(values[0].toInt(), values[4], values[5], values[6], values[7])
    else QnnPairServiceProjectionResult(-114, Double.NaN, 0.0, Double.NaN, 0.0)
  }
}
