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
    else -> "QNN HTP projection failed with status $status."
  }

  companion object {
    const val syntheticTolerance = 0.5
    const val checkpointTolerance = 0.0005
    const val fullBlockTolerance = 0.00263044941

    fun fromNative(values: DoubleArray, tolerance: Double = checkpointTolerance): QnnProjectionResult =
      if (values.size == 3) QnnProjectionResult(values[0].toInt(), values[1], values[2], tolerance)
      else QnnProjectionResult(-102, 0.0, Double.NaN, tolerance)
  }
}
