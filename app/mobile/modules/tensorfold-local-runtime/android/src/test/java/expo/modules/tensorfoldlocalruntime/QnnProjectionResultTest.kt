package expo.modules.tensorfoldlocalruntime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class QnnProjectionResultTest {
  @Test fun acceptsOnlyACompleteParityPassingNativeResult() {
    val result = QnnProjectionResult.fromNative(doubleArrayOf(0.0, 1.25, 0.0))
    assertTrue(result.passed)
    assertNull(result.reason())
    assertEquals(1.25, result.elapsedMs, 0.0)
  }

  @Test fun failsClosedForMalformedOrOutOfToleranceResults() {
    assertFalse(QnnProjectionResult.fromNative(doubleArrayOf(0.0, 1.0)).passed)
    assertFalse(QnnProjectionResult(0, 1.0, Double.NaN).passed)
    assertFalse(QnnProjectionResult(0, 1.0, QnnProjectionResult.checkpointTolerance + 0.01).passed)
  }

  @Test fun reportsTheFailingQnnStage() {
    assertEquals("QNN HTP graph execution failed.", QnnProjectionResult(-111, 0.0, Double.NaN).reason())
  }
}
