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

  @Test fun reportsLivePayloadFailuresWithoutTreatingThemAsQnnPasses() {
    val missing = QnnProjectionResult(-120, 0.0, Double.NaN, QnnProjectionResult.fullBlockTolerance)
    val malformed = QnnProjectionResult(-121, 0.0, Double.NaN, QnnProjectionResult.fullBlockTolerance)
    val stale = QnnProjectionResult(-122, 0.0, Double.NaN, QnnProjectionResult.fullBlockTolerance)

    assertFalse(missing.passed)
    assertEquals("Live checkpoint payload is not prepared in this app's private storage.", missing.reason())
    assertEquals("Live checkpoint payload is malformed or failed its integrity check.", malformed.reason())
    assertEquals("Live checkpoint payload is stale for the pinned model identity.", stale.reason())
  }

  @Test fun fullProjectionUsesContentBoundNativeToleranceAndRejectsPayloadFailures() {
    val passing = QnnProjectionResult.fromNative(
      doubleArrayOf(0.0, 10.0, 0.1, 0.2),
      QnnProjectionResult.liveFullTolerance,
    )
    assertTrue(passing.passed)
    assertEquals(0.2, passing.tolerance, 0.0)
    assertFalse(QnnProjectionResult.fromNative(doubleArrayOf(0.0, 10.0, 0.3, 0.2)).passed)
    assertFalse(QnnProjectionResult(-120, 0.0, Double.NaN, QnnProjectionResult.liveFullTolerance).passed)
    assertEquals(
      "Live checkpoint payload is stale for the pinned model identity.",
      QnnProjectionResult(-122, 0.0, Double.NaN, QnnProjectionResult.liveFullTolerance).reason(),
    )
  }
}
