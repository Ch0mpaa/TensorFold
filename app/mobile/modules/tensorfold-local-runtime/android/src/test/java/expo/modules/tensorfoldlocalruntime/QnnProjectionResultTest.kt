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

  @Test fun persistentProjectionParsesAllSeparatedMeasurementsAndFailsClosed() {
    val passing = QnnPersistentProjectionResult.fromNative(
      doubleArrayOf(0.0, 3.0, 4.0, 5.0, 2.0, 2.5, 0.0093, 0.01),
    )
    assertTrue(passing.passed)
    assertEquals(3.0, passing.createMs, 0.0)
    assertEquals(4.0, passing.finalizeMs, 0.0)
    assertEquals(2.0, passing.warmMedianMs, 0.0)
    assertFalse(QnnPersistentProjectionResult.fromNative(doubleArrayOf(0.0)).passed)
    assertFalse(QnnPersistentProjectionResult.fromNative(
      doubleArrayOf(0.0, 3.0, 4.0, 5.0, 2.0, 2.5, 0.02, 0.01),
    ).passed)
  }

  @Test fun persistentProjectionReportsFailClosedStateReasons() {
    assertEquals(
      "QNN HTP persistent session is in an invalid state.",
      QnnPersistentProjectionResult(-114, 0.0, 0.0, 0.0, 0.0, 0.0, Double.NaN, 0.0).reason(),
    )
    assertEquals(
      "QNN HTP persistent session received an invalid tensor shape.",
      QnnPersistentProjectionResult(-115, 0.0, 0.0, 0.0, 0.0, 0.0, Double.NaN, 0.0).reason(),
    )
  }
}
