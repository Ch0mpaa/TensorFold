package expo.modules.tensorfoldlocalruntime

import android.app.Service
import android.app.Application
import android.content.Intent
import android.os.Binder
import android.os.Bundle
import android.os.IBinder
import android.os.Process
import java.util.concurrent.Executors
import java.util.concurrent.ConcurrentHashMap

class QnnExecutorService : Service() {
  private var lastStatus = -114
  private val worker = Executors.newSingleThreadExecutor { task -> Thread(task, "tensorfold-qnn") }
  private val graphRecipes = ConcurrentHashMap<Long, String>()

  private fun sameUid() {
    if (Binder.getCallingUid() != Process.myUid()) {
      throw SecurityException("QNN executor only accepts its app UID.")
    }
  }

  private fun result(values: DoubleArray, keys: Array<String>): Bundle = Bundle().apply {
    keys.forEachIndexed { index, key -> putDouble(key, values.getOrElse(index) { Double.NaN }) }
    lastStatus = values.firstOrNull()?.toInt() ?: -114
    putInt("status", lastStatus)
  }

  private val binder = object : IQnnExecutor.Stub() {
    override fun createGraph(): Bundle {
      sameUid()
      return try {
        val payload = filesDir.resolve("tensorfold/qnn/live-layer0-rmsnorm-projection-pair-v7.bin").let {
          when {
            it.isFile -> it
            filesDir.resolve("tensorfold/qnn/live-layer0-projection-pair-v6.bin").isFile ->
              filesDir.resolve("tensorfold/qnn/live-layer0-projection-pair-v6.bin")
            filesDir.resolve("tensorfold/qnn/live-full-projection-v5.bin").isFile ->
              filesDir.resolve("tensorfold/qnn/live-full-projection-v5.bin")
            filesDir.resolve("tensorfold/qnn/live-full-projection-v4.bin").isFile ->
              filesDir.resolve("tensorfold/qnn/live-full-projection-v4.bin")
            filesDir.resolve("tensorfold/qnn/live-full-projection-v3.bin").isFile ->
              filesDir.resolve("tensorfold/qnn/live-full-projection-v3.bin")
            else -> filesDir.resolve("tensorfold/qnn/live-full-projection-v2.bin")
          }
        }
        val recipe = when {
          payload.name.endsWith("-rmsnorm-projection-pair-v7.bin") -> "fp16-qnn-rmsnorm-qkv-z"
          payload.name.endsWith("-pair-v6.bin") -> "fp16-real-input-qkv-z"
          payload.name.endsWith("-v5.bin") -> "fp16-input-weight-output"
          payload.name.endsWith("-v4.bin") -> "axis-u8-output-channel"
          else -> "symmetric-u8-per-tensor"
        }
        val values = worker.submit<DoubleArray> {
          QnnNative.create(applicationInfo.nativeLibraryDir, payload.absolutePath)
        }.get()
        lastStatus = values.getOrElse(0) { -114.0 }.toInt()
        Bundle().apply {
          val handle = values.getOrElse(1) { 0.0 }.toLong()
          if (lastStatus == 0 && handle != 0L) graphRecipes[handle] = recipe
          putInt("status", lastStatus)
          putLong("graphHandle", handle)
          putDouble("createMs", values.getOrElse(2) { 0.0 })
          putDouble("finalizeMs", values.getOrElse(3) { 0.0 })
          putDouble("tolerance", values.getOrElse(4) { 0.0 })
          putString("recipe", recipe)
          if (lastStatus == -121) {
            putString("error", "Selected $recipe layer-0 payload is malformed or failed its integrity check; no older payload was run.")
          } else if (lastStatus == -122) {
            putString("error", "Selected $recipe layer-0 payload does not match the pinned model identity; no older payload was run.")
          } else if (lastStatus != 0) {
            val stage = QnnProjectionResult(lastStatus, 0.0, Double.NaN, getDouble("tolerance")).reason()
            putString("error", "Selected $recipe graph failed without fallback: $stage")
          }
        }
      } catch (_: Throwable) {
        Bundle().apply {
          putInt("status", -100)
          putString("error", "QNN service could not load its native graph runtime.")
        }
      }
    }

    override fun execute(graphHandle: Long, repeats: Int): Bundle {
      sameUid()
      if (repeats !in 1..100) return Bundle().apply { putInt("status", -115); putString("error", "Repeat count is outside the bounded QNN oracle range.") }
      return try {
        val values = worker.submit<DoubleArray> { QnnNative.execute(graphHandle, repeats) }.get()
        val isPair = graphRecipes[graphHandle] in setOf("fp16-real-input-qkv-z", "fp16-qnn-rmsnorm-qkv-z")
        val expectedSize = if (isPair) 14 else 10
        if (values.size != expectedSize) Bundle().apply {
          lastStatus = -114
          putInt("status", lastStatus)
          putString("error", "QNN graph result does not match its recorded recipe contract.")
        } else if (isPair) Bundle().apply {
          lastStatus = values[0].toInt(); putInt("status", lastStatus)
          putDouble("firstExecuteMs", values[1]); putDouble("warmMedianMs", values[2]); putDouble("warmLastMs", values[3])
          putDouble("qkvQnnExecutionMaxAbsoluteError", values[4]); putDouble("qkvTolerance", values[5])
          putDouble("zQnnExecutionMaxAbsoluteError", values[6]); putDouble("zTolerance", values[7])
          putDouble("qkvExecutionOriginalMaxAbsoluteError", values[8]); putDouble("qkvQnnOriginalMaxAbsoluteError", values[9]); putDouble("qkvOriginalReferenceMaxMagnitude", values[10])
          putDouble("zExecutionOriginalMaxAbsoluteError", values[11]); putDouble("zQnnOriginalMaxAbsoluteError", values[12]); putDouble("zOriginalReferenceMaxMagnitude", values[13])
        } else result(values, arrayOf(
          "statusValue", "firstExecuteMs", "warmMedianMs", "warmLastMs", "qnnQuantizedMaxAbsoluteError",
          "quantizedOriginalMaxAbsoluteError", "qnnOriginalMaxAbsoluteError",
          "originalReferenceMaxMagnitude", "relativeQnnOriginalError", "tolerance",
        ))
      } catch (_: Throwable) { Bundle().apply { putInt("status", -114); putString("error", "QNN graph handle is unavailable or stale.") } }
    }

    override fun releaseGraph(graphHandle: Long) {
      sameUid()
      try {
        lastStatus = worker.submit<Int> { QnnNative.release(graphHandle) }.get()
      } finally {
        graphRecipes.remove(graphHandle)
      }
    }

    override fun health(): Bundle {
      sameUid()
      return try {
        val state = worker.submit<LongArray> { QnnNative.health() }.get()
        Bundle().apply {
          putString("backend", "QNN HTP v79")
          putString("processName", Application.getProcessName())
          putInt("pid", Process.myPid())
          putLong("graphCount", state.getOrElse(0) { 0 })
          putInt("lastStatus", lastStatus)
        }
      } catch (_: Throwable) {
        Bundle().apply {
          putString("backend", "unavailable")
          putString("processName", Application.getProcessName())
          putInt("pid", Process.myPid())
          putLong("graphCount", 0)
          putInt("lastStatus", -100)
        }
      }
    }
  }

  override fun onBind(intent: Intent?): IBinder = binder
  override fun onDestroy() {
    try {
      worker.submit { QnnNative.releaseAll() }.get()
    } catch (_: Throwable) {
    } finally {
      graphRecipes.clear()
      worker.shutdown()
      super.onDestroy()
    }
  }
}
