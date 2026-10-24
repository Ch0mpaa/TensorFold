package expo.modules.tensorfoldlocalruntime

import android.app.Service
import android.app.Application
import android.content.Intent
import android.os.Binder
import android.os.Bundle
import android.os.IBinder
import android.os.Process
import java.util.concurrent.Executors

class QnnExecutorService : Service() {
  private var lastStatus = -114
  private val worker = Executors.newSingleThreadExecutor { task -> Thread(task, "tensorfold-qnn") }

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
        val payload = filesDir.resolve("tensorfold/qnn/live-full-projection-v5.bin").let {
          when {
            it.isFile -> it
            filesDir.resolve("tensorfold/qnn/live-full-projection-v4.bin").isFile ->
              filesDir.resolve("tensorfold/qnn/live-full-projection-v4.bin")
            filesDir.resolve("tensorfold/qnn/live-full-projection-v3.bin").isFile ->
              filesDir.resolve("tensorfold/qnn/live-full-projection-v3.bin")
            else -> filesDir.resolve("tensorfold/qnn/live-full-projection-v2.bin")
          }
        }
        val recipe = when {
          payload.name.endsWith("-v5.bin") -> "fp16-input-weight-output"
          payload.name.endsWith("-v4.bin") -> "axis-u8-output-channel"
          else -> "symmetric-u8-per-tensor"
        }
        val values = worker.submit<DoubleArray> {
          QnnNative.create(applicationInfo.nativeLibraryDir, payload.absolutePath)
        }.get()
        lastStatus = values.getOrElse(0) { -114.0 }.toInt()
        Bundle().apply {
          putInt("status", lastStatus)
          putLong("graphHandle", values.getOrElse(1) { 0.0 }.toLong())
          putDouble("createMs", values.getOrElse(2) { 0.0 })
          putDouble("finalizeMs", values.getOrElse(3) { 0.0 })
          putDouble("tolerance", values.getOrElse(4) { 0.0 })
          putString("recipe", recipe)
          if (lastStatus == -121) {
            putString("error", "Selected $recipe layer-0 projection payload is malformed or failed its integrity check.")
          } else if (lastStatus == -122) {
            putString("error", "Selected $recipe layer-0 projection payload does not match the pinned model identity.")
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
        result(worker.submit<DoubleArray> { QnnNative.execute(graphHandle, repeats) }.get(), arrayOf(
          "statusValue", "firstExecuteMs", "warmMedianMs", "warmLastMs", "qnnQuantizedMaxAbsoluteError",
          "quantizedOriginalMaxAbsoluteError", "qnnOriginalMaxAbsoluteError",
          "originalReferenceMaxMagnitude", "relativeQnnOriginalError", "tolerance",
        ))
      } catch (_: Throwable) { Bundle().apply { putInt("status", -114); putString("error", "QNN graph handle is unavailable or stale.") } }
    }

    override fun releaseGraph(graphHandle: Long) { sameUid(); lastStatus = worker.submit<Int> { QnnNative.release(graphHandle) }.get() }

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
      worker.shutdown()
      super.onDestroy()
    }
  }
}
