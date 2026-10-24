package expo.modules.tensorfoldlocalruntime

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

internal object QnnExecutorClient {
  private const val timeoutSeconds = 15L

  fun runOracle(context: Context): Bundle {
    val connected = CountDownLatch(1)
    var service: IQnnExecutor? = null
    var dead = false
    val death = IBinder.DeathRecipient { dead = true; service = null; connected.countDown() }
    val connection = object : ServiceConnection {
      override fun onServiceConnected(name: ComponentName, binder: IBinder) {
        service = IQnnExecutor.Stub.asInterface(binder)
        try { binder.linkToDeath(death, 0) } catch (_: Throwable) { dead = true }
        connected.countDown()
      }
      override fun onServiceDisconnected(name: ComponentName) { dead = true; service = null }
      override fun onBindingDied(name: ComponentName) { dead = true; service = null }
      override fun onNullBinding(name: ComponentName) { dead = true; connected.countDown() }
    }
    val intent = Intent(context, QnnExecutorService::class.java)
    val bound = context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
    if (!bound) return unavailable("QNN service could not be bound.")
    if (!connected.await(timeoutSeconds, TimeUnit.SECONDS) || service == null || dead) {
      try { context.unbindService(connection) } catch (_: Throwable) {}
      return unavailable("QNN service is unavailable or died before graph creation.")
    }
    var handle = 0L
    return try {
      val created = service!!.createGraph()
      handle = created.getLong("graphHandle")
      if (created.getInt("status") != 0 || handle == 0L || dead) return unavailable(created.getString("error") ?: "QNN service could not create its graph.", created)
      val executed = service!!.execute(handle, 100)
      val health = if (dead) Bundle() else service!!.health()
      val healthyService =
        health.getString("backend") == "QNN HTP v79" &&
          health.getString("processName", "").endsWith(":qnn") &&
          health.getLong("graphCount") == 1L
      if (dead) unavailable("QNN service died while executing the graph.", executed) else Bundle(executed).apply {
        putString("serviceProcess", health.getString("processName", ":qnn"))
        putString("backend", health.getString("backend", "unavailable"))
        putInt("servicePid", health.getInt("pid"))
        putLong("graphCount", health.getLong("graphCount"))
        putBoolean(
          "available",
          healthyService && getInt("status") == 0 && getDouble("qnnQuantizedMaxAbsoluteError").isFinite() &&
            getDouble("qnnQuantizedMaxAbsoluteError") <= getDouble("tolerance"),
        )
        putString(
          "reason",
          when {
            getBoolean("available") -> null
            !healthyService -> "QNN service process or graph residency attestation failed."
            else -> getString("error") ?: QnnProjectionResult(
              getInt("status"),
              getDouble("firstExecuteMs"),
              getDouble("qnnQuantizedMaxAbsoluteError"),
              getDouble("tolerance"),
            ).reason()
          },
        )
        putDouble("createMs", created.getDouble("createMs"))
        putDouble("finalizeMs", created.getDouble("finalizeMs"))
        putString("recipe", created.getString("recipe"))
        putInt("repeats", 100)
      }
    } catch (_: Throwable) { unavailable("QNN service binder call failed or its graph handle became stale.") }
    finally {
      if (handle != 0L && !dead) try { service?.releaseGraph(handle) } catch (_: Throwable) {}
      try { service?.asBinder()?.unlinkToDeath(death, 0) } catch (_: Throwable) {}
      try { context.unbindService(connection) } catch (_: Throwable) {}
    }
  }

  private fun unavailable(reason: String, values: Bundle? = null) = Bundle(values ?: Bundle()).apply {
    putBoolean("available", false); putString("serviceProcess", ":qnn"); putString("reason", reason); putInt("status", getInt("status", -114)); putInt("repeats", 100)
  }
}
