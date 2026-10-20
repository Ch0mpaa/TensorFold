package expo.modules.tensorfoldlocalruntime

import android.app.ActivityManager
import android.os.BatteryManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.StatFs
import java.io.File
import java.util.Locale

internal object HardwareProbe {
  private fun vulkanVersion(context: Context): Int = context.packageManager.systemAvailableFeatures
    .firstOrNull { it.name == PackageManager.FEATURE_VULKAN_HARDWARE_VERSION }?.version ?: 0

  private fun versionText(version: Int): String = if (version == 0) "none" else
    "${version ushr 22}.${(version ushr 12) and 0x3ff}.${version and 0xfff}"

  fun batteryTemperature(context: Context): Double? {
    val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return null
    val tenths = battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
    return if (tenths == Int.MIN_VALUE) null else tenths / 10.0
  }

  fun classifyPlatform(socManufacturer: String, socModel: String, manufacturer: String, hardware: String): String {
    val identity = "$socManufacturer $socModel $manufacturer $hardware".lowercase(Locale.US)
    return when {
      identity.contains("qti") || identity.contains("qualcomm") || identity.contains("sm8") -> "qualcomm"
      identity.contains("google") || identity.contains("tensor") || Regex("\\bgs[0-9]+\\b").containsMatchIn(identity) -> "pixel-tensor"
      else -> "android-arm64"
    }
  }

  fun inspect(context: Context): Map<String, Any?> {
    val memory = ActivityManager.MemoryInfo().also {
      (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it)
    }
    val socManufacturer = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else Build.HARDWARE
    val socModel = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.BOARD
    val platform = classifyPlatform(socManufacturer, socModel, Build.MANUFACTURER, Build.HARDWARE)
    val vk = vulkanVersion(context)
    val arm64 = Build.SUPPORTED_ABIS.contains("arm64-v8a")
    val vulkan11 = vk >= ((1 shl 22) or (1 shl 12))
    // The Vulkan backend starts and is fast on Adreno 830, but its retained
    // 4B output diverges from CPU/HTP. Keep the public runtime exact by
    // selecting CPU until that backend passes the same token gate.
    val backend = if (arm64) "cpu" else "unsupported"
    val qualification = when {
      backend == "unsupported" -> "unsupported"
      platform == "qualcomm" && socModel.equals("SM8750", ignoreCase = true) -> "qualified-reference"
      platform == "qualcomm" -> "compatible-unqualified"
      platform == "pixel-tensor" -> "qualified-portable-cpu"
      else -> "compatible-unqualified"
    }
    val nativeDir = File(context.applicationInfo.nativeLibraryDir)
    val requiredRuntime = listOf("libtensorfold_openai.so", "libtensorfold_engine_2b.so", "libtensorfold_engine_4b.so")
    val missingRuntime = requiredRuntime.filterNot { File(nativeDir, it).canExecute() }
    val storage = StatFs(context.filesDir.absolutePath)
    return linkedMapOf(
      "platform" to platform,
      "manufacturer" to Build.MANUFACTURER,
      "model" to Build.MODEL,
      "socManufacturer" to socManufacturer,
      "socModel" to socModel,
      "apiLevel" to Build.VERSION.SDK_INT,
      "abis" to Build.SUPPORTED_ABIS.toList(),
      "totalRamBytes" to memory.totalMem.toDouble(),
      "availableRamBytes" to memory.availMem.toDouble(),
      "availableStorageBytes" to storage.availableBytes.toDouble(),
      "vulkanAvailable" to vulkan11,
      "vulkanQualified" to false,
      "vulkanReason" to if (vulkan11) "Retained-token mismatch; portable CPU selected." else "Vulkan 1.1 unavailable.",
      "batteryTemperatureC" to batteryTemperature(context),
      "vulkanVersion" to versionText(vk),
      "backend" to backend,
      "qualification" to qualification,
      "runtimeInstalled" to missingRuntime.isEmpty(),
      "missingRuntime" to missingRuntime,
      "htpAvailableToApp" to false,
      "htpReason" to "The direct FastRPC HTP prototype requires vendor or root privileges; the public app uses Vulkan without privileged device access.",
    )
  }
}
