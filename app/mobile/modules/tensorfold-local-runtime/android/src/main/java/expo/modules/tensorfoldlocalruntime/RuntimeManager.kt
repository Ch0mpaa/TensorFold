package expo.modules.tensorfoldlocalruntime

import android.content.Context
import android.os.StatFs
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.net.URI
import java.security.SecureRandom
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

internal object RuntimeManager {
  private const val PORT = 8082
  private const val RESERVE_BYTES = 512L shl 20
  private const val MAX_RESPONSE_CHARS = 2 shl 20
  private val lock = Any()
  private val executor = Executors.newSingleThreadExecutor()
  private val cancelDownload = AtomicBoolean(false)
  private val http = OkHttpClient.Builder()
    .connectTimeout(20, TimeUnit.SECONDS)
    .readTimeout(20, TimeUnit.MINUTES)
    .writeTimeout(2, TimeUnit.MINUTES)
    .followRedirects(false)
    .followSslRedirects(false)
    .build()

  @Volatile private var process: Process? = null
  @Volatile private var generation = 0L
  @Volatile private var engineState = "stopped"
  @Volatile private var activeModel: String? = null
  @Volatile private var runtimeError: String? = null
  @Volatile private var authToken: String? = null
  @Volatile private var transferState = "idle"
  @Volatile private var transferModel: String? = null
  @Volatile private var transferDone = 0L
  @Volatile private var transferTotal = 0L
  @Volatile private var transferError: String? = null

  private fun root(context: Context) = File(context.filesDir, "tensorfold")
  private fun modelsRoot(context: Context) = File(root(context), "models")
  private fun modelDir(context: Context, spec: LocalModelSpec) = File(modelsRoot(context), spec.id)
  private fun cacheDir(context: Context, spec: LocalModelSpec) = File(root(context), "cache/${spec.weights.sha256.take(16)}")
  private fun marker(context: Context, spec: LocalModelSpec) = File(modelDir(context, spec), ".verified")
  private fun cacheMarker(context: Context, spec: LocalModelSpec) = File(cacheDir(context, spec), ".tensorfold-ready")

  private fun cacheReady(context: Context, spec: LocalModelSpec): Boolean =
    cacheMarker(context, spec).takeIf { it.isFile }?.readText()?.trim() == spec.weights.sha256

  private fun newAuthToken(): String {
    val bytes = ByteArray(32)
    SecureRandom().nextBytes(bytes)
    return bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }
  }

  private fun installed(context: Context, spec: LocalModelSpec): Boolean {
    if (marker(context, spec).takeIf { it.isFile }?.readText()?.trim() != spec.weights.sha256) return false
    return spec.artifacts.all { File(modelDir(context, spec), it.name).let { file -> file.isFile && file.length() == it.bytes } }
  }

  private fun partialBytes(context: Context, spec: LocalModelSpec): Long = spec.artifacts.sumOf { artifact ->
    val complete = File(modelDir(context, spec), artifact.name)
    if (complete.isFile && complete.length() == artifact.bytes) artifact.bytes
    else File(modelDir(context, spec), "${artifact.name}.part").length().coerceAtMost(artifact.bytes)
  }

  private fun cacheBytes(context: Context, spec: LocalModelSpec): Long {
    val directory = cacheDir(context, spec)
    if (!directory.isDirectory) return 0
    return directory.walkTopDown().filter { it.isFile }.sumOf { it.length() }.coerceAtMost(spec.decodedCacheBytes)
  }

  private fun storageNeeded(context: Context, spec: LocalModelSpec): Long =
    (spec.totalBytes - partialBytes(context, spec)).coerceAtLeast(0) +
      (spec.decodedCacheBytes - cacheBytes(context, spec)).coerceAtLeast(0) +
      RESERVE_BYTES

  fun models(context: Context): List<Map<String, Any?>> {
    val storage = StatFs(context.filesDir.absolutePath).availableBytes
    val totalRam = (HardwareProbe.inspect(context)["totalRamBytes"] as Double).toLong()
    return ModelCatalog.all.map { spec ->
      val ready = installed(context, spec)
      val needed = storageNeeded(context, spec)
      val fits = storage >= needed && totalRam >= spec.minimumRamBytes
      linkedMapOf(
        "id" to spec.id,
        "repo" to spec.repo,
        "profile" to spec.profile,
        "downloadBytes" to spec.totalBytes.toDouble(),
        "downloadedBytes" to partialBytes(context, spec).toDouble(),
        "decodedCacheBytes" to spec.decodedCacheBytes.toDouble(),
        "installedBytes" to (spec.totalBytes + spec.decodedCacheBytes).toDouble(),
        "minimumRamBytes" to spec.minimumRamBytes.toDouble(),
        "installed" to ready,
        "fits" to fits,
        "reason" to when {
          ready -> null
          storage < needed -> "insufficient-storage"
          totalRam < spec.minimumRamBytes -> "insufficient-memory"
          else -> null
        },
      )
    }
  }

  fun status(context: Context): Map<String, Any?> {
    val health = if (engineState == "ready") getJson("/health") else null
    val metrics = if (engineState == "ready") getJson("/metrics") else null
    if (engineState == "ready" && health == null) {
      engineState = if (process?.isAlive == true) "unhealthy" else "stopped"
      if (runtimeError == null) runtimeError = "The local TensorFold server stopped responding."
    }
    return linkedMapOf(
      "state" to engineState,
      "modelId" to activeModel,
      "endpoint" to "http://127.0.0.1:$PORT",
      "error" to runtimeError,
      "health" to health,
      "metrics" to metrics,
      "batteryTemperatureC" to HardwareProbe.batteryTemperature(context),
      "transfer" to linkedMapOf(
        "state" to transferState,
        "modelId" to transferModel,
        "bytesDone" to transferDone.toDouble(),
        "bytesTotal" to transferTotal.toDouble(),
        "error" to transferError,
      ),
      "models" to models(context),
    )
  }

  fun beginDownload(context: Context, id: String) {
    val spec = ModelCatalog.require(id)
    synchronized(lock) {
      if (transferState == "downloading" || transferState == "verifying") throw IllegalStateException("Another model download is already active.")
      if (installed(context, spec)) return
      val free = StatFs(context.filesDir.absolutePath).availableBytes
      val needed = storageNeeded(context, spec)
      if (free < needed) throw IllegalStateException("This model needs $needed free bytes for its verified download, decoded cache, and safety reserve.")
      cancelDownload.set(false)
      transferState = "downloading"
      transferModel = spec.id
      transferDone = partialBytes(context, spec)
      transferTotal = spec.totalBytes
      transferError = null
    }
    executor.execute {
      try {
        download(context.applicationContext, spec)
        transferState = "ready"
      } catch (failure: Throwable) {
        transferState = if (cancelDownload.get()) "cancelled" else "failed"
        transferError = failure.message ?: failure.javaClass.simpleName
      }
    }
  }

  fun removeModel(context: Context, id: String) {
    val spec = ModelCatalog.require(id)
    synchronized(lock) {
      if (activeModel == id && engineState != "stopped") throw IllegalStateException("Stop this model before removing it.")
      if (transferModel == id && transferState in listOf("downloading", "verifying")) throw IllegalStateException("Pause this download before removing it.")
      if (!modelDir(context, spec).deleteRecursively()) throw IllegalStateException("The private model files could not be completely removed.")
      if (!cacheDir(context, spec).deleteRecursively()) throw IllegalStateException("The private model cache could not be completely removed.")
    }
  }

  fun cancelDownload() {
    cancelDownload.set(true)
  }


  private fun openDownload(url: String, offset: Long): Response {
    var current = URI(url)
    repeat(6) {
      if (!DownloadPolicy.allows(current)) throw SecurityException("The model catalog redirected outside approved HTTPS hosts.")
      val request = Request.Builder().url(current.toString()).apply {
        if (offset > 0) header("Range", "bytes=$offset-")
      }.build()
      val response = http.newCall(request).execute()
      if (response.code !in listOf(301, 302, 303, 307, 308)) return response
      val location = response.header("Location") ?: run {
        response.close(); throw IllegalStateException("Model download redirect omitted its destination.")
      }
      response.close()
      current = current.resolve(location)
    }
    throw IllegalStateException("Model download exceeded the redirect limit.")
  }

  private fun download(context: Context, spec: LocalModelSpec) {
    val directory = modelDir(context, spec)
    if (!directory.mkdirs() && !directory.isDirectory) throw IllegalStateException("Could not create the private model directory.")
    marker(context, spec).delete()
    var completed = 0L
    for (artifact in spec.artifacts) {
      if (cancelDownload.get()) throw InterruptedException("Model download cancelled; the verified partial file was retained for resume.")
      val final = File(directory, artifact.name)
      if (ModelIntegrity.matches(final, artifact)) {
        completed += artifact.bytes
        transferDone = completed
        continue
      }
      final.delete()
      val partial = File(directory, "${artifact.name}.part")
      if (partial.length() > artifact.bytes) partial.delete()
      var offset = partial.length()
      openDownload(artifact.url(spec.repo), offset).use { response ->
        if (response.code == 416 && offset == artifact.bytes) {
          // A completed partial still receives a full hash check below.
        } else {
          if (response.code == 200) offset = 0
          else if (response.code == 206) {
            val range = response.header("Content-Range") ?: throw IllegalStateException("Resume response omitted Content-Range.")
            if (!range.startsWith("bytes $offset-")) throw IllegalStateException("Resume response started at the wrong byte.")
          } else throw IllegalStateException("Model download returned HTTP ${response.code}.")
          val body = response.body ?: throw IllegalStateException("Model download returned no body.")
          RandomAccessFile(partial, "rw").use { output ->
            if (offset == 0L) output.setLength(0) else output.seek(offset)
            body.byteStream().use { input ->
              val buffer = ByteArray(1 shl 20)
              var written = offset
              while (true) {
                if (cancelDownload.get()) throw InterruptedException("Model download cancelled; the partial file was retained for resume.")
                val count = input.read(buffer)
                if (count < 0) break
                written += count
                if (written > artifact.bytes) throw IllegalStateException("Model download exceeded its declared size.")
                output.write(buffer, 0, count)
                transferDone = completed + written
              }
              output.fd.sync()
            }
          }
        }
      }
      transferState = "verifying"
      if (partial.length() != artifact.bytes) throw IllegalStateException("${artifact.name} is incomplete.")
      if (!ModelIntegrity.matches(partial, artifact)) {
        partial.delete()
        throw SecurityException("${artifact.name} failed its SHA-256 check and was removed.")
      }
      if (!partial.renameTo(final)) throw IllegalStateException("Could not publish the verified ${artifact.name} file.")
      completed += artifact.bytes
      transferDone = completed
      transferState = "downloading"
    }
    val temporary = File(directory, ".verified.tmp")
    FileOutputStream(temporary).use { output ->
      output.write((spec.weights.sha256 + "\n").toByteArray(Charsets.US_ASCII))
      output.fd.sync()
    }
    if (!temporary.renameTo(marker(context, spec))) throw IllegalStateException("Could not publish the model verification marker.")
    transferDone = spec.totalBytes
  }

  private fun publishCacheMarker(context: Context, spec: LocalModelSpec) {
    val temporary = File(cacheDir(context, spec), ".tensorfold-ready.tmp")
    FileOutputStream(temporary).use { output ->
      output.write((spec.weights.sha256 + "\n").toByteArray(Charsets.US_ASCII))
      output.fd.sync()
    }
    if (!temporary.renameTo(cacheMarker(context, spec))) throw IllegalStateException("Could not publish the decoded-cache verification marker.")
  }

  private fun prepareCache(context: Context, spec: LocalModelSpec, engine: File, nativeDir: File, token: Long) {
    if (cacheReady(context, spec)) return
    val log = File(root(context), "server.log")
    if (log.length() > 2L shl 20) log.delete()
    val command = listOf(
      engine.absolutePath,
      "cpu",
      File(modelDir(context, spec), "model.safetensors").absolutePath,
      "0",
      "6",
      cacheDir(context, spec).absolutePath,
      spec.weights.sha256,
      "1",
      "1",
      "prepare-cache",
      "1",
    )
    val child = ProcessBuilder(command)
      .directory(root(context))
      .redirectErrorStream(true)
      .apply { environment()["LD_LIBRARY_PATH"] = nativeDir.absolutePath }
      .start()
    synchronized(lock) {
      if (generation != token) {
        child.destroyForcibly()
        throw InterruptedException("Model startup was cancelled.")
      }
      process = child
    }
    child.inputStream.bufferedReader().useLines { lines ->
      FileOutputStream(log, true).bufferedWriter().use { output ->
        for (line in lines) {
          output.appendLine("cache: $line")
          output.flush()
        }
      }
    }
    val exit = child.waitFor()
    synchronized(lock) { if (process === child) process = null }
    if (generation != token) throw InterruptedException("Model startup was cancelled.")
    if (exit != 0) throw IllegalStateException("TensorFold could not construct the decoded cache. ${logTail(context)}")
    publishCacheMarker(context, spec)
  }


  fun start(context: Context, id: String) {
    val app = context.applicationContext
    val spec = ModelCatalog.require(id)
    val batteryTemperature = HardwareProbe.batteryTemperature(app)
    if (batteryTemperature != null && batteryTemperature >= 43.0) throw IllegalStateException("The phone is ${batteryTemperature} C; let it cool below 43 C before loading a model.")
    if (!installed(app, spec)) throw IllegalStateException("Download and verify this model before starting it.")
    val hardware = HardwareProbe.inspect(app)
    val backend = hardware["backend"] as String
    if (backend == "unsupported") throw IllegalStateException("This phone does not provide the required arm64 runtime.")
    if ((hardware["totalRamBytes"] as Double).toLong() < spec.minimumRamBytes) throw IllegalStateException("This phone does not have enough system memory for ${spec.id}.")
    val cacheRemaining = (spec.decodedCacheBytes - cacheBytes(app, spec)).coerceAtLeast(0)
    val freeStorage = StatFs(app.filesDir.absolutePath).availableBytes
    if (freeStorage < cacheRemaining + RESERVE_BYTES) throw IllegalStateException("This model needs ${cacheRemaining + RESERVE_BYTES} free bytes to finish its private decoded cache.")
    val nativeDir = File(app.applicationInfo.nativeLibraryDir)
    val server = File(nativeDir, "libtensorfold_openai.so")
    val engine = File(nativeDir, spec.engineLibrary)
    if (!server.canExecute() || !engine.canExecute()) throw IllegalStateException("This APK does not contain the TensorFold Android runtime for this model.")
    val localAuth = newAuthToken()
    val token = synchronized(lock) {
      stopLocked()
      generation += 1
      engineState = "starting"
      activeModel = spec.id
      runtimeError = null
      authToken = localAuth
      generation
    }
    executor.execute {
      try {
        if (!root(app).mkdirs() && !root(app).isDirectory) throw IllegalStateException("Could not create the private TensorFold directory.")
        if (!cacheDir(app, spec).mkdirs() && !cacheDir(app, spec).isDirectory) throw IllegalStateException("Could not create the private decoded-cache directory.")
        prepareCache(app, spec, engine, nativeDir, token)
        val command = listOf(
          server.absolutePath,
          engine.absolutePath,
          File(modelDir(app, spec), "model.safetensors").absolutePath,
          cacheDir(app, spec).absolutePath,
          spec.weights.sha256,
          "none",
          modelDir(app, spec).absolutePath,
          nativeDir.absolutePath,
          PORT.toString(),
          "low_balanced",
          "none",
          backend,
          "owned",
          localAuth,
        )
        val next = ProcessBuilder(command)
          .directory(root(app))
          .redirectErrorStream(true)
          .start()
        synchronized(lock) {
          if (generation != token) {
            next.destroyForcibly()
            return@execute
          }
          process = next
        }
        captureLog(app, next, token)
        val deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(20)
        while (System.nanoTime() < deadline && generation == token && next.isAlive) {
          val health = getJson("/health")
          if (health?.get("owned_parent") == true && health["auth_required"] == true && health["backend"] == backend) {
            engineState = "ready"
            return@execute
          }
          Thread.sleep(500)
        }
        if (generation == token) throw IllegalStateException(if (next.isAlive) "TensorFold model startup timed out." else "TensorFold exited during model startup. ${logTail(app)}")
      } catch (failure: Throwable) {
        if (generation == token) {
          engineState = "failed"
          runtimeError = failure.message ?: failure.javaClass.simpleName
          synchronized(lock) { process?.destroyForcibly(); process = null; authToken = null }
        }
      }
    }
  }

  private fun captureLog(context: Context, child: Process, token: Long) {
    val file = File(root(context), "server.log")
    if (file.length() > 2L shl 20) file.delete()
    thread(name = "tensorfold-local-log", isDaemon = true) {
      var readFailure: IOException? = null
      try {
        child.inputStream.bufferedReader().useLines { lines ->
          FileOutputStream(file, true).bufferedWriter().use { output ->
            for (line in lines) {
              output.appendLine(line)
              output.flush()
            }
          }
        }
      } catch (failure: IOException) {
        readFailure = failure
      }
      if (generation == token && engineState == "ready") {
        engineState = "failed"
        runtimeError = if (readFailure == null) {
          "TensorFold stopped unexpectedly. ${logTail(context)}"
        } else {
          "TensorFold log stream failed: ${readFailure.message}. ${logTail(context)}"
        }
      }
    }
  }

  private fun logTail(context: Context): String {
    val file = File(root(context), "server.log")
    if (!file.isFile) return "No runtime log was written."
    return file.readLines().takeLast(8).joinToString(" | ").takeLast(2_000)
  }

  fun stop() = synchronized(lock) {
    generation += 1
    stopLocked()
    engineState = "stopped"
    activeModel = null
    runtimeError = null
    authToken = null
  }

  private fun stopLocked() {
    val child = process ?: return
    child.destroy()
    if (!child.waitFor(5, TimeUnit.SECONDS)) child.destroyForcibly()
    process = null
  }

  private fun localRequest(path: String): Request.Builder = Request.Builder()
    .url("http://127.0.0.1:$PORT$path")
    .apply { authToken?.let { header("Authorization", "Bearer $it") } }

  private fun getJson(path: String): Map<String, Any?>? = try {
    val request = localRequest(path).build()
    http.newCall(request).execute().use { response ->
      if (!response.isSuccessful) return null
      val body = response.body?.string() ?: return null
      jsonObjectToMap(JSONObject(body))
    }
  } catch (_: Throwable) { null }

  fun chat(context: Context, messages: List<Map<String, String>>, maxTokens: Int, temperature: Double): Map<String, Any?> {
    if (engineState != "ready") throw IllegalStateException("Start a local model before sending a message.")
    val batteryTemperature = HardwareProbe.batteryTemperature(context)
    if (batteryTemperature != null && batteryTemperature >= 43.0) throw IllegalStateException("The phone is ${batteryTemperature} C; let it cool below 43 C before generating.")
    if (messages.isEmpty() || messages.size > 32) throw IllegalArgumentException("A chat needs between 1 and 32 messages.")
    if (maxTokens !in 1..1024) throw IllegalArgumentException("Output tokens must be between 1 and 1024.")
    if (!temperature.isFinite() || temperature !in 0.0..2.0) throw IllegalArgumentException("Temperature must be between 0 and 2.")
    val body = JSONObject().apply {
      put("model", "local")
      put("messages", JSONArray().apply {
        messages.forEach { message ->
          val role = message["role"] ?: throw IllegalArgumentException("Each message needs a role.")
          val content = message["content"] ?: throw IllegalArgumentException("Each message needs content.")
          if (role !in listOf("system", "user", "assistant") || content.length > 32_768) throw IllegalArgumentException("The chat message is outside the local safety bounds.")
          put(JSONObject().put("role", role).put("content", content))
        }
      })
      put("max_tokens", maxTokens)
      put("temperature", temperature)
      put("thinking", false)
      put("stream", true)
    }
    val request = localRequest("/v1/chat/completions")
      .post(body.toString().toRequestBody("application/json".toMediaType())).build()
    val content = StringBuilder()
    var runtime: Map<String, Any?>? = null
    http.newCall(request).execute().use { response ->
      if (!response.isSuccessful) throw IllegalStateException("TensorFold returned HTTP ${response.code}: ${response.body?.string()?.take(1_000)}")
      val source = response.body?.source() ?: throw IllegalStateException("TensorFold returned no stream.")
      while (!source.exhausted()) {
        val line = source.readUtf8Line() ?: break
        if (!line.startsWith("data: ")) continue
        val data = line.removePrefix("data: ")
        if (data == "[DONE]") break
        val event = JSONObject(data)
        val choices = event.optJSONArray("choices")
        if (choices != null && choices.length() > 0) {
          val delta = choices.getJSONObject(0).optJSONObject("delta")
          val piece = delta?.optString("content", "") ?: ""
          if (content.length + piece.length > MAX_RESPONSE_CHARS) throw IllegalStateException("TensorFold response exceeded the app limit.")
          content.append(piece)
        }
        event.optJSONObject("runtime")?.let { runtime = jsonObjectToMap(it) }
      }
    }
    val visible = ModelOutput.visibleText(content.toString())
    if (visible.isEmpty()) throw IllegalStateException("TensorFold returned an empty reply.")
    return linkedMapOf("content" to visible, "runtime" to runtime)
  }

  fun benchmark(context: Context): Map<String, Any?> {
    if (engineState != "ready") throw IllegalStateException("Start a local model before benchmarking.")
    val beforeHealth = getJson("/health")
    val temperatureBefore = HardwareProbe.batteryTemperature(context)
    val workloads = linkedMapOf<String, Map<String, Any?>>()
    val prompts = linkedMapOf(
      "code" to "Write a compact Kotlin function that validates a SHA-256 lowercase hexadecimal string without regular expressions. Return only code.",
      "chat" to "Explain in practical terms why an Android app should bind a local model server only to loopback.",
    )
    for ((name, prompt) in prompts) {
      val request = listOf(mapOf("role" to "user", "content" to prompt))
      chat(context, request, 64, 0.0) // Warm-up is excluded.
      val rates = mutableListOf<Double>()
      val prefills = mutableListOf<Double>()
      val durations = mutableListOf<Double>()
      repeat(3) {
        val runtime = chat(context, request, 64, 0.0)["runtime"] as? Map<*, *>
          ?: throw IllegalStateException("TensorFold omitted benchmark runtime measurements.")
        rates += (runtime["tokens_per_second"] as? Number)?.toDouble()
          ?: throw IllegalStateException("TensorFold omitted benchmark delivery speed.")
        prefills += (runtime["prefill_seconds"] as? Number)?.toDouble()
          ?: throw IllegalStateException("TensorFold omitted benchmark prefill time.")
        durations += (runtime["request_seconds"] as? Number)?.toDouble()
          ?: throw IllegalStateException("TensorFold omitted benchmark request duration.")
      }
      workloads[name] = linkedMapOf(
        "medianTokensPerSecond" to rates.sorted()[1],
        "medianPrefillSeconds" to prefills.sorted()[1],
        "medianRequestSeconds" to durations.sorted()[1],
        "repetitions" to 3,
        "outputTokens" to 64,
        "temperature" to 0,
      )
    }
    val temperatureAfter = HardwareProbe.batteryTemperature(context)
    return linkedMapOf(
      "suite" to "tensorfold-android-quick-v1",
      "modelId" to activeModel,
      "backend" to beforeHealth?.get("backend"),
      "temperatureBeforeC" to temperatureBefore,
      "temperatureAfterC" to temperatureAfter,
      "workloads" to workloads,
    )
  }

  private fun jsonObjectToMap(value: JSONObject): Map<String, Any?> = value.keys().asSequence().associateWith { key ->
    when (val item = value.get(key)) {
      JSONObject.NULL -> null
      is JSONObject -> jsonObjectToMap(item)
      is JSONArray -> (0 until item.length()).map { index -> item.get(index) }
      else -> item
    }
  }
}
