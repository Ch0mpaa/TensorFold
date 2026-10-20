package expo.modules.tensorfoldlocalruntime

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class TensorFoldLocalRuntimeModule : Module() {
  private fun context() = requireNotNull(appContext.reactContext?.applicationContext) {
    "TensorFold local runtime is not attached to the Android application."
  }

  override fun definition() = ModuleDefinition {
    Name("TensorFoldLocalRuntime")

    AsyncFunction("capabilities") {
      HardwareProbe.inspect(context())
    }
    AsyncFunction("models") {
      RuntimeManager.models(context())
    }
    AsyncFunction("status") {
      RuntimeManager.status(context())
    }
    AsyncFunction("downloadModel") { id: String ->
      RuntimeManager.beginDownload(context(), id)
      RuntimeManager.status(context())
    }
    AsyncFunction("cancelDownload") {
      RuntimeManager.cancelDownload()
      RuntimeManager.status(context())
    }
    AsyncFunction("removeModel") { id: String ->
      RuntimeManager.removeModel(context(), id)
      RuntimeManager.status(context())
    }
    AsyncFunction("start") { id: String ->
      RuntimeManager.start(context(), id)
      RuntimeManager.status(context())
    }
    AsyncFunction("stop") {
      RuntimeManager.stop()
      RuntimeManager.status(context())
    }
    AsyncFunction("benchmark") {
      RuntimeManager.benchmark(context())
    }
    AsyncFunction("chat") { messages: List<Map<String, String>>, maxTokens: Int, temperature: Double ->
      RuntimeManager.chat(context(), messages, maxTokens, temperature)
    }
  }
}
