package expo.modules.tensorfoldlocalruntime

internal data class ModelArtifact(
  val name: String,
  val bytes: Long,
  val sha256: String,
) {
  fun url(repo: String) = "https://huggingface.co/$repo/resolve/main/$name?download=true"
}

internal data class LocalModelSpec(
  val id: String,
  val repo: String,
  val profile: String,
  val engineLibrary: String,
  val minimumRamBytes: Long,
  val decodedCacheBytes: Long,
  val artifacts: List<ModelArtifact>,
) {
  val totalBytes: Long get() = artifacts.sumOf { it.bytes }
  val weights: ModelArtifact get() = artifacts.first { it.name == "model.safetensors" }
}

internal object ModelCatalog {
  private fun common(tokenizerConfigBytes: Long, tokenizerConfigSha: String, templateBytes: Long, templateSha: String) = listOf(
    ModelArtifact("tokenizer.json", 12_807_982, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    ModelArtifact("tokenizer_config.json", tokenizerConfigBytes, tokenizerConfigSha),
    ModelArtifact("chat_template.jinja", templateBytes, templateSha),
  )

  val all = listOf(
    LocalModelSpec(
      decodedCacheBytes = 8_229_362_432,
      id = "Qwen3.5-2B-exl3-8.00bpw",
      repo = "UnstableLlama/Qwen3.5-2B-exl3-8.00bpw",
      profile = "qwen3.5-2b",
      engineLibrary = "libtensorfold_engine_2b.so",
      minimumRamBytes = 8L shl 30,
      artifacts = listOf(
        ModelArtifact("config.json", 3_301, "6d6873295d7f81c0a6c227015dc9640dfb0785679a28abf07135911f0bb7d3fe"),
      ) + common(17_013, "ce3cbcd05aaa91d20c1605c5c46de4bacfe439c1c0efd8b27e6b5489de968f2c", 7_908, "306abaf09cf3a6c0d14128373a72925bda3d7de2616a5642804d9768c7d2f78c") + listOf(
        ModelArtifact("model.safetensors", 3_438_956_902, "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f"),
      ),
    ),
    LocalModelSpec(
      id = "Qwen3.5-4B-exl3-4.00bpw",
      decodedCacheBytes = 21_391_209_472,
      repo = "UnstableLlama/Qwen3.5-4B-exl3-4.00bpw",
      profile = "qwen3.5-4b",
      engineLibrary = "libtensorfold_engine_4b.so",
      minimumRamBytes = 10L shl 30,
      artifacts = listOf(
        ModelArtifact("config.json", 3_447, "528e0e9eece650cba206836fbca1b051cad08885607e053448deb46758247766"),
      ) + common(16_710, "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8", 7_756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715") + listOf(
        ModelArtifact("model.safetensors", 4_212_050_770, "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1"),
      ),
    ),
  )

  fun require(id: String): LocalModelSpec = all.firstOrNull { it.id == id }
    ?: throw IllegalArgumentException("This model is not in the qualified TensorFold Android catalog.")
}
