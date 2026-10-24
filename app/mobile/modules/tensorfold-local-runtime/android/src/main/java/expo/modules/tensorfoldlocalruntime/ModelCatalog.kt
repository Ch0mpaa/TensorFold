package expo.modules.tensorfoldlocalruntime

internal data class ModelArtifact(
  val name: String,
  val bytes: Long,
  val sha256: String,
) {
  fun url(repo: String, revision: String) = "https://huggingface.co/$repo/resolve/$revision/$name?download=true"
}

internal data class LocalModelSpec(
  val id: String,
  val repo: String,
  val profile: String,
  val engineLibrary: String,
  val minimumRamBytes: Long,
  val decodedCacheBytes: Long,
  val artifacts: List<ModelArtifact>,
  val revision: String = "main",
  val identitySha256: String? = null,
  val sharded: Boolean = false,
  val qualcommOnly: Boolean = false,
) {
  val totalBytes: Long get() = artifacts.sumOf { it.bytes }
  val weights: ModelArtifact get() = artifacts.first { it.name == "model.safetensors" || it.name.endsWith(".safetensors") }
  val modelIdentitySha256: String get() = identitySha256 ?: weights.sha256
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
    LocalModelSpec(
      id = "Qwen3.5-35B-A3B-exl3-2.54bpw",
      decodedCacheBytes = 0,
      repo = "turboderp/Qwen3.5-35B-A3B-exl3",
      revision = "2.54bpw",
      profile = "qwen3.5-35b-a3b",
      engineLibrary = "libtensorfold_engine_moe.so",
      minimumRamBytes = 14L shl 30,
      identitySha256 = "1539df4e836d11bceba0e54a57696eedc30f1b5df040b20fc3241a0fed1c7199",
      sharded = true,
      qualcommOnly = true,
      artifacts = listOf(
        ModelArtifact("config.json", 3_831, "ebdef7edaa3a4ebb48f1d62845983846019111f563d654771b6db024c3473353"),
        ModelArtifact("tokenizer.json", 12_807_982, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        ModelArtifact("tokenizer_config.json", 16_710, "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"),
        ModelArtifact("chat_template.jinja", 7_756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
        ModelArtifact("quantization_config.json", 40_315_310, "e73c30c1dc4a5e14ae4339049275ed5f271ead33b6394721145a2cfb7b1a5ba9"),
        ModelArtifact("model.safetensors.index.json", 13_401_028, "bb40179d63838c3fdc763007173fe5f800fa51c769d886986450eca6e24807b0"),
        ModelArtifact("model-00001-of-00002.safetensors", 8_469_182_534, "983c0126e41baa165ab9af07064d959605d2f1d5d0730eba040d4cfb3e2e98c3"),
        ModelArtifact("model-00002-of-00002.safetensors", 4_518_477_224, "dbbc2987b3b957c62c9cc8121c32f99e4169c9d8539c0ab722cf7b1942bb41a1"),
      ),
    ),
    LocalModelSpec(
      id = "Qwen3.5-35B-A3B-exl3-4.00bpw",
      decodedCacheBytes = 0,
      repo = "turboderp/Qwen3.5-35B-A3B-exl3",
      revision = "8a68a1aca688b768853bad4d926c0136ff62286c",
      profile = "qwen3.5-35b-a3b",
      engineLibrary = "libtensorfold_engine_moe.so",
      minimumRamBytes = 14L shl 30,
      identitySha256 = "af55d67bc3ccd363896a4b132d2c7c4335cc1e91e54cdc55e32796109fbeba5d",
      sharded = true,
      qualcommOnly = true,
      artifacts = listOf(
        ModelArtifact("config.json", 3_830, "0f5be0ff8050ea2912b8cd6fcee776d756db753c8567658a67f98dac395b31b6"),
        ModelArtifact("tokenizer.json", 12_807_982, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        ModelArtifact("tokenizer_config.json", 16_710, "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"),
        ModelArtifact("chat_template.jinja", 7_756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
        ModelArtifact("quantization_config.json", 40_315_309, "0fffd0ab7d2bdc1d9017434429f7f93a3fc227a33307a82f3366e908ad9fbb9a"),
        ModelArtifact("model.safetensors.index.json", 13_401_028, "5330149b9d1d12f4cc5ea3e5703196bce0b8f1e97522112b3aad7423635cb3fe"),
        ModelArtifact("model-00001-of-00003.safetensors", 8_258_803_070, "ae09fd598b8c0bd651904de8ec44cfbd8d7832ab0ae9de66df131cda40959be9"),
        ModelArtifact("model-00002-of-00003.safetensors", 8_518_626_585, "e47ffb46fe8dd9a97b97ce7e6e0a552bda20ac6eb9aae6ac8a19cb3c6549b92b"),
        ModelArtifact("model-00003-of-00003.safetensors", 2_552_016_355, "4202ac8815e0c91c183f8a2948c1abbe2461c01fbff6f8d73f5602857178e45a"),
      ),
    ),
  )

  fun require(id: String): LocalModelSpec = all.firstOrNull { it.id == id }
    ?: throw IllegalArgumentException("This model is not in the qualified TensorFold Android catalog.")
}
