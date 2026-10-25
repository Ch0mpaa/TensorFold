package expo.modules.tensorfoldlocalruntime

import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.File
import java.net.URI
import java.nio.file.Files

class ModelSecurityTest {
  @Test fun downloadPolicyAcceptsOnlyApprovedHttpsHosts() {
    assertTrue(DownloadPolicy.allows(URI("https://huggingface.co/owner/model")))
    assertTrue(DownloadPolicy.allows(URI("https://cdn-lfs.hf.co/file")))
    assertTrue(DownloadPolicy.allows(URI("https://cas-bridge.xethub.hf.co/file")))
    assertFalse(DownloadPolicy.allows(URI("http://huggingface.co/file")))
    assertFalse(DownloadPolicy.allows(URI("https://huggingface.co:444/file")))
    assertFalse(DownloadPolicy.allows(URI("https://user@huggingface.co/file")))
    assertFalse(DownloadPolicy.allows(URI("https://huggingface.co.attacker.example/file")))
    assertFalse(DownloadPolicy.allows(URI("https://example.com/file")))
  }

  @Test fun chipsetClassificationSeparatesQualcommAndPixelTensor() {
    assertTrue(HardwareProbe.classifyPlatform("QTI", "SM8750", "OnePlus", "qcom") == "qualcomm")
    assertTrue(HardwareProbe.classifyPlatform("Google", "Tensor G4", "Google", "zuma") == "pixel-tensor")
    assertTrue(HardwareProbe.classifyPlatform("MediaTek", "Dimensity", "Other", "mtk") == "android-arm64")
  }

  @Test fun tensorG5ProbeRequiresNativeAttestation() {
    val unattested = TensorG5ProjectionResult(
      -4, true, false, 1.0, 1.0, 1.0, 0.0,
    )
    assertFalse(unattested.available)
    assertEquals("LiteRT reports that this model was not fully accelerated outside CPU.", unattested.reason())
  }

  @Test fun tensorG5ProbeRejectsParityFailureAfterAttestation() {
    val divergent = TensorG5ProjectionResult(
      0, true, true, 1.0, 1.0, 1.0,
      TensorG5ProjectionResult.parityTolerance + 0.001,
    )
    assertFalse(divergent.available)
    assertTrue(divergent.reason()!!.contains("CPU oracle"))
  }

  @Test fun tensorG5ProbeReportsMissingPinnedModel() {
    assertEquals(16_371_837L, TensorG5ProjectionResult.probeBytes)
    assertEquals("c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0", TensorG5ProjectionResult.probeSha256)
    val missing = TensorG5ProjectionResult.unavailable(-12, true)
    assertFalse(missing.available)
    assertTrue(missing.reason()!!.contains("required selfie_multiclass.tflite"))
  }

  @Test fun chatOutputStopsAtModelControlTokens() {
    assertEquals("LOCAL_OK", ModelOutput.visibleText("LOCAL_OK<|im_end|>\\n<|endoftext|><|im_start|>user"))
    assertEquals("ordinary reply", ModelOutput.visibleText("ordinary reply  "))
  }

  @Test fun modelIntegrityRequiresExactSizeAndDigest() {
    val directory = Files.createTempDirectory("tensorfold-model-integrity").toFile()
    try {
      val file = File(directory, "artifact")
      file.writeText("abc")
      val artifact = ModelArtifact("artifact", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
      assertTrue(ModelIntegrity.matches(file, artifact))
      file.appendText("d")
      assertFalse(ModelIntegrity.matches(file, artifact))
      file.writeText("abd")
      assertFalse(ModelIntegrity.matches(file, artifact))
    } finally {
      directory.deleteRecursively()
    }
  }

  @Test fun catalogRefusesArbitraryOrTraversalIds() {
    assertThrows(IllegalArgumentException::class.java) { ModelCatalog.require("../foreign") }
    assertThrows(IllegalArgumentException::class.java) { ModelCatalog.require("owner/unqualified") }
  }

  @Test fun admittedMoeProfilesPinDistinctVerifiedArtifacts() {
    val low = ModelCatalog.require("Qwen3.5-35B-A3B-exl3-2.54bpw")
    val high = ModelCatalog.require("Qwen3.5-35B-A3B-exl3-4.00bpw")
    assertEquals("1539df4e836d11bceba0e54a57696eedc30f1b5df040b20fc3241a0fed1c7199", low.modelIdentitySha256)
    assertEquals("af55d67bc3ccd363896a4b132d2c7c4335cc1e91e54cdc55e32796109fbeba5d", high.modelIdentitySha256)
    assertEquals(2, low.artifacts.count { it.name.endsWith(".safetensors") })
    assertEquals(3, high.artifacts.count { it.name.endsWith(".safetensors") })
    assertEquals(3_830L, high.artifacts.single { it.name == "config.json" }.bytes)
    assertEquals(40_315_309L, high.artifacts.single { it.name == "quantization_config.json" }.bytes)
    assertEquals(19_329_446_010L, high.artifacts.filter { it.name.endsWith(".safetensors") }.sumOf { it.bytes })
  }
}
