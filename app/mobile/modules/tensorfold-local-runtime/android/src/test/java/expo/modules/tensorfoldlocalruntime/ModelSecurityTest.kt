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
}
