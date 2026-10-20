package expo.modules.tensorfoldlocalruntime

import java.io.File
import java.net.URI
import java.security.MessageDigest

internal object DownloadPolicy {
  fun allows(uri: URI): Boolean {
    if (uri.scheme != "https" || uri.userInfo != null || uri.port != -1) return false
    val host = uri.host?.lowercase() ?: return false
    return host == "huggingface.co" || host.endsWith(".huggingface.co") ||
      host == "hf.co" || host.endsWith(".hf.co") ||
      host == "xethub.hf.co" || host.endsWith(".xethub.hf.co")
  }
}

internal object ModelIntegrity {
  fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered(1 shl 20).use { input ->
      val buffer = ByteArray(1 shl 20)
      while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        digest.update(buffer, 0, count)
      }
    }
    return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
  }

  fun matches(file: File, artifact: ModelArtifact): Boolean =
    file.isFile && file.length() == artifact.bytes && sha256(file) == artifact.sha256
}

internal object ModelOutput {
  fun visibleText(raw: String): String = raw
    .substringBefore("<|im_end|>")
    .substringBefore("<|endoftext|>")
    .trimEnd()
}
