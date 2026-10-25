#include <jni.h>
#include <dlfcn.h>

namespace {

using LiteRtCompiledModelIsNonCpuFullyAccelerated = int (*)(void*, bool*);

constexpr jlong kSuccess = 0;
constexpr jlong kMissingHandle = -1;
constexpr jlong kLibraryUnavailable = -2;
constexpr jlong kSymbolUnavailable = -3;
constexpr jlong kAttestationFailed = -4;
constexpr jlong kNotFullyAccelerated = -5;

jlongArray result(JNIEnv* env, jlong status, jlong accelerated) {
  const jlong values[] = {status, accelerated};
  jlongArray array = env->NewLongArray(2);
  if (array != nullptr) env->SetLongArrayRegion(array, 0, 2, values);
  return array;
}

}  // namespace

extern "C" JNIEXPORT jlongArray JNICALL
Java_expo_modules_tensorfoldlocalruntime_TensorG5Native_nativeAttest(
    JNIEnv* env, jclass, jlong compiled_model_handle) {
  if (compiled_model_handle == 0) return result(env, kMissingHandle, 0);

  void* library = dlopen("libLiteRt.so", RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) return result(env, kLibraryUnavailable, 0);

  auto attest = reinterpret_cast<LiteRtCompiledModelIsNonCpuFullyAccelerated>(
      dlsym(library, "LiteRtCompiledModelIsNonCpuFullyAccelerated"));
  if (attest == nullptr) {
    dlclose(library);
    return result(env, kSymbolUnavailable, 0);
  }

  bool fully_accelerated = false;
  const int status = attest(reinterpret_cast<void*>(compiled_model_handle), &fully_accelerated);
  dlclose(library);
  if (status != 0) return result(env, kAttestationFailed, 0);
  return result(env, fully_accelerated ? kSuccess : kNotFullyAccelerated,
                fully_accelerated ? 1 : 0);
}
