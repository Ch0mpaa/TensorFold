#include <jni.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>

extern "C" JNIEXPORT jint JNICALL
Java_expo_modules_tensorfoldlocalruntime_QnnProbe_nativeCalculator(
    JNIEnv *env, jclass, jstring native_dir, jstring stub_name, jint count) {
  if (native_dir == nullptr || stub_name == nullptr || count < 2 || count > 1024) return -100;
  const char *dir = env->GetStringUTFChars(native_dir, nullptr);
  const char *stub = env->GetStringUTFChars(stub_name, nullptr);
  if (dir == nullptr || stub == nullptr) {
    if (dir != nullptr) env->ReleaseStringUTFChars(native_dir, dir);
    if (stub != nullptr) env->ReleaseStringUTFChars(stub_name, stub);
    return -101;
  }
  setenv("ADSP_LIBRARY_PATH", dir, 1);
  void *rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (rpc == nullptr) rpc = dlopen("libadsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  void *handle = rpc == nullptr ? nullptr : dlopen(stub, RTLD_NOW | RTLD_LOCAL);
  using Calculator = int (*)(int);
  Calculator calculator = handle == nullptr ? nullptr :
      reinterpret_cast<Calculator>(dlsym(handle, "Qnn_calculatorTest"));
  const int result = rpc == nullptr ? -103 : handle == nullptr ? -104 :
      calculator == nullptr ? -105 : calculator(count);
  if (handle != nullptr) dlclose(handle);
  if (rpc != nullptr) dlclose(rpc);
  env->ReleaseStringUTFChars(native_dir, dir);
  env->ReleaseStringUTFChars(stub_name, stub);
  return result;
}
