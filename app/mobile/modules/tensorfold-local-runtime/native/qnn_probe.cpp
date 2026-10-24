#include <jni.h>
#include <dlfcn.h>
#include <time.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <QNN/QnnInterface.h>
#include <QNN/QnnOpDef.h>

namespace {
constexpr uint8_t kInput[] = {1, 2, 3, 4};
constexpr uint8_t kWeights[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 2};
constexpr uint8_t kExpected[] = {34, 36, 50};
constexpr float kTolerance = 0.5f;
static_assert(kExpected[0] == kInput[0] * kWeights[0] + kInput[1] * kWeights[3] + kInput[2] * kWeights[6] + kInput[3] * kWeights[9]);
static_assert(kExpected[1] == kInput[0] * kWeights[1] + kInput[1] * kWeights[4] + kInput[2] * kWeights[7] + kInput[3] * kWeights[10]);
static_assert(kExpected[2] == kInput[0] * kWeights[2] + kInput[1] * kWeights[5] + kInput[2] * kWeights[8] + kInput[3] * kWeights[11]);

constexpr jint kLoadFailure = -100;
constexpr jint kProviderFailure = -101;
constexpr jint kApiFailure = -102;
constexpr jint kBackendFailure = -103;
constexpr jint kDeviceFailure = -104;
constexpr jint kContextFailure = -105;
constexpr jint kProfileFailure = -106;
constexpr jint kGraphFailure = -107;
constexpr jint kTensorFailure = -108;
constexpr jint kNodeFailure = -109;
constexpr jint kFinalizeFailure = -110;
constexpr jint kExecuteFailure = -111;
constexpr jint kParityFailure = -112;

void setQuantizedU8(Qnn_Tensor_t* tensor, const char* name, Qnn_TensorType_t type,
                    uint32_t* dimensions, uint32_t rank, void* data, uint64_t size) {
  *tensor = QNN_TENSOR_INIT;
  tensor->v1.name = name;
  tensor->v1.type = type;
  tensor->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor->v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
  tensor->v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  tensor->v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  tensor->v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
  tensor->v1.quantizeParams.scaleOffsetEncoding.offset = 0;
  tensor->v1.rank = rank;
  tensor->v1.dimensions = dimensions;
  tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
  tensor->v1.clientBuf.data = data;
  tensor->v1.clientBuf.dataSize = size;
}

double elapsedMs(const timespec& start, const timespec& end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

jdoubleArray result(JNIEnv* env, jint status, double elapsed_ms, double max_abs_error) {
  jdouble values[] = {static_cast<jdouble>(status), elapsed_ms, max_abs_error};
  jdoubleArray output = env->NewDoubleArray(3);
  if (output != nullptr) env->SetDoubleArrayRegion(output, 0, 3, values);
  return output;
}
}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_expo_modules_tensorfoldlocalruntime_QnnProbe_nativeProjection(
    JNIEnv* env, jclass, jstring native_dir, jstring stub_name) {
  if (native_dir == nullptr || stub_name == nullptr) return result(env, kLoadFailure, 0.0, NAN);
  const char* dir = env->GetStringUTFChars(native_dir, nullptr);
  const char* stub = env->GetStringUTFChars(stub_name, nullptr);
  if (dir == nullptr || stub == nullptr) {
    if (dir != nullptr) env->ReleaseStringUTFChars(native_dir, dir);
    if (stub != nullptr) env->ReleaseStringUTFChars(stub_name, stub);
    return result(env, kLoadFailure, 0.0, NAN);
  }

  setenv("ADSP_LIBRARY_PATH", dir, 1);
  void* rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (rpc == nullptr) rpc = dlopen("libadsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  void* htp_stub = rpc == nullptr ? nullptr : dlopen(stub, RTLD_NOW | RTLD_GLOBAL);
  if (rpc == nullptr || htp_stub == nullptr) {
    if (htp_stub != nullptr) dlclose(htp_stub);
    if (rpc != nullptr) dlclose(rpc);
    env->ReleaseStringUTFChars(native_dir, dir);
    env->ReleaseStringUTFChars(stub_name, stub);
    return result(env, kLoadFailure, 0.0, NAN);
  }

  const QnnInterface_t** providers = nullptr;
  uint32_t provider_count = 0;
  Qnn_ErrorHandle_t api_status = QnnInterface_getProviders(&providers, &provider_count);
  const QnnInterface_t* provider = (api_status == QNN_SUCCESS && provider_count > 0) ? providers[0] : nullptr;
  if (provider == nullptr) {
    dlclose(htp_stub); dlclose(rpc);
    env->ReleaseStringUTFChars(native_dir, dir); env->ReleaseStringUTFChars(stub_name, stub);
    return result(env, kProviderFailure, 0.0, NAN);
  }
  const auto& qnn = provider->QNN_INTERFACE_VER_NAME;
  if (qnn.backendCreate == nullptr || qnn.backendFree == nullptr || qnn.deviceCreate == nullptr ||
      qnn.deviceFree == nullptr || qnn.contextCreate == nullptr || qnn.contextFree == nullptr ||
      qnn.profileCreate == nullptr || qnn.profileFree == nullptr || qnn.graphCreate == nullptr ||
      qnn.tensorCreateGraphTensor == nullptr || qnn.graphAddNode == nullptr ||
      qnn.graphFinalize == nullptr || qnn.graphExecute == nullptr) {
    dlclose(htp_stub); dlclose(rpc);
    env->ReleaseStringUTFChars(native_dir, dir); env->ReleaseStringUTFChars(stub_name, stub);
    return result(env, kApiFailure, 0.0, NAN);
  }

  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_ProfileHandle_t profile = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  jint outcome = 0;
  double execution_ms = 0.0;
  double max_error = NAN;

  if (qnn.backendCreate(nullptr, nullptr, &backend) != QNN_SUCCESS) outcome = kBackendFailure;
  if (outcome == 0 && qnn.deviceCreate(nullptr, nullptr, &device) != QNN_SUCCESS) outcome = kDeviceFailure;
  if (outcome == 0 && qnn.contextCreate(backend, device, nullptr, &context) != QNN_SUCCESS) outcome = kContextFailure;
  if (outcome == 0 && qnn.profileCreate(backend, QNN_PROFILE_LEVEL_BASIC, &profile) != QNN_SUCCESS) outcome = kProfileFailure;

  uint32_t input_dims[] = {1, 4};
  uint32_t weight_dims[] = {4, 3};
  uint32_t output_dims[] = {1, 3};
  uint8_t output_data[sizeof(kExpected)] = {};
  Qnn_Tensor_t input;
  Qnn_Tensor_t weights;
  Qnn_Tensor_t output;
  setQuantizedU8(&input, "projection_input", QNN_TENSOR_TYPE_APP_WRITE, input_dims, 2, nullptr, 0);
  setQuantizedU8(&weights, "projection_weights", QNN_TENSOR_TYPE_STATIC, weight_dims, 2,
                 const_cast<uint8_t*>(kWeights), sizeof(kWeights));
  setQuantizedU8(&output, "projection_output", QNN_TENSOR_TYPE_APP_READ, output_dims, 2, nullptr, 0);

  if (outcome == 0 && qnn.graphCreate(context, "tensorfold_projection_probe", nullptr, &graph) != QNN_SUCCESS) outcome = kGraphFailure;
  if (outcome == 0 && (qnn.tensorCreateGraphTensor(graph, &input) != QNN_SUCCESS ||
                       qnn.tensorCreateGraphTensor(graph, &weights) != QNN_SUCCESS ||
                       qnn.tensorCreateGraphTensor(graph, &output) != QNN_SUCCESS)) outcome = kTensorFailure;
  Qnn_Tensor_t node_inputs[] = {input, weights};
  Qnn_Tensor_t node_outputs[] = {output};
  Qnn_OpConfig_t matmul = QNN_OPCONFIG_INIT;
  matmul.v1.name = "tensorfold_projection_matmul";
  matmul.v1.packageName = "qti.aisw";
  matmul.v1.typeName = QNN_OP_MAT_MUL;
  matmul.v1.numOfInputs = 2;
  matmul.v1.inputTensors = node_inputs;
  matmul.v1.numOfOutputs = 1;
  matmul.v1.outputTensors = node_outputs;
  if (outcome == 0 && qnn.graphAddNode(graph, matmul) != QNN_SUCCESS) outcome = kNodeFailure;
  if (outcome == 0 && qnn.graphFinalize(graph, profile, nullptr) != QNN_SUCCESS) outcome = kFinalizeFailure;

  input.v1.clientBuf.data = const_cast<uint8_t*>(kInput);
  input.v1.clientBuf.dataSize = sizeof(kInput);
  output.v1.clientBuf.data = output_data;
  output.v1.clientBuf.dataSize = sizeof(output_data);
  if (outcome == 0) {
    timespec start{}, end{};
    clock_gettime(CLOCK_MONOTONIC, &start);
    if (qnn.graphExecute(graph, &input, 1, &output, 1, profile, nullptr) != QNN_SUCCESS) outcome = kExecuteFailure;
    clock_gettime(CLOCK_MONOTONIC, &end);
    execution_ms = elapsedMs(start, end);
  }
  if (outcome == 0) {
    max_error = 0.0;
    for (size_t i = 0; i < sizeof(kExpected); ++i) {
      const double error = std::fabs(static_cast<double>(output_data[i]) - kExpected[i]);
      if (error > max_error) max_error = error;
    }
    if (max_error > kTolerance) outcome = kParityFailure;
  }

  if (profile != nullptr) qnn.profileFree(profile);
  if (context != nullptr) qnn.contextFree(context, nullptr);
  if (device != nullptr) qnn.deviceFree(device);
  if (backend != nullptr) qnn.backendFree(backend);
  dlclose(htp_stub);
  dlclose(rpc);
  env->ReleaseStringUTFChars(native_dir, dir);
  env->ReleaseStringUTFChars(stub_name, stub);
  return result(env, outcome, execution_ms, max_error);
}
