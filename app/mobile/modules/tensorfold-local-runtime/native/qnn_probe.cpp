#include <jni.h>
#include <dlfcn.h>
#include <time.h>

#include <cmath>
#include <cstdlib>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <vector>


#include <QNN/QnnInterface.h>
#include <QNN/QnnOpDef.h>
#include "qnn_checkpoint_projection.h"
#include "qnn_full_projection.h"
#include "qnn_matmul_bridge.h"

namespace {
namespace checkpoint = tensorfold::qnn_checkpoint_projection;
namespace full = tensorfold::qnn_full_projection;

constexpr uint8_t kSyntheticInput[] = {1, 2, 3, 4};
constexpr uint8_t kSyntheticWeights[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 2};
constexpr float kSyntheticReference[] = {34.0f, 36.0f, 50.0f};

struct ProjectionSpec {
  const uint8_t* input;
  const uint8_t* weights;
  const float* reference;
  uint32_t input_columns;
  uint32_t output_columns;
  float input_scale;
  int32_t input_offset;
  float weight_scale;
  int32_t weight_offset;
  float output_scale;
  int32_t output_offset;
  float tolerance;
};

constexpr ProjectionSpec kSynthetic = {kSyntheticInput, kSyntheticWeights, kSyntheticReference,
    4, 3, 1.0f, 0, 1.0f, 0, 1.0f, 0, 0.5f};
constexpr ProjectionSpec kCheckpoint = {checkpoint::kInput, checkpoint::kWeights,
    checkpoint::kQuantizedReference, checkpoint::kInputColumns, checkpoint::kOutputColumns,
    checkpoint::kInputScale, checkpoint::kInputOffset, checkpoint::kWeightScale,
    checkpoint::kWeightOffset, checkpoint::kOutputScale, checkpoint::kOutputOffset,
    checkpoint::kParityTolerance};
constexpr ProjectionSpec kFull = {full::kInput, full::kWeights, full::kQuantizedReference,
    full::kInputColumns, full::kOutputColumns, full::kInputScale, full::kInputOffset,
    full::kWeightScale, full::kWeightOffset, full::kOutputScale, full::kOutputOffset,
    full::kParityTolerance};

constexpr char kLivePayloadMagic[] = "TFQNNLP1";
constexpr uint32_t kLivePayloadVersion = 1;
constexpr size_t kLivePayloadHeaderBytes = 256;
constexpr size_t kLivePayloadBytes = kLivePayloadHeaderBytes + 128 + 128 * 128 + 128 * sizeof(float) + sizeof(uint64_t);
constexpr char kPinnedModelSha256[] = "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f";
constexpr char kPinnedBlockSha256[] = "4ad47947dfa769a9d7eb2754732e855775bb3df70e66902bb2ee4429d27dc776";
constexpr char kPinnedPayloadSha256[] = "556a06597ce4f7ddd01ed588305e92292cef5cf55cb00f71828a34e95b29795a";

uint32_t readU32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
      (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

int32_t readI32(const uint8_t* bytes) {
  return static_cast<int32_t>(readU32(bytes));
}

uint64_t readU64(const uint8_t* bytes) {
  uint64_t value = 0;
  for (size_t index = 0; index < sizeof(value); ++index) value |= static_cast<uint64_t>(bytes[index]) << (index * 8);
  return value;
}

float readF32(const uint8_t* bytes) {
  const uint32_t bits = readU32(bytes);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint64_t fnv1a64(const uint8_t* bytes, size_t size) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

bool isHexIdentity(const uint8_t* value) {
  for (size_t index = 0; index < 64; ++index) {
    if (!((value[index] >= '0' && value[index] <= '9') ||
          (value[index] >= 'a' && value[index] <= 'f'))) return false;
  }
  return true;
}

struct LivePayload {
  std::vector<uint8_t> bytes;
  std::array<float, 128> reference{};
  ProjectionSpec spec{};
};

int32_t loadLivePayload(const char* path, LivePayload* payload) {
  FILE* file = std::fopen(path, "rb");
  if (file == nullptr) return errno == ENOENT ? -120 : -121;
  payload->bytes.resize(kLivePayloadBytes);
  const size_t read = std::fread(payload->bytes.data(), 1, payload->bytes.size(), file);
  const int trailing = std::fgetc(file);
  std::fclose(file);
  if (read != payload->bytes.size() || trailing != EOF) return -121;
  const uint8_t* bytes = payload->bytes.data();
  if (std::memcmp(bytes, kLivePayloadMagic, 8) != 0 || readU32(bytes + 8) != kLivePayloadVersion ||
      readU32(bytes + 12) != kLivePayloadHeaderBytes || readU32(bytes + 16) != 128 ||
      readU32(bytes + 20) != 128 || readU32(bytes + 244) != 128 ||
      readU32(bytes + 248) != 128 * 128 || readU32(bytes + 252) != 128 * sizeof(float)) return -121;
  if (std::memcmp(bytes + 52, kPinnedModelSha256, 64) != 0 ||
      std::memcmp(bytes + 116, kPinnedBlockSha256, 64) != 0 ||
      std::memcmp(bytes + 180, kPinnedPayloadSha256, 64) != 0) return -122;
  if (!isHexIdentity(bytes + 116) || !isHexIdentity(bytes + 180) ||
      readU64(bytes + payload->bytes.size() - sizeof(uint64_t)) !=
          fnv1a64(bytes, payload->bytes.size() - sizeof(uint64_t))) return -121;
  const float input_scale = readF32(bytes + 24);
  const float weight_scale = readF32(bytes + 28);
  const float output_scale = readF32(bytes + 32);
  const float tolerance = readF32(bytes + 36);
  if (!std::isfinite(input_scale) || !std::isfinite(weight_scale) || !std::isfinite(output_scale) ||
      !std::isfinite(tolerance) || input_scale <= 0 || weight_scale <= 0 || output_scale <= 0 ||
      tolerance != output_scale * 1.5f || readI32(bytes + 40) != -128 || readI32(bytes + 44) != -128 ||
      readI32(bytes + 48) != -128) return -121;
  const uint8_t* input = bytes + kLivePayloadHeaderBytes;
  const uint8_t* weights = input + 128;
  const uint8_t* reference_bytes = weights + 128 * 128;
  for (size_t index = 0; index < payload->reference.size(); ++index) {
    payload->reference[index] = readF32(reference_bytes + index * sizeof(float));
    if (!std::isfinite(payload->reference[index])) return -121;
  }
  payload->spec = {input, weights, payload->reference.data(), 128, 128, input_scale, -128,
      weight_scale, -128, output_scale, -128, tolerance};
  return 0;
}

void setQuantizedU8(Qnn_Tensor_t* tensor, const char* name, Qnn_TensorType_t type,
                    uint32_t* dimensions, uint32_t rank, float scale, int32_t offset,
                    void* data, uint64_t size) {
  *tensor = QNN_TENSOR_INIT;
  tensor->v1.name = name;
  tensor->v1.type = type;
  tensor->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor->v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
  tensor->v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  tensor->v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  tensor->v1.quantizeParams.scaleOffsetEncoding.scale = scale;
  tensor->v1.quantizeParams.scaleOffsetEncoding.offset = offset;
  tensor->v1.rank = rank;
  tensor->v1.dimensions = dimensions;
  tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
  tensor->v1.clientBuf.data = data;
  tensor->v1.clientBuf.dataSize = size;
}

double elapsedMs(const timespec& start, const timespec& end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 +
      (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

void setResult(TfQnnU8MatmulResult* result, int32_t status, double execution_ms,
               double max_abs_error) {
  if (result != nullptr) {
    result->status = status;
    result->execution_ms = execution_ms;
    result->max_abs_error = max_abs_error;
  }
}

jdoubleArray jniResult(JNIEnv* env, const TfQnnU8MatmulResult& native_result) {
  const jdouble values[] = {static_cast<jdouble>(native_result.status),
      native_result.execution_ms, native_result.max_abs_error};
  jdoubleArray output = env->NewDoubleArray(3);
  if (output != nullptr) env->SetDoubleArrayRegion(output, 0, 3, values);
  return output;
}
}  // namespace

extern "C" int32_t tf_qnn_execute_u8_matmul(const TfQnnU8MatmulRequest* request,
                                             TfQnnU8MatmulResult* result) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  setResult(result, TF_QNN_MATMUL_INVALID_ARGUMENT, 0.0, kNaN);
  if (request == nullptr || result == nullptr || request->native_dir == nullptr ||
      request->stub_name == nullptr || request->input == nullptr || request->weights == nullptr ||
      request->reference == nullptr || request->output == nullptr || request->k == 0 ||
      request->m == 0 || !std::isfinite(request->input_scale) ||
      !std::isfinite(request->weight_scale) || !std::isfinite(request->output_scale) ||
      !std::isfinite(request->tolerance) || request->input_scale <= 0.0f ||
      request->weight_scale <= 0.0f || request->output_scale <= 0.0f ||
      request->tolerance < 0.0f) {
    return TF_QNN_MATMUL_INVALID_ARGUMENT;
  }
  const uint64_t weight_bytes = static_cast<uint64_t>(request->k) * request->m;
  if (weight_bytes > std::numeric_limits<size_t>::max()) return TF_QNN_MATMUL_INVALID_ARGUMENT;

  setenv("ADSP_LIBRARY_PATH", request->native_dir, 1);
  void* rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (rpc == nullptr) rpc = dlopen("libadsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  void* htp_stub = rpc == nullptr ? nullptr : dlopen(request->stub_name, RTLD_NOW | RTLD_GLOBAL);
  if (rpc == nullptr || htp_stub == nullptr) {
    if (htp_stub != nullptr) dlclose(htp_stub);
    if (rpc != nullptr) dlclose(rpc);
    setResult(result, TF_QNN_MATMUL_LOAD_FAILURE, 0.0, kNaN);
    return TF_QNN_MATMUL_LOAD_FAILURE;
  }

  const QnnInterface_t** providers = nullptr;
  uint32_t provider_count = 0;
  const Qnn_ErrorHandle_t api_status = QnnInterface_getProviders(&providers, &provider_count);
  const QnnInterface_t* provider =
      (api_status == QNN_SUCCESS && provider_count > 0) ? providers[0] : nullptr;
  if (provider == nullptr) {
    dlclose(htp_stub);
    dlclose(rpc);
    setResult(result, TF_QNN_MATMUL_PROVIDER_FAILURE, 0.0, kNaN);
    return TF_QNN_MATMUL_PROVIDER_FAILURE;
  }
  const auto& qnn = provider->QNN_INTERFACE_VER_NAME;
  if (qnn.backendCreate == nullptr || qnn.backendFree == nullptr || qnn.deviceCreate == nullptr ||
      qnn.deviceFree == nullptr || qnn.contextCreate == nullptr || qnn.contextFree == nullptr ||
      qnn.profileCreate == nullptr || qnn.profileFree == nullptr || qnn.graphCreate == nullptr ||
      qnn.tensorCreateGraphTensor == nullptr || qnn.graphAddNode == nullptr ||
      qnn.graphFinalize == nullptr || qnn.graphExecute == nullptr) {
    dlclose(htp_stub);
    dlclose(rpc);
    setResult(result, TF_QNN_MATMUL_API_FAILURE, 0.0, kNaN);
    return TF_QNN_MATMUL_API_FAILURE;
  }

  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_ProfileHandle_t profile = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  int32_t outcome = TF_QNN_MATMUL_OK;
  double execution_ms = 0.0;
  double max_error = kNaN;
  if (qnn.backendCreate(nullptr, nullptr, &backend) != QNN_SUCCESS) outcome = TF_QNN_MATMUL_BACKEND_FAILURE;
  if (outcome == 0 && qnn.deviceCreate(nullptr, nullptr, &device) != QNN_SUCCESS) outcome = TF_QNN_MATMUL_DEVICE_FAILURE;
  if (outcome == 0 && qnn.contextCreate(backend, device, nullptr, &context) != QNN_SUCCESS) outcome = TF_QNN_MATMUL_CONTEXT_FAILURE;
  if (outcome == 0 && qnn.profileCreate(backend, QNN_PROFILE_LEVEL_BASIC, &profile) != QNN_SUCCESS) outcome = TF_QNN_MATMUL_PROFILE_FAILURE;

  uint32_t input_dims[] = {1, request->k};
  uint32_t weight_dims[] = {request->k, request->m};
  uint32_t output_dims[] = {1, request->m};
  Qnn_Tensor_t input;
  Qnn_Tensor_t weights;
  Qnn_Tensor_t output;
  setQuantizedU8(&input, "projection_input", QNN_TENSOR_TYPE_APP_WRITE, input_dims, 2,
      request->input_scale, request->input_offset, nullptr, 0);
  setQuantizedU8(&weights, "projection_weights", QNN_TENSOR_TYPE_STATIC, weight_dims, 2,
      request->weight_scale, request->weight_offset, const_cast<uint8_t*>(request->weights), weight_bytes);
  setQuantizedU8(&output, "projection_output", QNN_TENSOR_TYPE_APP_READ, output_dims, 2,
      request->output_scale, request->output_offset, nullptr, 0);
  if (outcome == 0 && qnn.graphCreate(context, "tensorfold_projection_probe", nullptr, &graph) != QNN_SUCCESS) outcome = TF_QNN_MATMUL_GRAPH_FAILURE;
  if (outcome == 0 && (qnn.tensorCreateGraphTensor(graph, &input) != QNN_SUCCESS ||
      qnn.tensorCreateGraphTensor(graph, &weights) != QNN_SUCCESS ||
      qnn.tensorCreateGraphTensor(graph, &output) != QNN_SUCCESS)) outcome = TF_QNN_MATMUL_TENSOR_FAILURE;
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
  if (outcome == 0 && qnn.graphAddNode(graph, matmul) != QNN_SUCCESS) outcome = TF_QNN_MATMUL_NODE_FAILURE;
  if (outcome == 0 && qnn.graphFinalize(graph, profile, nullptr) != QNN_SUCCESS) outcome = TF_QNN_MATMUL_FINALIZE_FAILURE;

  input.v1.clientBuf.data = const_cast<uint8_t*>(request->input);
  input.v1.clientBuf.dataSize = request->k;
  output.v1.clientBuf.data = request->output;
  output.v1.clientBuf.dataSize = request->m;
  if (outcome == 0) {
    timespec start{}, end{};
    clock_gettime(CLOCK_MONOTONIC, &start);
    if (qnn.graphExecute(graph, &input, 1, &output, 1, profile, nullptr) != QNN_SUCCESS) outcome = TF_QNN_MATMUL_EXECUTE_FAILURE;
    clock_gettime(CLOCK_MONOTONIC, &end);
    execution_ms = elapsedMs(start, end);
  }
  if (outcome == 0) {
    max_error = 0.0;
    for (uint32_t i = 0; i < request->m; ++i) {
      const double actual = request->output_scale *
          (static_cast<int32_t>(request->output[i]) + request->output_offset);
      max_error = std::fmax(max_error, std::fabs(actual - request->reference[i]));
    }
    if (max_error > request->tolerance) outcome = TF_QNN_MATMUL_PARITY_FAILURE;
  }

  if (profile != nullptr) qnn.profileFree(profile);
  if (context != nullptr) qnn.contextFree(context, nullptr);
  if (device != nullptr) qnn.deviceFree(device);
  if (backend != nullptr) qnn.backendFree(backend);
  dlclose(htp_stub);
  dlclose(rpc);
  setResult(result, outcome, execution_ms, max_error);
  return outcome;
}

static jdoubleArray runProjection(JNIEnv* env, jstring native_dir, jstring stub_name,
                                  const ProjectionSpec& spec) {
  TfQnnU8MatmulResult native_result{};
  native_result.status = TF_QNN_MATMUL_LOAD_FAILURE;
  native_result.max_abs_error = std::numeric_limits<double>::quiet_NaN();
  if (native_dir == nullptr || stub_name == nullptr) return jniResult(env, native_result);
  const char* dir = env->GetStringUTFChars(native_dir, nullptr);
  const char* stub = env->GetStringUTFChars(stub_name, nullptr);
  if (dir == nullptr || stub == nullptr) {
    if (dir != nullptr) env->ReleaseStringUTFChars(native_dir, dir);
    if (stub != nullptr) env->ReleaseStringUTFChars(stub_name, stub);
    return jniResult(env, native_result);
  }
  uint8_t output[128] = {};
  TfQnnU8MatmulRequest request = {dir, stub, spec.input, spec.weights, spec.reference, output,
      spec.input_columns, spec.output_columns, spec.input_scale, spec.input_offset,
      spec.weight_scale, spec.weight_offset, spec.output_scale, spec.output_offset, spec.tolerance};
  tf_qnn_execute_u8_matmul(&request, &native_result);
  env->ReleaseStringUTFChars(native_dir, dir);
  env->ReleaseStringUTFChars(stub_name, stub);
  return jniResult(env, native_result);
}

static jdoubleArray runLiveCheckpointProjection(JNIEnv* env, jstring native_dir, jstring stub_name,
                                                jstring payload_path) {
  TfQnnU8MatmulResult native_result{};
  native_result.status = -121;
  native_result.max_abs_error = std::numeric_limits<double>::quiet_NaN();
  if (native_dir == nullptr || stub_name == nullptr || payload_path == nullptr) return jniResult(env, native_result);
  const char* path = env->GetStringUTFChars(payload_path, nullptr);
  if (path == nullptr) return jniResult(env, native_result);
  LivePayload payload;
  const int32_t payload_status = loadLivePayload(path, &payload);
  env->ReleaseStringUTFChars(payload_path, path);
  if (payload_status != 0) {
    native_result.status = payload_status;
    return jniResult(env, native_result);
  }
  const char* dir = env->GetStringUTFChars(native_dir, nullptr);
  const char* stub = env->GetStringUTFChars(stub_name, nullptr);
  if (dir == nullptr || stub == nullptr) {
    if (dir != nullptr) env->ReleaseStringUTFChars(native_dir, dir);
    if (stub != nullptr) env->ReleaseStringUTFChars(stub_name, stub);
    native_result.status = TF_QNN_MATMUL_LOAD_FAILURE;
    return jniResult(env, native_result);
  }
  uint8_t output[128] = {};
  TfQnnU8MatmulRequest request = {dir, stub, payload.spec.input, payload.spec.weights,
      payload.spec.reference, output, payload.spec.input_columns, payload.spec.output_columns,
      payload.spec.input_scale, payload.spec.input_offset, payload.spec.weight_scale,
      payload.spec.weight_offset, payload.spec.output_scale, payload.spec.output_offset,
      payload.spec.tolerance};
  tf_qnn_execute_u8_matmul(&request, &native_result);
  env->ReleaseStringUTFChars(native_dir, dir);
  env->ReleaseStringUTFChars(stub_name, stub);
  return jniResult(env, native_result);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_expo_modules_tensorfoldlocalruntime_QnnProbe_nativeProjection(
    JNIEnv* env, jclass, jstring native_dir, jstring stub_name) {
  return runProjection(env, native_dir, stub_name, kSynthetic);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_expo_modules_tensorfoldlocalruntime_QnnProbe_nativeCheckpointProjection(
    JNIEnv* env, jclass, jstring native_dir, jstring stub_name) {
  return runProjection(env, native_dir, stub_name, kCheckpoint);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_expo_modules_tensorfoldlocalruntime_QnnProbe_nativeFullProjection(
    JNIEnv* env, jclass, jstring native_dir, jstring stub_name) {
  return runProjection(env, native_dir, stub_name, kFull);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_expo_modules_tensorfoldlocalruntime_QnnProbe_nativeLiveCheckpointProjection(
    JNIEnv* env, jclass, jstring native_dir, jstring stub_name, jstring payload_path) {
  return runLiveCheckpointProjection(env, native_dir, stub_name, payload_path);
}
