#include "qnn_session_bridge.h"

#include <dlfcn.h>
#include <time.h>
#include <cstdlib>

#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <QNN/QnnInterface.h>
#include <QNN/QnnOpDef.h>

namespace {
constexpr uint64_t kMaximumWeightBytes = 64ULL * 1024ULL * 1024ULL;

double elapsedMs(const timespec& start, const timespec& end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

void clearTiming(TfQnnSessionTiming* timing, int32_t status) {
  if (timing != nullptr) *timing = {status, 0.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN()};
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

void setFloat16(Qnn_Tensor_t* tensor, const char* name, Qnn_TensorType_t type,
                uint32_t* dimensions, uint32_t rank, void* data, uint64_t size) {
  *tensor = QNN_TENSOR_INIT;
  tensor->v1.name = name;
  tensor->v1.type = type;
  tensor->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor->v1.dataType = QNN_DATATYPE_FLOAT_16;
  tensor->v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tensor->v1.rank = rank;
  tensor->v1.dimensions = dimensions;
  tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
  tensor->v1.clientBuf.data = data;
  tensor->v1.clientBuf.dataSize = size;
}

void setAxisQuantizedU8(Qnn_Tensor_t* tensor, const char* name, Qnn_TensorType_t type,
                        uint32_t* dimensions, uint32_t rank, uint32_t axis,
                        Qnn_ScaleOffset_t* scale_offsets, uint32_t count,
                        void* data, uint64_t size) {
  *tensor = QNN_TENSOR_INIT;
  tensor->v1.name = name;
  tensor->v1.type = type;
  tensor->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor->v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
  tensor->v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  tensor->v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  tensor->v1.quantizeParams.axisScaleOffsetEncoding.axis = axis;
  tensor->v1.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets = count;
  tensor->v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset = scale_offsets;
  tensor->v1.rank = rank;
  tensor->v1.dimensions = dimensions;
  tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
  tensor->v1.clientBuf.data = data;
  tensor->v1.clientBuf.dataSize = size;
}
}  // namespace

struct TfQnnSession {
  enum class State { kCreated, kWeightsSet, kFinalized, kDestroyed };
  std::mutex mutex;
  State state = State::kCreated;
  TfQnnSessionConfig config{};
  std::string native_dir;
  std::string stub_name;
  std::vector<uint8_t> weights;
  std::vector<Qnn_ScaleOffset_t> weight_axis_scale_offsets;
  void* rpc = nullptr;
  void* htp_stub = nullptr;
  const QnnInterface_t* provider = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_ProfileHandle_t profile = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  uint32_t input_dims[2]{};
  uint32_t weight_dims[2]{};
  uint32_t output_dims[2]{};
  Qnn_Tensor_t input = QNN_TENSOR_INIT;
  Qnn_Tensor_t weight = QNN_TENSOR_INIT;
  Qnn_Tensor_t output = QNN_TENSOR_INIT;

  void cleanup() {
    if (provider != nullptr) {
      const auto& qnn = provider->QNN_INTERFACE_VER_NAME;
      if (profile != nullptr && qnn.profileFree != nullptr) qnn.profileFree(profile);
      if (context != nullptr && qnn.contextFree != nullptr) qnn.contextFree(context, nullptr);
      if (device != nullptr && qnn.deviceFree != nullptr) qnn.deviceFree(device);
      if (backend != nullptr && qnn.backendFree != nullptr) qnn.backendFree(backend);
    }
    profile = nullptr; context = nullptr; device = nullptr; backend = nullptr; graph = nullptr;
    provider = nullptr;
    if (htp_stub != nullptr) dlclose(htp_stub);
    if (rpc != nullptr) dlclose(rpc);
    htp_stub = nullptr; rpc = nullptr;
  }
};

extern "C" int32_t tf_qnn_session_create(const TfQnnSessionConfig* config,
                                          TfQnnSession** out, TfQnnSessionTiming* timing) {
  clearTiming(timing, TF_QNN_MATMUL_INVALID_ARGUMENT);
  const bool axis_quantized = config != nullptr && config->weight_axis_scales != nullptr;
  const uint32_t element_bytes = config == nullptr || config->element_bytes == 0 ? 1 : config->element_bytes;
  const bool fp16 = config != nullptr && config->data_type == QNN_DATATYPE_FLOAT_16;
  if (out == nullptr || config == nullptr || config->native_dir == nullptr || config->stub_name == nullptr ||
      config->k == 0 || config->m == 0 || (config->data_type != 0 && !fp16) ||
      (fp16 && (element_bytes != sizeof(uint16_t) || axis_quantized)) ||
      (!fp16 && element_bytes != 1) || !std::isfinite(config->input_scale) ||
      !std::isfinite(config->weight_scale) || !std::isfinite(config->output_scale) ||
      config->input_scale <= 0 || config->weight_scale <= 0 || config->output_scale <= 0 ||
      (axis_quantized && (config->weight_axis_offsets == nullptr || config->weight_axis_count != config->m ||
                          config->weight_axis != 1)) ||
      (!axis_quantized && (config->weight_axis_offsets != nullptr || config->weight_axis_count != 0 ||
                           config->weight_axis != 0)) ||
      static_cast<uint64_t>(config->k) * config->m * element_bytes > kMaximumWeightBytes) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  *out = nullptr;
  timespec start{}, end{}; clock_gettime(CLOCK_MONOTONIC, &start);
  TfQnnSession* session = new (std::nothrow) TfQnnSession();
  if (session == nullptr) return TF_QNN_MATMUL_LOAD_FAILURE;
  session->config = *config;
  try {
    session->native_dir = config->native_dir;
    session->stub_name = config->stub_name;
    if (axis_quantized) {
      session->weight_axis_scale_offsets.resize(config->m);
      for (uint32_t i = 0; i < config->m; ++i) {
        if (!std::isfinite(config->weight_axis_scales[i]) || config->weight_axis_scales[i] <= 0)
          throw std::bad_alloc();
        session->weight_axis_scale_offsets[i] = {config->weight_axis_scales[i], config->weight_axis_offsets[i]};
      }
    }
  } catch (const std::bad_alloc&) { delete session; return TF_QNN_MATMUL_LOAD_FAILURE; }
  session->config.native_dir = session->native_dir.c_str();
  session->config.stub_name = session->stub_name.c_str();
  session->config.weight_axis_scales = nullptr;
  session->config.weight_axis_offsets = nullptr;
  setenv("ADSP_LIBRARY_PATH", session->config.native_dir, 1);
  session->rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (session->rpc == nullptr) session->rpc = dlopen("libadsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  session->htp_stub = session->rpc == nullptr ? nullptr : dlopen(session->config.stub_name, RTLD_NOW | RTLD_GLOBAL);
  int32_t status = TF_QNN_MATMUL_OK;
  if (session->rpc == nullptr || session->htp_stub == nullptr) status = TF_QNN_MATMUL_LOAD_FAILURE;
  const QnnInterface_t** providers = nullptr; uint32_t count = 0;
  if (status == 0 && (QnnInterface_getProviders(&providers, &count) != QNN_SUCCESS || count == 0)) status = TF_QNN_MATMUL_PROVIDER_FAILURE;
  if (status == 0) session->provider = providers[0];
  if (status == 0) {
    const auto& qnn = session->provider->QNN_INTERFACE_VER_NAME;
    if (qnn.backendCreate == nullptr || qnn.backendFree == nullptr || qnn.deviceCreate == nullptr ||
        qnn.deviceFree == nullptr || qnn.contextCreate == nullptr || qnn.contextFree == nullptr ||
        qnn.profileCreate == nullptr || qnn.profileFree == nullptr || qnn.graphCreate == nullptr ||
        qnn.tensorCreateGraphTensor == nullptr || qnn.graphAddNode == nullptr || qnn.graphFinalize == nullptr ||
        qnn.graphExecute == nullptr) status = TF_QNN_MATMUL_API_FAILURE;
    if (status == 0 && qnn.backendCreate(nullptr, nullptr, &session->backend) != QNN_SUCCESS) status = TF_QNN_MATMUL_BACKEND_FAILURE;
    if (status == 0 && qnn.deviceCreate(nullptr, nullptr, &session->device) != QNN_SUCCESS) status = TF_QNN_MATMUL_DEVICE_FAILURE;
    if (status == 0 && qnn.contextCreate(session->backend, session->device, nullptr, &session->context) != QNN_SUCCESS) status = TF_QNN_MATMUL_CONTEXT_FAILURE;
    if (status == 0 && qnn.profileCreate(session->backend, QNN_PROFILE_LEVEL_BASIC, &session->profile) != QNN_SUCCESS) status = TF_QNN_MATMUL_PROFILE_FAILURE;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (timing != nullptr) { timing->status = status; timing->create_ms = elapsedMs(start, end); }
  if (status != 0) { session->cleanup(); delete session; return status; }
  *out = session;
  return 0;
}

extern "C" int32_t tf_qnn_session_set_weights(TfQnnSession* session, const uint8_t* weights, uint64_t bytes) {
  if (session == nullptr || weights == nullptr) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(session->mutex, std::try_to_lock);
  if (!lock.owns_lock()) return TF_QNN_SESSION_BUSY;
  if (session->state == TfQnnSession::State::kDestroyed) return TF_QNN_SESSION_STATE;
  if (session->state != TfQnnSession::State::kCreated) return TF_QNN_SESSION_STATE;
  const uint64_t required = static_cast<uint64_t>(session->config.k) * session->config.m *
      (session->config.element_bytes == 0 ? 1 : session->config.element_bytes);
  if (bytes != required || bytes > kMaximumWeightBytes) return TF_QNN_SESSION_SHAPE;
  try { session->weights.assign(weights, weights + bytes); }
  catch (const std::bad_alloc&) { return TF_QNN_MATMUL_TENSOR_FAILURE; }
  session->state = TfQnnSession::State::kWeightsSet;
  return 0;
}

extern "C" int32_t tf_qnn_session_finalize(TfQnnSession* session, TfQnnSessionTiming* timing) {
  clearTiming(timing, TF_QNN_SESSION_STATE);
  if (session == nullptr) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(session->mutex, std::try_to_lock);
  if (!lock.owns_lock()) { if (timing != nullptr) timing->status = TF_QNN_SESSION_BUSY; return TF_QNN_SESSION_BUSY; }
  if (session->state != TfQnnSession::State::kWeightsSet) return TF_QNN_SESSION_STATE;
  timespec start{}, end{}; clock_gettime(CLOCK_MONOTONIC, &start);
  session->input_dims[0] = 1; session->input_dims[1] = session->config.k;
  session->weight_dims[0] = session->config.k; session->weight_dims[1] = session->config.m;
  session->output_dims[0] = 1; session->output_dims[1] = session->config.m;
  if (session->config.data_type == QNN_DATATYPE_FLOAT_16) {
    setFloat16(&session->input, "persistent_projection_input", QNN_TENSOR_TYPE_APP_WRITE,
               session->input_dims, 2, nullptr, 0);
    setFloat16(&session->weight, "persistent_projection_weights", QNN_TENSOR_TYPE_STATIC,
               session->weight_dims, 2, session->weights.data(), session->weights.size());
    setFloat16(&session->output, "persistent_projection_output", QNN_TENSOR_TYPE_APP_READ,
               session->output_dims, 2, nullptr, 0);
  } else {
    setQuantizedU8(&session->input, "persistent_projection_input", QNN_TENSOR_TYPE_APP_WRITE, session->input_dims, 2, session->config.input_scale, session->config.input_offset, nullptr, 0);
    if (session->config.weight_axis_count == 0) {
      setQuantizedU8(&session->weight, "persistent_projection_weights", QNN_TENSOR_TYPE_STATIC, session->weight_dims, 2, session->config.weight_scale, session->config.weight_offset, session->weights.data(), session->weights.size());
    } else {
      setAxisQuantizedU8(&session->weight, "persistent_projection_weights", QNN_TENSOR_TYPE_STATIC,
                         session->weight_dims, 2, session->config.weight_axis,
                         session->weight_axis_scale_offsets.data(), session->config.weight_axis_count,
                         session->weights.data(), session->weights.size());
    }
    setQuantizedU8(&session->output, "persistent_projection_output", QNN_TENSOR_TYPE_APP_READ, session->output_dims, 2, session->config.output_scale, session->config.output_offset, nullptr, 0);
  }
  const auto& qnn = session->provider->QNN_INTERFACE_VER_NAME;
  int32_t status = qnn.graphCreate(session->context, "tensorfold_persistent_projection", nullptr, &session->graph) == QNN_SUCCESS ? 0 : TF_QNN_MATMUL_GRAPH_FAILURE;
  if (status == 0 && (qnn.tensorCreateGraphTensor(session->graph, &session->input) != QNN_SUCCESS || qnn.tensorCreateGraphTensor(session->graph, &session->weight) != QNN_SUCCESS || qnn.tensorCreateGraphTensor(session->graph, &session->output) != QNN_SUCCESS)) status = TF_QNN_MATMUL_TENSOR_FAILURE;
  Qnn_Tensor_t inputs[] = {session->input, session->weight}; Qnn_Tensor_t outputs[] = {session->output};
  Qnn_OpConfig_t matmul = QNN_OPCONFIG_INIT;
  matmul.v1.name = "tensorfold_persistent_projection_matmul"; matmul.v1.packageName = "qti.aisw"; matmul.v1.typeName = QNN_OP_MAT_MUL;
  matmul.v1.numOfInputs = 2; matmul.v1.inputTensors = inputs; matmul.v1.numOfOutputs = 1; matmul.v1.outputTensors = outputs;
  if (status == 0 && qnn.graphAddNode(session->graph, matmul) != QNN_SUCCESS) status = TF_QNN_MATMUL_NODE_FAILURE;
  if (status == 0 && qnn.graphFinalize(session->graph, session->profile, nullptr) != QNN_SUCCESS) status = TF_QNN_MATMUL_FINALIZE_FAILURE;
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (timing != nullptr) { timing->status = status; timing->finalize_ms = elapsedMs(start, end); }
  if (status == 0) session->state = TfQnnSession::State::kFinalized;
  return status;
}

extern "C" int32_t tf_qnn_session_execute(TfQnnSession* session, const uint8_t* input, uint64_t input_bytes, uint8_t* output, uint64_t output_bytes, TfQnnSessionTiming* timing) {
  clearTiming(timing, TF_QNN_SESSION_STATE);
  if (session == nullptr || input == nullptr || output == nullptr) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(session->mutex, std::try_to_lock);
  if (!lock.owns_lock()) { if (timing != nullptr) timing->status = TF_QNN_SESSION_BUSY; return TF_QNN_SESSION_BUSY; }
  if (session->state != TfQnnSession::State::kFinalized) return TF_QNN_SESSION_STATE;
  const uint64_t element_bytes = session->config.element_bytes == 0 ? 1 : session->config.element_bytes;
  if (input_bytes != static_cast<uint64_t>(session->config.k) * element_bytes ||
      output_bytes != static_cast<uint64_t>(session->config.m) * element_bytes) return TF_QNN_SESSION_SHAPE;
  session->input.v1.clientBuf.data = const_cast<uint8_t*>(input); session->input.v1.clientBuf.dataSize = input_bytes;
  session->output.v1.clientBuf.data = output; session->output.v1.clientBuf.dataSize = output_bytes;
  timespec start{}, end{}; clock_gettime(CLOCK_MONOTONIC, &start);
  const int32_t status = session->provider->QNN_INTERFACE_VER_NAME.graphExecute(session->graph, &session->input, 1, &session->output, 1, session->profile, nullptr) == QNN_SUCCESS ? 0 : TF_QNN_MATMUL_EXECUTE_FAILURE;
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (timing != nullptr) { timing->status = status; timing->execute_ms = elapsedMs(start, end); }
  return status;
}

extern "C" int32_t tf_qnn_session_parity(TfQnnSession* session, const uint8_t* output, uint64_t output_bytes, const float* reference, uint64_t reference_count, float tolerance, TfQnnSessionTiming* timing) {
  clearTiming(timing, TF_QNN_SESSION_STATE);
  if (session == nullptr || output == nullptr || reference == nullptr || !std::isfinite(tolerance) || tolerance < 0) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(session->mutex, std::try_to_lock);
  if (!lock.owns_lock()) { if (timing != nullptr) timing->status = TF_QNN_SESSION_BUSY; return TF_QNN_SESSION_BUSY; }
  if (session->state != TfQnnSession::State::kFinalized) return TF_QNN_SESSION_STATE;
  const uint64_t element_bytes = session->config.element_bytes == 0 ? 1 : session->config.element_bytes;
  if (output_bytes != static_cast<uint64_t>(session->config.m) * element_bytes ||
      reference_count != session->config.m) return TF_QNN_SESSION_SHAPE;
  double maximum = 0.0;
  for (uint32_t i = 0; i < session->config.m; ++i) {
    double actual;
    if (session->config.data_type == QNN_DATATYPE_FLOAT_16) {
      uint16_t bits;
      std::memcpy(&bits, output + i * sizeof(bits), sizeof(bits));
      _Float16 value;
      std::memcpy(&value, &bits, sizeof(value));
      actual = static_cast<double>(value);
    } else {
      actual = session->config.output_scale * (static_cast<int32_t>(output[i]) + session->config.output_offset);
    }
    maximum = std::fmax(maximum, std::fabs(actual - reference[i]));
  }
  const int32_t status = maximum <= tolerance ? 0 : TF_QNN_MATMUL_PARITY_FAILURE;
  if (timing != nullptr) { timing->status = status; timing->max_abs_error = maximum; }
  return status;
}

extern "C" void tf_qnn_session_destroy(TfQnnSession* session) {
  if (session == nullptr) return;
  {
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->state != TfQnnSession::State::kDestroyed) {
      session->cleanup();
      session->weights.clear();
      session->state = TfQnnSession::State::kDestroyed;
    }
  }
  delete session;
}

struct TfQnnPairSession {
  enum class State { kCreated, kPayloadSet, kFinalized };
  std::mutex mutex;
  State state = State::kCreated;
  TfQnnPairSessionConfig config{};
  std::string native_dir;
  std::string stub_name;
  std::vector<uint8_t> input, qkv_weights, z_weights;
  void* rpc = nullptr;
  void* htp_stub = nullptr;
  const QnnInterface_t* provider = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_ProfileHandle_t profile = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  uint32_t input_dims[2]{1, 0}, qkv_dims[2]{}, z_dims[2]{}, qkv_out_dims[2]{}, z_out_dims[2]{};
  Qnn_Tensor_t input_tensor = QNN_TENSOR_INIT, qkv_weight = QNN_TENSOR_INIT,
      z_weight = QNN_TENSOR_INIT, qkv_output = QNN_TENSOR_INIT, z_output = QNN_TENSOR_INIT;
  void cleanup() {
    if (provider != nullptr) {
      const auto& qnn = provider->QNN_INTERFACE_VER_NAME;
      if (profile != nullptr && qnn.profileFree != nullptr) qnn.profileFree(profile);
      if (context != nullptr && qnn.contextFree != nullptr) qnn.contextFree(context, nullptr);
      if (device != nullptr && qnn.deviceFree != nullptr) qnn.deviceFree(device);
      if (backend != nullptr && qnn.backendFree != nullptr) qnn.backendFree(backend);
    }
    profile = nullptr; context = nullptr; device = nullptr; backend = nullptr; graph = nullptr; provider = nullptr;
    if (htp_stub != nullptr) dlclose(htp_stub);
    if (rpc != nullptr) dlclose(rpc);
    htp_stub = nullptr; rpc = nullptr;
  }
};

static void clearPairTiming(TfQnnPairSessionTiming* timing, int32_t status) {
  if (timing != nullptr) *timing = {status, 0.0, 0.0, 0.0,
      std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
}

extern "C" int32_t tf_qnn_pair_session_create(const TfQnnPairSessionConfig* config,
                                                TfQnnPairSession** out,
                                                TfQnnPairSessionTiming* timing) {
  clearPairTiming(timing, TF_QNN_MATMUL_INVALID_ARGUMENT);
  if (out == nullptr || config == nullptr || config->native_dir == nullptr || config->stub_name == nullptr ||
      config->k == 0 || config->qkv_m == 0 || config->z_m == 0 ||
      (static_cast<uint64_t>(config->k) * (config->qkv_m + config->z_m) * sizeof(uint16_t)) >
          kMaximumWeightBytes) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  *out = nullptr;
  timespec start{}, end{}; clock_gettime(CLOCK_MONOTONIC, &start);
  auto* session = new (std::nothrow) TfQnnPairSession();
  if (session == nullptr) return TF_QNN_MATMUL_LOAD_FAILURE;
  try { session->native_dir = config->native_dir; session->stub_name = config->stub_name; }
  catch (const std::bad_alloc&) { delete session; return TF_QNN_MATMUL_LOAD_FAILURE; }
  session->config = *config; session->config.native_dir = session->native_dir.c_str(); session->config.stub_name = session->stub_name.c_str();
  setenv("ADSP_LIBRARY_PATH", session->config.native_dir, 1);
  session->rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (session->rpc == nullptr) session->rpc = dlopen("libadsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  session->htp_stub = session->rpc == nullptr ? nullptr : dlopen(session->config.stub_name, RTLD_NOW | RTLD_GLOBAL);
  int32_t status = session->rpc != nullptr && session->htp_stub != nullptr ? 0 : TF_QNN_MATMUL_LOAD_FAILURE;
  const QnnInterface_t** providers = nullptr; uint32_t count = 0;
  if (status == 0 && (QnnInterface_getProviders(&providers, &count) != QNN_SUCCESS || count == 0)) status = TF_QNN_MATMUL_PROVIDER_FAILURE;
  if (status == 0) session->provider = providers[0];
  if (status == 0) {
    const auto& qnn = session->provider->QNN_INTERFACE_VER_NAME;
    if (qnn.backendCreate == nullptr || qnn.backendFree == nullptr || qnn.deviceCreate == nullptr ||
        qnn.deviceFree == nullptr || qnn.contextCreate == nullptr || qnn.contextFree == nullptr ||
        qnn.profileCreate == nullptr || qnn.profileFree == nullptr || qnn.graphCreate == nullptr ||
        qnn.tensorCreateGraphTensor == nullptr || qnn.graphAddNode == nullptr ||
        qnn.graphFinalize == nullptr || qnn.graphExecute == nullptr) status = TF_QNN_MATMUL_API_FAILURE;
    if (status == 0 && qnn.backendCreate(nullptr, nullptr, &session->backend) != QNN_SUCCESS) status = TF_QNN_MATMUL_BACKEND_FAILURE;
    if (status == 0 && qnn.deviceCreate(nullptr, nullptr, &session->device) != QNN_SUCCESS) status = TF_QNN_MATMUL_DEVICE_FAILURE;
    if (status == 0 && qnn.contextCreate(session->backend, session->device, nullptr, &session->context) != QNN_SUCCESS) status = TF_QNN_MATMUL_CONTEXT_FAILURE;
    if (status == 0 && qnn.profileCreate(session->backend, QNN_PROFILE_LEVEL_BASIC, &session->profile) != QNN_SUCCESS) status = TF_QNN_MATMUL_PROFILE_FAILURE;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  if (timing != nullptr) { timing->status = status; timing->create_ms = elapsedMs(start, end); }
  if (status != 0) { session->cleanup(); delete session; return status; }
  *out = session; return 0;
}

extern "C" int32_t tf_qnn_pair_session_set_payload(TfQnnPairSession* session,
    const uint8_t* input, uint64_t input_bytes, const uint8_t* qkv_weights, uint64_t qkv_bytes,
    const uint8_t* z_weights, uint64_t z_bytes) {
  if (session == nullptr || input == nullptr || qkv_weights == nullptr || z_weights == nullptr) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(session->mutex, std::try_to_lock);
  if (!lock.owns_lock()) return TF_QNN_SESSION_BUSY;
  if (session->state != TfQnnPairSession::State::kCreated) return TF_QNN_SESSION_STATE;
  const uint64_t input_required = static_cast<uint64_t>(session->config.k) * sizeof(uint16_t);
  const uint64_t qkv_required = static_cast<uint64_t>(session->config.k) * session->config.qkv_m * sizeof(uint16_t);
  const uint64_t z_required = static_cast<uint64_t>(session->config.k) * session->config.z_m * sizeof(uint16_t);
  if (input_bytes != input_required || qkv_bytes != qkv_required || z_bytes != z_required) return TF_QNN_SESSION_SHAPE;
  try { session->input.assign(input, input + input_bytes); session->qkv_weights.assign(qkv_weights, qkv_weights + qkv_bytes); session->z_weights.assign(z_weights, z_weights + z_bytes); }
  catch (const std::bad_alloc&) { return TF_QNN_MATMUL_TENSOR_FAILURE; }
  session->state = TfQnnPairSession::State::kPayloadSet; return 0;
}

extern "C" int32_t tf_qnn_pair_session_finalize(TfQnnPairSession* session, TfQnnPairSessionTiming* timing) {
  clearPairTiming(timing, TF_QNN_SESSION_STATE);
  if (session == nullptr) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(session->mutex, std::try_to_lock);
  if (!lock.owns_lock()) { if (timing != nullptr) timing->status = TF_QNN_SESSION_BUSY; return TF_QNN_SESSION_BUSY; }
  if (session->state != TfQnnPairSession::State::kPayloadSet) return TF_QNN_SESSION_STATE;
  timespec start{}, end{}; clock_gettime(CLOCK_MONOTONIC, &start);
  session->input_dims[1] = session->config.k; session->qkv_dims[0] = session->config.k; session->qkv_dims[1] = session->config.qkv_m;
  session->z_dims[0] = session->config.k; session->z_dims[1] = session->config.z_m; session->qkv_out_dims[0] = 1; session->qkv_out_dims[1] = session->config.qkv_m; session->z_out_dims[0] = 1; session->z_out_dims[1] = session->config.z_m;
  setFloat16(&session->input_tensor, "pair_input", QNN_TENSOR_TYPE_APP_WRITE, session->input_dims, 2, nullptr, 0);
  setFloat16(&session->qkv_weight, "pair_qkv_weight", QNN_TENSOR_TYPE_STATIC, session->qkv_dims, 2, session->qkv_weights.data(), session->qkv_weights.size());
  setFloat16(&session->z_weight, "pair_z_weight", QNN_TENSOR_TYPE_STATIC, session->z_dims, 2, session->z_weights.data(), session->z_weights.size());
  setFloat16(&session->qkv_output, "pair_qkv_output", QNN_TENSOR_TYPE_APP_READ, session->qkv_out_dims, 2, nullptr, 0);
  setFloat16(&session->z_output, "pair_z_output", QNN_TENSOR_TYPE_APP_READ, session->z_out_dims, 2, nullptr, 0);
  const auto& qnn = session->provider->QNN_INTERFACE_VER_NAME;
  int32_t status = qnn.graphCreate(session->context, "tensorfold_projection_pair", nullptr, &session->graph) == QNN_SUCCESS ? 0 : TF_QNN_MATMUL_GRAPH_FAILURE;
  if (status == 0 && (qnn.tensorCreateGraphTensor(session->graph, &session->input_tensor) != QNN_SUCCESS || qnn.tensorCreateGraphTensor(session->graph, &session->qkv_weight) != QNN_SUCCESS || qnn.tensorCreateGraphTensor(session->graph, &session->z_weight) != QNN_SUCCESS || qnn.tensorCreateGraphTensor(session->graph, &session->qkv_output) != QNN_SUCCESS || qnn.tensorCreateGraphTensor(session->graph, &session->z_output) != QNN_SUCCESS)) status = TF_QNN_MATMUL_TENSOR_FAILURE;
  Qnn_Tensor_t qkv_inputs[] = {session->input_tensor, session->qkv_weight}; Qnn_Tensor_t qkv_outputs[] = {session->qkv_output};
  Qnn_Tensor_t z_inputs[] = {session->input_tensor, session->z_weight}; Qnn_Tensor_t z_outputs[] = {session->z_output};
  Qnn_OpConfig_t qkv_matmul = QNN_OPCONFIG_INIT; qkv_matmul.v1.name = "pair_qkv_matmul"; qkv_matmul.v1.packageName = "qti.aisw"; qkv_matmul.v1.typeName = QNN_OP_MAT_MUL; qkv_matmul.v1.numOfInputs = 2; qkv_matmul.v1.inputTensors = qkv_inputs; qkv_matmul.v1.numOfOutputs = 1; qkv_matmul.v1.outputTensors = qkv_outputs;
  Qnn_OpConfig_t z_matmul = QNN_OPCONFIG_INIT; z_matmul.v1.name = "pair_z_matmul"; z_matmul.v1.packageName = "qti.aisw"; z_matmul.v1.typeName = QNN_OP_MAT_MUL; z_matmul.v1.numOfInputs = 2; z_matmul.v1.inputTensors = z_inputs; z_matmul.v1.numOfOutputs = 1; z_matmul.v1.outputTensors = z_outputs;
  if (status == 0 && qnn.graphAddNode(session->graph, qkv_matmul) != QNN_SUCCESS) status = TF_QNN_MATMUL_NODE_FAILURE;
  if (status == 0 && qnn.graphAddNode(session->graph, z_matmul) != QNN_SUCCESS) status = TF_QNN_MATMUL_NODE_FAILURE;
  if (status == 0 && qnn.graphFinalize(session->graph, session->profile, nullptr) != QNN_SUCCESS) status = TF_QNN_MATMUL_FINALIZE_FAILURE;
  clock_gettime(CLOCK_MONOTONIC, &end); if (timing != nullptr) { timing->status = status; timing->finalize_ms = elapsedMs(start, end); }
  if (status == 0) session->state = TfQnnPairSession::State::kFinalized; return status;
}

extern "C" int32_t tf_qnn_pair_session_execute(TfQnnPairSession* session, uint8_t* qkv_output,
    uint64_t qkv_bytes, uint8_t* z_output, uint64_t z_bytes, TfQnnPairSessionTiming* timing) {
  clearPairTiming(timing, TF_QNN_SESSION_STATE); if (session == nullptr || qkv_output == nullptr || z_output == nullptr) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(session->mutex, std::try_to_lock); if (!lock.owns_lock()) return TF_QNN_SESSION_BUSY;
  if (session->state != TfQnnPairSession::State::kFinalized) return TF_QNN_SESSION_STATE;
  if (qkv_bytes != static_cast<uint64_t>(session->config.qkv_m) * 2 || z_bytes != static_cast<uint64_t>(session->config.z_m) * 2) return TF_QNN_SESSION_SHAPE;
  session->input_tensor.v1.clientBuf.data = session->input.data();
  session->input_tensor.v1.clientBuf.dataSize = session->input.size();
  session->qkv_output.v1.clientBuf.data = qkv_output; session->qkv_output.v1.clientBuf.dataSize = qkv_bytes; session->z_output.v1.clientBuf.data = z_output; session->z_output.v1.clientBuf.dataSize = z_bytes;
  Qnn_Tensor_t outputs[] = {session->qkv_output, session->z_output}; timespec start{}, end{}; clock_gettime(CLOCK_MONOTONIC, &start);
  const int32_t status = session->provider->QNN_INTERFACE_VER_NAME.graphExecute(session->graph, &session->input_tensor, 1, outputs, 2, session->profile, nullptr) == QNN_SUCCESS ? 0 : TF_QNN_MATMUL_EXECUTE_FAILURE;
  clock_gettime(CLOCK_MONOTONIC, &end); if (timing != nullptr) { timing->status = status; timing->execute_ms = elapsedMs(start, end); } return status;
}

static double fp16MaxError(const uint8_t* output, uint64_t bytes, const float* reference, uint32_t count) {
  if (output == nullptr || reference == nullptr || bytes != static_cast<uint64_t>(count) * 2) return std::numeric_limits<double>::quiet_NaN();
  double maximum = 0.0; for (uint32_t i = 0; i < count; ++i) { _Float16 value; std::memcpy(&value, output + 2 * i, 2); maximum = std::fmax(maximum, std::fabs(static_cast<double>(value) - reference[i])); } return maximum;
}
extern "C" int32_t tf_qnn_pair_session_parity(TfQnnPairSession* session, const uint8_t* qkv_output, uint64_t qkv_bytes, const float* qkv_reference, float qkv_tolerance, const uint8_t* z_output, uint64_t z_bytes, const float* z_reference, float z_tolerance, TfQnnPairSessionTiming* timing) {
  clearPairTiming(timing, TF_QNN_SESSION_STATE); if (session == nullptr || !std::isfinite(qkv_tolerance) || !std::isfinite(z_tolerance) || qkv_tolerance < 0 || z_tolerance < 0) return TF_QNN_MATMUL_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lock(session->mutex, std::try_to_lock); if (!lock.owns_lock()) return TF_QNN_SESSION_BUSY;
  if (session->state != TfQnnPairSession::State::kFinalized) return TF_QNN_SESSION_STATE;
  const double qkv_error = fp16MaxError(qkv_output, qkv_bytes, qkv_reference, session->config.qkv_m), z_error = fp16MaxError(z_output, z_bytes, z_reference, session->config.z_m);
  const int32_t status = std::isfinite(qkv_error) && std::isfinite(z_error) && qkv_error <= qkv_tolerance && z_error <= z_tolerance ? 0 : TF_QNN_MATMUL_PARITY_FAILURE;
  if (timing != nullptr) { timing->status = status; timing->qkv_max_abs_error = qkv_error; timing->z_max_abs_error = z_error; } return status;
}
extern "C" void tf_qnn_pair_session_destroy(TfQnnPairSession* session) {
  if (session == nullptr) return;
  {
    std::lock_guard<std::mutex> lock(session->mutex);
    session->cleanup();
  }
  delete session;
}
