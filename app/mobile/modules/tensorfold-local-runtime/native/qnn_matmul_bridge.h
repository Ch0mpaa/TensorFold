#pragma once

#include <stdint.h>

#if defined(__GNUC__)
#define TF_QNN_BRIDGE_EXPORT __attribute__((visibility("default")))
#else
#define TF_QNN_BRIDGE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum TfQnnMatmulStatus {
  TF_QNN_MATMUL_OK = 0,
  TF_QNN_MATMUL_INVALID_ARGUMENT = -99,
  TF_QNN_MATMUL_LOAD_FAILURE = -100,
  TF_QNN_MATMUL_PROVIDER_FAILURE = -101,
  TF_QNN_MATMUL_API_FAILURE = -102,
  TF_QNN_MATMUL_BACKEND_FAILURE = -103,
  TF_QNN_MATMUL_DEVICE_FAILURE = -104,
  TF_QNN_MATMUL_CONTEXT_FAILURE = -105,
  TF_QNN_MATMUL_PROFILE_FAILURE = -106,
  TF_QNN_MATMUL_GRAPH_FAILURE = -107,
  TF_QNN_MATMUL_TENSOR_FAILURE = -108,
  TF_QNN_MATMUL_NODE_FAILURE = -109,
  TF_QNN_MATMUL_FINALIZE_FAILURE = -110,
  TF_QNN_MATMUL_EXECUTE_FAILURE = -111,
  TF_QNN_MATMUL_PARITY_FAILURE = -112,
};

// All buffers remain owned by the caller. Weights are K-major [K, M], and the
// supplied output buffer has M bytes. `reference` is an independently computed
// real-domain result; it is used only for parity, never as a fallback output.
typedef struct TfQnnU8MatmulRequest {
  const char* native_dir;
  const char* stub_name;
  const uint8_t* input;
  const uint8_t* weights;
  const float* reference;
  uint8_t* output;
  uint32_t k;
  uint32_t m;
  float input_scale;
  int32_t input_offset;
  float weight_scale;
  int32_t weight_offset;
  float output_scale;
  int32_t output_offset;
  float tolerance;
} TfQnnU8MatmulRequest;

typedef struct TfQnnU8MatmulResult {
  int32_t status;
  double execution_ms;
  double max_abs_error;
} TfQnnU8MatmulResult;

// Creates and destroys every QNN object inside this call. A nonzero status
// guarantees that no CPU-computed candidate was substituted for `output`.
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_execute_u8_matmul(
    const TfQnnU8MatmulRequest* request, TfQnnU8MatmulResult* result);

#ifdef __cplusplus
}
#endif
