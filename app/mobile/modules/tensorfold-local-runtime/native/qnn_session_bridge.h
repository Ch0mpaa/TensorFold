#pragma once

#include <stdint.h>

#include "qnn_matmul_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TfQnnSession TfQnnSession;

enum TfQnnSessionStatus {
  TF_QNN_SESSION_BUSY = -113,
  TF_QNN_SESSION_STATE = -114,
  TF_QNN_SESSION_SHAPE = -115,
};

typedef struct TfQnnSessionConfig {
  const char* native_dir;
  const char* stub_name;
  uint32_t k;
  uint32_t m;
  float input_scale;
  int32_t input_offset;
  float weight_scale;
  int32_t weight_offset;
  const float* weight_axis_scales;
  const int32_t* weight_axis_offsets;
  uint32_t weight_axis_count;
  uint32_t weight_axis;
  float output_scale;
  int32_t output_offset;
  // Zero selects the retained u8 path. FLOAT_16 requires two-byte elements and
  // is used only by the separately versioned v5 candidate.
  uint32_t data_type;
  uint32_t element_bytes;
} TfQnnSessionConfig;

typedef struct TfQnnSessionTiming {
  int32_t status;
  double create_ms;
  double finalize_ms;
  double execute_ms;
  double max_abs_error;
} TfQnnSessionTiming;

TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_session_create(
    const TfQnnSessionConfig* config, TfQnnSession** session, TfQnnSessionTiming* timing);
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_session_set_weights(
    TfQnnSession* session, const uint8_t* weights, uint64_t weight_bytes);
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_session_finalize(TfQnnSession* session, TfQnnSessionTiming* timing);
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_session_execute(
    TfQnnSession* session, const uint8_t* input, uint64_t input_bytes, uint8_t* output,
    uint64_t output_bytes, TfQnnSessionTiming* timing);
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_session_parity(
    TfQnnSession* session, const uint8_t* output, uint64_t output_bytes, const float* reference,
    uint64_t reference_count, float tolerance, TfQnnSessionTiming* timing);
// Releases the session object. The pointer is invalid after this call.
TF_QNN_BRIDGE_EXPORT void tf_qnn_session_destroy(TfQnnSession* session);

typedef struct TfQnnPairSession TfQnnPairSession;
typedef struct TfQnnPairSessionConfig {
  const char* native_dir;
  const char* stub_name;
  uint32_t k;
  uint32_t qkv_m;
  uint32_t z_m;
} TfQnnPairSessionConfig;

typedef struct TfQnnPairSessionTiming {
  int32_t status;
  double create_ms;
  double finalize_ms;
  double execute_ms;
  double qkv_max_abs_error;
  double z_max_abs_error;
} TfQnnPairSessionTiming;

TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_pair_session_create(
    const TfQnnPairSessionConfig* config, TfQnnPairSession** session,
    TfQnnPairSessionTiming* timing);
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_pair_session_set_payload(
    TfQnnPairSession* session, const uint8_t* input, uint64_t input_bytes,
    const uint8_t* qkv_weights, uint64_t qkv_weight_bytes,
    const uint8_t* z_weights, uint64_t z_weight_bytes);
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_pair_session_finalize(
    TfQnnPairSession* session, TfQnnPairSessionTiming* timing);
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_pair_session_execute(
    TfQnnPairSession* session, uint8_t* qkv_output, uint64_t qkv_output_bytes,
    uint8_t* z_output, uint64_t z_output_bytes, TfQnnPairSessionTiming* timing);
TF_QNN_BRIDGE_EXPORT int32_t tf_qnn_pair_session_parity(
    TfQnnPairSession* session, const uint8_t* qkv_output, uint64_t qkv_output_bytes,
    const float* qkv_reference, float qkv_tolerance, const uint8_t* z_output,
    uint64_t z_output_bytes, const float* z_reference, float z_tolerance,
    TfQnnPairSessionTiming* timing);
TF_QNN_BRIDGE_EXPORT void tf_qnn_pair_session_destroy(TfQnnPairSession* session);

#ifdef __cplusplus
}
#endif
