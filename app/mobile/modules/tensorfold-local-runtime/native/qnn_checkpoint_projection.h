#pragma once

#include <cstdint>

namespace tensorfold::qnn_checkpoint_projection {

// This compact int8-domain slice is derived deterministically from the checked-in
// public EXL3 oracle fixture, not from a locally downloaded model. Provenance:
//   model: UnstableLlama/Qwen3.5-2B-exl3-8.00bpw/model.safetensors
//   pinned revision: content-addressed model.safetensors SHA-256 below
//   model SHA-256: a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f
//   fixture: zig/tests/fixtures/exl3_oracle.bin
//   fixture SHA-256: 8eecb6ac56325a50634eaf75cd2c2ff4aed74e4341b5f2af6c05a3bf0985c810
//   source tensor: fixture's public, unlabeled real mcg 8.00-bpw block
//   source block: 128x128, K rows [0, 4), N columns [0, 3)
// The fixture format does not retain the original checkpoint tensor name. Its
// checked-in decoder test establishes these bytes as the public checkpoint block.
// The source weights are the fixture's dequantized, rotated-domain values. The
// fixed activation is [-0.5, -0.25, 0.25, 0.5]. Values use QNN scale-offset
// quantization: real = scale * (u8 + offset). The CPU reference below is an
// independently calculated dot over the quantized values before output rounding.
constexpr uint32_t kInputColumns = 4;
constexpr uint32_t kOutputColumns = 3;
constexpr float kInputScale = 0.01f;
constexpr int32_t kInputOffset = -128;
constexpr float kWeightScale = 0.0005f;
constexpr int32_t kWeightOffset = -128;
constexpr float kOutputScale = 0.0005f;
constexpr int32_t kOutputOffset = -128;
constexpr float kParityTolerance = 0.0005f;

constexpr uint8_t kInput[kInputColumns] = {78, 103, 153, 178};
constexpr uint8_t kWeights[kInputColumns * kOutputColumns] = {
    133, 166, 157,
    119, 109, 113,
    133, 122, 128,
    68, 93, 114,
};

constexpr int32_t kQuantizedDot[kOutputColumns] = {-2900, -3325, -1775};

// kQuantizedDot * kInputScale * kWeightScale, before output u8 rounding.
constexpr float kQuantizedReference[kOutputColumns] = {
    -0.0145f, -0.016625f, -0.008875f,
};

static_assert(kQuantizedDot[0] == (78 - 128) * (133 - 128) + (103 - 128) * (119 - 128) +
                                      (153 - 128) * (133 - 128) + (178 - 128) * (68 - 128));
static_assert(kQuantizedDot[1] == (78 - 128) * (166 - 128) + (103 - 128) * (109 - 128) +
                                      (153 - 128) * (122 - 128) + (178 - 128) * (93 - 128));
static_assert(kQuantizedDot[2] == (78 - 128) * (157 - 128) + (103 - 128) * (113 - 128) +
                                      (153 - 128) * (128 - 128) + (178 - 128) * (114 - 128));

}  // namespace tensorfold::qnn_checkpoint_projection
