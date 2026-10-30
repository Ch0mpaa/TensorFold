#pragma once

// NVIDIA NVFP4 (ModelOpt) routed-expert decode on HIP (gfx1151). Software dequant,
// fp32 accumulate. Consumes the RAW shard layout as stored on disk and uploaded by
// zig/src/core/quant/nvfp4.zig::upload — no CPU pre-pack is required:
//
//   codes   : [E][N][K/2] u8   (two 4-bit e2m1 nibbles per byte, low nibble first)
//   scales  : [E][N][K/16] u8  (one e4m3 byte per 16-input block)
//   globals : [E]      f32     (per-expert global, = 1 / input_scale)
//
// Arithmetic matches zig/src/core/quant/nvfp4.zig::reference exactly:
//   dequant(e, row, k) = e2m1(nib(e, row, k)) * e4m3(scale(e, row, k/16)) * globals[e]

#include "common.hpp"

#include <hip/hip_bf16.h>

namespace {

// e2m1 nibble (sign bit 3, magnitude bits 0..2) decoded to fp32.
__device__ __forceinline__ float nvfp4_e2m1(uint32_t nib) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = mag[nib & 7u];
    return (nib & 8u) ? -v : v;
}

// e4m3 byte (NVIDIA NVFP4 scale encoding, bias 7) decoded to fp32.
__device__ __forceinline__ float nvfp4_e4m3(uint8_t sf) {
    const uint32_t e = (static_cast<uint32_t>(sf) >> 3) & 0xFu;
    const uint32_t m = static_cast<uint32_t>(sf) & 0x7u;
    float f;
    if (e == 0u) {
        f = (m == 0u) ? 0.f : ldexpf(static_cast<float>(m), -9);
    } else {
        f = ldexpf(1.f + static_cast<float>(m) * 0.125f, static_cast<int>(e) - 7);
    }
    return (sf & 0x80u) ? -f : f;
}

// Dot one output row of one expert (reading its RAW shard bytes) against one bf16 input row.
// `codes_row` is the start of this (expert, out_row)'s nibbles ([K/2] bytes).
// `scales_row` is the start of this (expert, out_row)'s block scales ([K/16] bytes).
__device__ __forceinline__ float nvfp4_dot_row(const uint8_t* __restrict__ codes_row,
                                               const uint8_t* __restrict__ scales_row,
                                               const __hip_bfloat16* __restrict__ x_row,
                                               int K, float global_scale) {
    float acc = 0.f;
    for (int block = 0; block < K; block += 16) {
        const float sf = nvfp4_e4m3(scales_row[block >> 4]) * global_scale;
        #pragma unroll
        for (int off = 0; off < 16; ++off) {
            const int k_in = block + off;
            if (k_in >= K) break;
            const uint8_t byte = codes_row[k_in >> 1];
            const uint32_t nib = (k_in & 1) ? (static_cast<uint32_t>(byte) >> 4) : (static_cast<uint32_t>(byte) & 0xFu);
            const float w = nvfp4_e2m1(nib) * sf;
            const float xv = __bfloat162float(x_row[k_in]);
            acc = fmaf(w, xv, acc);
        }
    }
    return acc;
}

}  // namespace

// Routed-expert NVFP4 decode, reference (software dequant, fp32 accumulate, fp32 output).
// Grid: one block per (item, output-column-group); blockIdx.x = it * nb + cb; 32 threads/block.
// Each thread owns output column `cb * 32 + threadIdx.x` (equivalently, output row `cb*32 + c`
// of this expert's weight matrix).
//
// Args:
//   x        : bf16 input rows, laid out [max_pair, x_stride]
//   x_stride : row stride in bf16 elements
//   slots    : 0 when the pair row IS the token row; otherwise divisor (pair = token*slots + idx)
//   codes    : raw NVFP4 codes [E * N * K/2] u8, nibble-packed (low nibble first)
//   scales   : raw NVFP4 block scales [E * N * K/16] e4m3 bytes
//   globals  : per-expert global [E] fp32
//   K        : input width (must be a multiple of 16)
//   N        : output width per expert (must be a multiple of 32)
//   items    : [item_count][3] int32: (expert, first_member, count)
//   counts   : unused for this reference (kept for API parity)
//   members  : [sum(counts)] int32: pair indices
//   out      : fp32 output [pair_count * N]
//   limit    : SwiGLU clamp (unused in down epilogue; kept for API parity)
//   skip     : expert id to skip (-1 for none)
extern "C" __global__ void __launch_bounds__(32) tf_nvfp4_expert_rows_f32(
    const __hip_bfloat16* __restrict__ x, int x_stride, int slots,
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const float* __restrict__ globals,
    int K, int N,
    const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    float* __restrict__ out, float limit, int skip) {
    (void)counts;
    (void)limit;
    const int nb = N >> 5;                   // N / 32
    const int it = blockIdx.x / nb;
    const int cb = blockIdx.x % nb;
    const int c = threadIdx.x;               // output column within the 32-wide block
    const int e = items[3 * it + 0];
    const int first = items[3 * it + 1];
    const int cnt = items[3 * it + 2];
    if (e == skip || cnt == 0) return;
    if (c >= 32) return;

    const int out_row = cb * 32 + c;         // this thread's output row of expert e's weight
    const float g = globals[e];

    // Pointer to this (expert, out_row)'s raw codes and scales.
    const size_t codes_expert_stride = static_cast<size_t>(N) * static_cast<size_t>(K >> 1);
    const size_t scales_expert_stride = static_cast<size_t>(N) * static_cast<size_t>(K >> 4);
    const uint8_t* codes_row = codes + static_cast<size_t>(e) * codes_expert_stride
                                     + static_cast<size_t>(out_row) * static_cast<size_t>(K >> 1);
    const uint8_t* scales_row = scales + static_cast<size_t>(e) * scales_expert_stride
                                       + static_cast<size_t>(out_row) * static_cast<size_t>(K >> 4);

    for (int rr = 0; rr < cnt; ++rr) {
        const int member = members[first + rr];
        const int token = slots ? (member / slots) : member;
        const __hip_bfloat16* x_row = x + static_cast<size_t>(token) * static_cast<size_t>(x_stride);
        const float acc = nvfp4_dot_row(codes_row, scales_row, x_row, K, g);
        out[static_cast<size_t>(member) * static_cast<size_t>(N) + out_row] = acc;
    }
}

// Prompt-tile variant: same arithmetic, each launch covers up to BLOCK_ROWS pairs for the item.
// Grid: blockIdx.x = it * nb + cb, blockIdx.y = row_tile. 32 threads, one thread per output column.
extern "C" __global__ void __launch_bounds__(32) tf_nvfp4_expert_prompt_f32(
    const __hip_bfloat16* __restrict__ x, int x_stride, int slots,
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const float* __restrict__ globals,
    int K, int N,
    const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    float* __restrict__ out, float limit, int skip) {
    (void)counts;
    (void)limit;
    constexpr int kBlockRows = 128;          // mirrors zig/src/hip/moe.zig BLOCK_ROWS
    const int nb = N >> 5;
    const int it = blockIdx.x / nb;
    const int cb = blockIdx.x % nb;
    const int row_tile = blockIdx.y;
    const int c = threadIdx.x;
    const int e = items[3 * it + 0];
    const int first = items[3 * it + 1];
    const int cnt = items[3 * it + 2];
    if (e == skip || cnt == 0) return;
    if (c >= 32) return;
    const int row_lo = row_tile * kBlockRows;
    if (row_lo >= cnt) return;
    const int row_hi = (row_lo + kBlockRows < cnt) ? (row_lo + kBlockRows) : cnt;

    const int out_row = cb * 32 + c;
    const float g = globals[e];
    const size_t codes_expert_stride = static_cast<size_t>(N) * static_cast<size_t>(K >> 1);
    const size_t scales_expert_stride = static_cast<size_t>(N) * static_cast<size_t>(K >> 4);
    const uint8_t* codes_row = codes + static_cast<size_t>(e) * codes_expert_stride
                                     + static_cast<size_t>(out_row) * static_cast<size_t>(K >> 1);
    const uint8_t* scales_row = scales + static_cast<size_t>(e) * scales_expert_stride
                                       + static_cast<size_t>(out_row) * static_cast<size_t>(K >> 4);

    for (int rr = row_lo; rr < row_hi; ++rr) {
        const int member = members[first + rr];
        const int token = slots ? (member / slots) : member;
        const __hip_bfloat16* x_row = x + static_cast<size_t>(token) * static_cast<size_t>(x_stride);
        const float acc = nvfp4_dot_row(codes_row, scales_row, x_row, K, g);
        out[static_cast<size_t>(member) * static_cast<size_t>(N) + out_row] = acc;
    }
}

// Non-routed NVFP4 projection: out = x @ W^T, fp32 output.
// Grid: blockIdx.x = output-column-group (N / 32), blockIdx.y = token.
// 32 threads/block; thread c owns output column cb * 32 + c of the single-stack weight.
extern "C" __global__ void __launch_bounds__(32) tf_nvfp4_rows_f32(
    const __hip_bfloat16* __restrict__ x, int x_stride,
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const float* __restrict__ globals,
    int K, int N,
    float* __restrict__ out, int m) {
    const int nb = N >> 5;
    const int cb = blockIdx.x;
    const int token = blockIdx.y;
    const int c = threadIdx.x;
    if (cb >= nb || token >= m || c >= 32) return;
    const int out_row = cb * 32 + c;
    if (out_row >= N) return;
    const float g = globals[0];
    const uint8_t* codes_row = codes + static_cast<size_t>(out_row) * static_cast<size_t>(K >> 1);
    const uint8_t* scales_row = scales + static_cast<size_t>(out_row) * static_cast<size_t>(K >> 4);
    const __hip_bfloat16* x_row = x + static_cast<size_t>(token) * static_cast<size_t>(x_stride);
    const float acc = nvfp4_dot_row(codes_row, scales_row, x_row, K, g);
    out[static_cast<size_t>(token) * static_cast<size_t>(N) + out_row] = acc;
}
