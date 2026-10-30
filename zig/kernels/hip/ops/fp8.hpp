#pragma once

// NVIDIA ModelOpt FP8 (per-tensor scalar scale) decode on HIP (gfx1151). Software e4m3
// dequant, fp32 accumulate. Current scope matches zig/src/core/quant/fp8_block.zig:
// one scalar `weight_scale * input_scale` product per projection; the kernel receives
// it as `input_scale` and uses it as the sole multiplier.
//
//   weight       : [..., N, K]  u8 (e4m3 bytes, one per element)
//   input_scale  : scalar f32   (= weight_scale * input_scale, folded on host)
//
// Dequant: `fp(e, n, k) = e4m3(weight[e, n, k]) * input_scale`.
//
// Three entry points, all fp32-out, reference quality (no mfma, no shared tile):
//   tf_fp8_rows_f32         : single projection, [m, K] bf16 x [N, K] -> [m, N]
//   tf_fp8_expert_rows_f32  : routed-expert decode (one block per item, 32 cols a block)
//   tf_fp8_expert_prompt_f32: prompt-tile variant with BLOCK_ROWS row tiles

#include "common.hpp"
#include "nvfp4.hpp"  // reuses nvfp4_e4m3 (bias-7 e4m3 encoding is identical to IEEE/NVIDIA FP8 E4M3)

#include <hip/hip_bf16.h>

namespace {

// Dot one output row of one weight matrix against one bf16 input row.
//   w_row       : start of this (expert, out_row)'s e4m3 bytes ([K] bytes)
//   x_row       : bf16 inputs [K]
//   input_scale : folded scalar (weight_scale * input_scale)
__device__ __forceinline__ float fp8_dot_row(const uint8_t* __restrict__ w_row,
                                             const __hip_bfloat16* __restrict__ x_row,
                                             int K, float input_scale) {
    float acc = 0.f;
    for (int k_in = 0; k_in < K; ++k_in) {
        const float w = nvfp4_e4m3(w_row[k_in]) * input_scale;
        const float xv = __bfloat162float(x_row[k_in]);
        acc = fmaf(w, xv, acc);
    }
    return acc;
}

}  // namespace

// Non-routed FP8 projection: out = x @ W^T, fp32 output.
// Grid: blockIdx.x = output-column-group (N / 32), blockIdx.y = token.
// 32 threads/block; thread c owns output column cb * 32 + c.
extern "C" __global__ void __launch_bounds__(32) tf_fp8_rows_f32(
    const __hip_bfloat16* __restrict__ x, int x_stride,
    const uint8_t* __restrict__ weight, float input_scale,
    int K, int N,
    float* __restrict__ out, int m) {
    const int nb = N >> 5;
    const int cb = blockIdx.x;
    const int token = blockIdx.y;
    const int c = threadIdx.x;
    if (cb >= nb || token >= m || c >= 32) return;
    const int out_row = cb * 32 + c;
    if (out_row >= N) return;

    const uint8_t* w_row = weight + static_cast<size_t>(out_row) * static_cast<size_t>(K);
    const __hip_bfloat16* x_row = x + static_cast<size_t>(token) * static_cast<size_t>(x_stride);

    const float acc = fp8_dot_row(w_row, x_row, K, input_scale);
    out[static_cast<size_t>(token) * static_cast<size_t>(N) + out_row] = acc;
}

// Routed-expert FP8 decode (reference). Mirrors tf_nvfp4_expert_rows_f32.
// Grid: one block per (item, output-column-group); 32 threads/block.
extern "C" __global__ void __launch_bounds__(32) tf_fp8_expert_rows_f32(
    const __hip_bfloat16* __restrict__ x, int x_stride, int slots,
    const uint8_t* __restrict__ weight, float input_scale,
    int K, int N,
    const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    float* __restrict__ out, float limit, int skip) {
    (void)counts;
    (void)limit;
    const int nb = N >> 5;
    const int it = blockIdx.x / nb;
    const int cb = blockIdx.x % nb;
    const int c = threadIdx.x;
    const int e = items[3 * it + 0];
    const int first = items[3 * it + 1];
    const int cnt = items[3 * it + 2];
    if (e == skip || cnt == 0) return;
    if (c >= 32) return;

    const int out_row = cb * 32 + c;
    if (out_row >= N) return;

    const size_t weight_expert_stride = static_cast<size_t>(N) * static_cast<size_t>(K);
    const uint8_t* w_row = weight + static_cast<size_t>(e) * weight_expert_stride
                                  + static_cast<size_t>(out_row) * static_cast<size_t>(K);

    for (int rr = 0; rr < cnt; ++rr) {
        const int member = members[first + rr];
        const int token = slots ? (member / slots) : member;
        const __hip_bfloat16* x_row = x + static_cast<size_t>(token) * static_cast<size_t>(x_stride);
        const float acc = fp8_dot_row(w_row, x_row, K, input_scale);
        out[static_cast<size_t>(member) * static_cast<size_t>(N) + out_row] = acc;
    }
}

// Prompt-tile variant: same arithmetic, each launch covers up to BLOCK_ROWS pairs for the item.
// Grid: blockIdx.x = it * nb + cb, blockIdx.y = row_tile. 32 threads, one thread per output column.
extern "C" __global__ void __launch_bounds__(32) tf_fp8_expert_prompt_f32(
    const __hip_bfloat16* __restrict__ x, int x_stride, int slots,
    const uint8_t* __restrict__ weight, float input_scale,
    int K, int N,
    const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    float* __restrict__ out, float limit, int skip) {
    (void)counts;
    (void)limit;
    constexpr int kBlockRows = 128;  // mirrors zig/src/hip/moe.zig BLOCK_ROWS
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
    if (out_row >= N) return;

    const size_t weight_expert_stride = static_cast<size_t>(N) * static_cast<size_t>(K);
    const uint8_t* w_row = weight + static_cast<size_t>(e) * weight_expert_stride
                                  + static_cast<size_t>(out_row) * static_cast<size_t>(K);

    for (int rr = row_lo; rr < row_hi; ++rr) {
        const int member = members[first + rr];
        const int token = slots ? (member / slots) : member;
        const __hip_bfloat16* x_row = x + static_cast<size_t>(token) * static_cast<size_t>(x_stride);
        const float acc = fp8_dot_row(w_row, x_row, K, input_scale);
        out[static_cast<size_t>(member) * static_cast<size_t>(N) + out_row] = acc;
    }
}
