#pragma once

// NVIDIA NVFP4 (ModelOpt) routed-expert decode on HIP (gfx1151). Software dequant, fp32 accumulate.
//
// The packed weight layout mirrors zig/src/cuda/experts.zig::packPart:
//   [E][N/32][K/32][parts][144] uint32
// Within a (cb, k_group) block of 144 uint32s:
//   - block[0..128]   = code words. For column c = cb*32 + j*8 + gq, j in [0,4), gq in [0,8),
//     word index gq*16 + t*4 + j (t in [0,4)) carries 8 e2m1 nibbles for the 8 inputs of this
//     (col, k_block t). The nibble for (h, q), h in [0,2), q in [0,4), input k_in = h*16 + t*4 + q,
//     sits at bit `4 * (2h + q/2 + 4*(q%2))`.
//   - block[128..144] = 64 bytes of e4m3 per-16-input scales. For column c with
//     j = c/8, r = c%8, t_scale = r/2, ci = r%2, the scale byte for input half h in [0,2)
//     sits at offset `((t_scale*2 + h)*4 + j)*2 + ci` inside the 64-byte region.
//
// Arithmetic matches zig/src/core/quant/nvfp4.zig::reference: dequant = e2m1(nib) * e4m3(scale) * global.
// Parts = 1 for this reference; multi-part layouts (gate/up fused) are not covered here.

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
        f = (m == 0u) ? 0.f : ldexpf(static_cast<float>(m), -9);  // subnormal: m * 2^-9
    } else {
        f = ldexpf(1.f + static_cast<float>(m) * 0.125f, static_cast<int>(e) - 7);
    }
    return (sf & 0x80u) ? -f : f;
}

// Byte offset of the e4m3 scale for (column within cb, input half h) inside the 64-byte scale tail.
__device__ __forceinline__ int nvfp4_scale_offset(int col_within_cb, int h) {
    const int j = col_within_cb >> 3;         // col / 8, j in [0, 4)
    const int r = col_within_cb & 7;          // col % 8
    const int t_scale = r >> 1;               // r / 2, in [0, 4)
    const int ci = r & 1;                     // r % 2, in [0, 2)
    return ((t_scale * 2 + h) * 4 + j) * 2 + ci;
}

// Bit offset within a code word for (input half h, lane q), h in [0, 2), q in [0, 4).
__device__ __forceinline__ unsigned nvfp4_slot(unsigned h, unsigned q) {
    return 4u * (2u * h + (q >> 1) + 4u * (q & 1u));
}

// Dequant-and-dot one (column, k_group) tile for an output column against bf16 input `x_row`.
// Returns the partial fp32 sum over the 32 K-inputs of this k_group (one scalar add chain).
__device__ __forceinline__ float nvfp4_dot_group(const uint32_t* __restrict__ blk, int col_within_cb,
                                                 const __hip_bfloat16* __restrict__ x_row, int k_base,
                                                 float global_scale) {
    const int j = col_within_cb >> 3;
    const int gq = col_within_cb & 7;
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(blk + 128);
    float sf_h[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint8_t sb = sc[nvfp4_scale_offset(col_within_cb, h)];
        sf_h[h] = nvfp4_e4m3(sb) * global_scale;
    }
    float acc = 0.f;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t word = blk[gq * 16 + t * 4 + j];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float sf = sf_h[h];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t nib = (word >> nvfp4_slot(static_cast<unsigned>(h), static_cast<unsigned>(q))) & 0xFu;
                const float w = nvfp4_e2m1(nib) * sf;
                const int k_in = k_base + h * 16 + t * 4 + q;
                const float xv = __bfloat162float(x_row[k_in]);
                acc = fmaf(w, xv, acc);
            }
        }
    }
    return acc;
}

}  // namespace

// Routed-expert NVFP4 decode, reference (software dequant, fp32 accumulate, fp32 output).
// Grid: one block per (item, output-column-group) with blockIdx.x = it * nb + cb.
// Block: 32 threads, lane t owns output column cb*32 + t.
//
// Args:
//   x        : bf16 input rows, laid out [max_pair, x_stride]
//   x_stride : row stride in bf16 elements
//   slots    : 0 when the pair row IS the token row; otherwise divisor (pair = token * slots + slot_idx)
//   weight   : packed NVFP4 blocks [E][N/32][K/32][144] uint32 (parts = 1 for this reference)
//   scales   : per-expert fp32 global scale [E]
//   kg       : K / 32
//   nb       : N / 32
//   items    : [item_count][3] int32: (expert, first_member, count)
//   counts   : unused for this reference (kept for API parity)
//   members  : [sum(counts)] int32: pair indices
//   out      : fp32 output [pair_count * N]
//   N        : output width per expert
//   limit    : SwiGLU clamp (unused in down epilogue; kept for API parity)
//   skip     : expert id to skip (-1 for none)
extern "C" __global__ void __launch_bounds__(32) tf_nvfp4_expert_rows_f32(
    const __hip_bfloat16* __restrict__ x, int x_stride, int slots,
    const uint32_t* __restrict__ weight, const float* __restrict__ scales,
    int kg, int nb,
    const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    float* __restrict__ out, int N, float limit, int skip) {
    (void)counts;
    (void)limit;
    const int it = blockIdx.x / nb;
    const int cb = blockIdx.x % nb;
    const int c = threadIdx.x;              // output column within the 32-wide block
    const int e = items[3 * it + 0];
    const int first = items[3 * it + 1];
    const int cnt = items[3 * it + 2];
    if (e == skip || cnt == 0) return;
    if (c >= 32) return;
    const float g = scales[e];
    const size_t expert_stride = static_cast<size_t>(nb) * static_cast<size_t>(kg) * 144u;
    const size_t cb_stride = static_cast<size_t>(kg) * 144u;
    for (int rr = 0; rr < cnt; ++rr) {
        const int member = members[first + rr];
        const int token = slots ? (member / slots) : member;
        const __hip_bfloat16* x_row = x + static_cast<size_t>(token) * static_cast<size_t>(x_stride);
        float acc = 0.f;
        for (int k_group = 0; k_group < kg; ++k_group) {
            const uint32_t* blk = weight + static_cast<size_t>(e) * expert_stride
                                         + static_cast<size_t>(cb) * cb_stride
                                         + static_cast<size_t>(k_group) * 144u;
            acc += nvfp4_dot_group(blk, c, x_row, k_group * 32, g);
        }
        out[static_cast<size_t>(member) * static_cast<size_t>(N) + cb * 32 + c] = acc;
    }
}

// Prompt-tile variant: same arithmetic, but each launch covers up to BLOCK_ROWS pairs for the item.
// Grid: blockIdx.x = it * nb + cb, blockIdx.y = row_tile. 32 threads, one thread per output column.
// For the reference, this is a straight loop over the tile's pairs; no shared staging.
extern "C" __global__ void __launch_bounds__(32) tf_nvfp4_expert_prompt_f32(
    const __hip_bfloat16* __restrict__ x, int x_stride, int slots,
    const uint32_t* __restrict__ weight, const float* __restrict__ scales,
    int kg, int nb,
    const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    float* __restrict__ out, int N, float limit, int skip) {
    (void)counts;
    (void)limit;
    constexpr int kBlockRows = 128;         // mirrors zig/src/hip/moe.zig BLOCK_ROWS
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
    const float g = scales[e];
    const size_t expert_stride = static_cast<size_t>(nb) * static_cast<size_t>(kg) * 144u;
    const size_t cb_stride = static_cast<size_t>(kg) * 144u;
    for (int rr = row_lo; rr < row_hi; ++rr) {
        const int member = members[first + rr];
        const int token = slots ? (member / slots) : member;
        const __hip_bfloat16* x_row = x + static_cast<size_t>(token) * static_cast<size_t>(x_stride);
        float acc = 0.f;
        for (int k_group = 0; k_group < kg; ++k_group) {
            const uint32_t* blk = weight + static_cast<size_t>(e) * expert_stride
                                         + static_cast<size_t>(cb) * cb_stride
                                         + static_cast<size_t>(k_group) * 144u;
            acc += nvfp4_dot_group(blk, c, x_row, k_group * 32, g);
        }
        out[static_cast<size_t>(member) * static_cast<size_t>(N) + cb * 32 + c] = acc;
    }
}
