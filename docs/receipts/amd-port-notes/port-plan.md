# NVFP4 Flash Next on HIP — port plan (gfx1151)

Target: run NVIDIA's NVFP4 ModelOpt checkpoints on an AMD Ryzen AI MAX+ 395
(Radeon 8060S, gfx1151) with the same TensorFold engine that serves them on a
DGX Spark (GB10), and publish an honest A/B.

**First target (fits in 64 GiB container today):**
`nvidia/Qwen3.8-27B-NVFP4` — 22 GB on disk, ~14 GB loaded, family `qwen3_5`
(already fully supported on HIP; main-check 42/42 passed in Phase 1). Only
the NVFP4 format kernels are new work. Serves on CUDA since 1.1.1.

**Stretch target (needs container upsize to ≥128 GiB):**
`Chompa1422/Qwen3.8-Flash-Next-NVFP4-Abliterated` — 102 GB, family
`qwen4_exp`. Same NVFP4 kernels + Flash Next-specific MoE/n-gram paths.
DGX Spark reference serving at `http://192.168.60.228:8888` on recipe commit
`e1f564e`.

## What already exists

- Flash Next family (`qwen4_exp`) **has a HIP skeleton**:
  `zig/src/families/flashnext/hip/{load,tensors,model,engine,forward,head,cache,lanes,mixers,ngram,snapshot,worker,plan,native,tables,checkpoint_tables,state_test}.zig`.
- HIP backend lists `qwen4_exp: ["mlx-q3g32", "mlx-q4g32", "mlx-q6g32"]` in
  `capabilities --json`; serves mlx-affine today.
- CUDA has full NVFP4 Flash Next: `zig/kernels/cuda/nvfp4q.cuh`,
  `nvfp4_experts.cu`, `nvfp4_experts_tile.cu`; launchers at
  `zig/src/cuda/{nvfp4,experts,experts_tile,experts_waves,modelopt,qlinear,qmmf}.zig`.
- `Format` enum (`zig/src/cuda/experts.zig:18`): `affine4, affine3, fp8g, nvfp4, exl3, iq2xxs`.
- `csf_32x2_v1` scale codec in `core/` (shared between backends).
- MiaAI-Lab recipe (TensorFold branch) on the box at
  `/root/recipes/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` with 17
  patches on top of `v1.1.4` (`eab85828`); most are perf-only.

## NVFP4 data layout (from `zig/kernels/cuda/nvfp4q.cuh` + `zig/src/cuda/nvfp4.zig`)

- Codes `[n, k/2]`, two 4-bit e2m1 values per byte (low nibble first).
- Block scales `[n, k/16]`, one e4m3 byte per 16 inputs.
- One fp32 global per tensor (`1/input_scale`).
- Dequant per element: `fp = e2m1_decode(code) * e4m3_decode(block_sf) * (1/global_scale)`.
- Packed storage (`packWords` / `packScales` in `cuda/nvfp4.zig`) interleaves
  for 128x128 prompt tiles and 64x64 lane tiles.

## RDNA 3.5 (gfx1151) targets

- Wave 32. No native FP4 compute. Has WMMA11 FP16 matrix ops (`.matrix = .wmma11`
  in `zig/src/hip/caps.zig:40`). Software e2m1→fp16 and e4m3→fp16 dequant,
  then feed WMMA FP16 tensor ops with fp32 accumulate, bf16 output. Expected
  loss vs Blackwell FP4 tensor cores: ~2x throughput ceiling. Target: close
  the gap with aggressive tile fusion.

## Port scope (branch-local, incremental, upstreamable)

### M1 — ground truth: build main for HIP on gfx1151 (**DONE**)

- `zig build native-hip -Dhip-arch=gfx1151 ...` green with current paths.
- Qwen3.5-9B and Qwen3.8-27B model checks: 42/0/5 and 42/0/5 passed. Serve
  on loopback drains cleanly.

### M2 — NVFP4 reader shared with CUDA

- Register `.nvfp4` in a shared `Format` enum under `core/` (today it is
  CUDA-local in `zig/src/cuda/experts.zig`); CUDA keeps its existing shape.
- Lift `packWords` / `packScales` / decode scaffolding into a backend-agnostic
  module (`core/quant/nvfp4.zig`), with CUDA re-exporting from it.
- No kernel code yet.

### M3 — HIP NVFP4 dequant + reference projection

- Add `zig/kernels/hip/nvfp4q.hip` and `zig/kernels/hip/nvfp4_experts.hip`:
  - `nvfp4q.hip`: e2m1/e4m3 → fp16 inline device helpers (mirror
    `nvfp4q.cuh`, no `__nv_` intrinsics).
  - `nvfp4_experts.hip`: one correct reference kernel for routed-expert
    dequant + GEMM into fp16 accumulators (not yet WMMA). Prioritize bit
    equality with the CUDA path over speed. Output bf16.
- Zig launcher `zig/src/hip/nvfp4.zig` + `zig/src/hip/experts.zig` with the
  same tile sizes and plan structs as CUDA. Register in
  `zig/src/hip/kernels.zig` and `zig/build/hip.zig`.
- Family wiring: `zig/src/families/flashnext/hip/load.zig` dispatches on the
  checkpoint's `format` field for `.nvfp4` through this path.
- Capability: `qwen4_exp` on HIP advertises `modelopt-nvfp4`.
- Acceptance: `tf-flashnext-test` opens the Chompa checkpoint and the
  per-layer residual passes equal a BF16 reference within the family's
  tolerance (same gates CUDA uses).

### M4 — HIP NVFP4 performance tile

- `nvfp4_experts_tile.hip`: WMMA11 FP16 MMA on gfx1151 (16x16x16 FP16, fp32
  acc), persistent 64x64 tile, LDS double-buffered, software e2m1/e4m3
  dequant hoisted outside the inner MMA loop. Mirror CUDA's `K32`
  pipeline from `nvfp4_experts_tile.cu` where it maps.
- Equivalent for prompt (128x128). This is the throughput kernel; perf tuning
  loop uses `hip-affine-tune` as a template.

### M5 — memory to fit 100 GB on gfx1151

- Current container is 64 GiB. Options to raise:
  - Request upsize on the Degima dyno to >= 128 GiB (match DGX Spark).
  - Enable PLE (n-gram) table SSD streaming on HIP: port the CUDA PLE reader
    (`cuda_ple.zig`, `cuda_ple_ahead.zig`) with its `pread` mode to a HIP
    equivalent. On the recipe's DGX Spark path the PLE table is 27 GiB kept
    on SSD and streamed, which trims resident weight by that much.
  - FP8 KV cache (recipe patch `0003-kv-cache-fp8.patch`) ported to HIP; a
    262k-token window drops from `elements * 2` to `elements * 1` bytes.
- Acceptance: `tensorfold-native serve <chompa-nvfp4>` reaches `serving ... loaded`
  without OOM.

### M6 — A/B harness vs the NVIDIA box

- Build `tensorfold-native` on the NVIDIA host from the same current `main`
  SHA with the recipe's 17 patches applied (same recipe the model card
  advertises, pinned to `v1.1.4` / `eab85828`), serve the same checkpoint.
- Run identical prompts under `benchmark` on both:
  - prose / code sets from `docs/benchmarks/`.
  - `serve_check.py` for correctness (token-SHA256 agreement where the
    sampler is greedy).
  - `serve_cancel.py` for cancellation parity.
- Collect: load time, prefill tok/s at 2k/8k/32k, decode tok/s at
  `--parallel 1/2/5`, TTFT p50/p95, native token hashes.
- Publish side-by-side as a new receipt
  `docs/receipts/gfx1151-flash-next-nvfp4-<sha>/`.

## External action items (not blocking M2-M4)

1. **NVIDIA SSH** — password `Volk$peed13` denied through plink across 5
   variants (literal, no symbol, escaped, downcased). Need the actual
   working password or a key.
2. **AMD container memory** — current 64 GiB cap vs ~90 GiB needed to load
   the Chompa checkpoint. Request Degima to resize this dyno to >= 128 GiB
   (same as DGX Spark reference). M5's streaming work mitigates but does
   not eliminate.

## Why this plan is the port, not a reskin

The HIP engine already has the family's control plane. The missing piece is
one format family (NVFP4) and its two kernels (reference + perf tile). That
is the port. Everything on top (PLE streaming, FP8 KV, WMMA tile shapes) is
a performance milestone, not a new backend.
