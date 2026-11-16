# TFOLD kernel bottleneck analysis (TF_PIPE_TOKENS=5 stage, 48 logit rows)

Receipts: `nsys_bf16_tfold.nsys-rep`, `nsys_exl3_tfold.nsys-rep`, `kernels_bf16.csv`, `kernels_exl3.csv`.

## BF16 (lossless `.tfold` through `tf-qwen35-cuda`)

| rank | kernel | % GPU time | total (ns) | avg (ns) | count |
|---|---|---|---|---|---|
| 1 | `tf_qmmf::qmmf_kernel<4,64,64,1,4,4,0,0,1>` (BF16 GEMM, "the big one") | **75.1%** | 1,108,461,504 | 2,041,365 | 543 |
| 2 | `tf_pack_bf16_lane` | 4.7% | 68,813,056 | 367,984 | 187 |
| 3 | `tf_gdn_prefill::chain_kernel<bf16,bf16,64,32>` | 4.0% | 59,571,008 | 827,375 | 72 |
| 4 | `tf_q35_mlp_act` | 3.1% | 46,110,720 | 480,320 | 96 |
| 5 | `tf_q35_norm_shift` | 3.0% | 44,126,848 | 225,137 | 196 |
| 6 | `qmmf_kernel<4,16,64,…>` (narrow variant) | 2.4% | 35,276,832 | 8,819,208 | 4 |
| 7 | `qmmf_kernel<4,32,64,…>` (medium variant) | 2.4% | 35,068,736 | 188,541 | 186 |

Sum of all `qmmf_kernel` variants: **~80%**. The BF16 path is matmul-bound.

## EXL3-8bpw (TFOLD native)

| rank | kernel | % GPU time | total (ns) | avg (ns) | count |
|---|---|---|---|---|---|
| 1 | `tf_exl3_linear_16_1_4` (trellis-dequant+matmul fused) | **66.9%** | 6,855,719,072 | 1,071,206 | 6400 |
| 2 | `tf_exl3_rot_in` (input rotation for trellis decode) | **30.1%** | 3,082,108,382 | 481,278 | 6404 |
| 3 | `tf_q35_dense_bf16` | 0.5% | 54,252,992 | 34,250 | 1584 |
| 4 | `tf_gdn_prefill::chain_kernel<bf16,bf16,64,32>` | 0.5% | 54,215,872 | 752,998 | 72 |
| 5 | `tf_q35_mlp_act` | 0.5% | 51,836,062 | 539,959 | 96 |
| 6 | `tf_q35_norm` | 0.4% | 40,727,355 | 207,792 | 196 |
| 7 | `tf_q35_gdn_pre` | 0.3% | 30,242,784 | 420,038 | 72 |
| 8 | `tf_prefill_attention::hpattn_whole_kernel` | 0.3% | 29,828,704 | 1,242,862 | 24 |
| 9 | `tf_exl3_linear_12_1_8` | 0.2% | 17,825,248 | 4,456,312 | 4 |

Sum of EXL3 projection + rotation: **~97%**. The EXL3 path is bound by two
coupled kernels, launched in a near-1:1 pair ratio (6400 vs 6404 instances),
each averaging ~1.5 ms together for an average matmul tile.

## Top-1 bottleneck (declared)

- **For BF16**: `qmmf_kernel` (BF16 matmul). This is the cost of running
  dense arithmetic at half precision; there is no quantization overhead to
  attack.
- **For EXL3**: the `tf_exl3_linear_* + tf_exl3_rot_in` pair. The rotation
  kernel is launched separately for every projection, consuming 30% of GPU
  time by itself. Each launch is small (med 180 µs) but there are 6,400 of
  them.

## Memory side

- **BF16** cuda_gpu_mem_time_sum: HtoD 74.8% (194 MB/s streamed to device
  during stage), memsets 24.9%, DtoH 0.4%. A lot of per-step upload traffic.
- **EXL3** (not shown in CSV above, same `[CUDA memcpy Host-to-Device]`
  category dominates): tensors are pre-packed per layer into trellis tiles
  and dispatched; memcpy is tiny vs kernel time.

## Why this is the right candidate target

The EXL3 `rot_in` kernel is a Hadamard-like transform on the input
activation vector *that happens once per linear call*. It is a candidate
for **fuse-into-prologue**: the rotation can be absorbed into the first
MMA of `tf_exl3_linear_16_1_4` by premultiplying the input tile in
registers before the quant decode. Eliminating `rot_in` as a separate
kernel saves:

- Up to 30% of EXL3 GPU time (upper bound; real win depends on fusion
  overhead).
- Kernel-launch count (6,400 fewer launches per stage), which also removes
  scheduler/driver overhead not visible in `cuda_gpu_kern_sum`.

## What the optimization is NOT

- NOT a new quantizer. EXL3's trellis coding and bit layout are unchanged.
- NOT a fidelity change. The transform is algebraically identical to the
  separate kernel's output.
- NOT a GGUF/Q4_K_M change (different format, different kernel).

See `../candidate_optimization.md` for the proposal, scope and success
criteria.
