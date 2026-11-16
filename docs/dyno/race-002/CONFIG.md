# Dyno Race 002 — Frozen Configuration

Everything referenced below is pinned by sha256. No comparison is accepted
without the exact artifact sha.

## Upstream model (frozen)
- repo: `Qwen/Qwen3.5-2B`
- revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`
- license: Apache-2.0
- base shard sha256:  `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`
- config.json sha256: `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`
- tokenizer.json sha256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`

## Candidate inventory
See `receipts/quant_inventory.json`.

- **BF16-reference**: upstream safetensors directory + lossless .tfold at sha `86e1bba7a61309ec…` — 4.55 GB, bpw=16.
- **EXL3-8bpw**: `turboderp/Qwen3.5-2B-exl3@8.00bpw/model.safetensors` sha `a106b1c9d58a768e…` + TFOLD mcg .tfold sha `027ee296…` — 3.44 GB, bpw=8.
- **GGUF-Q8_0**: `bartowski/Qwen_Qwen3.5-2B-GGUF` revision `7d26695454df6de5fbcce2e58681e62dae06ce43`, file `Qwen_Qwen3.5-2B-Q8_0.gguf` — 2.08 GB, bpw=8.5 (8-bit weights + f32 norms).
- **GGUF-Q4_K_M**: same repo/revision, `Qwen_Qwen3.5-2B-Q4_K_M.gguf` — 1.40 GB, bpw=5.705 measured (Q4_K main + Q6_K embed/output).
- **EXL3-4bpw**: NOT PRODUCED (see REPORT; follow-up work).

## Runtimes
- **TFOLD**: `tf-qwen35-cuda` sha `81b2fbad316feb844040395351c4dc1a658bb394c657c7feae5d877aa849d2b0` (prebuilt Zig+nvcc, Race 001 binary).
- **vLLM**: `0.26.0`, dtype=bfloat16, enforce_eager=True, max_model_len=512, gpu_memory_utilization=0.08.
- **llama.cpp**: `0.3.0-dev build 1 commit 0f3a71b`, aarch64+cuda13 (lmstudio pre-built), `-ngl 99 --no-warmup --ctx-size 2048 --parallel 1`.

## Hardware (Race 001-identical)
- DGX Spark GB10 sm_121, driver 615.71.09, CUDA 13.0 toolkit + 13.4 UMD.
- Unified Memory Architecture; nvidia-smi `used_gpu_memory` is reliable but `memory.free/used` on the GPU-level query returns N/A.
- Concurrent tenant (tensorfold-native server) held 97 GiB during the race; runs used `gpu_memory_utilization=0.08` or absolute allocations.

## Sampling policy (fidelity + throughput)
- Deterministic greedy: `temperature=0.0`, `top_p=1.0`, `seed=0`.
- Identical prompt sets: `prompts.txt` (5 prompts) for throughput, `fidelity_bench.sh` embeds 4 × 5 prompts for the 4 fidelity domains.

## Warm-up protocol
- Throughput: `--no-warmup` on llama.cpp, one warm invocation before the timed bench on TFOLD side (Race 001 convention preserved).
- Fidelity: 1 single-prompt call during server readiness probe is implicit warm-up before scored prompts.

## Repro
```bash
# Download + quantize
bash race_002_setup.sh   # downloads bartowski GGUF variants
# Container-identical EXL3
TF_PIPE_TOKENS=5 tf-qwen35-cuda /home/chris/models/Qwen3.5-2B-exl3-8.00bpw stage /tmp/exl3_sft.bin
TF_PIPE_TOKENS=5 tf-qwen35-cuda /tmp/exl3_tfold_dir stage /tmp/exl3_tfold.bin
cmp /tmp/exl3_sft.bin /tmp/exl3_tfold.bin     # expect IDENTICAL
# Per-model throughput
bash gguf_bench.sh /path/to/Qwen_Qwen3.5-2B-Q4_K_M.gguf q4km
# 4-domain fidelity
bash fidelity_bench.sh /path/to/model.gguf label
python3 score_fidelity.py receipts/fidelity_*.json --out receipts/fidelity_summary.json
# Kernel bottleneck
nsys profile --output=receipts/nsys_bf16_tfold --trace=cuda,nvtx \
  tf-qwen35-cuda /tmp/tfold_dir stage /tmp/stage.bin
nsys stats --report cuda_gpu_kern_sum --format csv receipts/nsys_bf16_tfold.nsys-rep
```

## Race 001 corrections carried forward
See `RACE-001-CORRECTION.md` (originally in this directory); the cold-load 55× TFOLD claim was unsupported (prebuilt vs JIT asymmetry). All original Race 001 receipts preserved untouched.
