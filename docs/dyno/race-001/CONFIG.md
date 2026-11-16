# Dyno Race 001 — Frozen Configuration

All numbers below are the inputs. Any receipt that cannot cite these exact hashes is rejected.

## Model
- Repo: `Qwen/Qwen3.5-2B` (Apache-2.0, image-text-to-text, base-LM + vision tower in one shard)
- Revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`
- Local path: `/home/chris/models/qwen35-2b-orig/`
- `model.safetensors-00001-of-00001.safetensors` sha256: `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` (4,548,221,488 bytes)
- `config.json` sha256: `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`
- `tokenizer.json` sha256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- 632 tensors, 596 BF16 + 36 FP32

## Lossless `.tfold`
- Path: `/home/chris/models/tfold/Qwen3.5-2B-orig-bf16.tfold`
- sha256: `86e1bba7a61309eca99da3f263012ce4ed145df419d36b8456fc71c64dac7707`
- Size: 4,548,300,616 bytes
- Loader-boundary parity: 632/632 tensors byte-identical (verify_parity.py, previous milestone)

## Binary
- `tf-qwen35-cuda` sha256: `81b2fbad316feb844040395351c4dc1a658bb394c657c7feae5d877aa849d2b0`
- Size: 30,698,392 bytes
- Built: Zig 0.17.0, `zig build qwen35-cuda -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda-13.0/bin/nvcc`
- Includes the `normBf16OrF32` loader fix + 2 unit tests from commit `0bdfd7efc`.

## vLLM comparator
- vllm `0.26.0` (pip)
- torch `2.11.0+cu130`
- transformers `5.19.0`
- enforce_eager=True (no CUDA graphs)
- dtype=bfloat16 (what vLLM does to FP32 norms is audited separately — see `vllm_dtype_audit.json`)
- max_model_len=512, gpu_memory_utilization=0.08 (contested GPU)

## Hardware
- DGX Spark GB10 (ARM aarch64, sm_121, compute_cap 12.1, Unified Memory Architecture)
- Driver 615.71.09, CUDA UMD 13.4; nvcc 13.0 toolkit
- Host: Linux 6.11-nvidia aarch64

## Prompts (frozen)
Prompt set lives in `prompts.txt` in this directory. Five English prompts with
mixed lengths. Same file is fed to both TFOLD and vLLM.

## Warm-up protocol
- Race A: one throwaway run per path, then 10 timed warm runs per path.
- Race B: one throwaway batch per concurrency, then 3 timed repeats per concurrency.
- Cache: whatever the host has after the first run. No `drop_caches` (no root).
  The `.tfold` path's higher wall clock is an *integrity verification cost*,
  not a cache artifact (confirmed by `fnv_us` phase metric).

## Reproduction
```bash
# Race A (identity + per-phase timing on TFOLD runtime)
BIN=/home/chris/worktrees/tfold/zig-out/bin/tf-qwen35-cuda \
SFT_DIR=/home/chris/models/qwen35-2b-orig \
TFOLD_DIR=/tmp/tfold_dir \
TOKENS=5 OUT_JSONL=/tmp/race_a.jsonl \
bash race_a_bench.sh

# Race B dtype audit (what does vLLM actually hold?)
/home/chris/opt/exllamav3-venv/bin/python vllm_dtype_audit.py \
  --model /home/chris/models/qwen35-2b-orig \
  --dtype bfloat16 \
  --out receipts/vllm_dtype_audit.json

# Race B concurrency sweep
/home/chris/opt/exllamav3-venv/bin/python vllm_sweep.py \
  --model /home/chris/models/qwen35-2b-orig \
  --prompts-file prompts.txt \
  --max-tokens 32 --concurrencies 1,4,16,32 --repeats 3 \
  --out receipts/vllm_sweep.json
```
