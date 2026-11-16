# Lossless Qwen3.5-2B → TFOLD v0.2 conversion (unquantized)

## Scope
Produce a **lossless** `.tfold` checkpoint from the upstream Qwen/Qwen3.5-2B
safetensors shard — no EXL3, GGUF, or quantizer in the pipeline, no silent dtype
casts, and byte-for-byte parity with the upstream weights.

The existing EXL3-sourced `/home/chris/models/tfold/Qwen3.5-2B-exl3-mcg.tfold`
(sha `027ee2964e3868fa…`, built Nov 14 22:04 UTC) stays unchanged and serves as
the quantized baseline for future comparisons.

## Upstream pin
- repo: `Qwen/Qwen3.5-2B`
- revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`
- license: Apache-2.0
- pipeline: image-text-to-text (base-LM + vision tower in one safetensors shard)
- `architectures`: `Qwen3_5ForConditionalGeneration`
- `model_type`: `qwen3_5`
- `hidden_size`: 2048, `num_hidden_layers`: 24, `vocab_size`: 248320
- `dtype`: bfloat16
- shard: `model.safetensors-00001-of-00001.safetensors`, 4,548,221,488 bytes
  (`sha256 aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`)
- tokenizer.json sha: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- LICENSE sha: `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`
- config.json sha: `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`

## Tensor inventory (upstream shard)
- 632 tensors across 1 shard
- prefixes: `model` (617) — includes language-model and vision subtrees
  interleaved with `model.language_model.*` and `model.visual.*`; plus `mtp` (15)
  — the multi-token-predictor head Qwen3.5 uses for speculative decoding.
- dtype distribution: 596 `bf16`, 36 `fp32` (all the `linear_attn.A_log` scalars
  and `linear_attn.norm.weight` tensors — standard hybrid Mamba-style norms).

## TFOLD v0.2 container (spec extracted from `src/core/tfold_file.zig`)
```
bytes 0..8    : MAGIC = "TFOLD\x02\x00\x00"
bytes 8..12   : version u32 LE = 0x00020000
bytes 12..16  : header_len u32 LE
bytes 16..16+header_len : UTF-8 JSON header
padded to 64-byte boundary
body: raw tensor bytes at absolute offsets recorded in header; each
      tensor offset must be 64-byte aligned
trailer: last 8 bytes = FNV-1a-64 of every byte before them, little-endian
```
Header JSON (fields we populated):
```jsonc
{
  "tfold":    {"version": "0.2", "writer": "tfold-convert-lossless@0.1.0"},
  "model":    {"name", "arch": "qwen3_5", "parameters", "vocab_size",
               "context_length", "source_sha256", "source_url",
               "upstream_license"},
  "tokenizer": {"type": "qwen3_5-tokenizer", "sha256"},
  "quant":    {"scheme": "none"},
  "tensors":  [{"name", "dtype", "shape", "offset", "nbytes", "sha256",
                "source_byte_range"} ...]
}
```
Dtype allow-list for TFOLD v0.2 (from the Zig reader):
`bf16, fp16/f16, fp32/f32, u32, i16, i32, exl3_mcg_tile, exl3_mul1_tile,
exl3_3inst_tile`. BF16 and FP32 are native — no format extension is required
for a lossless unquantized Qwen3.5-2B checkpoint.

## Converter (`convert_safetensors_to_tfold.py`)
- Streams the safetensors shard; for every tensor:
  - preserves the original name, shape and raw byte payload verbatim
  - maps the safetensors dtype 1:1 to a TFOLD v0.2 dtype
  - **aborts** on any dtype outside the allow-list — no silent casts
  - records `sha256` and the original `source_byte_range`
- Sets `quant.scheme = "none"` to signal identity storage.
- Pads each tensor's absolute offset up to the 64-byte boundary the Zig reader
  enforces (`error.TfoldTensorMisaligned`).
- Appends an 8-byte little-endian FNV-1a-64 trailer over every byte of the
  preceding file, via a small `fnv_append.c` helper compiled with
  `gcc -O3 -o /tmp/fnv_append fnv_append.c`. The Python byte-loop version was
  prohibitively slow on a 4.5 GB file; the C helper runs in ~3.5 s and matches
  the Zig `fnv1a64_scalar` constants (`offset 0xcbf29ce484222325`, `prime
  0x100000001b3`) bit-for-bit.

## Independent parity check (`verify_parity.py`)
Re-opens both files and re-hashes every tensor's bytes:
```
PASS: 632 tensors byte-identical
{"ok": true, "n_tensors_checked": 632, "n_mismatches": 0}
```
No shape, nbytes, source-SHA or manifest-SHA mismatches.

## Output
- `/home/chris/models/tfold/Qwen3.5-2B-orig-bf16.tfold` — 4,548,300,616 bytes
- `/home/chris/models/tfold/Qwen3.5-2B-orig-bf16.manifest.json` — all 632 tensor
  hashes + source pins
- `/home/chris/models/tfold/Qwen3.5-2B-orig-bf16.parity.json` — PASS

## Native-loader probe (tf-qwen35-cuda)
The Zig CLI prefers `dir/model.tfold` when present. With the full HF directory
layout (config.json, tokenizer.json, …) and our `.tfold` hardlinked as
`model.tfold`, the loader:
1. opened the mmap successfully (`mmap=36us`),
2. parsed the 8-byte magic + 4-byte version + 155129-byte header JSON
   (`preamble=344us`, `header_parse=3094us`),
3. verified the full-file FNV-1a-64 trailer
   (`fnv=3.6s`, `fnv_bytes=4548300608`),
4. populated the tensor directory, then
5. halted with
   `error: model.language_model.layers.0.linear_attn.norm.weight: f32 { 128 },
   expected bf16 { 128 } → error.UnexpectedTensor`.

## Honest gap
Steps 1-4 prove the container, integrity and per-tensor placement match the
Zig spec exactly. Step 5 is a loader-side strictness: `cuda_weights.zig:413`
calls `tensor(..., "linear_attn.norm.weight", .bf16, ...)` for the dense CUDA
path. The EXL3 CUDA path already has the exact fallback we need
(`cuda_exl3_weights.zig:315` uses `bf16OrF32(... "linear_attn.norm.weight")`
with the comment "gated norm: unshifted (f32 widened to bf16)"). All 36 fp32
tensors in the upstream model are either `linear_attn.A_log` (already loaded
via `f32vec`) or `linear_attn.norm.weight` (the blocker).

**Minimum extension** to execute the unquantized `.tfold` on the dense CUDA
path: replace the strict `.bf16` call on `cuda_weights.zig:413` with the same
`bf16OrF32` helper the EXL3 path uses. No new kernel, no new dtype — the
fallback is already tested on EXL3 loads. This change is out of scope for the
lossless-conversion milestone but is a single-line follow-up.

## Explicit non-claims
- The conversion itself does not change numerical precision, speed or memory
  footprint. There is no basis to claim any model-quality or throughput
  improvement "because of" the lossless `.tfold`.
- No logit-level parity vs the upstream model was measured in this milestone;
  the loader gap above prevents it on the dense CUDA path. The EXL3 `.tfold`
  baseline remains unchanged and its throughput numbers
  (warm median 110.8 tok/s, VRAM 2208 MB) continue to stand for the quantized
  path.
- The 2B class remains **not graduated**. OFFSEC CWE-311/327 is an unresolved
  independent blocker.

## Files shipped
- `docs/tfold/convert_safetensors_to_tfold.py` — converter
- `docs/tfold/verify_parity.py` — byte-level parity verifier
- `docs/tfold/fnv_append.c` — FNV-1a-64 trailer helper (gcc -O3)
- `docs/tfold/receipts/parity.json` — PASS report
- `docs/tfold/receipts/container_head.txt` — first 64 bytes + file size of the
  output `.tfold`
- `learning-notes/tfold-original-qwen35-conversion.md` — beginner-friendly
  journal of this milestone

## Reproduction (Spark)
```
cd /home/chris/models/qwen35-2b-orig
gcc -O3 -o /tmp/fnv_append fnv_append.c
/home/chris/opt/exllamav3-venv/bin/python convert_safetensors_to_tfold.py \
  --safetensors model.safetensors-00001-of-00001.safetensors \
  --config config.json --tokenizer tokenizer.json --license LICENSE \
  --out      /home/chris/models/tfold/Qwen3.5-2B-orig-bf16.tfold \
  --manifest /home/chris/models/tfold/Qwen3.5-2B-orig-bf16.manifest.json \
  --model-name Qwen3.5-2B-orig-bf16

/home/chris/opt/exllamav3-venv/bin/python verify_parity.py \
  --safetensors model.safetensors-00001-of-00001.safetensors \
  --tfold      /home/chris/models/tfold/Qwen3.5-2B-orig-bf16.tfold \
  --report     /home/chris/models/tfold/Qwen3.5-2B-orig-bf16.parity.json
```
