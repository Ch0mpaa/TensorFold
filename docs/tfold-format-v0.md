# `.tfold` v0.1 — Model file format

**Status:** Stage 1 of 8. Specification only. Writer, reader, executor, Zig loader, converter, inference, mobile runtime, and dyno integration are separate stages.

**Design principles**

1. **Fast is useful. Fast and wrong is not.** Every field in the file enables verification of what the file says it contains. If the file cannot prove it, the file is not qualified for serving.
2. **Copy what worked from safetensors and GGUF; drop what didn't.**
   - From safetensors: fixed-length JSON metadata block, byte-aligned tensor region, mmap-friendly layout, one file is one blob.
   - From GGUF: versioned header, typed quant conventions, alignment discipline, up-front tensor count, backward-compatibility strategy.
   - **Dropped:** safetensors' opaque metadata (every consumer rolls their own); GGUF's embedded tokenizer monolith and variable-length KV type encoding.
3. **Identity before content.** The source checkpoint SHA-256 is pinned in the header. A `.tfold` file whose source identity does not match the one the consumer pinned MUST fail closed. No silent rebinds.
4. **Zero code in the file.** The format carries data, not executables, not deserializable Python pickles, not JIT bytecode.
5. **Little-endian everywhere.** No host-endian variants. ARM, x86, and WASM targets all read the same bytes.

---

## 1. File layout

```
┌─────────────────────────────────────────────────────────────┐
│  [ 0..  8)   magic         8 bytes   b"TFOLD\x00\x00\x00"   │
│  [ 8.. 12)   version       u32 LE    0x00010000  (v0.1)     │
│  [12.. 16)   header_length u32 LE    size of header JSON    │
│  [16..16+N)  header JSON   UTF-8     exactly N bytes        │
│  [align 64)  padding       zero-fill until 64-byte boundary │
│  [tensors]   tensor region see § 3 for layout               │
│  [trailing]  integrity     8 bytes   FNV-1a-64 (see § 4)    │
└─────────────────────────────────────────────────────────────┘
```

Total file size:
```
16 + header_length + pad_to_64 + sum(tensor.nbytes + pad_to_64) + 8
```

No trailing bytes beyond the 8-byte FNV-1a-64 integrity trailer. Readers that encounter extra bytes MUST reject the file.

### 1.1 Magic bytes

Exactly 8 bytes: `0x54 0x46 0x4F 0x4C 0x44 0x00 0x00 0x00` — ASCII `TFOLD` followed by three null bytes. Fixed width keeps the subsequent version field 4-byte aligned.

### 1.2 Version

`u32 LE`. Major-minor packed as `(major << 16) | minor`. v0.1 is `0x00010000`. v1.0 will be `0x00100000`. Readers MUST reject any version whose major number they do not implement. Minor-version additions are backward-compatible.

### 1.3 Header length

`u32 LE`. Exact byte count of the header JSON. Does not include trailing padding. A reader MUST allocate exactly this many bytes to parse the header; it MUST NOT read past the header_length looking for extra JSON.

### 1.4 Header JSON

UTF-8 encoded JSON, no BOM, no trailing newline, no trailing null terminator. Shape in § 2.

### 1.5 Tensor region alignment

The tensor region begins at the next 64-byte boundary after `(16 + header_length)`. Padding bytes MUST be zero. 64 bytes is chosen to cover the alignment needs of AVX-512 vector loads and NEON scalable loads with one rule.

Every individual tensor within the region MUST also begin at a 64-byte boundary. Writers MUST insert zero padding; readers MUST assume the padding is zero but MUST NOT validate it (writers may use `MAP_UNINITIALIZED` or similar on some platforms).

### 1.6 Integrity trailer

The final 8 bytes of the file are the FNV-1a-64 digest of every byte from offset 0 up to (but not including) the trailer itself. Writers MUST compute and write the trailer last. Readers MUST verify it before using any tensor data.

Algorithm (reference C):
```c
uint64_t fnv1a_64(const uint8_t *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
```

---

## 2. Header JSON schema

See `schemas/tfold-v0.json` for the authoritative JSON Schema (draft 2020-12). The sections below describe the semantic contract.

### 2.1 Top-level shape

```json
{
  "tfold":      { ... },
  "model":      { ... },
  "tokenizer":  { ... },
  "quant":      { ... },
  "tensors":    [ { ... }, ... ]
}
```

All five top-level keys are REQUIRED.

### 2.2 `tfold`

```json
{
  "version": "0.1",
  "writer":  "tfold-py@0.1.0"
}
```

- `version` (string, required) — matches the binary version field, in `"MAJOR.MINOR"` form
- `writer` (string, required) — name + semver of the tool that wrote the file; purely informational, never used for dispatch

### 2.3 `model`

```json
{
  "name":           "Qwen3.5-2B",
  "arch":           "qwen3",
  "parameters":     2140000000,
  "vocab_size":     151936,
  "context_length": 32768,
  "source_sha256":  "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f",
  "source_url":     "turboderp/Qwen3.5-2B-exl3@8.00bpw"
}
```

- `name` (string, required) — display name, arbitrary
- `arch` (string, required) — canonical architecture ID. Initial registry: `qwen3`, `qwen3_moe`, `llama`, `glm`, `gemma3`. Readers MUST reject unknown archs; writers MUST NOT invent new values without a registry PR.
- `parameters` (u64, required) — total parameter count including embedding and head. Verification: writer MUST assert this matches the actual tensor accounting.
- `vocab_size` (u32, required)
- `context_length` (u32, required)
- `source_sha256` (hex string, required, exactly 64 chars) — SHA-256 of the source checkpoint (safetensors shard concatenation in on-disk order, or single-file). Used by the loader to refuse bound-to-wrong-checkpoint files.
- `source_url` (string, required) — HuggingFace repo@revision or arbitrary URL. Informational.

### 2.4 `tokenizer`

```json
{
  "type":   "tiktoken",
  "sha256": "9c3a47dafba6..."
}
```

- `type` (string, required) — initial registry: `tiktoken`, `sentencepiece`, `bpe`, `unigram`
- `sha256` (hex string, required, 64 chars) — SHA-256 of the external tokenizer file. The tokenizer bytes are NOT embedded in `.tfold`; the file lives alongside the model (conventionally as `<basename>.tokenizer.json` or similar) and the reader MUST verify its digest before use.

**Rationale for external tokenizer:** GGUF's embedded tokenizer creates a 1-MB-to-10-MB monolith that is nearly always identical across files from the same model family. External + SHA-verified keeps `.tfold` small, lets vendors ship one tokenizer per family, and makes updates a file swap rather than a model reconvert.

### 2.5 `quant`

```json
{
  "scheme":          "exl3",
  "bits_per_weight": 8.0,
  "block_size":      128,
  "codebook_sha256": "f6e2a1...",
  "codebook_offset": 123456,
  "codebook_nbytes": 65536
}
```

- `scheme` (string, required) — initial registry: `fp16`, `bf16`, `fp32`, `exl3`, `exl3_mcg`, `gguf_q4_0`, `gguf_q4_k_m`, `gguf_q8_0`. Readers MUST reject unknown schemes.
- `bits_per_weight` (number, required) — effective BPW including scales and codebooks; writer MUST compute from actual tensor accounting.
- `block_size` (u32, required) — quant block size in elements. `0` means "not blocked" (for fp16/bf16/fp32).
- `codebook_sha256`, `codebook_offset`, `codebook_nbytes` (required for schemes that use a codebook, forbidden otherwise) — the codebook is stored as a reserved tensor named `__codebook__` in the tensor region (see § 3.3), these fields point into that region by byte offset within the file.

**Rationale for embedded codebook:** codebooks are tied 1:1 to the quant scheme and small (≤64 KB typical); embedding matches GGUF's single-file philosophy for the pieces that really belong together.

### 2.6 `tensors`

Array of tensor descriptors, in file order. Readers MUST assume order matches the on-disk tensor region order.

```json
[
  {
    "name":   "model.language_model.embed_tokens.weight",
    "dtype":  "fp16",
    "shape":  [151936, 2048],
    "offset": 0,
    "nbytes": 622329856,
    "sha256": "4ad47947dfa769a9d7eb2754732e855775bb3df70e66902bb2ee4429d27dc776"
  },
  ...
]
```

- `name` (string, required) — model-canonical dotted name; writers MUST NOT rename
- `dtype` (string, required) — initial registry: `fp16`, `bf16`, `fp32`, `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `exl3_tile`, `exl3_mcg_tile`. Quant-specific dtypes encode the pack layout.
- `shape` (array of u32, required, length 1..=8) — logical shape, not pack shape. For `exl3_tile` and other packed dtypes the on-disk byte layout is scheme-defined and may not be simply `product(shape) * element_size`.
- `offset` (u64, required) — byte offset from the start of the FILE (not the tensor region) where this tensor's bytes begin. MUST be 64-byte aligned.
- `nbytes` (u64, required) — exact byte count of the tensor blob, excluding trailing padding
- `sha256` (hex string, required, 64 chars) — SHA-256 of exactly the `nbytes` bytes of the tensor. Readers MAY defer verification until the tensor is actually touched, but SHOULD verify at least on first read.

---

## 3. Tensor region

### 3.1 Layout

Tensors are laid out back-to-back, each starting at a 64-byte boundary. The gap between `(offset_i + nbytes_i)` and `offset_{i+1}` is zero padding.

### 3.2 Sort order

Writers SHOULD sort tensors in approximate reverse size order (largest first) to minimize fragmentation under mmap, but readers MUST NOT rely on this.

### 3.3 Reserved names

- `__codebook__` — quant codebook, referenced by the `quant` block. Absent if `quant.scheme` does not require one.

Any other name starting with `__` is reserved for future format use. Readers encountering an unknown `__` name MUST reject the file.

---

## 4. Integrity, identity, and validation

A reader MUST perform the following checks in order on open. Failure at any step is a fatal fail-closed reject. There are no warnings, no fallbacks, no "best effort" paths.

| # | Check | Reject code |
|---|---|---|
| 1 | File is at least 32 bytes (magic + version + header_length + minimum header + trailer) | `TFOLD_TOO_SMALL` |
| 2 | Magic bytes match `TFOLD\0\0\0` exactly | `TFOLD_BAD_MAGIC` |
| 3 | Major version is one this reader implements | `TFOLD_VERSION_UNSUPPORTED` |
| 4 | Header length fits within file size minus trailer minus header position | `TFOLD_HEADER_OVERFLOW` |
| 5 | Header JSON parses | `TFOLD_HEADER_PARSE` |
| 6 | Header schema validates against `tfold-v0.json` | `TFOLD_HEADER_SCHEMA` |
| 7 | FNV-1a-64 trailer matches | `TFOLD_INTEGRITY` |
| 8 | Every tensor `offset + nbytes` is within the file bounds | `TFOLD_TENSOR_OOB` |
| 9 | Every tensor `offset` is 64-byte aligned | `TFOLD_TENSOR_ALIGN` |
| 10 | Tensor offsets, when sorted, are monotonic and non-overlapping | `TFOLD_TENSOR_OVERLAP` |
| 11 | No tensor name is duplicated | `TFOLD_TENSOR_DUP_NAME` |
| 12 | `model.arch`, `quant.scheme`, and every `tensors[].dtype` are in the registries this reader implements | `TFOLD_UNKNOWN_<kind>` |
| 13 | If the caller pinned a `source_sha256`, header's `model.source_sha256` matches it exactly | `TFOLD_CHECKPOINT_MISMATCH` |

Per-tensor SHA-256 checks (step 14) MAY be deferred to first access of the tensor but SHOULD run on first open for safety-critical callers. The check failure code is `TFOLD_TENSOR_INTEGRITY`.

---

## 5. Backward and forward compatibility

### 5.1 Minor-version additions (v0.1 → v0.2 etc.)

A v0.2 writer MAY add new OPTIONAL keys to the header. A v0.1 reader encountering unknown optional keys MUST either tolerate them (preferred) or reject the file explicitly — but MUST NOT silently misinterpret data. New REQUIRED keys bump the major version.

### 5.2 Major-version jumps

v1.0 and beyond MUST use a new magic byte sequence (e.g. `TFOLD\x01\x00\x00`). A reader encountering a magic it does not recognize MUST reject with `TFOLD_BAD_MAGIC`. There is no silent upgrade path.

### 5.3 Registry additions

Architecture, dtype, quant scheme, tokenizer type: every addition requires a PR to the central registry (`docs/tfold-registry.md`, to be added) and increments the minor version. Readers released before the addition MUST reject unknown values rather than guess.

---

## 6. What this format does NOT do

Being explicit about scope so nobody builds in these directions by accident.

- **No code execution.** The file does not carry executable bytes, serialized Python pickles, JIT IR, ONNX graph ops, or callable layers. An inference engine chooses how to execute based on `model.arch`; the file describes weights only.
- **No generation state.** No sampler, no stop tokens, no chat template, no prompt format. Those live in companion files (`<basename>.chat.jinja`, etc.) or in the inference engine's registry.
- **No multiple-tensor ordering semantics.** The on-disk order is for storage efficiency; execution order is defined by the model architecture, not the file.
- **No partial-file semantics.** There is no "chunked `.tfold`" or "sharded `.tfold`" in v0.1. A model is one file. Multi-file sharding is a v0.2+ decision pending real need.
- **No encryption or signing.** v0.1 is integrity-checked but not authenticated. Signed `.tfold` is a v0.2+ discussion.

---

## 7. Open design choices tabled for v0.2

Documented here so that v0.1 doesn't accidentally foreclose them:

- **Sharded files.** A `tfold.shard_index` block in the header + `*.tfold.001`, `*.tfold.002` siblings.
- **Mixed-precision per tensor.** v0.1 permits a single `quant.scheme` but each tensor carries its own `dtype`. A v0.2 revision may let the `quant` block be per-tensor rather than per-file.
- **Attestation.** Ed25519 signature over the header + tensor hashes.
- **Per-tensor metadata.** Arbitrary KV per tensor (e.g. `scale`, `zero_point`) for quant schemes that need them.
- **Compression.** The format assumes bytes on disk equal bytes in memory. Compressed `.tfold` is not planned but not foreclosed.

---

## 8. Reference implementations (stages 2+)

| Stage | Deliverable | Status |
|---|---|---|
| 1 | This document + `schemas/tfold-v0.json` | **this PR** |
| 2 | `tools/tfold_py/` writer + reader, round-trip tests | pending |
| 3 | `tools/tfold_py/execute.py` CPU matmul executor | pending |
| 4 | `zig/src/tfold/` native Zig loader | pending |
| 5 | `tools/tfold_py/convert_exl3.py` converter | pending |
| 6 | Zig autoregressive inference against `.tfold` | pending |
| 7 | Android APK consuming `.tfold` | pending |
| 8 | Dyno + QNN acceleration comparison | pending |

Each stage ships its own article in the Chop Shop series, with a reproducible commit and a result-or-failure receipt.

---

## 9. Why this series of design calls

| Design call | Rationale |
|---|---|
| Little-endian only | ARM, x86, WASM all converge on LE for model weights; big-endian adds code without serving any shipping platform |
| 8-byte magic | Keeps the version field naturally 4-byte aligned; mirrors safetensors' leading `u64` |
| UTF-8 JSON header | Human-readable debugging, battle-tested schema validation, no custom parser; cost is larger header vs. binary framing but headers are tiny vs. model weights |
| 64-byte alignment | Covers AVX-512 and ARM SVE vector requirements in one rule; small padding cost vs multi-rule complexity |
| FNV-1a-64 trailer | Fast (single CPU instruction per byte), no dependency, catches silent disk corruption; we are not defending against malicious tampering (that is v0.2 Ed25519) |
| External tokenizer + SHA | Shared across model files in a family, independently updatable, doesn't bloat every model file; GGUF's embedded tokenizer was a size-efficiency mistake |
| Embedded codebook | Codebook is 1:1 with quant scheme and small; embedding matches the "atomic file" model for pieces that genuinely belong together |
| `source_sha256` pinning | The QNN qualification work established that checkpoint-pinning is non-negotiable; `.tfold` carries that discipline forward from day one |
| Reject-unknown everywhere | The guiding principle: fast and wrong is not useful. Rejecting beats silently-misinterpreting at every single layer. |

---

**Spec version:** `0.1`
**Spec sha256:** (computed on commit)
**Schema file:** `schemas/tfold-v0.json`
