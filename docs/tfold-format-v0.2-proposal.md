# `.tfold` v0.2 extension proposal

**Status:** DRAFT. Not implemented in Stage 6 code. Documents the
minimum header-only change required to carry a full real EXL3
checkpoint in one file.

**Motivation:** Stage 5 proved v0.1 can carry one EXL3 group. Stage 6
analyzed the real Qwen3.5-2B-exl3-8.00bpw checkpoint and confirmed it
holds **267 tensor groups** (151 EXL3 + 116 plain), uniformly mcg
codebook except for the head which uses 6 bpw instead of 8.0 bpw.
Representing this in v0.1 would require 267 separate files, which
defeats the single-file packaging goal.

**Backward compatibility:** every v0.1 file remains valid v0.2. v0.2
adds OPTIONAL keys; readers that don't understand them should ignore
them or reject explicitly (spec §5.1). This proposal introduces zero
required keys.

---

## 1. Binary layout — UNCHANGED

- Magic bytes, version field, header-length prefix, 64-byte alignment,
  FNV-1a-64 trailer are identical to v0.1.
- Version u32 bumps minor only: `0x0001_0001` → v0.1.1 for the
  schema-only tightening already in Stage 2; a true v0.2 semantic
  bump would use `0x0002_0000` with a new magic `TFOLD\x02\x00\x00`
  per spec §5.2. **For this proposal we recommend a true major bump**
  so readers that implement only v0.1 cannot accidentally mis-interpret
  a v0.2 file.

## 2. Header schema additions — OPTIONAL fields

### 2.1 `quant.overrides` (NEW, optional)

An array of per-tensor override records. If present, each record
replaces the file-wide `quant.scheme`, `quant.bits_per_weight`, and
`quant.block_size` for exactly the named tensor.

```json
{
  "quant": {
    "scheme": "exl3_mcg",
    "bits_per_weight": 8.0,
    "block_size": 256,
    "codebook_sha256": "...",
    "codebook_offset": ...,
    "codebook_nbytes": ...,
    "overrides": [
      {
        "tensor_name": "lm_head.weight",
        "scheme": "exl3_mcg",
        "bits_per_weight": 6.0,
        "block_size": 192
      }
    ]
  }
}
```

### 2.2 `quant.groups` (NEW, optional)

Explicit group structure for EXL3 serving. Each group names its member
tensors by exact name. If absent, readers fall back to the v0.1
single-group convention (one `trellis` + `suh` + `svh` + optional
`bias`).

```json
"groups": [
  {
    "name": "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "trellis": "model.language_model.layers.0.linear_attn.in_proj_qkv.trellis",
    "suh":     "model.language_model.layers.0.linear_attn.in_proj_qkv.suh",
    "svh":     "model.language_model.layers.0.linear_attn.in_proj_qkv.svh",
    "bias":    null,
    "K": 2048,
    "N": 6144
  },
  ...
]
```

Rationale: the alternative — inferring groups from dotted-name
substrings — is fragile (what if a tensor name contains "trellis"?).
Explicit grouping is boring and correct.

### 2.3 `model.tensor_count` (NEW, optional)

A u64 giving the exact tensor count. Already derivable from
`tensors.length`, but exposing it lets readers pre-allocate without
scanning. Pure optimization; zero semantic value.

## 3. Writer behavior

- v0.2 writers MAY emit `overrides` and `groups` when they carry
  heterogeneous quant settings. v0.2 writers producing a
  homogeneous file (one scheme, one width, one trellis) SHOULD omit
  both keys to maximize v0.1-reader compatibility.

## 4. Reader behavior

- v0.2 readers that encounter a file without `overrides` MUST treat
  every tensor as using the file-wide `quant` settings — identical to
  v0.1 behavior.
- v0.2 readers that encounter `overrides` MUST apply them when
  decoding the named tensors.
- v0.2 readers that encounter `groups` SHOULD use that structure for
  EXL3 decoding orchestration; a reader that doesn't support
  `groups` MUST reject the file explicitly rather than guess.

## 5. What this proposal does NOT add

- No new dtypes.
- No new codebook schemes.
- No new file-structure elements (no sharding, no streaming, no
  compression).
- No encryption or authentication (deferred to a separate `v0.3
  attestation` proposal if ever needed).

## 6. Representability check against Qwen3.5-2B-exl3-8.00bpw

With this proposal:

| Thing | v0.1 | v0.2 proposal |
|---|---|---|
| 151 EXL3 groups + 116 plain tensors in one file | ❌ (would need 267 files) | ✅ |
| Uniform mcg body + 6-bpw head | ❌ | ✅ (one `overrides` entry for the head) |
| Mixed widths within body | ❌ | ✅ (one `overrides` entry per differing tensor) |
| Routed experts with per-expert codebook | ❌ | partial — scheme can differ per tensor; codebook table growth would need v0.3 |

The proposal is the **minimum** change that unblocks Stage 6 real-model
loading. Anything smaller forces multi-file hacks; anything larger
locks in commitments we shouldn't make before serving data proves them.

## 7. Implementation estimate

| Change | Effort |
|---|---|
| Schema JSON update + v0.2 example | 1 hr |
| Python writer: emit overrides/groups | 1 hr |
| Python reader: apply overrides, validate groups | 2 hr |
| Zig reader: same | 3 hr |
| Full Qwen3.5-2B conversion run | 2 hr (dominated by SHA hashing 2.7 GB) |
| Updated conformance tests | 2 hr |

Total: ~1 working day. Blocking Stage 6 real inference requires this
land first.

## 8. Decision

**Not implementing in Stage 6.** Stage 6's brief says "implement a
minimal backward-compatible v0.2 extension only if required." It IS
required for real-model inference, but inference on my dev hardware
is independently blocked by TensorFold's Metal-only Qwen3.5 forward
path (`zig/src/families/qwen3_5/backend.zig`), which I cannot execute
on Windows or Linux. Shipping v0.2 code now would be a partial-
infrastructure build with no way to validate it end-to-end.

The Stage 6 report (`docs/tfold-stage6-conformance.md`) carries this
decision and the full gap analysis.
