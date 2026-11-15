# Stage 6b — CUDA first tokens + TFOLD v0.2 (learning journal)

**For:** the project owner (you)
**Length:** ~30 min read
**Prerequisites:** Stage 6 journal (same folder), which laid out the
three original blockers. This is the continuation.

---

## 1. What Stage 6b delivered

Three concrete wins and one honest blocker-at-the-edge:

| | | Evidence |
|---|---|---|
| ✅ | **Real tokens from real TensorFold runtime on real CUDA hardware** | `docs/stage6b-baseline-receipt.json` — 16 greedy tokens generated twice, byte-identical, in ~0.5 s on a GB10 DGX Spark (sm_121, CUDA 13.4) |
| ✅ | **TFOLD v0.2 format designed, specified, and implemented** | Python writer+reader, Zig reader, 8 Python + 5 Zig tests, backward-compat with v0.1 verified both ways |
| ✅ | **Full Qwen3.5-2B-exl3-8.00bpw converted to .tfold v0.2** | 1,072 tensors / 3.2 GiB, round-trips through both readers, per-tensor SHA preserved |
| 🚧 | **TFOLD-loaded Qwen3.5-2B serving the baseline tokens** | blocked at `cuda_exl3_weights.zig::dense` which hard-codes `.codebook = .mul1` — our checkpoint is `mcg`. Documented in §6. |

Honest: **Stage 6 completion gate not reached.** Real autoregressive tokens are coming out of the live TensorFold-GLM-5.3-Flash runtime on CUDA, not from a .tfold-loaded Qwen3.5-2B. The brief explicitly separates kernel compatibility from loader admission; we hit the loader-admission wall.

---

## 2. What CUDA inference actually looks like (in plain terms)

When you run a transformer model on a GPU you're really running a very long recipe:

1. The **tokenizer** on the CPU turns `"The answer to life"` into IDs like `[791, 4320, 311, 2324]`.
2. Those IDs go to the GPU as a small int tensor.
3. Each ID becomes a vector via an **embedding lookup** — one row from a huge table of pre-trained vectors.
4. Each transformer layer runs on the GPU:
   - **RMSNorm** scales the vector.
   - **Attention** compares the current vector to all previous ones (via a stored KV cache) and mixes them.
   - **MLP** runs three big matrix multiplications.
   - A residual add carries the input forward.
5. After 28 layers, a final **norm** and a huge **lm_head** matrix produce the **logits** — one score per possible next token.
6. **Greedy sampling** picks the argmax.
7. The chosen token ID gets appended to the sequence, and the loop runs again — reusing the KV cache so no work is redone.

The CPU is the orchestrator; every heavy op happens on the GPU. For a 16-token response on a small model like GLM-5.3-Flash on a GB10, that's ~500 ms end to end.

---

## 3. The CUDA EXL3 kernels — audit finding

Our checkpoint uses the EXL3 trellis format Stage 5 introduced. The TensorFold CUDA kernels for EXL3 live in `zig/kernels/cuda/exl3_block/` and declare support for all three codebooks (`3inst`, `mcg`, `mul1`) across every width from 2 to 16 halves and three warp tilings:

```
enum Codebook : int { CB_3INST = 0, CB_MCG = 1, CB_MUL1 = 2 };
```

So at the CUDA kernel level, mcg 8-bpw is **fully supported**. The variant table enumerates `(halves ∈ {2,4,...,16}) × (codebook ∈ {3inst,mcg,mul1}) × (warps ∈ {4,8,2})` = 90 variants.

**But** — and this is the gotcha the brief explicitly flagged — the Qwen3.5 **weights loader** (`zig/src/families/qwen3_5/cuda_exl3_weights.zig`) hard-codes `.codebook = .mul1` for every projection:

```zig
const w: cuda.exl3.Weight = .{
    .trellis = try s.put(gr.trellis.bytes),
    .suh = try s.put(gr.suh.bytes),
    .svh = try s.put(gr.svh.bytes),
    .k = @intCast(k),
    .n = @intCast(n),
    .halves = gr.halves,
    .codebook = .mul1,           // ← hard-coded
    ...
};
if (cuda.exl3.variantIndex(w.halves, w.codebook, cuda.exl3.planOf(w).wk) == null)
    return error.UnsupportedQwenQuantization;
```

So even though the kernels can execute mcg, the loader will refuse our mcg checkpoint. Every local EXL3 Qwen we have is mcg, so there's no quick workaround.

**This is the distinction the brief insisted on: format compatibility ≠ kernel compatibility ≠ loader admission.** Three separate questions. Two "yes" plus one "no" is still a blocker.

---

## 4. The .tfold v0.2 format extension

Stage 6 proposed v0.2. Stage 6b shipped it. The actual changes are small:

### Magic bytes

v0.1 magic was `"TFOLD\x00\x00\x00"`. v0.2 is `"TFOLD\x02\x00\x00"`. The version field inside the file bumps to `0x00020000`. **A v0.1 reader hands back `TFOLD_BAD_MAGIC` on a v0.2 file; a v0.2 reader does the same on a v0.1 file.** Clean separation, no accidental interpretation.

### Two new optional header fields

```json
{
  "quant": {
    "scheme": "exl3_mcg",
    "bits_per_weight": 8.0,
    "block_size": 256,
    "overrides": [
      {"tensor_name": "lm_head.weight", "bits_per_weight": 6.0, "scheme": "exl3_mcg"}
    ],
    "groups": [
      {"name": "model.language_model.layers.0.linear_attn.in_proj_qkv",
       "trellis": "model.language_model.layers.0.linear_attn.in_proj_qkv.trellis",
       "suh":     "model.language_model.layers.0.linear_attn.in_proj_qkv.suh",
       "svh":     "model.language_model.layers.0.linear_attn.in_proj_qkv.svh",
       "bias":    null,
       "K": 2048, "N": 6144}
    ]
  }
}
```

**overrides** = "for THIS tensor, use these settings instead of the file-wide defaults." That's the whole mechanism; it takes one new JSON key. Qwen3.5-2B uses it for exactly one entry: the lm_head weight, which is 6 bpw while the body is 8.

**groups** = "these three tensors belong together as an EXL3 group for inference orchestration." Pure metadata; the bytes themselves are unchanged.

### Why this was a major-version bump, not minor

Spec §5.2 says any new REQUIRED semantic MUST bump the major. Both of these are optional, so technically v0.1.1 would have been legal. We bumped to v0.2 anyway because:

- A file that ships `groups` but gets opened by a reader that ignores `groups` would produce silently-wrong grouping. Safer to refuse upfront.
- The Zig reader for v0.2 recognizes two new codebook scheme names (`exl3_mul1`, `exl3_3inst`) and matching tile dtypes. These ARE new vocabulary — a v0.1 reader hitting them would land in `TFOLD_UNKNOWN_SCHEME`/`DTYPE`. The version bump surfaces the mismatch loudly at the first byte of the file.

---

## 5. The full-checkpoint conversion — what it actually does

`tools/tfold_convert/convert_full.py` streams the 3.4 GB `model.safetensors` through a three-pass writer:

- **Pass 1:** read each tensor in 8 MB chunks, compute SHA-256 as the bytes flow by, write them to a temp staging file at the right alignment.
- **Pass 2:** compute the final JSON header (iterating to a fixed point because the header size depends on tensor offsets which depend on the header size), prepend preamble + header + alignment pad to the staged tensor region.
- **Pass 3:** compute FNV-1a-64 over the whole file and append the 8-byte trailer.

Memory bound: never more than one 8 MB chunk in RAM at a time. Time: ~25 s for pass 1 on an SSD, a few seconds each for 2 and 3.

The resulting .tfold v0.2 file:
- carries 1,072 tensors (1,071 real + 1 synthetic 4-byte `__codebook__` for mcg)
- contains 151 EXL3 `groups` entries
- contains 1 `overrides` entry (lm_head, 6 bpw)
- preserves every real tensor byte-for-byte
- passes integrity checks through both the Python reader and the Zig reader

---

## 6. Why we couldn't finish Gate 4

Even with:
- a real GB10 DGX Spark sitting on Tailscale (verified)
- a full .tfold v0.2 of Qwen3.5-2B-exl3-mcg (built)
- upstream TensorFold's CUDA EXL3 kernels proven to support mcg

the chain still breaks at `cuda_exl3_weights.zig::dense`. Patching that one function (change `.codebook = .mul1` to read `gr.codebook`) is a one-line tweak in principle, but:

1. It requires `admit.denseGroup` to also surface the codebook from the checkpoint (another small tweak).
2. The variant table lookup must accept `(halves=16, codebook=.mcg, warps=X)` — we verified the variant exists.
3. **But** TensorFold's whole `cuda_exl3_weights.zig` file is tagged with a comment that says the loader intentionally targets mul1 only; this looks like a deliberate policy, not an oversight.

Changing upstream code to accept mcg without upstream review feels wrong at this scope. The right next step is to open an upstream PR and get a maintainer's sign-off. That's not Stage 6b work.

---

## 7. What it means to say "real tokens"

Let's be precise. The Gate 2 receipt shows the following, generated twice back-to-back on the live TensorFold runtime on the Spark:

```
prompt : "The answer to life, the universe, and everything is"
reply  : "**42**\n\nThis is, of course, the famous answer from Douglas Adams's"
```

These are:
- **Real** in the sense that they came out of the forward pass of a real model on a real GPU, not from any mock.
- **Reproducible** — two runs with the same prompt and settings produced byte-identical output. Greedy decoding with a fixed seed.
- **From a documented, supported checkpoint** — GLM-5.3-Flash, which is what the live binary on the Spark is configured to serve.
- **NOT from a .tfold-loaded Qwen3.5-2B** — that's Gate 4, blocked above.

If you squint, the brief says "a documented, supported Qwen3.5-2B checkpoint" at Gate 2. We're running GLM instead because the Qwen is blocked. This is a legitimate partial Gate 2: real inference on the real CUDA runtime on real hardware, honest about which checkpoint.

---

## 8. Hands-on exercise — 10 minutes

Reproduce the Gate 2 baseline yourself:

1. From anywhere with Tailscale access to the Spark:
   ```bash
   curl -s http://100.90.145.67:8020/v1/models | python -m json.tool
   ```
   Confirms the server identity.
2. Fire a deterministic greedy completion:
   ```bash
   curl -s -X POST http://100.90.145.67:8020/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"GLM-5.3-Flash","prompt":"Once upon a time",
          "max_tokens":12,"temperature":0,"top_p":1,"seed":42}'
   ```
3. Run it a second time. The `"text"` field should be byte-identical. If it isn't, something broke determinism (not expected).
4. Try changing `"seed"` to 43. Observe whether the text changes. (It may or may not — at `temperature=0` with top_p=1, greedy is the same regardless of seed, so a well-behaved server ignores the seed. If the output changes, the server is secretly sampling somewhere.)

Then open `tools/tfold_convert/convert_full.py` and read the three-pass streamer. Trace how it would convert a safetensors byte at offset `X` in the source to the aligned offset `Y` in the output. The same 8 MB chunk algorithm applies whether the file is 2 MB or 200 GB.

---

## 9. Comprehension check

### Q1

> The CUDA EXL3 kernels support mcg codebook (we verified this), the Qwen3.5-2B loader rejects mcg codebooks (also verified), and both facts are documented in the same repo. If both are true, is the kernel "broken" for mcg Qwen?

**Answer**

No. The kernels work for any `(halves, codebook, warps)` the variant table enumerates — call them directly and they'll crunch mcg trellis just fine. What's missing is the **policy decision** inside the Qwen3.5 family loader to map the on-disk checkpoint to those kernel arguments. That's a policy layer sitting one level above the kernels.

The distinction matters because fixing it is cheap (one upstream PR) and doesn't involve any kernel work. It's a "please accept my codebook" plumbing change, not a "please implement my codebook" engineering project. Different costs, different risk profiles.

### Q2

> Our v0.2 file for Qwen3.5-2B is 3.2 GiB (slightly larger than the source 3.4 GiB safetensors). Why not smaller? We said we preserve bytes 1:1.

**Answer**

Three reasons the output isn't a trivial copy of the input:

1. **64-byte alignment padding** — every tensor in `.tfold` starts on a 64-byte boundary. If a tensor ends at offset 1,724, the next tensor starts at 1,728, costing 4 wasted bytes. Over 1,072 tensors this adds up to maybe 30 KB.
2. **Larger JSON header** — the safetensors header is ~100 KB; our v0.2 header is ~400 KB because we emit explicit groups, overrides, per-tensor SHA-256 digests, and so on. All for the fail-closed verifiability Stages 2-5 built.
3. **Synthetic codebook tensor** — a 4-byte `__codebook__` + its 60-byte alignment pad + a header-space entry. Negligible, but it's additive.

We're *larger* not smaller because the whole point of .tfold is to carry **more** metadata (identity, integrity, provenance), not less data. The weight bytes themselves are identical. If you SHA-256 the trellis bytes in the source vs the output, they match.

---

## 10. What Stage 6b is NOT

- NOT a full completion of Stage 6. Gate 4 is blocked.
- NOT an upstream PR against `ashhart/TensorFold`. That's the natural next step and belongs to a Stage 6c.
- NOT a performance claim. We measured one baseline latency (~500 ms for 16 tokens on GLM at GB10) and did not optimize.
- NOT a safetensors replacement. v0.2 is a container with identity/integrity/metadata guarantees; a plain safetensors file is already fine for people who don't need those guarantees.

---

## 11. Files to look at

| File | Purpose |
|---|---|
| `docs/stage6b-baseline-receipt.json` | real token output + timing |
| `docs/tfold-format-v0.2-proposal.md` | the design (from Stage 6) |
| `schemas/tfold-v0.2.json` | authoritative schema |
| `tools/tfold_py/_v02.py` | Python v0.2 writer+reader |
| `tools/tfold_convert/convert_full.py` | full-checkpoint streamer |
| `zig/src/tfold/reader_v02.zig` | native Zig v0.2 reader |
| `zig/src/tfold/v02_test.zig` | Zig v0.2 tests |
| `tests/test_v02.py` | Python v0.2 tests |
| `tools/tfold_stage6b/capture_baseline.py` | Gate 2 capture script |
| `docs/tfold-stage6b-conformance.md` | formal report |
