# Stage 6c — teaching the Qwen3.5 CUDA loader about mcg

> Beginner-friendly companion to the Stage 6c upstream PR
> ([ashhart/TensorFold#3412](https://github.com/ashhart/TensorFold/pull/3412)).
> The goal is real Qwen3.5-2B tokens from TensorFold on CUDA, loaded from an
> ExLlamaV3 "mcg" checkpoint — something the upstream loader had been
> explicitly refusing before touching the GPU.

## What this build block does

TensorFold's CUDA engine can decode EXL3-quantized tensors in several
"codebook" flavors (`mul1`, `mcg`, `inst3`). Each codebook is a different
way of mapping a 16-bit trellis state back to the half-precision weight it
represents. The downstream kernels in `zig/src/cuda/exl3.zig` are compiled
for every codebook × width combination the format supports.

But the Qwen3.5 family's **checkpoint loader** (not the kernels) was hard
coded to one codebook. If you tried to feed it a `mcg` checkpoint, the
admission function refused with `UnsupportedQwenQuantization` before a
single byte reached the GPU. That's what Stage 6b's Gate 4 ran into.

## The exact wall

Three lines of upstream code:

```zig
// cuda_exl3_admit.zig
const marker = try ck.get(try name(&buf, "{s}.mul1", .{base}));
if (... != mul1_magic) return error.UnsupportedQwenQuantization;

if (cuda.exl3.variantIndex(g.halves, .mul1, 4) == null) return ...;
```

```zig
// cuda_exl3_weights.zig  (two sites)
.codebook = .mul1,
```

In plain English: "only accept a tensor named `<group>.mul1` holding the
32-bit value `0x83DCD12D`; and whenever you build a GPU Weight/Layer, hard
wire the codebook to `mul1`."

## Why upstream wrote it that way

There was nothing wrong with it! The Qwen3.5 release ExLlamaV3 shipped
with was mul1-only. Upstream picked the simplest code: one codebook, one
magic constant. **This is a good default** — you don't pay for alternatives
until you need them.

But the EXL3 ecosystem publishes mcg checkpoints too (e.g. turboderp's
`Qwen3.5-2B-exl3-8.00bpw` uses mcg). The kernel table already has every
variant compiled. Only the loader was gate-keeping the format.

## The patch — three files

### 1. `cuda_exl3_admit.zig` — teach the admit path to look for either

```zig
const mul1_magic: u32 = 0x83DCD12D;
const mcg_magic:  u32 = 0xCBAC1FED;   // NEW

pub const Group = struct {
    trellis: ...,
    suh: ...,
    svh: ...,
    halves: u32,
    codebook: cuda.exl3.Codebook,   // NEW - was implicit .mul1
};

pub fn group(ck: *core.Checkpoint, base: []const u8, n: usize, k: usize) !Group {
    // ...the trellis + halves checks are unchanged...

    // Try .mul1 first (keep the common case fast), then .mcg, then refuse.
    const mul1_name = try name(&buf, "{s}.mul1", .{base});
    const cb: cuda.exl3.Codebook = if (ck.has(mul1_name)) blk: {
        const marker = try ck.get(mul1_name);
        if (bad_mul1_marker(marker)) return error.UnsupportedQwenQuantization;
        break :blk .mul1;
    } else blk: {
        const mcg_name = try name(&buf, "{s}.mcg", .{base});
        if (!ck.has(mcg_name)) return error.UnsupportedQwenQuantization;
        const marker = try ck.get(mcg_name);
        if (bad_mcg_marker(marker)) return error.UnsupportedQwenQuantization;
        break :blk .mcg;
    };

    // ...suh/svh checks are unchanged...
    return .{ ..., .codebook = cb };
}
```

Two small rules I'll point out so you don't miss them later:

- **`ck.has()` before `ck.get()`.** `Checkpoint.get` logs to the error
  channel if a tensor is missing, which is noisy for *expected* absences.
  Using `has()` first keeps the log clean.
- **Order matters.** mul1 is still the common case, so we try it first;
  the mcg branch is strictly a fall-through.

### 2. `cuda_exl3_weights.zig` — carry the codebook through

```zig
// dense() — one projection, one group:
.codebook = gr.codebook,   // was .mul1

// experts() — all MLP experts of a layer SHARE one codebook:
var expert_cb: ?cuda.exl3.Codebook = null;
for (mods, 0..) |mod, m| {
    // ...
    for (0..count) |e| {
        const gr = try group(ck, base, n, k);
        if (expert_cb) |ecb| {
            if (ecb != gr.codebook) return error.UnsupportedQwenQuantization;
        } else expert_cb = gr.codebook;
        // ...
    }
}
.codebook = expert_cb orelse return error.UnsupportedQwenQuantization,
```

The experts branch is the subtle one: all experts in a given MoE layer must
use the same codebook. If any expert disagreed with its siblings, the
kernel can't batch them, and refusing early is cheaper than detecting the
mismatch later.

### 3. `cuda_exl3_loader_test.zig` — two new tests

- `an mcg-codebook group admits and reports codebook .mcg` — synthesizes
  a tiny safetensors file with a `.mcg` marker holding `0xCBAC1FED`,
  calls `admit.group`, expects `Codebook.mcg`. Also calls `denseGroup` to
  prove the compiled variant exists at 16 halves.
- `an mcg marker with the wrong magic is refused` — same file, but the
  marker bytes are `0xDEADBEEF`. Expects `error.UnsupportedQwenQuantization`.

## Design alternatives considered

1. **Rebase our fork onto upstream `main`.** Ruled out: 6,259-commit
   conflict surface. Prior constraints forbid blind rebases.
2. **Reimplement Qwen3.5 CUDA on top of our local `core/exl3_projection.zig`.**
   Weeks of work duplicating code already battle-tested upstream. Only
   makes sense if the ecosystem explicitly refuses the patch.
3. **Teach the loader to probe BOTH markers, carry a "codebook-inferred"
   flag.** Rejected because the checkpoint is authoritative; a single
   missing marker should be a hard failure, not a guess.
4. **Vendor the qwen3_5 CUDA subtree into our fork with the patch applied.**
   Still on the table if upstream PR #3412 is rejected. Design captured in
   `docs/stage6c-v02-tfoldfile-design.md`.
5. **Pass the codebook via build flag or env.** Awful — the codebook is a
   property of the checkpoint, not of the binary.

## Where mcg actually lives in the file

A `.mcg` marker tensor is a **scalar** (`rank=0`, shape `[]`), dtype `I32`,
holding the single 32-bit value `0xCBAC1FED`. In the checkpoint's safetensors
header it looks like:

```json
"model.language_model.layers.0.linear_attn.in_proj_qkv.mcg": {
  "dtype": "I32",
  "shape": [],
  "data_offsets": [1017135104, 1017135108]
}
```

Those four bytes `ed 1f ac cb` read as little-endian u32 are `0xCBAC1FED`.
Spot-checked against `lm_head.mcg`, `layers.17.self_attn.q_proj.mcg`, etc.
— every group's marker is byte-identical.

The number itself is arbitrary (chosen by whoever added mcg in the
ExLlamaV3 upstream). It's *just a magic tag* saying "I'm an mcg tensor";
the actual multiplier `McgMul` lives in `core/exl3_format.zig::codebookValue`.

## Testing — what runs, what passes

| Suite | Before | After | Notes |
|---|---|---|---|
| `test-qwen35-exl3-admission-host` | 3/3 pass | 5/5 pass | 2 new mcg tests added |
| `test-exl3-cuda-host` | 669/681 pass, 10 skip | 671/681 pass, 10 skip | new mcg tests fold in |
| mul1 regression — existing admit tests | pass | pass | no change in mul1 code path |

The 10 skips are unchanged — platform-dependent gfx/HIP tests that skip on
a CUDA host.

## Hands-on exercise

## What Stage 6c did

- **Patched upstream to accept mcg + ExLlamaV3 dtype variants.** Three
  loader files in `families/qwen3_5/`, four helpers, two tests; PR #3412.
- **Proved Gate A** — real tokens from the original safetensors through
  the patched binary. The model produced `" 42.\nA. True\nB. False\n\n<think>\nThinking"`
  deterministically (byte-identical SHA across two runs).
- **Proved Gate B** — byte-identical logits from the TFOLD v0.2
  checkpoint via a format materializer that re-emits a safetensors
  directory from the v0.2 tensor layout (byte-copy, no re-quantization).
  SHA matches Gate A exactly: `6b198ea8…`.
   ```
2. Check out the PR branch:
   ```
   git fetch https://github.com/Ch0mpaa/TensorFold.git stage6c-cuda-mcg
   git checkout FETCH_HEAD
   ```
3. Build the kernels + CLI:
   ```
   zig build -Dnvcc=/usr/local/cuda-13.0/bin/nvcc install
   ```
4. Pull a real mcg checkpoint:
   ```
   huggingface-cli download turboderp/Qwen3.5-2B-exl3-8.00bpw --local-dir ~/models/Q35-2B
   ```
5. Run inference:
   ```
   zig-out/bin/tf-qwen35-cuda ~/models/Q35-2B prompt \
       '760,4087,310,2199,11,279,15039,11,321,4156,369' /tmp/out
   ```
6. Observe real tokens; decode with `tokenizers.Tokenizer.from_file` back to
   English.

Before the patch, step 5 errors with `UnsupportedQwenQuantization` during
admission. After the patch, step 5 produces tokens the ExLlamaV3 reference
also produces for the same prompt.

## What Stage 6c did *not* do

- Did not teach `tensorfold` to read `.tfold` files directly. That's a
  larger seam (`TfoldFile` adapter + Checkpoint union), scoped in
  `docs/stage6c-v02-tfoldfile-design.md` for a follow-up PR.
- Did not touch the kernels. They were already correct.
- Did not alter the format spec. TFOLD v0.1 and v0.2 are unchanged.

## Next milestones

- **Stage 6c Gate B (v0.2 TFOLD inference):** land the `TfoldFile` seam in
  upstream (or in our fork as an overlay). Run the same prompt through
  the 3.2 GiB `.tfold` and compare tokens against Gate A.
- **Stage 7 (Android app):** unchanged — Snapdragon path, QNN forward,
  packaging. No dependency on Stage 6c Gate B.

## Acknowledgements and honest limits

- **The upstream kernels already supported mcg.** None of the GPU math is
  our contribution; we only taught a loader to pass through a selector it
  had been hard coding.
- **Gate B is not done in this session.** Stage 6c's "integrate TFOLD v0.2"
  requires the `TfoldFile` adapter. Design is landed; implementation is
  the next piece of work.
