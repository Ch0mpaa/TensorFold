# PR #3412 — Final CUDA regression on DGX Spark

Upstream PR: https://github.com/ashhart/TensorFold/pull/3412
Head tested: **`ff439ed55`** (amended from the Synthena-reviewed
`5d6dc705c` with a buffer-aliasing fix; Stage 7F documented the catch).
Dtype follow-up branch (kept separate): **`e9558e288`** (rebased from
`99fc6e3c6` on top of `ff439ed55`).

## 1. Hardware + toolchain

```
host:        spark
gpu:         NVIDIA GB10 Grace Blackwell, compute_cap 12.1 (sm_121)
driver/UMD:  615.71.09, CUDA UMD 13.4
toolchain:   zig 0.17.0 at /home/chris/opt/zig-0.17.0/zig
             nvcc 13 at /usr/local/cuda/bin/nvcc
repo:        /home/chris/src/tensorfold
worktree:    /home/chris/worktrees/pr3412 (detached HEAD ff439ed55)
```

## 2. Checkpoint under test

Local — the only Qwen3.5 EXL3 checkpoint available on Spark:

```
path:        /home/chris/models/Qwen3.5-2B-exl3-8.00bpw/model.safetensors
sha256:      a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f
size:        3,438,956,902 bytes
author:      UnstableLlama/Qwen3.5-2B-exl3-8.00bpw
revision:    main (pinned via sha256 above)
marker:      mcg (151 tensors)
norms (f32/bf16): 18 / 159
embed dtype: F16
A_log dtype: F32
```

## 3. Host regression suites (all Zig; no GPU required)

Commands:

```bash
cd ~/worktrees/pr3412
export PATH=/usr/local/cuda/bin:$PATH
/home/chris/opt/zig-0.17.0/zig build <step> --summary all
```

| Step | Pass | Skip | Fail | Notes |
|---|---|---|---|---|
| `test-qwen35-exl3-admission-host` | 6 | 0 | 0 | admission + dual-marker (new) |
| `test-exl3-cuda-host` | 1000 | 10 | 0 | broad EXL3 CUDA host coverage |
| `test-qwen35-cache-host` | 23 | 0 | 0 | cache contracts |
| `test-qwen35-exl3-owner-host` | 1 | 0 | 0 | ownership lifecycle |
| `test-qwen35-formats-host` | 11 | 0 | 0 | embedding dispatch + formats |
| `test-exl3-prompt-host` | 7 | 0 | 0 | common prompt dispatch |
| `test-glm-exl3-prompt-host` | 5 | 0 | 0 | GLM EXL3 whole prompt |
| `test-glm-formats-host` | 12 | 0 | 0 | GLM EXL3 offer registry |
| **Totals** | **1065** | **10** | **0** | — |

## 4. Format checks

```
$ zig fmt --check zig/src/families/qwen3_5/cuda_exl3_admit.zig \
                 zig/src/families/qwen3_5/cuda_exl3_weights.zig \
                 zig/src/families/qwen3_5/cuda_exl3_loader_test.zig
exit=0
```

```
$ python3 tools/zig/lean_check.py 2>&1 \
    | grep -E 'cuda_exl3_(admit|weights|loader_test)'
(no matches)
```

## 5. Negative tests: conflicting markers refused

`writeBoth()` fixture writes a safetensors file carrying BOTH a
`p.mul1` and a `p.mcg` marker tensor in one group. The admission path
must refuse before touching the markers' bytes.

Result: **pass**.

```
EXL3 admission: a group carrying both mul1 and mcg markers is refused: PASS
```

The refusal error is `error.UnsupportedQwenQuantization`, same error
as missing-marker and wrong-magic cases. The check fires on name
presence only (`ck.has(mul1) and ck.has(mcg)`), before any `ck.get()`.

### Buffer-alias bug caught

The initial Stage 7E hygiene commit (5d6dc705c) used one stack buffer
for both `mul1_name` and `mcg_name`:

```zig
const mul1_name = try name(&buf, "{s}.mul1", .{base});
const mcg_name  = try name(&buf, "{s}.mcg",  .{base});  // overwrites buf
```

`std.fmt.bufPrint` writes into the shared buffer. On a mul1-only
checkpoint, `ck.has(mul1_name)` read bytes spelling `p.mcg`, admission
fell through to the else branch, and a valid mul1 checkpoint was
refused. Three admission tests failed on first Spark run — two
pre-existing mul1 tests and the new dual-marker test.

**Fix** (now the PR head `ff439ed55`):

```zig
var buf:     [240]u8 = undefined;
var mcg_buf: [240]u8 = undefined;
...
const mul1_name = try name(&buf,     "{s}.mul1", .{base});
const mcg_name  = try name(&mcg_buf, "{s}.mcg",  .{base});
```

After fix: 6/6 admission tests pass on Spark. No upstream regression.

## 6. Real-checkpoint load attempts

### MCG — Qwen3.5-2B

```
$ zig build qwen35-cuda -Doptimize=ReleaseFast -Dnvcc=nvcc -Dsm=121
$ ./zig-out/bin/tf-qwen35-cuda \
    ~/models/Qwen3.5-2B-exl3-8.00bpw \
    step "15226,2746,279" \
    /tmp/stage7g-mcg.bin
error: model.language_model.embed_tokens.weight: f16 { 248320, 2048 }, expected bf16 { 248320, 2048 }
error: UnexpectedTensor
exit=0
```

**Expected and correct.** The admission branch does NOT widen F16
embeddings; that's the dtype branch's scope. The checkpoint stores
`embed_tokens.weight` as F16, so the loader refuses at the embedding
tensor. The checkpoint never reaches the F32 norm tensors (which
would ALSO refuse per Synthena's exactness requirement; see Stage 7F).

### MUL1 — Qwen3.5-2B

**Not available.** Exhaustive HuggingFace search:

- `search="Qwen3.5 mul1"` → 0 results.
- Every UnstableLlama Qwen3.5 EXL3 variant (0.8B, 2B, 4B, 8B, 27B, all
  bpw settings) uses mcg + F32 norms + F16 embed.
- `TheMelonGod/Qwen3.5-2B-exl3`, `UwUtismXD/Qwen3.5-2B-exl3`: empty
  repos with no model.safetensors.
- Non-Qwen3.5 mul1 checkpoints exist (`Lygodactylus/Qwen3.8-27B-*`,
  `pogon90/Qwen3-Coder-30B-*`) but are different families — outside the
  qwen3_5 loader's architecture and the operator's scope restriction.

MUL1 admission is instead exercised through the synthetic host tests
(`invalid widths refuse before upload...`, `mixed supported widths and
allocator failures stay host owned`, `marker dtype and scale shape...`
and the new `dual-marker` case). All pass on `ff439ed55`.

### Explicit non-comparison

**We do NOT compare EXL3-mul1 and EXL3-mcg outputs against each
other.** They use different trellis codebooks (`exl3.Codebook.mul1`
vs `exl3.Codebook.mcg`), so even for byte-identical pre-quantized
source weights the quantized bytes and decoded reference values
diverge. Each quantization must only be compared against its own
reference implementation. For mcg Qwen3.5-2B a reference run would
require `exllamav3`'s mcg decoder with the same checkpoint + the same
sampling settings; that comparison is Mission 2 (Dyno) scope.

## 7. Dtype branch (not in this PR)

`stage6c-cuda-dtype @ e9558e288` sits on top of `ff439ed55` and adds:

- `embedTokens()` + `mappedEmbed()` accept F16 embed; widen to BF16
  via `f16Rows()` (bit-exact on the audited checkpoint; see the dtype
  commit message).
- `f32vec()` A_log passthrough when checkpoint stores F32.

With the dtype branch applied, the mcg Qwen3.5-2B load refuses at
`linear_attn.norm.weight` (F32 vs BF16) — honest refusal because
narrowing F32→BF16 loses 85 % of bit-exactness on the audited
checkpoint (Stage 7F receipt).

Full-model mcg Qwen3.5-2B inference requires a kernel-side F32 norm
path in `tf_q35_gdn_post`; that is a separate (third) PR scope, not
on either reviewed branch. The operator brief for Mission 1
reaffirmed: F32 norm support is reported as unsupported until that
kernel path exists.

## 8. Upstream CI

Status at `ff439ed55` as of this run:

```
$ gh pr view 3412 --repo ashhart/TensorFold \
      --json statusCheckRollup
{"statusCheckRollup": []}
```

All configured workflows (`No GPU`, `Zig format`, etc.) show
**`action_required`** — the upstream repo requires maintainer
approval for workflow runs from forked PRs. The workflows have not
executed at `ff439ed55`.

Per operator brief: **do not re-request upstream review until
required CI checks execute successfully on the final PR commit**.
This document reports local Spark validation as green and upstream
CI as pending-approval. The two are reported independently.

## 9. Scope guards honored

- ✅ PR #3412 scope unchanged: mcg admission + dual-marker refusal + buffer-alias fix.
- ✅ No merge of PR #3412.
- ✅ Dtype changes kept on `stage6c-cuda-dtype` only.
- ✅ No F32-to-BF16 norm narrowing reintroduced on either branch.
- ✅ F32 norms reported unsupported.
- ✅ No re-request of upstream review.
- ✅ Previously-passing host tests preserved (1065 pass, 10 skip).
- ✅ Different-quantization outputs never compared as if identical.

## 10. Verdict

Local CUDA regression on Spark: **green**.
Upstream CI: **action_required** (no verdict yet).
Mission 2 (Dyno) can proceed on the dtype branch head
`e9558e288` for the EXL3 side, using `exllamav3` as the reference
runtime.
