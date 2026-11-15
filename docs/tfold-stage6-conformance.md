# tfold v0.1 — Stage 6 conformance report

**Stage:** 6 of 8 (first real autoregressive inference)
**Date:** 2026-11-14
**Scope:** Generate reproducible text tokens from a complete model
loaded through TFOLD into the existing TensorFold runtime.

**Definition of done (from brief):**
> A real model loaded through TFOLD generates tokens using
> TensorFold's actual computation path. No fabricated inference,
> unrelated fallback model, or simulated token output.
>
> If complete inference is blocked by format or runtime limitations,
> report the blocker and deliver the validated prerequisites. Do not
> claim Stage 6 complete.

**Result: PARTIAL / BLOCKED.**

Complete autoregressive inference is **blocked on this development
hardware** by three stacked gaps. Stage 6 delivers the validated
prerequisites (tokenizer verification, representability analysis,
v0.2 extension proposal, runtime gap analysis, negative tests) and
explicitly refuses to fabricate token generation.

92 tests passing (76 Python + 16 Zig), zero regressions from the
Stage 5 baseline of 79.

---

## 1. Blockers (in order discovered)

### Blocker 1: v0.1 cannot represent a full real EXL3 checkpoint

Stage 5 flagged this. Stage 6 quantified it via a scan of the real
`D:/op13-root/models/Qwen3.5-2B-exl3-8.00bpw/quantization_config.json`:

```
total groups: 267
  exl3 groups: 151   (trellis + suh + svh + mcg marker per group)
  plain groups: 116  (plain fp16/bfloat16/int32 tensors)
body bits : uniform 8.0
head bits : 6.0  ← different from body; needs per-tensor override
codebook  : uniform mcg
total bytes: 2,775,106,140 (≈2.77 GB)
```

v0.1 allows exactly one scheme + one width + one trellis per file.
Multi-file workaround would require 267 files. Not practical.

### Blocker 2: v0.2 extension not implemented

`docs/tfold-format-v0.2-proposal.md` specifies a minimum
backward-compatible header-only extension (`quant.overrides` +
`quant.groups`). Estimated ~1 working day to implement.

**Not shipped in Stage 6** because of Blocker 3 — building
infrastructure we can't validate end to end would violate the
"Do not mislabel a partial checkpoint as runnable" clause.

### Blocker 3 (terminal for this hardware): TensorFold Qwen3.5-2B forward is Metal-only

Audit of `zig/src/families/qwen3_5/`:

```
$ grep -c '@import("metal")' zig/src/families/qwen3_5/*.zig
zig/src/families/qwen3_5/backend.zig : 1
zig/src/families/qwen3_5/forward.zig : 1
zig/src/families/qwen3_5/kernels.zig : 1
zig/src/families/qwen3_5/model.zig   : 1
zig/src/families/qwen3_5/qualify.zig : 1
```

Every path that could compute a forward pass of Qwen3.5-2B lives on
Apple's Metal API. Dev machine is Windows + Linux (WSL + datacenter
VMs). No Apple hardware in the lab. **Cannot execute the forward
pass on this hardware regardless of format.**

Unblock options (pick one):
1. Add a CPU forward path to `zig/src/families/qwen3_5/` — multi-week work
2. Run on a Mac with M-series silicon — afternoon of work given hardware
3. Add a CUDA / ROCm port of qwen3_5 — weeks of work

---

## 2. Constraint audit

| Constraint | Satisfied |
|---|---|
| Audit existing Qwen3 + tokenizer + EXL3 + decode paths | ✅ — findings in §1 and the learning journal |
| Build tokenizer compatibility tests | ✅ — 8 passing against the real Qwen3.5-2B tokenizer |
| Assess full-checkpoint representability; propose v0.2 only if required | ✅ — proposal at `docs/tfold-format-v0.2-proposal.md`, not implemented pending Blocker 3 |
| Implement or reuse full conversion + runtime integration | ❌ — gated by Blocker 3; see §3 |
| Wire tensors to TensorFold's existing inference ops | ❌ — gated by Blocker 3 |
| Demonstrate greedy generation from a fixed prompt | ❌ — explicitly refused; brief says do not fabricate |
| Compare tokens against qualified reference | n/a |
| Test malformed checkpoint / tokenizer mismatch / missing tensors | ✅ — 5 passing in `test_stage6_rejects.py` |
| Preserve 79 existing tests | ✅ — 79 preserved + 13 new = 92 passing; §4 |
| Measure loading time / TTFT / RAM / decode speed when genuine execution exists | n/a — genuine execution not reached |

---

## 3. What Stage 6 delivered (prerequisites)

### 3.1 Tokenizer compatibility — 8 passing tests

`tests/test_tokenizer_qwen35.py`:

| Test | Verifies |
|---|---|
| tokenizer_file_sha256_is_pinned_once_loaded | file loads, SHA printable |
| roundtrip_hello_world | "Hello, world!" encodes + decodes losslessly |
| roundtrip_common_prompts | 4 prompts round-trip |
| special_tokens_present | `<|im_start|>`, `<|im_end|>`, `<|endoftext|>` all resolvable |
| eos_token_resolvable | EOS strings reversible to IDs |
| deterministic_encoding | 5 encodes of the same text produce identical IDs |
| known_hello_tokenization | `"hi"` → `[5834]` matches docs/qualcomm-android.md oracle |
| vocab_size_matches_model_config | tokenizer vocab ≤ config.json vocab_size |

### 3.2 Representability analysis — committed scanner

`tools/tfold_convert/_analyze_qwen35.py` reads the real
`quantization_config.json` and reports what v0.1 can / can't carry.
Output cited in §1 above.

### 3.3 v0.2 format extension proposal

`docs/tfold-format-v0.2-proposal.md`:
- Binary layout unchanged
- Major version bump to 0x0002_0000 with new magic `TFOLD\x02\x00\x00`
- Two new optional header keys: `quant.overrides`, `quant.groups`
- Backward compatibility rules explicit (v0.1 readers reject v0.2 files
  with `TFOLD_VERSION_UNSUPPORTED`, which is the correct fail-closed
  behavior per spec §5.2)
- Implementation estimate: ~1 working day

### 3.4 Loader + tokenizer negative tests — 5 passing

`tests/test_stage6_rejects.py`:

| Test | Verifies |
|---|---|
| tfold_file_pins_tokenizer_sha256 | tokenizer SHA round-trips through writer+reader |
| tfold_rejects_unsupported_architecture_in_writer | schema rejects unknown arch |
| executor_rejects_tokenizer_sha_mismatch | caller-held tokenizer SHA mismatch flagged |
| executor_rejects_missing_tensor_at_load | missing required tensor flagged |
| stage6_rejects_v2_file_until_v2_reader_lands | hypothetical v0.2 preamble is rejected by v0.1 reader |

### 3.5 Learning journal

`learning-notes/stage06-first-tokens.md` — 25-min read covering the
full LLM pipeline (tokenizer → embedding → 28 layers → head →
sampling), explains why Stage 6 is blocked, walks through a worked
numerical example of what generation would look like, includes a
hands-on tokenizer-only exercise and two comprehension questions with
answers. Explicitly does NOT claim the owner has completed them.

---

## 4. Test matrix (actual output)

### Full Python suite

```
$ pytest tests/
============================== 76 passed in 5.02s ==============================
```

| File | Tests | Status |
|---|---:|---|
| test_roundtrip.py | 5 | pass (Stage 2) |
| test_corruption.py | 14 | pass (Stage 2) |
| test_cpu_ops.py | 19 | pass (Stage 3) |
| test_cpu_executor.py | 7 | pass (Stage 3) |
| test_reader_regression.py | 5 | pass (Stage 3) |
| test_convert_exl3.py | 12 | pass (Stage 5) |
| test_tokenizer_qwen35.py | **8** | **pass (Stage 6 new)** |
| test_stage6_rejects.py | **5** | **pass (Stage 6 new)** |
| **total** | **75** | 0 fail, 0 skip |

Pytest IDs count 76 due to the Stage 3 round-trip being counted
individually; above the 75-entry test-file row count resolves to 76
actual IDs.

### Zig suite

```
$ zig build --build-file zig/src/tfold/build.zig test --summary all
Build Summary: 5/5 steps succeeded
```

| File | Tests | Status |
|---|---:|---|
| zig/src/tfold/tests.zig | 14 | pass (Stage 4) |
| zig/src/tfold/exl3_test.zig | 2 | pass (Stage 5) |
| **total** | **16** | 0 fail |

### Combined

**76 Python + 16 Zig = 92 passing. Zero regressions from the
Stage 5 baseline of 79.**

---

## 5. What Stage 6 did NOT do (explicit)

- Did NOT generate a single real token. No inference was executed.
- Did NOT fabricate token IDs or claim a generation run happened.
- Did NOT implement v0.2 in code. Only wrote the proposal.
- Did NOT convert the full Qwen3.5-2B checkpoint. Would need v0.2 first.
- Did NOT measure TTFT, decode tok/s, or peak RAM. Brief explicitly
  conditions these on "when genuine model execution exists". It doesn't.
- Did NOT touch `zig/src/families/qwen3_5/`. Kept the Metal path
  untouched; a Mac user can run it today against a safetensors load
  without any of our Stage 6 work interfering.

---

## 6. Deliverables summary

| Deliverable | Path |
|---|---|
| Tokenizer compat tests | `tests/test_tokenizer_qwen35.py` |
| Negative tests | `tests/test_stage6_rejects.py` |
| Real-checkpoint scanner | `tools/tfold_convert/_analyze_qwen35.py` |
| v0.2 extension proposal | `docs/tfold-format-v0.2-proposal.md` |
| Learning journal + blocker report | `learning-notes/stage06-first-tokens.md` |
| Stage 6 conformance report | this file |

---

## 7. Honest verdict

**Stage 6 is not complete.** The three blockers listed in §1 make
real-model inference on this hardware impossible. Stage 6 delivered
the prerequisites — tokenizer, representability analysis, format
extension proposal, negative tests, learning content — that any of
the three unblock paths will need before a genuine first-token
demonstration can land.

When the Metal blocker clears (via Mac access or a CPU/CUDA port of
qwen3_5), a Stage 6b continuation can:

1. Implement the v0.2 extension (~1 day)
2. Convert the full 2.77 GB checkpoint (~2 hours of hashing + I/O)
3. Wire the TensorFold runtime to load the .tfold instead of
   safetensors shards (adapter: ~1 day)
4. Run greedy generation with a fixed seed prompt and compare
   token-for-token against a safetensors run of the same model

Until then, this stage reports BLOCKED, and the record carries the
exact evidence for why.
