# Stage 6 — first tokens (learning journal + blocker report)

**For:** the project owner (you)
**Length:** ~25 min read
**Prerequisites:** Stage 3, 4, 5 journals (same folder); no prior LLM
background assumed

---

## 1. What this stage was supposed to do

Make a real model generate real tokens, loaded through our `.tfold`
format, using TensorFold's existing inference pipeline.

## 2. What this stage actually delivered

Tokenizer compatibility verification against the real
Qwen3.5-2B-exl3-8.00bpw tokenizer + a documented v0.2 format proposal
+ a formal gap analysis showing why end-to-end token generation on
this hardware is blocked.

**Stage 6 is reported as PARTIAL / BLOCKED, not complete.** This
journal explains why that's the honest answer and what you'd need to
do to unblock it.

---

## 3. The LLM pipeline in one picture

Even though we didn't execute it end to end, you need to understand it
to make sense of what the blocker is. Here's the whole chain that has
to work for a single generated token:

```
"Hello" ─┐
         │
         ▼
    ┌─────────────┐
    │  tokenizer  │  text → integer token IDs    e.g. [15339]
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  embedding  │  each ID → fp16 vector of hidden_dim      (table lookup)
    └─────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  for each transformer layer (28 in Qwen3.5-2B):
    │     RMSNorm                                  │
    │     attention / linear-attn (projections + KV cache)
    │     RMSNorm                                  │
    │     MLP (gate_proj + up_proj + down_proj)    │
    │     residual add                             │
    └─────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────┐
    │  final RMS  │  hidden_dim fp16 vector
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │   lm_head   │  hidden_dim → vocab_size logits   (big linear projection)
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │   sample    │  pick next token ID from logits (greedy = argmax)
    └─────────────┘
         │
         ▼
    next token ID, e.g. 11  →  append to history, run the whole thing again
```

Each arrow is a different thing that has to be correct. A bug in any
box corrupts everything downstream.

---

## 4. What each stage of the pipeline means, in plain words

### 4.1 Tokenizer

Models don't understand letters. They understand integers.
`"Hello, world!"` has to be chopped into a sequence of integer tokens
so the model can look them up. Qwen3.5's tokenizer is a **byte-pair
encoding (BPE)**: common chunks get single token IDs, rare chunks get
decomposed into smaller pieces. Example from our actual tokenizer:

```
"hi"           → [5834]              (one token)
"Hello, world!" → [9707, 11, 1917, 0]  (four tokens)
"def add(a, b): return a + b" → [750, 912, 2877, 11, 293, 1648, 470, 264, 488, 293]  (ten tokens)
```

Tokens 5834, 9707, etc. are indexes into a vocabulary of 151,936
entries. The tokenizer is **deterministic** — same text in, same IDs
out, every time. If it weren't, two runs of the same model on the
same prompt would be computing different things.

Our Stage 6 Python tests (`tests/test_tokenizer_qwen35.py`) verify
this directly against the real Qwen3.5 tokenizer we have on disk.

### 4.2 Embedding

Each integer token ID gets converted to a vector of `hidden_dim`
numbers (2048 for Qwen3.5-2B). It's a lookup table:

```
embedding[5834] = [0.0123, -0.0456, 0.0789, ..., 2048 numbers ...]
```

The embedding table is huge: `vocab_size × hidden_dim = 151,936 × 2,048
= 311 million numbers`. In our Qwen3.5-2B checkpoint it's stored as
FP16, taking 622 MB just for this one table. That's why every byte of
compression matters.

### 4.3 The 28 transformer layers

Each layer takes the current vector and produces another vector of
the same shape. The inside of a layer is where all the interesting
work happens:

- **Attention**: the layer looks at the current token AND every
  previous token in the sequence. For a chat turn of 1000 tokens, it
  computes how much the current token "pays attention to" each of the
  previous 1000. This requires the model to KEEP a **KV cache** — the
  keys and values of all previous tokens — in memory across each
  generated token.
- **Normalization (RMSNorm)**: scales the vector to a fixed length so
  the numbers don't explode or vanish as they pass through 28 layers.
- **MLP (feed-forward)**: three big linear projections
  (`gate_proj`, `up_proj`, `down_proj`) that mix all 2048 dimensions
  through a hidden 8192-dim intermediate space. This is where most
  of the FLOPs live.
- **Residual add**: the input to each layer is added to its output.
  This gives the model a direct path to copy information forward.

Every one of those projections is an EXL3-quantized matrix. 28 layers
× (3 attention projections + 1 attn output + 3 MLP projections) + 2
norms + 1 embedding + 1 head = **151 EXL3 groups in our real
checkpoint**, matching exactly what the Stage 6 scan found.

### 4.4 Final normalization + head

After the last layer, one more RMSNorm, then the **lm_head** — a huge
linear projection from hidden_dim → vocab_size. The output is 151,936
numbers called **logits**: one per possible next token, where the
value means "how much does the model think this token is the right
next token".

In our checkpoint the lm_head is encoded at 6 bpw (not 8 like
everything else) because the vocab is big and the few bits saved
matter. That one-off width is why Stage 6 concluded v0.2 needs a
per-tensor `quant.overrides` map.

### 4.5 Greedy sampling

For greedy decoding you pick `argmax(logits)` — the token with the
highest score. That's it. Deterministic, no randomness, reproducible
across runs. If logits[11] is the highest value, the model generates
token 11 (which happens to be ","), and the whole pipeline runs
again with `[your prompt tokens…, 11]` as input.

More sophisticated sampling (top-k, top-p, temperature) multiplies
the logits by weights and samples from the resulting distribution,
but we don't need any of that to prove the pipeline works.

---

## 5. Why Stage 6 is blocked on this hardware

Three independent blockers, discovered in order:

### Blocker 1: v0.1 can't carry the full checkpoint

v0.1 = one `quant.scheme`, one `bits_per_weight`, one `trellis`.
Real checkpoint = 151 EXL3 groups + 116 plain tensors, uniform 8.0
bpw except head at 6 bpw.

Stage 5 already flagged this as a v0.2 concern. Stage 6's
`tools/tfold_convert/_analyze_qwen35.py` scan confirmed the exact
numbers. `docs/tfold-format-v0.2-proposal.md` specifies the minimum
backward-compatible extension.

This blocker is ~1 working day of implementation. I did NOT ship it
in Stage 6 because of Blocker 3.

### Blocker 2: no full-checkpoint conversion

Even if v0.2 were ready, the conversion step itself hasn't been
built. Loading the 2.77 GB safetensors + computing SHA-256 of 267
tensor payloads + writing them into a new container is another few
hours.

Not shipped for the same reason as Blocker 3.

### Blocker 3 (terminal): no runnable forward path on this hardware

**This is the one that stops Stage 6 cold.**

TensorFold's Qwen3.5-2B forward pass lives in
`zig/src/families/qwen3_5/backend.zig` + `forward.zig` + `kernels.zig`.
All three files unconditionally import `@import("metal")` — Apple's
Metal graphics API. There is no CPU forward path for Qwen3.5-2B in
the TensorFold tree.

Our dev machine is Windows + Linux (WSL + the datacenter VMs).
Neither runs Metal. We have no Apple hardware in the lab.

Consequences:
- Even if I built the perfect v0.2 .tfold file carrying all 267
  tensors with the right scheme overrides, I have no way to
  **execute** it on this hardware.
- Shipping v0.2 + conversion code now would produce infrastructure
  nobody can validate end to end. The brief forbids that: *"Do not
  mislabel a partial checkpoint as runnable."*

### What would unblock it?

Pick exactly one:

1. **Add a CPU forward path** for Qwen3.5-2B in
   `zig/src/families/qwen3_5/`. Non-trivial: reuse
   `zig/src/core/exl3_format.zig` decoder on the trellis + implement
   attention + KV cache + sampling on pure CPU. Multi-week project.
2. **Run on a Mac.** Someone with M-series hardware builds the
   TensorFold tree, runs the existing Metal path. Takes an afternoon
   once we have the hardware access.
3. **Pair with an existing TensorFold runtime caller** in CUDA/ROCm.
   Qwen27 and Flashnext families have CUDA engines — a port of
   qwen3_5 to CUDA would unlock H100/RTX execution. Weeks of work.

---

## 6. What Stage 6 DID deliver

Prerequisites that any of the three unblock paths above will need:

### 6.1 Verified tokenizer path

`tests/test_tokenizer_qwen35.py` — 8 passing tests against the real
Qwen3.5-2B tokenizer on disk. Confirms:
- Round-trip on common prompts
- Deterministic across repeated encodes
- `hi` tokenizes to `[5834]` (matches docs/qualcomm-android.md)
- All known special tokens resolvable
- Vocab size ≤ model's config.json vocab_size

When real inference lands, we won't be debugging the tokenizer.

### 6.2 Honest representability analysis

`tools/tfold_convert/_analyze_qwen35.py` — scans the real
quantization_config.json and emits:
- 267 tensor groups (151 EXL3 + 116 plain)
- Uniform mcg codebook, 8.0 bpw body, 6.0 bpw head
- 2.77 GB total payload
- v0.1 would require 267 files; v0.2 can hold one

Reruns deterministically on any host that has the model dir.

### 6.3 v0.2 format proposal

`docs/tfold-format-v0.2-proposal.md` — concrete header-only addition
spec. Adds two optional keys (`quant.overrides` and `quant.groups`),
zero binary-layout changes, strict backward compatibility, explicit
reject behavior for readers that don't implement it.

### 6.4 Tokenizer + loader negative tests

`tests/test_stage6_rejects.py` — 5 passing tests covering:
- Tokenizer SHA pinned in the .tfold header, round-trips through writer+reader
- Writer rejects unsupported arch values
- Executor-side tokenizer SHA mismatch flagged
- Missing required tensor flagged
- v0.2-stamped file rejected by the v0.1 reader

### 6.5 This journal + the conformance report

You're reading the journal. The formal report lives at
`docs/tfold-stage6-conformance.md` and itemizes every claim with
evidence.

---

## 7. A small numerical example — what would generation look like?

Pretend for a moment we had everything wired. Let's walk through
generating ONE token from the prompt `"hi"`.

```
Step 1: tokenize
  "hi" → [5834]

Step 2: embed
  embedding[5834] → fp16 vector of 2048 numbers
  e.g. [0.012, -0.045, 0.079, ..., 0.003]

Step 3: run through 28 layers
  Each layer takes the 2048-vector and produces another 2048-vector.
  It also stores K+V tensors in the KV cache so the NEXT token can
  attend to this one.
  (details omitted; this is where the 2.77 GB of weights get used)

Step 4: final norm
  RMSNorm applied to the final 2048-vector.

Step 5: lm_head
  hidden × lm_head.weight.T → logits of length 151,936
  [0.1, 2.3, -1.4, ..., 5.7, 0.8, ...]

Step 6: greedy argmax
  argmax(logits) → say, 11
  11 → "," (via tokenizer decode)

Step 7: appended to history, run again
  input = [5834, 11]
  repeat from step 2 using KV cache so we don't recompute 5834's
  attention
```

The reason KV cache matters: if you generated 100 tokens and had to
re-run the entire attention stack for all 100 of them each time,
you'd do 100² = 10,000 attention operations per generation step. With
the cache, you do only 100. Modern LLM inference is 90% engineering
around making the KV cache fast.

---

## 8. Hands-on exercise — tokenizer round-trip (15 min)

Even without model execution, you can watch the tokenizer at work.

1. From the repo root:
   ```powershell
   cd D:/op13-root/src/tensorfold
   $env:PYTHONPATH = "tools"
   wsl /tmp/tfold-venv/bin/python
   ```
2. In the Python REPL:
   ```python
   from tokenizers import Tokenizer
   tok = Tokenizer.from_file("D:/op13-root/models/Qwen3.5-2B-exl3-8.00bpw/tokenizer.json")

   # Encode a prompt
   ids = tok.encode("Hello, world!", add_special_tokens=False).ids
   print("IDs:", ids)

   # See what each ID means
   for i in ids:
       print(f"  {i:6d}  →  {tok.id_to_token(i)!r}")

   # Decode back
   print("back:", tok.decode(ids))

   # The 'hi' oracle from docs/qualcomm-android.md
   print("hi IDs:", tok.encode("hi", add_special_tokens=False).ids)
   # Expected: [5834]

   # Explore the 'im_end' special token
   print("im_end id:", tok.get_vocab()["<|im_end|>"])
   ```
3. **Bonus:** write a tiny loop that would GENERATE tokens, if we had
   a model. You can fake the logits step by just picking a random
   token. The point is to see the control flow.
   ```python
   import random
   history = tok.encode("Once upon a time", add_special_tokens=False).ids
   vocab_size = tok.get_vocab_size(with_added_tokens=True)

   for step in range(10):
       # In a real model: logits = model.forward(history)
       # Here: pretend we got random logits and picked the top one
       fake_next_token = random.randint(0, vocab_size - 1)
       history.append(fake_next_token)
       print(f"step {step}: next_id={fake_next_token} → "
             f"text so far: {tok.decode(history)!r}")
   ```
   What this shows: once the model forward is working, the autoregressive
   loop itself is tiny. Fifteen lines of Python. The model IS the
   hard part.

---

## 9. Comprehension check

### Q1

> Why can't we just patch v0.1 to accept multiple schemes in one
> `quant` block, instead of waiting for a formal v0.2?

**Answer**

Because v0.1 is a **published format spec**. Any file written today
that claims to be v0.1 must be readable by any reader that also
claims to implement v0.1. If we quietly started emitting new
semantics under the old version bit, we'd break the meaning of the
version number — the single most important invariant a file format
has. A v0.2 reader has a different major version in the magic bytes
so v0.1 readers reject it loudly (`TFOLD_VERSION_UNSUPPORTED`)
rather than quietly consuming data they don't actually understand.

This is the same principle Stage 5 held when it hit the one-group
limit: document the gap, propose the extension, don't patch under
cover of night.

### Q2

> If we had a Mac and could run the existing Metal forward path on a
> normal safetensors load of Qwen3.5-2B (bypassing .tfold entirely),
> what would that prove about .tfold?

**Answer**

**Almost nothing about .tfold directly.** It would prove the model
weights are correct and the forward pass is correct — those are
properties of the checkpoint and the Zig code, not the format. .tfold
would only come into play once we routed that same forward pass
through a converted .tfold file instead.

That would be the correct validation ladder:
1. Run the Metal forward on a safetensors load → check tokens are sane
   (proves the model + forward are healthy)
2. Convert the same safetensors → .tfold v0.2
3. Run the Metal forward on the .tfold-loaded tensors → check tokens
   are IDENTICAL to step 1
4. Celebrate: .tfold is bit-equivalent to safetensors for this model

Without step 1 you can't tell whether a token difference at step 3 is
a .tfold bug or a model bug or a Metal bug. That's why "just get a
Mac" is step one of unblocking Stage 6.

---

## 10. What Stage 6 is NOT

- Not a working autoregressive run. We did zero inference.
- Not a v0.2 implementation. Only a proposal.
- Not a performance claim. No timing, no RAM, no token/s.
- Not a lab failure. Blockers are documented upfront in the Stage 6
  brief; we hit them exactly where the brief anticipated.

What it IS: honest prerequisites + a clear statement of what has to
land before Stage 6 can genuinely complete.

---

## 11. Files to look at

| File | What's in it |
|---|---|
| `tests/test_tokenizer_qwen35.py` | 8 passing tokenizer compat tests |
| `tests/test_stage6_rejects.py` | 5 passing negative tests |
| `tools/tfold_convert/_analyze_qwen35.py` | real checkpoint scan |
| `docs/tfold-format-v0.2-proposal.md` | the minimum v0.2 extension |
| `docs/tfold-stage6-conformance.md` | formal partial / blocked report |
| `zig/src/families/qwen3_5/` | the Metal-only Qwen3.5 forward (blocker evidence) |
| `D:/op13-root/models/Qwen3.5-2B-exl3-8.00bpw/` | the actual model + tokenizer on disk |
