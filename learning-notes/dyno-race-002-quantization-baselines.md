# Journal: Dyno Race 002 — First real quantization baselines

**Audience:** someone who has read the three previous journals and knows the
TFOLD container, the FP32-norm narrowing policy, and the Race 001 vLLM vs
TFOLD comparison. Never dealt with GGUF or EXL3 bit layouts directly before.

## What Race 002 actually asks for

Race 001 was runtime vs runtime on one checkpoint. Race 002 asks a different
question: **if we hold the model constant, what does each quantization
scheme buy us, and where is the first honest optimization target?**

Five sub-goals, in order:

1. Freeze the BF16 baseline (done in Race 001, carried forward).
2. Inventory the quantizers we can actually run.
3. Produce comparable low-bit candidates.
4. Measure size, memory, speed, and fidelity.
5. Pick ONE optimization that the evidence supports — and keep its branch
   separate from the baseline receipts.

Correction obligation: the Race 001 "TFOLD wins cold load 55×" line was
unsupported (TFOLD is a prebuilt binary, vLLM compiles on first use).
`RACE-001-CORRECTION.md` withdraws the claim without rewriting the
receipt.

## Why each candidate is in the race

The upstream model is `Qwen/Qwen3.5-2B` — a 2 B parameter vision-text LM.
Four candidates share that one parent:

- **BF16-reference**: our lossless `.tfold` and the safetensors directory
  from Race 001. 4.55 GB on disk, 16 bits per weight, used as the quality
  north star.
- **EXL3-8bpw**: a trellis-quantized version produced by exllamav3. 3.44 GB,
  ~8 bits per weight. We had this from Race 001.
- **GGUF-Q8_0**: llama.cpp's simple 8-bit round-to-nearest quant with 32-
  element blocks. We downloaded bartowski's published build.
- **GGUF-Q4_K_M**: llama.cpp's k-quant mix — Q4_K for most layers, Q6_K for
  embedding and output. Effective bpw came out to **5.705** when we counted
  actual tensor bytes divided by actual parameter count (there's no way to
  guess this from the filename).

We planned to also produce EXL3-4bpw. The quantizer ran layer 0 of 24 in
about 5 minutes on a contested GPU. Cancelled; documented as follow-up.
That's how it works: the honest answer when time runs out is "didn't ship",
not "can't be done".

A fifth entry on the brief was **EAQ**. It turns out EAQ is not a
competitor quantizer; it is TensorFold's own "Efficient Any-Quant"
container around EXL3 (`zig/src/core/eaq_container.zig`). Comparing TFOLD
against its own container would not teach anything.

## What each metric actually measures

- **Effective bpw**: tensor bytes ÷ parameter count. For Q4_K_M this is
  5.705, not 4.0 — because K-quants mix widths across tensors. If you quote
  "4-bit" without disclosing effective bpw, you are shading the comparison.
- **Peak VRAM**: with `-ngl 99` llama.cpp offloads all layers. That number
  is "what the runtime actually holds on-device", not what the file takes.
- **Decode tok/s**: a *steady-state* number from llama-server's HTTP
  `/completion`. Includes prefill of a short prompt; longer prompts would
  raise it (more tokens shared the prefill cost).
- **Container identity (SHA of logit dump)**: the hardest check. If two
  runs of the *same quantized weights* through the *same binary* produce
  the same SHA, the container wrapper is lossless. We showed this for
  EXL3: safetensors directory and `.tfold` both produced sha `c22df21e…`.

## Why the fidelity table is depressing — and still useful

Four micro-domains × five prompts × three models = 60 outputs. Every model
scored at or near the floor on the rubric:

| model | coding | reasoning | ifeval | cybersec |
|---|---|---|---|---|
| BF16 | 0/5 | 1/5 | 1/5 | 0/5 |
| Q8_0 | 0/5 | 1/5 | 1/5 | 0/5 |
| Q4_K_M | 1/5 | 1/5 | 1/5 | 0/5 |

The reason is not quantization. The base `Qwen3.5-2B` is a *vision-text*
LM. We ask it "Classify this vulnerability. Respond with only the CWE
number." It responds with "CWE-79 CWE-79 CWE-79 ..." no matter what. All
three models do this. Quantization does not break something that was
never there.

What the table **is** useful for is the cross-agreement column:

- Q8_0 produces token-identical output for 15 of 20 prompts vs BF16.
- Q4_K_M matches BF16 on only 4 of 20.

That is a *behavioral drift* signal, not a quality signal. Q4_K_M is not
"worse", it's "the same ability expressed slightly differently" because the
narrower weights push the argmax a different direction at some token
positions. For a strong instruct-tuned model this exact measurement would
mean something; for the base 2B VL model it means "both floors at the same
rubric-pass rate, but the shape of the floor differs".

The REPORT names that honestly. Our conclusion is "no measurable fidelity
loss at this test resolution", not "Q4_K_M is as good as BF16 for coding".

## The honest bottleneck

We nsys-profiled `tf-qwen35-cuda` running both BF16 and EXL3-8bpw. The
numbers are striking:

- BF16 is 75% `tf_qmmf::qmmf_kernel`. That's the BF16 matmul. Classic.
- EXL3 is **67% linear + 30% rotation** (`tf_exl3_linear_16_1_4` +
  `tf_exl3_rot_in`). Together: 97%.

The 30% rotation kernel is launched 6,400 times in a 48-row stage, paired
1:1 with the linear. Each launch is ~180 µs — individually tiny, but
collectively 30% of the whole GPU budget.

If this feels obvious, it's because it is. nsys reports do most of the
work; the skill is in deciding what to *do* about it.

## One candidate optimization

The rotation output is the input tile of the matmul. Merging the two means
the rotation never materialises in global memory; it lives as a
shared-memory transform in the linear kernel's prologue.

The candidate is on its own branch (`race-002/candidate/exl3-fuse-rot-in`),
with four pre-committed success criteria:

1. Logit SHA stays `c22df21edf589ec…` (bit-identical, non-negotiable).
2. Scenario_ms drops at least 10% from 5,819 ms.
3. BF16 path unchanged ±2%.
4. nsys shows `tf_exl3_rot_in` instance count = 0.

If any fails, the branch stays off `stage6b-cuda`. The baseline receipts
in this directory never move.

## Takeaways

1. **Hold the model constant.** All five candidates in Race 002 come from
   the same upstream repo at the same commit. That is the only way bpw and
   fidelity numbers can be compared at all.
2. **Trust the effective bpw you measure, not the one on the label.** Q4_K_M
   turned out to be 5.7 bpw on this model.
3. **Container identity is a strong, cheap check.** Same bytes in, same
   bytes out, verified by sha.
4. **Low rubric scores can still be a real result.** They tell you about
   the test resolution and the model's base ability, not just the
   quantization.
5. **ONE optimization, pre-committed criteria, separate branch.** Keeps
   baselines honest.

## Exercise

Open `score_fidelity.py`. Change the `IFEVAL_RUBRIC[3]` haiku check from
"3 non-empty lines" to a syllable-count checker. Rerun on the three
existing receipts. You should see every model drop to **0/5** on IFEval
(because none of them produce actual 5-7-5 haiku syllable counts
consistently). That's not a quantization story — it's the test resolution
story told again from the strict end.

## Comprehension questions

1. **Why does the "effective bpw" for GGUF Q4_K_M come out to 5.7 and not
   4.0?** (Think about which tensors in a transformer are *not* 4-bit
   quantized, and what K-quants do to the embedding and output heads.)
2. **The candidate optimization promises "bit-identical logits". Why is that
   a stronger success criterion than "logits within 1e-3"?** (What classes
   of bugs does a bit-identical requirement rule out that a near-equality
   tolerance cannot?)
