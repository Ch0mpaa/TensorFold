# Journal: taking the official Qwen3.5-2B straight into TFOLD

**Audience:** someone who has used Python and Git, has never looked inside a
safetensors or `.tfold` file, and wants to follow every decision this milestone
made and the two real surprises along the way.

## What we set out to do

The TensorFold stack had a `.tfold` checkpoint for Qwen3.5-2B, but it had been
built *through* an EXL3 quantizer. That means the weights on disk were already
a compact 8 bpw representation, not the plain BF16 the model's authors released.
We wanted a checkpoint that

1. starts from the original `Qwen/Qwen3.5-2B` safetensors on HuggingFace,
2. preserves every tensor's **bytes** (no silent float cast anywhere),
3. still fits the TFOLD `.tfold` container format, and
4. can be loaded by the same `tf-qwen35-cuda` binary we already use.

We also had a strict ground rule: **never claim a conversion makes a model
faster or better.** Lossless byte-for-byte rewriting cannot change throughput
or accuracy; it only changes the container.

## Step 1 — read the format before writing it

Instead of guessing at the `.tfold` layout, we opened the existing
`Qwen3.5-2B-exl3-mcg.tfold` and reverse-engineered it, then cross-checked the
Zig source at `src/core/tfold_file.zig`. The spec lives in the file itself:

```
bytes 0..8    : "TFOLD\x02\x00\x00"          (magic)
bytes 8..12   : 0x00020000                    (version u32 LE)
bytes 12..16  : header_len                    (u32 LE)
header_len bytes of UTF-8 JSON              (the schema)
<pad to 64-byte boundary>
tensor bytes at absolute offsets listed in the JSON
<last 8 bytes>  : FNV-1a-64 of every preceding byte (LE)
```

The JSON itself lists each tensor's name, dtype, shape, `offset`, `nbytes` and
`sha256`. Lesson: *when a format is small, write a reader for it before you
write a writer. The reader tells you exactly what the writer must produce.*

## Step 2 — pin upstream precisely

We fetched `Qwen/Qwen3.5-2B` from HuggingFace at the exact revision commit
`15852e8c16360a2fea060d615a32b45270f8a8fc`, and recorded `sha256` for the
shard, the config, the tokenizer and the LICENSE. The hashes let anyone
rebuild the same `.tfold` later without having to trust that the HF mirror
hasn't moved.

Surprise #1: the official `Qwen/Qwen3.5-2B` is actually an *image-text-to-text*
model. The safetensors contains both a vision tower and the language model in
one 4.55 GB file, totalling 632 tensors (596 BF16 + 36 FP32). Our lossless
goal means we keep *all* of it; later loaders can decide which subtree to
ignore.

## Step 3 — write the converter

`docs/tfold/convert_safetensors_to_tfold.py` streams the shard tensor-by-tensor
and writes a `.tfold` without ever decoding the floats:

- read the safetensors JSON header
- for each tensor, copy exactly `data_offsets[1] - data_offsets[0]` raw bytes
- record its SHA-256 in the output header
- **abort** if the dtype isn't already a TFOLD v0.2 dtype — we would rather
  stop than cast

Three details became bugs the first time:

- **Preamble bytes.** The spec's "8-byte magic" includes two trailing `\x00`
  bytes (`TFOLD\x02\x00\x00`), then a separate 4-byte version word
  (`0x00020000`). My first writer put the `\x02` byte and six reserved zeros,
  which left the version word all-zero and the loader rejected it.
- **64-byte alignment.** The Zig reader errors out with
  `TfoldTensorMisaligned` if a tensor's absolute offset is not a multiple of
  64. The converter now rounds each tensor offset up to the next 64-byte
  boundary and writes zeros to fill the gap.
- **FNV-1a-64 trailer in pure Python.** FNV is a per-byte loop of XOR +
  multiply. Running that in Python over a 4.5 GB file takes longer than making
  coffee. We dropped a tiny 20-line C program (`fnv_append.c`) that reads the
  file once, computes the hash and appends the 8-byte little-endian trailer in
  about 3.5 s. **Lesson:** cryptographic-shaped work doesn't need a crypto
  library; it needs a language that can touch a byte in nanoseconds.

The final output size is 4,548,300,616 bytes (upstream + header + pad + FNV).

## Step 4 — prove byte-level parity

`docs/tfold/verify_parity.py` opens both files back up and re-hashes every
tensor's bytes from each. It does not share any code with the writer, so a
match is real proof — not a fixpoint. The run prints:

```
PASS: 632 tensors byte-identical
```

If ever this script says FAIL for a tensor, the converter is wrong, not the
model. Keep this script in CI if you ever touch the writer.

## Step 5 — try the native runner

Here is the honest result. We put `model.tfold` into a directory with the HF
tokenizer/config files and invoked
`tf-qwen35-cuda /tmp/tfold_dir stage /tmp/stage_out.bin`. The loader printed:

```
tfold_file open: mmap=36us preamble=344us fnv=3.6s header_parse=3094us
fnv_bytes=4548300608
error: model.language_model.layers.0.linear_attn.norm.weight:
       f32 { 128 }, expected bf16 { 128 }
error: UnexpectedTensor
```

Steps 1–4 (mmap, preamble, FNV integrity, JSON header) all succeeded. The
runtime then halted because the *dense* CUDA weights file
(`cuda_weights.zig:413`) expects `linear_attn.norm.weight` in BF16, while the
official checkpoint ships it in FP32.

Surprise #2: *the EXL3 path already handles this exact tensor in both dtypes*
— `cuda_exl3_weights.zig:315` uses a helper called `bf16OrF32` that upcasts
FP32 norms to BF16 on load. The dense path was simply never generalised the
same way. The minimum change needed to execute our lossless checkpoint is to
mirror that one line.

We *did not* silently cast the tensor in the converter to dodge the error.
That would have created a forever-mystery for the next reader. Instead we
documented the gap and stopped.

## Takeaways for someone new

1. **Lossless does not mean fast.** The whole point of this milestone is to
   preserve bytes. It does not make inference quicker or cheaper. Don't
   advertise it as such.
2. **Reader before writer.** When writing a serializer, keep the reader
   source open; if the reader can reject your output, let it tell you why.
3. **Hard integrity > soft integrity.** FNV-1a-64 over the whole file is
   cheap (seconds on 4.5 GB in C) and catches *every* byte flip. Per-tensor
   SHA-256 in the header is useful for diffing, not for integrity.
4. **"Minimum extension" is a real engineering answer.** If a loader refuses
   your valid input, name the one-line change that fixes it rather than
   either hiding the problem or writing a parallel loader.
5. **The oracle matters.** "The conversion produced a 4.5 GB file" is not
   proof. "The independent parity script re-hashes both files and reports
   632/632 byte-identical" is proof. Keep that script, keep its output.

## Exercise for the reader

Pick a 10-tensor subset of your own choice from the manifest JSON. Open the
original safetensors with `safe_open(...).get_tensor(name)`, copy the raw
bytes and SHA-256 them. Then open the `.tfold`, seek to the recorded
`offset`, read `nbytes`, and SHA-256 that. They must match. If they don't,
you have found the first real bug in the converter — write a failing test
before you touch the code.
