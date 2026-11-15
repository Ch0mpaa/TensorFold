# Stage 5 — real EXL3 conversion (learning journal)

**For:** the project owner (you)
**Length:** ~20 min read
**Prerequisites:** Stages 3 and 4 journals (same folder). Comfort with
Stage 2's `.tfold` file concept. No prior quantization experience.

---

## 1. What this stage actually did

For the first four stages the `.tfold` format carried toy data: eight
random FP32 numbers in a matrix. Useful for proving the plumbing, but
the real world stores weights **quantized** — compressed to 8 bits, or
4, or even fewer, per weight. Stage 5 proves `.tfold` can carry a
**real EXL3 checkpoint block** without touching a single bit of the
original compressed data.

What shipped:

- `tools/tfold_convert/` — Python package that reads the committed
  EXL3 oracle fixture and writes it into a valid `.tfold` v0.1 file.
- `examples/oracle-mcg-128x128.tfold` — the real converted file,
  18,312 bytes, checked into the repo.
- 12 new Python tests + 2 new Zig tests, all passing.
- An honest note documenting that **one** v0.1 file holds **one** EXL3
  group; a full multi-layer checkpoint with mixed widths is a v0.2
  conversation.

---

## 2. Quantization vs packaging — the crucial distinction

There are two completely separate concerns and people mix them up all
the time:

| Concern | Who owns it | Example |
|---|---|---|
| **Quantization** — how a single number is encoded | the quantizer (EXL3 algorithm) | 8-bit value → `0x3B` |
| **Packaging** — how a whole model is stored on disk | the file format | safetensors / GGUF / .tfold |

Think of quantization as "how a number shows up as a shrunken sticker"
and packaging as "how all those stickers get organized into a shoebox
with labels". `.tfold` is the shoebox. EXL3 is the sticker design. Our
Stage 5 job was:

> Pick up the EXL3 stickers from one specific checkpoint, put them
> into the `.tfold` shoebox without altering a single sticker, add a
> label that says what format they are, and prove any reader can take
> them back out and recognize them.

**No re-quantization.** If we decoded the EXL3 stickers and re-encoded
them, we'd have a new set of stickers — they'd be close to the originals
but almost certainly not bit-identical. The point of Stage 5 is
*lossless container conversion*: in, out, same bits.

---

## 3. What EXL3 actually looks like

EXL3 ("ExLlamaV3's trellis quantization") is what many community
low-bit NVIDIA checkpoints ship as. Full technical description lives in
`docs/recipes/exl3.md`. The short version:

A quantized linear layer with K inputs and N outputs is stored as a
small group of tensors:

```
trellis  int16 [K/16, N/16, 16 * bits]      ← the packed codes
suh      fp16  [K]                          ← per-input scales
svh      fp16  [N]                          ← per-output scales
bias     fp16  [N]                          ← optional, not in our oracle
```

plus a zero-size marker tensor (or a quant-block field in our case)
saying which **codebook** was used: `3inst`, `mcg`, or `mul1`.

### Trellis: not a lookup table

"Codebook" usually makes people think of a table: "state 0x1234 maps
to value 0.5718", "state 0x5678 maps to -2.1409", etc. EXL3 **isn't** a
table. The codebook is a tiny arithmetic recipe that converts a 16-bit
state `s` into an FP16 value:

```
mcg:    x = s * 0xCBAC1FED mod 2^32
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        value = fp16(x & 0xFFFF) + fp16(x >> 16)   (one rounded fp16 add)
```

That's it. No table lookup. The multiplier `0xCBAC1FED` IS the entire
codebook for mcg. If you know the multiplier and the arithmetic, you
can decode any state.

### Packed bits per tile

A "tile" holds 256 weights. For 8-bit mcg, each tile is
`16 * 8 = 128` int16 words = 256 bytes = 2,048 bits = 256 states of
8 bits each (actually, states are pulled from a *circular bitstream*
that wraps around, but the count works out).

Our 128×128 oracle block has 128/16 = 8 tile rows × 8 tile cols = 64
tiles. 64 × 256 bytes = 16,384 bytes of packed trellis. That's the
`len(g.trellis_i16)` number in the smoke test output.

---

## 4. The fixture we used

Stage 5's input is already in the TensorFold repo at
`zig/tests/fixtures/exl3_oracle.bin`. It's **371,478 bytes**, with
SHA-256
`8eecb6ac56325a50634eaf75cd2c2ff4aed74e4341b5f2af6c05a3bf0985c810`.

Provenance (in `tools/tfold_convert/oracle.py`):

- **Source model:** `turboderp/Qwen3.5-2B-exl3@8.00bpw/model.safetensors`
- **Source SHA-256:** `a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f`
- **License:** Apache-2.0 (base Qwen weights), EXL3 encoding by turboderp
- **What we extracted:** the publicly-verifiable 128×128 mcg
  8.00-bpw block carried in the oracle file as its final record

The fixture also contains an FP64 reference dequantization of that
same block — precisely so an independent implementation can verify its
own decoder. Stage 5 doesn't need to decode the weights; it just
carries the packed bits across. But the FP64 reference is right there
if you ever want to check.

**Why this fixture and not a HuggingFace download?** Three reasons:

1. **Determinism.** Checked-in bytes with a pinned SHA-256 can't drift.
2. **Size.** 371 KB vs ~4 GB for a full model. Fast, bounded, obvious.
3. **Legal clarity.** This single block has been community-tested as
   the reference oracle for EXL3 and is appropriate to redistribute.

---

## 5. A worked numerical example

Let's decode a single state the way the Zig decoder does. Picking
state `s = 0x1234` with the mcg codebook:

```
Step 1: x = s * 0xCBAC1FED  (mod 2^32)
         = 0x1234 * 0xCBAC1FED
         = 0xE8A2F4DC0D54  (truncate low 32 bits)
         → x = 0xF4DC0D54

Step 2: x = (x & 0x8FFF8FFF) ^ 0x3B603B60
         = (0xF4DC0D54 & 0x8FFF8FFF) ^ 0x3B603B60
         = 0x84DC0D54 ^ 0x3B603B60
         = 0xBFBC3634

Step 3: value = fp16(0x3634) + fp16(0xBFBC)
         fp16(0x3634) = 0.388...  (sign=0, exp=13, mantissa=1588/1024 ≈ 1.551)
         fp16(0xBFBC) = -1.934...  (sign=1, exp=14, mantissa=1980/1024 ≈ 1.933)
         sum = -1.546  → rounded to nearest-even fp16 → value
```

**Try this in your head, then run it:** `zig build -Dtarget=native
--build-file zig/src/tfold/build.zig test` prints the full oracle
battery and confirms the Zig decoder reproduces the FP64 reference
across all 27 codebook-and-width combinations.

**The important thing:** our Stage 5 converter never touched any of
this math. The 16,384 bytes of trellis went in one side and out the
other, bit for bit. The decoder only runs when someone later wants to
read the weights; we didn't need it to do the conversion.

---

## 6. v0.1 format limits (honest disclosure)

While building Stage 5 we discovered what v0.1 can and can't express:

| Thing | v0.1 can store | Needs v0.2 |
|---|---|---|
| Single EXL3 group (our oracle) | ✅ one scheme, one width | — |
| Multi-tensor weights with different widths | ❌ | **yes** — needs per-tensor `quant` overrides |
| Mixed codebooks in one file | ❌ | **yes** — needs per-group scheme tag |
| Routed experts (MoE) | partial (if all share scheme/width) | mostly yes |
| Codebook tables for custom quantizers | no table support | **yes** — needs `__codebook__` as a variable-size tensor with table lookup |

I documented this in `docs/tfold-stage5-conformance.md § v0.1 limits`.
**I did not change the spec.** The temptation when hitting a limit is
to patch the spec to make the current problem go away. Instead we
flagged what's missing and left it as a v0.2 proposal that will land
only when there's real pressure to need it.

Our Stage 5 goal was to prove the format works for one real group.
That bar is cleared. A full multi-layer Qwen3.5-2B conversion will
land in v0.2.

---

## 7. How the converter works, in code

`tools/tfold_convert/convert.py::convert_group_to_tfold` is 50 lines
of straightforward Python:

```python
def convert_group_to_tfold(group, out_path, ...):
    if group.codebook != "mcg":
        raise ValueError(...)  # v0.1 scope limit, surfaced loudly

    tensors_list = [
        {"name": "__codebook__", "dtype": "u32", "shape": [1], ...},
        {"name": "trellis", "dtype": "exl3_mcg_tile", "shape": [...], ...},
        {"name": "suh", "dtype": "fp16", "shape": [K], ...},
        {"name": "svh", "dtype": "fp16", "shape": [N], ...},
    ]
    buffers = {
        "__codebook__": np.frombuffer(struct.pack("<I", 0xCBAC1FED), ...),
        "trellis": group.trellis_i16,         # bytes, preserved as-is
        "suh": np.frombuffer(group.suh_fp16, dtype=np.dtype("<f2")),
        "svh": np.frombuffer(group.svh_fp16, dtype=np.dtype("<f2")),
    }

    header = {
        "tfold": {"version": "0.1", ...},
        "model": {"arch": "qwen3", "source_sha256": MODEL_SHA256, ...},
        "tokenizer": {...},
        "quant": {"scheme": "exl3_mcg", "bits_per_weight": 8.0, "block_size": 256},
        "tensors": tensors_list,
    }

    return tfold_py.write(out_path, header, buffers)
```

Note the critical lines:

- `"trellis": group.trellis_i16` — the writer accepts raw bytes for
  the packed `exl3_mcg_tile` dtype. **No numpy roundtrip for the
  packed data.** Bits in, bits out.
- `struct.pack("<I", 0xCBAC1FED)` — the mcg "codebook" is exactly
  four bytes: the multiplier. Embedding it as a `__codebook__` tensor
  satisfies the v0.1 schema's "exl3_* requires codebook_sha256" rule.
- `scheme="exl3_mcg"` + `bits_per_weight=8.0` captures the entire
  quantization identity. Any reader seeing this file knows it's
  mcg at 8 bpw.

---

## 8. Hands-on exercise — 15 minutes

**Goal:** prove to yourself that the trellis bytes are truly preserved.

1. From the repo root, generate the converted file and dump both the
   source bytes and the converted bytes:
   ```powershell
   cd D:/op13-root/src/tensorfold
   $env:PYTHONPATH="tools"
   D:/op13-root/tools/zig-0.17.0/zig.exe build --build-file zig/src/tfold/build.zig test
   ```
2. In a Python REPL:
   ```python
   import sys; sys.path.insert(0, "tools")
   import tfold_convert as tc
   g = tc.load_oracle_mcg_block()
   back = tc.read_converted_group_bytes("examples/oracle-mcg-128x128.tfold")

   # Compare byte-for-byte
   print("trellis bytes identical:", back["trellis"] == g.trellis_i16)
   print("suh      bytes identical:", back["suh"] == g.suh_fp16)
   print("svh      bytes identical:", back["svh"] == g.svh_fp16)

   # Now hash both to prove it
   import hashlib
   print("orig trellis SHA:", hashlib.sha256(g.trellis_i16).hexdigest()[:16], "...")
   print("back trellis SHA:", hashlib.sha256(back["trellis"]).hexdigest()[:16], "...")
   ```

   Both prints should say `True`. Both SHA-256 prefixes should match.
3. **Bonus:** open `examples/oracle-mcg-128x128.tfold` in a hex editor
   and spot the `TFOLD` magic at offset 0, then search for the `mcg`
   multiplier bytes `ED 1F AC CB` (little-endian `0xCBAC1FED`) at
   roughly offset 1,344 — that's the embedded `__codebook__` marker.

What you've just verified is the Stage 5 one-sentence promise: the
original EXL3 bytes come back out of the `.tfold` container exactly
as they went in.

---

## 9. Comprehension check

### Q1

> Our converter refuses to run if the committed fixture's SHA-256
> doesn't match the pinned constant `FIXTURE_SHA256`. Why would we
> gate on the fixture's bytes rather than the model source URL?

**Answer**

URLs can lie. Someone could rename a HuggingFace repo, replace a file
in place, or publish a "Qwen3.5-2B-exl3" model that's silently a
different quantization. If we trusted the URL, we'd convert whatever
bytes we got and ship a file claiming it carried mcg 8-bpw data when
the source might have been something else. The SHA-256 is the bytes'
true identity. Pinning it means: this converter guarantees the output
file carries THESE exact bytes, repeatable across anyone running the
code, independent of what name the bytes go by online.

This is the same checkpoint-pinning discipline the QNN qualification
work established (see `docs/qualcomm-android.md`). We carry it
through here so that any future audit of the converted `.tfold` can
trace back to one specific set of bytes.

### Q2

> Our .tfold file lists `__codebook__` as a 4-byte tensor containing
> `0xCBAC1FED` in little-endian. If someone wrote a converter that
> embedded the mcg algorithm name as the string `"mcg"` (3 bytes)
> instead of the multiplier, would downstream readers still work?

**Answer**

**Depends on what the readers check.** The Stage 2 schema would
validate either as a `u8` tensor (since `"mcg"` is 3 ASCII bytes).
The Stage 4 Zig reader that our Stage 5 Zig test implements
explicitly checks that `__codebook__` is a `u32` tensor containing
`0xCBAC1FED` — because the reader uses the *value* as a contract.
A reader that just trusted the `quant.scheme` string without
verifying `__codebook__` wouldn't notice, but ours does.

**This is on purpose.** If a future reader optimizes using the mcg
multiplier (say, pre-computing the quantization lookup tables at
startup), it needs an unambiguous signal that the file really uses
that multiplier. A human-readable string "mcg" is easy to change by
accident; the raw 4-byte multiplier is the actual mathematical
invariant. Hard to confuse.

The lesson: when a format has a choice between "this is the name of
the thing" and "this is the arithmetic that defines the thing",
prefer the arithmetic. Names drift; math doesn't.

---

## 10. What Stage 5 is NOT

- Not a full-model conversion. See the v0.1 limits table in §6.
- Not a performance benchmark. We measured byte-for-byte fidelity,
  not inference speed.
- Not an inference engine. The decoder exists; the engine that would
  use it ships in later stages.
- Not a quantization library. We did zero quantization math; the bits
  came in from the fixture and went out unchanged.

---

## 11. Files to look at

| File | What it does |
|---|---|
| `tools/tfold_convert/__init__.py` | public API |
| `tools/tfold_convert/oracle.py` | reads the committed EXL3 fixture, pins SHA-256 |
| `tools/tfold_convert/convert.py` | writes the EXL3 group into `.tfold` v0.1 |
| `tools/tfold_convert/_smoke.py` | the smoke script shown in §8 |
| `examples/oracle-mcg-128x128.tfold` | the actual 18,312-byte converted file |
| `tests/test_convert_exl3.py` | 12 pytest tests (positive + negative) |
| `zig/src/tfold/exl3_test.zig` | 2 Zig tests that load the converted file natively |
| `docs/tfold-stage5-conformance.md` | one-page formal report |
| `docs/recipes/exl3.md` | the authoritative EXL3 format description |
