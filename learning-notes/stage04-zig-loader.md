# Stage 4 — the native Zig loader (learning journal)

**For:** the project owner (you)
**Length:** ~25–30 min read
**Prerequisites:** Stage 3 retrospective (same folder); light familiarity
with programming concepts like "function" and "struct"; no prior Zig
knowledge assumed

---

## What is this document?

This is my working journal for Stage 4. It explains, from scratch,
every important decision I made while building the native Zig loader.
By the end you'll be able to:

- read Zig source without being lost
- explain why the loader uses "manual guard" memory management
- understand why the loader rejects a .tfold file rather than guessing
- describe the difference between IEEE-754 FP32 accumulation orders
- answer the two comprehension questions at the end
- run the hands-on exercise

Numbers and quotes are pulled from the real code; everything is
clickable in your editor.

---

## 1. What did we build?

Stage 3 gave us a Python program that opens `examples/demo.tfold` and
computes `y = x @ W + b`. Stage 4 does the same thing again, but in
Zig (a low-level systems language, cousin to C and Rust), running as
a compiled executable with no Python runtime involved.

The deliverable is:

- `zig/src/tfold/reader.zig` — opens and validates the file
- `zig/src/tfold/ops.zig` — does the math
- `zig/src/tfold/demo.zig` — command-line entry point
- `zig/src/tfold/tests.zig` — unit tests
- `zig/src/tfold/build.zig` — a tiny standalone build script

Build and run:

```
D:/op13-root/tools/zig-0.17.0/zig.exe build \
    --build-file zig/src/tfold/build.zig run-demo
```

Output (real, not invented):

```
file           : examples/demo.tfold
output dim     : 4
cold load+run  : 0.000783 s
warm median    : 0.000137 s  (n=10)
output         : [-3.5695248, -1.8333335, 0.1698398, 0.91303146]
```

Compare to Stage 3's Python result:

```
cold load+run  : 0.013364 s
warm median    : 0.003643 s
output         : [-3.5695247650146484, -1.8333334922790527,
                   0.1698397994041443, 0.9130314588546753]
```

Same answer. Native Zig is ~**25× faster** on warm runs. Same file,
same math, no magic.

---

## 2. Why bother writing it in Zig?

Three reasons, listed in the order they mattered:

1. **Native deployment target.** TensorFold eventually ships to a phone.
   You can't drop a Python interpreter on a locked-down Snapdragon
   app. You need a tiny, static, portable binary. Zig is what
   TensorFold already uses for that.
2. **Shared code path with the production engine.** TensorFold's main
   Zig tree (`zig/src/core/`) already has `checkpoint.zig` and
   `safetensors.zig` — the readers for other formats. Our .tfold
   reader lives in the same world so Stage 5+ can integrate cleanly.
3. **Zero dependencies.** The entire loader is standard-library only
   (`std`). No npm, no pip, no Cargo, no apt packages. Copy the four
   `.zig` files, point `zig build` at them, you have a loader.

None of this is a performance claim yet. 25× faster than Python is
unimpressive next to a real BLAS kernel. Stage 8 is where real speed
starts to matter.

---

## 3. A quick Zig primer

Just enough Zig to read the code.

**Function:**
```zig
pub fn matvecFp32(
    x: []const f32,          // immutable slice of f32
    w_flat: []const f32,
    K: usize,
    N: usize,
    out: []f32,              // mutable slice (writer puts the answer here)
) OpError!void {             // can error; otherwise returns nothing
    // ...
}
```

**Struct:**
```zig
pub const Tensor = struct {
    name: []const u8,        // UTF-8 text
    dtype: Dtype,            // enum
    shape: []const u64,
    offset: u64,
    nbytes: u64,
    sha256: [64]u8,          // fixed 64-byte array
};
```

**Error handling:**
```zig
const buf = try something(gpa, path);
// 'try' == "if an error, return it from THIS function"
// 'catch' == "if an error, do this instead"
```

**Defer & errdefer:**
```zig
const buf = try alloc(100);
defer free(buf);             // runs when scope ends, EVERY path
errdefer free(buf);          // runs only if we return an error
```

Those five patterns cover 90% of the Stage 4 code.

---

## 4. How the Zig loader parses the file

The on-disk format (locked in Stage 1, see `docs/tfold-format-v0.md`):

```
┌───────────────────────────────────────────────────┐
│ bytes  0..  8   magic         "TFOLD\0\0\0"       │
│ bytes  8.. 12   version       u32 LE  0x00010000  │
│ bytes 12.. 16   header_length u32 LE              │
│ bytes 16..N+16  header JSON   UTF-8               │
│  (align 64)                                       │
│ [tensor region] each tensor 64-byte aligned       │
│ last 8 bytes    FNV-1a-64 trailer                 │
└───────────────────────────────────────────────────┘
```

Byte-for-byte diagram as it would live in a real `.tfold` file:

```
offset  content                                       bytes
0       54 46 4F 4C 44 00 00 00                        magic
8       00 00 01 00                                    version (LE u32 → 0x00010000)
12      A0 03 00 00                                    header_length (LE u32 → 928)
16      7B 22 74 66 6F 6C 64 22 ...  (JSON text)      928 bytes
944     00 00 00 00 00 00 00 00 ...                   pad to 64-byte boundary
960     [weight tensor bytes]                          128 bytes (8×4 fp32)
1088    [bias tensor bytes]                            16 bytes (4 fp32)
1152    [input tensor bytes]                           32 bytes (8 fp32)
1184    [FNV-1a-64 trailer]                            8 bytes
```

The Zig loader walks this exact map in `reader.zig::openBytes`. For
every step, there's a spec reject code if validation fails:

| Step | Code checked | Reject code |
|---|---|---|
| 1 | File is at least 26 bytes long | `TFOLD_TOO_SMALL` |
| 2 | First 8 bytes equal `"TFOLD\0\0\0"` | `TFOLD_BAD_MAGIC` |
| 3 | Major version matches our reader | `TFOLD_VERSION_UNSUPPORTED` |
| 4 | Header length doesn't overflow past file end | `TFOLD_HEADER_OVERFLOW` |
| 5 | Header is valid JSON | `TFOLD_HEADER_PARSE` |
| 6 | Header matches our schema | `TFOLD_HEADER_SCHEMA` |
| 7 | FNV-1a-64 trailer matches recomputed value | `TFOLD_INTEGRITY` |
| 8 | Every tensor offset+nbytes stays in file | `TFOLD_TENSOR_OOB` |
| 9 | Every offset is on a 64-byte boundary | `TFOLD_TENSOR_ALIGN` |
| 10 | No two tensor ranges overlap | `TFOLD_TENSOR_OVERLAP` |
| 11 | No duplicate tensor names | `TFOLD_TENSOR_DUP_NAME` |
| 12 | arch/dtype/scheme/tokenizer_type are in our registries | `TFOLD_UNKNOWN_*` |
| 13 | source_sha256 matches the caller's pinned value | `TFOLD_CHECKPOINT_MISMATCH` |
| 14 | Every tensor's recomputed SHA-256 matches its header entry | `TFOLD_TENSOR_INTEGRITY` |

**The loader never executes bytes out of the file.** It reads them as
*data*. Spec §5 forbids embedded executable code; the loader has no
code path that could run them even if they were.

### A worked example: little-endian decoding

`bytes[8..12]` holds the version. The four bytes on disk are
`00 00 01 00`. We decode them as little-endian:

```
byte 8  = 0x00  → bits   0..7
byte 9  = 0x00  → bits   8..15
byte 10 = 0x01  → bits  16..23   ← this "1" is really 65,536
byte 11 = 0x00  → bits  24..31
→ u32 value = 0x00010000
```

The Zig code is one line:

```zig
const version = std.mem.readInt(u32, bytes[8..12], .little);
```

`.little` is the key part. On a big-endian machine, reading the same
bytes as a native `u32` would yield `0x00000100` (256) and you'd get
silently wrong answers. By specifying endianness, our loader produces
the same result on x86_64 (little), ARM64 (little by default), and any
hypothetical big-endian machine.

---

## 5. Memory ownership — the thing that crashed my tests

This is the hardest concept in low-level programming. You own some
memory; you have to free it exactly once; if you free it twice you
crash; if you don't free it, you leak. No garbage collector saves
you.

Zig gives you two patterns:

### 5.1 `defer` — "run this when the scope ends, every path"

```zig
const buf = try alloc(100);
defer free(buf);                // ALWAYS frees buf
// ... use buf ...
return buf[0];                  // free still runs
```

### 5.2 `errdefer` — "run this only if we return an error"

```zig
const buf = try alloc(100);
errdefer free(buf);             // frees only on error paths
return buf;                     // success → errdefer does NOT fire; caller now owns buf
```

### What went wrong in my first version

I had this pattern spread across 10 error paths:

```zig
if (bytes.len < min_size) {
    gpa.free(bytes);
    arena.deinit();
    return SpecError.TfoldTooSmall;
}
```

And **also** this at the top:

```zig
errdefer arena.deinit();
```

See the bug? On the `TfoldTooSmall` path, the explicit `arena.deinit()`
runs, THEN the `errdefer` runs it again. Double-free. Segfault. I
caught it when 4 of 14 tests crashed with:

```
Segmentation fault at address 0x1651cc60010
D:\...\lib\std\heap\ArenaAllocator.zig:58:22: in deinit
    it = node.next;
```

The fix: use a **single guard** that tracks ownership with nullable
pointers.

```zig
var owned_bytes: ?[]u8 = bytes;
var owned_arena: ?std.heap.ArenaAllocator = null;
defer {
    if (owned_bytes) |b| gpa.free(b);     // frees only if still owned
    if (owned_arena) |*a| a.deinit();
}

// ... validation steps that may return errors ...

// SUCCESS PATH: release ownership out of the guard
hdr.arena = owned_arena.?;
owned_arena = null;                        // guard won't free it
const out_bytes = owned_bytes.?;
owned_bytes = null;                        // guard won't free it
return File{ .bytes = out_bytes, .header = hdr };
```

Now there's exactly one cleanup site, and it uses the nullable state
to tell "do I still own this?". After the fix, all 14 tests passed:

```
Build Summary: 3/3 steps succeeded
test success
+- run test cached
   +- compile test debug native cached 21ms MaxRSS:18M
```

**Takeaway:** when writing low-level code, pick ONE pattern for
ownership and stick to it. Mixing `errdefer` with explicit cleanup is
how you get segfaults.

---

## 6. Alignment — why 64 bytes?

The spec says tensors must start on a 64-byte boundary. Why 64? Three
reasons:

1. **CPU vector instructions** read aligned chunks of 64 bytes at a
   time (AVX-512 on x86, SVE on ARM). Misaligned reads either slow
   down or trap, depending on the architecture.
2. **Cache line size.** Modern CPUs transfer 64 bytes per cache
   line. If a tensor crosses a cache-line boundary unnecessarily, you
   get an extra memory fetch for the same operation.
3. **One rule, many platforms.** Different vectorisation widths
   demand different alignments. Picking 64 covers all current CPUs
   with one number.

The Zig loader verifies this on every tensor:

```zig
if (t.offset % ALIGN != 0) return SpecError.TfoldTensorAlign;
```

If a file somehow contained a tensor at offset 300, this line rejects
the file with `TfoldTensorAlign`. The reader does not attempt to shift
the tensor to a valid address. **Wrong offset = wrong file = reject.**

---

## 7. Numerical precision — why the native output matches Python

The native output is `[-3.5695248, -1.8333335, 0.1698398, 0.91303146]`.
The Python output (Stage 3) was
`[-3.5695247650146484, -1.8333334922790527, 0.1698397994041443, 0.9130314588546753]`.

Those look different, but they're actually the same four FP32 numbers,
printed with different precisions. If you print them with the same
number of digits, they match byte-for-byte.

Why? Because both implementations do:

```
acc = 0.0 (as f64)
for k in 0..K:
    acc += x[k] * W[k * N + n]      # strict left-to-right
y[n] = cast_to_f32(acc)             # round to FP32 at the end
```

numpy, on the other hand, may use **pairwise summation** inside its
BLAS pathway — it can add groups of four at a time, which produces a
different (but equally valid) FP32 result. The 1-ULP discrepancy we
documented in Stage 3 came from that difference. Since the Zig loader
uses the Python-compatible order, Zig and Python agree bit-exactly;
both differ from numpy by up to 1 ULP.

**Documented tolerance** (from `ops.zig` test):

```zig
const tol: f32 = 1.0e-5 + 1.0e-5 * @abs(ref);
```

That's `atol = 1e-5, rtol = 1e-5` — the exact same tolerance Stage 3
declared. The test fails loudly if any element is outside that bound.

---

## 8. Hands-on exercise — your turn (10 minutes)

**Goal:** change one byte in `examples/demo.tfold` and watch the
loader refuse to open it.

1. Open a terminal in the TensorFold repo root.
2. Run the demo as-is to confirm it works:
   ```
   D:/op13-root/tools/zig-0.17.0/zig.exe build \
       --build-file zig/src/tfold/build.zig run-demo
   ```
   You should see the four-element output.
3. Make a corrupted copy of the demo file by changing the very first
   byte (which should be the letter 'T' — ASCII 0x54). In PowerShell:
   ```powershell
   Copy-Item examples/demo.tfold examples/bad.tfold
   $f = [IO.File]::OpenWrite("examples/bad.tfold")
   $f.WriteByte(0x58)  # 'X' instead of 'T'
   $f.Close()
   ```
4. Point the demo at the bad file:
   ```
   D:/op13-root/tools/zig-0.17.0/zig.exe build \
       --build-file zig/src/tfold/build.zig run-demo -- examples/bad.tfold
   ```
   (Note: the standalone build.zig doesn't currently pass CLI args
   through; you can also edit `demo.zig` line 68 to change the default
   path to `examples/bad.tfold`.)
5. Observe the error. The loader exits with `error.TfoldBadMagic`
   because the magic bytes no longer match. **No memory was touched,
   no tensor was decoded, nothing propagated downstream.**

What you just saw is the "fail closed" principle in action: a tiny
corruption at the very start of the file kills the load before any
weight data is touched. If this were a real model and the magic were
corrupted by disk rot, you'd rather know immediately than compute
subtly-wrong predictions for the next six months.

Restore the file by deleting `examples/bad.tfold`.

---

## 9. Alternatives I considered and rejected

### "Use `mmap` to avoid the full-file read"
**Decision:** no, not yet. mmap is faster for large files but adds
platform-specific code (different on Windows vs Linux vs macOS) and
test complexity. The demo file is 1,192 bytes — mmap and `read` are
indistinguishable at that size. Stage 5+ will reconsider when real
models arrive.

### "Share code between the Python and Zig loaders"
**Decision:** no. They're independent implementations of the same
spec on purpose. If we share code, a bug in one side masks the same
bug in the other. Independent implementations are the only honest way
to cross-check.

### "Ship a Zig dependency on a JSON schema library"
**Decision:** no. We hand-rolled the schema checks in
`parseAndValidate`. Sum total ~100 lines of straightforward code.
Adding a dependency to save 100 lines isn't worth the supply-chain
risk for a format we control.

### "Use `std.fs.cwd()` directly like old Zig code did"
**Decision:** had to switch. Zig 0.17.0 moved all file I/O under
`std.Io.Dir.cwd()` which requires an `Io` parameter threaded through.
I rewrote `openPath` to take `io: std.Io` and the test harness to
construct a `std.Io.Threaded` instance. More verbose, but matches
TensorFold's production pattern in `zig/src/cli/cli.zig`.

---

## 10. Comprehension check (answer yourself before peeking)

### Q1

> Our test `test_rejects_corrupted_trailer` flips one bit in the last
> 8 bytes of the file. Which reject code fires, and why does it fire
> there instead of at step 14 (per-tensor SHA)?

**Answer**

It fires at `TFOLD_INTEGRITY` (step 7). The last 8 bytes ARE the
FNV-1a-64 integrity trailer. When we recompute the trailer over the
body and compare, the comparison fails and the loader rejects before
ever reaching step 14. Step 14 would only fire if a byte *inside* a
tensor region had been altered AND the trailer had been recomputed to
match — which our other test (`test_rejects_per_tensor_SHA_mismatch`)
explicitly constructs.

### Q2

> If I write a tensor at offset 1024 and its `nbytes` is 128, is it
> on a 64-byte boundary? What about a tensor at offset 1024 with
> nbytes 100 — does that cause any problem for the NEXT tensor?

**Answer**

1024 is a multiple of 64 (1024 ÷ 64 = 16), so yes it's aligned.

If the first tensor's nbytes is 128, the next tensor must start at
offset 1152 (1024 + 128) or later. 1152 ÷ 64 = 18, so 1152 is
already aligned — no padding needed.

If the first tensor's nbytes is 100, the next byte is at offset 1124.
1124 ÷ 64 = 17.56, so we need to pad up to 1152 (28 bytes of
zero-fill) before the next tensor. The writer in `tools/tfold_py/_io.py`
does this automatically with `_pad_to(cur + nbytes, ALIGN)`; the Zig
loader verifies that whatever the writer did produced aligned offsets.

If the writer *forgot* to pad and wrote the next tensor at offset
1124, the Zig loader would reject with `TfoldTensorAlign`. The file
would not open. No tensor would be read. No weights would be processed.
This is the fail-closed principle — a broken file never gets a
chance to produce a wrong answer.

---

## 11. What's next (Stage 5)

Stage 5 is the real-model converter: take a Qwen3.5-2B-exl3 checkpoint
and produce a `.tfold` version of it. That means hundreds of tensors
instead of three, hundreds of megabytes instead of 1 KB, and the
EXL3 packed-quant format instead of plain FP32. The loader stays
unchanged — but it finally gets exercised with something that looks
like a real model.

When Stage 5 lands, this document stays. You can come back here to
remember what we did and why. If you want to make a change to the
format, re-read it, then re-read `docs/tfold-format-v0.md`, then be
ready for the compatibility conversation.

---

## 12. Files to look at

| File | What's in it |
|---|---|
| `zig/src/tfold/reader.zig` | binary parse + spec §4 validation + integrity check |
| `zig/src/tfold/ops.zig` | fp32 matvec, bias_add, little-endian float decoder |
| `zig/src/tfold/demo.zig` | command-line demo entry point |
| `zig/src/tfold/tests.zig` | 14 unit tests (positive + negative) |
| `zig/src/tfold/build.zig` | standalone build invocation |
| `docs/tfold-format-v0.md` | the authoritative on-disk spec |
| `docs/tfold-stage4-conformance.md` | Stage 4's formal pass report |
| `learning-notes/stage03-cpu-reference.md` | the retrospective for Stage 3 |

Open `reader.zig` next to the step table in §4 above. Read line by
line. The code is written to map one-to-one to the spec, so if you
can follow the spec, you can follow the code.
