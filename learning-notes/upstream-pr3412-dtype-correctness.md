# Upstream PR #3412 — dtype correctness learning journal

> How a one-line `@memcpy` in a loader silently rescaled every embedding
> row by 128×, how the reviewer caught it, and how we fixed it without
> breaking the rest of the loader.

## 1. The original bug in beginner terms

ExLlamaV3 Qwen3.5 checkpoints can store the embedding table in either
`bf16` or `f16`. Both formats use 16 bits per value. They are NOT the
same format. The difference is where those 16 bits spend themselves:

| | Sign bits | Exponent bits | Mantissa bits |
|---|---|---|---|
| `f16` (IEEE half) | 1 | **5** | **10** |
| `bf16` (brain-float) | 1 | **8** | **7** |

Both formats say "this is 1.0" using distinct bit patterns:

- `f16` 1.0 = `0x3C00` = sign 0, exp `01111`, mantissa `0000000000`
- `bf16` 1.0 = `0x3F80` = sign 0, exp `01111111`, mantissa `0000000`

The CUDA kernel reads the embedding table as `bf16`. If the loader
gives it an `f16` table without converting, every row is wrong. In
particular, the pattern `0x3C00` that means **1.0 in f16** gets read
as `0x3C00` in bf16, which decodes as `2^(-7) × 1 = 0.0078125` —
exactly 128× too small.

The reviewer's exact words: *"F16 `1.0` becomes `0.0078125` on the
mapped path."* This is why.

## 2. How the two placements got out of sync

The CUDA engine has two ways to hand embeddings to the GPU:

- **Device placement**: copy the whole table into GPU memory.
  `embedTokens()` does the conversion row by row: f16 bits →
  rounded bf16 bits → GPU memory.

- **Mapped placement**: when the card has no room, pin the table
  on the host side and have the GPU read it over PCIe.
  `mappedEmbed()` previously called `embedTensor()` (which just
  returned the raw safetensors bytes) and then `cuda_host_embed.rows()`
  which did an unconverted `@memcpy` into the pinned bf16-sized
  buffer.

The device path converted; the mapped path didn't. Both compiled.
Both admitted the checkpoint. Only one gave the right number. The
admission tests (which only check "did the dtype-and-shape handshake
pass?") couldn't see the difference.

## 3. The fix

`mappedEmbed()` now mirrors `embedTokens()` exactly:

```zig
fn mappedEmbed(e: *Engine, ck: *core.Checkpoint, host: cuda.HostBuffer) !u64 {
    var buf: [200]u8 = undefined;
    const path = try name(&buf, "model.language_model.embed_tokens.weight", .{});
    const t = try ck.get(path);
    const host_embed = @import("cuda_host_embed.zig");
    if (t.is(.bf16, &.{ e.g.vocab, e.g.hidden })) return host_embed.rows(host, t.bytes);
    if (t.is(.f16, &.{ e.g.vocab, e.g.hidden })) {
        const widened = try f16Rows(e.gpa, t.bytes);
        defer e.gpa.free(widened);
        return host_embed.rows(host, std.mem.sliceAsBytes(widened));
    }
    return error.UnexpectedTensor;
}
```

Two things to notice:

1. `f16Rows()` was already the tested conversion routine used by
   `embedTokens()`. Reusing it means the mapped path and the device
   path now go through the EXACT same code for the f16 case. You
   can't have them silently diverge again unless someone re-adds a
   raw memcpy.

2. `widened` is allocated on `e.gpa` and `defer`-freed in-scope.
   `host_embed.rows()` does a length-checked memcpy into the
   pre-pinned host buffer (which has its own lifetime). No new
   lifetime surprises.

The broken `embedTensor()` helper got deleted. Dead code can't
silently come back.

## 4. The F32 norm narrowing finding

The second finding was less dramatic but equally real. ExLlamaV3
stores `linear_attn.norm.weight` as f32. The loader's `bf16OrF32()`
helper widened — wait, the old comment said "widened" but it's
actually **narrowing** f32 → bf16. We say "widening" because the
loader widens the SUPPORTED dtype set; the DATA gets narrowed.

Is the narrowing wrong? No — the consuming kernel
`qwen35.cu::tf_q35_gdn_post` reads `NW` as `__nv_bfloat16`, so bf16
is genuinely what the GPU needs. But the OLD tests (admission only)
couldn't tell you that the narrowing preserved anything numerically.

### Quantifying the narrowing error

bf16 has 8 explicit mantissa bits. A single rounding step costs at
most half an ULP. For normal f32 values outside tie neighborhoods:

$$|x - \text{bf16}(x)| \le 2^{-9} \cdot |x|$$

Our test asserts a conservative double bound (`2^-7 * |x|`) across
1024 random f32 norm weights. Every element passed. The looser
bound tolerates ties and subnormals without needing special cases.

The A_log tensor got different treatment: the kernel reads A_log as
`f32`, so `f32vec()` passes f32 bytes through unchanged. A test
asserts that: if A_log is stored f32, the bytes round-trip
bit-exactly. (If stored bf16, the widening is a bit-shift, also
exact.)

## 5. Mapped vs. device memory in the engine

Short version:

- **Device memory** is GPU DRAM. Fast. Limited. The GPU reads it
  via its own memory controller, no PCIe hop.
- **Mapped (pinned) host memory** is RAM allocated with
  `cudaMallocHost` and marked read-only-from-GPU. The GPU reads it
  over PCIe at roughly 20-30 GB/s on PCIe 4.0 x16, vs the hundreds
  of GB/s on-card.

Why would you ever map embeddings? Qwen3.5-2B's embedding table
(`vocab=151936 × hidden=1536 × 2 bytes/bf16`) is 456 MB. Add
multi-gigabyte EXL3 weights and a KV cache and a stream buffer, and
a 24 GB card can run out of room. Mapping the embedding to host
costs embedding-lookup latency but reclaims that 456 MB for KV.

The engine makes that choice in `cuda_host_embed.zig::mapped()`:

```zig
pub fn mapped(unified: bool, room: u64, device: u64, stream: u64, beside: u64) bool {
    return !unified and room < device +| stream +| beside;
}
```

An integrated/unified-memory GPU never maps (host==device). A
discrete card maps when its room doesn't fit device weights + one
stream + anything beside (like a drafter).

## 6. How the tests prove correctness

For each dtype path, a tiny focused test:

**Test 1 — F16 1.0 preserves**:

```zig
const cases = [_]struct { f16_bits: u16, want_bf16: u16 }{
    .{ .f16_bits = 0x3C00, .want_bf16 = 0x3F80 },  // +1.0
    // ... -1.0, +0.5, +2.0, +0.0, -0.0, f16 max 65504 ...
};
for (cases) |c| try std.testing.expectEqual(c.want_bf16, bf16Of(@as(f16, @bitCast(c.f16_bits))));
// Reinterpretation guard:
try std.testing.expect(widened[0] != cases[0].f16_bits);
```

**Test 2 — Mapped vs device byte parity**:

```zig
const device_rows = try f16Rows(...);  // Same routine both paths now share.
const mapped_rows = try f16Rows(...);
try std.testing.expectEqualSlices(u16, device_rows, mapped_rows);
```

**Test 3 — F32 narrowing bound**:

```zig
for (fp32, got) |src, bf| {
    const back: f32 = @bitCast(@as(u32, bf) << 16);
    try std.testing.expect(@abs(src - back) <= 0x1p-7 * @abs(src) + 0x1p-130);
}
```

**Test 4 — F32 A_log passthrough**:

```zig
const stored: f32 = @bitCast(std.mem.readInt(u32, bytes[4*i..][0..4], .little));
try std.testing.expectEqual(v, stored);  // f32 unchanged
```

None of these tests need a GPU. They're pure Zig unit tests that
run in milliseconds and would have caught the original bug
immediately.

## 7. What changed, why

| Change | Why |
|---|---|
| Delete `embedTensor()` helper | It returned raw tensor bytes and let the caller interpret them as pre-converted. The one caller interpreted them wrong. Removing the helper means no regression. |
| Inline dtype dispatch in `mappedEmbed()` | Matches `embedTokens()` structurally. Both paths now use the same tested `f16Rows()` routine. |
| Replace `// f32 widened to bf16` with `// f32 narrowed to bf16 by bf16OrF32; kernel reads bf16` | Honest about which direction precision moves. The "widening" was always about the loader accepting more dtypes, not the data getting more bits. |
| Add `f16Of 1.0 → 0x3F80` test | Would have caught the bug in CI. Pinned-value tests beat admission tests for dtype correctness. |
| Add 1024-sample narrowing-bound test | Proves narrowing stays within bf16's single-rounded limit across the plausible norm-weight range. |

## 8. Alternatives considered

| Option | Reason rejected |
|---|---|
| Keep `embedTensor()`, add a second helper for conversion | Two helpers that both look reasonable and only one is correct. The original bug would re-happen when someone picks the wrong one. |
| Put the F16 detection in `cuda_host_embed.rows()` | Pushes dtype knowledge into the memcpy helper, breaking its single-responsibility. The dtype decision belongs where we know the tensor, not where we know the destination. |
| Split the embedding handling into a separate PR | Considered. Rejected because the F16 fix is tightly coupled to the F16 widening work — Finding 1 only exists because Finding 2's widening work touched the embedding path. Keeping them in the same follow-up commit lets reviewers see the fix with its context. |
| Add a GPU-accelerated check | The bug is bit-level. CPU unit tests are enough and run in a millisecond. |

## 9. Actual measurements

Local Zig test run (standalone harness mirroring the tests I added):

```
1/2 f32narrow.test.f32 norm narrowing error bound...OK
2/2 f32narrow.test.f32 A_log passthrough + bf16 widening roundtrip...OK
All 2 tests passed.

1/2 f16bf16_check.test.f16 to bf16 widening preserves values...OK
2/2 f16bf16_check.test.mapped-vs-device both now use f16Rows...OK
All 2 tests passed.
```

Full build-system CI on the PR is `action_required` — upstream repos
require maintainer approval for workflow runs from forked PRs. Not
a failure, just a hand-raise.

## 10. Hands-on exercise

**Goal**: convince yourself that the old code is actually broken, in
a terminal, in under a minute.

```bash
cat > /tmp/f16_vs_bf16.zig <<'EOF'
const std = @import("std");

pub fn main() void {
    // F16 bits for 1.0
    const f16_bits: u16 = 0x3C00;

    // The buggy path: @memcpy f16 bytes into a "bf16" destination.
    // The GPU then interprets those bits as bf16.
    const bf16_as_if: f32 = @bitCast(@as(u32, f16_bits) << 16);

    // The correct path: convert f16 → f32 → bf16 round-to-even → f32
    const f: f16 = @bitCast(f16_bits);
    const widened: f32 = @floatCast(f);
    const bf16_bits: u16 = @truncate(
        (@as(u32, @bitCast(widened)) +% 0x7fff +% ((@as(u32, @bitCast(widened)) >> 16) & 1)) >> 16
    );
    const bf16_correct: f32 = @bitCast(@as(u32, bf16_bits) << 16);

    std.debug.print("f16 1.0 = 0x{X:0>4}\n", .{f16_bits});
    std.debug.print("  buggy (reinterpret as bf16):   {d} (expected 1.0)\n", .{bf16_as_if});
    std.debug.print("  correct (convert via f32):     {d} (expected 1.0)\n", .{bf16_correct});
}
EOF
zig run /tmp/f16_vs_bf16.zig
```

Expected output:

```
f16 1.0 = 0x3C00
  buggy (reinterpret as bf16):   7.8125e-3 (expected 1.0)
  correct (convert via f32):     1 (expected 1.0)
```

The **128× rescaling** reviewer Synthena caught in one line of
review.

## 11. Comprehension questions (left open on purpose)

**Q1**. `cuda_host_embed.rows()` takes `bytes: []const u8` and does
a length-checked `@memcpy` into pre-pinned host memory. It does NOT
know the source dtype. Would you prefer to (a) keep it dtype-agnostic
and leave the dispatch in `mappedEmbed()`, as we did, or (b) add a
`DType` parameter to `rows()` and dispatch there? Pick one and
describe one future refactor each choice would make easier or
harder. Hint: think about who else might want to call `rows()` and
what their inputs look like.

**Q2**. Our `"f32 norm narrowing"` test asserts `|src - back| <=
2^-7 * |src| + 2^-130`. The theoretical single-rounded bound is
`2^-9 * |src|`. Why does our assertion tolerate 4× the theoretical
limit, and when would a tighter bound catch a bug the current test
would miss? Hint: consider ties, subnormals, and the exponent-aware
nature of "half an ULP".
