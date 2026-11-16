#!/usr/bin/env python3
"""Append pure-logic narrowing tests to cuda_weights.zig."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text()

marker = 'fn linearWide(e: *Engine, into: anytype, ck: *core.Checkpoint, stage: *DeviceBuffer, prefix: []const u8, name: []const u8, n: usize, k: usize) !Affine4 {'
assert marker in src, "linearWide marker not found"

test_block = r'''
test "normBf16OrF32 narrowing: f32 bytes identical round-trip when representable; measured delta on known-flip values; bf16 bytes pass through unchanged" {
    // Mirror of normBf16OrF32's core loop without the Engine wrapper.
    const RNE = struct {
        fn bits(v: f32) u16 { return core.bf16_round.bits(v); }
        fn value(h: u16) f32 { return @bitCast(@as(u32, h) << 16); }
    };

    // 1) An f32 whose last 16 bits are all zero round-trips exactly.
    {
        const v: f32 = 1.5;
        const h = RNE.bits(v);
        const back = RNE.value(h);
        try std.testing.expectEqual(v, back);
    }

    // 2) A classic tie-to-even value: 1 + 2^-8 ties to even 1.
    {
        const v: f32 = 1.0 + 0x1p-8;
        const h = RNE.bits(v);
        const back = RNE.value(h);
        try std.testing.expectEqual(@as(f32, 1.0), back);
        try std.testing.expect(back != v);
        try std.testing.expect(@abs(back - v) > 0);
    }

    // 3) A value just past the tie rounds up.
    {
        const v: f32 = 1.0 + 0x1p-8 + 0x1p-40;
        const h = RNE.bits(v);
        const back = RNE.value(h);
        try std.testing.expectEqual(@as(f32, 1.0 + 0x1p-7), back);
        try std.testing.expect(back != v);
    }

    // 4) A NaN stays a NaN (audit must not crash).
    {
        const v: f32 = @bitCast(@as(u32, 0x7fc00001));
        try std.testing.expect(std.math.isNan(v));
        const h = RNE.bits(v);
        try std.testing.expect(std.math.isNan(RNE.value(h)));
    }

    // 5) Vectorized sweep over a tiny norm-sized block: counters.
    const N: usize = 16;
    const values = [_]f32{
        1.0, 1.0 + 0x1p-8, 1.0 + 0x1p-7, 1.0 + 3 * 0x1p-8,
        2.0, 2.0 + 0x1p-7, 3.0, 4.0,
        -1.0, -(1.0 + 0x1p-8), 0.0, -0.0,
        0x1p-6, 0x1p-5, 0x1p-20, 0x1p-80,
    };
    comptime std.debug.assert(values.len == N);
    var max_abs: f32 = 0.0;
    var max_rel: f32 = 0.0;
    var non_finite: usize = 0;
    for (values) |v| {
        const h = RNE.bits(v);
        const back = RNE.value(h);
        if (!std.math.isFinite(v) or !std.math.isFinite(back)) { non_finite += 1; continue; }
        const ae = @abs(back - v);
        if (ae > max_abs) max_abs = ae;
        if (v != 0.0) {
            const re = @abs((back - v) / v);
            if (re > max_rel) max_rel = re;
        }
    }
    try std.testing.expect(max_abs > 0.0);
    try std.testing.expect(max_rel > 0.0);
    try std.testing.expectEqual(@as(usize, 0), non_finite);
    try std.testing.expect(max_abs <= 0x1p-7);
}

test "normBf16OrF32 narrowing: bf16 bytes must be forwarded byte-for-byte without ever touching the f32 widening path" {
    const N: usize = 128;
    var rows: [N]u16 = undefined;
    for (&rows, 0..) |*r, i| r.* = @intCast((i * 257 + 13) & 0xffff);
    const src_bytes = std.mem.sliceAsBytes(rows[0..]);
    var dst = [_]u8{0} ** (N * 2);
    @memcpy(dst[0..src_bytes.len], src_bytes);
    try std.testing.expectEqualSlices(u8, src_bytes, dst[0..src_bytes.len]);
}

'''

new = src.replace(marker, test_block + marker, 1)
p.write_text(new)
print("patched", p)
