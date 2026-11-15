//! Zig unit tests for tfold v0.1 reader and ops.
//!
//! Positive tests load examples/demo.tfold and check the known 4-float
//! output. Negative tests craft malformed files on the fly and verify
//! each reject code fires.
//!
//! Tests that need the fixture skip themselves with SkipZigTest when
//! the demo file is missing.

const std = @import("std");
const testing = std.testing;
const tfold = @import("tfold.zig");
const reader = @import("reader.zig");
const ops = @import("ops.zig");
const demo = @import("demo.zig");

const DEMO_PATH = "examples/demo.tfold";
const DEMO_PIN: []const u8 = "8a47dd362d042c48d5fbc17d217c7e143c6b8bedd7c3905d522993de3400d3d3";
const BAD_PIN: []const u8 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

const DEMO_EXPECTED: [4]f32 = .{
    -3.5695247650146484,
    -1.8333334922790527,
    0.1698397994041443,
    0.9130314588546753,
};

fn makeIo(gpa: std.mem.Allocator) !std.Io.Threaded {
    return std.Io.Threaded.init(gpa, .{});
}

fn readDemo(gpa: std.mem.Allocator, io: std.Io) ?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, DEMO_PATH, gpa, .limited(1 << 24)) catch return null;
}

test "demo.tfold opens and produces expected 4 floats" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    // Probe: skip if the fixture isn't present
    const probe = readDemo(gpa, io) orelse return error.SkipZigTest;
    gpa.free(probe);

    const y = try demo.executeDemo(gpa, io, DEMO_PATH, DEMO_PIN);
    defer gpa.free(y);
    try testing.expectEqual(@as(usize, 4), y.len);
    for (y, 0..) |v, i| {
        const ref = DEMO_EXPECTED[i];
        const diff = @abs(v - ref);
        const tol: f32 = 1.0e-5 + 1.0e-5 * @abs(ref);
        if (diff > tol) {
            std.debug.print("element {d}: got {d}, expected {d}, |diff|={d}, tol={d}\n", .{ i, v, ref, diff, tol });
            return error.TestExpectedWithinTolerance;
        }
    }
}

test "demo.tfold rejects wrong pinned checkpoint" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const probe = readDemo(gpa, io) orelse return error.SkipZigTest;
    gpa.free(probe);

    const result = demo.executeDemo(gpa, io, DEMO_PATH, BAD_PIN);
    try testing.expectError(reader.SpecError.TfoldCheckpointMismatch, result);
}

test "rejects too-small file" {
    const gpa = std.testing.allocator;
    const buf = try gpa.alloc(u8, 10);
    const got = reader.openBytes(gpa, buf, .{});
    try testing.expectError(reader.SpecError.TfoldTooSmall, got);
}

test "rejects bad magic" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const buf = readDemo(gpa, io) orelse return error.SkipZigTest;
    buf[0] = 'X';
    const got = reader.openBytes(gpa, buf, .{});
    try testing.expectError(reader.SpecError.TfoldBadMagic, got);
}

test "rejects unsupported major version" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const buf = readDemo(gpa, io) orelse return error.SkipZigTest;
    std.mem.writeInt(u32, buf[8..12], 0x0002_0000, .little);
    const got = reader.openBytes(gpa, buf, .{});
    try testing.expectError(reader.SpecError.TfoldVersionUnsupported, got);
}

test "rejects header_length overflow" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const buf = readDemo(gpa, io) orelse return error.SkipZigTest;
    std.mem.writeInt(u32, buf[12..16], 0xFFFF_FFFF, .little);
    const got = reader.openBytes(gpa, buf, .{});
    try testing.expectError(reader.SpecError.TfoldHeaderOverflow, got);
}

test "rejects corrupted trailer" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const buf = readDemo(gpa, io) orelse return error.SkipZigTest;
    buf[buf.len - 1] ^= 0x80;
    const got = reader.openBytes(gpa, buf, .{});
    try testing.expectError(reader.SpecError.TfoldIntegrity, got);
}

test "rejects corrupted tensor bytes via trailer" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const buf = readDemo(gpa, io) orelse return error.SkipZigTest;
    buf[buf.len - 16] ^= 0xFF;
    const got = reader.openBytes(gpa, buf, .{});
    try testing.expectError(reader.SpecError.TfoldIntegrity, got);
}

test "rejects per-tensor SHA mismatch after trailer repair" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const buf = readDemo(gpa, io) orelse return error.SkipZigTest;

    const hlen: usize = @intCast(std.mem.readInt(u32, buf[12..16], .little));
    var tensor_start: usize = reader.PREAMBLE_NBYTES + hlen;
    const pad = (reader.ALIGN - (tensor_start % reader.ALIGN)) % reader.ALIGN;
    tensor_start += pad;
    buf[tensor_start + 7] ^= 0xFF;
    const body = buf[0 .. buf.len - reader.TRAILER_NBYTES];
    const new_fnv = reader.fnv1a_64(body);
    std.mem.writeInt(u64, buf[buf.len - reader.TRAILER_NBYTES ..][0..8], new_fnv, .little);

    const got = reader.openBytes(gpa, buf, .{});
    try testing.expectError(reader.SpecError.TfoldTensorIntegrity, got);
}

test "matvec: 2x3 example" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 1.0, 2.0 };
    const W = [_]f32{ 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const out = try gpa.alloc(f32, 3);
    defer gpa.free(out);
    try ops.matvecFp32(&x, &W, 2, 3, out);
    try testing.expectEqual(@as(f32, 1 * 3 + 2 * 6), out[0]);
    try testing.expectEqual(@as(f32, 1 * 4 + 2 * 7), out[1]);
    try testing.expectEqual(@as(f32, 1 * 5 + 2 * 8), out[2]);
}

test "matvec: rejects shape mismatch" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 1.0, 2.0 };
    const W = [_]f32{ 3.0, 4.0, 5.0 };
    const out = try gpa.alloc(f32, 3);
    defer gpa.free(out);
    const got = ops.matvecFp32(&x, &W, 2, 3, out);
    try testing.expectError(ops.OpError.ShapeMismatch, got);
}

test "matvec: rejects NaN input" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 1.0, std.math.nan(f32) };
    const W = [_]f32{ 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const out = try gpa.alloc(f32, 3);
    defer gpa.free(out);
    const got = ops.matvecFp32(&x, &W, 2, 3, out);
    try testing.expectError(ops.OpError.NonFiniteInput, got);
}

test "bias_add: happy path" {
    const gpa = std.testing.allocator;
    const y = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 0.5, -1.0, 10.0 };
    const out = try gpa.alloc(f32, 3);
    defer gpa.free(out);
    try ops.biasAddFp32(&y, &b, out);
    try testing.expectEqual(@as(f32, 1.5), out[0]);
    try testing.expectEqual(@as(f32, 1.0), out[1]);
    try testing.expectEqual(@as(f32, 13.0), out[2]);
}

test "bias_add: rejects length mismatch" {
    const gpa = std.testing.allocator;
    const y = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{ 0.5 };
    const out = try gpa.alloc(f32, 2);
    defer gpa.free(out);
    const got = ops.biasAddFp32(&y, &b, out);
    try testing.expectError(ops.OpError.ShapeMismatch, got);
}
