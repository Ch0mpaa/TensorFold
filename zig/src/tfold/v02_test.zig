//! Stage 6b Gate 3 — Zig-side v0.2 reader tests.

const std = @import("std");
const testing = std.testing;
const reader = @import("reader.zig");
const v02 = @import("reader_v02.zig");

fn makeIo(gpa: std.mem.Allocator) !std.Io.Threaded {
    return std.Io.Threaded.init(gpa, .{});
}

const V02_FIXTURE = "examples/v02-mixed-widths-example.tfold";
test "v0.2 opens with Zig reader when Python wrote it" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const probe = std.Io.Dir.cwd().readFileAlloc(io, V02_FIXTURE, gpa, .limited(1 << 22)) catch return error.SkipZigTest;
    gpa.free(probe);

    var f = try v02.openPath(gpa, io, V02_FIXTURE, .{});
    defer f.close();
    try testing.expect(f.overrides_present);
}

test "v0.2 Zig reader rejects v0.1 bytes" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const v01_path = "examples/demo.tfold";
    const probe = std.Io.Dir.cwd().readFileAlloc(io, v01_path, gpa, .limited(1 << 22)) catch return error.SkipZigTest;
    gpa.free(probe);

    const result = v02.openPath(gpa, io, v01_path, .{});
    try testing.expectError(reader.SpecError.TfoldBadMagic, result);
}

test "v0.1 Zig reader rejects v0.2 bytes" {
    const gpa = std.testing.allocator;
    var threaded = try makeIo(gpa);
    defer threaded.deinit();
    const io = threaded.io();

    const probe = std.Io.Dir.cwd().readFileAlloc(io, V02_FIXTURE, gpa, .limited(1 << 22)) catch return error.SkipZigTest;
    gpa.free(probe);

    const result = reader.openPath(gpa, io, V02_FIXTURE, .{});
    try testing.expectError(reader.SpecError.TfoldBadMagic, result);
}

test "v0.2 recognizes new scheme names" {
    try testing.expect(v02.schemeIsKnownV02("exl3_mcg"));
    try testing.expect(v02.schemeIsKnownV02("exl3_mul1"));
    try testing.expect(v02.schemeIsKnownV02("exl3_3inst"));
    try testing.expect(!v02.schemeIsKnownV02("exl3_unknown"));
}

test "v0.2 recognizes new dtype names" {
    try testing.expect(v02.dtypeIsKnownV02("exl3_mcg_tile"));
    try testing.expect(v02.dtypeIsKnownV02("exl3_mul1_tile"));
    try testing.expect(v02.dtypeIsKnownV02("exl3_3inst_tile"));
    try testing.expect(!v02.dtypeIsKnownV02("exl3_unknown_tile"));
}
