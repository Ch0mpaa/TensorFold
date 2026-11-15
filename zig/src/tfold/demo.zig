//! tfold-cpu-demo: open examples/demo.tfold, run y = x @ W + b in
//! native Zig, print the result and timing.
//!
//! Usage:
//!   tfold-cpu-demo [PATH] [--pin HEX64] [--repeat N] [--json]

const std = @import("std");
const tfold = @import("tfold.zig");

const DEMO_PIN: []const u8 = "8a47dd362d042c48d5fbc17d217c7e143c6b8bedd7c3905d522993de3400d3d3";

fn usage(err_w: *std.Io.Writer) !void {
    try err_w.writeAll("usage: tfold-cpu-demo [PATH] [--pin HEX64] [--repeat N] [--json]\n");
}

fn elapsedNs(a: std.Io.Clock.Timestamp, b: std.Io.Clock.Timestamp) u64 {
    const dur = a.durationTo(b);
    const ns = dur.raw.toNanoseconds();
    return if (ns < 0) 0 else @intCast(ns);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;

    var out_buf: [4096]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &out_buf);
    var err = std.Io.File.stderr().writerStreaming(io, &err_buf);
    defer out.interface.flush() catch {};
    defer err.interface.flush() catch {};
    const w = &out.interface;
    const ew = &err.interface;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer iter.deinit();
    _ = iter.next(); // argv0

    var path_buf: ?[]u8 = null;
    defer if (path_buf) |p| gpa.free(p);
    var pin: []const u8 = DEMO_PIN;
    var repeat: usize = 10;
    var json_out: bool = false;

    while (iter.next()) |raw| {
        if (std.mem.eql(u8, raw, "--pin")) {
            const v = iter.next() orelse { try usage(ew); return 2; };
            pin = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, raw, "--repeat")) {
            const v = iter.next() orelse { try usage(ew); return 2; };
            repeat = std.fmt.parseInt(usize, v, 10) catch { try usage(ew); return 2; };
        } else if (std.mem.eql(u8, raw, "--json")) {
            json_out = true;
        } else if (std.mem.eql(u8, raw, "--help") or std.mem.eql(u8, raw, "-h")) {
            try usage(ew);
            return 0;
        } else if (raw.len > 0 and raw[0] != '-') {
            path_buf = try gpa.dupe(u8, raw);
        } else {
            try usage(ew);
            return 2;
        }
    }

    const path: []const u8 = if (path_buf) |p| p else "examples/demo.tfold";

    // Cold run
    const t_cold0 = std.Io.Clock.Timestamp.now(io, .awake);
    const y_cold = try executeDemo(gpa, io, path, pin);
    const t_cold1 = std.Io.Clock.Timestamp.now(io, .awake);
    const cold_took_ns = elapsedNs(t_cold0, t_cold1);
    defer gpa.free(y_cold);

    // Warm repeats
    const warm_times = try gpa.alloc(u64, repeat);
    defer gpa.free(warm_times);
    var r: usize = 0;
    while (r < repeat) : (r += 1) {
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        const y = try executeDemo(gpa, io, path, pin);
        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        warm_times[r] = elapsedNs(t0, t1);
        gpa.free(y);
    }
    std.mem.sort(u64, warm_times, {}, std.sort.asc(u64));
    const warm_median_ns = warm_times[warm_times.len / 2];

    if (json_out) {
        try w.print("{{\n", .{});
        try w.print("  \"file\": \"{s}\",\n", .{path});
        try w.print("  \"dim\": {d},\n", .{y_cold.len});
        try w.print("  \"cold_s\": {d:.9},\n", .{nsToS(cold_took_ns)});
        try w.print("  \"warm_runs\": {d},\n", .{repeat});
        try w.print("  \"warm_median_s\": {d:.9},\n", .{nsToS(warm_median_ns)});
        try w.print("  \"warm_min_s\": {d:.9},\n", .{nsToS(warm_times[0])});
        try w.print("  \"warm_max_s\": {d:.9},\n", .{nsToS(warm_times[warm_times.len - 1])});
        try w.print("  \"output\": [", .{});
        for (y_cold, 0..) |v, idx| {
            if (idx != 0) try w.print(", ", .{});
            try w.print("{d}", .{v});
        }
        try w.print("]\n", .{});
        try w.print("}}\n", .{});
    } else {
        try w.print("file           : {s}\n", .{path});
        try w.print("output dim     : {d}\n", .{y_cold.len});
        try w.print("cold load+run  : {d:.6} s\n", .{nsToS(cold_took_ns)});
        try w.print("warm median    : {d:.6} s  (n={d})\n", .{ nsToS(warm_median_ns), repeat });
        try w.print("warm min/max   : {d:.6} / {d:.6} s\n", .{ nsToS(warm_times[0]), nsToS(warm_times[warm_times.len - 1]) });
        try w.print("output         : [", .{});
        for (y_cold, 0..) |v, idx| {
            if (idx != 0) try w.print(", ", .{});
            try w.print("{d}", .{v});
        }
        try w.print("]\n", .{});
        try w.print("note           : Native Zig CPU reference; not optimized inference.\n", .{});
    }
    return 0;
}

fn nsToS(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

pub fn executeDemo(gpa: std.mem.Allocator, io: std.Io, path: []const u8, pin: []const u8) ![]f32 {
    var f = try tfold.openPath(gpa, io, path, .{ .pin_source_sha256 = pin });
    defer f.close();

    const w_t = f.tensorByName("weight.linear_0") orelse return error.MissingTensor;
    const b_t = f.tensorByName("bias.linear_0") orelse return error.MissingTensor;
    const x_t = f.tensorByName("input.x") orelse return error.MissingTensor;

    if (w_t.dtype != .fp32 or b_t.dtype != .fp32 or x_t.dtype != .fp32) return error.BadDtype;
    if (w_t.shape.len != 2) return error.BadShape;
    if (b_t.shape.len != 1 or x_t.shape.len != 1) return error.BadShape;
    const K: usize = @intCast(w_t.shape[0]);
    const N: usize = @intCast(w_t.shape[1]);
    if (x_t.shape[0] != K or b_t.shape[0] != N) return error.BadShape;

    const w_bytes = f.tensorSlice(w_t);
    const b_bytes = f.tensorSlice(b_t);
    const x_bytes = f.tensorSlice(x_t);

    const W = try gpa.alloc(f32, K * N);
    defer gpa.free(W);
    const B = try gpa.alloc(f32, N);
    defer gpa.free(B);
    const X = try gpa.alloc(f32, K);
    defer gpa.free(X);

    tfold.ops.decodeF32LE(w_bytes, W);
    tfold.ops.decodeF32LE(b_bytes, B);
    tfold.ops.decodeF32LE(x_bytes, X);

    const y_mv = try gpa.alloc(f32, N);
    defer gpa.free(y_mv);
    try tfold.ops.matvecFp32(X, W, K, N, y_mv);

    const y_final = try gpa.alloc(f32, N);
    errdefer gpa.free(y_final);
    try tfold.ops.biasAddFp32(y_mv, B, y_final);
    return y_final;
}
