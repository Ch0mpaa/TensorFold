const std = @import("std");
const exl3 = @import("exl3_format");

extern fn tf_qualcomm_matmul(
    backend_name: [*:0]const u8,
    k: usize,
    m: usize,
    n: usize,
    iterations: usize,
    cpu_threads: i32,
    weight: [*]const f32,
    input: [*]const f32,
    output: [*]f32,
    ms_per_iteration: *f64,
) c_int;

fn readInt(comptime T: type, bytes: []const u8, at: usize) T {
    return std.mem.readInt(T, bytes[at..][0..@sizeOf(T)], .little);
}

fn realBlockOffset(bytes: []const u8) !usize {
    if (bytes.len < 16 or !std.mem.eql(u8, bytes[0..8], "EXL3ORAC")) return error.BadFixture;
    const records = readInt(u64, bytes, 8);
    var at: usize = 16;
    for (0..records) |_| {
        if (at + 12 > bytes.len) return error.BadFixture;
        const width: exl3.Width = .{ .halves = readInt(u16, bytes, at) };
        const tiles_k = readInt(u32, bytes, at + 4);
        const tiles_n = readInt(u32, bytes, at + 8);
        at += 12 + @as(usize, tiles_k) * @as(usize, tiles_n) * (width.tileWords() * 2 + 256 * 4);
        if (at > bytes.len) return error.BadFixture;
    }
    if (at + 4 > bytes.len) return error.BadFixture;
    at += 4 + @as(usize, readInt(u32, bytes, at)) * 6;
    if (at > bytes.len) return error.BadFixture;
    return at;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 6) {
        std.debug.print("usage: {s} cpu|htp FIXTURE ITERATIONS BATCH CPU_THREADS\n", .{args[0]});
        return 64;
    }
    if (!std.mem.eql(u8, args[1], "htp") and !std.mem.eql(u8, args[1], "cpu")) {
        std.debug.print("backend must be cpu or htp\n", .{});
        return 64;
    }
    const iterations = try std.fmt.parseInt(usize, args[3], 10);
    const batch = try std.fmt.parseInt(usize, args[4], 10);
    const threads = try std.fmt.parseInt(i32, args[5], 10);
    if (iterations == 0 or batch == 0 or threads <= 0) return error.InvalidArgument;

    const allocator = init.gpa;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], allocator, .limited(4 << 20));
    defer allocator.free(bytes);
    var at = try realBlockOffset(bytes);

    const width: exl3.Width = .{ .halves = readInt(u16, bytes, at) };
    const tiles_k: usize = readInt(u32, bytes, at + 2);
    const tiles_n: usize = readInt(u32, bytes, at + 6);
    const trellis_len: usize = readInt(u32, bytes, at + 10);
    at += 14;
    const trellis = try allocator.alloc(i16, trellis_len / 2);
    defer allocator.free(trellis);
    @memcpy(std.mem.sliceAsBytes(trellis), bytes[at .. at + trellis_len]);
    at += trellis_len;

    const suh_len: usize = readInt(u32, bytes, at);
    at += 4;
    const suh_bits = try allocator.alloc(u16, suh_len / 2);
    defer allocator.free(suh_bits);
    @memcpy(std.mem.sliceAsBytes(suh_bits), bytes[at .. at + suh_len]);
    at += suh_len;
    const svh_len: usize = readInt(u32, bytes, at);
    at += 4;
    const svh_bits = try allocator.alloc(u16, svh_len / 2);
    defer allocator.free(svh_bits);
    @memcpy(std.mem.sliceAsBytes(svh_bits), bytes[at .. at + svh_len]);
    at += svh_len;

    const wq_len: usize = readInt(u32, bytes, at);
    at += 4 + wq_len;
    const oracle_len: usize = readInt(u32, bytes, at);
    at += 4;
    const oracle = std.mem.bytesAsSlice(f64, bytes[at .. at + oracle_len]);

    const k = tiles_k * 16;
    const output_width = tiles_n * 16;
    if (k != 128 or output_width != 128) return error.UnexpectedFixtureShape;
    const decoded = try allocator.alloc(f64, k * output_width);
    defer allocator.free(decoded);
    const scratch = try allocator.alloc(f64, 2 * k * output_width);
    defer allocator.free(scratch);
    const suh = std.mem.bytesAsSlice(f16, std.mem.sliceAsBytes(suh_bits));
    const svh = std.mem.bytesAsSlice(f16, std.mem.sliceAsBytes(svh_bits));
    exl3.dequantize(width, .mcg, trellis, tiles_k, tiles_n, suh, svh, decoded, scratch);

    var decode_max_abs: f64 = 0;
    for (decoded, oracle) |got, want| decode_max_abs = @max(decode_max_abs, @abs(got - want));

    const weight = try allocator.alloc(f32, k * output_width);
    defer allocator.free(weight);
    for (0..output_width) |out_col| {
        for (0..k) |inner| weight[out_col * k + inner] = @floatCast(decoded[inner * output_width + out_col]);
    }
    const input = try allocator.alloc(f32, k * batch);
    defer allocator.free(input);
    for (input, 0..) |*value, i| {
        const signed: i32 = @intCast(i % 29);
        value.* = @as(f32, @floatFromInt(signed - 14)) / 16.0;
    }
    const output = try allocator.alloc(f32, output_width * batch);
    defer allocator.free(output);
    const reference = try allocator.alloc(f32, output_width * batch);
    defer allocator.free(reference);
    for (0..batch) |row| {
        for (0..output_width) |out_col| {
            var sum: f32 = 0;
            for (0..k) |inner| sum += weight[out_col * k + inner] * input[row * k + inner];
            reference[row * output_width + out_col] = sum;
        }
    }

    var ms: f64 = 0;
    const bridge_status = tf_qualcomm_matmul(
        args[1].ptr,
        k,
        output_width,
        batch,
        iterations,
        threads,
        weight.ptr,
        input.ptr,
        output.ptr,
        &ms,
    );
    if (bridge_status != 0) {
        std.debug.print("GGML bridge failed with status {d}\n", .{bridge_status});
        return 7;
    }

    var projection_max_abs: f32 = 0;
    for (output, reference) |got, want| projection_max_abs = @max(projection_max_abs, @abs(got - want));
    const passed = decode_max_abs <= 1e-9 and projection_max_abs <= 1e-3;
    std.debug.print(
        "{{\"tensorfold\":\"qualcomm-exl3-projection\",\"backend\":\"{s}\",\"source\":\"real-mcg-8.00bpw\",\"k\":{d},\"m\":{d},\"batch\":{d},\"iterations\":{d},\"ms_per_iteration\":{d:.6},\"decode_max_abs_error\":{d},\"projection_max_abs_error\":{d},\"passed\":{s}}}\n",
        .{ args[1], k, output_width, batch, iterations, ms, decode_max_abs, projection_max_abs, if (passed) "true" else "false" },
    );
    return if (passed) 0 else 8;
}
