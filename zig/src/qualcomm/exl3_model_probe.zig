const std = @import("std");
const exl3 = @import("exl3_format");
const st = @import("safetensors");

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

const block = 128;
const source_k = 2048;
const source_m = 6144;
const projection_name = "model.language_model.layers.0.linear_attn.in_proj_qkv";

fn require(file: *const st.File, suffix: []const u8, dtype: st.DType, shape: []const usize) !st.Tensor {
    var name: [256]u8 = undefined;
    const full = try std.fmt.bufPrint(&name, "{s}.{s}", .{ projection_name, suffix });
    const tensor = file.get(full) orelse return error.MissingTensor;
    if (!tensor.is(dtype, shape)) return error.UnexpectedTensor;
    return tensor;
}

fn fnv1a(bytes: []const u8) u64 {
    var hash: u64 = 14695981039346656037;
    for (bytes) |byte| {
        hash ^= byte;
        hash *%= 1099511628211;
    }
    return hash;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 7) {
        std.debug.print("usage: {s} cpu|htp MODEL_SAFETENSORS ITERATIONS BATCH CPU_THREADS block|full\n", .{args[0]});
        return 64;
    }
    if (!std.mem.eql(u8, args[1], "cpu") and !std.mem.eql(u8, args[1], "htp")) return error.InvalidBackend;
    const iterations = try std.fmt.parseInt(usize, args[3], 10);
    const batch = try std.fmt.parseInt(usize, args[4], 10);
    const threads = try std.fmt.parseInt(i32, args[5], 10);
    if (iterations == 0 or batch == 0 or threads <= 0) return error.InvalidArgument;
    const full = if (std.mem.eql(u8, args[6], "full")) true else if (std.mem.eql(u8, args[6], "block")) false else return error.InvalidMode;
    const projection_k: usize = if (full) source_k else block;
    const projection_m: usize = if (full) source_m else block;

    const allocator = init.gpa;
    var file = try st.File.open(allocator, init.io, args[2]);
    defer file.close(init.io);
    const suh_tensor = try require(&file, "suh", .f16, &.{source_k});
    const svh_tensor = try require(&file, "svh", .f16, &.{source_m});
    const trellis_tensor = try require(&file, "trellis", .i16, &.{ 128, 384, 128 });
    const mcg_tensor = try require(&file, "mcg", .i32, &.{});
    if (std.mem.readInt(u32, mcg_tensor.bytes[0..4], .little) != exl3.McgMul) return error.UnexpectedCodebook;

    const all_trellis = std.mem.bytesAsSlice(i16, trellis_tensor.bytes);
    const tile_rows = projection_k / 16;
    const tile_cols = projection_m / 16;
    const total_tile_cols = trellis_tensor.dim(1);
    const words = trellis_tensor.dim(2);
    const selected = try allocator.alloc(i16, tile_rows * tile_cols * words);
    defer allocator.free(selected);
    for (0..tile_rows) |tile_row| {
        for (0..tile_cols) |tile_col| {
            const source = (tile_row * total_tile_cols + tile_col) * words;
            const target = (tile_row * tile_cols + tile_col) * words;
            @memcpy(selected[target..][0..words], all_trellis[source..][0..words]);
        }
    }

    const suh = try allocator.alloc(f16, projection_k);
    defer allocator.free(suh);
    const svh = try allocator.alloc(f16, projection_m);
    defer allocator.free(svh);
    @memcpy(std.mem.sliceAsBytes(suh), suh_tensor.bytes[0 .. projection_k * @sizeOf(f16)]);
    @memcpy(std.mem.sliceAsBytes(svh), svh_tensor.bytes[0 .. projection_m * @sizeOf(f16)]);
    const decoded = try allocator.alloc(f64, projection_k * projection_m);
    defer allocator.free(decoded);
    const decode_scratch = try allocator.alloc(f64, 2 * projection_k * projection_m);
    defer allocator.free(decode_scratch);
    exl3.dequantizeFast(.{ .halves = 16 }, .mcg, selected, tile_rows, tile_cols, suh, svh, decoded, decode_scratch);

    const weight = try allocator.alloc(f32, projection_k * projection_m);
    defer allocator.free(weight);
    for (0..projection_m) |out_col| for (0..projection_k) |inner| {
        weight[out_col * projection_k + inner] = @floatCast(decoded[inner * projection_m + out_col]);
    };
    const input = try allocator.alloc(f32, projection_k * batch);
    defer allocator.free(input);
    for (input, 0..) |*value, i| {
        const signed: i32 = @intCast(i % 31);
        value.* = @as(f32, @floatFromInt(signed - 15)) / 16.0;
    }
    const output = try allocator.alloc(f32, projection_m * batch);
    defer allocator.free(output);
    const reference = try allocator.alloc(f32, projection_m * batch);
    defer allocator.free(reference);
    for (0..batch) |row| for (0..projection_m) |out_col| {
        var sum: f32 = 0;
        for (0..projection_k) |inner| sum += weight[out_col * projection_k + inner] * input[row * projection_k + inner];
        reference[row * projection_m + out_col] = sum;
    };

    var ms: f64 = 0;
    const bridge_status = tf_qualcomm_matmul(args[1].ptr, projection_k, projection_m, batch, iterations, threads, weight.ptr, input.ptr, output.ptr, &ms);
    if (bridge_status != 0) return error.BridgeFailed;
    var max_abs: f32 = 0;
    var reference_max_abs: f32 = 0;
    for (output, reference) |got, want| {
        max_abs = @max(max_abs, @abs(got - want));
        reference_max_abs = @max(reference_max_abs, @abs(want));
    }
    const checksum = fnv1a(std.mem.sliceAsBytes(weight));
    const tolerance = 5e-4 * (1.0 + reference_max_abs);
    const passed = max_abs <= tolerance;
    std.debug.print(
        "{{\"tensorfold\":\"qualcomm-exl3-model-probe\",\"backend\":\"{s}\",\"checkpoint_tensors\":{d},\"source_tensor\":\"{s}\",\"source_shape\":[2048,6144],\"projection_shape\":[{d},{d}],\"batch\":{d},\"iterations\":{d},\"ms_per_iteration\":{d:.6},\"weight_fnv1a\":\"{x}\",\"reference_max_abs\":{d},\"projection_max_abs_error\":{d},\"tolerance\":{d},\"passed\":{s}}}\n",
        .{ args[1], file.names.count(), projection_name, projection_k, projection_m, batch, iterations, ms, checksum, reference_max_abs, max_abs, tolerance, if (passed) "true" else "false" },
    );
    return if (passed) 0 else 8;
}
