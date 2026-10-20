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
extern fn tf_qualcomm_backend_create(backend_name: [*:0]const u8, cpu_threads: i32) ?*anyopaque;
extern fn tf_qualcomm_backend_destroy(context: ?*anyopaque) void;
extern fn tf_qualcomm_backend_matmul_q4(
    context: ?*anyopaque,
    k: usize,
    m: usize,
    n: usize,
    iterations: usize,
    weight: [*]const f32,
    input: [*]const f32,
    output: [*]f32,
    ms_per_iteration: *f64,
) c_int;
extern fn tf_qualcomm_backend_matmul_q8(
    context: ?*anyopaque,
    k: usize,
    m: usize,
    n: usize,
    iterations: usize,
    weight: [*]const f32,
    input: [*]const f32,
    output: [*]f32,
    ms_per_iteration: *f64,
) c_int;
extern fn tf_qualcomm_backend_exl3_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    k: usize,
    m: usize,
    n: usize,
    words: usize,
    trellis: ?[*]const u16,
    suh: ?[*]const u16,
    svh: ?[*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weight_uploaded: *c_int,
) c_int;

const block = 128;
const projection_name = "model.language_model.layers.0.linear_attn.in_proj_qkv";

fn require(file: *const st.File, suffix: []const u8, dtype: st.DType) !st.Tensor {
    var name: [256]u8 = undefined;
    const full = try std.fmt.bufPrint(&name, "{s}.{s}", .{ projection_name, suffix });
    const tensor = file.get(full) orelse return error.MissingTensor;
    if (tensor.dtype != dtype) return error.UnexpectedTensor;
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
        std.debug.print("usage: {s} cpu|htp MODEL_SAFETENSORS ITERATIONS BATCH CPU_THREADS block|full|packed|q4|q8\n", .{args[0]});
        return 64;
    }
    if (!std.mem.eql(u8, args[1], "cpu") and !std.mem.eql(u8, args[1], "htp")) return error.InvalidBackend;
    const iterations = try std.fmt.parseInt(usize, args[3], 10);
    const batch = try std.fmt.parseInt(usize, args[4], 10);
    const threads = try std.fmt.parseInt(i32, args[5], 10);
    if (iterations == 0 or batch == 0 or threads <= 0) return error.InvalidArgument;
    const use_packed = std.mem.eql(u8, args[6], "packed");
    const use_q4 = std.mem.eql(u8, args[6], "q4");
    const use_q8 = std.mem.eql(u8, args[6], "q8");
    const full = use_packed or use_q4 or use_q8 or std.mem.eql(u8, args[6], "full");
    if (!full and !std.mem.eql(u8, args[6], "block")) return error.InvalidMode;

    const allocator = init.gpa;
    var file = try st.File.open(allocator, init.io, args[2]);
    defer file.close(init.io);
    const trellis_tensor = try require(&file, "trellis", .i16);
    if (trellis_tensor.rank != 3 or trellis_tensor.dim(2) != 64) return error.UnexpectedTensor;
    const source_k = trellis_tensor.dim(0) * 16;
    const source_m = trellis_tensor.dim(1) * 16;
    const projection_k: usize = if (full) source_k else block;
    const projection_m: usize = if (full) source_m else block;
    const suh_tensor = try require(&file, "suh", .f16);
    const svh_tensor = try require(&file, "svh", .f16);
    if (suh_tensor.rank != 1 or suh_tensor.dim(0) != source_k or svh_tensor.rank != 1 or svh_tensor.dim(0) != source_m) return error.UnexpectedTensor;
    const mcg_tensor = try require(&file, "mcg", .i32);
    if (mcg_tensor.rank != 0) return error.UnexpectedTensor;
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
    exl3.dequantizeFast(.{ .halves = @intCast(words / 8) }, .mcg, selected, tile_rows, tile_cols, suh, svh, decoded, decode_scratch);

    const weight = if (use_packed) null else try allocator.alloc(f32, projection_k * projection_m);
    defer if (weight) |w| allocator.free(w);
    if (weight) |w| for (0..projection_m) |out_col| for (0..projection_k) |inner| {
        w[out_col * projection_k + inner] = @floatCast(decoded[inner * projection_m + out_col]);
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
        for (0..projection_k) |inner| sum += @as(f32, @floatCast(decoded[inner * projection_m + out_col])) * input[row * projection_k + inner];
        reference[row * projection_m + out_col] = sum;
    };

    var ms: f64 = 0;
    var bridge_status: c_int = 0;
    if (use_packed) {
        const backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BridgeFailed;
        defer tf_qualcomm_backend_destroy(backend);
        var total_ms: f64 = 0;
        for (0..iterations) |iteration| {
            var run_ms: f64 = 0;
            var uploaded: c_int = 0;
            bridge_status = tf_qualcomm_backend_exl3_resident_once(
                backend,
                0,
                projection_k,
                projection_m,
                batch,
                words,
                if (iteration == 0) @ptrCast(selected.ptr) else null,
                if (iteration == 0) @ptrCast(suh.ptr) else null,
                if (iteration == 0) @ptrCast(svh.ptr) else null,
                input.ptr,
                output.ptr,
                &run_ms,
                &uploaded,
            );
            if (bridge_status != 0) break;
            total_ms += run_ms;
        }
        ms = total_ms / @as(f64, @floatFromInt(iterations));
    } else if (use_q4) {
        const backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BridgeFailed;
        defer tf_qualcomm_backend_destroy(backend);
        bridge_status = tf_qualcomm_backend_matmul_q4(
            backend, projection_k, projection_m, batch, iterations, weight.?.ptr, input.ptr, output.ptr, &ms,
        );
    } else if (use_q8) {
        const backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BridgeFailed;
        defer tf_qualcomm_backend_destroy(backend);
        bridge_status = tf_qualcomm_backend_matmul_q8(
            backend, projection_k, projection_m, batch, iterations, weight.?.ptr, input.ptr, output.ptr, &ms,
        );
    } else {
        bridge_status = tf_qualcomm_matmul(args[1].ptr, projection_k, projection_m, batch, iterations, threads, weight.?.ptr, input.ptr, output.ptr, &ms);
    }
    if (bridge_status != 0) return error.BridgeFailed;
    var max_abs: f32 = 0;
    var reference_max_abs: f32 = 0;
    for (output, reference) |got, want| {
        max_abs = @max(max_abs, @abs(got - want));
        reference_max_abs = @max(reference_max_abs, @abs(want));
    }
    const checksum = if (weight) |w| fnv1a(std.mem.sliceAsBytes(w)) else fnv1a(std.mem.sliceAsBytes(selected));
    const tolerance = 2e-2 * (1.0 + reference_max_abs);
    const passed = max_abs <= tolerance;
    std.debug.print(
        "{{\"tensorfold\":\"qualcomm-exl3-model-probe\",\"backend\":\"{s}\",\"kernel\":\"{s}\",\"checkpoint_tensors\":{d},\"source_tensor\":\"{s}\",\"source_shape\":[{d},{d}],\"projection_shape\":[{d},{d}],\"batch\":{d},\"iterations\":{d},\"ms_per_iteration\":{d:.6},\"weight_fnv1a\":\"{x}\",\"reference_max_abs\":{d},\"projection_max_abs_error\":{d},\"tolerance\":{d},\"passed\":{s}}}\n",
        .{ args[1], if (use_packed) "packed-exl3-hvx-hmx" else if (use_q4) "decoded-q4_0-resident" else if (use_q8) "decoded-q8_0-resident" else "decoded", file.names.count(), projection_name, source_k, source_m, projection_k, projection_m, batch, iterations, ms, checksum, reference_max_abs, max_abs, tolerance, if (passed) "true" else "false" },
    );
    return if (passed) 0 else 8;
}
