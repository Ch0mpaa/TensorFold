const std = @import("std");
const st = @import("safetensors");

extern fn tf_qualcomm_backend_create(backend_name: [*:0]const u8, cpu_threads: i32) ?*anyopaque;
extern fn tf_qualcomm_backend_destroy(context: ?*anyopaque) void;
extern fn tf_qualcomm_backend_matmul_f16_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    k: usize,
    m: usize,
    n: usize,
    weight: ?[*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weight_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_matmul_q8_f16_source_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    k: usize,
    m: usize,
    n: usize,
    weight: ?[*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weight_uploaded: *c_int,
) c_int;

const Expected = struct {
    name: []const u8,
    shape: []const usize,
};

const expected = [_]Expected{
    .{ .name = "mtp.fc.weight", .shape = &.{ 2560, 5120 } },
    .{ .name = "mtp.layers.0.input_layernorm.weight", .shape = &.{2560} },
    .{ .name = "mtp.layers.0.mlp.down_proj.weight", .shape = &.{ 2560, 9216 } },
    .{ .name = "mtp.layers.0.mlp.gate_proj.weight", .shape = &.{ 9216, 2560 } },
    .{ .name = "mtp.layers.0.mlp.up_proj.weight", .shape = &.{ 9216, 2560 } },
    .{ .name = "mtp.layers.0.post_attention_layernorm.weight", .shape = &.{2560} },
    .{ .name = "mtp.layers.0.self_attn.k_norm.weight", .shape = &.{256} },
    .{ .name = "mtp.layers.0.self_attn.k_proj.weight", .shape = &.{ 1024, 2560 } },
    .{ .name = "mtp.layers.0.self_attn.o_proj.weight", .shape = &.{ 2560, 4096 } },
    .{ .name = "mtp.layers.0.self_attn.q_norm.weight", .shape = &.{256} },
    .{ .name = "mtp.layers.0.self_attn.q_proj.weight", .shape = &.{ 8192, 2560 } },
    .{ .name = "mtp.layers.0.self_attn.v_proj.weight", .shape = &.{ 1024, 2560 } },
    .{ .name = "mtp.norm.weight", .shape = &.{2560} },
    .{ .name = "mtp.pre_fc_norm_embedding.weight", .shape = &.{2560} },
    .{ .name = "mtp.pre_fc_norm_hidden.weight", .shape = &.{2560} },
};

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn validate(file: *const st.File) !void {
    if (file.names.count() != expected.len) return error.UnexpectedTensorCount;
    for (expected) |want| {
        const tensor = file.get(want.name) orelse return error.MissingTensor;
        if (!tensor.is(.bf16, want.shape)) return error.UnexpectedTensor;
    }
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
    if (args.len < 7 or args.len > 8) {
        std.debug.print("usage: {s} cpu|htp MTP_SAFETENSORS ITERATIONS BATCH CPU_THREADS f16|q8 [TENSOR_NAME]\n", .{args[0]});
        return 64;
    }
    if (!std.mem.eql(u8, args[1], "cpu") and !std.mem.eql(u8, args[1], "htp")) return error.InvalidBackend;
    const iterations = try std.fmt.parseInt(usize, args[3], 10);
    const batch = try std.fmt.parseInt(usize, args[4], 10);
    const threads = try std.fmt.parseInt(i32, args[5], 10);
    const use_q8 = std.mem.eql(u8, args[6], "q8");
    if (!use_q8 and !std.mem.eql(u8, args[6], "f16")) return error.InvalidMode;
    if (iterations == 0 or batch == 0 or threads <= 0) return error.InvalidArgument;

    const allocator = init.gpa;
    var file = try st.File.open(allocator, init.io, args[2]);
    defer file.close(init.io);
    try validate(&file);

    const source_name = if (args.len == 8) args[7] else "mtp.fc.weight";
    const tensor = file.get(source_name) orelse return error.MissingTensor;
    if (tensor.dtype != .bf16 or tensor.rank != 2) return error.UnexpectedTensor;
    const m = tensor.dim(0);
    const k = tensor.dim(1);
    const bf16 = std.mem.bytesAsSlice(u16, tensor.bytes);
    const weight_f16 = try allocator.alloc(u16, tensor.numel());
    defer allocator.free(weight_f16);
    for (bf16, weight_f16) |bits, *converted| {
        converted.* = @bitCast(@as(f16, @floatCast(bf16ToF32(bits))));
    }

    const input = try allocator.alloc(f32, k * batch);
    defer allocator.free(input);
    for (0..batch) |row| for (0..k) |column| {
        const signed: i32 = @intCast(column % 31);
        input[row * k + column] = @as(f32, @floatFromInt(signed - 15)) / 32.0;
    };
    const output = try allocator.alloc(f32, m * batch);
    defer allocator.free(output);
    const reference = try allocator.alloc(f32, m);
    defer allocator.free(reference);
    for (0..m) |out_row| {
        var sum: f32 = 0;
        for (0..k) |column| sum += bf16ToF32(bf16[out_row * k + column]) * input[column];
        reference[out_row] = sum;
    }

    const backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BridgeFailed;
    defer tf_qualcomm_backend_destroy(backend);
    var total_ms: f64 = 0;
    var uploaded_first: c_int = 0;
    for (0..iterations) |iteration| {
        var compute_ms: f64 = 0;
        var uploaded: c_int = 0;
        const weight_ptr: ?[*]const u16 = if (iteration == 0) weight_f16.ptr else null;
        const status = if (use_q8)
            tf_qualcomm_backend_matmul_q8_f16_source_resident_once(backend, 0, k, m, batch, weight_ptr, input.ptr, output.ptr, &compute_ms, &uploaded)
        else
            tf_qualcomm_backend_matmul_f16_resident_once(backend, 0, k, m, batch, weight_ptr, input.ptr, output.ptr, &compute_ms, &uploaded);
        if (status != 0) return error.BridgeFailed;
        if (iteration == 0) uploaded_first = uploaded;
        total_ms += compute_ms;
    }

    var max_abs: f32 = 0;
    var reference_max_abs: f32 = 0;
    for (0..batch) |row| for (0..m) |column| {
        const want = reference[column];
        max_abs = @max(max_abs, @abs(output[row * m + column] - want));
        reference_max_abs = @max(reference_max_abs, @abs(want));
    };
    const relative_tolerance: f32 = if (use_q8) 0.04 else 0.01;
    const tolerance = relative_tolerance * (1.0 + reference_max_abs);
    const passed = uploaded_first == 1 and max_abs <= tolerance;
    std.debug.print(
        "{{\"tensorfold\":\"qwen35-mtp-probe\",\"backend\":\"{s}\",\"kernel\":\"{s}\",\"validated_tensors\":{d},\"source_tensor\":\"{s}\",\"shape\":[{d},{d}],\"batch\":{d},\"iterations\":{d},\"ms_per_iteration\":{d:.6},\"weight_uploaded\":{d},\"source_fnv1a\":\"{x}\",\"reference_max_abs\":{d},\"max_abs_error\":{d},\"tolerance\":{d},\"passed\":{s}}}\n",
        .{ args[1], if (use_q8) "bf16-to-resident-q8" else "bf16-to-resident-f16", expected.len, source_name, m, k, batch, iterations, total_ms / @as(f64, @floatFromInt(iterations)), uploaded_first, fnv1a(tensor.bytes), reference_max_abs, max_abs, tolerance, if (passed) "true" else "false" },
    );
    return if (passed) 0 else 8;
}
