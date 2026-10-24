const std = @import("std");
const exl3 = @import("exl3_format");
const st = @import("safetensors");

const block = 128;
const source_k = 2048;
const source_m = 6144;
const projection_name = "model.language_model.layers.0.linear_attn.in_proj_qkv";
const model_sha256_hex = "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f";

const BridgeRequest = extern struct {
    native_dir: [*:0]const u8,
    stub_name: [*:0]const u8,
    input: [*]const u8,
    weights: [*]const u8,
    reference: [*]const f32,
    output: [*]u8,
    k: u32,
    m: u32,
    input_scale: f32,
    input_offset: i32,
    weight_scale: f32,
    weight_offset: i32,
    output_scale: f32,
    output_offset: i32,
    tolerance: f32,
};

const BridgeResult = extern struct {
    status: i32,
    execution_ms: f64,
    max_abs_error: f64,
};

const Execute = *const fn (*const BridgeRequest, *BridgeResult) callconv(.c) i32;

const payload_magic = "TFQNNLP1";
const payload_version: u32 = 1;
const payload_header_bytes: u32 = 256;
const payload_identity_offset = 180;
const payload_integrity_bytes = 8;

const Failure = error{
    Usage,
    Model,
    Bridge,
    Parity,
    Payload,
};

fn require(file: *const st.File, suffix: []const u8, dtype: st.DType) Failure!st.Tensor {
    var name: [256]u8 = undefined;
    const full = std.fmt.bufPrint(&name, "{s}.{s}", .{ projection_name, suffix }) catch return error.Model;
    const tensor = file.get(full) orelse return error.Model;
    if (tensor.dtype != dtype) return error.Model;
    return tensor;
}

fn symmetricQuantize(value: f64, scale: f64) Failure!u8 {
    if (!std.math.isFinite(value) or !std.math.isFinite(scale) or scale <= 0) return error.Model;
    const magnitude = @floor(@abs(value) / scale + 0.5);
    const signed: i32 = @intFromFloat(if (value < 0) -magnitude else magnitude);
    if (signed < -127 or signed > 127) return error.Model;
    return @intCast(signed + 128);
}

fn sha256Hex(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn appendInt(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, comptime T: type, value: T) Failure!void {
    var encoded: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, encoded[0..], value, .little);
    bytes.appendSlice(allocator, &encoded) catch return error.Payload;
}

fn appendFloat(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, value: f32) Failure!void {
    try appendInt(bytes, allocator, u32, @bitCast(value));
}

fn fnv1a64(bytes: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (bytes) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    }
    return hash;
}

fn quantizeProjection(decoded: []const f64, allocator: std.mem.Allocator) Failure!struct {
    input: []u8,
    weights: []u8,
    reference: []f32,
    input_scale: f32,
    weight_scale: f32,
    output_scale: f32,
    block_sha256: [64]u8,
    payload_sha256: [64]u8,
} {
    if (decoded.len != block * block) return error.Model;
    const input = allocator.alloc(u8, block) catch return error.Model;
    errdefer allocator.free(input);
    const weights = allocator.alloc(u8, block * block) catch return error.Model;
    errdefer allocator.free(weights);
    const reference = allocator.alloc(f32, block) catch return error.Model;
    errdefer allocator.free(reference);
    const input_scale: f64 = 1.0 / 127.0;
    var max_weight: f64 = 0;
    for (decoded) |value| max_weight = @max(max_weight, @abs(value));
    if (max_weight == 0) return error.Model;
    const weight_scale: f64 = max_weight / 127.0;
    for (input, 0..) |*entry, index| {
        const signed: i32 = @intCast((index * 73 + 19) % 255);
        entry.* = @intCast(signed + 1); // ((i * 73 + 19) % 255) - 127, shifted by 128.
    }
    for (decoded, 0..) |value, index| weights[index] = try symmetricQuantize(value, weight_scale);
    var max_reference: f64 = 0;
    var dots: [block]i32 = undefined;
    for (0..block) |out_col| {
        var dot: i32 = 0;
        for (0..block) |inner| {
            const lhs: i32 = @as(i32, input[inner]) - 128;
            const rhs: i32 = @as(i32, weights[inner * block + out_col]) - 128;
            dot += lhs * rhs;
        }
        dots[out_col] = dot;
        const value = @as(f64, @floatFromInt(dot)) * input_scale * weight_scale;
        max_reference = @max(max_reference, @abs(value));
        reference[out_col] = @floatCast(value);
    }
    if (max_reference == 0) return error.Model;
    const output_scale: f64 = max_reference / 100.0;
    // Match the generated fixture's range proof before asking QNN to quantize output.
    for (reference) |value| {
        const rounded: i32 = @intFromFloat(@floor(@as(f64, value) / output_scale + 0.5));
        if (rounded < -127 or rounded > 127) return error.Model;
    }
    var payload: std.ArrayList(u8) = .empty;
    payload.ensureTotalCapacity(allocator, block + block * block + block * 4 + 3 * 8) catch return error.Model;
    defer payload.deinit(allocator);
    payload.appendSlice(allocator, input) catch return error.Model;
    payload.appendSlice(allocator, weights) catch return error.Model;
    for (dots) |dot| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(i32, bytes[0..], dot, .little);
        payload.appendSlice(allocator, bytes[0..]) catch return error.Model;
    }
    for ([_]f64{ input_scale, weight_scale, output_scale }) |scale| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, bytes[0..], @bitCast(scale), .little);
        payload.appendSlice(allocator, bytes[0..]) catch return error.Model;
    }
    return .{ .input = input, .weights = weights, .reference = reference, .input_scale = @floatCast(input_scale), .weight_scale = @floatCast(weight_scale), .output_scale = @floatCast(output_scale), .block_sha256 = sha256Hex(std.mem.sliceAsBytes(decoded)), .payload_sha256 = sha256Hex(payload.items) };
}

fn writePreparedPayload(path: []const u8, model_sha: [64]u8, quantized: anytype, allocator: std.mem.Allocator, io: std.Io) Failure!void {
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(allocator);
    payload.ensureTotalCapacity(allocator, payload_header_bytes + block + block * block + block * @sizeOf(f32) + payload_integrity_bytes) catch return error.Payload;
    payload.appendSlice(allocator, payload_magic) catch return error.Payload;
    try appendInt(&payload, allocator, u32, payload_version);
    try appendInt(&payload, allocator, u32, payload_header_bytes);
    try appendInt(&payload, allocator, u32, block);
    try appendInt(&payload, allocator, u32, block);
    try appendFloat(&payload, allocator, quantized.input_scale);
    try appendFloat(&payload, allocator, quantized.weight_scale);
    try appendFloat(&payload, allocator, quantized.output_scale);
    try appendFloat(&payload, allocator, quantized.output_scale * 1.5);
    try appendInt(&payload, allocator, i32, -128);
    try appendInt(&payload, allocator, i32, -128);
    try appendInt(&payload, allocator, i32, -128);
    payload.appendSlice(allocator, &model_sha) catch return error.Payload;
    payload.appendSlice(allocator, &quantized.block_sha256) catch return error.Payload;
    payload.appendSlice(allocator, &quantized.payload_sha256) catch return error.Payload;
    try appendInt(&payload, allocator, u32, block);
    try appendInt(&payload, allocator, u32, block * block);
    try appendInt(&payload, allocator, u32, block * @sizeOf(f32));
    std.debug.assert(payload.items.len == payload_header_bytes);
    payload.appendSlice(allocator, quantized.input) catch return error.Payload;
    payload.appendSlice(allocator, quantized.weights) catch return error.Payload;
    for (quantized.reference) |value| try appendFloat(&payload, allocator, value);
    try appendInt(&payload, allocator, u64, fnv1a64(payload.items));

    if (std.fs.path.dirname(path)) |parent| std.Io.Dir.cwd().createDirPath(io, parent) catch return error.Payload;
    const temporary = std.fmt.allocPrint(allocator, "{s}.tmp", .{path}) catch return error.Payload;
    defer allocator.free(temporary);
    defer std.Io.Dir.cwd().deleteFile(io, temporary) catch {};
    var output = std.Io.Dir.cwd().createFile(io, temporary, .{ .truncate = true }) catch return error.Payload;
    defer output.close(io);
    output.writePositionalAll(io, payload.items, 0) catch return error.Payload;
    output.sync(io) catch return error.Payload;
    std.Io.Dir.cwd().rename(temporary, std.Io.Dir.cwd(), path, io) catch return error.Payload;
}

const fidelity_payload_magic = "TFQNNLF3";
const fidelity_payload_version: u32 = 3;
const fidelity_payload_header_bytes: u32 = 384;
const channelwise_payload_magic = "TFQNNLC4";
const channelwise_payload_version: u32 = 4;
const channelwise_payload_header_bytes: u32 = 448;
const full_payload_magic = "TFQNNLF2";
const full_payload_version: u32 = 2;
const full_payload_header_bytes: u32 = 320;
const full_tensor_identity = "5ef6279518fbef7940d8da4011b6f0bb77375ffb3a6953ce9fc2c7752844343a";

fn projectionTensorIdentity(trellis: st.Tensor, suh: st.Tensor, svh: st.Tensor, mcg: st.Tensor) [64]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_]struct { name: []const u8, bytes: []const u8 }{
        .{ .name = projection_name ++ ".trellis", .bytes = trellis.bytes },
        .{ .name = projection_name ++ ".suh", .bytes = suh.bytes },
        .{ .name = projection_name ++ ".svh", .bytes = svh.bytes },
        .{ .name = projection_name ++ ".mcg", .bytes = mcg.bytes },
    }) |part| {
        hash.update(part.name);
        hash.update(&.{0});
        hash.update(part.bytes);
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn writeFullPreparedPayload(path: []const u8, model_sha: [64]u8, trellis: st.Tensor, suh: st.Tensor, svh: st.Tensor, words: usize, allocator: std.mem.Allocator, io: std.Io) Failure!void {
    const weight_count = source_k * source_m;
    const input = allocator.alloc(u8, source_k) catch return error.Payload;
    defer allocator.free(input);
    const weights = allocator.alloc(u8, weight_count) catch return error.Payload;
    defer allocator.free(weights);
    const reference = allocator.alloc(f32, source_m) catch return error.Payload;
    defer allocator.free(reference);
    const trellis_values = allocator.alloc(i16, trellis.numel()) catch return error.Payload;
    defer allocator.free(trellis_values);
    @memcpy(std.mem.sliceAsBytes(trellis_values), trellis.bytes);
    const suh_values = allocator.alloc(f16, source_k) catch return error.Payload;
    defer allocator.free(suh_values);
    @memcpy(std.mem.sliceAsBytes(suh_values), suh.bytes);
    const svh_values = allocator.alloc(f16, source_m) catch return error.Payload;
    defer allocator.free(svh_values);
    @memcpy(std.mem.sliceAsBytes(svh_values), svh.bytes);
    const decoded = allocator.alloc(f64, weight_count) catch return error.Payload;
    defer allocator.free(decoded);
    const scratch = allocator.alloc(f64, 2 * weight_count) catch return error.Payload;
    defer allocator.free(scratch);
    exl3.dequantizeFast(
        .{ .halves = @intCast(words / 8) },
        .mcg,
        trellis_values,
        trellis.dim(0),
        trellis.dim(1),
        suh_values,
        svh_values,
        decoded,
        scratch,
    );
    var max_weight: f64 = 0;
    for (decoded) |value| {
        if (!std.math.isFinite(value)) return error.Model;
        max_weight = @max(max_weight, @abs(value));
    }
    if (max_weight == 0) return error.Model;
    const input_scale: f64 = 1.0 / 127.0;
    const weight_scale: f64 = max_weight / 127.0;
    for (input, 0..) |*entry, index| entry.* = @intCast((index * 73 + 19) % 255 + 1);
    for (decoded, 0..) |value, index| weights[index] = try symmetricQuantize(value, weight_scale);
    var dots: [source_m]i32 = undefined;
    var max_reference: f64 = 0;
    for (0..source_m) |m| {
        var dot: i32 = 0;
        for (0..source_k) |k| dot += (@as(i32, input[k]) - 128) * (@as(i32, weights[k * source_m + m]) - 128);
        dots[m] = dot;
        const value = @as(f64, @floatFromInt(dot)) * input_scale * weight_scale;
        reference[m] = @floatCast(value);
        max_reference = @max(max_reference, @abs(value));
    }
    if (max_reference == 0) return error.Model;
    const output_scale: f64 = max_reference / 100.0;
    for (reference) |value| if (@as(f64, value) / output_scale < -127 or @as(f64, value) / output_scale > 127) return error.Model;
    var identity: std.ArrayList(u8) = .empty;
    defer identity.deinit(allocator);
    identity.ensureTotalCapacity(allocator, source_k + weight_count + source_m * 4 + 24) catch return error.Payload;
    identity.appendSlice(allocator, input) catch return error.Payload;
    identity.appendSlice(allocator, weights) catch return error.Payload;
    for (dots) |dot| try appendInt(&identity, allocator, i32, dot);
    for ([_]f64{ input_scale, weight_scale, output_scale }) |scale| try appendInt(&identity, allocator, u64, @bitCast(scale));
    const payload_sha = sha256Hex(identity.items);
    const output_scale_f32: f32 = @floatCast(output_scale);
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(allocator);
    payload.ensureTotalCapacity(allocator, full_payload_header_bytes + source_k + weight_count + source_m * @sizeOf(f32) + payload_integrity_bytes) catch return error.Payload;
    payload.appendSlice(allocator, full_payload_magic) catch return error.Payload;
    try appendInt(&payload, allocator, u32, full_payload_version);
    try appendInt(&payload, allocator, u32, full_payload_header_bytes);
    try appendInt(&payload, allocator, u32, source_k);
    try appendInt(&payload, allocator, u32, source_m);
    try appendFloat(&payload, allocator, @floatCast(input_scale));
    try appendFloat(&payload, allocator, @floatCast(weight_scale));
    try appendFloat(&payload, allocator, output_scale_f32);
    try appendFloat(&payload, allocator, output_scale_f32 * 1.5);
    try appendInt(&payload, allocator, i32, -128);
    try appendInt(&payload, allocator, i32, -128);
    try appendInt(&payload, allocator, i32, -128);
    payload.appendSlice(allocator, &model_sha) catch return error.Payload;
    payload.appendSlice(allocator, full_tensor_identity) catch return error.Payload;
    payload.appendSlice(allocator, &payload_sha) catch return error.Payload;
    try appendInt(&payload, allocator, u32, source_k);
    try appendInt(&payload, allocator, u32, weight_count);
    try appendInt(&payload, allocator, u32, source_m * @sizeOf(f32));
    while (payload.items.len < full_payload_header_bytes) payload.append(allocator, 0) catch return error.Payload;
    payload.appendSlice(allocator, input) catch return error.Payload;
    payload.appendSlice(allocator, weights) catch return error.Payload;
    for (reference) |value| try appendFloat(&payload, allocator, value);
    try appendInt(&payload, allocator, u64, fnv1a64(payload.items));
    const temporary = std.fmt.allocPrint(allocator, "{s}.tmp", .{path}) catch return error.Payload;
    defer allocator.free(temporary);
    defer std.Io.Dir.cwd().deleteFile(io, temporary) catch {};
    if (std.fs.path.dirname(path)) |parent| std.Io.Dir.cwd().createDirPath(io, parent) catch return error.Payload;
    var output = std.Io.Dir.cwd().createFile(io, temporary, .{ .truncate = true }) catch return error.Payload;
    defer output.close(io);
    output.writePositionalAll(io, payload.items, 0) catch return error.Payload;
    output.sync(io) catch return error.Payload;
    std.Io.Dir.cwd().rename(temporary, std.Io.Dir.cwd(), path, io) catch return error.Payload;
}

fn writeFidelityPreparedPayload(path: []const u8, model_sha: [64]u8, trellis: st.Tensor, suh: st.Tensor, svh: st.Tensor, words: usize, allocator: std.mem.Allocator, io: std.Io) Failure!void {
    const weight_count = source_k * source_m;
    const input = allocator.alloc(u8, source_k) catch return error.Payload;
    defer allocator.free(input);
    const weights = allocator.alloc(u8, weight_count) catch return error.Payload;
    defer allocator.free(weights);
    const quantized_reference = allocator.alloc(f32, source_m) catch return error.Payload;
    defer allocator.free(quantized_reference);
    const original_reference = allocator.alloc(f32, source_m) catch return error.Payload;
    defer allocator.free(original_reference);
    const trellis_values = allocator.alloc(i16, trellis.numel()) catch return error.Payload;
    defer allocator.free(trellis_values);
    @memcpy(std.mem.sliceAsBytes(trellis_values), trellis.bytes);
    const suh_values = allocator.alloc(f16, source_k) catch return error.Payload;
    defer allocator.free(suh_values);
    @memcpy(std.mem.sliceAsBytes(suh_values), suh.bytes);
    const svh_values = allocator.alloc(f16, source_m) catch return error.Payload;
    defer allocator.free(svh_values);
    @memcpy(std.mem.sliceAsBytes(svh_values), svh.bytes);
    const decoded = allocator.alloc(f64, weight_count) catch return error.Payload;
    defer allocator.free(decoded);
    const scratch = allocator.alloc(f64, 2 * weight_count) catch return error.Payload;
    defer allocator.free(scratch);
    exl3.dequantizeFast(.{ .halves = @intCast(words / 8) }, .mcg, trellis_values, trellis.dim(0), trellis.dim(1), suh_values, svh_values, decoded, scratch);
    var max_weight: f64 = 0;
    for (decoded) |value| {
        if (!std.math.isFinite(value)) return error.Model;
        max_weight = @max(max_weight, @abs(value));
    }
    if (max_weight == 0) return error.Model;
    const input_scale: f64 = 1.0 / 127.0;
    const weight_scale: f64 = max_weight / 127.0;
    for (input, 0..) |*entry, index| entry.* = @intCast((index * 73 + 19) % 255 + 1);
    for (decoded, 0..) |value, index| weights[index] = try symmetricQuantize(value, weight_scale);
    var quantized_identity: std.ArrayList(u8) = .empty;
    defer quantized_identity.deinit(allocator);
    quantized_identity.ensureTotalCapacity(allocator, source_k + weight_count + source_m * 4 + 24) catch return error.Payload;
    quantized_identity.appendSlice(allocator, input) catch return error.Payload;
    quantized_identity.appendSlice(allocator, weights) catch return error.Payload;
    var max_quantized_reference: f64 = 0;
    for (0..source_m) |m| {
        var quantized_dot: i32 = 0;
        var original_dot: f64 = 0;
        for (0..source_k) |k| {
            const activation = @as(f64, @floatFromInt(@as(i32, input[k]) - 128)) * input_scale;
            quantized_dot += (@as(i32, input[k]) - 128) * (@as(i32, weights[k * source_m + m]) - 128);
            original_dot += activation * decoded[k * source_m + m];
        }
        const quantized_value = @as(f64, @floatFromInt(quantized_dot)) * input_scale * weight_scale;
        if (!std.math.isFinite(original_dot) or !std.math.isFinite(quantized_value)) return error.Model;
        quantized_reference[m] = @floatCast(quantized_value);
        original_reference[m] = @floatCast(original_dot);
        max_quantized_reference = @max(max_quantized_reference, @abs(quantized_value));
        try appendInt(&quantized_identity, allocator, i32, quantized_dot);
    }
    if (max_quantized_reference == 0) return error.Model;
    const output_scale: f64 = max_quantized_reference / 100.0;
    for (quantized_reference) |value| if (@as(f64, value) / output_scale < -127 or @as(f64, value) / output_scale > 127) return error.Model;
    for ([_]f64{ input_scale, weight_scale, output_scale }) |scale| try appendInt(&quantized_identity, allocator, u64, @bitCast(scale));
    const quantized_identity_sha = sha256Hex(quantized_identity.items);
    const original_identity_sha = sha256Hex(std.mem.sliceAsBytes(original_reference));
    const output_scale_f32: f32 = @floatCast(output_scale);
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(allocator);
    payload.ensureTotalCapacity(allocator, fidelity_payload_header_bytes + source_k + weight_count + 2 * source_m * @sizeOf(f32) + payload_integrity_bytes) catch return error.Payload;
    payload.appendSlice(allocator, fidelity_payload_magic) catch return error.Payload;
    try appendInt(&payload, allocator, u32, fidelity_payload_version);
    try appendInt(&payload, allocator, u32, fidelity_payload_header_bytes);
    try appendInt(&payload, allocator, u32, source_k);
    try appendInt(&payload, allocator, u32, source_m);
    try appendFloat(&payload, allocator, @floatCast(input_scale));
    try appendFloat(&payload, allocator, @floatCast(weight_scale));
    try appendFloat(&payload, allocator, output_scale_f32);
    try appendFloat(&payload, allocator, output_scale_f32 * 1.5);
    try appendInt(&payload, allocator, i32, -128);
    try appendInt(&payload, allocator, i32, -128);
    try appendInt(&payload, allocator, i32, -128);
    payload.appendSlice(allocator, &model_sha) catch return error.Payload;
    payload.appendSlice(allocator, full_tensor_identity) catch return error.Payload;
    payload.appendSlice(allocator, &quantized_identity_sha) catch return error.Payload;
    payload.appendSlice(allocator, &original_identity_sha) catch return error.Payload;
    try appendInt(&payload, allocator, u32, source_k);
    try appendInt(&payload, allocator, u32, weight_count);
    try appendInt(&payload, allocator, u32, source_m * @sizeOf(f32));
    try appendInt(&payload, allocator, u32, source_m * @sizeOf(f32));
    while (payload.items.len < fidelity_payload_header_bytes) payload.append(allocator, 0) catch return error.Payload;
    payload.appendSlice(allocator, input) catch return error.Payload;
    payload.appendSlice(allocator, weights) catch return error.Payload;
    for (quantized_reference) |value| try appendFloat(&payload, allocator, value);
    for (original_reference) |value| try appendFloat(&payload, allocator, value);
    try appendInt(&payload, allocator, u64, fnv1a64(payload.items));
    const temporary = std.fmt.allocPrint(allocator, "{s}.tmp", .{path}) catch return error.Payload;
    defer allocator.free(temporary);
    defer std.Io.Dir.cwd().deleteFile(io, temporary) catch {};
    if (std.fs.path.dirname(path)) |parent| std.Io.Dir.cwd().createDirPath(io, parent) catch return error.Payload;
    var output = std.Io.Dir.cwd().createFile(io, temporary, .{ .truncate = true }) catch return error.Payload;
    defer output.close(io);
    output.writePositionalAll(io, payload.items, 0) catch return error.Payload;
    output.sync(io) catch return error.Payload;
    std.Io.Dir.cwd().rename(temporary, std.Io.Dir.cwd(), path, io) catch return error.Payload;
}
fn writeChannelwisePreparedPayload(path: []const u8, model_sha: [64]u8, trellis: st.Tensor, suh: st.Tensor, svh: st.Tensor, words: usize, allocator: std.mem.Allocator, io: std.Io) Failure!void {
    const weight_count = source_k * source_m;
    const input = allocator.alloc(u8, source_k) catch return error.Payload;
    defer allocator.free(input);
    const weights = allocator.alloc(u8, weight_count) catch return error.Payload;
    defer allocator.free(weights);
    const scales = allocator.alloc(f32, source_m) catch return error.Payload;
    defer allocator.free(scales);
    const offsets = allocator.alloc(i32, source_m) catch return error.Payload;
    defer allocator.free(offsets);
    const quantized_reference = allocator.alloc(f32, source_m) catch return error.Payload;
    defer allocator.free(quantized_reference);
    const original_reference = allocator.alloc(f32, source_m) catch return error.Payload;
    defer allocator.free(original_reference);
    const trellis_values = allocator.alloc(i16, trellis.numel()) catch return error.Payload;
    defer allocator.free(trellis_values);
    @memcpy(std.mem.sliceAsBytes(trellis_values), trellis.bytes);
    const suh_values = allocator.alloc(f16, source_k) catch return error.Payload;
    defer allocator.free(suh_values);
    @memcpy(std.mem.sliceAsBytes(suh_values), suh.bytes);
    const svh_values = allocator.alloc(f16, source_m) catch return error.Payload;
    defer allocator.free(svh_values);
    @memcpy(std.mem.sliceAsBytes(svh_values), svh.bytes);
    const decoded = allocator.alloc(f64, weight_count) catch return error.Payload;
    defer allocator.free(decoded);
    const scratch = allocator.alloc(f64, 2 * weight_count) catch return error.Payload;
    defer allocator.free(scratch);
    exl3.dequantizeFast(.{ .halves = @intCast(words / 8) }, .mcg, trellis_values, trellis.dim(0), trellis.dim(1), suh_values, svh_values, decoded, scratch);
    const input_scale: f64 = 1.0 / 127.0;
    for (input, 0..) |*entry, index| entry.* = @intCast((index * 73 + 19) % 255 + 1);
    var max_reference: f64 = 0;
    for (0..source_m) |m| {
        var maximum: f64 = 0;
        for (0..source_k) |k| {
            const value = decoded[k * source_m + m];
            if (!std.math.isFinite(value)) return error.Model;
            maximum = @max(maximum, @abs(value));
        }
        const scale = if (maximum > 0) maximum / 127.0 else 1.0;
        scales[m] = @floatCast(scale);
        offsets[m] = -128;
        var quantized_dot: i32 = 0;
        var original_dot: f64 = 0;
        for (0..source_k) |k| {
            const index = k * source_m + m;
            weights[index] = try symmetricQuantize(decoded[index], scale);
            const activation = @as(f64, @floatFromInt(@as(i32, input[k]) - 128)) * input_scale;
            quantized_dot += (@as(i32, input[k]) - 128) * (@as(i32, weights[index]) - 128);
            original_dot += activation * decoded[index];
        }
        const quantized_value = @as(f64, @floatFromInt(quantized_dot)) * input_scale * scale;
        if (!std.math.isFinite(quantized_value) or !std.math.isFinite(original_dot)) return error.Model;
        quantized_reference[m] = @floatCast(quantized_value);
        original_reference[m] = @floatCast(original_dot);
        max_reference = @max(max_reference, @abs(quantized_value));
    }
    if (max_reference == 0) return error.Model;
    const output_scale: f64 = max_reference / 100.0;
    for (quantized_reference) |value| if (@as(f64, value) / output_scale < -127 or @as(f64, value) / output_scale > 127) return error.Model;
    var quantized_identity: std.ArrayList(u8) = .empty;
    defer quantized_identity.deinit(allocator);
    quantized_identity.appendSlice(allocator, input) catch return error.Payload;
    quantized_identity.appendSlice(allocator, weights) catch return error.Payload;
    quantized_identity.appendSlice(allocator, std.mem.sliceAsBytes(scales)) catch return error.Payload;
    quantized_identity.appendSlice(allocator, std.mem.sliceAsBytes(offsets)) catch return error.Payload;
    for (quantized_reference) |value| try appendFloat(&quantized_identity, allocator, value);
    const quantized_identity_sha = sha256Hex(quantized_identity.items);
    const original_identity_sha = sha256Hex(std.mem.sliceAsBytes(original_reference));
    const output_scale_f32: f32 = @floatCast(output_scale);
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(allocator);
    payload.appendSlice(allocator, channelwise_payload_magic) catch return error.Payload;
    try appendInt(&payload, allocator, u32, channelwise_payload_version);
    try appendInt(&payload, allocator, u32, channelwise_payload_header_bytes);
    try appendInt(&payload, allocator, u32, source_k);
    try appendInt(&payload, allocator, u32, source_m);
    try appendFloat(&payload, allocator, @floatCast(input_scale));
    try appendFloat(&payload, allocator, 1.0);
    try appendFloat(&payload, allocator, output_scale_f32);
    try appendFloat(&payload, allocator, output_scale_f32 * 1.5);
    try appendInt(&payload, allocator, i32, -128);
    try appendInt(&payload, allocator, i32, 0);
    try appendInt(&payload, allocator, i32, -128);
    payload.appendSlice(allocator, &model_sha) catch return error.Payload;
    payload.appendSlice(allocator, full_tensor_identity) catch return error.Payload;
    payload.appendSlice(allocator, &quantized_identity_sha) catch return error.Payload;
    payload.appendSlice(allocator, &original_identity_sha) catch return error.Payload;
    try appendInt(&payload, allocator, u32, source_k);
    try appendInt(&payload, allocator, u32, weight_count);
    try appendInt(&payload, allocator, u32, source_m * @sizeOf(f32));
    try appendInt(&payload, allocator, u32, source_m * @sizeOf(f32));
    try appendInt(&payload, allocator, u32, source_m);
    try appendInt(&payload, allocator, u32, source_m * @sizeOf(i32));
    try appendInt(&payload, allocator, u32, 1);
    try appendInt(&payload, allocator, u32, 1);
    while (payload.items.len < channelwise_payload_header_bytes) payload.append(allocator, 0) catch return error.Payload;
    payload.appendSlice(allocator, input) catch return error.Payload;
    payload.appendSlice(allocator, weights) catch return error.Payload;
    for (quantized_reference) |value| try appendFloat(&payload, allocator, value);
    for (original_reference) |value| try appendFloat(&payload, allocator, value);
    payload.appendSlice(allocator, std.mem.sliceAsBytes(scales)) catch return error.Payload;
    payload.appendSlice(allocator, std.mem.sliceAsBytes(offsets)) catch return error.Payload;
    try appendInt(&payload, allocator, u64, fnv1a64(payload.items));
    const temporary = std.fmt.allocPrint(allocator, "{s}.tmp", .{path}) catch return error.Payload;
    defer allocator.free(temporary);
    defer std.Io.Dir.cwd().deleteFile(io, temporary) catch {};
    if (std.fs.path.dirname(path)) |parent| std.Io.Dir.cwd().createDirPath(io, parent) catch return error.Payload;
    var output = std.Io.Dir.cwd().createFile(io, temporary, .{ .truncate = true }) catch return error.Payload;
    defer output.close(io);
    output.writePositionalAll(io, payload.items, 0) catch return error.Payload;
    output.sync(io) catch return error.Payload;
    std.Io.Dir.cwd().rename(temporary, std.Io.Dir.cwd(), path, io) catch return error.Payload;
}

fn run(init: std.process.Init) Failure!void {
    const args = init.minimal.args.toSlice(init.arena.allocator()) catch return error.Usage;
    const preparing = args.len == 4 and std.mem.eql(u8, args[1], "prepare");
    const preparing_full = args.len == 4 and std.mem.eql(u8, args[1], "prepare-full");
    const preparing_fidelity = args.len == 4 and std.mem.eql(u8, args[1], "prepare-fidelity");
    const preparing_channelwise = args.len == 4 and std.mem.eql(u8, args[1], "prepare-channelwise");
    if ((!(preparing or preparing_full or preparing_fidelity or preparing_channelwise) and args.len != 5) or ((preparing or preparing_full or preparing_fidelity or preparing_channelwise) and args.len != 4)) return error.Usage;
    const allocator = init.gpa;
    const model_path = if (preparing or preparing_full or preparing_fidelity or preparing_channelwise) args[2] else args[1];
    var file = st.File.open(allocator, init.io, model_path) catch return error.Model;
    defer file.close(init.io);
    const model_sha = sha256Hex(file.map.memory);
    if (!std.mem.eql(u8, &model_sha, model_sha256_hex)) return error.Model;

    const trellis = try require(&file, "trellis", .i16);
    const suh = try require(&file, "suh", .f16);
    const svh = try require(&file, "svh", .f16);
    const mcg = try require(&file, "mcg", .i32);
    if (trellis.rank != 3 or trellis.dim(0) * 16 != source_k or trellis.dim(1) * 16 != source_m or
        trellis.dim(2) == 0 or trellis.dim(2) % 8 != 0 or suh.rank != 1 or svh.rank != 1 or
        suh.dim(0) != source_k or svh.dim(0) != source_m or mcg.rank != 0 or
        mcg.bytes.len != 4 or std.mem.readInt(u32, mcg.bytes[0..4], .little) != exl3.McgMul)
        return error.Model;

    const words = trellis.dim(2);
    const tensor_identity = projectionTensorIdentity(trellis, suh, svh, mcg);
    if (!std.mem.eql(u8, &tensor_identity, full_tensor_identity)) return error.Model;
    if (preparing_channelwise) {
        try writeChannelwisePreparedPayload(args[3], model_sha, trellis, suh, svh, words, allocator, init.io);
        std.debug.print(
            "{{\"tensorfold\":\"qualcomm-exl3-qnn-projection\",\"mode\":\"prepare-channelwise\",\"recipe\":\"axis-u8-output-channel\",\"model_sha256\":\"{s}\",\"tensor\":\"{s}\",\"source_shape\":[2048,6144],\"scale_count\":6144,\"axis\":1,\"payload\":\"{s}\",\"passed\":true}}\n",
            .{ model_sha[0..], projection_name, args[3] },
        );
        return;
    }
    if (preparing_fidelity) {
        try writeFidelityPreparedPayload(args[3], model_sha, trellis, suh, svh, words, allocator, init.io);
        std.debug.print(
            "{{\"tensorfold\":\"qualcomm-exl3-qnn-projection\",\"mode\":\"prepare-fidelity\",\"model_sha256\":\"{s}\",\"tensor\":\"{s}\",\"source_shape\":[2048,6144],\"payload\":\"{s}\",\"passed\":true}}\n",
            .{ model_sha[0..], projection_name, args[3] },
        );
        return;
    }
    if (preparing_full) {
        try writeFullPreparedPayload(args[3], model_sha, trellis, suh, svh, words, allocator, init.io);
        std.debug.print(
            "{{\"tensorfold\":\"qualcomm-exl3-qnn-projection\",\"mode\":\"prepare-full\",\"model_sha256\":\"{s}\",\"tensor\":\"{s}\",\"source_shape\":[2048,6144],\"payload\":\"{s}\",\"passed\":true}}\n",
            .{ model_sha[0..], projection_name, args[3] },
        );
        return;
    }
    const all_trellis = std.mem.bytesAsSlice(i16, trellis.bytes);
    const selected = allocator.alloc(i16, (block / 16) * (block / 16) * words) catch return error.Model;
    defer allocator.free(selected);
    for (0..block / 16) |tile_row| for (0..block / 16) |tile_col| {
        const source = (tile_row * trellis.dim(1) + tile_col) * words;
        const destination = (tile_row * (block / 16) + tile_col) * words;
        @memcpy(selected[destination..][0..words], all_trellis[source..][0..words]);
    };
    const selected_suh = allocator.alloc(f16, block) catch return error.Model;
    defer allocator.free(selected_suh);
    const selected_svh = allocator.alloc(f16, block) catch return error.Model;
    defer allocator.free(selected_svh);
    @memcpy(std.mem.sliceAsBytes(selected_suh), suh.bytes[0 .. block * @sizeOf(f16)]);
    @memcpy(std.mem.sliceAsBytes(selected_svh), svh.bytes[0 .. block * @sizeOf(f16)]);
    const decoded = allocator.alloc(f64, block * block) catch return error.Model;
    defer allocator.free(decoded);
    const scratch = allocator.alloc(f64, 2 * block * block) catch return error.Model;
    defer allocator.free(scratch);
    exl3.dequantizeFast(.{ .halves = @intCast(words / 8) }, .mcg, selected, block / 16, block / 16, selected_suh, selected_svh, decoded, scratch);
    const quantized = try quantizeProjection(decoded, allocator);
    defer allocator.free(quantized.input);
    defer allocator.free(quantized.weights);
    defer allocator.free(quantized.reference);
    if (preparing) {
        try writePreparedPayload(args[3], model_sha, quantized, allocator, init.io);
        std.debug.print(
            "{{\"tensorfold\":\"qualcomm-exl3-qnn-projection\",\"mode\":\"prepare\",\"model_sha256\":\"{s}\",\"tensor\":\"{s}\",\"block_sha256\":\"{s}\",\"payload_sha256\":\"{s}\",\"payload\":\"{s}\",\"passed\":true}}\n",
            .{ model_sha[0..], projection_name, quantized.block_sha256[0..], quantized.payload_sha256[0..], args[3] },
        );
        return;
    }

    var bridge = std.DynLib.open(args[4]) catch return error.Bridge;
    defer bridge.close();
    const execute = bridge.lookup(Execute, "tf_qnn_execute_u8_matmul") orelse return error.Bridge;
    const output = allocator.alloc(u8, block) catch return error.Bridge;
    defer allocator.free(output);
    var result: BridgeResult = .{ .status = -99, .execution_ms = 0, .max_abs_error = std.math.nan(f64) };
    const request: BridgeRequest = .{ .native_dir = args[2].ptr, .stub_name = args[3].ptr, .input = quantized.input.ptr, .weights = quantized.weights.ptr, .reference = quantized.reference.ptr, .output = output.ptr, .k = block, .m = block, .input_scale = quantized.input_scale, .input_offset = -128, .weight_scale = quantized.weight_scale, .weight_offset = -128, .output_scale = quantized.output_scale, .output_offset = -128, .tolerance = quantized.output_scale * 1.5 };
    const status = execute(&request, &result);
    const passed = status == 0 and result.status == 0 and std.math.isFinite(result.max_abs_error) and
        result.max_abs_error <= request.tolerance;
    std.debug.print(
        "{{\"tensorfold\":\"qualcomm-exl3-qnn-projection\",\"model_sha256\":\"{s}\",\"tensor\":\"{s}\",\"source_shape\":[{d},{d}],\"block\":[0,0,128,128],\"block_sha256\":\"{s}\",\"payload_sha256\":\"{s}\",\"scales\":[{d:.9},{d:.9},{d:.9}],\"bridge_status\":{d},\"qnn_status\":{d},\"max_abs_error\":{d:.9},\"tolerance\":{d:.9},\"execution_ms\":{d:.6},\"passed\":{s}}}\n",
        .{ model_sha[0..], projection_name, source_k, source_m, quantized.block_sha256[0..], quantized.payload_sha256[0..], quantized.input_scale, quantized.weight_scale, quantized.output_scale, status, result.status, result.max_abs_error, request.tolerance, result.execution_ms, if (passed) "true" else "false" },
    );
    if (status != 0 or result.status != 0)
        return if (status == -112 or result.status == -112) error.Parity else error.Bridge;
    if (!passed) return error.Parity;
}

pub fn main(init: std.process.Init) !u8 {
    run(init) catch |failure| switch (failure) {
        error.Usage => {
            std.debug.print("usage: tensorfold-qualcomm-exl3-qnn-projection MODEL_SAFETENSORS NATIVE_DIR STUB_NAME QNN_BRIDGE_SO\n       tensorfold-qualcomm-exl3-qnn-projection prepare MODEL_SAFETENSORS APP_PRIVATE_PAYLOAD\n       tensorfold-qualcomm-exl3-qnn-projection prepare-full MODEL_SAFETENSORS APP_PRIVATE_PAYLOAD\n       tensorfold-qualcomm-exl3-qnn-projection prepare-fidelity MODEL_SAFETENSORS APP_PRIVATE_PAYLOAD\n       tensorfold-qualcomm-exl3-qnn-projection prepare-channelwise MODEL_SAFETENSORS APP_PRIVATE_PAYLOAD\n", .{});
            return 64;
        },
        error.Model => return 4,
        error.Bridge => return 5,
        error.Parity => return 6,
        error.Payload => return 7,
    };
    return 0;
}

test "generator-contract activation is stable" {
    var decoded: [block * block]f64 = @splat(1.0);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const q = try quantizeProjection(&decoded, arena.allocator());
    try std.testing.expectEqualSlices(u8, &.{ 20, 93, 166, 239, 57 }, q.input[0..5]);
    try std.testing.expectEqual(@as(u8, 255), q.weights[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 127.0), q.input_scale, 1e-7);
}

test "prepared payload format is fixed and integrity is deterministic" {
    try std.testing.expectEqual(@as(u32, 256), payload_header_bytes);
    try std.testing.expectEqual(@as(usize, 180), payload_identity_offset);
    try std.testing.expectEqual(@as(u64, 0xced1370968eccfa1), fnv1a64(payload_magic));
}

test "full prepared payload has fixed source dimensions and a distinct schema" {
    try std.testing.expectEqual(@as(u32, 2), full_payload_version);
    try std.testing.expectEqual(@as(u32, 320), full_payload_header_bytes);
    try std.testing.expectEqual(@as(usize, 12_582_912), source_k * source_m);
    try std.testing.expectEqualStrings("TFQNNLF2", full_payload_magic);
    try std.testing.expectEqual(@as(usize, 64), full_tensor_identity.len);
}

test "fidelity payload binds two references in a versioned schema" {
    try std.testing.expectEqual(@as(u32, 3), fidelity_payload_version);
    try std.testing.expectEqual(@as(u32, 384), fidelity_payload_header_bytes);
    try std.testing.expectEqualStrings("TFQNNLF3", fidelity_payload_magic);
}

test "channelwise payload fixes output-axis scale schema" {
    try std.testing.expectEqual(@as(u32, 4), channelwise_payload_version);
    try std.testing.expectEqual(@as(u32, 448), channelwise_payload_header_bytes);
    try std.testing.expectEqualStrings("TFQNNLC4", channelwise_payload_magic);
    try std.testing.expectEqual(@as(usize, 6144), source_m);
}
