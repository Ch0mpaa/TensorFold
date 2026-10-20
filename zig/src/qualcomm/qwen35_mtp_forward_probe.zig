//! Compose the complete Qwen3.5-4B native-MTP block on Qualcomm HTP and
//! compare it with a scalar BF16-weight oracle over a short synthetic stream.

const std = @import("std");
const st = @import("safetensors");

extern fn tf_qualcomm_monotonic_ms() f64;
extern fn tf_qualcomm_backend_create(backend_name: [*:0]const u8, cpu_threads: i32) ?*anyopaque;
extern fn tf_qualcomm_backend_destroy(context: ?*anyopaque) void;
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
extern fn tf_qualcomm_tied_head_f16(
    embedding: [*]const u16,
    state: [*]const f32,
    vocab_size: usize,
    hidden_size: usize,
    cpu_threads: i32,
    top_ids: *[5]usize,
    top_logits: *[5]f32,
) c_int;

const hidden = 2560;
const intermediate = 9216;
const vocab = 248320;
const attention_heads = 16;
const kv_heads = 4;
const attention_width = 256;
const attention_query_width = attention_heads * attention_width;
const kv_width = kv_heads * attention_width;
const rope_width = 64;
const eps: f32 = 1e-6;

const Matrix = enum(usize) {
    fc,
    q,
    k,
    v,
    o,
    gate,
    up,
    down,

    fn name(self: Matrix) []const u8 {
        return switch (self) {
            .fc => "mtp.fc.weight",
            .q => "mtp.layers.0.self_attn.q_proj.weight",
            .k => "mtp.layers.0.self_attn.k_proj.weight",
            .v => "mtp.layers.0.self_attn.v_proj.weight",
            .o => "mtp.layers.0.self_attn.o_proj.weight",
            .gate => "mtp.layers.0.mlp.gate_proj.weight",
            .up => "mtp.layers.0.mlp.up_proj.weight",
            .down => "mtp.layers.0.mlp.down_proj.weight",
        };
    }
};

fn bf16Value(bytes: []const u8, index: usize) f32 {
    const bits = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
    return @bitCast(@as(u32, bits) << 16);
}

fn f16Value(bytes: []const u8, index: usize) f32 {
    const bits = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
    return @floatCast(@as(f16, @bitCast(bits)));
}

fn value(tensor: st.Tensor, index: usize) f32 {
    return switch (tensor.dtype) {
        .bf16 => bf16Value(tensor.bytes, index),
        .f16 => f16Value(tensor.bytes, index),
        else => unreachable,
    };
}

fn roundF16(x: f32) f32 {
    return @floatCast(@as(f16, @floatCast(x)));
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

fn require(file: *const st.File, name: []const u8, dtype: st.DType, shape: []const usize) !st.Tensor {
    const tensor = file.get(name) orelse return error.MissingTensor;
    if (!tensor.is(dtype, shape)) return error.UnexpectedTensor;
    return tensor;
}

pub fn validateMtp(file: *const st.File) !void {
    if (file.names.count() != 15) return error.UnexpectedTensorCount;
    _ = try require(file, "mtp.fc.weight", .bf16, &.{ hidden, 2 * hidden });
    _ = try require(file, "mtp.layers.0.input_layernorm.weight", .bf16, &.{hidden});
    _ = try require(file, "mtp.layers.0.mlp.down_proj.weight", .bf16, &.{ hidden, intermediate });
    _ = try require(file, "mtp.layers.0.mlp.gate_proj.weight", .bf16, &.{ intermediate, hidden });
    _ = try require(file, "mtp.layers.0.mlp.up_proj.weight", .bf16, &.{ intermediate, hidden });
    _ = try require(file, "mtp.layers.0.post_attention_layernorm.weight", .bf16, &.{hidden});
    _ = try require(file, "mtp.layers.0.self_attn.k_norm.weight", .bf16, &.{attention_width});
    _ = try require(file, "mtp.layers.0.self_attn.k_proj.weight", .bf16, &.{ kv_width, hidden });
    _ = try require(file, "mtp.layers.0.self_attn.o_proj.weight", .bf16, &.{ hidden, attention_query_width });
    _ = try require(file, "mtp.layers.0.self_attn.q_norm.weight", .bf16, &.{attention_width});
    _ = try require(file, "mtp.layers.0.self_attn.q_proj.weight", .bf16, &.{ 2 * attention_query_width, hidden });
    _ = try require(file, "mtp.layers.0.self_attn.v_proj.weight", .bf16, &.{ kv_width, hidden });
    _ = try require(file, "mtp.norm.weight", .bf16, &.{hidden});
    _ = try require(file, "mtp.pre_fc_norm_embedding.weight", .bf16, &.{hidden});
    _ = try require(file, "mtp.pre_fc_norm_hidden.weight", .bf16, &.{hidden});
}

fn rmsNorm(file: *const st.File, name: []const u8, input: []const f32, output: []f32) !void {
    const weight = try require(file, name, .bf16, &.{input.len});
    if (output.len != input.len) return error.UnexpectedTensor;
    var sum: f32 = 0;
    for (input) |x| sum += x * x;
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(input.len)) + eps);
    for (output, 0..) |*out, i| out.* = roundF16(input[i] * inv * (value(weight, i) + 1.0));
}

fn headNorm(file: *const st.File, name: []const u8, input: []const f32, heads: usize, output: []f32) !void {
    if (input.len != output.len or input.len % heads != 0) return error.UnexpectedTensor;
    const width = input.len / heads;
    const weight = try require(file, name, .bf16, &.{width});
    for (0..heads) |head| {
        var sum: f32 = 0;
        for (0..width) |d| {
            const x = input[head * width + d];
            sum += x * x;
        }
        const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(width)) + eps);
        for (0..width) |d| {
            const i = head * width + d;
            output[i] = roundF16(input[i] * inv * (value(weight, d) + 1.0));
        }
    }
}

fn rotateHalf(input: []const f32, output: []f32, position: usize) void {
    @memcpy(output, input);
    for (0..input.len / attention_width) |head| {
        const base = head * attention_width;
        for (0..rope_width / 2) |i| {
            const angle = @as(f32, @floatFromInt(position)) * std.math.pow(f32, 10_000_000.0, -@as(f32, @floatFromInt(i)) / @as(f32, rope_width / 2));
            const c = @cos(angle);
            const s = @sin(angle);
            const a = input[base + i];
            const b = input[base + rope_width / 2 + i];
            output[base + i] = roundF16(a * c - b * s);
            output[base + rope_width / 2 + i] = roundF16(b * c + a * s);
        }
    }
}

fn residualAdd(state: []f32, update: []const f32) void {
    for (state, update) |*x, y| x.* = roundF16(x.* + y);
}

pub const Cache = struct {
    keys: []f32,
    values: []f32,
    positions: usize,

    pub fn init(allocator: std.mem.Allocator, positions: usize) !Cache {
        const keys = try allocator.alloc(f32, positions * kv_width);
        errdefer allocator.free(keys);
        const values = try allocator.alloc(f32, positions * kv_width);
        @memset(keys, 0.0);
        @memset(values, 0.0);
        return .{ .keys = keys, .values = values, .positions = positions };
    }

    pub fn deinit(self: *Cache, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        allocator.free(self.keys);
        self.* = undefined;
    }

    fn key(self: *Cache, position: usize) []f32 {
        return self.keys[position * kv_width ..][0..kv_width];
    }

    fn valueRow(self: *Cache, position: usize) []f32 {
        return self.values[position * kv_width ..][0..kv_width];
    }
};

pub const Runtime = struct {
    allocator: std.mem.Allocator,
    file: *const st.File,
    backend: ?*anyopaque,
    loaded: [8]bool = @splat(false),
    projection_calls: usize = 0,
    projection_uploads: usize = 0,
    projection_compute_ms: f64 = 0,

    fn projectHtp(self: *Runtime, matrix: Matrix, input: []const f32, output: []f32) !void {
        const tensor = self.file.get(matrix.name()) orelse return error.MissingTensor;
        if (tensor.dtype != .bf16 or tensor.rank != 2) return error.UnexpectedTensor;
        const m = tensor.dim(0);
        const k = tensor.dim(1);
        if (input.len == 0 or input.len % k != 0 or output.len != input.len / k * m) return error.UnexpectedTensor;
        const slot = @backingInt(matrix);
        var converted: ?[]u16 = null;
        defer if (converted) |weight| self.allocator.free(weight);
        if (!self.loaded[slot]) {
            const bf16 = std.mem.bytesAsSlice(u16, tensor.bytes);
            const weight = try self.allocator.alloc(u16, bf16.len);
            for (bf16, weight) |bits, *target| target.* = @bitCast(@as(f16, @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)))));
            converted = weight;
        }
        var compute_ms: f64 = 0;
        var uploaded: c_int = 0;
        const status = tf_qualcomm_backend_matmul_q8_f16_source_resident_once(
            self.backend,
            slot,
            k,
            m,
            input.len / k,
            if (converted) |weight| weight.ptr else null,
            input.ptr,
            output.ptr,
            &compute_ms,
            &uploaded,
        );
        if (status != 0) return error.BridgeFailed;
        if (uploaded != 0) {
            self.loaded[slot] = true;
            self.projection_uploads += 1;
        }
        self.projection_calls += 1;
        self.projection_compute_ms += compute_ms;
    }
};

fn projectCpu(file: *const st.File, matrix: Matrix, input: []const f32, output: []f32) !void {
    const tensor = file.get(matrix.name()) orelse return error.MissingTensor;
    if (tensor.dtype != .bf16 or tensor.rank != 2) return error.UnexpectedTensor;
    const m = tensor.dim(0);
    const k = tensor.dim(1);
    if (input.len == 0 or input.len % k != 0 or output.len != input.len / k * m) return error.UnexpectedTensor;
    const rows = input.len / k;
    for (0..rows) |row| for (0..m) |out_row| {
        var sum: f32 = 0;
        for (0..k) |column| sum += bf16Value(tensor.bytes, out_row * k + column) * input[row * k + column];
        output[row * m + out_row] = sum;
    };
}

fn project(comptime use_htp: bool, runtime: *Runtime, matrix: Matrix, input: []const f32, output: []f32) !void {
    if (use_htp) return runtime.projectHtp(matrix, input, output);
    return projectCpu(runtime.file, matrix, input, output);
}

pub fn runStep(
    comptime use_htp: bool,
    scratch: std.mem.Allocator,
    runtime: *Runtime,
    target_embedding: st.Tensor,
    cache: *Cache,
    target_hidden: []const f32,
    token: usize,
    position: usize,
    output: []f32,
) !void {
    if (target_hidden.len != hidden or output.len != hidden or token >= vocab or position >= cache.positions) return error.InvalidArgument;

    const embedding = try scratch.alloc(f32, hidden);
    for (embedding, 0..) |*x, d| x.* = f16Value(target_embedding.bytes, token * hidden + d);
    const hidden_norm = try scratch.alloc(f32, hidden);
    const embedding_norm = try scratch.alloc(f32, hidden);
    try rmsNorm(runtime.file, "mtp.pre_fc_norm_hidden.weight", target_hidden, hidden_norm);
    try rmsNorm(runtime.file, "mtp.pre_fc_norm_embedding.weight", embedding, embedding_norm);
    const joined = try scratch.alloc(f32, 2 * hidden);
    @memcpy(joined[0..hidden], embedding_norm);
    @memcpy(joined[hidden .. 2 * hidden], hidden_norm);
    const state = try scratch.alloc(f32, hidden);
    try project(use_htp, runtime, .fc, joined, state);

    const normed = try scratch.alloc(f32, hidden);
    try rmsNorm(runtime.file, "mtp.layers.0.input_layernorm.weight", state, normed);
    const qraw = try scratch.alloc(f32, 2 * attention_query_width);
    const kraw = try scratch.alloc(f32, kv_width);
    const vraw = try scratch.alloc(f32, kv_width);
    try project(use_htp, runtime, .q, normed, qraw);
    try project(use_htp, runtime, .k, normed, kraw);
    try project(use_htp, runtime, .v, normed, vraw);
    for (qraw) |*x| x.* = roundF16(x.*);
    for (kraw) |*x| x.* = roundF16(x.*);
    for (vraw) |*x| x.* = roundF16(x.*);

    const query = try scratch.alloc(f32, attention_query_width);
    const attention_gate = try scratch.alloc(f32, attention_query_width);
    for (0..attention_heads) |head| for (0..attention_width) |d| {
        query[head * attention_width + d] = qraw[head * 2 * attention_width + d];
        attention_gate[head * attention_width + d] = qraw[head * 2 * attention_width + attention_width + d];
    };
    const query_norm = try scratch.alloc(f32, attention_query_width);
    const key_norm = try scratch.alloc(f32, kv_width);
    try headNorm(runtime.file, "mtp.layers.0.self_attn.q_norm.weight", query, attention_heads, query_norm);
    try headNorm(runtime.file, "mtp.layers.0.self_attn.k_norm.weight", kraw, kv_heads, key_norm);
    const query_rotated = try scratch.alloc(f32, attention_query_width);
    const key_rotated = try scratch.alloc(f32, kv_width);
    rotateHalf(query_norm, query_rotated, position);
    rotateHalf(key_norm, key_rotated, position);
    @memcpy(cache.key(position), key_rotated);
    @memcpy(cache.valueRow(position), vraw);

    const attended = try scratch.alloc(f32, attention_query_width);
    const scores = try scratch.alloc(f32, position + 1);
    for (0..attention_heads) |query_head| {
        const kv_head = query_head / (attention_heads / kv_heads);
        var maximum = -std.math.inf(f32);
        for (0..position + 1) |token_position| {
            const cached_key = cache.key(token_position);
            var score: f32 = 0;
            for (0..attention_width) |d| score += query_rotated[query_head * attention_width + d] * cached_key[kv_head * attention_width + d];
            scores[token_position] = score / 16.0;
            maximum = @max(maximum, scores[token_position]);
        }
        var denominator: f32 = 0;
        for (scores) |*score| {
            score.* = @exp(score.* - maximum);
            denominator += score.*;
        }
        for (0..attention_width) |d| {
            var sum: f32 = 0;
            for (0..position + 1) |token_position| {
                const cached_value = cache.valueRow(token_position);
                sum += scores[token_position] / denominator * cached_value[kv_head * attention_width + d];
            }
            const i = query_head * attention_width + d;
            attended[i] = roundF16(roundF16(sum) * sigmoid(attention_gate[i]));
        }
    }

    const update = try scratch.alloc(f32, hidden);
    try project(use_htp, runtime, .o, attended, update);
    residualAdd(state, update);
    try rmsNorm(runtime.file, "mtp.layers.0.post_attention_layernorm.weight", state, normed);
    const gate = try scratch.alloc(f32, intermediate);
    const up = try scratch.alloc(f32, intermediate);
    const activated = try scratch.alloc(f32, intermediate);
    try project(use_htp, runtime, .gate, normed, gate);
    try project(use_htp, runtime, .up, normed, up);
    for (activated, 0..) |*x, i| x.* = roundF16(silu(roundF16(gate[i])) * roundF16(up[i]));
    try project(use_htp, runtime, .down, activated, update);
    residualAdd(state, update);
    try rmsNorm(runtime.file, "mtp.norm.weight", state, output);
}

/// Prefill the native-MTP layer as one projection batch while retaining the
/// exact causal attention and FP16 rounding used by runStep. The target hidden
/// rows are already shifted: row zero is zero and row i is target row i - 1.
pub fn runBatch(
    comptime use_htp: bool,
    scratch: std.mem.Allocator,
    runtime: *Runtime,
    target_embedding: st.Tensor,
    cache: *Cache,
    target_hidden_rows: []const f32,
    tokens: []const usize,
    start_position: usize,
    output_rows: []f32,
) !void {
    const count = tokens.len;
    if (count == 0 or target_hidden_rows.len != count * hidden or output_rows.len != count * hidden or start_position + count > cache.positions) return error.InvalidArgument;
    const embeddings = try scratch.alloc(f32, count * hidden);
    const hidden_norm = try scratch.alloc(f32, count * hidden);
    const embedding_norm = try scratch.alloc(f32, count * hidden);
    const joined = try scratch.alloc(f32, count * 2 * hidden);
    for (tokens, 0..) |token, row| {
        if (token >= vocab) return error.InvalidArgument;
        for (embeddings[row * hidden ..][0..hidden], 0..) |*x, d| x.* = f16Value(target_embedding.bytes, token * hidden + d);
        try rmsNorm(runtime.file, "mtp.pre_fc_norm_hidden.weight", target_hidden_rows[row * hidden ..][0..hidden], hidden_norm[row * hidden ..][0..hidden]);
        try rmsNorm(runtime.file, "mtp.pre_fc_norm_embedding.weight", embeddings[row * hidden ..][0..hidden], embedding_norm[row * hidden ..][0..hidden]);
        @memcpy(joined[row * 2 * hidden ..][0..hidden], embedding_norm[row * hidden ..][0..hidden]);
        @memcpy(joined[row * 2 * hidden + hidden ..][0..hidden], hidden_norm[row * hidden ..][0..hidden]);
    }
    const states = try scratch.alloc(f32, count * hidden);
    try project(use_htp, runtime, .fc, joined, states);

    const normed = try scratch.alloc(f32, count * hidden);
    for (0..count) |row| try rmsNorm(runtime.file, "mtp.layers.0.input_layernorm.weight", states[row * hidden ..][0..hidden], normed[row * hidden ..][0..hidden]);
    const qraw = try scratch.alloc(f32, count * 2 * attention_query_width);
    const kraw = try scratch.alloc(f32, count * kv_width);
    const vraw = try scratch.alloc(f32, count * kv_width);
    try project(use_htp, runtime, .q, normed, qraw);
    try project(use_htp, runtime, .k, normed, kraw);
    try project(use_htp, runtime, .v, normed, vraw);
    for (qraw) |*x| x.* = roundF16(x.*);
    for (kraw) |*x| x.* = roundF16(x.*);
    for (vraw) |*x| x.* = roundF16(x.*);

    const queries = try scratch.alloc(f32, count * attention_query_width);
    const attention_gates = try scratch.alloc(f32, count * attention_query_width);
    const query_norm = try scratch.alloc(f32, count * attention_query_width);
    const key_norm = try scratch.alloc(f32, count * kv_width);
    const query_rotated = try scratch.alloc(f32, count * attention_query_width);
    const key_rotated = try scratch.alloc(f32, count * kv_width);
    for (0..count) |row| {
        for (0..attention_heads) |head| for (0..attention_width) |d| {
            queries[row * attention_query_width + head * attention_width + d] = qraw[row * 2 * attention_query_width + head * 2 * attention_width + d];
            attention_gates[row * attention_query_width + head * attention_width + d] = qraw[row * 2 * attention_query_width + head * 2 * attention_width + attention_width + d];
        };
        try headNorm(runtime.file, "mtp.layers.0.self_attn.q_norm.weight", queries[row * attention_query_width ..][0..attention_query_width], attention_heads, query_norm[row * attention_query_width ..][0..attention_query_width]);
        try headNorm(runtime.file, "mtp.layers.0.self_attn.k_norm.weight", kraw[row * kv_width ..][0..kv_width], kv_heads, key_norm[row * kv_width ..][0..kv_width]);
        rotateHalf(query_norm[row * attention_query_width ..][0..attention_query_width], query_rotated[row * attention_query_width ..][0..attention_query_width], start_position + row);
        rotateHalf(key_norm[row * kv_width ..][0..kv_width], key_rotated[row * kv_width ..][0..kv_width], start_position + row);
        @memcpy(cache.key(start_position + row), key_rotated[row * kv_width ..][0..kv_width]);
        @memcpy(cache.valueRow(start_position + row), vraw[row * kv_width ..][0..kv_width]);
    }

    const attended = try scratch.alloc(f32, count * attention_query_width);
    const scores = try scratch.alloc(f32, start_position + count);
    for (0..count) |row| {
        const position = start_position + row;
        for (0..attention_heads) |query_head| {
            const kv_head = query_head / (attention_heads / kv_heads);
            var maximum = -std.math.inf(f32);
            for (0..position + 1) |token_position| {
                const cached_key = cache.key(token_position);
                var score: f32 = 0;
                for (0..attention_width) |d| score += query_rotated[row * attention_query_width + query_head * attention_width + d] * cached_key[kv_head * attention_width + d];
                scores[token_position] = score / 16.0;
                maximum = @max(maximum, scores[token_position]);
            }
            var denominator: f32 = 0;
            for (scores[0 .. position + 1]) |*score| {
                score.* = @exp(score.* - maximum);
                denominator += score.*;
            }
            for (0..attention_width) |d| {
                var sum: f32 = 0;
                for (0..position + 1) |token_position| sum += scores[token_position] / denominator * cache.valueRow(token_position)[kv_head * attention_width + d];
                const i = row * attention_query_width + query_head * attention_width + d;
                attended[i] = roundF16(roundF16(sum) * sigmoid(attention_gates[i]));
            }
        }
    }

    const updates = try scratch.alloc(f32, count * hidden);
    try project(use_htp, runtime, .o, attended, updates);
    residualAdd(states, updates);
    for (0..count) |row| try rmsNorm(runtime.file, "mtp.layers.0.post_attention_layernorm.weight", states[row * hidden ..][0..hidden], normed[row * hidden ..][0..hidden]);
    const gates = try scratch.alloc(f32, count * intermediate);
    const ups = try scratch.alloc(f32, count * intermediate);
    const activated = try scratch.alloc(f32, count * intermediate);
    try project(use_htp, runtime, .gate, normed, gates);
    try project(use_htp, runtime, .up, normed, ups);
    for (activated, 0..) |*x, i| x.* = roundF16(silu(roundF16(gates[i])) * roundF16(ups[i]));
    try project(use_htp, runtime, .down, activated, updates);
    residualAdd(states, updates);
    for (0..count) |row| try rmsNorm(runtime.file, "mtp.norm.weight", states[row * hidden ..][0..hidden], output_rows[row * hidden ..][0..hidden]);
}

fn topFive(embedding: st.Tensor, state: []const f32, threads: i32) !struct { ids: [5]usize, logits: [5]f32 } {
    const words: [*]const u16 = @ptrCast(@alignCast(embedding.bytes.ptr));
    var ids: [5]usize = undefined;
    var logits: [5]f32 = undefined;
    if (tf_qualcomm_tied_head_f16(words, state.ptr, vocab, hidden, threads, &ids, &logits) != 0) return error.HeadFailed;
    return .{ .ids = ids, .logits = logits };
}

fn parseTokens(allocator: std.mem.Allocator, encoded: []const u8) ![]usize {
    const count = std.mem.count(u8, encoded, ",") + 1;
    if (count == 0 or count > 128) return error.InvalidArgument;
    const tokens = try allocator.alloc(usize, count);
    var split = std.mem.splitScalar(u8, encoded, ',');
    var i: usize = 0;
    while (split.next()) |part| : (i += 1) {
        if (part.len == 0 or i >= count) return error.InvalidArgument;
        tokens[i] = try std.fmt.parseInt(usize, part, 10);
        if (tokens[i] >= vocab) return error.InvalidArgument;
    }
    if (i != count) return error.InvalidArgument;
    return tokens;
}

fn loadFixtureHidden(bytes: []const u8, position: usize, output: []f32) void {
    if (position == 0) {
        @memset(output, 0.0);
        return;
    }
    const row = bytes[(position - 1) * hidden * @sizeOf(f32) ..][0 .. hidden * @sizeOf(f32)];
    for (output, 0..) |*x, d| x.* = @bitCast(std.mem.readInt(u32, row[d * 4 ..][0..4], .little));
}

const ChainResult = struct {
    rounds: usize = 0,
    draft_steps: usize = 0,
    offered: usize = 0,
    accepted: usize = 0,
    trace_hash: u64 = 14695981039346656037,
};

fn simulateChain(
    comptime use_htp: bool,
    allocator: std.mem.Allocator,
    runtime: *Runtime,
    embedding: st.Tensor,
    fixture: []const u8,
    tokens: []const usize,
    prompt_len: usize,
    depth: usize,
    threads: i32,
) !ChainResult {
    var result: ChainResult = .{};
    var current = prompt_len;
    const target_hidden = try allocator.alloc(f32, hidden);
    defer allocator.free(target_hidden);
    const state_a = try allocator.alloc(f32, hidden);
    defer allocator.free(state_a);
    const state_b = try allocator.alloc(f32, hidden);
    defer allocator.free(state_b);
    var scratch: std.heap.ArenaAllocator = .init(allocator);
    defer scratch.deinit();

    while (current + 1 < tokens.len) {
        // Rebuild the native-MTP prefix from target-confirmed states. This is
        // deliberately conservative: no speculative KV row survives a round.
        var cache = try Cache.init(allocator, tokens.len + depth);
        defer cache.deinit(allocator);
        for (0..current) |position| {
            loadFixtureHidden(fixture, position, target_hidden);
            _ = scratch.reset(.retain_capacity);
            try runStep(use_htp, scratch.allocator(), runtime, embedding, &cache, target_hidden, tokens[position], position, state_a);
        }

        loadFixtureHidden(fixture, current, target_hidden);
        var input_token = tokens[current];
        var input_hidden: []const f32 = target_hidden;
        var output = state_a;
        const available = @min(depth, tokens.len - current - 1);
        var accepted_prefix: usize = 0;
        var prefix_open = true;
        var compared: usize = 0;
        for (0..available) |draft_index| {
            _ = scratch.reset(.retain_capacity);
            try runStep(use_htp, scratch.allocator(), runtime, embedding, &cache, input_hidden, input_token, current + draft_index, output);
            const top = try topFive(embedding, output, threads);
            const expected = tokens[current + draft_index + 1];
            const comparable = prefix_open;
            const matched = comparable and top.ids[0] == expected;
            if (comparable) {
                compared += 1;
                if (matched) {
                    accepted_prefix += 1;
                } else {
                    prefix_open = false;
                }
            }
            result.draft_steps += 1;
            result.trace_hash ^= @as(u64, top.ids[0]);
            result.trace_hash *%= 1099511628211;
            std.debug.print(
                "{{\"tensorfold\":\"qwen35-mtp-chain-step\",\"backend\":\"{s}\",\"round\":{d},\"draft_index\":{d},\"position\":{d},\"input_token\":{d},\"proposed_token\":{d},\"expected_token\":{d},\"comparable\":{},\"matched\":{}}}\n",
                .{ if (use_htp) "htp" else "cpu", result.rounds, draft_index, current + draft_index, input_token, top.ids[0], expected, comparable, matched },
            );
            input_token = top.ids[0];
            input_hidden = output;
            output = if (output.ptr == state_a.ptr) state_b else state_a;
        }
        result.rounds += 1;
        result.accepted += accepted_prefix;
        result.offered += compared;
        // Verification always lands one target token in addition to the
        // accepted proposal prefix, or one extra target token when all drafts
        // match. That token becomes the pending input for the next round.
        current += accepted_prefix + 1;
    }
    return result;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 6 and args.len != 8 and args.len != 9) {
        std.debug.print("usage: {s} cpu|htp TARGET_SAFETENSORS MTP_SAFETENSORS CPU_THREADS TOKEN_IDS [TARGET_HIDDEN_F32 PROMPT_LEN [CHAIN_DEPTH]]\n", .{args[0]});
        return 64;
    }
    if (!std.mem.eql(u8, args[1], "cpu") and !std.mem.eql(u8, args[1], "htp")) return error.InvalidBackend;
    const threads = try std.fmt.parseInt(i32, args[4], 10);
    if (threads <= 0) return error.InvalidArgument;
    const tokens = try parseTokens(init.gpa, args[5]);
    defer init.gpa.free(tokens);
    const fixture_mode = args.len >= 8;
    const chain_depth = if (args.len == 9) try std.fmt.parseInt(usize, args[8], 10) else 0;
    if (chain_depth > 16) return error.InvalidArgument;
    const prompt_len = if (fixture_mode) try std.fmt.parseInt(usize, args[7], 10) else tokens.len;
    if (fixture_mode and (prompt_len == 0 or prompt_len >= tokens.len)) return error.InvalidArgument;
    const positions = if (fixture_mode) tokens.len - 1 else tokens.len;
    const hidden_fixture: ?[]u8 = if (fixture_mode)
        try std.Io.Dir.cwd().readFileAlloc(init.io, args[6], init.gpa, .limited(128 * hidden * @sizeOf(f32)))
    else
        null;
    defer if (hidden_fixture) |bytes| init.gpa.free(bytes);
    if (hidden_fixture) |bytes| {
        const expected_bytes = try std.math.mul(usize, tokens.len - 1, hidden * @sizeOf(f32));
        if (bytes.len != expected_bytes) return error.UnexpectedFixtureSize;
    }

    var target = try st.File.open(init.gpa, init.io, args[2]);
    defer target.close(init.io);
    const embedding = try require(&target, "model.language_model.embed_tokens.weight", .f16, &.{ vocab, hidden });
    var mtp = try st.File.open(init.gpa, init.io, args[3]);
    defer mtp.close(init.io);
    try validateMtp(&mtp);
    const backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BackendInitFailed;
    defer tf_qualcomm_backend_destroy(backend);
    var runtime: Runtime = .{ .allocator = init.gpa, .file = &mtp, .backend = backend };
    var cpu_cache = try Cache.init(init.gpa, positions);
    defer cpu_cache.deinit(init.gpa);
    var htp_cache = try Cache.init(init.gpa, positions);
    defer htp_cache.deinit(init.gpa);
    const target_hidden = try init.gpa.alloc(f32, hidden);
    defer init.gpa.free(target_hidden);
    const cpu_output = try init.gpa.alloc(f32, hidden);
    defer init.gpa.free(cpu_output);
    const htp_output = try init.gpa.alloc(f32, hidden);
    defer init.gpa.free(htp_output);
    var scratch: std.heap.ArenaAllocator = .init(init.gpa);
    defer scratch.deinit();

    var matching_heads: usize = 0;
    var passing_positions: usize = 0;
    var cpu_draft_accepts: usize = 0;
    var htp_draft_accepts: usize = 0;
    var draft_offered: usize = 0;
    const started = tf_qualcomm_monotonic_ms();
    for (tokens[0..positions], 0..) |token, position| {
        if (hidden_fixture) |bytes| {
            loadFixtureHidden(bytes, position, target_hidden);
        } else {
            for (target_hidden, 0..) |*x, d| {
                const signed: i32 = @intCast((d + position * 17) % 37);
                x.* = @as(f32, @floatFromInt(signed - 18)) / 64.0;
            }
        }
        _ = scratch.reset(.retain_capacity);
        try runStep(false, scratch.allocator(), &runtime, embedding, &cpu_cache, target_hidden, token, position, cpu_output);
        _ = scratch.reset(.retain_capacity);
        try runStep(true, scratch.allocator(), &runtime, embedding, &htp_cache, target_hidden, token, position, htp_output);

        var max_abs: f32 = 0;
        var error_squared: f64 = 0;
        var cpu_squared: f64 = 0;
        var htp_squared: f64 = 0;
        var dot: f64 = 0;
        for (cpu_output, htp_output) |want, got| {
            const delta = got - want;
            max_abs = @max(max_abs, @abs(delta));
            error_squared += @as(f64, delta) * @as(f64, delta);
            cpu_squared += @as(f64, want) * @as(f64, want);
            htp_squared += @as(f64, got) * @as(f64, got);
            dot += @as(f64, want) * @as(f64, got);
        }
        const rmse = @sqrt(error_squared / @as(f64, @floatFromInt(hidden)));
        const cosine = dot / @sqrt(cpu_squared * htp_squared);
        const cpu_top = try topFive(embedding, cpu_output, threads);
        const htp_top = try topFive(embedding, htp_output, threads);
        const head_match = cpu_top.ids[0] == htp_top.ids[0];
        if (head_match) matching_heads += 1;
        const expected_next = if (fixture_mode) tokens[position + 1] else 0;
        const draft_scored = fixture_mode and position >= prompt_len;
        const cpu_draft_match = draft_scored and cpu_top.ids[0] == expected_next;
        const htp_draft_match = draft_scored and htp_top.ids[0] == expected_next;
        if (draft_scored) {
            draft_offered += 1;
            if (cpu_draft_match) cpu_draft_accepts += 1;
            if (htp_draft_match) htp_draft_accepts += 1;
        }
        const position_passed = std.math.isFinite(cosine) and cosine >= 0.98;
        if (position_passed) passing_positions += 1;
        std.debug.print(
            "{{\"tensorfold\":\"qwen35-mtp-forward-probe-step\",\"backend\":\"{s}\",\"fixture_mode\":{},\"position\":{d},\"input_token\":{d},\"expected_next\":{d},\"draft_scored\":{},\"cpu_draft_match\":{},\"htp_draft_match\":{},\"cpu_top1\":{d},\"htp_top1\":{d},\"top1_match\":{},\"max_abs_error\":{d},\"rmse\":{d},\"cosine\":{d},\"passed\":{}}}\n",
            .{ args[1], fixture_mode, position, token, expected_next, draft_scored, cpu_draft_match, htp_draft_match, cpu_top.ids[0], htp_top.ids[0], head_match, max_abs, rmse, cosine, position_passed },
        );
    }
    var chain_cpu: ChainResult = .{};
    var chain_htp: ChainResult = .{};
    if (chain_depth != 0) {
        const fixture = hidden_fixture orelse return error.InvalidArgument;
        chain_cpu = try simulateChain(false, init.gpa, &runtime, embedding, fixture, tokens, prompt_len, chain_depth, threads);
        chain_htp = try simulateChain(true, init.gpa, &runtime, embedding, fixture, tokens, prompt_len, chain_depth, threads);
    }
    const chain_match = chain_depth == 0 or (chain_cpu.rounds == chain_htp.rounds and chain_cpu.offered == chain_htp.offered and chain_cpu.accepted == chain_htp.accepted and chain_cpu.trace_hash == chain_htp.trace_hash);
    const elapsed = tf_qualcomm_monotonic_ms() - started;
    const passed = passing_positions == positions and chain_match;
    std.debug.print(
        "{{\"tensorfold\":\"qwen35-mtp-forward-probe\",\"backend\":\"{s}\",\"fixture_mode\":{},\"positions\":{d},\"prompt_len\":{d},\"validated_tensors\":15,\"projection_calls\":{d},\"projection_uploads\":{d},\"projection_compute_ms\":{d:.6},\"top1_matches\":{d},\"passing_positions\":{d},\"teacher_forced_offered\":{d},\"cpu_teacher_forced_matches\":{d},\"htp_teacher_forced_matches\":{d},\"chain_depth\":{d},\"chain_cpu_rounds\":{d},\"chain_cpu_draft_steps\":{d},\"chain_cpu_offered\":{d},\"chain_cpu_accepted\":{d},\"chain_htp_rounds\":{d},\"chain_htp_draft_steps\":{d},\"chain_htp_offered\":{d},\"chain_htp_accepted\":{d},\"chain_trace_match\":{},\"elapsed_ms\":{d:.3},\"passed\":{}}}\n",
        .{ args[1], fixture_mode, positions, prompt_len, runtime.projection_calls, runtime.projection_uploads, runtime.projection_compute_ms, matching_heads, passing_positions, draft_offered, cpu_draft_accepts, htp_draft_accepts, chain_depth, chain_cpu.rounds, chain_cpu.draft_steps, chain_cpu.offered, chain_cpu.accepted, chain_htp.rounds, chain_htp.draft_steps, chain_htp.offered, chain_htp.accepted, chain_match, elapsed, passed },
    );
    return if (passed) 0 else 8;
}
