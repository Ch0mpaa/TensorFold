//! A deliberately small, auditable Qwen3.5-2B EXL3 decode path for Qualcomm
//! Android bring-up. It executes autoregressive decode end-to-end and keeps the
//! model math in TensorFold while routing every quantized projection through
//! either GGML CPU or Hexagon HTP. An optional content-addressed cache keeps
//! decoded row-major weights between runs so EXL3 reconstruction is paid once.

const std = @import("std");
const exl3 = @import("exl3_format");
const st = @import("checkpoint");
const expert_pager = @import("expert_pager");
const profile = @import("model_profile");
const mtp = @import("qwen35_mtp_forward_probe.zig");
const moe_routing = @import("moe_routing.zig");

extern fn tf_qualcomm_monotonic_ms() f64;
extern fn tf_qualcomm_peak_rss_kib() usize;
extern fn tf_qualcomm_tied_head_f16(
    embedding: [*]const u16,
    state: [*]const f32,
    vocab_size: usize,
    hidden_size: usize,
    cpu_threads: i32,
    top_ids: *[5]usize,
    top_logits: *[5]f32,
) c_int;
extern fn tf_qualcomm_gdn_step_f32(
    recurrent: [*]f32,
    query: [*]const f32,
    key: [*]const f32,
    value: [*]const f32,
    beta: [*]const f32,
    decay: [*]const f32,
    output: [*]f32,
    key_heads: usize,
    value_heads: usize,
    key_width: usize,
    value_width: usize,
    thread_count: usize,
) void;
extern fn tf_qualcomm_dense_f16(weight: [*]const u16, input: [*]const f32, output: [*]f32, rows: usize, columns: usize) void;
extern fn tf_qualcomm_silu_norm_f16_fast(gate: [*]const f32, input: [*]const f32, norm: [*]const u8, inverse_rms: f32, output: [*]f32, count: usize) void;
extern fn tf_qualcomm_backend_create(backend_name: [*:0]const u8, cpu_threads: i32) ?*anyopaque;
extern fn tf_qualcomm_backend_destroy(context: ?*anyopaque) void;
extern fn tf_qualcomm_backend_reset_graphs(context: ?*anyopaque) void;
extern fn tf_qualcomm_backend_matmul_once(
    context: ?*anyopaque,
    k: usize,
    m: usize,
    n: usize,
    weight: [*]const f32,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
) c_int;
extern fn tf_qualcomm_f32_to_f16(input: [*]const f32, output: [*]u16, count: usize) void;
extern fn tf_qualcomm_backend_matmul_f16_once(
    context: ?*anyopaque,
    k: usize,
    m: usize,
    n: usize,
    weight: [*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
) c_int;
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
extern fn tf_qualcomm_backend_matmul_q8_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    k: usize,
    m: usize,
    n: usize,
    weight: ?[*]const f32,
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
extern fn tf_qualcomm_backend_matmul_q4_f16_source_resident_once(
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
extern fn tf_qualcomm_backend_matmul_q4_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    k: usize,
    m: usize,
    n: usize,
    weight: ?[*]const f32,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weight_uploaded: *c_int,
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
extern fn tf_qualcomm_backend_exl3_paged_once(
    context: ?*anyopaque,
    slot_id: usize,
    k: usize,
    m: usize,
    n: usize,
    words: usize,
    trellis: [*]const u16,
    suh: [*]const u16,
    svh: [*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
) c_int;
extern fn tf_qualcomm_backend_matmul_f16_resident_group_once(
    context: ?*anyopaque,
    slot_id: usize,
    matrix_count: usize,
    k: usize,
    m0: usize,
    m1: usize,
    m2: usize,
    n: usize,
    weight0: ?[*]const u16,
    weight1: ?[*]const u16,
    weight2: ?[*]const u16,
    input: [*]const f32,
    output0: [*]f32,
    output1: [*]f32,
    output2: ?[*]f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_matmul_q8_resident_group_once(
    context: ?*anyopaque,
    slot_id: usize,
    matrix_count: usize,
    k: usize,
    m0: usize,
    m1: usize,
    m2: usize,
    n: usize,
    weight0: ?[*]const f32,
    weight1: ?[*]const f32,
    weight2: ?[*]const f32,
    input: [*]const f32,
    output0: [*]f32,
    output1: [*]f32,
    output2: ?[*]f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_matmul_q4_resident_group_once(
    context: ?*anyopaque,
    slot_id: usize,
    matrix_count: usize,
    k: usize,
    m0: usize,
    m1: usize,
    m2: usize,
    n: usize,
    weight0: ?[*]const f32,
    weight1: ?[*]const f32,
    weight2: ?[*]const f32,
    input: [*]const f32,
    output0: [*]f32,
    output1: [*]f32,
    output2: ?[*]f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_mlp_f16_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    input_width: usize,
    intermediate_width: usize,
    output_width: usize,
    n: usize,
    gate_weight: ?[*]const u16,
    up_weight: ?[*]const u16,
    down_weight: ?[*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_mlp_q8_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    input_width: usize,
    intermediate_width: usize,
    output_width: usize,
    n: usize,
    gate_weight: ?[*]const f32,
    up_weight: ?[*]const f32,
    down_weight: ?[*]const f32,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_mlp_q4_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    input_width: usize,
    intermediate_width: usize,
    output_width: usize,
    n: usize,
    gate_weight: ?[*]const f32,
    up_weight: ?[*]const f32,
    down_weight: ?[*]const f32,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_layer_tail_q4_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    attention_width: usize,
    hidden_width: usize,
    intermediate_width: usize,
    n: usize,
    output_q8: c_int,
    output_weight: ?[*]const f32,
    gate_weight: ?[*]const f32,
    up_weight: ?[*]const f32,
    down_weight: ?[*]const f32,
    post_norm_weight: ?[*]const f32,
    state: [*]f32,
    attention: [*]const f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_linear_tail_q4_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    width: usize,
    heads: usize,
    hidden_width: usize,
    intermediate_width: usize,
    n: usize,
    output_weight: ?[*]const f32,
    gate_weight: ?[*]const f32,
    up_weight: ?[*]const f32,
    down_weight: ?[*]const f32,
    linear_norm_weight: ?[*]const f32,
    post_norm_weight: ?[*]const f32,
    state: [*]f32,
    z: [*]const f32,
    query: [*]const f32,
    key: [*]const f32,
    value_input: [*]const f32,
    log_decay: [*]const f32,
    beta: [*]const f32,
    initial_gdn_state: ?[*]const f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_gdn_f32_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    width: usize,
    heads: usize,
    n: usize,
    query: [*]const f32,
    key: [*]const f32,
    value_input: [*]const f32,
    log_decay: [*]const f32,
    beta: [*]const f32,
    initial_state: ?[*]const f32,
    output: [*]f32,
    compute_ms: *f64,
) c_int;
extern fn tf_qualcomm_backend_gdn_state_get(
    context: ?*anyopaque,
    slot_id: usize,
    state: [*]f32,
    elements: usize,
) c_int;
extern fn tf_qualcomm_backend_gdn_state_set(
    context: ?*anyopaque,
    slot_id: usize,
    state: [*]const f32,
    elements: usize,
) c_int;
extern fn tf_qualcomm_backend_gdn_state_restore_snapshot(
    context: ?*anyopaque,
    slot_id: usize,
    n: usize,
    tokens_back: usize,
    scratch: [*]f32,
    elements: usize,
) c_int;
extern fn tf_qualcomm_backend_linear_tail_gdn_state_restore_snapshot(
    context: ?*anyopaque,
    slot_id: usize,
    n: usize,
    tokens_back: usize,
    scratch: [*]f32,
    elements: usize,
) c_int;

const prefix = "model.language_model";
const hidden = profile.hidden;
const intermediate = profile.intermediate;
const layers = profile.layers;
const vocab = profile.vocab;
const experts = profile.experts;
const top_k = profile.top_k;
const shared_intermediate = profile.shared_intermediate;
const max_worker_prompt_tokens = 512;
const eps: f32 = 1e-6;
const linear_key_heads = profile.linear_key_heads;
const linear_value_heads = profile.linear_value_heads;
const linear_key_width = profile.linear_key_width;
const linear_value_width = profile.linear_value_width;
const linear_query_width = linear_key_heads * linear_key_width;
const linear_output_width = linear_value_heads * linear_value_width;
const linear_state_elems = linear_value_heads * linear_value_width * linear_key_width;
const conv_channels = 2 * linear_query_width + linear_output_width;
const conv_history_rows = 3;
const attention_heads = profile.attention_heads;
const kv_heads = profile.kv_heads;
const attention_width = profile.attention_width;
const attention_query_width = attention_heads * attention_width;
const kv_width = kv_heads * attention_width;
const rope_width = profile.rope_width;
const cache_magic = "TFEXL3F1";
const f16_cache_magic = "TFEXL3H1";
const cache_header_bytes = 64;
const projections_per_position = layers * 6 + layers / 4;

fn f16Value(bytes: []const u8, index: usize) f32 {
    const bits = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
    const decoded: f16 = @bitCast(bits);
    return @floatCast(decoded);
}

fn bf16Value(bytes: []const u8, index: usize) f32 {
    const bits = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
    return @bitCast(@as(u32, bits) << 16);
}

fn f32Value(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
}

fn value(tensor: st.Tensor, index: usize) f32 {
    return switch (tensor.dtype) {
        .f16 => f16Value(tensor.bytes, index),
        .bf16 => bf16Value(tensor.bytes, index),
        .f32 => f32Value(tensor.bytes, index),
        else => unreachable,
    };
}

fn roundF16(x: f32) f32 {
    if (comptime experts > 0) return roundBF16(x);
    const h: f16 = @floatCast(x);
    return @floatCast(h);
}
fn quantizeF16(x: f32) f32 {
    const h: f16 = @floatCast(x);
    return @floatCast(h);
}

fn roundBF16(x: f32) f32 {
    var bits: u32 = @bitCast(x);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    bits &= 0xffff0000;
    return @bitCast(bits);
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

// Exact-value LUT over all bf16 bit patterns: sigmoid(roundBF16(x)) keyed by the
// u16 high half. Built lazily with the same sigmoid; identical results, no exp.
var sigmoid_lut: [65536]f32 = undefined;
var sigmoid_lut_ready = false;

fn sigmoidBf16Lut(x: f32) f32 {
    if (!sigmoid_lut_ready) {
        var pattern: usize = 0;
        while (pattern < 65536) : (pattern += 1) {
            const bits: u32 = @intCast(pattern << 16);
            const x2: f32 = @bitCast(bits);
            sigmoid_lut[pattern] = sigmoid(x2);
        }
        sigmoid_lut_ready = true;
    }
    const bits: u32 = @bitCast(x);
    return sigmoid_lut[(bits >> 16) & 0xFFFF];
}

fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

// Exact-value LUT over all f16 bit patterns for the gate sigmoid.
var sigmoid_f16_lut: [65536]f32 = undefined;
var sigmoid_f16_lut_ready = false;

fn sigmoidF16Lut(x: f32) f32 {
    if (!sigmoid_f16_lut_ready) {
        var pattern: usize = 0;
        while (pattern < 65536) : (pattern += 1) {
            const h: f16 = @bitCast(@as(u16, @intCast(pattern)));
            sigmoid_f16_lut[pattern] = sigmoid(@as(f32, h));
        }
        sigmoid_f16_lut_ready = true;
    }
    const h: f16 = @floatCast(x);
    return sigmoid_f16_lut[@as(u16, @bitCast(h))];
}

// RoPE tables: inv_freq constant; cos/sin depend only on (position, i) so cache
// per position across layers/rows. Backed by a c_allocator global arena grown once.
var rope_arena: std.heap.ArenaAllocator = undefined;
var rope_arena_ready = false;
var rope_inv_freq: [rope_width / 2]f32 = undefined;
var rope_inv_freq_ready = false;
var rope_cos: []f32 = &.{};
var rope_sin: []f32 = &.{};
var rope_positions: usize = 0;

fn ropeTables(position: usize) void {
    if (!rope_arena_ready) {
        rope_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        rope_arena_ready = true;
    }
    if (!rope_inv_freq_ready) {
        for (0..rope_width / 2) |i| {
            rope_inv_freq[i] = std.math.pow(f32, 10_000_000.0, -@as(f32, @floatFromInt(i)) / @as(f32, rope_width / 2));
        }
        rope_inv_freq_ready = true;
    }
    if (position >= rope_positions) {
        const new_len = position + 64;
        const half: usize = rope_width / 2;
        const cos_buf = rope_arena.allocator().realloc(rope_cos, new_len * half) catch return;
        const sin_buf = rope_arena.allocator().realloc(rope_sin, new_len * half) catch return;
        for (rope_positions..new_len) |p| {
            for (0..half) |i| {
                const angle = @as(f32, @floatFromInt(p)) * rope_inv_freq[i];
                cos_buf[p * half + i] = @cos(angle);
                sin_buf[p * half + i] = @sin(angle);
            }
        }
        rope_cos = cos_buf;
        rope_sin = sin_buf;
        rope_positions = new_len;
    }
}

fn require(file: *const st.Checkpoint, name: []const u8, dtype: st.DType, shape: []const usize) !st.Tensor {
    const tensor = file.lookup(name) orelse return error.MissingTensor;
    if (!tensor.is(dtype, shape)) return error.UnexpectedTensor;
    return tensor;
}

fn tensorName(buffer: []u8, comptime format: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buffer, format, args);
}

const PackedCpuTask = struct {
    trellis: []const i16,
    rotated_input: []const f32,
    output: []f32,
    k: usize,
    m: usize,
    words: usize,
    rows: usize,
    tile_m_start: usize,
    tile_m_end: usize,
};

fn packedCpuProjectRange(task: *const PackedCpuTask) void {
    var placement: [256]exl3.Placement = undefined;
    exl3.tilePlacement(&placement);
    const width: exl3.Width = .{ .halves = @intCast(task.words / 8) };
    const tiles_k = task.k / 16;
    const tiles_m = task.m / 16;
    for (task.tile_m_start..task.tile_m_end) |tile_m| for (0..tiles_k) |tile_k| {
        const tile = task.trellis[(tile_k * tiles_m + tile_m) * task.words ..][0..task.words];
        var states: [256]u32 = undefined;
        exl3.tileStates(width, tile, &states);
        for (states, placement) |state, place| {
            const inner = tile_k * 16 + place.row;
            const column = tile_m * 16 + place.col;
            const weight: f32 = @floatCast(exl3.codebookValue(.mcg, @intCast(state)));
            for (0..task.rows) |row| {
                task.output[row * task.m + column] += task.rotated_input[row * task.k + inner] * weight;
            }
        }
    };
}

fn packedCpuProject(
    allocator: std.mem.Allocator,
    trellis_bytes: []const u8,
    suh_bytes: []const u8,
    svh_bytes: []const u8,
    k: usize,
    m: usize,
    words: usize,
    input: []const f32,
    output: []f32,
    cpu_threads: usize,
) !void {
    if (k % exl3.hadamard_dim != 0 or m % exl3.hadamard_dim != 0 or
        trellis_bytes.len != (k / 16) * (m / 16) * words * @sizeOf(i16) or
        suh_bytes.len != k * @sizeOf(f16) or svh_bytes.len != m * @sizeOf(f16) or
        input.len == 0 or input.len % k != 0 or output.len != (input.len / k) * m) return error.UnexpectedTensor;
    const rows = input.len / k;
    const rotated_input = try allocator.alloc(f32, input.len);
    defer allocator.free(rotated_input);
    for (rotated_input, input, 0..) |*out, x, i| out.* = x * f16Value(suh_bytes, i % k);
    exl3.rotateFastF32(rotated_input, rows, k, 1);
    @memset(output, 0);
    const trellis: []const i16 = @as([*]const i16, @ptrCast(@alignCast(trellis_bytes.ptr)))[0 .. trellis_bytes.len / @sizeOf(i16)];
    const tiles_m = m / 16;
    const thread_count = @min(@min(cpu_threads, tiles_m), 16);
    if (thread_count == 0) return error.InvalidArgument;
    var tasks: [16]PackedCpuTask = undefined;
    var worker_threads: [15]?std.Thread = @splat(null);
    var spawned: usize = 0;
    errdefer for (worker_threads[0..spawned]) |thread| thread.?.join();
    for (0..thread_count) |thread_index| {
        tasks[thread_index] = .{
            .trellis = trellis,
            .rotated_input = rotated_input,
            .output = output,
            .k = k,
            .m = m,
            .words = words,
            .rows = rows,
            .tile_m_start = tiles_m * thread_index / thread_count,
            .tile_m_end = tiles_m * (thread_index + 1) / thread_count,
        };
        if (thread_index != 0) {
            worker_threads[thread_index - 1] = try std.Thread.spawn(.{}, packedCpuProjectRange, .{&tasks[thread_index]});
            spawned += 1;
        }
    }
    packedCpuProjectRange(&tasks[0]);
    for (worker_threads[0..spawned]) |thread| thread.?.join();
    exl3.rotateFastF32(output, rows, m, 1);
    for (output, 0..) |*out, i| out.* *= f16Value(svh_bytes, i % m);
}

const Runner = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    file: *const st.Checkpoint,
    backend_is_cpu: bool,
    backend: ?*anyopaque,
    cpu_threads: usize,
    use_f16_weights: bool = false,
    use_resident_weights: bool = false,
    use_packed_exl3: bool = false,
    use_f16_head_cache: bool = false,
    use_q8_weights: bool = false,
    use_q4_weights: bool = false,
    q8_attention_inputs: bool = false,
    q8_linear_outputs: bool = false,
    q8_full_outputs: bool = false,
    q8_output_layer_start: usize = 0,
    q8_output_layer_end: usize = layers,
    q8_output_layer_mask: u32 = 0,
    q8_mlp: bool = false,
    use_resident_head: bool = false,
    use_q8_head: bool = false,
    use_q4_head: bool = false,
    use_adaptive_head: bool = false,
    use_refined_head: bool = false,
    use_fused_tail: bool = false,
    use_fused_linear_tail: bool = false,
    fused_linear_layer_mask: u32 = 0,
    fast_activations: bool = false,
    cache_dir: ?[]const u8 = null,
    cache_model_id: ?[]const u8 = null,
    expert_cache: ?*expert_pager.Pager = null,
    projection_count: usize = 0,
    projection_ms: f64 = 0,
    projection_wall_ms: f64 = 0,
    weight_convert_ms: f64 = 0,
    dequant_ms: f64 = 0,
    cache_hash_ms: f64 = 0,
    cache_read_ms: f64 = 0,
    cache_write_ms: f64 = 0,
    cache_hits: usize = 0,
    cache_misses: usize = 0,
    cache_bytes_read: usize = 0,
    cache_bytes_written: usize = 0,
    cache_trusted_hits: usize = 0,
    cache_aliases_created: usize = 0,
    f16_cache_hits: usize = 0,
    f16_cache_misses: usize = 0,
    f16_cache_bytes_read: usize = 0,
    f16_cache_bytes_written: usize = 0,
    resident_loaded: [projections_per_position]bool = @splat(false),
    resident_slot_cursor: usize = 0,
    resident_weight_uploads: usize = 0,
    resident_weight_hits: usize = 0,
    resident_weight_fallbacks: usize = 0,
    resident_weight_bytes_uploaded: usize = 0,
    resident_head_loaded: bool = false,
    resident_head_q4_loaded: bool = false,
    resident_head_q8_loaded: bool = false,
    resident_head_uploads: usize = 0,
    resident_head_hits: usize = 0,
    resident_head_fallbacks: usize = 0,
    resident_head_compute_ms: f64 = 0,
    resident_gdn_loaded: [layers]bool = @splat(false),
    fused_tail_slot: [layers]usize = @splat(0),
    resident_gdn_hits: usize = 0,
    resident_gdn_compute_ms: f64 = 0,
    resident_gdn_wall_ms: f64 = 0,
    pair_quant_calls: usize = 0,
    pair_quant_compute_ms: f64 = 0,
    pair_quant_wall_ms: f64 = 0,
    triple_quant_calls: usize = 0,
    triple_quant_compute_ms: f64 = 0,
    triple_quant_wall_ms: f64 = 0,
    layer_tail_calls: usize = 0,
    layer_tail_compute_ms: f64 = 0,
    layer_tail_wall_ms: f64 = 0,

    fn outputUsesQ8(self: *const Runner, base: []const u8) bool {
        const kind_matches = (self.q8_linear_outputs and std.mem.indexOf(u8, base, ".linear_attn.") != null) or
            (self.q8_full_outputs and std.mem.indexOf(u8, base, ".self_attn.") != null);
        if (!kind_matches) return false;
        const marker = ".layers.";
        const at = std.mem.indexOf(u8, base, marker) orelse return false;
        const tail = base[at + marker.len ..];
        const dot = std.mem.indexOfScalar(u8, tail, '.') orelse return false;
        const layer = std.fmt.parseInt(usize, tail[0..dot], 10) catch return false;
        if (self.q8_output_layer_mask != 0) {
            return (self.q8_output_layer_mask & (@as(u32, 1) << @intCast(layer))) != 0;
        }
        return layer >= self.q8_output_layer_start and layer < self.q8_output_layer_end;
    }

    fn usesFusedLinearTail(self: *const Runner, layer: usize) bool {
        return self.use_fused_linear_tail and
            (self.fused_linear_layer_mask & (@as(u32, 1) << @intCast(layer))) != 0;
    }

    fn resetMetrics(self: *Runner) void {
        self.projection_count = 0;
        self.projection_ms = 0;
        self.projection_wall_ms = 0;
        self.weight_convert_ms = 0;
        self.dequant_ms = 0;
        self.cache_hash_ms = 0;
        self.cache_read_ms = 0;
        self.cache_write_ms = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.cache_bytes_read = 0;
        self.cache_bytes_written = 0;
        self.cache_trusted_hits = 0;
        self.cache_aliases_created = 0;
        self.f16_cache_hits = 0;
        self.f16_cache_misses = 0;
        self.f16_cache_bytes_read = 0;
        self.f16_cache_bytes_written = 0;
        self.resident_weight_uploads = 0;
        self.resident_weight_hits = 0;
        self.resident_weight_fallbacks = 0;
        self.resident_weight_bytes_uploaded = 0;
        self.resident_slot_cursor = 0;
        self.resident_head_uploads = 0;
        self.resident_head_hits = 0;
        self.resident_head_fallbacks = 0;
        self.resident_head_compute_ms = 0;
        self.resident_gdn_loaded = @splat(false);
        self.resident_gdn_hits = 0;
        self.resident_gdn_compute_ms = 0;
        self.resident_gdn_wall_ms = 0;
        self.pair_quant_calls = 0;
        self.pair_quant_compute_ms = 0;
        self.pair_quant_wall_ms = 0;
        self.triple_quant_calls = 0;
        self.triple_quant_compute_ms = 0;
        self.triple_quant_wall_ms = 0;
        self.layer_tail_calls = 0;
        self.layer_tail_compute_ms = 0;
        self.layer_tail_wall_ms = 0;
    }

    fn invalidateResident(self: *Runner) void {
        self.resident_loaded = @splat(false);
        self.resident_slot_cursor = 0;
        self.resident_head_loaded = false;
        self.resident_head_q4_loaded = false;
        self.resident_head_q8_loaded = false;
        self.resident_gdn_loaded = @splat(false);
        self.fused_tail_slot = @splat(0);
    }

    const CachedWeight = struct {
        file: std.Io.File,
        map: std.Io.File.MemoryMap,
        weight: []const f32,

        fn close(self: *CachedWeight, io: std.Io) void {
            self.map.destroy(io);
            self.file.close(io);
            self.* = undefined;
        }
    };

    const CachedF16Weight = struct {
        file: std.Io.File,
        map: std.Io.File.MemoryMap,
        weight: []const u16,

        fn close(self: *CachedF16Weight, io: std.Io) void {
            self.map.destroy(io);
            self.file.close(io);
            self.* = undefined;
        }
    };

    const ResidentProjection = struct {
        k: usize,
        m: usize,
        key: [32]u8,
    };

    fn residentProjection(self: *Runner, base: []const u8) !ResidentProjection {
        var name_buffer: [256]u8 = undefined;
        const trellis_tensor = self.file.lookup(try tensorName(&name_buffer, "{s}.trellis", .{base})) orelse return error.MissingTensor;
        if (trellis_tensor.dtype != .i16 or trellis_tensor.rank != 3) return error.UnexpectedTensor;
        const tiles_k = trellis_tensor.dim(0);
        const tiles_m = trellis_tensor.dim(1);
        const words = trellis_tensor.dim(2);
        if (words % 8 != 0) return error.UnexpectedTensor;
        const k = tiles_k * 16;
        const m = tiles_m * 16;
        const model_id = self.cache_model_id orelse return error.InvalidModelSha256;
        return .{ .k = k, .m = m, .key = f16ProjectionKey(model_id, base, k, m, words) };
    }

    fn residentProjectionF32(self: *Runner, base: []const u8) !ResidentProjection {
        var name_buffer: [256]u8 = undefined;
        const trellis_tensor = self.file.lookup(try tensorName(&name_buffer, "{s}.trellis", .{base})) orelse return error.MissingTensor;
        if (trellis_tensor.dtype != .i16 or trellis_tensor.rank != 3) return error.UnexpectedTensor;
        const k = trellis_tensor.dim(0) * 16;
        const m = trellis_tensor.dim(1) * 16;
        const words = trellis_tensor.dim(2);
        if (words % 8 != 0) return error.UnexpectedTensor;
        const model_id = self.cache_model_id orelse return error.InvalidModelSha256;
        return .{ .k = k, .m = m, .key = trustedProjectionKey(model_id, base, k, m, words) };
    }

    fn projectionKey(
        trellis_tensor: st.Tensor,
        suh_tensor: st.Tensor,
        svh_tensor: st.Tensor,
        mcg_tensor: st.Tensor,
        k: usize,
        m: usize,
        words: usize,
    ) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("TensorFold EXL3 decoded row-major f32 cache v1\n");
        var dimensions: [24]u8 = undefined;
        std.mem.writeInt(u64, dimensions[0..8], k, .little);
        std.mem.writeInt(u64, dimensions[8..16], m, .little);
        std.mem.writeInt(u64, dimensions[16..24], words, .little);
        hash.update(&dimensions);
        hash.update(trellis_tensor.bytes);
        hash.update(suh_tensor.bytes);
        hash.update(svh_tensor.bytes);
        hash.update(mcg_tensor.bytes);
        var key: [32]u8 = undefined;
        hash.final(&key);
        return key;
    }

    fn trustedProjectionKey(model_id: []const u8, base: []const u8, k: usize, m: usize, words: usize) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("TensorFold EXL3 decoded row-major f32 trusted cache v1\n");
        hash.update(model_id);
        hash.update("\n");
        hash.update(base);
        var dimensions: [24]u8 = undefined;
        std.mem.writeInt(u64, dimensions[0..8], k, .little);
        std.mem.writeInt(u64, dimensions[8..16], m, .little);
        std.mem.writeInt(u64, dimensions[16..24], words, .little);
        hash.update(&dimensions);
        var key: [32]u8 = undefined;
        hash.final(&key);
        return key;
    }

    fn f16ProjectionKey(model_id: []const u8, base: []const u8, k: usize, m: usize, words: usize) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("TensorFold EXL3 decoded row-major f16 cache v1\n");
        hash.update(model_id);
        hash.update("\n");
        hash.update(base);
        var dimensions: [24]u8 = undefined;
        std.mem.writeInt(u64, dimensions[0..8], k, .little);
        std.mem.writeInt(u64, dimensions[8..16], m, .little);
        std.mem.writeInt(u64, dimensions[16..24], words, .little);
        hash.update(&dimensions);
        var key: [32]u8 = undefined;
        hash.final(&key);
        return key;
    }

    fn cachePath(self: *Runner, key: [32]u8) !?[]u8 {
        const dir = self.cache_dir orelse return null;
        return try std.fmt.allocPrint(self.allocator, "{s}/{s}.tfexl3-f32", .{ dir, std.fmt.bytesToHex(key, .lower) });
    }

    fn cacheAliasPath(self: *Runner, trusted_path: []const u8) ![]u8 {
        return try std.fmt.allocPrint(self.allocator, "{s}.alias", .{trusted_path});
    }

    fn f16CachePath(self: *Runner, key: [32]u8) !?[]u8 {
        const dir = self.cache_dir orelse return null;
        return try std.fmt.allocPrint(self.allocator, "{s}/{s}.tfexl3-f16", .{ dir, std.fmt.bytesToHex(key, .lower) });
    }

    fn cacheHeader(key: [32]u8, k: usize, m: usize, payload_bytes: usize) [cache_header_bytes]u8 {
        var header: [cache_header_bytes]u8 = @splat(0);
        @memcpy(header[0..cache_magic.len], cache_magic);
        std.mem.writeInt(u64, header[8..16], k, .little);
        std.mem.writeInt(u64, header[16..24], m, .little);
        std.mem.writeInt(u64, header[24..32], payload_bytes, .little);
        @memcpy(header[32..64], &key);
        return header;
    }

    fn saveCacheAlias(self: *Runner, content_key: [32]u8, trusted_path: []const u8) !void {
        const alias_path = try self.cacheAliasPath(trusted_path);
        defer self.allocator.free(alias_path);
        if (std.Io.Dir.cwd().statFile(self.io, alias_path, .{})) |_| return else |_| {}
        const temp = try std.fmt.allocPrint(self.allocator, "{s}.{d}.tmp", .{ alias_path, std.c.getpid() });
        defer self.allocator.free(temp);
        var file = try std.Io.Dir.cwd().createFile(self.io, temp, .{ .truncate = true });
        errdefer std.Io.Dir.cwd().deleteFile(self.io, temp) catch {};
        defer file.close(self.io);
        try file.writePositionalAll(self.io, &content_key, 0);
        try file.setLength(self.io, content_key.len);
        file.sync(self.io) catch {};
        try std.Io.Dir.cwd().rename(temp, std.Io.Dir.cwd(), alias_path, self.io);
    }

    fn f16CacheHeader(key: [32]u8, k: usize, m: usize, payload_bytes: usize) [cache_header_bytes]u8 {
        var header = cacheHeader(key, k, m, payload_bytes);
        @memcpy(header[0..f16_cache_magic.len], f16_cache_magic);
        return header;
    }

    fn loadCachedF16(self: *Runner, key: [32]u8, k: usize, m: usize) !?CachedF16Weight {
        const path = try self.f16CachePath(key) orelse return null;
        defer self.allocator.free(path);
        var file = std.Io.Dir.cwd().openFile(self.io, path, .{}) catch return null;
        var file_owned = true;
        defer if (file_owned) file.close(self.io);
        const payload_bytes = try std.math.mul(usize, try std.math.mul(usize, k, m), @sizeOf(u16));
        const whole_bytes = try std.math.add(usize, cache_header_bytes, payload_bytes);
        if ((file.length(self.io) catch return null) != whole_bytes) return null;
        const started = tf_qualcomm_monotonic_ms();
        var map = std.Io.File.MemoryMap.create(self.io, file, .{
            .len = whole_bytes,
            .protection = .{ .read = true, .write = false },
            .populate = false,
        }) catch return null;
        var map_owned = true;
        defer if (map_owned) map.destroy(self.io);
        const want = f16CacheHeader(key, k, m, payload_bytes);
        if (!std.mem.eql(u8, map.memory[0..cache_header_bytes], &want)) return null;
        const weight = std.mem.bytesAsSlice(u16, map.memory[cache_header_bytes..]);
        self.cache_read_ms += tf_qualcomm_monotonic_ms() - started;
        self.f16_cache_hits += 1;
        self.f16_cache_bytes_read += payload_bytes;
        file_owned = false;
        map_owned = false;
        return .{ .file = file, .map = map, .weight = weight };
    }

    fn saveCachedF16(self: *Runner, key: [32]u8, k: usize, m: usize, weight: []const u16) !void {
        const path = try self.f16CachePath(key) orelse return;
        defer self.allocator.free(path);
        const temp = try std.fmt.allocPrint(self.allocator, "{s}.{d}.tmp", .{ path, std.c.getpid() });
        defer self.allocator.free(temp);
        const payload = std.mem.sliceAsBytes(weight);
        const header = f16CacheHeader(key, k, m, payload.len);
        const started = tf_qualcomm_monotonic_ms();
        var file = try std.Io.Dir.cwd().createFile(self.io, temp, .{ .truncate = true });
        errdefer std.Io.Dir.cwd().deleteFile(self.io, temp) catch {};
        defer file.close(self.io);
        try file.writePositionalAll(self.io, &header, 0);
        try file.writePositionalAll(self.io, payload, cache_header_bytes);
        try file.setLength(self.io, cache_header_bytes + payload.len);
        file.sync(self.io) catch {};
        try std.Io.Dir.cwd().rename(temp, std.Io.Dir.cwd(), path, self.io);
        self.cache_write_ms += tf_qualcomm_monotonic_ms() - started;
        self.f16_cache_bytes_written += payload.len;
    }

    fn loadCached(self: *Runner, key: [32]u8, k: usize, m: usize) !?CachedWeight {
        const path = try self.cachePath(key) orelse return null;
        defer self.allocator.free(path);
        var alias_target: ?[]u8 = null;
        defer if (alias_target) |target| self.allocator.free(target);
        var resolved_key: ?[32]u8 = null;
        var file = std.Io.Dir.cwd().openFile(self.io, path, .{}) catch blk: {
            const alias_path = try self.cacheAliasPath(path);
            defer self.allocator.free(alias_path);
            const raw = std.Io.Dir.cwd().readFileAlloc(self.io, alias_path, self.allocator, .limited(33)) catch return null;
            defer self.allocator.free(raw);
            if (raw.len != 32) return null;
            var content_key: [32]u8 = undefined;
            @memcpy(content_key[0..], raw);
            const target = try self.cachePath(content_key) orelse return null;
            alias_target = target;
            resolved_key = content_key;
            break :blk std.Io.Dir.cwd().openFile(self.io, target, .{}) catch return null;
        };
        var file_owned = true;
        defer if (file_owned) file.close(self.io);
        const payload_bytes = try std.math.mul(usize, try std.math.mul(usize, k, m), @sizeOf(f32));
        const whole_bytes = try std.math.add(usize, cache_header_bytes, payload_bytes);
        if ((file.length(self.io) catch return null) != whole_bytes) return null;
        const started = tf_qualcomm_monotonic_ms();
        var map = std.Io.File.MemoryMap.create(self.io, file, .{
            .len = whole_bytes,
            .protection = .{ .read = true, .write = false },
            .populate = false,
        }) catch return null;
        var map_owned = true;
        defer if (map_owned) map.destroy(self.io);
        const want = cacheHeader(key, k, m, payload_bytes);
        // The first 32 bytes validate format and geometry. A hard-link alias
        // gets filename identity from the trusted model hash; a sidecar alias
        // also names and validates the content key stored in the header.
        if (!std.mem.eql(u8, map.memory[0..32], want[0..32])) return null;
        if (resolved_key) |content_key| if (!std.mem.eql(u8, map.memory[32..64], &content_key)) return null;
        const payload = map.memory[cache_header_bytes..];
        const weight = std.mem.bytesAsSlice(f32, payload);
        self.cache_read_ms += tf_qualcomm_monotonic_ms() - started;
        self.cache_hits += 1;
        self.cache_bytes_read += payload_bytes;
        file_owned = false;
        map_owned = false;
        return .{ .file = file, .map = map, .weight = weight };
    }

    fn saveCached(self: *Runner, key: [32]u8, k: usize, m: usize, weight: []const f32) !void {
        const path = try self.cachePath(key) orelse return;
        defer self.allocator.free(path);
        const temp = try std.fmt.allocPrint(self.allocator, "{s}.{d}.tmp", .{ path, std.c.getpid() });
        defer self.allocator.free(temp);
        const payload = std.mem.sliceAsBytes(weight);
        const header = cacheHeader(key, k, m, payload.len);
        const started = tf_qualcomm_monotonic_ms();
        var file = try std.Io.Dir.cwd().createFile(self.io, temp, .{ .truncate = true });
        errdefer std.Io.Dir.cwd().deleteFile(self.io, temp) catch {};
        defer file.close(self.io);
        try file.writePositionalAll(self.io, &header, 0);
        try file.writePositionalAll(self.io, payload, cache_header_bytes);
        try file.setLength(self.io, cache_header_bytes + payload.len);
        file.sync(self.io) catch {};
        try std.Io.Dir.cwd().rename(temp, std.Io.Dir.cwd(), path, self.io);
        self.cache_write_ms += tf_qualcomm_monotonic_ms() - started;
        self.cache_bytes_written += payload.len;
    }

    fn linkCached(self: *Runner, content_key: [32]u8, trusted_key: [32]u8) !void {
        const content_path = try self.cachePath(content_key) orelse return;
        defer self.allocator.free(content_path);
        const trusted_path = try self.cachePath(trusted_key) orelse return;
        defer self.allocator.free(trusted_path);
        std.Io.Dir.cwd().hardLink(content_path, std.Io.Dir.cwd(), trusted_path, self.io, .{}) catch |err| switch (err) {
            error.PathAlreadyExists => return,
            error.AccessDenied => return self.saveCacheAlias(content_key, trusted_path),
            else => return err,
        };
        self.cache_aliases_created += 1;
    }

    fn runWeight(self: *Runner, k: usize, m: usize, weight: []const f32, input: []const f32, output: []f32) !void {
        if (input.len == 0 or input.len % k != 0) return error.UnexpectedTensor;
        const n = input.len / k;
        if (output.len != m * n) return error.UnexpectedTensor;
        var ms: f64 = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_matmul_once(self.backend, k, m, n, weight.ptr, input.ptr, output.ptr, &ms);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 1;
    }

    fn runWeightQuant(self: *Runner, k: usize, m: usize, weight: ?[]const f32, input: []const f32, output: []f32, q4: bool) !void {
        if (input.len == 0 or input.len % k != 0) return error.UnexpectedTensor;
        const n = input.len / k;
        if (output.len != m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const weight_ptr: ?[*]const f32 = if (weight) |values| values.ptr else null;
        const started = tf_qualcomm_monotonic_ms();
        const status = if (q4)
            tf_qualcomm_backend_matmul_q4_resident_once(self.backend, slot, k, m, n, weight_ptr, input.ptr, output.ptr, &ms, &uploaded)
        else
            tf_qualcomm_backend_matmul_q8_resident_once(self.backend, slot, k, m, n, weight_ptr, input.ptr, output.ptr, &ms, &uploaded);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 1;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 1;
            self.resident_weight_bytes_uploaded += (k / 32) * (if (q4) @as(usize, 18) else 34) * m;
        } else {
            self.resident_weight_hits += 1;
        }
    }

    fn runDecodedWeight(self: *Runner, k: usize, m: usize, weight: []const f32, input: []const f32, output: []f32, q4: bool) !void {
        if (self.use_q8_weights or self.use_q4_weights) return self.runWeightQuant(k, m, weight, input, output, q4);
        return self.runWeight(k, m, weight, input, output);
    }

    fn runResidentWeightF16(self: *Runner, slot_id: usize, k: usize, m: usize, weight: ?[]const u16, input: []const f32, output: []f32) bool {
        const n = input.len / k;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const weight_ptr: ?[*]const u16 = if (weight) |words| words.ptr else null;
        const status = tf_qualcomm_backend_matmul_f16_resident_once(self.backend, slot_id, k, m, n, weight_ptr, input.ptr, output.ptr, &ms, &uploaded);
        if (status != 0) {
            self.resident_weight_fallbacks += 1;
            self.resident_loaded[slot_id] = false;
            return false;
        }
        self.projection_ms += ms;
        self.projection_count += 1;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot_id] = true;
            self.resident_weight_uploads += 1;
            self.resident_weight_bytes_uploaded += k * m * @sizeOf(u16);
        } else {
            self.resident_weight_hits += 1;
        }
        return true;
    }

    fn runWeightF16(self: *Runner, k: usize, m: usize, weight: []const u16, input: []const f32, output: []f32) !void {
        if (input.len == 0 or input.len % k != 0) return error.UnexpectedTensor;
        const n = input.len / k;
        if (weight.len != k * m or output.len != m * n) return error.UnexpectedTensor;
        const started = tf_qualcomm_monotonic_ms();
        const slot_id = self.resident_slot_cursor;
        if (self.use_resident_weights and self.runResidentWeightF16(slot_id, k, m, weight, input, output)) {
            self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
            return;
        }
        var ms: f64 = 0;
        const status = tf_qualcomm_backend_matmul_f16_once(self.backend, k, m, n, weight.ptr, input.ptr, output.ptr, &ms);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 1;
        if (self.use_resident_weights) self.resident_slot_cursor += 1;
    }

    fn projectPairResident(self: *Runner, base0: []const u8, base1: []const u8, input: []const f32, output0: []f32, output1: []f32) !void {
        const p0 = try self.residentProjection(base0);
        const p1 = try self.residentProjection(base1);
        if (p0.k != p1.k or input.len == 0 or input.len % p0.k != 0) return error.UnexpectedTensor;
        const n = input.len / p0.k;
        if (output0.len != p0.m * n or output1.len != p1.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var loaded0: ?CachedF16Weight = null;
        defer if (loaded0) |*loaded| loaded.close(self.io);
        var loaded1: ?CachedF16Weight = null;
        defer if (loaded1) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            loaded0 = try self.loadCachedF16(p0.key, p0.k, p0.m) orelse return error.MissingResidentWeight;
            loaded1 = try self.loadCachedF16(p1.key, p1.k, p1.m) orelse return error.MissingResidentWeight;
        }
        const weight0: ?[*]const u16 = if (loaded0) |loaded| loaded.weight.ptr else null;
        const weight1: ?[*]const u16 = if (loaded1) |loaded| loaded.weight.ptr else null;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_matmul_f16_resident_group_once(self.backend, slot, 2, p0.k, p0.m, p1.m, 0, n, weight0, weight1, null, input.ptr, output0.ptr, output1.ptr, null, &ms, &uploaded);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("resident pair bridge failed: status {d}, slot {d}, k {d}, m {d}/{d}, n {d}\n", .{ status, slot, p0.k, p0.m, p1.m, n });
            return error.BridgeFailed;
        }
        self.projection_ms += ms;
        self.projection_count += 2;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 2;
            self.resident_weight_bytes_uploaded += (p0.k * p0.m + p1.k * p1.m) * @sizeOf(u16);
        } else {
            self.resident_weight_hits += 2;
        }
    }

    fn projectTripleResident(self: *Runner, base0: []const u8, base1: []const u8, base2: []const u8, input: []const f32, output0: []f32, output1: []f32, output2: []f32) !void {
        const p0 = try self.residentProjection(base0);
        const p1 = try self.residentProjection(base1);
        const p2 = try self.residentProjection(base2);
        if (p0.k != p1.k or p0.k != p2.k or input.len == 0 or input.len % p0.k != 0) return error.UnexpectedTensor;
        const n = input.len / p0.k;
        if (output0.len != p0.m * n or output1.len != p1.m * n or output2.len != p2.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var loaded0: ?CachedF16Weight = null;
        defer if (loaded0) |*loaded| loaded.close(self.io);
        var loaded1: ?CachedF16Weight = null;
        defer if (loaded1) |*loaded| loaded.close(self.io);
        var loaded2: ?CachedF16Weight = null;
        defer if (loaded2) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            loaded0 = try self.loadCachedF16(p0.key, p0.k, p0.m) orelse return error.MissingResidentWeight;
            loaded1 = try self.loadCachedF16(p1.key, p1.k, p1.m) orelse return error.MissingResidentWeight;
            loaded2 = try self.loadCachedF16(p2.key, p2.k, p2.m) orelse return error.MissingResidentWeight;
        }
        const weight0: ?[*]const u16 = if (loaded0) |loaded| loaded.weight.ptr else null;
        const weight1: ?[*]const u16 = if (loaded1) |loaded| loaded.weight.ptr else null;
        const weight2: ?[*]const u16 = if (loaded2) |loaded| loaded.weight.ptr else null;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_matmul_f16_resident_group_once(self.backend, slot, 3, p0.k, p0.m, p1.m, p2.m, n, weight0, weight1, weight2, input.ptr, output0.ptr, output1.ptr, output2.ptr, &ms, &uploaded);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("resident triple bridge failed: status {d}, slot {d}, k {d}, m {d}/{d}/{d}, n {d}\n", .{ status, slot, p0.k, p0.m, p1.m, p2.m, n });
            return error.BridgeFailed;
        }
        self.projection_ms += ms;
        self.projection_count += 3;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 3;
            self.resident_weight_bytes_uploaded += (p0.k * p0.m + p1.k * p1.m + p2.k * p2.m) * @sizeOf(u16);
        } else {
            self.resident_weight_hits += 3;
        }
    }

    fn mlpResident(self: *Runner, gate_base: []const u8, up_base: []const u8, down_base: []const u8, input: []const f32, output: []f32) !void {
        const gate = try self.residentProjection(gate_base);
        const up = try self.residentProjection(up_base);
        const down = try self.residentProjection(down_base);
        if (gate.k != up.k or gate.m != up.m or down.k != gate.m or input.len == 0 or input.len % gate.k != 0) return error.UnexpectedTensor;
        const n = input.len / gate.k;
        if (output.len != down.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var gate_loaded: ?CachedF16Weight = null;
        defer if (gate_loaded) |*loaded| loaded.close(self.io);
        var up_loaded: ?CachedF16Weight = null;
        defer if (up_loaded) |*loaded| loaded.close(self.io);
        var down_loaded: ?CachedF16Weight = null;
        defer if (down_loaded) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            gate_loaded = try self.loadCachedF16(gate.key, gate.k, gate.m) orelse return error.MissingResidentWeight;
            up_loaded = try self.loadCachedF16(up.key, up.k, up.m) orelse return error.MissingResidentWeight;
            down_loaded = try self.loadCachedF16(down.key, down.k, down.m) orelse return error.MissingResidentWeight;
        }
        const gate_weight: ?[*]const u16 = if (gate_loaded) |loaded| loaded.weight.ptr else null;
        const up_weight: ?[*]const u16 = if (up_loaded) |loaded| loaded.weight.ptr else null;
        const down_weight: ?[*]const u16 = if (down_loaded) |loaded| loaded.weight.ptr else null;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_mlp_f16_resident_once(self.backend, slot, gate.k, gate.m, down.m, n, gate_weight, up_weight, down_weight, input.ptr, output.ptr, &ms, &uploaded);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("resident MLP bridge failed: status {d}, slot {d}, widths {d}/{d}/{d}, n {d}\n", .{ status, slot, gate.k, gate.m, down.m, n });
            return error.BridgeFailed;
        }
        self.projection_ms += ms;
        self.projection_count += 3;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 3;
            self.resident_weight_bytes_uploaded += (gate.k * gate.m + up.k * up.m + down.k * down.m) * @sizeOf(u16);
        } else {
            self.resident_weight_hits += 3;
        }
    }

    fn projectPairQuant(self: *Runner, base0: []const u8, base1: []const u8, input: []const f32, output0: []f32, output1: []f32) !void {
        const p0 = try self.residentProjectionF32(base0);
        const p1 = try self.residentProjectionF32(base1);
        if (p0.k != p1.k or input.len == 0 or input.len % p0.k != 0) return error.UnexpectedTensor;
        const n = input.len / p0.k;
        if (output0.len != p0.m * n or output1.len != p1.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var loaded0: ?CachedWeight = null;
        defer if (loaded0) |*loaded| loaded.close(self.io);
        var loaded1: ?CachedWeight = null;
        defer if (loaded1) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            loaded0 = try self.loadCached(p0.key, p0.k, p0.m) orelse return error.MissingResidentWeight;
            loaded1 = try self.loadCached(p1.key, p1.k, p1.m) orelse return error.MissingResidentWeight;
        }
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const q4 = self.use_q4_weights and !self.q8_attention_inputs;
        const status = if (q4)
            tf_qualcomm_backend_matmul_q4_resident_group_once(self.backend, slot, 2, p0.k, p0.m, p1.m, 0, n, if (loaded0) |loaded| loaded.weight.ptr else null, if (loaded1) |loaded| loaded.weight.ptr else null, null, input.ptr, output0.ptr, output1.ptr, null, &ms, &uploaded)
        else
            tf_qualcomm_backend_matmul_q8_resident_group_once(self.backend, slot, 2, p0.k, p0.m, p1.m, 0, n, if (loaded0) |loaded| loaded.weight.ptr else null, if (loaded1) |loaded| loaded.weight.ptr else null, null, input.ptr, output0.ptr, output1.ptr, null, &ms, &uploaded);
        const wall_ms = tf_qualcomm_monotonic_ms() - started;
        self.projection_wall_ms += wall_ms;
        self.pair_quant_calls += 1;
        self.pair_quant_compute_ms += ms;
        self.pair_quant_wall_ms += wall_ms;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 2;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 2;
            self.resident_weight_bytes_uploaded += ((p0.k / 32) * p0.m + (p1.k / 32) * p1.m) * (if (q4) @as(usize, 18) else 34);
        } else self.resident_weight_hits += 2;
    }

    fn projectTripleQuant(self: *Runner, base0: []const u8, base1: []const u8, base2: []const u8, input: []const f32, output0: []f32, output1: []f32, output2: []f32) !void {
        const p0 = try self.residentProjectionF32(base0);
        const p1 = try self.residentProjectionF32(base1);
        const p2 = try self.residentProjectionF32(base2);
        if (p0.k != p1.k or p0.k != p2.k or input.len == 0 or input.len % p0.k != 0) return error.UnexpectedTensor;
        const n = input.len / p0.k;
        if (output0.len != p0.m * n or output1.len != p1.m * n or output2.len != p2.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var loaded0: ?CachedWeight = null;
        defer if (loaded0) |*loaded| loaded.close(self.io);
        var loaded1: ?CachedWeight = null;
        defer if (loaded1) |*loaded| loaded.close(self.io);
        var loaded2: ?CachedWeight = null;
        defer if (loaded2) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            loaded0 = try self.loadCached(p0.key, p0.k, p0.m) orelse return error.MissingResidentWeight;
            loaded1 = try self.loadCached(p1.key, p1.k, p1.m) orelse return error.MissingResidentWeight;
            loaded2 = try self.loadCached(p2.key, p2.k, p2.m) orelse return error.MissingResidentWeight;
        }
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const q4 = self.use_q4_weights and !self.q8_attention_inputs;
        const status = if (q4)
            tf_qualcomm_backend_matmul_q4_resident_group_once(self.backend, slot, 3, p0.k, p0.m, p1.m, p2.m, n, if (loaded0) |loaded| loaded.weight.ptr else null, if (loaded1) |loaded| loaded.weight.ptr else null, if (loaded2) |loaded| loaded.weight.ptr else null, input.ptr, output0.ptr, output1.ptr, output2.ptr, &ms, &uploaded)
        else
            tf_qualcomm_backend_matmul_q8_resident_group_once(self.backend, slot, 3, p0.k, p0.m, p1.m, p2.m, n, if (loaded0) |loaded| loaded.weight.ptr else null, if (loaded1) |loaded| loaded.weight.ptr else null, if (loaded2) |loaded| loaded.weight.ptr else null, input.ptr, output0.ptr, output1.ptr, output2.ptr, &ms, &uploaded);
        const wall_ms = tf_qualcomm_monotonic_ms() - started;
        self.projection_wall_ms += wall_ms;
        self.triple_quant_calls += 1;
        self.triple_quant_compute_ms += ms;
        self.triple_quant_wall_ms += wall_ms;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 3;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 3;
            self.resident_weight_bytes_uploaded += ((p0.k / 32) * p0.m + (p1.k / 32) * p1.m + (p2.k / 32) * p2.m) * (if (q4) @as(usize, 18) else 34);
        } else self.resident_weight_hits += 3;
    }

    fn mlpQuant(self: *Runner, gate_base: []const u8, up_base: []const u8, down_base: []const u8, input: []const f32, output: []f32) !void {
        const gate = try self.residentProjectionF32(gate_base);
        const up = try self.residentProjectionF32(up_base);
        const down = try self.residentProjectionF32(down_base);
        if (gate.k != up.k or gate.m != up.m or down.k != gate.m or input.len == 0 or input.len % gate.k != 0) return error.UnexpectedTensor;
        const n = input.len / gate.k;
        if (output.len != down.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var gate_loaded: ?CachedWeight = null;
        defer if (gate_loaded) |*loaded| loaded.close(self.io);
        var up_loaded: ?CachedWeight = null;
        defer if (up_loaded) |*loaded| loaded.close(self.io);
        var down_loaded: ?CachedWeight = null;
        defer if (down_loaded) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            gate_loaded = try self.loadCached(gate.key, gate.k, gate.m) orelse return error.MissingResidentWeight;
            up_loaded = try self.loadCached(up.key, up.k, up.m) orelse return error.MissingResidentWeight;
            down_loaded = try self.loadCached(down.key, down.k, down.m) orelse return error.MissingResidentWeight;
        }
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const q4 = self.use_q4_weights and !self.q8_mlp;
        const status = if (q4)
            tf_qualcomm_backend_mlp_q4_resident_once(self.backend, slot, gate.k, gate.m, down.m, n, if (gate_loaded) |loaded| loaded.weight.ptr else null, if (up_loaded) |loaded| loaded.weight.ptr else null, if (down_loaded) |loaded| loaded.weight.ptr else null, input.ptr, output.ptr, &ms, &uploaded)
        else
            tf_qualcomm_backend_mlp_q8_resident_once(self.backend, slot, gate.k, gate.m, down.m, n, if (gate_loaded) |loaded| loaded.weight.ptr else null, if (up_loaded) |loaded| loaded.weight.ptr else null, if (down_loaded) |loaded| loaded.weight.ptr else null, input.ptr, output.ptr, &ms, &uploaded);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 3;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 3;
            self.resident_weight_bytes_uploaded += ((gate.k / 32) * gate.m + (up.k / 32) * up.m + (down.k / 32) * down.m) * (if (q4) @as(usize, 18) else 34);
        } else self.resident_weight_hits += 3;
    }

    fn layerTailQuant(self: *Runner, layer: usize, output_base: []const u8, post_norm_name: []const u8, state: []f32, attention: []const f32) !void {
        var base_buffer: [256]u8 = undefined;
        const mlp_base = try tensorName(&base_buffer, "{s}.layers.{d}.mlp", .{ prefix, layer });
        var gate_buffer: [256]u8 = undefined;
        var up_buffer: [256]u8 = undefined;
        var down_buffer: [256]u8 = undefined;
        const gate_base = try tensorName(&gate_buffer, "{s}.gate_proj", .{mlp_base});
        const up_base = try tensorName(&up_buffer, "{s}.up_proj", .{mlp_base});
        const down_base = try tensorName(&down_buffer, "{s}.down_proj", .{mlp_base});
        const output_projection = try self.residentProjectionF32(output_base);
        const gate = try self.residentProjectionF32(gate_base);
        const up = try self.residentProjectionF32(up_base);
        const down = try self.residentProjectionF32(down_base);
        if (output_projection.m != hidden or gate.k != hidden or up.k != hidden or gate.m != intermediate or
            up.m != intermediate or down.k != intermediate or down.m != hidden or state.len == 0 or
            state.len % hidden != 0) return error.UnexpectedTensor;
        const n = state.len / hidden;
        if (attention.len != n * output_projection.k) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;

        var output_loaded: ?CachedWeight = null;
        defer if (output_loaded) |*loaded| loaded.close(self.io);
        var gate_loaded: ?CachedWeight = null;
        defer if (gate_loaded) |*loaded| loaded.close(self.io);
        var up_loaded: ?CachedWeight = null;
        defer if (up_loaded) |*loaded| loaded.close(self.io);
        var down_loaded: ?CachedWeight = null;
        defer if (down_loaded) |*loaded| loaded.close(self.io);
        var norm_weight: ?[]f32 = null;
        defer if (norm_weight) |values| self.allocator.free(values);
        if (!self.resident_loaded[slot]) {
            output_loaded = try self.loadCached(output_projection.key, output_projection.k, output_projection.m) orelse return error.MissingResidentWeight;
            gate_loaded = try self.loadCached(gate.key, gate.k, gate.m) orelse return error.MissingResidentWeight;
            up_loaded = try self.loadCached(up.key, up.k, up.m) orelse return error.MissingResidentWeight;
            down_loaded = try self.loadCached(down.key, down.k, down.m) orelse return error.MissingResidentWeight;
            const norm = self.file.lookup(post_norm_name) orelse return error.MissingTensor;
            if ((norm.dtype != .f16 and norm.dtype != .bf16 and norm.dtype != .f32) or norm.rank != 1 or norm.dim(0) != hidden) return error.UnexpectedTensor;
            const values = try self.allocator.alloc(f32, hidden);
            for (values, 0..) |*x, i| x.* = value(norm, i) + 1.0;
            norm_weight = values;
        }

        const output_q8 = self.outputUsesQ8(output_base);
        var compute_ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_layer_tail_q4_resident_once(
            self.backend,
            slot,
            output_projection.k,
            hidden,
            intermediate,
            n,
            @intFromBool(output_q8),
            if (output_loaded) |loaded| loaded.weight.ptr else null,
            if (gate_loaded) |loaded| loaded.weight.ptr else null,
            if (up_loaded) |loaded| loaded.weight.ptr else null,
            if (down_loaded) |loaded| loaded.weight.ptr else null,
            if (norm_weight) |values| values.ptr else null,
            state.ptr,
            attention.ptr,
            &compute_ms,
            &uploaded,
        );
        const wall_ms = tf_qualcomm_monotonic_ms() - started;
        self.projection_wall_ms += wall_ms;
        self.layer_tail_calls += 1;
        self.layer_tail_compute_ms += compute_ms;
        self.layer_tail_wall_ms += wall_ms;
        if (status != 0) {
            std.debug.print("resident fused tail bridge failed: status {d}, layer {d}, slot {d}, n {d}\n", .{ status, layer, slot, n });
            return error.BridgeFailed;
        }
        self.projection_ms += compute_ms;
        self.projection_count += 4;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 4;
            const output_block_bytes: usize = if (output_q8) 34 else 18;
            self.resident_weight_bytes_uploaded += (output_projection.k / 32) * output_projection.m * output_block_bytes +
                ((gate.k / 32) * gate.m + (up.k / 32) * up.m + (down.k / 32) * down.m) * 18 + hidden * @sizeOf(f32);
        } else self.resident_weight_hits += 4;
    }

    fn linearTailQuant(self: *Runner, layer: usize, count: usize, state: []f32, z: []const f32, query: []const f32, key: []const f32, value_input: []const f32, log_decay: []const f32, beta: []const f32, initial_gdn_state: []const f32) !void {
        const expanded_len = count * linear_output_width;
        if (state.len != count * hidden or z.len != expanded_len or query.len != count * linear_query_width or
            key.len != count * linear_query_width or value_input.len != expanded_len or
            log_decay.len != count * linear_value_heads or beta.len != count * linear_value_heads) return error.UnexpectedTensor;
        const expanded_query = try self.allocator.alloc(f32, expanded_len);
        defer self.allocator.free(expanded_query);
        const expanded_key = try self.allocator.alloc(f32, expanded_len);
        defer self.allocator.free(expanded_key);
        const values_per_key = linear_value_heads / linear_key_heads;
        for (0..count) |position| for (0..linear_value_heads) |head| {
            const key_head = head / values_per_key;
            const source = (position * linear_key_heads + key_head) * linear_key_width;
            const target = (position * linear_value_heads + head) * linear_value_width;
            @memcpy(expanded_query[target..][0..linear_value_width], query[source..][0..linear_key_width]);
            @memcpy(expanded_key[target..][0..linear_value_width], key[source..][0..linear_key_width]);
        };

        var base_buffer: [256]u8 = undefined;
        const layer_base = try tensorName(&base_buffer, "{s}.layers.{d}", .{ prefix, layer });
        var output_buffer: [256]u8 = undefined;
        var gate_buffer: [256]u8 = undefined;
        var up_buffer: [256]u8 = undefined;
        var down_buffer: [256]u8 = undefined;
        var linear_norm_buffer: [256]u8 = undefined;
        var post_norm_buffer: [256]u8 = undefined;
        const output_base = try tensorName(&output_buffer, "{s}.linear_attn.out_proj", .{layer_base});
        const gate_base = try tensorName(&gate_buffer, "{s}.mlp.gate_proj", .{layer_base});
        const up_base = try tensorName(&up_buffer, "{s}.mlp.up_proj", .{layer_base});
        const down_base = try tensorName(&down_buffer, "{s}.mlp.down_proj", .{layer_base});
        const linear_norm_name = try tensorName(&linear_norm_buffer, "{s}.linear_attn.norm.weight", .{layer_base});
        const post_norm_name = try tensorName(&post_norm_buffer, "{s}.post_attention_layernorm.weight", .{layer_base});
        if (self.outputUsesQ8(output_base)) return error.UnsupportedFusedPrecision;
        const output_projection = try self.residentProjectionF32(output_base);
        const gate = try self.residentProjectionF32(gate_base);
        const up = try self.residentProjectionF32(up_base);
        const down = try self.residentProjectionF32(down_base);
        const slot = self.resident_slot_cursor;
        self.fused_tail_slot[layer] = slot;
        var output_loaded: ?CachedWeight = null;
        defer if (output_loaded) |*loaded| loaded.close(self.io);
        var gate_loaded: ?CachedWeight = null;
        defer if (gate_loaded) |*loaded| loaded.close(self.io);
        var up_loaded: ?CachedWeight = null;
        defer if (up_loaded) |*loaded| loaded.close(self.io);
        var down_loaded: ?CachedWeight = null;
        defer if (down_loaded) |*loaded| loaded.close(self.io);
        var linear_norm_values: ?[]f32 = null;
        defer if (linear_norm_values) |values| self.allocator.free(values);
        var post_norm_values: ?[]f32 = null;
        defer if (post_norm_values) |values| self.allocator.free(values);
        if (!self.resident_loaded[slot]) {
            output_loaded = try self.loadCached(output_projection.key, output_projection.k, output_projection.m) orelse return error.MissingResidentWeight;
            gate_loaded = try self.loadCached(gate.key, gate.k, gate.m) orelse return error.MissingResidentWeight;
            up_loaded = try self.loadCached(up.key, up.k, up.m) orelse return error.MissingResidentWeight;
            down_loaded = try self.loadCached(down.key, down.k, down.m) orelse return error.MissingResidentWeight;
            const linear_norm = try require(self.file, linear_norm_name, .f32, &.{linear_value_width});
            const linear_values = try self.allocator.alloc(f32, linear_value_width);
            for (linear_values, 0..) |*x, i| x.* = value(linear_norm, i);
            linear_norm_values = linear_values;
            const post_norm = self.file.lookup(post_norm_name) orelse return error.MissingTensor;
            if ((post_norm.dtype != .f16 and post_norm.dtype != .bf16 and post_norm.dtype != .f32) or post_norm.rank != 1 or post_norm.dim(0) != hidden) return error.UnexpectedTensor;
            const post_values = try self.allocator.alloc(f32, hidden);
            for (post_values, 0..) |*x, i| x.* = value(post_norm, i) + 1.0;
            post_norm_values = post_values;
        }
        var compute_ms: f64 = 0;
        var uploaded: c_int = 0;
        const state_ptr: ?[*]const f32 = if (self.resident_gdn_loaded[layer]) null else initial_gdn_state.ptr;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_linear_tail_q4_resident_once(
            self.backend,
            slot,
            linear_value_width,
            linear_value_heads,
            hidden,
            intermediate,
            count,
            if (output_loaded) |loaded| loaded.weight.ptr else null,
            if (gate_loaded) |loaded| loaded.weight.ptr else null,
            if (up_loaded) |loaded| loaded.weight.ptr else null,
            if (down_loaded) |loaded| loaded.weight.ptr else null,
            if (linear_norm_values) |values| values.ptr else null,
            if (post_norm_values) |values| values.ptr else null,
            state.ptr,
            z.ptr,
            expanded_query.ptr,
            expanded_key.ptr,
            value_input.ptr,
            log_decay.ptr,
            beta.ptr,
            state_ptr,
            &compute_ms,
            &uploaded,
        );
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("resident fused linear tail bridge failed: status {d}, layer {d}, slot {d}, n {d}\n", .{ status, layer, slot, count });
            return error.BridgeFailed;
        }
        self.projection_ms += compute_ms;
        self.projection_count += 4;
        self.resident_slot_cursor += 1;
        self.resident_gdn_loaded[layer] = true;
        self.resident_gdn_hits += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 4;
            self.resident_weight_bytes_uploaded += ((output_projection.k / 32) * output_projection.m +
                (gate.k / 32) * gate.m + (up.k / 32) * up.m + (down.k / 32) * down.m) * 18 +
                (linear_value_width + hidden) * @sizeOf(f32);
        } else self.resident_weight_hits += 4;
    }

    fn gdnResident(self: *Runner, layer: usize, count: usize, query: []const f32, key: []const f32, value_input: []const f32, log_decay: []const f32, beta: []const f32, initial_state: []const f32, output: []f32) !void {
        const expanded_len = count * linear_output_width;
        if (query.len != count * linear_query_width or key.len != count * linear_query_width or
            value_input.len != expanded_len or log_decay.len != count * linear_value_heads or
            beta.len != count * linear_value_heads or output.len != expanded_len) return error.UnexpectedTensor;
        const expanded_query = try self.allocator.alloc(f32, expanded_len);
        defer self.allocator.free(expanded_query);
        const expanded_key = try self.allocator.alloc(f32, expanded_len);
        defer self.allocator.free(expanded_key);
        const values_per_key = linear_value_heads / linear_key_heads;
        for (0..count) |position| {
            for (0..linear_value_heads) |head| {
                const key_head = head / values_per_key;
                const source = (position * linear_key_heads + key_head) * linear_key_width;
                const target = (position * linear_value_heads + head) * linear_value_width;
                @memcpy(expanded_query[target..][0..linear_value_width], query[source..][0..linear_key_width]);
                @memcpy(expanded_key[target..][0..linear_value_width], key[source..][0..linear_key_width]);
            }
        }
        var compute_ms: f64 = 0;
        const state_ptr: ?[*]const f32 = if (self.resident_gdn_loaded[layer]) null else initial_state.ptr;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_gdn_f32_resident_once(self.backend, layer, linear_value_width, linear_value_heads, count, expanded_query.ptr, expanded_key.ptr, value_input.ptr, log_decay.ptr, beta.ptr, state_ptr, output.ptr, &compute_ms);
        self.resident_gdn_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("resident GDN bridge failed: status {d}, layer {d}, tokens {d}\n", .{ status, layer, count });
            return error.BackendRunFailed;
        }
        self.resident_gdn_loaded[layer] = true;
        self.resident_gdn_hits += 1;
        self.resident_gdn_compute_ms += compute_ms;
        const undo_scale = @sqrt(@as(f32, @floatFromInt(linear_value_width)));
        for (output) |*x| x.* = roundBF16(x.* * undo_scale);
    }

    fn convertSaveRunF16(self: *Runner, key: [32]u8, k: usize, m: usize, weight: []const f32, input: []const f32, output: []f32) !void {
        const converted = try self.allocator.alloc(u16, weight.len);
        defer self.allocator.free(converted);
        const convert_started = tf_qualcomm_monotonic_ms();
        tf_qualcomm_f32_to_f16(weight.ptr, converted.ptr, weight.len);
        self.weight_convert_ms += tf_qualcomm_monotonic_ms() - convert_started;
        try self.saveCachedF16(key, k, m, converted);
        try self.runWeightF16(k, m, converted, input, output);
    }

    fn project(self: *Runner, base: []const u8, input: []const f32, output: []f32) !void {
        var name_buffer: [256]u8 = undefined;
        const trellis_tensor = self.file.lookup(try tensorName(&name_buffer, "{s}.trellis", .{base})) orelse return error.MissingTensor;
        if (trellis_tensor.dtype != .i16 or trellis_tensor.rank != 3) return error.UnexpectedTensor;
        const tiles_k = trellis_tensor.dim(0);
        const tiles_m = trellis_tensor.dim(1);
        const words = trellis_tensor.dim(2);
        if (words % 8 != 0) return error.UnexpectedTensor;
        const k = tiles_k * 16;
        const m = tiles_m * 16;
        if (input.len == 0 or input.len % k != 0 or output.len != m * (input.len / k)) return error.UnexpectedTensor;

        const suh_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.suh", .{base}), .f16, &.{k});
        const svh_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.svh", .{base}), .f16, &.{m});
        const mcg_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.mcg", .{base}), .i32, &.{});
        if (std.mem.readInt(u32, mcg_tensor.bytes[0..4], .little) != exl3.McgMul) return error.UnexpectedCodebook;

        if (self.use_packed_exl3) {
            if (self.backend_is_cpu) {
                const started = tf_qualcomm_monotonic_ms();
                try packedCpuProject(self.allocator, trellis_tensor.bytes, suh_tensor.bytes, svh_tensor.bytes, k, m, words, input, output, self.cpu_threads);
                const elapsed = tf_qualcomm_monotonic_ms() - started;
                self.projection_ms += elapsed;
                self.projection_wall_ms += elapsed;
                self.projection_count += 1;
                self.resident_slot_cursor += 1;
                self.resident_weight_uploads += 1;
                self.resident_weight_bytes_uploaded += trellis_tensor.bytes.len + suh_tensor.bytes.len + svh_tensor.bytes.len;
                return;
            }
            const slot = self.resident_slot_cursor;
            var compute_ms: f64 = 0;
            var uploaded: c_int = 0;
            const trellis_ptr: ?[*]const u16 = if (self.resident_loaded[slot]) null else @ptrCast(@alignCast(trellis_tensor.bytes.ptr));
            const suh_ptr: ?[*]const u16 = if (self.resident_loaded[slot]) null else @ptrCast(@alignCast(suh_tensor.bytes.ptr));
            const svh_ptr: ?[*]const u16 = if (self.resident_loaded[slot]) null else @ptrCast(@alignCast(svh_tensor.bytes.ptr));
            const started = tf_qualcomm_monotonic_ms();
            const status = tf_qualcomm_backend_exl3_resident_once(self.backend, slot, k, m, input.len / k, words, trellis_ptr, suh_ptr, svh_ptr, input.ptr, output.ptr, &compute_ms, &uploaded);
            self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
            if (status != 0) {
                std.debug.print("packed resident bridge failed: status={d} slot={d} k={d} m={d} n={d} words={d}\n", .{ status, slot, k, m, input.len / k, words });
                return error.BridgeFailed;
            }
            self.projection_ms += compute_ms;
            self.projection_count += 1;
            self.resident_slot_cursor += 1;
            if (uploaded != 0) {
                self.resident_loaded[slot] = true;
                self.resident_weight_uploads += 1;
                self.resident_weight_bytes_uploaded += trellis_tensor.bytes.len + suh_tensor.bytes.len + svh_tensor.bytes.len;
            } else {
                self.resident_weight_hits += 1;
            }
            return;
        }

        const resident_slot = self.resident_slot_cursor;
        const q8_output = self.outputUsesQ8(base);
        const q4_quant = self.use_q4_weights and !q8_output;
        if ((self.use_q8_weights or self.use_q4_weights) and self.resident_loaded[resident_slot]) {
            return self.runWeightQuant(k, m, null, input, output, q4_quant);
        }

        const f16_key: ?[32]u8 = if (self.use_f16_weights) if (self.cache_model_id) |model_id|
            f16ProjectionKey(model_id, base, k, m, words)
        else
            null else null;
        if (self.use_resident_weights and self.resident_loaded[resident_slot]) {
            const started = tf_qualcomm_monotonic_ms();
            if (self.runResidentWeightF16(resident_slot, k, m, null, input, output)) {
                self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
                return;
            }
        }
        if (f16_key) |key| {
            if (try self.loadCachedF16(key, k, m)) |loaded_value| {
                var loaded = loaded_value;
                defer loaded.close(self.io);
                return self.runWeightF16(k, m, loaded.weight, input, output);
            }
            self.f16_cache_misses += 1;
        }

        var cache_key: ?[32]u8 = null;
        if (self.cache_dir != null) {
            const trusted_key = if (self.cache_model_id) |model_id| trustedProjectionKey(model_id, base, k, m, words) else null;
            if (trusted_key) |key| if (try self.loadCached(key, k, m)) |loaded_value| {
                self.cache_trusted_hits += 1;
                var loaded = loaded_value;
                defer loaded.close(self.io);
                if (f16_key) |f16_key_value| return self.convertSaveRunF16(f16_key_value, k, m, loaded.weight, input, output);
                return self.runDecodedWeight(k, m, loaded.weight, input, output, q4_quant);
            };
            const hash_started = tf_qualcomm_monotonic_ms();
            const key = projectionKey(trellis_tensor, suh_tensor, svh_tensor, mcg_tensor, k, m, words);
            self.cache_hash_ms += tf_qualcomm_monotonic_ms() - hash_started;
            cache_key = key;
            if (try self.loadCached(key, k, m)) |loaded_value| {
                if (trusted_key) |alias| try self.linkCached(key, alias);
                var loaded = loaded_value;
                defer loaded.close(self.io);
                if (f16_key) |f16_key_value| return self.convertSaveRunF16(f16_key_value, k, m, loaded.weight, input, output);
                return self.runDecodedWeight(k, m, loaded.weight, input, output, q4_quant);
            }
            self.cache_misses += 1;
        }

        const trellis = try self.allocator.alloc(i16, trellis_tensor.numel());
        defer self.allocator.free(trellis);
        @memcpy(std.mem.sliceAsBytes(trellis), trellis_tensor.bytes);
        const suh = try self.allocator.alloc(f16, k);
        defer self.allocator.free(suh);
        @memcpy(std.mem.sliceAsBytes(suh), suh_tensor.bytes);
        const svh = try self.allocator.alloc(f16, m);
        defer self.allocator.free(svh);
        @memcpy(std.mem.sliceAsBytes(svh), svh_tensor.bytes);

        const elements = try std.math.mul(usize, k, m);
        const decoded = try self.allocator.alloc(f32, elements);
        defer self.allocator.free(decoded);
        const unpacked = try self.allocator.alloc(f16, elements);
        defer self.allocator.free(unpacked);
        const decode_started = tf_qualcomm_monotonic_ms();
        exl3.dequantizeFastF32(.{ .halves = @intCast(words / 8) }, .mcg, trellis, tiles_k, tiles_m, suh, svh, decoded, unpacked);
        self.dequant_ms += tf_qualcomm_monotonic_ms() - decode_started;

        // EXL3 reconstructs [K,M]; GGML's matrix operand is contiguous [M,K].
        const weight = try self.allocator.alloc(f32, elements);
        defer self.allocator.free(weight);
        for (0..m) |out_col| for (0..k) |inner| {
            weight[out_col * k + inner] = decoded[inner * m + out_col];
        };
        if (f16_key == null) {
            if (cache_key) |key| try self.saveCached(key, k, m, weight);
            if (cache_key) |key| if (self.cache_model_id) |model_id| {
                try self.linkCached(key, trustedProjectionKey(model_id, base, k, m, words));
            };
        }
        if (f16_key) |f16_key_value| {
            try self.convertSaveRunF16(f16_key_value, k, m, weight, input, output);
        } else {
            try self.runDecodedWeight(k, m, weight, input, output, q4_quant);
        }
    }

    fn projectPaged(self: *Runner, base: []const u8, input: []const f32, output: []f32) !void {
        const cache = self.expert_cache orelse return error.MissingExpertPager;
        var trellis_name_buf: [256]u8 = undefined;
        var suh_name_buf: [256]u8 = undefined;
        var svh_name_buf: [256]u8 = undefined;
        var mcg_name_buf: [256]u8 = undefined;
        const trellis_name = try tensorName(&trellis_name_buf, "{s}.trellis", .{base});
        const suh_name = try tensorName(&suh_name_buf, "{s}.suh", .{base});
        const svh_name = try tensorName(&svh_name_buf, "{s}.svh", .{base});
        const mcg_name = try tensorName(&mcg_name_buf, "{s}.mcg", .{base});
        const trellis_span = cache.catalog.get(trellis_name) orelse return error.MissingTensor;
        const suh_span = cache.catalog.get(suh_name) orelse return error.MissingTensor;
        const svh_span = cache.catalog.get(svh_name) orelse return error.MissingTensor;
        if (trellis_span.dtype != .i16 or trellis_span.rank != 3 or suh_span.dtype != .f16 or svh_span.dtype != .f16) return error.UnexpectedTensor;
        const tiles_k = trellis_span.shape[0];
        const tiles_m = trellis_span.shape[1];
        const words = trellis_span.shape[2];
        const k = tiles_k * 16;
        const m = tiles_m * 16;
        if (words % 8 != 0 or suh_span.rank != 1 or svh_span.rank != 1 or suh_span.shape[0] != k or svh_span.shape[0] != m or input.len == 0 or input.len % k != 0 or output.len != m * (input.len / k)) return error.UnexpectedTensor;

        var mcg_lease = try cache.acquire(mcg_name, null);
        defer mcg_lease.release();
        if (mcg_lease.data().len != 4 or std.mem.readInt(u32, mcg_lease.data()[0..4], .little) != exl3.McgMul) return error.UnexpectedCodebook;
        var trellis_lease = try cache.acquire(trellis_name, null);
        defer trellis_lease.release();
        var suh_lease = try cache.acquire(suh_name, null);
        defer suh_lease.release();
        var svh_lease = try cache.acquire(svh_name, null);
        defer svh_lease.release();
        if (self.backend_is_cpu) {
            const started = tf_qualcomm_monotonic_ms();
            try packedCpuProject(self.allocator, trellis_lease.data(), suh_lease.data(), svh_lease.data(), k, m, words, input, output, self.cpu_threads);
            const elapsed = tf_qualcomm_monotonic_ms() - started;
            self.projection_ms += elapsed;
            self.projection_wall_ms += elapsed;
            self.projection_count += 1;
            self.resident_weight_uploads += 1;
            self.resident_weight_bytes_uploaded += trellis_lease.data().len + suh_lease.data().len + svh_lease.data().len;
            return;
        }
        const trellis_ptr: [*]const u16 = @ptrCast(@alignCast(trellis_lease.data().ptr));
        const suh_ptr: [*]const u16 = @ptrCast(@alignCast(suh_lease.data().ptr));
        const svh_ptr: [*]const u16 = @ptrCast(@alignCast(svh_lease.data().ptr));
        const slot: usize = 232 + (if (k == hidden) @as(usize, 0) else 8) + words / 8 - 1;
        var compute_ms: f64 = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_exl3_paged_once(self.backend, slot, k, m, input.len / k, words, trellis_ptr, suh_ptr, svh_ptr, input.ptr, output.ptr, &compute_ms);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("packed paged bridge failed: status={d} slot={d} k={d} m={d} n={d} words={d}\n", .{ status, slot, k, m, input.len / k, words });
            return error.BridgeFailed;
        }
        self.projection_ms += compute_ms;
        self.projection_count += 1;
        self.resident_weight_uploads += 1;
        self.resident_weight_bytes_uploaded += trellis_lease.data().len + suh_lease.data().len + svh_lease.data().len;
    }
};

var prof_gdn: f64 = 0;
var prof_dense: f64 = 0;
var prof_pair: f64 = 0;
var prof_conv: f64 = 0;
var prof_fa: f64 = 0;
var prof_on = false;

fn denseBatched(runner: *Runner, name: []const u8, input: []const f32, output: []f32) !void {
    const tensor = runner.file.lookup(name) orelse return error.MissingTensor;
    const count = input.len / hidden;
    if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 2 or tensor.dim(0) != output.len / count or tensor.dim(1) != hidden) return error.UnexpectedTensor;
    const out_w = tensor.dim(0);
    if (tensor.dtype != .f16) {
        for (0..count) |row| try denseVector(runner.file, name, input[row * hidden ..][0..hidden], output[row * out_w ..][0..out_w]);
        return;
    }
    const words: [*]const u16 = @ptrCast(@alignCast(tensor.bytes.ptr));
    var j: usize = 0;
    while (j < out_w) : (j += 1) {
        const w = words[j * hidden ..][0..hidden];
        for (0..count) |row| {
            const x = input[row * hidden ..][0..hidden];
            var acc: @Vector(8, f32) = @splat(0.0);
            var tail: f32 = 0;
            var dd: usize = 0;
            while (dd + 8 <= hidden) : (dd += 8) {
                var wv: [8]f32 = undefined;
                var xv: [8]f32 = undefined;
                for (0..8) |k| {
                    wv[k] = @as(f16, @bitCast(w[dd + k]));
                    xv[k] = x[dd + k];
                }
                acc += @as(@Vector(8, f32), wv) * @as(@Vector(8, f32), xv);
            }
            while (dd < hidden) : (dd += 1) tail += @as(f16, @bitCast(w[dd])) * x[dd];
            output[row * out_w + j] = @reduce(.Add, acc) + tail;
        }
    }
}

fn dense(runner: *Runner, name: []const u8, input: []const f32, output: []f32) !void {
    const tensor = runner.file.lookup(name) orelse return error.MissingTensor;
    if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 2 or tensor.dim(0) != output.len or tensor.dim(1) != input.len) return error.UnexpectedTensor;
    if (runner.use_resident_weights and tensor.dtype == .f16) {
        const t0 = if (prof_on) tf_qualcomm_monotonic_ms() else 0;
        const words: [*]const u16 = @ptrCast(@alignCast(tensor.bytes.ptr));
        tf_qualcomm_dense_f16(words, input.ptr, output.ptr, output.len, input.len);
        if (prof_on) prof_dense += tf_qualcomm_monotonic_ms() - t0;
        return;
    }
    for (output, 0..) |*out, row| {
        var sum: f32 = 0;
        for (input, 0..) |x, col| sum += value(tensor, row * input.len + col) * x;
        out.* = sum;
    }
}

fn rmsNorm(file: *const st.Checkpoint, name: []const u8, input: []const f32, output: []f32, constant_bias: f32) !void {
    const tensor = file.lookup(name) orelse return error.MissingTensor;
    if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 1 or tensor.dim(0) != input.len or output.len != input.len) return error.UnexpectedTensor;
    var sum: f32 = 0;
    for (input) |x| sum += x * x;
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(input.len)) + eps);
    for (output, 0..) |*out, i| out.* = quantizeF16(input[i] * inv * (value(tensor, i) + constant_bias));
}

fn residualAdd(hidden_state: []f32, update: []const f32) void {
    for (hidden_state, update) |*h, u| h.* += u;
}

const DecodeState = struct {
    gdn: []f32,
    conv: []f32,
    keys: []f32,
    values: []f32,
    max_positions: usize,

    fn init(allocator: std.mem.Allocator, max_positions: usize) !DecodeState {
        const gdn = try allocator.alloc(f32, layers * linear_state_elems);
        errdefer allocator.free(gdn);
        const conv = try allocator.alloc(f32, layers * conv_history_rows * conv_channels);
        errdefer allocator.free(conv);
        const keys = try allocator.alloc(f32, layers * max_positions * kv_width);
        errdefer allocator.free(keys);
        const values = try allocator.alloc(f32, layers * max_positions * kv_width);
        errdefer allocator.free(values);
        @memset(gdn, 0.0);
        @memset(conv, 0.0);
        @memset(keys, 0.0);
        @memset(values, 0.0);
        return .{ .gdn = gdn, .conv = conv, .keys = keys, .values = values, .max_positions = max_positions };
    }

    fn deinit(self: *DecodeState, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        allocator.free(self.keys);
        allocator.free(self.conv);
        allocator.free(self.gdn);
        self.* = undefined;
    }

    fn reset(self: *DecodeState, reset_gdn: bool) void {
        if (reset_gdn) @memset(self.gdn, 0.0);
        @memset(self.conv, 0.0);
        @memset(self.keys, 0.0);
        @memset(self.values, 0.0);
    }

    fn gdnLayer(self: *DecodeState, layer: usize) []f32 {
        const start = layer * linear_state_elems;
        return self.gdn[start .. start + linear_state_elems];
    }

    fn convLayer(self: *DecodeState, layer: usize) []f32 {
        const start = layer * conv_history_rows * conv_channels;
        return self.conv[start .. start + conv_history_rows * conv_channels];
    }

    fn keyPosition(self: *DecodeState, layer: usize, position: usize) []f32 {
        const start = (layer * self.max_positions + position) * kv_width;
        return self.keys[start .. start + kv_width];
    }

    fn valuePosition(self: *DecodeState, layer: usize, position: usize) []f32 {
        const start = (layer * self.max_positions + position) * kv_width;
        return self.values[start .. start + kv_width];
    }
};

const PromptSnapshot = struct {
    valid: bool = false,
    prompt_len: usize = 0,
    tokens: [max_worker_prompt_tokens]usize = @splat(0),
    decode: DecodeState,
    hidden_history: []f32,
    normed: []f32,
    next_token: usize = 0,
    mtp_cache: mtp.Cache,
    mtp_valid: bool = false,

    fn init(allocator: std.mem.Allocator) !PromptSnapshot {
        const decode = try DecodeState.init(allocator, max_worker_prompt_tokens);
        errdefer {
            var owned = decode;
            owned.deinit(allocator);
        }
        const hidden_history = try allocator.alloc(f32, max_worker_prompt_tokens * hidden);
        errdefer allocator.free(hidden_history);
        const final_normed = try allocator.alloc(f32, hidden);
        errdefer allocator.free(final_normed);
        const mtp_cache = try mtp.Cache.init(allocator, max_worker_prompt_tokens + 2);
        return .{ .decode = decode, .hidden_history = hidden_history, .normed = final_normed, .mtp_cache = mtp_cache };
    }

    fn deinit(self: *PromptSnapshot, allocator: std.mem.Allocator) void {
        self.mtp_cache.deinit(allocator);
        allocator.free(self.normed);
        allocator.free(self.hidden_history);
        self.decode.deinit(allocator);
        self.* = undefined;
    }

    fn prefixLength(self: *const PromptSnapshot, prompt: []const usize) ?usize {
        if (!self.valid or self.prompt_len > prompt.len) return null;
        return if (std.mem.eql(usize, self.tokens[0..self.prompt_len], prompt[0..self.prompt_len])) self.prompt_len else null;
    }

    fn capture(
        self: *PromptSnapshot,
        backend: ?*anyopaque,
        prompt: []const usize,
        source: *DecodeState,
        hidden_history: []const f32,
        final_normed: []const f32,
        next_token: usize,
    ) !void {
        if (prompt.len == 0 or prompt.len > self.tokens.len or hidden_history.len < prompt.len * hidden or final_normed.len != hidden) return error.InvalidArgument;
        @memcpy(self.tokens[0..prompt.len], prompt);
        @memcpy(self.decode.conv, source.conv);
        for (0..layers) |layer| {
            if ((layer + 1) % 4 != 0) {
                const state = self.decode.gdnLayer(layer);
                if (tf_qualcomm_backend_gdn_state_get(backend, layer, state.ptr, linear_state_elems) != 0) return error.BridgeFailed;
            }
            for (0..prompt.len) |position| {
                @memcpy(self.decode.keyPosition(layer, position), source.keyPosition(layer, position));
                @memcpy(self.decode.valuePosition(layer, position), source.valuePosition(layer, position));
            }
        }
        @memcpy(self.hidden_history[0 .. prompt.len * hidden], hidden_history[0 .. prompt.len * hidden]);
        @memcpy(self.normed, final_normed);
        self.prompt_len = prompt.len;
        self.next_token = next_token;
        self.mtp_valid = false;
        self.valid = true;
    }

    fn restore(
        self: *PromptSnapshot,
        backend: ?*anyopaque,
        runner: *Runner,
        target: *DecodeState,
        hidden_history: []f32,
        final_normed: []f32,
    ) !void {
        if (!self.valid or hidden_history.len < self.prompt_len * hidden or final_normed.len != hidden) return error.InvalidArgument;
        @memcpy(target.conv, self.decode.conv);
        for (0..layers) |layer| {
            if ((layer + 1) % 4 != 0) {
                const state = self.decode.gdnLayer(layer);
                if (tf_qualcomm_backend_gdn_state_set(backend, layer, state.ptr, linear_state_elems) != 0) return error.BridgeFailed;
                runner.resident_gdn_loaded[layer] = true;
            }
            for (0..self.prompt_len) |position| {
                @memcpy(target.keyPosition(layer, position), self.decode.keyPosition(layer, position));
                @memcpy(target.valuePosition(layer, position), self.decode.valuePosition(layer, position));
            }
        }
        @memcpy(hidden_history[0 .. self.prompt_len * hidden], self.hidden_history[0 .. self.prompt_len * hidden]);
        @memcpy(final_normed, self.normed);
    }

    fn captureMtp(self: *PromptSnapshot, source: *const mtp.Cache) !void {
        if (!self.valid or self.prompt_len > self.mtp_cache.positions or self.prompt_len > source.positions) return error.InvalidArgument;
        const key_stride = source.keys.len / source.positions;
        const value_stride = source.values.len / source.positions;
        @memcpy(self.mtp_cache.keys[0 .. self.prompt_len * key_stride], source.keys[0 .. self.prompt_len * key_stride]);
        @memcpy(self.mtp_cache.values[0 .. self.prompt_len * value_stride], source.values[0 .. self.prompt_len * value_stride]);
        self.mtp_valid = true;
    }

    fn restoreMtp(self: *const PromptSnapshot, target: *mtp.Cache) !void {
        if (!self.mtp_valid or self.prompt_len > target.positions) return error.InvalidArgument;
        const key_stride = target.keys.len / target.positions;
        const value_stride = target.values.len / target.positions;
        @memcpy(target.keys[0 .. self.prompt_len * key_stride], self.mtp_cache.keys[0 .. self.prompt_len * key_stride]);
        @memcpy(target.values[0 .. self.prompt_len * value_stride], self.mtp_cache.values[0 .. self.prompt_len * value_stride]);
    }
};

const PromptCache = struct {
    entries: []PromptSnapshot,
    next: usize = 0,

    fn init(allocator: std.mem.Allocator, count: usize) !PromptCache {
        if (count == 0) return error.InvalidArgument;
        const entries = try allocator.alloc(PromptSnapshot, count);
        var initialized: usize = 0;
        errdefer {
            for (entries[0..initialized]) |*entry| entry.deinit(allocator);
            allocator.free(entries);
        }
        while (initialized < count) : (initialized += 1) entries[initialized] = try PromptSnapshot.init(allocator);
        return .{ .entries = entries };
    }

    fn deinit(self: *PromptCache, allocator: std.mem.Allocator) void {
        for (self.entries) |*entry| entry.deinit(allocator);
        allocator.free(self.entries);
        self.* = undefined;
    }

    fn find(self: *PromptCache, prompt: []const usize) ?*PromptSnapshot {
        var best: ?*PromptSnapshot = null;
        for (self.entries) |*entry| {
            const length = entry.prefixLength(prompt) orelse continue;
            if (best == null or length > best.?.prompt_len) best = entry;
        }
        return best;
    }

    fn reserve(self: *PromptCache) *PromptSnapshot {
        const entry = &self.entries[self.next];
        self.next = (self.next + 1) % self.entries.len;
        entry.valid = false;
        entry.mtp_valid = false;
        return entry;
    }
};

fn linearAttention(runner: *Runner, decode: *DecodeState, layer: usize, input: []const f32, residual_state: ?[]f32, output: []f32) !void {
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.linear_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;

    const qkv = try a.alloc(f32, conv_channels);
    defer a.free(qkv);
    const z = try a.alloc(f32, linear_output_width);
    defer a.free(z);
    var qkv_name_buffer: [256]u8 = undefined;
    var z_name_buffer: [256]u8 = undefined;
    const qkv_name = try tensorName(&qkv_name_buffer, "{s}.in_proj_qkv", .{base});
    const z_name = try tensorName(&z_name_buffer, "{s}.in_proj_z", .{base});
    if (runner.use_resident_weights and !runner.use_packed_exl3) {
        if (runner.use_q8_weights or runner.use_q4_weights) try runner.projectPairQuant(qkv_name, z_name, input, qkv, z) else try runner.projectPairResident(qkv_name, z_name, input, qkv, z);
    } else {
        try runner.project(qkv_name, input, qkv);
        try runner.project(z_name, input, z);
    }
    for (qkv) |*x| x.* = roundBF16(roundF16(x.*));

    var a_gate: [linear_value_heads]f32 = undefined;
    var b_gate: [linear_value_heads]f32 = undefined;
    try dense(runner, try tensorName(&projection_buffer, "{s}.in_proj_a.weight", .{base}), input, &a_gate);
    try dense(runner, try tensorName(&projection_buffer, "{s}.in_proj_b.weight", .{base}), input, &b_gate);

    const conv_name = try tensorName(&projection_buffer, "{s}.conv1d.weight", .{base});
    const conv = try require(runner.file, conv_name, .bf16, &.{ conv_channels, 1, 4 });
    const mixed = try a.alloc(f32, conv_channels);
    defer a.free(mixed);
    const history = decode.convLayer(layer);
    for (mixed, 0..) |*out, c| {
        var sum: f32 = value(conv, c * 4 + 3) * qkv[c];
        for (0..conv_history_rows) |tap| sum += value(conv, c * 4 + tap) * history[tap * conv_channels + c];
        const conv_value = roundBF16(sum);
        const conv_sigmoid = roundBF16(sigmoid(conv_value));
        out.* = roundBF16(conv_value * conv_sigmoid);
    }
    std.mem.copyForwards(f32, history[0 * conv_channels .. 1 * conv_channels], history[1 * conv_channels .. 2 * conv_channels]);
    std.mem.copyForwards(f32, history[1 * conv_channels .. 2 * conv_channels], history[2 * conv_channels .. 3 * conv_channels]);
    @memcpy(history[2 * conv_channels .. 3 * conv_channels], qkv);

    var dt_name: [256]u8 = undefined;
    const dt = try require(runner.file, try tensorName(&dt_name, "{s}.dt_bias", .{base}), .bf16, &.{linear_value_heads});
    var alog_name: [256]u8 = undefined;
    const alog = try require(runner.file, try tensorName(&alog_name, "{s}.A_log", .{base}), .f32, &.{linear_value_heads});
    var beta: [linear_value_heads]f32 = undefined;
    var log_decay: [linear_value_heads]f32 = undefined;
    for (0..linear_value_heads) |head| {
        beta[head] = roundBF16(sigmoid(b_gate[head]));
        const s = roundBF16(a_gate[head] + value(dt, head));
        const softplus = roundBF16(@max(s, 0.0) + @log(1.0 + @exp(-@abs(s))));
        log_decay[head] = -@exp(value(alog, head)) * softplus;
    }
    const q = try a.alloc(f32, linear_query_width);
    defer a.free(q);
    const k = try a.alloc(f32, linear_query_width);
    defer a.free(k);
    const v = try a.alloc(f32, linear_output_width);
    defer a.free(v);
    const q_scale = roundBF16(1.0 / @as(f32, @floatFromInt(linear_key_width)));
    const k_scale = roundBF16(1.0 / @sqrt(@as(f32, @floatFromInt(linear_key_width))));
    for (0..linear_key_heads) |head| {
        var qss: f32 = 0;
        var kss: f32 = 0;
        for (0..linear_key_width) |d| {
            const qv = mixed[head * linear_key_width + d];
            const kv = mixed[linear_query_width + head * linear_key_width + d];
            qss += qv * qv;
            kss += kv * kv;
        }
        const width: f32 = @floatFromInt(linear_key_width);
        const qinv = 1.0 / @sqrt(qss / width + eps / width);
        const kinv = 1.0 / @sqrt(kss / width + eps / width);
        for (0..linear_key_width) |d| {
            q[head * linear_key_width + d] = roundBF16(q_scale * roundBF16(mixed[head * linear_key_width + d] * qinv));
            k[head * linear_key_width + d] = roundBF16(k_scale * roundBF16(mixed[linear_query_width + head * linear_key_width + d] * kinv));
        }
    }
    @memcpy(v, mixed[2 * linear_query_width .. conv_channels]);

    if (runner.usesFusedLinearTail(layer)) {
        const target_state = residual_state orelse return error.UnexpectedTensor;
        try runner.linearTailQuant(layer, 1, target_state, z, q, k, v, &log_decay, &beta, decode.gdnLayer(layer));
        return;
    }

    const y = try a.alloc(f32, linear_output_width);
    defer a.free(y);
    const recurrent = decode.gdnLayer(layer);
    if (runner.use_resident_weights) {
        try runner.gdnResident(layer, 1, q, k, v, &log_decay, &beta, recurrent, y);
    } else {
        for (0..linear_value_heads) |head| {
            const decay = @exp(log_decay[head]);
            const key_head = head / (linear_value_heads / linear_key_heads);
            for (0..linear_value_width) |dv| {
                const row = recurrent[(head * linear_value_width + dv) * linear_key_width ..][0..linear_key_width];
                var kv_mem: f32 = 0;
                for (0..linear_key_width) |dk| {
                    row[dk] *= decay;
                    kv_mem += row[dk] * k[key_head * linear_key_width + dk];
                }
                const delta = (v[head * linear_value_width + dv] - kv_mem) * beta[head];
                var out: f32 = 0;
                for (0..linear_key_width) |dk| {
                    row[dk] += k[key_head * linear_key_width + dk] * delta;
                    out += row[dk] * q[key_head * linear_key_width + dk];
                }
                y[head * linear_value_width + dv] = roundBF16(out);
            }
        }
    }

    var norm_name: [256]u8 = undefined;
    const norm = try require(runner.file, try tensorName(&norm_name, "{s}.norm.weight", .{base}), .f32, &.{linear_value_width});
    const gated = try a.alloc(f32, linear_output_width);
    defer a.free(gated);
    for (0..linear_value_heads) |head| {
        var ss: f32 = 0;
        for (0..linear_value_width) |d| {
            const x = y[head * linear_value_width + d];
            ss += x * x;
        }
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(linear_value_width)) + eps);
        const at = head * linear_value_width;
        if (runner.fast_activations) {
            tf_qualcomm_silu_norm_f16_fast(z[at..].ptr, y[at..].ptr, norm.bytes.ptr, inv, gated[at..].ptr, linear_value_width);
        } else {
            for (0..linear_value_width) |d| {
                const i = at + d;
                gated[i] = roundF16(silu(z[i]) * (value(norm, d) * y[i] * inv));
            }
        }
    }
    if (runner.use_fused_tail) {
        if (output.len != gated.len) return error.UnexpectedTensor;
        @memcpy(output, gated);
    } else try runner.project(try tensorName(&projection_buffer, "{s}.out_proj", .{base}), gated, output);
}

fn linearAttentionBatch(runner: *Runner, decode: *DecodeState, layer: usize, input: []const f32, residual_state: ?[]f32, conv_commit: ?[]f32, output: []f32) !void {
    if (input.len == 0 or input.len % hidden != 0) return error.UnexpectedTensor;
    const count = input.len / hidden;
    if (output.len != count * (if (runner.use_fused_tail) linear_output_width else hidden)) return error.UnexpectedTensor;
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.linear_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;

    const qkv = try a.alloc(f32, count * conv_channels);
    defer a.free(qkv);
    const z = try a.alloc(f32, count * linear_output_width);
    defer a.free(z);
    var qkv_name_buffer: [256]u8 = undefined;
    var z_name_buffer: [256]u8 = undefined;
    const qkv_name = try tensorName(&qkv_name_buffer, "{s}.in_proj_qkv", .{base});
    const z_name = try tensorName(&z_name_buffer, "{s}.in_proj_z", .{base});
    if (runner.use_resident_weights and !runner.use_packed_exl3) {
        const pair_t0 = if (prof_on) tf_qualcomm_monotonic_ms() else 0;
        if (runner.use_q8_weights or runner.use_q4_weights) try runner.projectPairQuant(qkv_name, z_name, input, qkv, z) else try runner.projectPairResident(qkv_name, z_name, input, qkv, z);
        if (prof_on) prof_pair += tf_qualcomm_monotonic_ms() - pair_t0;
    } else {
        try runner.project(qkv_name, input, qkv);
        try runner.project(z_name, input, z);
    }
    for (qkv) |*x| x.* = roundBF16(roundF16(x.*));
    if (conv_commit) |commit| {
        if (commit.len != qkv.len) return error.UnexpectedTensor;
        @memcpy(commit, qkv);
    }

    const a_gate = try a.alloc(f32, count * linear_value_heads);
    defer a.free(a_gate);
    const b_gate = try a.alloc(f32, count * linear_value_heads);
    defer a.free(b_gate);
    const a_name = try tensorName(&projection_buffer, "{s}.in_proj_a.weight", .{base});
    var b_name_buffer: [256]u8 = undefined;
    const b_name = try tensorName(&b_name_buffer, "{s}.in_proj_b.weight", .{base});
    var a_base_buffer: [256]u8 = undefined;
    var b_base_buffer: [256]u8 = undefined;
    const a_base = try tensorName(&a_base_buffer, "{s}.in_proj_a", .{base});
    const b_base = try tensorName(&b_base_buffer, "{s}.in_proj_b", .{base});
    if (false and runner.use_resident_weights and !runner.use_packed_exl3) {
        // EXL3 trellis tensors absent for in_proj_a/b (plain F16) — resident pair unusable.
        if (runner.use_q8_weights or runner.use_q4_weights) try runner.projectPairQuant(a_base, b_base, input, a_gate, b_gate) else try runner.projectPairResident(a_base, b_base, input, a_gate, b_gate);
    } else {
        try denseBatched(runner, a_name, input, a_gate);
        try denseBatched(runner, b_name, input, b_gate);
    }

    const conv_name = try tensorName(&projection_buffer, "{s}.conv1d.weight", .{base});
    const conv = try require(runner.file, conv_name, .bf16, &.{ conv_channels, 1, 4 });
    const mixed = try a.alloc(f32, count * conv_channels);
    defer a.free(mixed);
    const history = decode.convLayer(layer);
    const conv_t0 = if (prof_on) tf_qualcomm_monotonic_ms() else 0;
    // Hoist the per-channel taps: constant across rows, bf16-decode once.
    const w0 = try a.alloc(f32, conv_channels);
    defer a.free(w0);
    const w1 = try a.alloc(f32, conv_channels);
    defer a.free(w1);
    const w2 = try a.alloc(f32, conv_channels);
    defer a.free(w2);
    const w3 = try a.alloc(f32, conv_channels);
    defer a.free(w3);
    for (0..conv_channels) |c| {
        w0[c] = value(conv, c * 4 + 0);
        w1[c] = value(conv, c * 4 + 1);
        w2[c] = value(conv, c * 4 + 2);
        w3[c] = value(conv, c * 4 + 3);
    }
    for (0..count) |position| {
        const qkv_row = qkv[position * conv_channels ..][0..conv_channels];
        const mixed_row = mixed[position * conv_channels ..][0..conv_channels];
        var c: usize = 0;
        while (c + 8 <= conv_channels) : (c += 8) {
            const w0v: @Vector(8, f32) = w0[c..][0..8].*;
            const w1v: @Vector(8, f32) = w1[c..][0..8].*;
            const w2v: @Vector(8, f32) = w2[c..][0..8].*;
            const w3v: @Vector(8, f32) = w3[c..][0..8].*;
            const hv0: @Vector(8, f32) = history[c..][0..8].*;
            const hv1: @Vector(8, f32) = history[conv_channels + c ..][0..8].*;
            const hv2: @Vector(8, f32) = history[2 * conv_channels + c ..][0..8].*;
            const qv: @Vector(8, f32) = qkv_row[c..][0..8].*;
            const sum = w3v * qv + w0v * hv0 + w1v * hv1 + w2v * hv2;
            const fields = [8]f32{ sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], sum[6], sum[7] };
            for (0..8) |lane| {
                const conv_value = roundBF16(fields[lane]);
                const conv_sigmoid = roundBF16(sigmoidBf16Lut(conv_value));
                mixed_row[c + lane] = roundBF16(conv_value * conv_sigmoid);
            }
        }
        while (c < conv_channels) : (c += 1) {
            const conv_value = roundBF16(w3[c] * qkv_row[c] + w0[c] * history[c] + w1[c] * history[conv_channels + c] + w2[c] * history[2 * conv_channels + c]);
            const conv_sigmoid = roundBF16(sigmoidBf16Lut(conv_value));
            mixed_row[c] = roundBF16(conv_value * conv_sigmoid);
        }
        std.mem.copyForwards(f32, history[0 * conv_channels .. 2 * conv_channels], history[1 * conv_channels .. 3 * conv_channels]);
        @memcpy(history[2 * conv_channels .. 3 * conv_channels], qkv_row);
    }
    if (prof_on) prof_conv += tf_qualcomm_monotonic_ms() - conv_t0;

    var dt_name: [256]u8 = undefined;
    const dt = try require(runner.file, try tensorName(&dt_name, "{s}.dt_bias", .{base}), .bf16, &.{linear_value_heads});
    var alog_name: [256]u8 = undefined;
    const alog = try require(runner.file, try tensorName(&alog_name, "{s}.A_log", .{base}), .f32, &.{linear_value_heads});
    const q = try a.alloc(f32, count * linear_query_width);
    defer a.free(q);
    const k = try a.alloc(f32, count * linear_query_width);
    defer a.free(k);
    const v = try a.alloc(f32, count * linear_output_width);
    defer a.free(v);
    const y = try a.alloc(f32, count * linear_output_width);
    defer a.free(y);
    const beta = try a.alloc(f32, count * linear_value_heads);
    defer a.free(beta);
    const log_decay = try a.alloc(f32, count * linear_value_heads);
    defer a.free(log_decay);
    const recurrent = decode.gdnLayer(layer);
    const q_scale = roundBF16(1.0 / @as(f32, @floatFromInt(linear_key_width)));
    const k_scale = roundBF16(1.0 / @sqrt(@as(f32, @floatFromInt(linear_key_width))));
    for (0..count) |position| {
        const mixed_row = mixed[position * conv_channels ..][0..conv_channels];
        const q_row = q[position * linear_query_width ..][0..linear_query_width];
        const k_row = k[position * linear_query_width ..][0..linear_query_width];
        const v_row = v[position * linear_output_width ..][0..linear_output_width];
        for (0..linear_key_heads) |head| {
            var qss: f32 = 0;
            var kss: f32 = 0;
            for (0..linear_key_width) |d| {
                const qv = mixed_row[head * linear_key_width + d];
                const kv = mixed_row[linear_query_width + head * linear_key_width + d];
                qss += qv * qv;
                kss += kv * kv;
            }
            const width: f32 = @floatFromInt(linear_key_width);
            const qinv = 1.0 / @sqrt(qss / width + eps / width);
            const kinv = 1.0 / @sqrt(kss / width + eps / width);
            for (0..linear_key_width) |d| {
                q_row[head * linear_key_width + d] = roundBF16(q_scale * roundBF16(mixed_row[head * linear_key_width + d] * qinv));
                k_row[head * linear_key_width + d] = roundBF16(k_scale * roundBF16(mixed_row[linear_query_width + head * linear_key_width + d] * kinv));
            }
        }
        @memcpy(v_row, mixed_row[2 * linear_query_width .. conv_channels]);
        for (0..linear_value_heads) |head| {
            const gate_index = position * linear_value_heads + head;
            beta[gate_index] = roundBF16(sigmoid(b_gate[gate_index]));
            const s = roundBF16(a_gate[gate_index] + value(dt, head));
            const softplus = roundBF16(@max(s, 0.0) + @log(1.0 + @exp(-@abs(s))));
            log_decay[gate_index] = -@exp(value(alog, head)) * softplus;
        }
    }
    if (runner.usesFusedLinearTail(layer)) {
        const target_state = residual_state orelse return error.UnexpectedTensor;
        try runner.linearTailQuant(layer, count, target_state, z, q, k, v, log_decay, beta, decode.gdnLayer(layer));
        return;
    }
    if (runner.use_resident_weights) {
        const gdn_t0 = if (prof_on) tf_qualcomm_monotonic_ms() else 0;
        try runner.gdnResident(layer, count, q, k, v, log_decay, beta, recurrent, y);
        if (prof_on) prof_gdn += tf_qualcomm_monotonic_ms() - gdn_t0;
    } else {
        for (0..count) |position| {
            const q_row = q[position * linear_query_width ..][0..linear_query_width];
            const k_row = k[position * linear_query_width ..][0..linear_query_width];
            const v_row = v[position * linear_output_width ..][0..linear_output_width];
            const y_row = y[position * linear_output_width ..][0..linear_output_width];
            for (0..linear_value_heads) |head| {
                const gate_index = position * linear_value_heads + head;
                const decay = @exp(log_decay[gate_index]);
                const key_head = head / (linear_value_heads / linear_key_heads);
                for (0..linear_value_width) |dv| {
                    const row = recurrent[(head * linear_value_width + dv) * linear_key_width ..][0..linear_key_width];
                    var kv_mem: f32 = 0;
                    for (0..linear_key_width) |dk| {
                        row[dk] *= decay;
                        kv_mem += row[dk] * k_row[key_head * linear_key_width + dk];
                    }
                    const delta = (v_row[head * linear_value_width + dv] - kv_mem) * beta[gate_index];
                    var result: f32 = 0;
                    for (0..linear_key_width) |dk| {
                        row[dk] += k_row[key_head * linear_key_width + dk] * delta;
                        result += row[dk] * q_row[key_head * linear_key_width + dk];
                    }
                    y_row[head * linear_value_width + dv] = roundBF16(result);
                }
            }
        }
    }

    var norm_name: [256]u8 = undefined;
    const norm = try require(runner.file, try tensorName(&norm_name, "{s}.norm.weight", .{base}), .f32, &.{linear_value_width});
    const gated = try a.alloc(f32, count * linear_output_width);
    defer a.free(gated);
    for (0..count) |position| {
        const y_row = y[position * linear_output_width ..][0..linear_output_width];
        const z_row = z[position * linear_output_width ..][0..linear_output_width];
        const gated_row = gated[position * linear_output_width ..][0..linear_output_width];
        for (0..linear_value_heads) |head| {
            var ss: f32 = 0;
            for (0..linear_value_width) |d| {
                const x = y_row[head * linear_value_width + d];
                ss += x * x;
            }
            const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(linear_value_width)) + eps);
            const at = head * linear_value_width;
            if (runner.fast_activations) {
                tf_qualcomm_silu_norm_f16_fast(z_row[at..].ptr, y_row[at..].ptr, norm.bytes.ptr, inv, gated_row[at..].ptr, linear_value_width);
            } else {
                for (0..linear_value_width) |d| {
                    const i = at + d;
                    gated_row[i] = roundF16(silu(z_row[i]) * (value(norm, d) * y_row[i] * inv));
                }
            }
        }
    }
    if (runner.use_fused_tail) {
        @memcpy(output, gated);
    } else try runner.project(try tensorName(&projection_buffer, "{s}.out_proj", .{base}), gated, output);
}

fn headNorm(file: *const st.Checkpoint, name: []const u8, input: []const f32, heads: usize, output: []f32) !void {
    const tensor = file.lookup(name) orelse return error.MissingTensor;
    const width = input.len / heads;
    if (input.len != output.len or width != attention_width or tensor.dtype != .bf16 or !tensor.is(.bf16, &.{width})) return error.UnexpectedTensor;
    for (0..heads) |head| {
        var ss: f32 = 0;
        for (0..width) |d| {
            const x = input[head * width + d];
            ss += x * x;
        }
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(width)) + eps);
        for (0..width) |d| output[head * width + d] = roundF16(input[head * width + d] * inv * (value(tensor, d) + 1.0));
    }
}

fn rotateHalf(input: []const f32, output: []f32, position: usize) void {
    @memcpy(output, input);
    ropeTables(position);
    const half: usize = rope_width / 2;
    for (0..input.len / attention_width) |head| {
        const base = head * attention_width;
        for (0..half) |i| {
            const c = rope_cos[position * half + i];
            const s = rope_sin[position * half + i];
            const av = input[base + i];
            const bv = input[base + half + i];
            output[base + i] = roundF16(av * c - bv * s);
            output[base + half + i] = roundF16(bv * c + av * s);
        }
    }
}

fn fullAttention(runner: *Runner, decode: *DecodeState, layer: usize, position: usize, input: []const f32, output: []f32) !void {
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.self_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;
    const qraw = try a.alloc(f32, 2 * attention_query_width);
    defer a.free(qraw);
    const kraw = try a.alloc(f32, kv_width);
    defer a.free(kraw);
    const vraw = try a.alloc(f32, kv_width);
    defer a.free(vraw);
    var q_name_buffer: [256]u8 = undefined;
    var k_name_buffer: [256]u8 = undefined;
    var v_name_buffer: [256]u8 = undefined;
    const q_name = try tensorName(&q_name_buffer, "{s}.q_proj", .{base});
    const k_name = try tensorName(&k_name_buffer, "{s}.k_proj", .{base});
    const v_name = try tensorName(&v_name_buffer, "{s}.v_proj", .{base});
    if (runner.use_resident_weights and !runner.use_packed_exl3) {
        if (runner.use_q8_weights or runner.use_q4_weights) try runner.projectTripleQuant(q_name, k_name, v_name, input, qraw, kraw, vraw) else try runner.projectTripleResident(q_name, k_name, v_name, input, qraw, kraw, vraw);
    } else {
        try runner.project(q_name, input, qraw);
        try runner.project(k_name, input, kraw);
        try runner.project(v_name, input, vraw);
    }
    for (qraw) |*x| x.* = roundF16(x.*);
    for (kraw) |*x| x.* = roundF16(x.*);
    for (vraw) |*x| x.* = roundF16(x.*);

    const q = try a.alloc(f32, attention_query_width);
    defer a.free(q);
    const gate = try a.alloc(f32, attention_query_width);
    defer a.free(gate);
    for (0..attention_heads) |head| for (0..attention_width) |d| {
        q[head * attention_width + d] = qraw[head * 2 * attention_width + d];
        gate[head * attention_width + d] = qraw[head * 2 * attention_width + attention_width + d];
    };
    const qnorm = try a.alloc(f32, attention_query_width);
    defer a.free(qnorm);
    const knorm = try a.alloc(f32, kv_width);
    defer a.free(knorm);
    var norm_buffer: [256]u8 = undefined;
    try headNorm(runner.file, try tensorName(&norm_buffer, "{s}.q_norm.weight", .{base}), q, attention_heads, qnorm);
    try headNorm(runner.file, try tensorName(&norm_buffer, "{s}.k_norm.weight", .{base}), kraw, kv_heads, knorm);
    const qrot = try a.alloc(f32, attention_query_width);
    defer a.free(qrot);
    const krot = try a.alloc(f32, kv_width);
    defer a.free(krot);
    rotateHalf(qnorm, qrot, position);
    rotateHalf(knorm, krot, position);
    @memcpy(decode.keyPosition(layer, position), krot);
    @memcpy(decode.valuePosition(layer, position), vraw);

    const attended = try a.alloc(f32, attention_query_width);
    defer a.free(attended);
    const scores = try a.alloc(f32, position + 1);
    defer a.free(scores);
    for (0..attention_heads) |query_head| {
        const kv_head = query_head / (attention_heads / kv_heads);
        var maximum = -std.math.inf(f32);
        for (0..position + 1) |token_position| {
            const cached_key = decode.keyPosition(layer, token_position);
            var score: f32 = 0;
            for (0..attention_width) |d| score += qrot[query_head * attention_width + d] * cached_key[kv_head * attention_width + d];
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
                const cached_value = decode.valuePosition(layer, token_position);
                sum += scores[token_position] / denominator * cached_value[kv_head * attention_width + d];
            }
            const i = query_head * attention_width + d;
            attended[i] = roundF16(roundF16(sum) * sigmoid(gate[i]));
        }
    }
    if (runner.use_fused_tail) {
        if (output.len != attended.len) return error.UnexpectedTensor;
        @memcpy(output, attended);
    } else try runner.project(try tensorName(&projection_buffer, "{s}.o_proj", .{base}), attended, output);
}

fn fullAttentionBatch(runner: *Runner, decode: *DecodeState, layer: usize, start_position: usize, input: []const f32, output: []f32) !void {
    if (input.len == 0 or input.len % hidden != 0) return error.UnexpectedTensor;
    const count = input.len / hidden;
    if (output.len != count * (if (runner.use_fused_tail) attention_query_width else hidden) or start_position + count > decode.max_positions) return error.UnexpectedTensor;
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.self_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;
    const qraw = try a.alloc(f32, count * 2 * attention_query_width);
    defer a.free(qraw);
    const kraw = try a.alloc(f32, count * kv_width);
    defer a.free(kraw);
    const vraw = try a.alloc(f32, count * kv_width);
    defer a.free(vraw);
    var q_name_buffer: [256]u8 = undefined;
    var k_name_buffer: [256]u8 = undefined;
    var v_name_buffer: [256]u8 = undefined;
    const q_name = try tensorName(&q_name_buffer, "{s}.q_proj", .{base});
    const k_name = try tensorName(&k_name_buffer, "{s}.k_proj", .{base});
    const v_name = try tensorName(&v_name_buffer, "{s}.v_proj", .{base});
    if (runner.use_resident_weights and !runner.use_packed_exl3) {
        if (runner.use_q8_weights or runner.use_q4_weights) try runner.projectTripleQuant(q_name, k_name, v_name, input, qraw, kraw, vraw) else try runner.projectTripleResident(q_name, k_name, v_name, input, qraw, kraw, vraw);
    } else {
        try runner.project(q_name, input, qraw);
        try runner.project(k_name, input, kraw);
        try runner.project(v_name, input, vraw);
    }
    for (qraw) |*x| x.* = roundF16(x.*);
    for (kraw) |*x| x.* = roundF16(x.*);
    for (vraw) |*x| x.* = roundF16(x.*);

    const q = try a.alloc(f32, count * attention_query_width);
    defer a.free(q);
    const gate = try a.alloc(f32, count * attention_query_width);
    defer a.free(gate);
    const qnorm = try a.alloc(f32, count * attention_query_width);
    defer a.free(qnorm);
    const knorm = try a.alloc(f32, count * kv_width);
    defer a.free(knorm);
    const qrot = try a.alloc(f32, count * attention_query_width);
    defer a.free(qrot);
    const krot = try a.alloc(f32, count * kv_width);
    defer a.free(krot);
    var q_norm_name_buffer: [256]u8 = undefined;
    const q_norm_name = try tensorName(&q_norm_name_buffer, "{s}.q_norm.weight", .{base});
    var k_norm_name_buffer: [256]u8 = undefined;
    const k_norm_name = try tensorName(&k_norm_name_buffer, "{s}.k_norm.weight", .{base});
    for (0..count) |row| {
        const qraw_row = qraw[row * 2 * attention_query_width ..][0 .. 2 * attention_query_width];
        const q_row = q[row * attention_query_width ..][0..attention_query_width];
        const gate_row = gate[row * attention_query_width ..][0..attention_query_width];
        for (0..attention_heads) |head| for (0..attention_width) |d| {
            q_row[head * attention_width + d] = qraw_row[head * 2 * attention_width + d];
            gate_row[head * attention_width + d] = qraw_row[head * 2 * attention_width + attention_width + d];
        };
        const qnorm_row = qnorm[row * attention_query_width ..][0..attention_query_width];
        const knorm_row = knorm[row * kv_width ..][0..kv_width];
        try headNorm(runner.file, q_norm_name, q_row, attention_heads, qnorm_row);
        try headNorm(runner.file, k_norm_name, kraw[row * kv_width ..][0..kv_width], kv_heads, knorm_row);
        const position = start_position + row;
        const qrot_row = qrot[row * attention_query_width ..][0..attention_query_width];
        const krot_row = krot[row * kv_width ..][0..kv_width];
        rotateHalf(qnorm_row, qrot_row, position);
        rotateHalf(knorm_row, krot_row, position);
        @memcpy(decode.keyPosition(layer, position), krot_row);
        @memcpy(decode.valuePosition(layer, position), vraw[row * kv_width ..][0..kv_width]);
    }

    const attended = try a.alloc(f32, count * attention_query_width);
    defer a.free(attended);
    const scores = try a.alloc(f32, start_position + count);
    defer a.free(scores);
    const Vec = @Vector(8, f32);
    for (0..count) |row| {
        const position = start_position + row;
        const qrot_row = qrot[row * attention_query_width ..][0..attention_query_width];
        const gate_row = gate[row * attention_query_width ..][0..attention_query_width];
        const attended_row = attended[row * attention_query_width ..][0..attention_query_width];
        for (0..attention_heads) |query_head| {
            const kv_head = query_head / (attention_heads / kv_heads);
            var maximum = -std.math.inf(f32);
            const q_base = query_head * attention_width;
            for (0..position + 1) |token_position| {
                const cached_key = decode.keyPosition(layer, token_position);
                const k_base = kv_head * attention_width;
                var acc: Vec = @splat(0.0);
                var d: usize = 0;
                while (d + 8 <= attention_width) : (d += 8) {
                    const qv: Vec = qrot_row[q_base + d ..][0..8].*;
                    const kv: Vec = cached_key[k_base + d ..][0..8].*;
                    acc += qv * kv;
                }
                var score = @reduce(.Add, acc);
                while (d < attention_width) : (d += 1) score += qrot_row[q_base + d] * cached_key[k_base + d];
                scores[token_position] = score / 16.0;
                maximum = @max(maximum, scores[token_position]);
            }
            var denominator: f32 = 0;
            for (scores[0 .. position + 1]) |*score| {
                score.* = @exp(score.* - maximum);
                denominator += score.*;
            }
            var d2: usize = 0;
            const acc_chunks = attention_width / 8;
            const acc_vec = try a.alloc(Vec, acc_chunks);
            defer a.free(acc_vec);
            for (0..acc_chunks) |ci| acc_vec[ci] = @splat(0.0);
            for (0..position + 1) |token_position| {
                const cached_value = decode.valuePosition(layer, token_position);
                const v_base = kv_head * attention_width;
                const w = scores[token_position] / denominator;
                const wv: Vec = @splat(w);
                for (0..acc_chunks) |ci| {
                    const vv: Vec = cached_value[v_base + ci * 8 ..][0..8].*;
                    acc_vec[ci] += wv * vv;
                }
                d2 = acc_chunks * 8;
            }
            for (0..acc_chunks) |ci| {
                const out = acc_vec[ci];
                const base_i = query_head * attention_width + ci * 8;
                const fields = [8]f32{ out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7] };
                for (0..8) |lane| {
                    const i = base_i + lane;
                    attended_row[i] = roundF16(roundF16(fields[lane]) * sigmoidF16Lut(gate_row[i]));
                }
            }
            while (d2 < attention_width) : (d2 += 1) {
                var sum: f32 = 0;
                for (0..position + 1) |token_position| {
                    const cached_value = decode.valuePosition(layer, token_position);
                    sum += scores[token_position] / denominator * cached_value[kv_head * attention_width + d2];
                }
                const i = query_head * attention_width + d2;
                attended_row[i] = roundF16(roundF16(sum) * sigmoidF16Lut(gate_row[i]));
            }
        }
    }
    if (runner.use_fused_tail) {
        @memcpy(output, attended);
    } else try runner.project(try tensorName(&projection_buffer, "{s}.o_proj", .{base}), attended, output);
}

fn denseVector(file: *const st.Checkpoint, name: []const u8, input: []const f32, output: []f32) !void {
    const tensor = file.lookup(name) orelse return error.MissingTensor;
    if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 2 or tensor.dim(0) != output.len or tensor.dim(1) != input.len) return error.UnexpectedTensor;
    for (output, 0..) |*out, row| {
        var sum: f32 = 0;
        for (input, 0..) |x, col| sum += x * value(tensor, row * input.len + col);
        out.* = sum;
    }
}

fn moeMlpBatch(runner: *Runner, layer: usize, input: []const f32, output: []f32) !void {
    if (comptime experts == 0 or top_k == 0 or shared_intermediate == 0) return error.NotMoeProfile;
    if (input.len == 0 or input.len % hidden != 0 or output.len != input.len) return error.UnexpectedTensor;
    const count = input.len / hidden;
    const allocator = runner.allocator;
    @memset(output, 0);

    var router_name_buf: [256]u8 = undefined;
    var shared_gate_name_buf: [256]u8 = undefined;
    const router_name = try tensorName(&router_name_buf, "{s}.layers.{d}.mlp.gate.weight", .{ prefix, layer });
    const shared_gate_name = try tensorName(&shared_gate_name_buf, "{s}.layers.{d}.mlp.shared_expert_gate.weight", .{ prefix, layer });
    const router_logits = try allocator.alloc(f32, count * experts);
    for (0..count) |row| try denseVector(runner.file, router_name, input[row * hidden ..][0..hidden], router_logits[row * experts ..][0..experts]);
    for (router_logits) |*logit| logit.* = quantizeF16(logit.*);
    const route_ids = try allocator.alloc(usize, count * top_k);
    const route_weights = try allocator.alloc(f32, count * top_k);
    for (0..count) |row| {
        const logits = router_logits[row * experts ..][0..experts];
        const ids = route_ids[row * top_k ..][0..top_k];
        const weights = route_weights[row * top_k ..][0..top_k];
        @memset(ids, 0);
        @memset(weights, -std.math.inf(f32));
        for (logits, 0..) |logit, expert| {
            if (logit < weights[top_k - 1]) continue;
            weights[top_k - 1] = logit;
            ids[top_k - 1] = expert;
            var at = top_k - 1;
            while (at > 0 and (weights[at] > weights[at - 1] or
                (weights[at] == weights[at - 1] and ids[at] < ids[at - 1]))) : (at -= 1)
            {
                std.mem.swap(f32, &weights[at], &weights[at - 1]);
                std.mem.swap(usize, &ids[at], &ids[at - 1]);
            }
        }
        try moe_routing.normalizeTopK(weights);
    }

    // Group all rows routed to the same expert. A packed tensor is paged and
    // decoded once per expert per layer instead of once per token. Contributions
    // stay in route slots so the final accumulation order remains identical to
    // the scalar reference.
    const contributions = try allocator.alloc(f32, count * top_k * hidden);
    @memset(contributions, 0);
    const expert_rows = try allocator.alloc(usize, count);
    const expert_routes = try allocator.alloc(usize, count);
    const expert_input = try allocator.alloc(f32, count * hidden);
    const expert_gate = try allocator.alloc(f32, count * intermediate);
    const expert_up = try allocator.alloc(f32, count * intermediate);
    const expert_activated = try allocator.alloc(f32, count * intermediate);
    const expert_down = try allocator.alloc(f32, count * hidden);
    for (0..experts) |expert| {
        var routed_count: usize = 0;
        for (0..count) |row| for (0..top_k) |route| {
            if (route_ids[row * top_k + route] != expert) continue;
            expert_rows[routed_count] = row;
            expert_routes[routed_count] = route;
            @memcpy(expert_input[routed_count * hidden ..][0..hidden], input[row * hidden ..][0..hidden]);
            routed_count += 1;
        };
        if (routed_count == 0) continue;

        var gate_name_buf: [256]u8 = undefined;
        var up_name_buf: [256]u8 = undefined;
        var down_name_buf: [256]u8 = undefined;
        const gate_name = try tensorName(&gate_name_buf, "{s}.layers.{d}.mlp.experts.{d}.gate_proj", .{ prefix, layer, expert });
        const up_name = try tensorName(&up_name_buf, "{s}.layers.{d}.mlp.experts.{d}.up_proj", .{ prefix, layer, expert });
        const down_name = try tensorName(&down_name_buf, "{s}.layers.{d}.mlp.experts.{d}.down_proj", .{ prefix, layer, expert });
        try runner.projectPaged(gate_name, expert_input[0 .. routed_count * hidden], expert_gate[0 .. routed_count * intermediate]);
        try runner.projectPaged(up_name, expert_input[0 .. routed_count * hidden], expert_up[0 .. routed_count * intermediate]);
        for (expert_activated[0 .. routed_count * intermediate], 0..) |*out, i| {
            out.* = quantizeF16(silu(quantizeF16(expert_gate[i])) * quantizeF16(expert_up[i]));
        }
        try runner.projectPaged(down_name, expert_activated[0 .. routed_count * intermediate], expert_down[0 .. routed_count * hidden]);
        for (0..routed_count) |assignment| {
            const row = expert_rows[assignment];
            const route = expert_routes[assignment];
            const destination = contributions[(row * top_k + route) * hidden ..][0..hidden];
            const source = expert_down[assignment * hidden ..][0..hidden];
            const weight = route_weights[row * top_k + route];
            for (destination, source) |*out, value_| out.* = weight * value_;
        }
    }

    var shared_gate_proj_buf: [256]u8 = undefined;
    var shared_up_proj_buf: [256]u8 = undefined;
    var shared_down_proj_buf: [256]u8 = undefined;
    const shared_gate_proj = try tensorName(&shared_gate_proj_buf, "{s}.layers.{d}.mlp.shared_expert.gate_proj", .{ prefix, layer });
    const shared_up_proj = try tensorName(&shared_up_proj_buf, "{s}.layers.{d}.mlp.shared_expert.up_proj", .{ prefix, layer });
    const shared_down_proj = try tensorName(&shared_down_proj_buf, "{s}.layers.{d}.mlp.shared_expert.down_proj", .{ prefix, layer });
    const shared_gate = try allocator.alloc(f32, count * shared_intermediate);
    const shared_up = try allocator.alloc(f32, count * shared_intermediate);
    const shared_activated = try allocator.alloc(f32, count * shared_intermediate);
    const shared_down = try allocator.alloc(f32, count * hidden);
    const shared_scale = try allocator.alloc(f32, count);
    try runner.projectPaged(shared_gate_proj, input, shared_gate);
    try runner.projectPaged(shared_up_proj, input, shared_up);
    for (shared_activated, 0..) |*out, i| {
        out.* = quantizeF16(silu(quantizeF16(shared_gate[i])) * quantizeF16(shared_up[i]));
    }
    try runner.projectPaged(shared_down_proj, shared_activated, shared_down);
    for (0..count) |row| try denseVector(runner.file, shared_gate_name, input[row * hidden ..][0..hidden], shared_scale[row..][0..1]);
    for (0..count) |row| {
        const y = output[row * hidden ..][0..hidden];
        for (0..top_k) |route| {
            const contribution = contributions[(row * top_k + route) * hidden ..][0..hidden];
            for (y, contribution) |*out, value_| out.* += value_;
        }
        const scale = sigmoid(shared_scale[row]);
        const shared = shared_down[row * hidden ..][0..hidden];
        for (y, shared) |*out, value_| out.* += scale * value_;
    }
}

fn mlp(runner: *Runner, layer: usize, input: []const f32, output: []f32) !void {
    if (comptime experts > 0) return moeMlpBatch(runner, layer, input, output);
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.mlp", .{ prefix, layer });
    var gate_name_buffer: [256]u8 = undefined;
    var up_name_buffer: [256]u8 = undefined;
    var down_name_buffer: [256]u8 = undefined;
    const gate_name = try tensorName(&gate_name_buffer, "{s}.gate_proj", .{base});
    const up_name = try tensorName(&up_name_buffer, "{s}.up_proj", .{base});
    const down_name = try tensorName(&down_name_buffer, "{s}.down_proj", .{base});
    if (runner.use_resident_weights and !runner.use_packed_exl3) {
        if (runner.use_q8_weights or runner.use_q4_weights) return runner.mlpQuant(gate_name, up_name, down_name, input, output);
        return runner.mlpResident(gate_name, up_name, down_name, input, output);
    }
    const gate = try a.alloc(f32, intermediate);
    defer a.free(gate);
    const up = try a.alloc(f32, intermediate);
    defer a.free(up);
    const activated = try a.alloc(f32, intermediate);
    defer a.free(activated);
    try runner.project(gate_name, input, gate);
    try runner.project(up_name, input, up);
    for (activated, 0..) |*out, i| out.* = roundF16(silu(roundF16(gate[i])) * roundF16(up[i]));
    try runner.project(down_name, activated, output);
}

fn mlpBatch(runner: *Runner, layer: usize, input: []const f32, output: []f32) !void {
    if (comptime experts > 0) return moeMlpBatch(runner, layer, input, output);
    if (input.len == 0 or input.len % hidden != 0) return error.UnexpectedTensor;
    const count = input.len / hidden;
    if (output.len != count * hidden) return error.UnexpectedTensor;
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.mlp", .{ prefix, layer });
    var gate_name_buffer: [256]u8 = undefined;
    var up_name_buffer: [256]u8 = undefined;
    var down_name_buffer: [256]u8 = undefined;
    const gate_name = try tensorName(&gate_name_buffer, "{s}.gate_proj", .{base});
    const up_name = try tensorName(&up_name_buffer, "{s}.up_proj", .{base});
    const down_name = try tensorName(&down_name_buffer, "{s}.down_proj", .{base});
    if (runner.use_resident_weights and !runner.use_packed_exl3) {
        if (runner.use_q8_weights or runner.use_q4_weights) return runner.mlpQuant(gate_name, up_name, down_name, input, output);
        return runner.mlpResident(gate_name, up_name, down_name, input, output);
    }
    const gate = try a.alloc(f32, count * intermediate);
    defer a.free(gate);
    const up = try a.alloc(f32, count * intermediate);
    defer a.free(up);
    const activated = try a.alloc(f32, count * intermediate);
    defer a.free(activated);
    try runner.project(gate_name, input, gate);
    try runner.project(up_name, input, up);
    for (activated, 0..) |*out, i| out.* = roundF16(silu(roundF16(gate[i])) * roundF16(up[i]));
    try runner.project(down_name, activated, output);
}

const Top = struct { id: usize, logit: f32 };

fn topFive(all_logits: []const f32) [5]Top {
    var top: [5]Top = @splat(.{ .id = 0, .logit = -std.math.inf(f32) });
    for (all_logits, 0..) |logit, token| {
        if (logit > top[4].logit) {
            top[4] = .{ .id = token, .logit = logit };
            var i: usize = 4;
            while (i > 0 and top[i].logit > top[i - 1].logit) : (i -= 1) std.mem.swap(Top, &top[i], &top[i - 1]);
        }
    }
    return top;
}

fn topOne(all_logits: []const f32) Top {
    var best: Top = .{ .id = 0, .logit = -std.math.inf(f32) };
    for (all_logits, 0..) |logit, token| if (logit > best.logit) {
        best = .{ .id = token, .logit = logit };
    };
    return best;
}

fn topTwo(all_logits: []const f32) [2]Top {
    var top: [2]Top = @splat(.{ .id = 0, .logit = -std.math.inf(f32) });
    for (all_logits, 0..) |logit, token| {
        if (logit > top[1].logit) {
            top[1] = .{ .id = token, .logit = logit };
            if (top[1].logit > top[0].logit) std.mem.swap(Top, &top[0], &top[1]);
        }
    }
    return top;
}

fn refineHeadCandidates(embedding: [*]const u16, state: []const f32, candidates: [5]Top) [5]Top {
    var refined: [5]Top = @splat(.{ .id = 0, .logit = -std.math.inf(f32) });
    for (candidates) |candidate| {
        var logit: f32 = 0;
        tf_qualcomm_dense_f16(embedding + candidate.id * hidden, state.ptr, @ptrCast(&logit), 1, hidden);
        refined[4] = .{ .id = candidate.id, .logit = logit };
        var i: usize = 4;
        while (i > 0 and refined[i].logit > refined[i - 1].logit) : (i -= 1) std.mem.swap(Top, &refined[i], &refined[i - 1]);
    }
    return refined;
}

fn tiedHeadScalar(file: *const st.Checkpoint, state: []const f32) ![5]Top {
    const embedding = try require(file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
    var top: [5]Top = @splat(.{ .id = 0, .logit = -std.math.inf(f32) });
    for (0..vocab) |token| {
        var logit: f32 = 0;
        for (0..hidden) |d| logit += f16Value(embedding.bytes, token * hidden + d) * state[d];
        if (logit > top[4].logit) {
            top[4] = .{ .id = token, .logit = logit };
            var i: usize = 4;
            while (i > 0 and top[i].logit > top[i - 1].logit) : (i -= 1) std.mem.swap(Top, &top[i], &top[i - 1]);
        }
    }
    return top;
}

fn tiedHead(runner: *Runner, state: []const f32, cpu_threads: i32) ![5]Top {
    if (comptime !profile.tied_embeddings) {
        const logits = try runner.allocator.alloc(f32, vocab);
        defer runner.allocator.free(logits);
        const backend_was_cpu = runner.backend_is_cpu;
        const packed_was_enabled = runner.use_packed_exl3;
        const f16_was_enabled = runner.use_f16_weights;
        const resident_was_enabled = runner.use_resident_weights;
        if (comptime experts > 0) {
            runner.backend_is_cpu = true;
            if (runner.use_f16_head_cache) {
                runner.use_packed_exl3 = false;
                runner.use_f16_weights = true;
                runner.use_resident_weights = false;
            }
        }
        defer {
            runner.backend_is_cpu = backend_was_cpu;
            runner.use_packed_exl3 = packed_was_enabled;
            runner.use_f16_weights = f16_was_enabled;
            runner.use_resident_weights = resident_was_enabled;
        }
        try runner.project("lm_head", state, logits);
        return topFive(logits);
    }
    const embedding = try require(runner.file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
    const embedding_words: [*]const u16 = @ptrCast(@alignCast(embedding.bytes.ptr));
    if (runner.use_resident_head) {
        const all_logits = try runner.allocator.alloc(f32, vocab);
        defer runner.allocator.free(all_logits);
        if (runner.use_adaptive_head) {
            var compute_ms: f64 = 0;
            var uploaded: c_int = 0;
            const q4_weight: ?[*]const u16 = if (runner.resident_head_q4_loaded) null else embedding_words;
            const q4_status = tf_qualcomm_backend_matmul_q4_f16_source_resident_once(runner.backend, projections_per_position, hidden, vocab, 1, q4_weight, state.ptr, all_logits.ptr, &compute_ms, &uploaded);
            if (q4_status == 0) {
                runner.resident_head_compute_ms += compute_ms;
                if (uploaded != 0) {
                    runner.resident_head_q4_loaded = true;
                    runner.resident_head_uploads += 1;
                } else runner.resident_head_hits += 1;
                const q4_top = topFive(all_logits);
                if (q4_top[0].logit - q4_top[1].logit >= 0.5) return q4_top;
                if (runner.use_refined_head) return refineHeadCandidates(embedding_words, state, q4_top);

                compute_ms = 0;
                uploaded = 0;
                const q8_weight: ?[*]const u16 = if (runner.resident_head_q8_loaded) null else embedding_words;
                const q8_status = tf_qualcomm_backend_matmul_q8_f16_source_resident_once(runner.backend, projections_per_position + 1, hidden, vocab, 1, q8_weight, state.ptr, all_logits.ptr, &compute_ms, &uploaded);
                if (q8_status == 0) {
                    runner.resident_head_compute_ms += compute_ms;
                    if (uploaded != 0) {
                        runner.resident_head_q8_loaded = true;
                        runner.resident_head_uploads += 1;
                    } else runner.resident_head_hits += 1;
                    return topFive(all_logits);
                }
                runner.resident_head_q8_loaded = false;
            } else runner.resident_head_q4_loaded = false;
            runner.resident_head_fallbacks += 1;
        }
        var compute_ms: f64 = 0;
        var uploaded: c_int = 0;
        const weight_ptr: ?[*]const u16 = if (runner.resident_head_loaded) null else embedding_words;
        const status = if (runner.use_q4_head)
            tf_qualcomm_backend_matmul_q4_f16_source_resident_once(runner.backend, projections_per_position, hidden, vocab, 1, weight_ptr, state.ptr, all_logits.ptr, &compute_ms, &uploaded)
        else if (runner.use_q8_head)
            tf_qualcomm_backend_matmul_q8_f16_source_resident_once(runner.backend, projections_per_position, hidden, vocab, 1, weight_ptr, state.ptr, all_logits.ptr, &compute_ms, &uploaded)
        else
            tf_qualcomm_backend_matmul_f16_resident_once(runner.backend, projections_per_position, hidden, vocab, 1, weight_ptr, state.ptr, all_logits.ptr, &compute_ms, &uploaded);
        if (status == 0) {
            runner.resident_head_compute_ms += compute_ms;
            if (uploaded != 0) {
                runner.resident_head_loaded = true;
                runner.resident_head_uploads += 1;
            } else {
                runner.resident_head_hits += 1;
            }
            return topFive(all_logits);
        }
        runner.resident_head_loaded = false;
        runner.resident_head_fallbacks += 1;
    }
    var ids: [5]usize = undefined;
    var logits: [5]f32 = undefined;
    if (tf_qualcomm_tied_head_f16(embedding_words, state.ptr, vocab, hidden, cpu_threads, &ids, &logits) != 0) {
        return tiedHeadScalar(runner.file, state);
    }
    var top: [5]Top = undefined;
    for (&top, ids, logits) |*candidate, id, logit| candidate.* = .{ .id = id, .logit = logit };
    return top;
}

/// Draft tokens need only an argmax: target verification preserves exact
/// output semantics even when this lossy Q4 proposal differs. Avoid the
/// top-five insertion scan and candidate refinement used for target tokens.
fn tiedHeadDraft(runner: *Runner, state: []const f32, cpu_threads: i32) ![2]Top {
    if (runner.use_resident_head) {
        const embedding = try require(runner.file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
        const embedding_words: [*]const u16 = @ptrCast(@alignCast(embedding.bytes.ptr));
        const all_logits = try runner.allocator.alloc(f32, vocab);
        defer runner.allocator.free(all_logits);
        var compute_ms: f64 = 0;
        var uploaded: c_int = 0;
        const weight_ptr: ?[*]const u16 = if (runner.resident_head_q4_loaded) null else embedding_words;
        const status = tf_qualcomm_backend_matmul_q4_f16_source_resident_once(
            runner.backend,
            projections_per_position,
            hidden,
            vocab,
            1,
            weight_ptr,
            state.ptr,
            all_logits.ptr,
            &compute_ms,
            &uploaded,
        );
        if (status == 0) {
            runner.resident_head_compute_ms += compute_ms;
            if (uploaded != 0) {
                runner.resident_head_q4_loaded = true;
                runner.resident_head_uploads += 1;
            } else runner.resident_head_hits += 1;
            return topTwo(all_logits);
        }
        runner.resident_head_q4_loaded = false;
        runner.resident_head_fallbacks += 1;
    }
    const fallback = try tiedHead(runner, state, cpu_threads);
    return .{ fallback[0], fallback[1] };
}

fn tiedHeadBatch(runner: *Runner, states: []const f32, cpu_threads: i32) ![][5]Top {
    if (states.len == 0 or states.len % hidden != 0) return error.UnexpectedTensor;
    const count = states.len / hidden;
    const tops = try runner.allocator.alloc([5]Top, count);
    const embedding = try require(runner.file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
    const embedding_words: [*]const u16 = @ptrCast(@alignCast(embedding.bytes.ptr));
    const all_logits = try runner.allocator.alloc(f32, count * vocab);
    var compute_ms: f64 = 0;
    var uploaded: c_int = 0;
    const weight_ptr: ?[*]const u16 = if (runner.resident_head_q4_loaded) null else embedding_words;
    const status = tf_qualcomm_backend_matmul_q4_f16_source_resident_once(
        runner.backend,
        projections_per_position,
        hidden,
        vocab,
        count,
        weight_ptr,
        states.ptr,
        all_logits.ptr,
        &compute_ms,
        &uploaded,
    );
    if (status == 0) {
        runner.resident_head_compute_ms += compute_ms;
        if (uploaded != 0) {
            runner.resident_head_q4_loaded = true;
            runner.resident_head_uploads += 1;
        } else runner.resident_head_hits += 1;
        for (tops, 0..) |*top, row| {
            const q4_top = topFive(all_logits[row * vocab ..][0..vocab]);
            top.* = if (q4_top[0].logit - q4_top[1].logit >= 0.5)
                q4_top
            else
                refineHeadCandidates(embedding_words, states[row * hidden ..][0..hidden], q4_top);
        }
        return tops;
    }
    runner.resident_head_q4_loaded = false;
    runner.resident_head_fallbacks += 1;
    for (tops, 0..) |*top, row| top.* = try tiedHead(runner, states[row * hidden ..][0..hidden], cpu_threads);
    return tops;
}

fn targetStep(
    runner: *Runner,
    decode: *DecodeState,
    scratch_arena: *std.heap.ArenaAllocator,
    embedding: st.Tensor,
    input_token: usize,
    position: usize,
    state: []f32,
    normed: []f32,
    update: []f32,
    cpu_threads: i32,
) ![5]Top {
    for (state, 0..) |*x, d| x.* = f16Value(embedding.bytes, input_token * hidden + d);
    runner.resident_slot_cursor = 0;
    for (0..layers) |layer| {
        _ = scratch_arena.reset(.retain_capacity);
        var input_norm_buffer: [256]u8 = undefined;
        const input_norm = try tensorName(&input_norm_buffer, "{s}.layers.{d}.input_layernorm.weight", .{ prefix, layer });
        try rmsNorm(runner.file, input_norm, state, normed, 1.0);
        if ((layer + 1) % 4 == 0) {
            try fullAttention(runner, decode, layer, position, normed, update);
        } else {
            try linearAttention(runner, decode, layer, normed, if (runner.usesFusedLinearTail(layer)) state else null, update);
        }
        if (runner.usesFusedLinearTail(layer) and (layer + 1) % 4 != 0) {
            // Completed by linearAttention's fused persistent HTP graph.
        } else if (runner.use_fused_tail) {
            var post_norm_buffer: [256]u8 = undefined;
            var output_name_buffer: [256]u8 = undefined;
            const post_norm = try tensorName(&post_norm_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
            const output_name = if ((layer + 1) % 4 == 0)
                try tensorName(&output_name_buffer, "{s}.layers.{d}.self_attn.o_proj", .{ prefix, layer })
            else
                try tensorName(&output_name_buffer, "{s}.layers.{d}.linear_attn.out_proj", .{ prefix, layer });
            try runner.layerTailQuant(layer, output_name, post_norm, state, update);
        } else {
            residualAdd(state, update);
            var post_norm_buffer: [256]u8 = undefined;
            const post_norm = try tensorName(&post_norm_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
            try rmsNorm(runner.file, post_norm, state, normed, 1.0);
            try mlp(runner, layer, normed, update);
            residualAdd(state, update);
        }
    }
    try rmsNorm(runner.file, prefix ++ ".norm.weight", state, normed, 1.0);
    return tiedHead(runner, normed, cpu_threads);
}

fn parseTokenList(allocator: std.mem.Allocator, encoded: []const u8, maximum: usize) ![]usize {
    const count = std.mem.count(u8, encoded, ",") + 1;
    if (count == 0 or count > maximum) return error.InvalidArgument;
    const tokens = try allocator.alloc(usize, count);
    errdefer allocator.free(tokens);
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

const WorkerCommand = struct {
    id: u64,
    steps: usize,
    tokens: []usize,
};

fn parseWorkerCommand(allocator: std.mem.Allocator, line: []const u8, max_steps: usize) !WorkerCommand {
    var fields = std.mem.splitScalar(u8, line, ' ');
    const id_text = fields.next() orelse return error.InvalidArgument;
    const steps_text = fields.next() orelse return error.InvalidArgument;
    const tokens_text = fields.next() orelse return error.InvalidArgument;
    if (fields.next() != null) return error.InvalidArgument;
    const request_steps = try std.fmt.parseInt(usize, steps_text, 10);
    if (request_steps == 0 or request_steps > max_steps) return error.InvalidArgument;
    return .{
        .id = try std.fmt.parseInt(u64, id_text, 10),
        .steps = request_steps,
        .tokens = try parseTokenList(allocator, tokens_text, max_worker_prompt_tokens),
    };
}

fn stateHash(state: []const f32) u64 {
    var hash: u64 = 14695981039346656037;
    for (state) |x| {
        const h: f16 = @floatCast(x);
        const bits: u16 = @bitCast(h);
        hash ^= @as(u8, @truncate(bits));
        hash *%= 1099511628211;
        hash ^= @as(u8, @truncate(bits >> 8));
        hash *%= 1099511628211;
    }
    return hash;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5 or args.len > 13) {
        std.debug.print("usage: {s} cpu|htp|vulkan MODEL_SAFETENSORS TOKEN_ID CPU_THREADS [DECODED_CACHE_DIR [MODEL_SHA256 [STEPS [PROMPT_SUFFIX_TOKEN_IDS [prepare-cache|serve|serve-resident|serve-resident-fast|serve-packed|serve-q8|serve-q4|serve-q4-mlp-q8|serve-q4-attn-q8|serve-q4-attn-in-q8|serve-q4-attn-out-q8|serve-q4-linear-out-q8|serve-q4-full-out-q8|serve-q4-full-out-q8-early|serve-q4-full-out-q8-late|serve-q4-full-out-q8-q0-head-q8|serve-q4-full-out-q8-q0-head-q4|serve-q4-full-out-q8-q0-head-adaptive|serve-q4-full-out-q8-q0-head-adaptive-tail|serve-q4-full-out-q8-q0-head-adaptive-linear [REQUESTS [PROMPTS_JSON [serial-prefill]]]]]]]]\n", .{args[0]});
        return 64;
    }
    if (!std.mem.eql(u8, args[1], "cpu") and !std.mem.eql(u8, args[1], "htp") and !std.mem.eql(u8, args[1], "vulkan")) return error.InvalidBackend;
    const token = try std.fmt.parseInt(usize, args[3], 10);
    const threads = try std.fmt.parseInt(i32, args[4], 10);
    const steps = if (args.len >= 8) try std.fmt.parseInt(usize, args[7], 10) else 1;
    const suffix_count = if (args.len >= 9) std.mem.count(u8, args[8], ",") + 1 else 0;
    const prepare_cache_mode = args.len >= 10 and std.mem.eql(u8, args[9], "prepare-cache");
    const fast_activations = args.len >= 10 and std.mem.eql(u8, args[9], "serve-resident-fast");
    const packed_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-packed");
    const q8_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q8");
    const q4_mlp_q8_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-mlp-q8");
    const q4_attn_q8_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-attn-q8");
    const q4_attn_in_q8_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-attn-in-q8");
    const q4_attn_out_q8_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-attn-out-q8");
    const q4_linear_out_q8_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-linear-out-q8");
    const q4_full_out_q8_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8");
    const q4_full_out_q8_early_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-early");
    const q4_full_out_q8_late_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-late");
    const q4_full_out_q8_q0_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-q0");
    const q4_full_out_q8_q0_head_q8_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-q0-head-q8");
    const q4_full_out_q8_q0_head_q4_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-q0-head-q4");
    const q4_full_out_q8_q0_head_adaptive_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-q0-head-adaptive");
    const q4_full_out_q8_q0_head_adaptive_tail_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-q0-head-adaptive-tail");
    const q4_full_out_q8_q0_head_adaptive_linear_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-q0-head-adaptive-linear");
    const fused_tail_mode = q4_full_out_q8_q0_head_adaptive_tail_mode or q4_full_out_q8_q0_head_adaptive_linear_mode;
    const q4_full_out_q8_q1_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-q1");
    const q4_full_out_q8_l3_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-l3");
    const q4_full_out_q8_l7_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-q4-full-out-q8-l7");
    const q4_mode = args.len >= 10 and (std.mem.eql(u8, args[9], "serve-q4") or q4_mlp_q8_mode or q4_attn_q8_mode or q4_attn_in_q8_mode or q4_attn_out_q8_mode or q4_linear_out_q8_mode or q4_full_out_q8_mode or q4_full_out_q8_early_mode or q4_full_out_q8_late_mode or q4_full_out_q8_q0_mode or q4_full_out_q8_q0_head_q8_mode or q4_full_out_q8_q0_head_q4_mode or q4_full_out_q8_q0_head_adaptive_mode or q4_full_out_q8_q0_head_adaptive_tail_mode or q4_full_out_q8_q0_head_adaptive_linear_mode or q4_full_out_q8_q1_mode or q4_full_out_q8_l3_mode or q4_full_out_q8_l7_mode);
    const serve_mode = args.len >= 10 and (prepare_cache_mode or std.mem.eql(u8, args[9], "serve") or std.mem.eql(u8, args[9], "serve-resident") or fast_activations or packed_mode or q8_mode or q4_mode);
    const resident_mode = args.len >= 10 and (std.mem.eql(u8, args[9], "serve-resident") or fast_activations or packed_mode or q8_mode or q4_mode);
    const requests = if (args.len >= 11) try std.fmt.parseInt(usize, args[10], 10) else 1;
    const worker_mode = init.environ_map.get("TENSORFOLD_PERSISTENT_WORKER") != null;
    if (args.len >= 10 and !serve_mode) return error.InvalidArgument;
    const cli_prompt_len = 1 + suffix_count;

    var corpus_prompts: ?std.json.Parsed(std.json.ArrayHashMap([]usize)) = null;
    if (args.len >= 12) {
        const corpus_text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[11], init.gpa, .limited(32 << 20));
        defer init.gpa.free(corpus_text);
        corpus_prompts = try std.json.parseFromSlice(std.json.ArrayHashMap([]usize), init.gpa, corpus_text, .{ .allocate = .alloc_always });
    }
    defer if (corpus_prompts) |*parsed| parsed.deinit();
    const corpus_mode = corpus_prompts != null;
    const serial_corpus_prefill = args.len == 13 and std.mem.eql(u8, args[12], "serial-prefill");
    if (args.len > 13 or (args.len == 13 and !serial_corpus_prefill)) return error.InvalidArgument;
    const corpus_names = if (corpus_prompts) |*parsed| parsed.value.map.keys() else &.{};
    const corpus_values = if (corpus_prompts) |*parsed| parsed.value.map.values() else &.{};
    const request_count = if (corpus_mode) corpus_values.len else requests;
    const corpus_filter_len: ?usize = if (init.environ_map.get("TENSORFOLD_CORPUS_LENGTH")) |encoded| try std.fmt.parseInt(usize, encoded, 10) else null;
    if (corpus_filter_len != null and !corpus_mode) return error.InvalidArgument;
    const hidden_dump_path = init.environ_map.get("TENSORFOLD_HIDDEN_DUMP");
    const layer_dump_path = init.environ_map.get("TENSORFOLD_LAYER_DUMP");
    const norm_dump_path = init.environ_map.get("TENSORFOLD_NORM_DUMP");
    const verify_inputs_text = init.environ_map.get("TENSORFOLD_VERIFY_INPUTS");
    const verify_inputs: ?[]usize = if (verify_inputs_text) |encoded| try parseTokenList(init.gpa, encoded, 16) else null;
    defer if (verify_inputs) |tokens_to_verify| init.gpa.free(tokens_to_verify);
    const mtp_sidecar_path = init.environ_map.get("TENSORFOLD_MTP_SIDECAR");
    const use_live_mtp = mtp_sidecar_path != null;
    const mtp_depth = if (init.environ_map.get("TENSORFOLD_MTP_DEPTH")) |encoded| try std.fmt.parseInt(usize, encoded, 10) else 2;
    if (mtp_depth < 1 or mtp_depth > 6) return error.InvalidArgument;
    const mtp_initial_rank = if (init.environ_map.get("TENSORFOLD_MTP_INITIAL_RANK")) |encoded| try std.fmt.parseInt(usize, encoded, 10) else 1;
    if (mtp_initial_rank < 1 or mtp_initial_rank > 2 or (mtp_initial_rank != 1 and !use_live_mtp)) return error.InvalidArgument;
    const q8_output_layer_start_override: ?usize = if (init.environ_map.get("TENSORFOLD_Q8_OUTPUT_LAYER_START")) |encoded| try std.fmt.parseInt(usize, encoded, 10) else null;
    const q8_output_layer_end_override: ?usize = if (init.environ_map.get("TENSORFOLD_Q8_OUTPUT_LAYER_END")) |encoded| try std.fmt.parseInt(usize, encoded, 10) else null;
    const q8_output_layer_mask: u32 = if (init.environ_map.get("TENSORFOLD_Q8_OUTPUT_LAYER_MASK")) |encoded| try std.fmt.parseInt(u32, encoded, 0) else 0;
    const fused_linear_layer_mask: u32 = if (init.environ_map.get("TENSORFOLD_FUSED_LINEAR_LAYER_MASK")) |encoded| try std.fmt.parseInt(u32, encoded, 0) else 0xFFFFFFFF;
    if ((q8_output_layer_start_override orelse 0) > (q8_output_layer_end_override orelse layers) or
        (q8_output_layer_end_override orelse layers) > layers) return error.InvalidArgument;
    const mtp_first_margin: ?f32 = if (init.environ_map.get("TENSORFOLD_MTP_FIRST_MARGIN")) |encoded| try std.fmt.parseFloat(f32, encoded) else null;
    const mtp_second_margin: ?f32 = if (init.environ_map.get("TENSORFOLD_MTP_SECOND_MARGIN")) |encoded| try std.fmt.parseFloat(f32, encoded) else null;
    const target_ambiguity_margin: ?f32 = if (init.environ_map.get("TENSORFOLD_TARGET_AMBIGUITY_MARGIN")) |encoded| try std.fmt.parseFloat(f32, encoded) else null;
    if (mtp_first_margin) |threshold| {
        if (!std.math.isFinite(threshold) or threshold < 0 or !use_live_mtp or mtp_depth != 2) return error.InvalidArgument;
    }
    if (mtp_second_margin) |threshold| {
        if (!std.math.isFinite(threshold) or threshold < 0 or !use_live_mtp or mtp_depth != 2) return error.InvalidArgument;
    }
    if (target_ambiguity_margin) |threshold| {
        if (!std.math.isFinite(threshold) or threshold < 0 or !use_live_mtp or fused_linear_layer_mask != 0) return error.InvalidArgument;
    }
    const benchmark_quiet = init.environ_map.get("TENSORFOLD_BENCH_QUIET") != null;
    const stream_tokens = init.environ_map.get("TENSORFOLD_STREAM_TOKENS") != null;
    const mtp_policy_dump = init.environ_map.get("TENSORFOLD_MTP_POLICY_DUMP") != null;
    if (use_live_mtp and verify_inputs != null) return error.InvalidArgument;
    if ((hidden_dump_path != null or layer_dump_path != null or norm_dump_path != null) and request_count != 1) return error.InvalidArgument;
    if (verify_inputs != null and (corpus_mode or serial_corpus_prefill or !serve_mode or cli_prompt_len <= 1)) return error.InvalidArgument;
    if (use_live_mtp and (serial_corpus_prefill or !serve_mode)) return error.InvalidArgument;
    if (worker_mode and (corpus_mode or verify_inputs != null or hidden_dump_path != null or layer_dump_path != null or norm_dump_path != null or !resident_mode)) return error.InvalidArgument;
    var max_prompt_len = if (worker_mode) max_worker_prompt_tokens else cli_prompt_len;
    for (corpus_values) |prompt| {
        if (prompt.len == 0 or prompt.len > 2048) return error.InvalidArgument;
        max_prompt_len = @max(max_prompt_len, prompt.len);
        for (prompt) |prompt_token| if (prompt_token >= vocab) return error.InvalidArgument;
    }
    if (target_ambiguity_margin != null and max_prompt_len > max_worker_prompt_tokens) return error.InvalidArgument;
    const verify_positions = if (verify_inputs) |tokens_to_verify| cli_prompt_len + tokens_to_verify.len else 0;
    // A speculative verifier may compute up to `mtp_depth` look-ahead rows
    // beyond the requested output prefix; those tokens are discarded from the
    // report.  Size the decode/cache storage for the configured depth rather
    // than the historical depth-two default.
    const max_positions = @max(max_prompt_len + steps - 1, verify_positions) + (if (use_live_mtp) mtp_depth else 0);
    if (token >= vocab or threads <= 0 or steps == 0 or
        steps > 1024 or
        (!corpus_mode and cli_prompt_len > max_worker_prompt_tokens) or request_count == 0 or
        (!corpus_mode and requests > 16) or (request_count > 1 and !resident_mode)) return error.InvalidArgument;

    var file = if (std.mem.endsWith(u8, args[2], ".safetensors"))
        try st.Checkpoint.openPath(init.gpa, init.io, args[2])
    else
        try st.Checkpoint.openModel(init.gpa, init.io, args[2]);
    defer file.close();
    const embedding = try require(&file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
    const backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BackendInitFailed;
    defer tf_qualcomm_backend_destroy(backend);
    const cache_dir: ?[]const u8 = if (args.len >= 6) args[5] else null;
    const cache_model_id: ?[]const u8 = if (args.len >= 7) args[6] else null;
    if (cache_model_id) |id| {
        if (id.len != 64) return error.InvalidModelSha256;
        for (id) |c| if (!std.ascii.isHex(c)) return error.InvalidModelSha256;
    }
    if (cache_dir) |dir| try std.Io.Dir.cwd().createDirPath(init.io, dir);
    if (comptime experts > 0) {
        if (std.mem.endsWith(u8, args[2], ".safetensors") or use_live_mtp) return error.InvalidMoeConfiguration;
    }
    var expert_catalog: expert_pager.Catalog = undefined;
    var paged_experts: expert_pager.Pager = undefined;
    var expert_cache_ptr: ?*expert_pager.Pager = null;
    if (comptime experts > 0) {
        expert_catalog = try expert_pager.Catalog.open(init.gpa, init.io, args[2]);
        const cache_mib = if (init.environ_map.get("TENSORFOLD_EXPERT_CACHE_MIB")) |encoded| try std.fmt.parseInt(usize, encoded, 10) else 512;
        if (cache_mib < 64 or cache_mib > 4096) return error.InvalidExpertCache;
        paged_experts = try expert_pager.Pager.init(init.gpa, &expert_catalog, cache_mib << 20);
        expert_cache_ptr = &paged_experts;
    }
    defer if (comptime experts > 0) {
        paged_experts.deinit();
        expert_catalog.deinit();
    };
    var mtp_file: st.Checkpoint = undefined;
    var mtp_backend: ?*anyopaque = null;
    if (mtp_sidecar_path) |sidecar_path| {
        mtp_file = try st.Checkpoint.openPath(init.gpa, init.io, sidecar_path);
        try mtp.validateMtp(&mtp_file);
        mtp_backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BackendInitFailed;
    }
    defer if (use_live_mtp) {
        tf_qualcomm_backend_destroy(mtp_backend);
        mtp_file.close();
    };
    var mtp_runtime: mtp.Runtime = .{ .allocator = init.gpa, .file = &mtp_file, .backend = mtp_backend };
    var scratch_arena: std.heap.ArenaAllocator = .init(init.gpa);
    defer scratch_arena.deinit();
    // All projection uploads are consumed synchronously by the backend.  Keep
    // transient layer tensors in the resettable arena even for Q8; using the
    // general-purpose allocator here turns every decode layer into dozens of
    // malloc/free pairs after the resident weights are already loaded.
    var runner: Runner = .{
        .allocator = scratch_arena.allocator(),
        .io = init.io,
        .file = &file,
        .backend = backend,
        .cpu_threads = @intCast(threads),
        .backend_is_cpu = std.mem.eql(u8, args[1], "cpu"),
        .use_f16_weights = serve_mode and !prepare_cache_mode and !packed_mode and !q8_mode and !q4_mode,
        .use_resident_weights = resident_mode,
        .use_packed_exl3 = packed_mode or experts > 0,
        .use_f16_head_cache = init.environ_map.get("TENSORFOLD_DECODED_HEAD_CACHE") != null,
        .use_q8_weights = q8_mode,
        .use_q4_weights = q4_mode,
        .q8_attention_inputs = q4_attn_q8_mode or q4_attn_in_q8_mode,
        .q8_linear_outputs = q4_attn_q8_mode or q4_attn_out_q8_mode or q4_linear_out_q8_mode,
        .q8_full_outputs = q4_attn_q8_mode or q4_attn_out_q8_mode or q4_full_out_q8_mode or q4_full_out_q8_early_mode or q4_full_out_q8_late_mode or q4_full_out_q8_q0_mode or q4_full_out_q8_q0_head_q8_mode or q4_full_out_q8_q0_head_q4_mode or q4_full_out_q8_q0_head_adaptive_mode or fused_tail_mode or q4_full_out_q8_q1_mode or q4_full_out_q8_l3_mode or q4_full_out_q8_l7_mode,
        .q8_output_layer_start = q8_output_layer_start_override orelse (if (q4_full_out_q8_late_mode) layers / 2 else if (q4_full_out_q8_q1_mode) layers / 4 else if (q4_full_out_q8_l7_mode) 4 else 0),
        .q8_output_layer_end = q8_output_layer_end_override orelse (if (q4_full_out_q8_l3_mode) 4 else if (q4_full_out_q8_l7_mode) 8 else if (q4_full_out_q8_q0_mode or q4_full_out_q8_q0_head_q8_mode or q4_full_out_q8_q0_head_q4_mode or q4_full_out_q8_q0_head_adaptive_mode or fused_tail_mode) layers / 4 else if (q4_full_out_q8_early_mode or q4_full_out_q8_q1_mode) layers / 2 else layers),
        .q8_output_layer_mask = q8_output_layer_mask,
        .q8_mlp = q4_mlp_q8_mode,
        .use_resident_head = profile.tied_embeddings and resident_mode and (request_count > 1 or use_live_mtp),
        .use_q8_head = q4_full_out_q8_q0_head_q8_mode,
        .use_q4_head = q4_full_out_q8_q0_head_q4_mode,
        .use_adaptive_head = q4_full_out_q8_q0_head_adaptive_mode or fused_tail_mode,
        .use_refined_head = fused_tail_mode,
        .use_fused_tail = fused_tail_mode,
        .use_fused_linear_tail = q4_full_out_q8_q0_head_adaptive_linear_mode,
        .fused_linear_layer_mask = fused_linear_layer_mask,
        .fast_activations = (fast_activations or q8_mode or q4_mode) and init.environ_map.get("TENSORFOLD_EXACT_ACTIVATIONS") == null,
        .cache_dir = cache_dir,
        .cache_model_id = cache_model_id,
        .expert_cache = expert_cache_ptr,
    };
    var decode = try DecodeState.init(init.gpa, max_positions);
    defer decode.deinit(init.gpa);
    const state = try init.gpa.alloc(f32, hidden);
    defer init.gpa.free(state);
    const normed = try init.gpa.alloc(f32, hidden);
    defer init.gpa.free(normed);
    const update = try init.gpa.alloc(f32, if (runner.use_fused_tail) attention_query_width else hidden);
    defer init.gpa.free(update);
    const sequence = try init.gpa.alloc(usize, max_positions + 1);
    defer init.gpa.free(sequence);
    var hidden_dump: ?std.Io.File = if (hidden_dump_path) |path|
        try std.Io.Dir.cwd().createFile(init.io, path, .{ .truncate = true })
    else
        null;
    defer if (hidden_dump) |*dump| dump.close(init.io);
    var layer_dump: ?std.Io.File = if (layer_dump_path) |path|
        try std.Io.Dir.cwd().createFile(init.io, path, .{ .truncate = true })
    else
        null;
    defer if (layer_dump) |*dump| dump.close(init.io);
    var norm_dump: ?std.Io.File = if (norm_dump_path) |path|
        try std.Io.Dir.cwd().createFile(init.io, path, .{ .truncate = true })
    else
        null;
    defer if (norm_dump) |*dump| dump.close(init.io);
    if (init.environ_map.get("TENSORFOLD_HEAD_INPUT")) |path| {
        const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, path, init.gpa, .limited(hidden * @sizeOf(f32) * max_worker_prompt_tokens));
        defer init.gpa.free(raw);
        if (raw.len == 0 or raw.len % (hidden * @sizeOf(f32)) != 0) return error.InvalidArgument;
        const head_state = try init.gpa.alloc(f32, hidden);
        defer init.gpa.free(head_state);
        @memcpy(std.mem.sliceAsBytes(head_state), raw[raw.len - hidden * @sizeOf(f32) ..]);
        const top = try tiedHead(&runner, head_state, threads);
        std.debug.print(
            "{{\"tensorfold\":\"qualcomm-exl3-head\",\"top5\":[[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}]]}}\n",
            .{ top[0].id, top[0].logit, top[1].id, top[1].logit, top[2].id, top[2].logit, top[3].id, top[3].logit, top[4].id, top[4].logit },
        );
        return 0;
    }
    var prompt_cache: ?PromptCache = if (worker_mode and use_live_mtp and fused_linear_layer_mask == 0) try PromptCache.init(init.gpa, 4) else null;
    defer if (prompt_cache) |*cache| cache.deinit(init.gpa);
    var exactness_snapshot: ?PromptSnapshot = if (target_ambiguity_margin != null) try PromptSnapshot.init(init.gpa) else null;
    defer if (exactness_snapshot) |*snapshot| snapshot.deinit(init.gpa);

    var worker_stdin_buffer: [8192]u8 = undefined;
    var worker_stdin = std.Io.File.stdin().readerStreaming(init.io, &worker_stdin_buffer);
    if (worker_mode) {
        std.debug.print("{{\"tensorfold\":\"qualcomm-exl3-worker-ready\",\"max_prompt_tokens\":{d},\"max_steps\":{d}}}\n", .{ max_worker_prompt_tokens, steps });
    }
    var completed_requests: usize = 0;
    var request_index: usize = 0;
    while (worker_mode or request_index < request_count) : (request_index += 1) {
        var request_steps = steps;
        var request_id: u64 = @intCast(request_index);
        var worker_tokens: ?[]usize = null;
        defer if (worker_tokens) |dynamic_prompt| init.gpa.free(dynamic_prompt);
        if (worker_mode) {
            const raw = worker_stdin.interface.takeDelimiterInclusive('\n') catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            const line = std.mem.trim(u8, raw, " \r\n");
            const worker_command = try parseWorkerCommand(init.gpa, line, steps);
            worker_tokens = worker_command.tokens;
            request_steps = worker_command.steps;
            request_id = worker_command.id;
            std.debug.print("{{\"tensorfold\":\"qualcomm-exl3-worker-start\",\"id\":{d},\"prompt_tokens\":{d},\"steps\":{d}}}\n", .{ request_id, worker_command.tokens.len, request_steps });
        }
        if (corpus_filter_len) |selected_len| if (corpus_values[request_index].len != selected_len) continue;
        runner.resetMetrics();
        if (use_live_mtp) {
            mtp_runtime.projection_calls = 0;
            mtp_runtime.projection_uploads = 0;
            mtp_runtime.projection_compute_ms = 0;
        }
        const started = tf_qualcomm_monotonic_ms();
        decode.reset(true);
        if (runner.use_fused_linear_tail) runner.resident_gdn_loaded = @splat(false);
        if ((verify_inputs != null or use_live_mtp) and completed_requests != 0) {
            for (0..layers) |layer| {
                if ((layer + 1) % 4 == 0) continue;
                // Fused-linear slots carry their own resident GDN state and
                // are keyed by the projection cursor rather than `layer`.
                // Marking them unloaded above makes the first prompt call
                // upload decode.gdnLayer(layer), which reset() has zeroed.
                if (runner.usesFusedLinearTail(layer)) continue;
                const zero_state = decode.gdnLayer(layer);
                if (tf_qualcomm_backend_gdn_state_set(backend, layer, zero_state.ptr, linear_state_elems) != 0) return error.BridgeFailed;
            }
        }
        const prompt_len = if (worker_tokens) |dynamic_prompt| dynamic_prompt.len else if (corpus_mode) corpus_values[request_index].len else cli_prompt_len;
        const positions = prompt_len + request_steps - 1;
        if (worker_tokens) |dynamic_prompt| {
            @memcpy(sequence[0..prompt_len], dynamic_prompt);
        } else if (corpus_mode) {
            @memcpy(sequence[0..prompt_len], corpus_values[request_index]);
        } else {
            sequence[0] = token;
        }
        if (worker_tokens == null and !corpus_mode and args.len >= 9) {
            var suffix = std.mem.splitScalar(u8, args[8], ',');
            var i: usize = 1;
            while (suffix.next()) |encoded| : (i += 1) {
                const suffix_token = try std.fmt.parseInt(usize, encoded, 10);
                if (suffix_token >= vocab) return error.InvalidArgument;
                sequence[i] = suffix_token;
            }
        }
        var name_buffer: [256]u8 = undefined;
        var total_head_ms: f64 = 0;
        var skipped_heads: usize = 0;
        var decode_start: usize = 0;
        var generated_count: usize = 0;
        var first_token_ms: f64 = 0;
        var stopped_on_eos = false;
        var hidden_rows_dumped: usize = 0;
        const target_hidden_history: ?[]f32 = if (use_live_mtp) try init.gpa.alloc(f32, (max_positions + 2) * hidden) else null;
        defer if (target_hidden_history) |history| init.gpa.free(history);
        const matched_snapshot: ?*PromptSnapshot = if (prompt_cache) |*cache| cache.find(sequence[0..prompt_len]) else null;
        const prefill_start = if (matched_snapshot) |snapshot| snapshot.prompt_len else 0;
        if (matched_snapshot) |snapshot| try snapshot.restore(backend, &runner, &decode, target_hidden_history.?, normed);
        var request_prompt_snapshot: ?*PromptSnapshot = if (matched_snapshot != null and prefill_start == prompt_len) matched_snapshot else null;
        var capture_prompt_snapshot = false;
        const head_kind = if (runner.use_resident_head)
            (if (runner.use_refined_head) "tied-q4-f16-top5-refine" else if (runner.use_adaptive_head) "tied-q4-q8-adaptive-htp-resident" else if (runner.use_q4_head) "tied-q4-htp-resident" else if (runner.use_q8_head) "tied-q8-htp-resident" else "tied-f16-htp-resident")
        else
            "tied-f16-neon";

        if (matched_snapshot != null and prefill_start == prompt_len) {
            const prefill_started = tf_qualcomm_monotonic_ms();
            const snapshot = matched_snapshot.?;
            sequence[prompt_len] = snapshot.next_token;
            generated_count = 1;
            skipped_heads = prompt_len - 1;
            first_token_ms = tf_qualcomm_monotonic_ms() - started;
            const prefill_elapsed = tf_qualcomm_monotonic_ms() - prefill_started;
            std.debug.print(
                "{{\"tensorfold\":\"qualcomm-exl3-prefill\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"prompt_tokens\":{d},\"batch\":0,\"layers\":{d},\"projections\":0,\"head\":\"prompt-cache\",\"heads_skipped\":{d},\"next_token\":{d},\"state_f16_fnv1a\":\"{x}\",\"matmul_compute_ms\":0,\"projection_wall_ms\":0,\"head_ms\":0,\"elapsed_ms\":{d:.3},\"cache_hit\":true,\"cache_prefix_tokens\":{d},\"passed\":true}}\n",
                .{ profile.name, args[1], prompt_len, layers, skipped_heads, snapshot.next_token, stateHash(normed), prefill_elapsed, prefill_start },
            );
            decode_start = prompt_len;
        } else if (serve_mode and prompt_len > prefill_start and !serial_corpus_prefill) {
            const prefill_started = tf_qualcomm_monotonic_ms();
            const projection_start = runner.projection_count;
            const compute_start = runner.projection_ms;
            const wall_start = runner.projection_wall_ms;
            var pprof_norm: f64 = 0;
            var pprof_att: f64 = 0;
            var pprof_mlp: f64 = 0;
            var pprof_embed: f64 = 0;
            const pprof_on = init.environ_map.get("TENSORFOLD_PREFILL_PROFILE") != null;
            if (pprof_on) {
                prof_on = true;
                prof_gdn = 0;
                prof_dense = 0;
                prof_pair = 0;
                prof_conv = 0;
                prof_fa = 0;
            }
            const prefill_count = prompt_len - prefill_start;
            const chunk_capacity = @min(prefill_count, max_worker_prompt_tokens / 4);
            const prompt_states = try init.gpa.alloc(f32, chunk_capacity * hidden);
            const prompt_normed = try init.gpa.alloc(f32, chunk_capacity * hidden);
            const prompt_update = try init.gpa.alloc(f32, chunk_capacity * (if (runner.use_fused_tail) attention_query_width else hidden));
            var chunk_offset: usize = 0;
            while (chunk_offset < prefill_count) {
                const chunk_count = @min(chunk_capacity, prefill_count - chunk_offset);
                const chunk_start = prefill_start + chunk_offset;
                const states = prompt_states[0 .. chunk_count * hidden];
                const normalized = prompt_normed[0 .. chunk_count * hidden];
                const updates = prompt_update[0 .. chunk_count * (if (runner.use_fused_tail) attention_query_width else hidden)];
                const embed_t0 = tf_qualcomm_monotonic_ms();
                for (0..chunk_count) |position| {
                    const row = states[position * hidden ..][0..hidden];
                    for (row, 0..) |*x, d| x.* = f16Value(embedding.bytes, sequence[chunk_start + position] * hidden + d);
                }
                pprof_embed += tf_qualcomm_monotonic_ms() - embed_t0;
                runner.resident_slot_cursor = 0;
                for (0..layers) |layer| {
                    _ = scratch_arena.reset(.retain_capacity);
                    const layer_t0 = tf_qualcomm_monotonic_ms();
                    const input_norm = try tensorName(&name_buffer, "{s}.layers.{d}.input_layernorm.weight", .{ prefix, layer });
                    for (0..chunk_count) |position| {
                        try rmsNorm(&file, input_norm, states[position * hidden ..][0..hidden], normalized[position * hidden ..][0..hidden], 1.0);
                    }
                    if (norm_dump) |*dump| {
                        try dump.writeStreamingAll(init.io, std.mem.sliceAsBytes(normalized));
                    }
                    const norm_t1 = tf_qualcomm_monotonic_ms();
                    const norm_dl = norm_t1 - layer_t0;
                    pprof_norm += norm_dl;
                    if ((layer + 1) % 4 == 0) {
                        const fa_t0 = if (prof_on) tf_qualcomm_monotonic_ms() else 0;
                        try fullAttentionBatch(&runner, &decode, layer, chunk_start, normalized, updates);
                        if (prof_on) prof_fa += tf_qualcomm_monotonic_ms() - fa_t0;
                    } else {
                        try linearAttentionBatch(&runner, &decode, layer, normalized, if (runner.usesFusedLinearTail(layer)) states else null, null, updates);
                    }
                    const att_t2 = tf_qualcomm_monotonic_ms();
                    const att_dl = att_t2 - norm_t1;
                    pprof_att += att_dl;
                    if (runner.usesFusedLinearTail(layer) and (layer + 1) % 4 != 0) {
                        // The fused graph completed the residual and MLP.
                    } else if (runner.use_fused_tail) {
                        var post_norm_buffer: [256]u8 = undefined;
                        var output_name_buffer: [256]u8 = undefined;
                        const post_norm = try tensorName(&post_norm_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
                        const output_name = if ((layer + 1) % 4 == 0)
                            try tensorName(&output_name_buffer, "{s}.layers.{d}.self_attn.o_proj", .{ prefix, layer })
                        else
                            try tensorName(&output_name_buffer, "{s}.layers.{d}.linear_attn.out_proj", .{ prefix, layer });
                        try runner.layerTailQuant(layer, output_name, post_norm, states, updates);
                    } else {
                        residualAdd(states, updates);
                        const post_norm = try tensorName(&name_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
                        for (0..chunk_count) |position| {
                            try rmsNorm(&file, post_norm, states[position * hidden ..][0..hidden], normalized[position * hidden ..][0..hidden], 1.0);
                        }
                        if (norm_dump) |*dump| {
                            try dump.writeStreamingAll(init.io, std.mem.sliceAsBytes(normalized));
                        }
                        const mlp_t3 = tf_qualcomm_monotonic_ms();
                        try mlpBatch(&runner, layer, normalized, updates);
                        pprof_mlp += tf_qualcomm_monotonic_ms() - mlp_t3;
                        residualAdd(states, updates);
                    }
                    if (layer_dump) |*dump| {
                        try dump.writeStreamingAll(init.io, std.mem.sliceAsBytes(states));
                    }
                    if (pprof_on) {
                        const layer_end = tf_qualcomm_monotonic_ms();
                        std.debug.print("PREFILL_LAYER {d}: norm={d:.3} att={d:.3} mlp=0 rest={d:.3}\n", .{ layer, norm_dl, att_dl, layer_end - layer_t0 - norm_dl - att_dl });
                    }
                }
                if (hidden_dump != null or use_live_mtp) {
                    for (0..chunk_count) |row_index| {
                        const row = normalized[row_index * hidden ..][0..hidden];
                        try rmsNorm(&file, prefix ++ ".norm.weight", states[row_index * hidden ..][0..hidden], row, 1.0);
                        if (hidden_dump) |*dump| {
                            try dump.writeStreamingAll(init.io, std.mem.sliceAsBytes(row));
                            hidden_rows_dumped += 1;
                        }
                        const position = chunk_start + row_index;
                        if (target_hidden_history) |history| @memcpy(history[position * hidden ..][0..hidden], row);
                    }
                    @memcpy(normed, normalized[(chunk_count - 1) * hidden ..][0..hidden]);
                } else {
                    try rmsNorm(&file, prefix ++ ".norm.weight", states[(chunk_count - 1) * hidden ..][0..hidden], normed, 1.0);
                }
                chunk_offset += chunk_count;
            }
            if (pprof_on) {
                std.debug.print("PREFILL_PROF pair={d:.3} dense={d:.3} conv={d:.3} gdn={d:.3} fa={d:.3}\n", .{ prof_pair, prof_dense, prof_conv, prof_gdn, prof_fa });
                prof_on = false;
            }
            const head_started = tf_qualcomm_monotonic_ms();
            const top = try tiedHead(&runner, normed, threads);
            const head_ms = tf_qualcomm_monotonic_ms() - head_started;
            total_head_ms += head_ms;
            skipped_heads = prompt_len - 1;
            sequence[prompt_len] = top[0].id;
            generated_count = 1;
            first_token_ms = tf_qualcomm_monotonic_ms() - started;
            stopped_on_eos = corpus_mode and (top[0].id == 248046 or top[0].id == 248044);
            const prefill_elapsed = tf_qualcomm_monotonic_ms() - prefill_started;
            std.debug.print(
                "{{\"tensorfold\":\"qualcomm-exl3-prefill\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"prompt_tokens\":{d},\"batch\":{d},\"layers\":{d},\"projections\":{d},\"head\":\"{s}\",\"heads_skipped\":{d},\"next_token\":{d},\"top5\":[[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}]],\"state_f16_fnv1a\":\"{x}\",\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"head_ms\":{d:.3},\"elapsed_ms\":{d:.3},\"cache_prefix_tokens\":{d},\"passed\":true}}\n",
                .{ profile.name, args[1], prompt_len, prefill_count, layers, runner.projection_count - projection_start, head_kind, skipped_heads, top[0].id, top[0].id, top[0].logit, top[1].id, top[1].logit, top[2].id, top[2].logit, top[3].id, top[3].logit, top[4].id, top[4].logit, stateHash(normed), runner.projection_ms - compute_start, runner.projection_wall_ms - wall_start, head_ms, prefill_elapsed, prefill_start },
            );
            capture_prompt_snapshot = prompt_cache != null;
            init.gpa.free(prompt_update);
            init.gpa.free(prompt_normed);
            init.gpa.free(prompt_states);
            decode_start = prompt_len;
        }
        if (worker_mode and !use_live_mtp and stream_tokens) {
            std.debug.print(
                "{{\"tensorfold\":\"qualcomm-exl3-speculative-chunk\",\"start\":0,\"count\":1,\"tokens\":[{d},0,0,0,0,0,0]}}\n",
                .{sequence[prompt_len]},
            );
        }

        if (verify_inputs != null or use_live_mtp) {
            const tokens_to_verify = verify_inputs orelse &.{};
            if (decode_start != prompt_len) return error.InvalidArgument;
            if (!use_live_mtp and (tokens_to_verify[0] != sequence[prompt_len] or tokens_to_verify.len < 2 or (tokens_to_verify.len > 3 and tokens_to_verify.len % 3 != 0))) return error.InvalidArgument;
            const speculative_started = tf_qualcomm_monotonic_ms();
            // Keep one host staging range per layer. Hexagon tensor_set may
            // retain the source until the next graph submission; reusing one
            // scratch range across layers can race the queued state uploads.
            const gdn_restore_scratch = try init.gpa.alloc(f32, layers * linear_state_elems);
            defer init.gpa.free(gdn_restore_scratch);
            const conv_snapshot = try init.gpa.alloc(f32, decode.conv.len);
            defer init.gpa.free(conv_snapshot);
            const conv_commits = try init.gpa.alloc(f32, layers * 7 * conv_channels);
            defer init.gpa.free(conv_commits);
            const verify_states = try init.gpa.alloc(f32, 7 * hidden);
            defer init.gpa.free(verify_states);
            const verify_normed = try init.gpa.alloc(f32, 7 * hidden);
            defer init.gpa.free(verify_normed);
            const verify_update = try init.gpa.alloc(f32, 7 * (if (runner.use_fused_tail) attention_query_width else hidden));
            defer init.gpa.free(verify_update);
            const generated = try init.gpa.alloc(usize, if (use_live_mtp) request_steps + 2 else 1 + tokens_to_verify.len);
            defer init.gpa.free(generated);
            generated[0] = sequence[prompt_len];
            var generated_len: usize = 1;
            var current_position = prompt_len;
            var pending_token = sequence[prompt_len];
            var token_offset: usize = 0;
            var round: usize = 0;
            var total_accepted: usize = 0;
            var mtp_scratch: std.heap.ArenaAllocator = .init(init.gpa);
            defer mtp_scratch.deinit();
            const mtp_state_a: []f32 = if (use_live_mtp) try init.gpa.alloc(f32, hidden) else @constCast(&.{});
            defer if (use_live_mtp) init.gpa.free(mtp_state_a);
            const mtp_state_b: []f32 = if (use_live_mtp) try init.gpa.alloc(f32, hidden) else @constCast(&.{});
            defer if (use_live_mtp) init.gpa.free(mtp_state_b);
            const mtp_catchup_hidden: []f32 = if (use_live_mtp) try init.gpa.alloc(f32, 7 * hidden) else @constCast(&.{});
            defer if (use_live_mtp) init.gpa.free(mtp_catchup_hidden);
            const mtp_catchup_output: []f32 = if (use_live_mtp) try init.gpa.alloc(f32, 7 * hidden) else @constCast(&.{});
            defer if (use_live_mtp) init.gpa.free(mtp_catchup_output);
            var live_mtp_cache: mtp.Cache = undefined;
            var mtp_prefill_ms: f64 = 0;
            if (use_live_mtp) {
                const mtp_prefill_started = tf_qualcomm_monotonic_ms();
                live_mtp_cache = try mtp.Cache.init(init.gpa, max_positions + 2);
                var mtp_prefill_start: usize = 0;
                if (matched_snapshot) |snapshot| {
                    if (snapshot.mtp_valid) {
                        try snapshot.restoreMtp(&live_mtp_cache);
                        mtp_prefill_start = snapshot.prompt_len;
                    }
                }
                if (mtp_prefill_start < current_position) {
                    const mtp_prefill_count = current_position - mtp_prefill_start;
                    const mtp_prefill_hidden = try init.gpa.alloc(f32, mtp_prefill_count * hidden);
                    defer init.gpa.free(mtp_prefill_hidden);
                    const mtp_prefill_output = try init.gpa.alloc(f32, mtp_prefill_count * hidden);
                    defer init.gpa.free(mtp_prefill_output);
                    for (0..mtp_prefill_count) |row_index| {
                        const position = mtp_prefill_start + row_index;
                        const row = mtp_prefill_hidden[row_index * hidden ..][0..hidden];
                        if (position == 0) @memset(row, 0.0) else @memcpy(row, target_hidden_history.?[(position - 1) * hidden ..][0..hidden]);
                    }
                    var mtp_offset: usize = 0;
                    while (mtp_offset < mtp_prefill_count) {
                        const mtp_chunk_count = @min(@as(usize, 7), mtp_prefill_count - mtp_offset);
                        _ = mtp_scratch.reset(.retain_capacity);
                        try mtp.runBatch(
                            true,
                            mtp_scratch.allocator(),
                            &mtp_runtime,
                            embedding,
                            &live_mtp_cache,
                            mtp_prefill_hidden[mtp_offset * hidden ..][0 .. mtp_chunk_count * hidden],
                            sequence[mtp_prefill_start + mtp_offset ..][0..mtp_chunk_count],
                            mtp_prefill_start + mtp_offset,
                            mtp_prefill_output[mtp_offset * hidden ..][0 .. mtp_chunk_count * hidden],
                        );
                        mtp_offset += mtp_chunk_count;
                    }
                }
                mtp_prefill_ms = tf_qualcomm_monotonic_ms() - mtp_prefill_started;
                if (capture_prompt_snapshot) {
                    const snapshot = prompt_cache.?.reserve();
                    try snapshot.capture(backend, sequence[0..prompt_len], &decode, target_hidden_history.?, normed, sequence[prompt_len]);
                    try snapshot.captureMtp(&live_mtp_cache);
                    request_prompt_snapshot = snapshot;
                }
                if (target_ambiguity_margin != null and request_prompt_snapshot == null) {
                    const snapshot = &exactness_snapshot.?;
                    try snapshot.capture(backend, sequence[0..prompt_len], &decode, target_hidden_history.?, normed, sequence[prompt_len]);
                    request_prompt_snapshot = snapshot;
                }
            }
            defer if (use_live_mtp) live_mtp_cache.deinit(init.gpa);
            var mtp_next_position = current_position;
            var adaptive_depth1_rounds: usize = 0;
            var adaptive_depth2_rounds: usize = 0;
            var serial_fallbacks: usize = 0;
            if (stream_tokens or !benchmark_quiet) {
                std.debug.print("{{\"tensorfold\":\"qualcomm-exl3-speculative-chunk\",\"start\":0,\"count\":1,\"tokens\":[{d},0,0,0,0,0,0]}}\n", .{generated[0]});
            }
            speculative_rounds: while (if (use_live_mtp) generated_len < request_steps else token_offset < tokens_to_verify.len) : (round += 1) {
                if (round >= request_steps + 8) return error.InvalidArgument;
                var live_window: [7]usize = undefined;
                // Never verify more rows than can contribute to the requested
                // output.  Four- and five-row resident graphs are not kept;
                // use the already-qualified three-row graph for those tails.
                const remaining_outputs = if (use_live_mtp) request_steps - generated_len else tokens_to_verify.len - token_offset;
                var count: usize = @min(mtp_depth + 1, remaining_outputs);
                if (count == 4 or count == 5) count = 3;
                var window: []const usize = undefined;
                if (use_live_mtp) {
                    live_window[0] = pending_token;
                    var draft_input_token = pending_token;
                    var draft_input_hidden: []const f32 = &.{};
                    var draft_output: []f32 = undefined;
                    var drafts_computed: usize = 0;
                    var first_draft_margin: f32 = 0;
                    var second_draft_margin: f32 = 0;
                    for (0..count - 1) |draft_index| {
                        _ = mtp_scratch.reset(.retain_capacity);
                        if (draft_index == 0) {
                            const catchup_count = current_position - mtp_next_position + 1;
                            if (catchup_count == 0 or catchup_count > 7) return error.InvalidArgument;
                            for (0..catchup_count) |catchup_index| {
                                const position = mtp_next_position + catchup_index;
                                @memcpy(mtp_catchup_hidden[catchup_index * hidden ..][0..hidden], target_hidden_history.?[(position - 1) * hidden ..][0..hidden]);
                            }
                            // Resident MTP projections keep qualified graph
                            // shapes for 1/2/3/6/7 rows.  A rejected proposal
                            // can leave four or five target-confirmed rows to
                            // catch up; split those as 3+1 or 3+2 rather than
                            // requesting an unsupported resident graph.
                            var catchup_offset: usize = 0;
                            while (catchup_offset < catchup_count) {
                                const catchup_remaining = catchup_count - catchup_offset;
                                const catchup_chunk: usize = if (catchup_remaining >= 6)
                                    catchup_remaining
                                else if (catchup_remaining >= 3)
                                    3
                                else
                                    catchup_remaining;
                                try mtp.runBatch(
                                    true,
                                    mtp_scratch.allocator(),
                                    &mtp_runtime,
                                    embedding,
                                    &live_mtp_cache,
                                    mtp_catchup_hidden[catchup_offset * hidden ..][0 .. catchup_chunk * hidden],
                                    sequence[mtp_next_position + catchup_offset ..][0..catchup_chunk],
                                    mtp_next_position + catchup_offset,
                                    mtp_catchup_output[catchup_offset * hidden ..][0 .. catchup_chunk * hidden],
                                );
                                catchup_offset += catchup_chunk;
                            }
                            draft_output = mtp_catchup_output[(catchup_count - 1) * hidden ..][0..hidden];
                            mtp_next_position = current_position + 1;
                        } else {
                            draft_output = if (draft_index % 2 == 1) mtp_state_a else mtp_state_b;
                            try mtp.runStep(true, mtp_scratch.allocator(), &mtp_runtime, embedding, &live_mtp_cache, draft_input_hidden, draft_input_token, current_position + draft_index, draft_output);
                        }
                        const draft_top = try tiedHeadDraft(&runner, draft_output, threads);
                        const selected_rank = if (round == 0 and draft_index == 0) mtp_initial_rank - 1 else 0;
                        live_window[draft_index + 1] = draft_top[selected_rank].id;
                        drafts_computed += 1;
                        if (draft_index == 0) first_draft_margin = draft_top[0].logit - draft_top[1].logit;
                        if (draft_index == 1) second_draft_margin = draft_top[0].logit - draft_top[1].logit;
                        if (!benchmark_quiet) {
                            std.debug.print(
                                "{{\"tensorfold\":\"qualcomm-exl3-mtp-draft\",\"round\":{d},\"draft_index\":{d},\"position\":{d},\"input_token\":{d},\"proposal_rank\":{d},\"proposed_token\":{d},\"top1_token\":{d},\"top1_logit\":{d},\"top2_token\":{d},\"top2_logit\":{d},\"margin\":{d}}}\n",
                                .{ round, draft_index, current_position + draft_index, draft_input_token, selected_rank + 1, live_window[draft_index + 1], draft_top[0].id, draft_top[0].logit, draft_top[1].id, draft_top[1].logit, draft_top[0].logit - draft_top[1].logit },
                            );
                        }
                        draft_input_token = draft_top[selected_rank].id;
                        draft_input_hidden = draft_output;
                        if (draft_index == 0) {
                            if (mtp_first_margin) |threshold| {
                                if (first_draft_margin < threshold) {
                                    count = 2;
                                    break;
                                }
                            }
                        }
                    }
                    if (count == 3) {
                        if (mtp_second_margin) |threshold| {
                            if (second_draft_margin < threshold) count = 2;
                        }
                    }
                    if (mtp_first_margin != null or mtp_second_margin != null) {
                        if (count == 2) adaptive_depth1_rounds += 1 else adaptive_depth2_rounds += 1;
                        if (!benchmark_quiet) {
                            std.debug.print(
                                "{{\"tensorfold\":\"qualcomm-exl3-mtp-depth-decision\",\"round\":{d},\"drafts_computed\":{d},\"first_margin\":{d},\"second_margin\":{d},\"verify_rows\":{d}}}\n",
                                .{ round, drafts_computed, first_draft_margin, second_draft_margin, count },
                            );
                        }
                    }
                    window = live_window[0..count];
                } else {
                    count = if (tokens_to_verify.len <= 3) tokens_to_verify.len else 3;
                    window = tokens_to_verify[token_offset .. token_offset + count];
                }
                if (window[0] != pending_token) return error.InvalidArgument;
                const verify_started = tf_qualcomm_monotonic_ms();
                const projection_start = runner.projection_count;
                const compute_start = runner.projection_ms;
                const wall_start = runner.projection_wall_ms;
                var snapshot_ms: f64 = 0;
                if (count > 1) {
                    const snapshot_started = tf_qualcomm_monotonic_ms();
                    @memcpy(conv_snapshot, decode.conv);
                    snapshot_ms = tf_qualcomm_monotonic_ms() - snapshot_started;
                }
                for (window, 0..) |verify_token, row_index| {
                    const row = verify_states[row_index * hidden ..][0..hidden];
                    for (row, 0..) |*x, d| x.* = f16Value(embedding.bytes, verify_token * hidden + d);
                }
                runner.resident_slot_cursor = 0;
                for (0..layers) |layer| {
                    _ = scratch_arena.reset(.retain_capacity);
                    const input_norm = try tensorName(&name_buffer, "{s}.layers.{d}.input_layernorm.weight", .{ prefix, layer });
                    for (0..count) |row_index| try rmsNorm(&file, input_norm, verify_states[row_index * hidden ..][0..hidden], verify_normed[row_index * hidden ..][0..hidden], 1.0);
                    if ((layer + 1) % 4 == 0) {
                        try fullAttentionBatch(&runner, &decode, layer, current_position, verify_normed[0 .. count * hidden], verify_update[0 .. count * (if (runner.use_fused_tail) attention_query_width else hidden)]);
                    } else {
                        const commit = conv_commits[layer * 7 * conv_channels ..][0 .. count * conv_channels];
                        try linearAttentionBatch(&runner, &decode, layer, verify_normed[0 .. count * hidden], if (runner.usesFusedLinearTail(layer)) verify_states[0 .. count * hidden] else null, commit, verify_update[0 .. count * (if (runner.use_fused_tail) attention_query_width else hidden)]);
                    }
                    if (runner.usesFusedLinearTail(layer) and (layer + 1) % 4 != 0) {
                        // Completed by the fused persistent HTP graph.
                    } else if (runner.use_fused_tail) {
                        var post_norm_buffer: [256]u8 = undefined;
                        var output_name_buffer: [256]u8 = undefined;
                        const post_norm = try tensorName(&post_norm_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
                        const output_name = if ((layer + 1) % 4 == 0)
                            try tensorName(&output_name_buffer, "{s}.layers.{d}.self_attn.o_proj", .{ prefix, layer })
                        else
                            try tensorName(&output_name_buffer, "{s}.layers.{d}.linear_attn.out_proj", .{ prefix, layer });
                        try runner.layerTailQuant(layer, output_name, post_norm, verify_states[0 .. count * hidden], verify_update[0 .. count * (if (runner.use_fused_tail) attention_query_width else hidden)]);
                    } else {
                        residualAdd(verify_states[0 .. count * hidden], verify_update[0 .. count * hidden]);
                        const post_norm = try tensorName(&name_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
                        for (0..count) |row_index| try rmsNorm(&file, post_norm, verify_states[row_index * hidden ..][0..hidden], verify_normed[row_index * hidden ..][0..hidden], 1.0);
                        try mlpBatch(&runner, layer, verify_normed[0 .. count * hidden], verify_update[0 .. count * hidden]);
                        residualAdd(verify_states[0 .. count * hidden], verify_update[0 .. count * hidden]);
                    }
                }
                for (0..count) |row_index| try rmsNorm(&file, prefix ++ ".norm.weight", verify_states[row_index * hidden ..][0..hidden], verify_normed[row_index * hidden ..][0..hidden], 1.0);
                const head_started = tf_qualcomm_monotonic_ms();
                const tops = try tiedHeadBatch(&runner, verify_normed[0 .. count * hidden], threads);
                const verify_head_ms = tf_qualcomm_monotonic_ms() - head_started;
                var accepted_prefix: usize = 0;
                var prefix_open = true;
                for (tops, 0..) |top, row_index| {
                    const has_proposal = row_index + 1 < count;
                    const proposal = if (has_proposal) window[row_index + 1] else 0;
                    const matched = has_proposal and prefix_open and top[0].id == proposal;
                    if (has_proposal and prefix_open) {
                        if (matched) accepted_prefix += 1 else prefix_open = false;
                    }
                    if (!benchmark_quiet) {
                        std.debug.print(
                            "{{\"tensorfold\":\"qualcomm-exl3-verify-row\",\"round\":{d},\"row\":{d},\"position\":{d},\"input_token\":{d},\"proposal_token\":{d},\"has_proposal\":{},\"proposal_match\":{},\"target_token\":{d},\"top5\":[[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}]],\"state_f16_fnv1a\":\"{x}\"}}\n",
                            .{ round, row_index, current_position + row_index, window[row_index], proposal, has_proposal, matched, top[0].id, top[0].id, top[0].logit, top[1].id, top[1].logit, top[2].id, top[2].logit, top[3].id, top[3].logit, top[4].id, top[4].logit, stateHash(verify_normed[row_index * hidden ..][0..hidden]) },
                        );
                    }
                }
                if (mtp_policy_dump) accepted_prefix = 0;
                if (target_ambiguity_margin) |threshold| {
                    var target_margin = std.math.inf(f32);
                    for (tops[0 .. accepted_prefix + 1]) |top| {
                        target_margin = @min(target_margin, top[0].logit - top[1].logit);
                    }
                    if (target_margin < threshold) {
                        const snapshot = request_prompt_snapshot orelse return error.InvalidArgument;
                        try snapshot.restore(backend, &runner, &decode, target_hidden_history.?, normed);
                        const confirmed_len = generated_len;
                        var replay_index: usize = 0;
                        while (replay_index < confirmed_len) : (replay_index += 1) {
                            const position = prompt_len + replay_index;
                            const input_token = generated[replay_index];
                            sequence[position] = input_token;
                            const serial_top = try targetStep(
                                &runner,
                                &decode,
                                &scratch_arena,
                                embedding,
                                input_token,
                                position,
                                state,
                                normed,
                                update,
                                threads,
                            );
                            @memcpy(target_hidden_history.?[position * hidden ..][0..hidden], normed);
                            if (replay_index + 1 < confirmed_len) {
                                if (serial_top[0].id != generated[replay_index + 1]) return error.ExactnessGuardMismatch;
                            } else {
                                generated[generated_len] = serial_top[0].id;
                                sequence[position + 1] = serial_top[0].id;
                                if (stream_tokens or !benchmark_quiet) {
                                    std.debug.print(
                                        "{{\"tensorfold\":\"qualcomm-exl3-speculative-chunk\",\"start\":{d},\"count\":1,\"tokens\":[{d},0,0,0,0,0,0]}}\n",
                                        .{ generated_len, serial_top[0].id },
                                    );
                                }
                                generated_len += 1;
                            }
                        }
                        while (generated_len < request_steps) {
                            const position = prompt_len + generated_len - 1;
                            const input_token = generated[generated_len - 1];
                            sequence[position] = input_token;
                            const serial_top = try targetStep(
                                &runner,
                                &decode,
                                &scratch_arena,
                                embedding,
                                input_token,
                                position,
                                state,
                                normed,
                                update,
                                threads,
                            );
                            @memcpy(target_hidden_history.?[position * hidden ..][0..hidden], normed);
                            generated[generated_len] = serial_top[0].id;
                            sequence[position + 1] = serial_top[0].id;
                            if (stream_tokens or !benchmark_quiet) {
                                std.debug.print(
                                    "{{\"tensorfold\":\"qualcomm-exl3-speculative-chunk\",\"start\":{d},\"count\":1,\"tokens\":[{d},0,0,0,0,0,0]}}\n",
                                    .{ generated_len, serial_top[0].id },
                                );
                            }
                            generated_len += 1;
                        }
                        serial_fallbacks += 1;
                        round += 1;
                        if (!benchmark_quiet) {
                            std.debug.print(
                                "{{\"tensorfold\":\"qualcomm-exl3-serial-fallback\",\"round\":{d},\"target_margin\":{d},\"threshold\":{d},\"confirmed_tokens\":{d},\"generated_tokens\":{d}}}\n",
                                .{ round - 1, target_margin, threshold, confirmed_len, generated_len },
                            );
                        }
                        break :speculative_rounds;
                    }
                }
                const landed_target_token = tops[accepted_prefix][0].id;
                const verify_compute_ms = runner.projection_ms - compute_start;
                const verify_projection_wall_ms = runner.projection_wall_ms - wall_start;
                const verify_projection_count = runner.projection_count - projection_start;
                const window_elapsed_ms = tf_qualcomm_monotonic_ms() - verify_started;
                var rollback_commit_ms: f64 = 0;
                var committed_rows = count;
                var committed_state_hash = stateHash(verify_normed[(count - 1) * hidden ..][0..hidden]);
                if (accepted_prefix != count - 1) {
                    const rollback_started = tf_qualcomm_monotonic_ms();
                    committed_rows = accepted_prefix + 1;
                    for (0..layers) |layer| {
                        if ((layer + 1) % 4 == 0) continue;
                        const layer_scratch = gdn_restore_scratch[layer * linear_state_elems ..][0..linear_state_elems];
                        const restore_status = if (runner.usesFusedLinearTail(layer))
                            tf_qualcomm_backend_linear_tail_gdn_state_restore_snapshot(backend, runner.fused_tail_slot[layer], count, count - committed_rows, layer_scratch.ptr, linear_state_elems)
                        else
                            tf_qualcomm_backend_gdn_state_restore_snapshot(backend, layer, count, count - committed_rows, layer_scratch.ptr, linear_state_elems);
                        if (restore_status != 0) {
                            std.debug.print("resident GDN snapshot restore failed: status {d}, layer {d}, rows {d}, tokens_back {d}\n", .{ restore_status, layer, count, count - committed_rows });
                            return error.BridgeFailed;
                        }
                    }
                    @memcpy(decode.conv, conv_snapshot);
                    for (0..layers) |layer| {
                        if ((layer + 1) % 4 == 0) continue;
                        const history = decode.convLayer(layer);
                        for (0..committed_rows) |row_index| {
                            std.mem.copyForwards(f32, history[0 * conv_channels .. 1 * conv_channels], history[1 * conv_channels .. 2 * conv_channels]);
                            std.mem.copyForwards(f32, history[1 * conv_channels .. 2 * conv_channels], history[2 * conv_channels .. 3 * conv_channels]);
                            const commit = conv_commits[layer * 7 * conv_channels + row_index * conv_channels ..][0..conv_channels];
                            @memcpy(history[2 * conv_channels .. 3 * conv_channels], commit);
                        }
                    }
                    committed_state_hash = stateHash(verify_normed[(committed_rows - 1) * hidden ..][0..hidden]);
                    rollback_commit_ms = tf_qualcomm_monotonic_ms() - rollback_started;
                }
                if (target_hidden_history) |history| {
                    for (0..committed_rows) |row_index| {
                        @memcpy(history[(current_position + row_index) * hidden ..][0..hidden], verify_normed[row_index * hidden ..][0..hidden]);
                    }
                }
                if (generated_len + accepted_prefix + 1 > generated.len) return error.InvalidArgument;
                const stream_start = generated_len;
                for (0..accepted_prefix) |proposal_index| {
                    const accepted_token = window[proposal_index + 1];
                    generated[generated_len] = accepted_token;
                    sequence[current_position + proposal_index + 1] = accepted_token;
                    generated_len += 1;
                }
                generated[generated_len] = landed_target_token;
                sequence[current_position + committed_rows] = landed_target_token;
                generated_len += 1;
                const stream_end = @min(generated_len, request_steps);
                if ((stream_tokens or !benchmark_quiet) and stream_start < stream_end) {
                    var stream_chunk: [7]usize = @splat(0);
                    const stream_count = stream_end - stream_start;
                    @memcpy(stream_chunk[0..stream_count], generated[stream_start..stream_end]);
                    std.debug.print("{{\"tensorfold\":\"qualcomm-exl3-speculative-chunk\",\"start\":{d},\"count\":{d},\"tokens\":[{d},{d},{d},{d},{d},{d},{d}]}}\n", .{ stream_start, stream_count, stream_chunk[0], stream_chunk[1], stream_chunk[2], stream_chunk[3], stream_chunk[4], stream_chunk[5], stream_chunk[6] });
                }
                total_accepted += accepted_prefix;
                pending_token = landed_target_token;
                current_position += committed_rows;
                const verify_elapsed = tf_qualcomm_monotonic_ms() - verify_started;
                if (!benchmark_quiet) {
                    std.debug.print(
                        "{{\"tensorfold\":\"qualcomm-exl3-verify-batch\",\"round\":{d},\"rows\":{d},\"draft_offered\":{d},\"accepted_prefix\":{d},\"landed_target_token\":{d},\"projections\":{d},\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"head_ms\":{d:.3},\"snapshot_ms\":{d:.3},\"window_elapsed_ms\":{d:.3},\"rollback_commit_rows\":{d},\"rollback_commit_ms\":{d:.3},\"committed_state_f16_fnv1a\":\"{x}\",\"elapsed_ms\":{d:.3},\"passed\":true}}\n",
                        .{ round, count, count - 1, accepted_prefix, landed_target_token, verify_projection_count, verify_compute_ms, verify_projection_wall_ms, verify_head_ms, snapshot_ms, window_elapsed_ms, committed_rows, rollback_commit_ms, committed_state_hash, verify_elapsed },
                    );
                }
                token_offset += count;
            }
            var token_hash: u64 = 14695981039346656037;
            const reported_generated_len = @min(generated_len, request_steps);
            for (generated[0..reported_generated_len]) |generated_token| {
                token_hash ^= @as(u64, generated_token);
                token_hash *%= 1099511628211;
            }
            std.debug.print(
                "{{\"tensorfold\":\"qualcomm-exl3-speculative-summary\",\"rounds\":{d},\"generated_tokens\":{d},\"accepted_drafts\":{d},\"adaptive_depth1_rounds\":{d},\"adaptive_depth2_rounds\":{d},\"serial_fallbacks\":{d},\"token_hash\":\"{x}\",\"mtp_prefill_ms\":{d:.3},\"mtp_projections\":{d},\"mtp_compute_ms\":{d:.3},\"speculative_elapsed_ms\":{d:.3},\"request_elapsed_ms\":{d:.3},\"passed\":true}}\n",
                .{ round, reported_generated_len, total_accepted, adaptive_depth1_rounds, adaptive_depth2_rounds, serial_fallbacks, token_hash, mtp_prefill_ms, mtp_runtime.projection_calls, mtp_runtime.projection_compute_ms, tf_qualcomm_monotonic_ms() - speculative_started, tf_qualcomm_monotonic_ms() - started },
            );
            if (worker_mode) {
                std.debug.print("{{\"tensorfold\":\"qualcomm-exl3-worker-done\",\"id\":{d},\"generated_tokens\":{d},\"resident_weight_uploads\":{d},\"resident_weight_hits\":{d},\"resident_head_uploads\":{d},\"resident_head_hits\":{d},\"mtp_projection_uploads\":{d}}}\n", .{ request_id, reported_generated_len, runner.resident_weight_uploads, runner.resident_weight_hits, runner.resident_head_uploads, runner.resident_head_hits, mtp_runtime.projection_uploads });
            }
            completed_requests += 1;
            continue;
        }

        const decode_end = if (stopped_on_eos) decode_start else positions;
        for (decode_start..decode_end) |position| {
            const input_token = sequence[position];
            const step_started = tf_qualcomm_monotonic_ms();
            const projection_start = runner.projection_count;
            const compute_start = runner.projection_ms;
            const wall_start = runner.projection_wall_ms;
            for (state, 0..) |*x, d| x.* = f16Value(embedding.bytes, input_token * hidden + d);
            runner.resident_slot_cursor = 0;
            for (0..layers) |layer| {
                _ = scratch_arena.reset(.retain_capacity);
                const input_norm = try tensorName(&name_buffer, "{s}.layers.{d}.input_layernorm.weight", .{ prefix, layer });
                try rmsNorm(&file, input_norm, state, normed, 1.0);
                if (norm_dump) |*dump| {
                    try dump.writeStreamingAll(init.io, std.mem.sliceAsBytes(normed));
                }
                if ((layer + 1) % 4 == 0) {
                    try fullAttention(&runner, &decode, layer, position, normed, update);
                } else {
                    try linearAttention(&runner, &decode, layer, normed, if (runner.usesFusedLinearTail(layer)) state else null, update);
                }
                if (runner.usesFusedLinearTail(layer) and (layer + 1) % 4 != 0) {
                    // Completed by linearAttention's fused persistent HTP graph.
                } else if (runner.use_fused_tail) {
                    var post_norm_buffer: [256]u8 = undefined;
                    var output_name_buffer: [256]u8 = undefined;
                    const post_norm = try tensorName(&post_norm_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
                    const output_name = if ((layer + 1) % 4 == 0)
                        try tensorName(&output_name_buffer, "{s}.layers.{d}.self_attn.o_proj", .{ prefix, layer })
                    else
                        try tensorName(&output_name_buffer, "{s}.layers.{d}.linear_attn.out_proj", .{ prefix, layer });
                    try runner.layerTailQuant(layer, output_name, post_norm, state, update);
                } else {
                    residualAdd(state, update);
                    const post_norm = try tensorName(&name_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
                    try rmsNorm(&file, post_norm, state, normed, 1.0);
                    if (norm_dump) |*dump| {
                        try dump.writeStreamingAll(init.io, std.mem.sliceAsBytes(normed));
                    }
                    try mlp(&runner, layer, normed, update);
                    residualAdd(state, update);
                }
                if (layer_dump) |*dump| {
                    try dump.writeStreamingAll(init.io, std.mem.sliceAsBytes(state));
                }
            }

            try rmsNorm(&file, prefix ++ ".norm.weight", state, normed, 1.0);
            if (hidden_dump) |*dump| {
                try dump.writeStreamingAll(init.io, std.mem.sliceAsBytes(normed));
                hidden_rows_dumped += 1;
            }
            const forced_next = position + 1 < prompt_len;
            if (forced_next and serve_mode) {
                skipped_heads += 1;
                const step_elapsed = tf_qualcomm_monotonic_ms() - step_started;
                if (!benchmark_quiet) {
                    std.debug.print(
                        "{{\"tensorfold\":\"qualcomm-exl3-forward-step\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"input_token\":{d},\"position\":{d},\"forced_next\":true,\"layers\":{d},\"projections\":{d},\"head\":\"skipped-teacher-forced\",\"state_f16_fnv1a\":\"{x}\",\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"head_ms\":0.000,\"elapsed_ms\":{d:.3},\"passed\":true}}\n",
                        .{ profile.name, args[1], input_token, position, layers, runner.projection_count - projection_start, stateHash(normed), runner.projection_ms - compute_start, runner.projection_wall_ms - wall_start, step_elapsed },
                    );
                }
                continue;
            }
            const head_started = tf_qualcomm_monotonic_ms();
            const top = try tiedHead(&runner, normed, threads);
            const head_ms = tf_qualcomm_monotonic_ms() - head_started;
            total_head_ms += head_ms;
            if (!forced_next) {
                sequence[position + 1] = top[0].id;
                generated_count += 1;
                if (first_token_ms == 0) first_token_ms = tf_qualcomm_monotonic_ms() - started;
                stopped_on_eos = corpus_mode and (top[0].id == 248046 or top[0].id == 248044);
                if (worker_mode and stream_tokens) {
                    std.debug.print(
                        "{{\"tensorfold\":\"qualcomm-exl3-speculative-chunk\",\"start\":{d},\"count\":1,\"tokens\":[{d},0,0,0,0,0,0]}}\n",
                        .{ generated_count - 1, top[0].id },
                    );
                }
            }
            const step_elapsed = tf_qualcomm_monotonic_ms() - step_started;
            if (!benchmark_quiet) {
                std.debug.print(
                    "{{\"tensorfold\":\"qualcomm-exl3-forward-step\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"input_token\":{d},\"position\":{d},\"forced_next\":{},\"layers\":{d},\"projections\":{d},\"head\":\"{s}\",\"next_token\":{d},\"top5\":[[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}]],\"state_f16_fnv1a\":\"{x}\",\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"head_ms\":{d:.3},\"elapsed_ms\":{d:.3},\"passed\":true}}\n",
                    .{ profile.name, args[1], input_token, position, forced_next, layers, runner.projection_count - projection_start, head_kind, top[0].id, top[0].id, top[0].logit, top[1].id, top[1].logit, top[2].id, top[2].logit, top[3].id, top[3].logit, top[4].id, top[4].logit, stateHash(normed), runner.projection_ms - compute_start, runner.projection_wall_ms - wall_start, head_ms, step_elapsed },
                );
            }
            if (stopped_on_eos) break;
        }
        const elapsed = tf_qualcomm_monotonic_ms() - started;
        const positions_evaluated = prompt_len + generated_count - 1;
        const decode_after_first_ms = @max(0.0, elapsed - first_token_ms);
        const decode_tokens_per_second = if (generated_count > 1 and decode_after_first_ms > 0)
            @as(f64, @floatFromInt(generated_count - 1)) * 1000.0 / decode_after_first_ms
        else
            0.0;
        const sample_id = if (corpus_mode) corpus_names[request_index] else "";
        std.debug.print("{{\"tensorfold\":\"qualcomm-exl3-generate\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"fast_activations\":{},\"prefill_mode\":\"{s}\",\"sample_id\":\"{s}\",\"request_index\":{d},\"requests\":{d},\"checkpoint_tensors\":{d},\"prompt_tokens\":{d},\"generated_tokens\":{d},\"requested_generated_tokens\":{d},\"stopped_on_eos\":{},\"positions\":{d},\"tokens\":[", .{ profile.name, args[1], runner.fast_activations, if (serial_corpus_prefill) "serial" else "batched", sample_id, request_index, request_count, file.count(), prompt_len, generated_count, request_steps, stopped_on_eos, positions_evaluated });
        for (sequence[0 .. prompt_len + generated_count], 0..) |generated_token, i| {
            if (i != 0) std.debug.print(",", .{});
            std.debug.print("{d}", .{generated_token});
        }
        std.debug.print(
            "],\"projections\":{d},\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"weight_convert_ms\":{d:.3},\"head_ms\":{d:.3},\"heads_skipped\":{d},\"dequant_ms\":{d:.3},\"cache_hash_ms\":{d:.3},\"cache_read_ms\":{d:.3},\"cache_write_ms\":{d:.3},\"cache_hits\":{d},\"cache_trusted_hits\":{d},\"cache_misses\":{d},\"cache_aliases_created\":{d},\"cache_bytes_read\":{d},\"cache_bytes_written\":{d},\"f16_cache_hits\":{d},\"f16_cache_misses\":{d},\"f16_cache_bytes_read\":{d},\"f16_cache_bytes_written\":{d},\"resident_weight_uploads\":{d},\"resident_weight_hits\":{d},\"resident_weight_fallbacks\":{d},\"resident_weight_bytes_uploaded\":{d},\"resident_head_uploads\":{d},\"resident_head_hits\":{d},\"resident_head_fallbacks\":{d},\"resident_head_compute_ms\":{d:.3}",
            .{ runner.projection_count, runner.projection_ms, runner.projection_wall_ms, runner.weight_convert_ms, total_head_ms, skipped_heads, runner.dequant_ms, runner.cache_hash_ms, runner.cache_read_ms, runner.cache_write_ms, runner.cache_hits, runner.cache_trusted_hits, runner.cache_misses, runner.cache_aliases_created, runner.cache_bytes_read, runner.cache_bytes_written, runner.f16_cache_hits, runner.f16_cache_misses, runner.f16_cache_bytes_read, runner.f16_cache_bytes_written, runner.resident_weight_uploads, runner.resident_weight_hits, runner.resident_weight_fallbacks, runner.resident_weight_bytes_uploaded, runner.resident_head_uploads, runner.resident_head_hits, runner.resident_head_fallbacks, runner.resident_head_compute_ms },
        );
        std.debug.print(
            ",\"resident_gdn_hits\":{d},\"resident_gdn_compute_ms\":{d:.3},\"resident_gdn_wall_ms\":{d:.3},\"pair_quant_calls\":{d},\"pair_quant_compute_ms\":{d:.3},\"pair_quant_wall_ms\":{d:.3},\"triple_quant_calls\":{d},\"triple_quant_compute_ms\":{d:.3},\"triple_quant_wall_ms\":{d:.3},\"layer_tail_calls\":{d},\"layer_tail_compute_ms\":{d:.3},\"layer_tail_wall_ms\":{d:.3},\"hidden_rows_dumped\":{d},\"ttft_ms\":{d:.3},\"decode_after_first_ms\":{d:.3},\"decode_tokens_per_second\":{d:.6},\"elapsed_ms\":{d:.3},\"positions_per_second\":{d:.6},\"end_to_end_generated_tokens_per_second\":{d:.6},\"peak_rss_kib\":{d},\"passed\":true}}\n",
            .{ runner.resident_gdn_hits, runner.resident_gdn_compute_ms, runner.resident_gdn_wall_ms, runner.pair_quant_calls, runner.pair_quant_compute_ms, runner.pair_quant_wall_ms, runner.triple_quant_calls, runner.triple_quant_compute_ms, runner.triple_quant_wall_ms, runner.layer_tail_calls, runner.layer_tail_compute_ms, runner.layer_tail_wall_ms, hidden_rows_dumped, first_token_ms, decode_after_first_ms, decode_tokens_per_second, elapsed, @as(f64, @floatFromInt(positions_evaluated)) * 1000.0 / elapsed, @as(f64, @floatFromInt(generated_count)) * 1000.0 / elapsed, tf_qualcomm_peak_rss_kib() },
        );
        if (worker_mode) {
            var token_hash: u64 = 14695981039346656037;
            for (sequence[prompt_len .. prompt_len + generated_count]) |generated_token| {
                token_hash ^= @as(u64, generated_token);
                token_hash *%= 1099511628211;
            }
            std.debug.print(
                "{{\"tensorfold\":\"qualcomm-exl3-speculative-summary\",\"rounds\":{d},\"generated_tokens\":{d},\"accepted_drafts\":0,\"adaptive_depth1_rounds\":0,\"adaptive_depth2_rounds\":0,\"serial_fallbacks\":0,\"token_hash\":\"{x}\",\"mtp_prefill_ms\":0,\"mtp_projections\":0,\"mtp_compute_ms\":0,\"speculative_elapsed_ms\":{d:.3},\"request_elapsed_ms\":{d:.3},\"passed\":true}}\n",
                .{ generated_count -| 1, generated_count, token_hash, decode_after_first_ms, elapsed },
            );
            std.debug.print(
                "{{\"tensorfold\":\"qualcomm-exl3-worker-done\",\"id\":{d},\"generated_tokens\":{d},\"resident_weight_uploads\":{d},\"resident_weight_hits\":{d},\"resident_head_uploads\":{d},\"resident_head_hits\":{d},\"mtp_projection_uploads\":0}}\n",
                .{ request_id, generated_count, runner.resident_weight_uploads, runner.resident_weight_hits, runner.resident_head_uploads, runner.resident_head_hits },
            );
        }
        completed_requests += 1;
    }
    return 0;
}
