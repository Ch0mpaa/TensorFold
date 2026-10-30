//! NVIDIA ModelOpt FP8 (weight-only) on the host: per-tensor scalar scale, fp64 reference.
//!
//! Current scope: per-tensor FP8 as emitted by NVIDIA's ModelOpt mixed-precision export
//! (e.g. `nvidia/Qwen3.8-27B-NVFP4`'s attention linears). The projection reads as e4m3
//! codes `[..., N, K]`, one fp32 `weight_scale` (rank 0) and one fp32 `input_scale`
//! (rank 0, defaults to 1.0). Dequantization is one scalar multiplier per tensor:
//! `fp(n, k) = e4m3(weight[n, k]) * weight_scale * input_scale`.
//!
//! The 2D block-scaled variant (one fp32 per 128x128 tile) is NOT supported yet — any
//! `weight_scale` with more than one element is refused at `read` time. Add that path
//! here when a mixed-precision checkpoint needs it.

const std = @import("std");
const types = @import("types.zig");
const fp8 = @import("fp8.zig");
const slice = @import("slice.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;
const refuse = types.refuse;

pub const id: []const u8 = "fp8";
pub const decoder: types.Decoder = .fp8;

/// A ModelOpt FP8 checkpoint: an optional default input scale (per tensor; 1.0 for weight-only).
pub const Config = struct { input_scale: ?f32 = null };

/// FP8 is declared by `quantization.quant_algo == "FP8"`, by `quantization_config` naming FP8 or ModelOpt FP8,
/// or by a ModelOpt MIXED_PRECISION config whose `config_groups` carry at least one FP8 group (8-bit float).
pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    _ = a;
    if (src.quantization.get("quant_algo")) |v| {
        if (v == .string and std.ascii.eqlIgnoreCase(v.string, "FP8")) return .{};
    }
    if (src.root.get("quantization_config")) |qc| if (qc == .object) {
        if (qc.object.get("quant_algo")) |algo| if (algo == .string) {
            if (std.ascii.eqlIgnoreCase(algo.string, "FP8")) return .{};
        };
        if (qc.object.get("quant_method")) |m| if (m == .string) {
            if (std.ascii.eqlIgnoreCase(m.string, "fp8")) return .{};
        };
        const method = qc.object.get("quant_method");
        const is_modelopt = method != null and method.? == .string and std.ascii.eqlIgnoreCase(method.?.string, "modelopt");
        if (is_modelopt) if (qc.object.get("config_groups")) |groups| if (groups == .object) {
            var it = groups.object.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.* != .object) continue;
                const weights = entry.value_ptr.object.get("weights") orelse continue;
                if (weights != .object) continue;
                const nb = weights.object.get("num_bits") orelse continue;
                const ty = weights.object.get("type") orelse continue;
                if (nb == .integer and nb.integer == 8 and ty == .string and
                    std.ascii.eqlIgnoreCase(ty.string, "float")) return .{};
            }
        };
    };
    return null;
}

/// As stored on this path: `weight` e4m3 [..., N, K]; `scale` is the pre-folded scalar
/// product `weight_scale * input_scale` carried on `.input_scale` (no separate device tensor).
pub const Host = struct { weight: Tensor, scale: Tensor, input_scale: f32 = 1.0 };

/// Whether `key` holds a per-tensor FP8 projection: `.weight` is e4m3 (or u8 bytes) beside an fp32 `.weight_scale`.
pub fn matches(t: anytype, key: []const u8) bool {
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    const w = t.get(join(&buf, &.{ key, ".weight" })) catch return false;
    if (w.dtype != .f8_e4m3 and w.dtype != .u8) return false;
    const s = t.get(join(&other, &.{ key, ".weight_scale" })) catch return false;
    return s.dtype == .f32;
}

fn stackCount(t: Tensor) usize {
    if (t.rank < 2) return 1;
    var n: usize = 1;
    for (t.shape[0 .. t.rank - 2]) |d| n *= d;
    return n;
}

fn readScalarF32(t: Tensor) !f32 {
    if (t.dtype != .f32 or t.bytes.len < 4) return error.UnexpectedTensor;
    return @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
}

fn readOptionalScalarF32(t: anytype, key: []const u8, suffix: []const u8, default_value: f32) !f32 {
    var buf: [256]u8 = undefined;
    const k = join(&buf, &.{ key, suffix });
    if (!t.has(k)) return default_value;
    const tensor = try t.get(k);
    return readScalarF32(tensor) catch return refuse(key, "fp8 scalar must be fp32");
}

/// Reads an FP8 projection with a single per-tensor scalar. Folds `weight_scale * input_scale`
/// into `Host.input_scale` and keeps `Host.scale` as the raw scalar tensor for provenance.
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    _ = a;
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    const weight = try t.get(join(&buf, &.{ key, ".weight" }));
    const scale = try t.get(join(&other, &.{ key, ".weight_scale" }));
    if (weight.dtype != .f8_e4m3 and weight.dtype != .u8) return refuse(key, "fp8 expects e4m3 codes");
    if (scale.dtype != .f32) return refuse(key, "fp8 expects fp32 weight_scale");
    if (weight.rank < 2) return refuse(key, "fp8 weight must be rank-2 or higher");
    // Only the per-tensor scalar variant is supported today. 2D block-scaled checkpoints
    // (scale shape [N/128, K/128]) must come through a later block-aware kernel path.
    if (scale.numel() != 1) return refuse(key, "fp8 block-scaled scales not yet supported on this backend");
    const w_scale = try readScalarF32(scale);
    const i_scale = try readOptionalScalarF32(t, key, ".input_scale", 1.0);
    return .{ .weight = weight, .scale = scale, .input_scale = w_scale * i_scale };
}

/// Output rows N of a projection (a stack's expert count is the leading dim).
pub fn rows(h: Host) usize {
    return h.weight.shape[h.weight.rank - 2];
}

/// The bytes of its weight tensor (scalar scale lives in the Device struct itself).
pub fn bytes(h: Host) usize {
    return h.weight.bytes.len + h.scale.bytes.len;
}

/// `_halves`: a fused [embedding | hidden] projection split in two along K.
pub fn halves(a: Allocator, h: Host) ![2]Host {
    const rank = h.weight.rank;
    const code_cols = h.weight.shape[rank - 1];
    if (code_cols % 2 != 0) return refuse("fp8 halves", "K is not even in codes");
    const half_codes = code_cols / 2;
    var out: [2]Host = undefined;
    for (&out, 0..) |*o, i| {
        const w = if (i == 0) [2]usize{ 0, half_codes } else [2]usize{ half_codes, code_cols };
        o.* = .{
            .weight = try takeCols(a, h.weight, w[0], w[1]),
            .scale = h.scale,
            .input_scale = h.input_scale,
        };
    }
    return out;
}

/// `takeCols` across the last dim for any rank, byte for byte.
fn takeCols(a: Allocator, t: Tensor, from: usize, to: usize) Allocator.Error!Tensor {
    const size = t.dtype.size();
    const last = t.shape[t.rank - 1];
    const stride = last * size;
    const width = (to - from) * size;
    var outer: usize = 1;
    for (t.shape[0 .. t.rank - 1]) |d| outer *= d;
    const buf = try a.alloc(u8, outer * width);
    for (0..outer) |r| @memcpy(buf[r * width ..][0..width], t.bytes[r * stride + from * size ..][0..width]);
    var out = t;
    out.shape[t.rank - 1] = to - from;
    out.bytes = buf;
    return out;
}

/// `_rows`: output rows `spans` of a projection; scalar scale rides unchanged.
pub fn sliceRows(a: Allocator, h: Host, spans: []const slice.Span) slice.Error!Host {
    return .{
        .weight = try slice.takeRows(a, h.weight, spans),
        .scale = h.scale,
        .input_scale = h.input_scale,
    };
}

/// `_cols`: one rank's whole input groups of a row-split projection; scalar scale rides unchanged.
pub fn sliceCols(a: Allocator, h: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    const code_cols = h.weight.shape[h.weight.rank - 1];
    _ = try slice.even(code_cols, r.world, what);
    const codes = code_cols / r.world;
    return .{
        .weight = try takeCols(a, h.weight, r.rank * codes, (r.rank + 1) * codes),
        .scale = h.scale,
        .input_scale = h.input_scale,
    };
}

/// `_stack`: a rank's `part` experts of an E + 1 stack; scalar scale rides unchanged.
pub fn sliceStack(a: Allocator, h: Host, r: slice.Rank, part: usize) slice.Error!Host {
    return .{
        .weight = try slice.takeExperts(a, h.weight, r, part),
        .scale = h.scale,
        .input_scale = h.input_scale,
    };
}

/// An uploaded FP8 projection: weight device address, folded scalar in `input_scale`, N and K per expert.
pub const Device = struct { weight: u64, scale: u64, input_scale: f32, n: u32, k: u32, stack: u32 };

/// Place the weight on the device. The scalar scale is NOT uploaded; it lives in `input_scale`.
pub fn upload(u: types.Uploader, h: Host) !Device {
    const w = try u.tensor(h.weight);
    const rank = h.weight.rank;
    const n: u32 = @intCast(h.weight.shape[rank - 2]);
    const k: u32 = @intCast(h.weight.shape[rank - 1]);
    const stack: u32 = @intCast(stackCount(h.weight));
    return .{ .weight = w.ptr, .scale = 0, .input_scale = h.input_scale, .n = n, .k = k, .stack = stack };
}

/// What the kernels read of an FP8 projection: the weight address, the per-tensor folded scale, per-expert N and K.
pub const Matrix = struct { weight: u64, scale: u64, input_scale: f32, n: u32, k: u32 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

/// A single projection or one expert's shape at the stack's addresses (stacked=true).
pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    _ = stacked;
    const matrix: Matrix = .{ .weight = d.weight, .scale = d.scale, .input_scale = d.input_scale, .n = d.n, .k = d.k };
    return .{ .n = d.n, .k = d.k, .matrix = matrix };
}

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    return h.weight.numel();
}

/// fp64 dequantization of every row of `h` (per-tensor scalar form): `e4m3(w) * input_scale`.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len) return error.UnexpectedTensor;
    if (h.weight.dtype != .f8_e4m3 and h.weight.dtype != .u8) return error.UnexpectedTensor;
    if (out.len != h.weight.numel()) return error.UnexpectedTensor;
    const inp: f64 = @floatCast(h.input_scale);
    for (0..out.len) |i| out[i] = fp8.e4m3(h.weight.bytes[i]) * inp;
}

test "FP8 per-tensor reference dequantization" {
    const codes = [_]u8{ 0x38, 0x40, 0xc0, 0x44 }; // 1.0, 2.0, -2.0, 2.5
    const scale = [_]f32{0.5};
    const h: Host = .{
        .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 2, 2, 1, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f32, .rank = 0, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&scale) },
        .input_scale = 0.5 * 2.0, // weight_scale * input_scale folded
    };
    var out: [4]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 1.0 * 1.0), out[0]);
    try std.testing.expectEqual(@as(f64, 2.0 * 1.0), out[1]);
    try std.testing.expectEqual(@as(f64, -2.0 * 1.0), out[2]);
    try std.testing.expectEqual(@as(f64, 2.5 * 1.0), out[3]);
}
