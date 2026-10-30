//! ModelOpt NVFP4 weights (e2m1 codes, an e4m3 scale every 16 inputs, a global scale) on the host: fp64 references.
const std = @import("std");
const types = @import("types.zig");
const fp8 = @import("fp8.zig");
const slice = @import("slice.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;
const refuse = types.refuse;

pub const id: []const u8 = "nvfp4";
pub const decoder: types.Decoder = .nvfp4;

/// A ModelOpt NVFP4 checkpoint: the table carries no per-tensor width, only an optional default input scale.
pub const Config = struct { input_scale: ?f32 = null };

/// NVFP4 is declared by `quantization.quant_algo == "NVFP4"`, by `quantization_config` naming modelopt + NVFP4,
/// or by a modelopt MIXED_PRECISION config whose `config_groups` carry at least one NVFP4 group (4-bit float, 16).
pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    _ = a;
    if (src.quantization.get("quant_algo")) |v| {
        if (v == .string and std.ascii.eqlIgnoreCase(v.string, "NVFP4")) return .{};
    }
    if (src.root.get("quantization_config")) |qc| if (qc == .object) {
        if (qc.object.get("quant_algo")) |algo| if (algo == .string) {
            if (std.ascii.eqlIgnoreCase(algo.string, "NVFP4")) return .{};
        };
        if (qc.object.get("quant_method")) |m| if (m == .string) {
            if (std.ascii.eqlIgnoreCase(m.string, "NVFP4")) return .{};
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
                const gs = weights.object.get("group_size") orelse continue;
                if (nb == .integer and nb.integer == 4 and ty == .string and
                    std.ascii.eqlIgnoreCase(ty.string, "float") and
                    gs == .integer and gs.integer == 16) return .{};
            }
        };
    };
    return null;
}

/// Inputs one e4m3 scale covers.
pub const group = 16;

/// An e2m1 nibble's value.
pub fn e2m1(n: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return (if (n & 8 != 0) @as(f64, -1) else 1) * mag[n & 7];
}

/// As stored: `weight` u8 [..., N, K / 2] (even columns low nibbles), `scale` e4m3 [..., N, K / 16], `global` a matrix.
pub const Host = struct { weight: Tensor, scale: Tensor, global: []const f32 };

/// Whether `key` holds an NVFP4 projection: `.weight` is u8 codes beside an e4m3 `.weight_scale`.
pub fn matches(t: anytype, key: []const u8) bool {
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    const w = t.get(join(&buf, &.{ key, ".weight" })) catch return false;
    if (w.dtype != .u8) return false;
    const s = t.get(join(&other, &.{ key, ".weight_scale" })) catch return false;
    return s.dtype == .f8_e4m3;
}

fn stackCount(t: Tensor) usize {
    if (t.rank < 2) return 1;
    var n: usize = 1;
    for (t.shape[0 .. t.rank - 2]) |d| n *= d;
    return n;
}

fn readGlobal(a: Allocator, t: anytype, key: []const u8, stack: usize) ![]const f32 {
    var buf: [256]u8 = undefined;
    inline for ([_][]const u8{ ".input_scale", ".weight_scale_2" }) |suffix| {
        const k = join(&buf, &.{ key, suffix });
        if (t.has(k)) {
            const tensor = try t.get(k);
            if (tensor.dtype != .f32 or tensor.bytes.len % 4 != 0) return refuse(key, "nvfp4 global scale must be fp32");
            const vals = try a.alloc(f32, stack);
            const count = tensor.bytes.len / 4;
            if (count == stack) {
                @memcpy(std.mem.sliceAsBytes(vals), tensor.bytes);
            } else if (count == 1) {
                const one: f32 = @bitCast(std.mem.readInt(u32, tensor.bytes[0..4], .little));
                @memset(vals, one);
            } else return refuse(key, "nvfp4 global scale length does not match the stack");
            return vals;
        }
    }
    const vals = try a.alloc(f32, stack);
    @memset(vals, 1.0);
    return vals;
}

/// `_packed`: NVFP4 codes [..., N, K/2], e4m3 scales [..., N, K/16] and a per-stack fp32 global scale.
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    const weight = try t.get(join(&buf, &.{ key, ".weight" }));
    const scale = try t.get(join(&other, &.{ key, ".weight_scale" }));
    if (weight.dtype != .u8 or scale.dtype != .f8_e4m3) return refuse(key, "nvfp4 expects u8 codes and e4m3 scales");
    if (weight.rank < 2 or weight.rank != scale.rank) return refuse(key, "nvfp4 weight and scale ranks differ");
    const global = try readGlobal(a, t, key, stackCount(weight));
    std.log.info("nvfp4.read: {s} w={d} s={d} g={d} stack={d}", .{ key, weight.bytes.len, scale.bytes.len, global.len, stackCount(weight) });
    return .{ .weight = weight, .scale = scale, .global = global };
}

/// Output rows N of a projection (a stack's expert count is the leading dim).
pub fn rows(h: Host) usize {
    return h.weight.shape[h.weight.rank - 2];
}

/// The bytes of its three tensors.
pub fn bytes(h: Host) usize {
    return h.weight.bytes.len + h.scale.bytes.len + h.global.len * @sizeOf(f32);
}

/// `_halves`: a fused [embedding | hidden] projection split in two along K; codes and scale table alike.
pub fn halves(a: Allocator, h: Host) ![2]Host {
    const rank = h.weight.rank;
    const code_cols = h.weight.shape[rank - 1]; // K / 2 bytes
    const scale_cols = h.scale.shape[rank - 1]; // K / 16 scales
    if (code_cols % 2 != 0 or scale_cols % 2 != 0) return refuse("nvfp4 halves", "K / 2 is not even");
    const half_codes = code_cols / 2;
    const half_scales = scale_cols / 2;
    var out: [2]Host = undefined;
    for (&out, 0..) |*o, i| {
        const w = if (i == 0) [2]usize{ 0, half_codes } else [2]usize{ half_codes, code_cols };
        const s = if (i == 0) [2]usize{ 0, half_scales } else [2]usize{ half_scales, scale_cols };
        o.* = .{
            .weight = try takeCols(a, h.weight, w[0], w[1]),
            .scale = try takeCols(a, h.scale, s[0], s[1]),
            .global = try a.dupe(f32, h.global),
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

/// `_rows`: output rows `spans` of a projection, the share of a column-split one.
pub fn sliceRows(a: Allocator, h: Host, spans: []const slice.Span) slice.Error!Host {
    return .{
        .weight = try slice.takeRows(a, h.weight, spans),
        .scale = try slice.takeRows(a, h.scale, spans),
        .global = try a.dupe(f32, h.global),
    };
}

/// `_cols`: one rank's whole input groups of a row-split projection; weight cut at nibble, scale at group boundaries.
pub fn sliceCols(a: Allocator, h: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    const scale_cols = h.scale.shape[h.scale.rank - 1];
    const groups = try slice.even(scale_cols, r.world, what);
    const code_cols = h.weight.shape[h.weight.rank - 1];
    _ = try slice.even(code_cols, r.world, what);
    const codes = code_cols / r.world;
    return .{
        .weight = try takeCols(a, h.weight, r.rank * codes, (r.rank + 1) * codes),
        .scale = try takeCols(a, h.scale, r.rank * groups, (r.rank + 1) * groups),
        .global = try a.dupe(f32, h.global),
    };
}

/// `_stack`: a rank's `part` experts of an E + 1 stack (shared last), and the shared one on rank 0.
pub fn sliceStack(a: Allocator, h: Host, r: slice.Rank, part: usize) slice.Error!Host {
    return .{
        .weight = try slice.takeExperts(a, h.weight, r, part),
        .scale = try slice.takeExperts(a, h.scale, r, part),
        .global = try takeGlobal(a, h.global, r, part),
    };
}

fn takeGlobal(a: Allocator, g: []const f32, r: slice.Rank, part: usize) slice.Error![]const f32 {
    const total = g.len - 1;
    const mine = g[r.rank * part ..][0..part];
    if (r.rank != 0) return a.dupe(f32, mine);
    const buf = try a.alloc(f32, part + 1);
    @memcpy(buf[0..part], mine);
    buf[part] = g[total];
    return buf;
}

/// An uploaded NVFP4 projection: the three device addresses, N and K per expert, and the stack's leading count.
pub const Device = struct { weight: u64, scale: u64, global: u64, n: u32, k: u32, stack: u32 };

/// Place every tensor on the device: codes, e4m3 scales and the fp32 global scales (one per expert of the stack).
pub fn upload(u: types.Uploader, h: Host) !Device {
    const w = try u.tensor(h.weight);
    const s = try u.tensor(h.scale);
    const g_tensor: Tensor = .{
        .dtype = .f32,
        .rank = 1,
        .shape = .{ h.global.len, 1, 1, 1, 1 },
        .bytes = std.mem.sliceAsBytes(h.global),
    };
    const g = try u.tensor(g_tensor);
    const rank = h.weight.rank;
    const n: u32 = @intCast(h.weight.shape[rank - 2]);
    const k: u32 = @intCast(h.weight.shape[rank - 1] * 2);
    const stack: u32 = @intCast(stackCount(h.weight));
    return .{ .weight = w.ptr, .scale = s.ptr, .global = g.ptr, .n = n, .k = k, .stack = stack };
}

/// What the kernels read of an NVFP4 projection: the three addresses and the per-expert N and K.
pub const Matrix = struct { weight: u64, scale: u64, global: u64, n: u32, k: u32 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

/// A single projection or one expert's shape at the stack's addresses (stacked=true).
pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    _ = stacked;
    const matrix: Matrix = .{ .weight = d.weight, .scale = d.scale, .global = d.global, .n = d.n, .k = d.k };
    return .{ .n = d.n, .k = d.k, .matrix = matrix };
}

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    return h.weight.numel() * 2;
}

/// fp64 dequantization of every row of `h` (a stack's leading dimensions folded), code * scale * global, into `out`.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len or h.scale.rank != rank or h.weight.dtype != .u8 or h.scale.dtype != .f8_e4m3) return error.UnexpectedTensor;
    const n = h.weight.shape[rank - 2];
    const k = h.weight.shape[rank - 1] * 2;
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    for (0..rank - 2) |d| if (h.weight.shape[d] == 0 or h.weight.shape[d] != h.scale.shape[d]) return error.UnexpectedTensor;
    if (h.scale.shape[rank - 2] != n or h.scale.shape[rank - 1] * group != k or out.len != elements(h)) return error.UnexpectedTensor;
    const stack = elements(h) / (n * k);
    if (h.global.len != stack) return error.UnexpectedTensor;
    for (0..stack) |e| for (0..n) |r| for (0..k) |c| {
        const row = e * n + r;
        const byte = h.weight.bytes[row * (k / 2) + c / 2];
        const nib: u8 = if (c & 1 == 0) byte & 0xF else byte >> 4;
        out[row * k + c] = e2m1(nib) * fp8.e4m3(h.scale.bytes[row * (k / group) + c / group]) * h.global[e];
    };
}

test "an NVFP4 stack against hand-decoded values" {
    // two experts of [1, 32]: two scale groups a row; byte 0 holds columns 0 (low) and 1 (high)
    var codes: [2 * 16]u8 = @splat(0);
    codes[0] = 0xF2; // column 0: e2m1 2 -> 1.0; column 1: e2m1 0xF -> -6
    codes[8] = 0x07; // column 16: 6
    codes[16 + 15] = 0x30; // expert 1, column 31: e2m1 3 -> 1.5
    const scales = [_]u8{ 0x38, 0x40, 0x38, 0x30 }; // 1, 2 | 1, 0.5
    const global = [_]f32{ 0.25, 2 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 16, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 2, 1, 2, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [64]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 0.25), out[0]);
    try std.testing.expectEqual(@as(f64, -1.5), out[1]);
    try std.testing.expectEqual(@as(f64, 3), out[16]);
    try std.testing.expectEqual(@as(f64, 1.5), out[32 + 31]);
    try std.testing.expectEqual(@as(f64, 0), out[2]);
}

test "a reference refuses mismatched expert stacks" {
    const codes: [16]u8 = @splat(0x22);
    const scales: [2]u8 = @splat(0x38);
    const global = [_]f32{ 1, 1 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 8, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [32]f64 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &out));
}
