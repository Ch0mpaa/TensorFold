//! Native FP32 CPU ops matching tools/tfold_cpu/ops.py.
//!
//! Accumulation order: left-to-right in FP64, cast to FP32 at store
//! time. Matches the Python reference's accumulation strategy, so
//! equivalent inputs produce bit-identical outputs.

const std = @import("std");

pub const OpError = error{
    ShapeMismatch,
    Empty,
    NonFiniteInput,
    NonFiniteOutput,
};

fn assertFinite1d(v: []const f32, label: []const u8) OpError!void {
    for (v) |x| {
        if (!std.math.isFinite(x)) {
            _ = label;
            return OpError.NonFiniteInput;
        }
    }
}

fn assertFinite2d(m: []const []const f32, label: []const u8) OpError!void {
    for (m) |row| try assertFinite1d(row, label);
}

/// y[n] = sum_k x[k] * W[k * N + n]
/// W stored row-major as [K, N] (K rows of N columns). x is [K]. y is [N].
///
/// Matches Python matvec(x, W, transpose_weight=False):
///   for n: acc = 0; for k: acc += x[k]*W[k][n]; y[n] = _as_fp32(acc)
pub fn matvecFp32(
    x: []const f32,
    w_flat: []const f32,
    K: usize,
    N: usize,
    out: []f32,
) OpError!void {
    if (K == 0 or N == 0) return OpError.Empty;
    if (x.len != K) return OpError.ShapeMismatch;
    if (w_flat.len != K * N) return OpError.ShapeMismatch;
    if (out.len != N) return OpError.ShapeMismatch;
    try assertFinite1d(x, "matvec.x");
    // Finite check on w: do it inline during the first touch
    for (w_flat) |v| {
        if (!std.math.isFinite(v)) return OpError.NonFiniteInput;
    }

    var n: usize = 0;
    while (n < N) : (n += 1) {
        var acc: f64 = 0.0;
        var k: usize = 0;
        while (k < K) : (k += 1) {
            acc += @as(f64, x[k]) * @as(f64, w_flat[k * N + n]);
        }
        const rounded: f32 = @floatCast(acc); // f64 -> f32 rounds to nearest-even per IEEE 754
        if (!std.math.isFinite(rounded)) return OpError.NonFiniteOutput;
        out[n] = rounded;
    }
}

/// out[i] = y[i] + b[i]; no broadcasting; length-mismatch is an error.
pub fn biasAddFp32(y: []const f32, b: []const f32, out: []f32) OpError!void {
    if (y.len == 0) return OpError.Empty;
    if (y.len != b.len or y.len != out.len) return OpError.ShapeMismatch;
    try assertFinite1d(y, "bias_add.y");
    try assertFinite1d(b, "bias_add.b");
    var i: usize = 0;
    while (i < y.len) : (i += 1) {
        const v: f32 = @floatCast(@as(f64, y[i]) + @as(f64, b[i]));
        if (!std.math.isFinite(v)) return OpError.NonFiniteOutput;
        out[i] = v;
    }
}

// Little-endian f32 decoder; the loader's byte region is guaranteed LE
// by the spec. We copy element-wise to avoid alignment assumptions.
pub fn decodeF32LE(bytes: []const u8, out: []f32) void {
    std.debug.assert(bytes.len == out.len * 4);
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        const raw = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        out[i] = @bitCast(raw);
    }
}
