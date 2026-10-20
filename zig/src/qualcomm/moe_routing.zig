const std = @import("std");

fn quantizeF16(x: f32) f32 {
    const value: f16 = @floatCast(x);
    return @floatCast(value);
}

pub fn normalizeTopK(logits: []f32) !void {
    if (logits.len == 0) return error.EmptyRoutingSet;
    const maximum = logits[0];
    var denominator: f32 = 0;
    for (logits) |*logit| {
        logit.* = @exp(logit.* - maximum);
        denominator += logit.*;
    }
    if (!std.math.isFinite(denominator) or denominator <= 0) return error.InvalidRoutingWeights;
    for (logits) |*weight| weight.* = quantizeF16(weight.* / denominator);
}

test "normalization keeps the original maximum for every expert" {
    var logits = [_]f32{ 4.0, 4.0 - @log(2.0), 4.0 - @log(4.0) };
    try normalizeTopK(&logits);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 7.0), logits[0], 0.0005);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 7.0), logits[1], 0.0005);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 7.0), logits[2], 0.0005);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[0] + logits[1] + logits[2], 0.001);
}

test "normalization rejects an empty route set" {
    try std.testing.expectError(error.EmptyRoutingSet, normalizeTopK(&.{}));
}
