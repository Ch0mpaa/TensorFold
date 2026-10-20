const std = @import("std");
const Conn = @import("http_support").Conn;
const Allocator = std.mem.Allocator;

pub const Summary = struct {
    prefill_ms: ?f64 = null,
    decode_ms: ?f64 = null,
    request_ms: ?f64 = null,
    prompt_cache_hit: bool = false,
    prompt_cache_prefix_tokens: ?u64 = null,
    rounds: ?u64 = null,
    accepted_drafts: ?u64 = null,
    serial_fallbacks: ?u64 = null,
    weight_uploads: ?u64 = null,
    mtp_uploads: ?u64 = null,
};

pub fn integer(value: ?std.json.Value) ?u64 {
    const v = value orelse return null;
    return switch (v) {
        .integer => |number_value| if (number_value >= 0) @intCast(number_value) else null,
        else => null,
    };
}

pub fn number(value: ?std.json.Value) ?f64 {
    const v = value orelse return null;
    return switch (v) {
        .integer => |number_value| @floatFromInt(number_value),
        .float => |number_value| number_value,
        else => null,
    };
}

pub fn boolean(value: ?std.json.Value) bool {
    const v = value orelse return false;
    return v == .bool and v.bool;
}

fn jsonBytes(a: Allocator, value: anytype) ![]u8 {
    return std.json.Stringify.valueAlloc(a, value, .{});
}

pub fn sendJson(conn: *Conn, a: Allocator, code: u16, value: anytype) void {
    const body = jsonBytes(a, value) catch return;
    conn.startResponse(code, null) catch return;
    conn.addHeader("Content-Type", "application/json") catch return;
    var length: [32]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&length, "{d}", .{body.len}) catch return) catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}

const ui_html = @embedFile("openai_ui.html");

pub fn sendHtml(conn: *Conn) void {
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    conn.addHeader("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'") catch return;
    var length: [32]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&length, "{d}", .{ui_html.len}) catch return) catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish(ui_html) catch {};
}

pub fn sendWebManifest(conn: *Conn, a: Allocator) void {
    const body = jsonBytes(a, .{
        .name = "TensorFold Local",
        .short_name = "TensorFold",
        .start_url = "/",
        .scope = "/",
        .display = "standalone",
        .background_color = "#100d18",
        .theme_color = "#161224",
    }) catch return;
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "application/manifest+json") catch return;
    var length: [32]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&length, "{d}", .{body.len}) catch return) catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}

pub fn sendServiceWorker(conn: *Conn) void {
    const body = "self.addEventListener('install',()=>self.skipWaiting());self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));";
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/javascript; charset=utf-8") catch return;
    conn.addHeader("Service-Worker-Allowed", "/") catch return;
    var length: [32]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&length, "{d}", .{body.len}) catch return) catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}

pub fn startSse(conn: *Conn) !void {
    try conn.startResponse(200, null);
    try conn.addHeader("Content-Type", "text/event-stream");
    try conn.addHeader("Cache-Control", "no-cache");
    try conn.addHeader("Connection", "close");
    try conn.finish("");
}

pub fn sse(conn: *Conn, a: Allocator, value: anytype) !void {
    const payload = try jsonBytes(a, value);
    const frame = try std.fmt.allocPrint(a, "data: {s}\n\n", .{payload});
    try conn.writeAll(frame);
}

pub fn sendChunk(conn: *Conn, a: Allocator, model: []const u8, chat: bool, piece: []const u8) !void {
    if (chat) {
        try sse(conn, a, .{
            .id = "tf-phone",
            .object = "chat.completion.chunk",
            .model = model,
            .choices = [_]struct { index: u8, delta: struct { content: []const u8 }, finish_reason: ?[]const u8 }{
                .{ .index = 0, .delta = .{ .content = piece }, .finish_reason = null },
            },
        });
    } else {
        try sse(conn, a, .{
            .id = "tf-phone",
            .object = "text_completion",
            .model = model,
            .choices = [_]struct { index: u8, text: []const u8, finish_reason: ?[]const u8 }{
                .{ .index = 0, .text = piece, .finish_reason = null },
            },
        });
    }
}

fn tokenSha(tokens: []const u32) [64]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [16]u8 = undefined;
    for (tokens, 0..) |token, i| {
        if (i != 0) hash.update(",");
        hash.update(std.fmt.bufPrint(&buffer, "{d}", .{token}) catch unreachable);
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn sendFinal(conn: *Conn, a: Allocator, model: []const u8, drafted: bool, chat: bool, prompt_tokens: usize, generated: []const u32, summary: Summary) !void {
    const hash = tokenSha(generated);
    const decode_seconds = (summary.decode_ms orelse 0) / 1000.0;
    const runtime = .{
        .engine = "tensorfold-qualcomm-exl3-native",
        .sampling = "greedy",
        .drafts = drafted,
        .prefill_seconds = (summary.prefill_ms orelse 0) / 1000.0,
        .prompt_cache_hit = summary.prompt_cache_hit,
        .prompt_cache_prefix_tokens = summary.prompt_cache_prefix_tokens orelse 0,
        .decode_seconds = decode_seconds,
        .tokens_per_second = if (decode_seconds > 0) @as(f64, @floatFromInt(generated.len)) / decode_seconds else 0,
        .token_sha = hash[0..],
        .mtp_depth = if (drafted) @as(u8, 2) else 0,
        .mtp_initial_rank = if (drafted) @as(u8, 1) else 0,
        .rounds = summary.rounds orelse 0,
        .accepted_drafts = summary.accepted_drafts orelse 0,
        .serial_fallbacks = summary.serial_fallbacks orelse 0,
        .request_seconds = (summary.request_ms orelse 0) / 1000.0,
        .resident_weight_uploads = summary.weight_uploads orelse 0,
        .mtp_projection_uploads = summary.mtp_uploads orelse 0,
    };
    const usage = .{ .prompt_tokens = prompt_tokens, .completion_tokens = generated.len, .total_tokens = prompt_tokens + generated.len };
    if (chat) {
        try sse(conn, a, .{
            .id = "tf-phone",
            .object = "chat.completion.chunk",
            .model = model,
            .choices = [_]struct { index: u8, delta: struct {}, finish_reason: []const u8 }{
                .{ .index = 0, .delta = .{}, .finish_reason = "length" },
            },
            .usage = usage,
            .runtime = runtime,
        });
    } else {
        try sse(conn, a, .{
            .id = "tf-phone",
            .object = "text_completion",
            .model = model,
            .choices = [_]struct { index: u8, text: []const u8, finish_reason: []const u8 }{
                .{ .index = 0, .text = "", .finish_reason = "length" },
            },
            .usage = usage,
            .runtime = runtime,
        });
    }
    try conn.writeAll("data: [DONE]\n\n");
}
