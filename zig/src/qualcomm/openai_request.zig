const std = @import("std");
const tokenizer_mod = @import("tokenizer");
const responses = @import("responses");
const Allocator = std.mem.Allocator;

fn renderChat(a: Allocator, value: std.json.Value) ![]u8 {
    if (value != .object) return error.BadRequest;
    const messages = value.object.get("messages") orelse return error.BadRequest;
    if (messages != .array) return error.BadRequest;
    var out: std.ArrayList(u8) = .empty;
    for (messages.array.items) |message| {
        if (message != .object) return error.BadRequest;
        const role_value = message.object.get("role") orelse return error.BadRequest;
        const content_value = message.object.get("content") orelse return error.BadRequest;
        if (role_value != .string or content_value != .string) return error.BadRequest;
        const role = role_value.string;
        if (!std.mem.eql(u8, role, "system") and !std.mem.eql(u8, role, "user") and
            !std.mem.eql(u8, role, "assistant") and !std.mem.eql(u8, role, "tool")) return error.BadRequest;
        try out.print(a, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ role, content_value.string });
    }
    var thinking = false;
    if (value.object.get("chat_template_kwargs")) |template_options| {
        if (template_options == .object) {
            if (template_options.object.get("enable_thinking")) |enabled| {
                if (enabled == .bool) thinking = enabled.bool;
            }
        }
    }
    try out.appendSlice(a, "<|im_start|>assistant\n");
    try out.appendSlice(a, if (thinking) "<think>\n" else "<think>\n\n</think>\n\n");
    return out.items;
}

pub fn encode(text: *tokenizer_mod.Tokenizer, a: Allocator, value: std.json.Value, chat: bool) ![]u32 {
    const prompt = if (chat)
        try renderChat(a, value)
    else blk: {
        if (value != .object) return error.BadRequest;
        const field = value.object.get("prompt") orelse return error.BadRequest;
        if (field != .string) return error.BadRequest;
        break :blk field.string;
    };
    return text.encode(a, prompt);
}

pub fn options(value: std.json.Value, maximum: usize) !struct { steps: usize, stream: bool } {
    if (value != .object) return error.BadRequest;
    var steps: usize = 256;
    if (value.object.get("max_tokens")) |field| {
        const parsed = responses.integer(field) orelse return error.BadRequest;
        if (parsed == 0 or parsed > maximum) return error.BadRequest;
        steps = @intCast(parsed);
    }
    var stream = false;
    if (value.object.get("stream")) |field| {
        if (field != .bool) return error.BadRequest;
        stream = field.bool;
    }
    if (value.object.get("temperature")) |field| {
        const temperature = responses.number(field) orelse return error.BadRequest;
        if (temperature != 0) return error.BadRequest;
    }
    return .{ .steps = steps, .stream = stream };
}

pub fn batteryTemperature(io: std.Io, a: Allocator) ?f64 {
    for ([_][]const u8{
        "/sys/class/power_supply/battery/temp",
        "/sys/class/power_supply/battery/batt_temp",
    }) |path| {
        const raw = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64)) catch continue;
        const value = std.fmt.parseInt(i32, std.mem.trim(u8, raw, " \r\n"), 10) catch continue;
        return @as(f64, @floatFromInt(value)) / 10.0;
    }
    return null;
}
