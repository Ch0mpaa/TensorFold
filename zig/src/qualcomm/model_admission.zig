const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Info = struct {
    profile: []const u8,
    hidden: u64,
    intermediate: u64,
    layers: u64,
    vocab: u64,
};

fn requiredInt(object: std.json.ObjectMap, key: []const u8) !u64 {
    const value = object.get(key) orelse return error.MissingModelField;
    if (value != .integer or value.integer < 0) return error.InvalidModelField;
    return @intCast(value.integer);
}

pub fn inspect(io: std.Io, a: Allocator, model_path: []const u8) !Info {
    const dir = if (std.mem.endsWith(u8, model_path, ".safetensors"))
        std.fs.path.dirname(model_path) orelse return error.InvalidModelPath
    else
        model_path;
    const config_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(config_path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(16 << 20));
    defer a.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidModelConfig;
    const root = parsed.value.object;
    const model_type = root.get("model_type") orelse return error.MissingModelField;
    if (model_type != .string or !std.mem.eql(u8, model_type.string, "qwen3_5")) return error.UnsupportedArchitecture;
    const quant = root.get("quantization_config") orelse return error.MissingQuantization;
    if (quant != .object) return error.MissingQuantization;
    const method = quant.object.get("quant_method") orelse return error.MissingQuantization;
    if (method != .string or !std.mem.eql(u8, method.string, "exl3")) return error.UnsupportedQuantization;
    const text = root.get("text_config") orelse return error.MissingModelField;
    if (text != .object) return error.InvalidModelField;
    const hidden = try requiredInt(text.object, "hidden_size");
    const intermediate = try requiredInt(text.object, "intermediate_size");
    const layers = try requiredInt(text.object, "num_hidden_layers");
    const vocab = try requiredInt(text.object, "vocab_size");
    const profile = if (hidden == 2560 and intermediate == 9216 and layers == 32 and vocab == 248320)
        "qwen3.5-4b"
    else if (hidden == 2048 and intermediate == 6144 and layers == 24 and vocab == 248320)
        "qwen3.5-2b"
    else
        return error.UnsupportedQwenGeometry;
    return .{ .profile = profile, .hidden = hidden, .intermediate = intermediate, .layers = layers, .vocab = vocab };
}
