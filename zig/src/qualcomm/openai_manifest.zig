const std = @import("std");
const build_options = @import("build_options");
const responses = @import("responses");
const Conn = @import("http_support").Conn;
const Allocator = std.mem.Allocator;

fn sha256File(a: Allocator, io: std.Io, path: []const u8) ?[]const u8 {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(256 << 20)) catch return null;
    defer a.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(digest, .lower)) catch null;
}

fn tokenizerSha256(a: Allocator, io: std.Io, dir: []const u8) ?[]const u8 {
    const names = [_][]const u8{
        "tokenizer.json",          "tokenizer.model", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json",      "merges.txt",
        "chat_template.jinja",
    };
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var found: usize = 0;
    for (names) |name| {
        const path = std.fs.path.join(a, &.{ dir, name }) catch return null;
        defer a.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(256 << 20)) catch continue;
        defer a.free(bytes);
        var header: [64]u8 = undefined;
        hash.update(std.fmt.bufPrint(&header, "{s}\x00{d}\x00", .{ name, bytes.len }) catch return null);
        hash.update(bytes);
        found += 1;
    }
    if (found == 0) return null;
    return a.dupe(u8, &std.fmt.bytesToHex(hash.finalResult(), .lower)) catch null;
}

pub fn send(conn: *Conn, a: Allocator, io: std.Io, model_path: []const u8, tokenizer_dir: []const u8, backend: []const u8) void {
    const model_dir = std.fs.path.dirname(model_path) orelse "";
    const config_path = std.fs.path.join(a, &.{ model_dir, "config.json" }) catch return;
    const config_sha = sha256File(a, io, config_path);
    const tokenizer_sha = tokenizerSha256(a, io, tokenizer_dir);
    const Gpu = struct { name: []const u8, memory_bytes: ?u64, cores: ?u64 };
    responses.sendJson(conn, a, 200, .{
        .schema_version = 1,
        .rank_count = 1,
        .runtime = .{
            .tensorfold_version = build_options.version,
            .backend = backend,
            .python_version = "none",
            .platform = "android",
        },
        .model = .{
            .revision = @as(?[]const u8, null),
            .config_sha256 = config_sha,
            .tokenizer_sha256 = tokenizer_sha,
            .family = "qwen3_5",
            .quantization = "exl3",
        },
        .hardware = .{
            .cpu_model = @as(?[]const u8, null),
            .system_memory_bytes = @as(?u64, null),
            .memory_type = "unified",
            .gpus = [_]Gpu{},
            .gpu_count = 0,
            .source = "partial",
        },
    });
}
