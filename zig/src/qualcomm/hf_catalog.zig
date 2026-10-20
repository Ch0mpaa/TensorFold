const std = @import("std");
const admission = @import("model_admission");
const registry = @import("model_registry");
const Allocator = std.mem.Allocator;

const curl = "/data/data/com.termux/files/usr/bin/curl";
const reserve_bytes: u64 = 512 << 20;

const c = struct {
    extern "c" fn fork() c_int;
    extern "c" fn execv(path: [*:0]const u8, argv: [*c]const ?[*:0]u8) c_int;
    extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
    extern "c" fn _exit(status: c_int) noreturn;
};

const Count = u64;
const Vfs = extern struct {
    bsize: u64,
    frsize: u64,
    blocks: Count,
    bfree: Count,
    bavail: Count,
    files: Count,
    ffree: Count,
    favail: Count,
    fsid: u64,
    flag: u64,
    namemax: u64,
    tail: [24]u8,
};
extern "c" fn statvfs(path: [*:0]const u8, result: *Vfs) c_int;

const Known = struct {
    repo_id: []const u8,
    local_id: []const u8,
    profile: []const u8,
    sha256: []const u8,
    bytes: u64,
};

const known = [_]Known{
    .{
        .repo_id = "UnstableLlama/Qwen3.5-2B-exl3-8.00bpw",
        .local_id = "Qwen3.5-2B-exl3-8.00bpw",
        .profile = "qwen3.5-2b",
        .sha256 = "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f",
        .bytes = 3_438_956_902,
    },
    .{
        .repo_id = "UnstableLlama/Qwen3.5-4B-exl3-4.00bpw",
        .local_id = "Qwen3.5-4B-exl3-4.00bpw",
        .profile = "qwen3.5-4b",
        .sha256 = "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1",
        .bytes = 4_212_050_770,
    },
};

pub const Item = struct {
    repo_id: []const u8,
    local_id: []const u8,
    profile: []const u8,
    download_bytes: u64,
    available_bytes: ?u64,
    installed: bool,
    pullable: bool,
    reason: ?[]const u8,
};

fn availableBytes() ?u64 {
    var stat: Vfs = undefined;
    if (statvfs("/data/local/tmp", &stat) != 0) return null;
    return std.math.mul(u64, stat.frsize, stat.bavail) catch null;
}

fn fileExists(io: std.Io, path: []const u8) bool {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return stat.kind == .file;
}

pub fn list(a: Allocator, io: std.Io) ![]Item {
    const available = availableBytes();
    const out = try a.alloc(Item, known.len);
    for (known, out) |model, *item| {
        const dir = try std.fs.path.join(a, &.{ registry.model_root, model.local_id });
        const weights = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
        const installed = fileExists(io, weights);
        const cache_ready = registry.cacheReady(a, io, model.sha256);
        const fits = installed or available == null or available.? >= model.bytes + reserve_bytes;
        item.* = .{
            .repo_id = model.repo_id,
            .local_id = model.local_id,
            .profile = model.profile,
            .download_bytes = model.bytes,
            .available_bytes = available,
            .installed = installed,
            .pullable = cache_ready and fits,
            .reason = if (!cache_ready) "DecodedCacheMissing" else if (!fits) "InsufficientStorage" else null,
        };
    }
    return out;
}

fn find(repo_id: []const u8) ?Known {
    for (known) |model| if (std.mem.eql(u8, model.repo_id, repo_id)) return model;
    return null;
}

fn runCurl(a: Allocator, argv_text: []const []const u8) !void {
    const argv = try a.alloc(?[*:0]u8, argv_text.len + 1);
    defer a.free(argv);
    var encoded = try a.alloc([:0]u8, argv_text.len);
    defer a.free(encoded);
    for (argv_text, 0..) |value, index| {
        encoded[index] = try a.dupeSentinel(u8, value, 0);
        argv[index] = encoded[index].ptr;
    }
    defer for (encoded) |value| a.free(value);
    argv[argv_text.len] = null;
    const pid = c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        _ = c.execv(encoded[0].ptr, @ptrCast(argv.ptr));
        c._exit(127);
    }
    var status: c_int = 0;
    if (c.waitpid(pid, &status, 0) != pid) return error.DownloadWaitFailed;
    if ((status & 0x7f) != 0 or ((status >> 8) & 0xff) != 0) return error.DownloadFailed;
}

fn download(a: Allocator, io: std.Io, model: Known, dir: []const u8, name: []const u8, required: bool) !void {
    const final = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(final);
    if (fileExists(io, final)) return;
    const partial = try std.fmt.allocPrint(a, "{s}.part", .{final});
    defer a.free(partial);
    const url = try std.fmt.allocPrint(a, "https://huggingface.co/{s}/resolve/main/{s}?download=true", .{ model.repo_id, name });
    defer a.free(url);
    runCurl(a, &.{ curl, "--location", "--fail", "--show-error", "--silent", "--retry", "3", "--continue-at", "-", "--proto", "=https", "--proto-redir", "=https", "--output", partial, url }) catch |err| {
        if (!required) {
            std.Io.Dir.cwd().deleteFile(io, partial) catch {};
            return;
        }
        return err;
    };
    try std.Io.Dir.renameAbsolute(partial, final, io);
}

pub fn pull(a: Allocator, io: std.Io, repo_id: []const u8) ![]const u8 {
    const model = find(repo_id) orelse return error.UnqualifiedRepository;
    if (!registry.cacheReady(a, io, model.sha256)) return error.DecodedCacheMissing;
    const available = availableBytes();
    if (available != null and available.? < model.bytes + reserve_bytes) return error.InsufficientStorage;
    const dir = try std.fs.path.join(a, &.{ registry.model_root, model.local_id });
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .fromMode(0o755));
    try download(a, io, model, dir, "config.json", true);
    const info = try admission.inspect(io, a, dir);
    if (!std.mem.eql(u8, info.profile, model.profile)) return error.CatalogProfileMismatch;
    for ([_][]const u8{
        "tokenizer.json",      "tokenizer_config.json",  "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json", "README.md",  "LICENSE",
    }) |name| {
        try download(a, io, model, dir, name, false);
    }
    try download(a, io, model, dir, "model.safetensors", true);
    const weights = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
    const hash = try registry.hashModel(a, io, dir, weights);
    if (!std.mem.eql(u8, hash, model.sha256)) {
        std.Io.Dir.cwd().deleteFile(io, weights) catch {};
        return error.CheckpointHashMismatch;
    }
    var plan = try registry.prepare(a, io, model.local_id);
    plan.deinit();
    return a.dupe(u8, model.local_id);
}
