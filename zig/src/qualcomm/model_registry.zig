const std = @import("std");
const admission = @import("model_admission");
const Allocator = std.mem.Allocator;

pub const model_root = "/data/local/tmp/tensorfold-models";
pub const profile_path = "/data/local/tmp/tensorfold-local/profile.conf";
const product_root = "/data/local/tmp/exl3-full-power";
const fallback_tokenizer = model_root ++ "/Qwen3.5-2B-exl3-8.00bpw";
const mtp_4b = "/data/local/tmp/tensorfold-mtp-poc/mtp.safetensors";
const sha_4b = "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1";
const sha_2b = "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f";

pub const Entry = struct {
    id: []const u8,
    path: []const u8,
    profile: ?[]const u8,
    selected: bool,
    ready: bool,
    drafted: bool,
    weights_bytes: ?u64,
    error_name: ?[]const u8,
};

pub const Plan = struct {
    allocator: Allocator,
    id: []const u8,
    model_dir: []const u8,
    engine: []const u8,
    model: []const u8,
    cache: []const u8,
    model_sha: []const u8,
    mtp: []const u8,
    tokenizer: []const u8,
    profile: []const u8,

    pub fn deinit(plan: *Plan) void {
        inline for (.{ "id", "model_dir", "engine", "model", "cache", "model_sha", "mtp", "tokenizer", "profile" }) |field| {
            plan.allocator.free(@field(plan, field));
        }
        plan.* = undefined;
    }
};

fn validId(id: []const u8) bool {
    if (id.len == 0 or std.mem.eql(u8, id, ".") or std.mem.eql(u8, id, "..")) return false;
    for (id) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '_' or ch == '-')) return false;
    return true;
}

fn fileSize(io: std.Io, path: []const u8) ?u64 {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch return null;
    return if (stat.kind == .file) stat.size else null;
}

fn isDir(io: std.Io, path: []const u8) bool {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch return false;
    dir.close(io);
    return true;
}

fn readHashSidecar(a: Allocator, io: std.Io, model_dir: []const u8) ?[]const u8 {
    const path = std.fs.path.join(a, &.{ model_dir, ".tensorfold.sha256" }) catch return null;
    defer a.free(path);
    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(128)) catch return null;
    const value = std.mem.trim(u8, raw, " \r\n");
    if (value.len != 64) {
        a.free(raw);
        return null;
    }
    for (value) |ch| if (!(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'))) {
        a.free(raw);
        return null;
    };
    const out = a.dupe(u8, value) catch {
        a.free(raw);
        return null;
    };
    a.free(raw);
    return out;
}

pub fn hashModel(a: Allocator, io: std.Io, model_dir: []const u8, model_path: []const u8) ![]const u8 {
    if (readHashSidecar(a, io, model_dir)) |cached| return cached;
    const file = try std.Io.Dir.cwd().openFile(io, model_path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.MissingModelWeights;
    const chunk = try a.alloc(u8, 1 << 20);
    defer a.free(chunk);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var offset: u64 = 0;
    while (offset < stat.size) {
        const count = try file.readPositional(io, &.{chunk}, offset);
        if (count == 0) return error.ShortModelRead;
        hash.update(chunk[0..count]);
        offset += count;
    }
    const hex = try a.dupe(u8, &std.fmt.bytesToHex(hash.finalResult(), .lower));
    const sidecar = try std.fs.path.join(a, &.{ model_dir, ".tensorfold.sha256" });
    defer a.free(sidecar);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = sidecar, .data = hex }) catch {};
    return hex;
}

fn cachePath(a: Allocator, sha: []const u8) ![]const u8 {
    if (std.mem.eql(u8, sha, sha_4b)) return a.dupe(u8, "/data/local/tmp/tensorfold-exl3-cache-4b-v1");
    if (std.mem.eql(u8, sha, sha_2b)) return a.dupe(u8, "/data/local/tmp/tensorfold-exl3-cache-v1");
    return std.fmt.allocPrint(a, "/data/local/tmp/tensorfold-exl3-cache-{s}", .{sha[0..16]});
}

pub fn cacheReady(a: Allocator, io: std.Io, sha: []const u8) bool {
    const cache = cachePath(a, sha) catch return false;
    defer a.free(cache);
    return isDir(io, cache);
}

fn enginePath(a: Allocator, profile: []const u8) ![]const u8 {
    if (std.mem.eql(u8, profile, "qwen3.5-4b")) return a.dupe(u8, product_root ++ "/tensorfold-qualcomm-exl3-forward-4b");
    if (std.mem.eql(u8, profile, "qwen3.5-2b")) return a.dupe(u8, product_root ++ "/tensorfold-qualcomm-exl3-forward-2b");
    return error.UnsupportedQwenGeometry;
}

fn entryReady(a: Allocator, io: std.Io, model_dir: []const u8, info: admission.Info, selected: bool) bool {
    if (selected) return true;
    const sha = readHashSidecar(a, io, model_dir) orelse return false;
    defer a.free(sha);
    const cache = cachePath(a, sha) catch return false;
    defer a.free(cache);
    const engine = enginePath(a, info.profile) catch return false;
    defer a.free(engine);
    if (fileSize(io, engine) == null or !isDir(io, cache)) return false;
    return !std.mem.eql(u8, info.profile, "qwen3.5-4b") or fileSize(io, mtp_4b) != null;
}

fn before(_: void, left: Entry, right: Entry) bool {
    return std.mem.lessThan(u8, left.id, right.id);
}

pub fn list(a: Allocator, io: std.Io, selected_model: []const u8) ![]Entry {
    var root = try std.Io.Dir.cwd().openDir(io, model_root, .{ .iterate = true });
    defer root.close(io);
    var entries: std.ArrayList(Entry) = .empty;
    var iterator = root.iterate();
    while (try iterator.next(io)) |item| {
        if (item.kind != .directory or !validId(item.name)) continue;
        const id = try a.dupe(u8, item.name);
        const dir = try std.fs.path.join(a, &.{ model_root, id });
        const model = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
        const weights_bytes = fileSize(io, model);
        const selected = std.mem.eql(u8, model, selected_model);
        const info = admission.inspect(io, a, dir) catch |err| {
            try entries.append(a, .{
                .id = id,
                .path = dir,
                .profile = null,
                .selected = selected,
                .ready = false,
                .drafted = false,
                .weights_bytes = weights_bytes,
                .error_name = @errorName(err),
            });
            continue;
        };
        try entries.append(a, .{
            .id = id,
            .path = dir,
            .profile = info.profile,
            .selected = selected,
            .ready = weights_bytes != null and entryReady(a, io, dir, info, selected),
            .drafted = std.mem.eql(u8, info.profile, "qwen3.5-4b"),
            .weights_bytes = weights_bytes,
            .error_name = if (weights_bytes == null) "MissingModelWeights" else null,
        });
    }
    std.mem.sort(Entry, entries.items, {}, before);
    return entries.items;
}

pub fn prepare(a: Allocator, io: std.Io, id: []const u8) !Plan {
    if (!validId(id)) return error.InvalidModelId;
    const model_dir = try std.fs.path.join(a, &.{ model_root, id });
    errdefer a.free(model_dir);
    const info = try admission.inspect(io, a, model_dir);
    const model = try std.fs.path.join(a, &.{ model_dir, "model.safetensors" });
    errdefer a.free(model);
    if (fileSize(io, model) == null) return error.MissingModelWeights;
    const model_sha = try hashModel(a, io, model_dir, model);
    errdefer a.free(model_sha);
    const cache = try cachePath(a, model_sha);
    errdefer a.free(cache);
    if (!isDir(io, cache)) return error.DecodedCacheMissing;
    const engine = try enginePath(a, info.profile);
    errdefer a.free(engine);
    if (fileSize(io, engine) == null) return error.MissingEngine;
    const tokenizer_json = try std.fs.path.join(a, &.{ model_dir, "tokenizer.json" });
    defer a.free(tokenizer_json);
    const tokenizer = if (fileSize(io, tokenizer_json) != null) try a.dupe(u8, model_dir) else try a.dupe(u8, fallback_tokenizer);
    errdefer a.free(tokenizer);
    const drafted = std.mem.eql(u8, info.profile, "qwen3.5-4b");
    if (drafted and fileSize(io, mtp_4b) == null) return error.MissingDrafter;
    return .{
        .allocator = a,
        .id = try a.dupe(u8, id),
        .model_dir = model_dir,
        .engine = engine,
        .model = model,
        .cache = cache,
        .model_sha = model_sha,
        .mtp = try a.dupe(u8, if (drafted) mtp_4b else "none"),
        .tokenizer = tokenizer,
        .profile = try a.dupe(u8, info.profile),
    };
}

pub fn persist(io: std.Io, a: Allocator, plan: *const Plan, power_mode: []const u8, ambiguity_margin: []const u8) !void {
    const body = try std.fmt.allocPrint(a, "MODEL_DIR='{s}'\nENGINE='{s}'\nMTP='{s}'\nTOKENIZER_DIR='{s}'\nMODEL_SHA='{s}'\nCACHE='{s}'\nPOWER_MODE='{s}'\nAMBIGUITY_MARGIN='{s}'\n", .{ plan.model_dir, plan.engine, plan.mtp, plan.tokenizer, plan.model_sha, plan.cache, power_mode, ambiguity_margin });
    defer a.free(body);
    const temporary = profile_path ++ ".tmp";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = temporary, .data = body });
    try std.Io.Dir.renameAbsolute(temporary, profile_path, io);
}
