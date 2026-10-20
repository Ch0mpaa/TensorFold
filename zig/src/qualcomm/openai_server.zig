//! Phone-local OpenAI HTTP/SSE control plane for the persistent Qualcomm worker.
const std = @import("std");
const tokenizer_mod = @import("tokenizer");
const listener_mod = @import("http_support");
const Conn = listener_mod.Conn;
const model_admission = @import("model_admission");
const model_registry = @import("model_registry");
const hf_catalog = @import("hf_catalog");
const responses = @import("responses");
const request_mod = @import("request");
const manifest = @import("manifest");
const c = struct {
    const STDIN_FILENO = 0;
    const STDOUT_FILENO = 1;
    const STDERR_FILENO = 2;
    const SIGTERM = 15;
    const PR_SET_PDEATHSIG = 1;

    extern "c" fn pipe(fds: [*]c_int) c_int;
    extern "c" fn fork() c_int;
    extern "c" fn dup2(old_fd: c_int, new_fd: c_int) c_int;
    extern "c" fn close(fd: c_int) c_int;
    extern "c" fn read(fd: c_int, buffer: [*]u8, count: usize) isize;
    extern "c" fn write(fd: c_int, buffer: [*]const u8, count: usize) isize;
    extern "c" fn execv(path: [*:0]const u8, argv: [*c]const ?[*:0]u8) c_int;
    extern "c" fn kill(pid: c_int, signal: c_int) c_int;
    extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    extern "c" fn prctl(option: c_int, arg2: usize, arg3: usize, arg4: usize, arg5: usize) c_int;
    extern "c" fn _exit(status: c_int) noreturn;
};

const Allocator = std.mem.Allocator;
const Summary = responses.Summary;
const integer = responses.integer;
const number = responses.number;
const boolean = responses.boolean;
const sendJson = responses.sendJson;
const sendHtml = responses.sendHtml;
const sendWebManifest = responses.sendWebManifest;
const sendServiceWorker = responses.sendServiceWorker;
const startSse = responses.startSse;
const sse = responses.sse;
const sendChunk = responses.sendChunk;
const encodeRequest = request_mod.encode;
const batteryTemperature = request_mod.batteryTemperature;
const sendFinal = responses.sendFinal;
const max_prompt_tokens = 512;
const max_output_tokens = 1024;

const Config = struct {
    engine: []const u8,
    model: []const u8,
    cache: []const u8,
    model_sha: []const u8,
    mtp: []const u8,
    tokenizer: []const u8,
    lib_dir: []const u8,
    port: u16,
    backend: []const u8 = "htp",
    profile: []const u8 = "",
    power_mode: []const u8 = "low_balanced",
    ambiguity_margin: []const u8 = "0.125",
    exit_with_parent: bool = false,
    auth_token: ?[]const u8 = null,
};

const Worker = struct {
    allocator: Allocator,
    config: *Config,
    io: std.Io,
    pid: c_int = 0,
    input_fd: c_int = -1,
    output_fd: c_int = -1,
    read_buffer: [65536]u8 = undefined,
    read_start: usize = 0,
    read_end: usize = 0,
    next_id: u64 = 1,
    starts: u64 = 0,
    cancellations: u64 = 0,
    requests: u64 = 0,
    generated_tokens: u64 = 0,
    last_tokens_per_second: f64 = 0,
    last_weight_uploads: u64 = 0,
    last_mtp_uploads: u64 = 0,

    fn setEnv(a: Allocator, name: []const u8, value: []const u8) !void {
        const zname = try a.dupeSentinel(u8, name, 0);
        defer a.free(zname);
        const zvalue = try a.dupeSentinel(u8, value, 0);
        defer a.free(zvalue);
        if (c.setenv(zname.ptr, zvalue.ptr, 1) != 0) return error.EnvironmentFailed;
    }

    fn unsetEnv(a: Allocator, name: []const u8) !void {
        const zname = try a.dupeSentinel(u8, name, 0);
        defer a.free(zname);
        if (c.unsetenv(zname.ptr) != 0) return error.EnvironmentFailed;
    }

    fn configure(self: *Worker) !void {
        const a = self.allocator;
        try setEnv(a, "LD_LIBRARY_PATH", self.config.lib_dir);
        try setEnv(a, "ADSP_LIBRARY_PATH", self.config.lib_dir);
        try setEnv(a, "GGML_HEXAGON_OPPOLL", "1");
        try setEnv(a, "GGML_HEXAGON_HMX_PIPELINE_MIN_ROWS", "4");
        try setEnv(a, "GGML_HEXAGON_POWER_MODE", self.config.power_mode);
        if (std.mem.eql(u8, self.config.mtp, "none")) {
            try unsetEnv(a, "TENSORFOLD_MTP_SIDECAR");
            try unsetEnv(a, "TENSORFOLD_MTP_FIRST_MARGIN");
            try unsetEnv(a, "TENSORFOLD_TARGET_AMBIGUITY_MARGIN");
        } else {
            try setEnv(a, "TENSORFOLD_MTP_SIDECAR", self.config.mtp);
            try setEnv(a, "TENSORFOLD_MTP_DEPTH", "2");
            try setEnv(a, "TENSORFOLD_MTP_INITIAL_RANK", "1");
            try setEnv(a, "TENSORFOLD_MTP_FIRST_MARGIN", "0.1");
            if (std.mem.eql(u8, self.config.ambiguity_margin, "none")) {
                try unsetEnv(a, "TENSORFOLD_TARGET_AMBIGUITY_MARGIN");
            } else {
                try setEnv(a, "TENSORFOLD_TARGET_AMBIGUITY_MARGIN", self.config.ambiguity_margin);
            }
        }
        try setEnv(a, "TENSORFOLD_Q8_OUTPUT_LAYER_END", "24");
        if (std.mem.eql(u8, self.config.profile, "qwen3.5-35b-a3b")) try setEnv(a, "TENSORFOLD_EXPERT_CACHE_MIB", "512");
        try setEnv(a, "TENSORFOLD_FUSED_LINEAR_LAYER_MASK", "0");
        try setEnv(a, "TENSORFOLD_BENCH_QUIET", "1");
        try setEnv(a, "TENSORFOLD_STREAM_TOKENS", "1");
        try setEnv(a, "TENSORFOLD_PERSISTENT_WORKER", "1");
    }

    fn start(self: *Worker) !void {
        self.stop();
        try self.configure();
        var to_child: [2]c_int = undefined;
        var from_child: [2]c_int = undefined;
        if (c.pipe(&to_child) != 0) return error.PipeFailed;
        errdefer {
            _ = c.close(to_child[0]);
            _ = c.close(to_child[1]);
        }
        if (c.pipe(&from_child) != 0) return error.PipeFailed;
        errdefer {
            _ = c.close(from_child[0]);
            _ = c.close(from_child[1]);
        }
        const pid = c.fork();
        if (pid < 0) return error.ForkFailed;
        if (pid == 0) {
            _ = c.prctl(c.PR_SET_PDEATHSIG, c.SIGTERM, 0, 0, 0);
            _ = c.dup2(to_child[0], c.STDIN_FILENO);
            _ = c.dup2(from_child[1], c.STDOUT_FILENO);
            _ = c.dup2(from_child[1], c.STDERR_FILENO);
            _ = c.close(to_child[0]);
            _ = c.close(to_child[1]);
            _ = c.close(from_child[0]);
            _ = c.close(from_child[1]);
            const a = self.allocator;
            const values = [_][]const u8{
                self.config.engine,
                self.config.backend,
                self.config.model,
                "0",
                if (std.mem.eql(u8, self.config.profile, "qwen3.5-35b-a3b")) "16" else "6",
                self.config.cache,
                self.config.model_sha,
                "1024",
                "0",
                if (std.mem.eql(u8, self.config.profile, "qwen3.5-35b-a3b")) "serve-packed" else "serve-q4-full-out-q8-q0-head-adaptive-tail",
                "1",
            };
            var owned: [values.len][*:0]u8 = undefined;
            var argv: [values.len + 1]?[*:0]u8 = @splat(null);
            for (values, 0..) |value, i| {
                const encoded = a.dupeSentinel(u8, value, 0) catch c._exit(126);
                owned[i] = encoded.ptr;
                argv[i] = encoded.ptr;
            }
            _ = c.execv(owned[0], @ptrCast(&argv));
            c._exit(127);
        }
        _ = c.close(to_child[0]);
        _ = c.close(from_child[1]);
        self.pid = pid;
        self.input_fd = to_child[1];
        self.output_fd = from_child[0];
        self.read_start = 0;
        self.read_end = 0;
        while (true) {
            const line = try self.readLine(self.allocator);
            defer self.allocator.free(line);
            if (std.mem.indexOf(u8, line, "\"tensorfold\":\"qualcomm-exl3-worker-ready\"") != null) {
                self.starts += 1;
                return;
            }
            std.debug.print("worker startup: {s}\n", .{line});
        }
    }

    fn stop(self: *Worker) void {
        if (self.input_fd >= 0) _ = c.close(self.input_fd);
        if (self.output_fd >= 0) _ = c.close(self.output_fd);
        self.input_fd = -1;
        self.output_fd = -1;
        if (self.pid > 0) {
            _ = c.kill(self.pid, c.SIGTERM);
            var status: c_int = 0;
            _ = c.waitpid(self.pid, &status, 0);
        }
        self.pid = 0;
        self.read_start = 0;
        self.read_end = 0;
    }

    fn cancel(self: *Worker) void {
        self.cancellations += 1;
        self.stop();
    }

    fn ensure(self: *Worker) !void {
        if (self.pid <= 0) try self.start();
    }

    fn writeAll(self: *Worker, bytes: []const u8) !void {
        var sent: usize = 0;
        while (sent < bytes.len) {
            const rc = c.write(self.input_fd, bytes[sent..].ptr, bytes.len - sent);
            if (rc > 0) {
                sent += @intCast(rc);
                continue;
            }
            return error.WorkerClosed;
        }
    }

    fn readLine(self: *Worker, a: Allocator) ![]u8 {
        var line: std.ArrayList(u8) = .empty;
        errdefer line.deinit(a);
        while (true) {
            const available = self.read_buffer[self.read_start..self.read_end];
            if (std.mem.indexOfScalar(u8, available, '\n')) |index| {
                try line.appendSlice(a, available[0 .. index + 1]);
                self.read_start += index + 1;
                return line.toOwnedSlice(a);
            }
            if (available.len > 0) {
                try line.appendSlice(a, available);
                self.read_start = self.read_end;
            }
            if (line.items.len > 1 << 20) return error.WorkerLineTooLong;
            const rc = c.read(self.output_fd, &self.read_buffer, self.read_buffer.len);
            if (rc > 0) {
                self.read_start = 0;
                self.read_end = @intCast(rc);
                continue;
            }
            return error.WorkerClosed;
        }
    }

    fn request(self: *Worker, a: Allocator, conn: *Conn, text: *tokenizer_mod.Tokenizer, tokens: []const u32, steps: usize, chat: bool) !void {
        try self.ensure();
        const request_id = self.next_id;
        self.next_id += 1;
        var command: std.ArrayList(u8) = .empty;
        defer command.deinit(a);
        try command.print(a, "{d} {d} ", .{ request_id, steps });
        for (tokens, 0..) |token, i| {
            if (i != 0) try command.append(a, ',');
            try command.print(a, "{d}", .{token});
        }
        try command.append(a, '\n');
        try self.writeAll(command.items);

        var generated: std.ArrayList(u32) = .empty;
        var previous: []const u8 = "";
        var summary: Summary = .{};
        var done = false;
        while (!done) {
            if (conn.peerGone()) {
                self.cancel();
                return error.ClientGone;
            }
            const line = try self.readLine(a);
            if (line.len == 0 or line[0] != '{') continue;
            const parsed = std.json.parseFromSlice(std.json.Value, a, line, .{}) catch continue;
            const value = parsed.value;
            if (value != .object) continue;
            const kind_value = value.object.get("tensorfold") orelse continue;
            if (kind_value != .string) continue;
            const kind = kind_value.string;
            if (std.mem.eql(u8, kind, "qualcomm-exl3-prefill")) {
                summary.prefill_ms = number(value.object.get("elapsed_ms"));
                summary.prompt_cache_hit = boolean(value.object.get("cache_hit"));
                summary.prompt_cache_prefix_tokens = integer(value.object.get("cache_prefix_tokens"));
            } else if (std.mem.eql(u8, kind, "qualcomm-exl3-speculative-chunk")) {
                const count: usize = @intCast(integer(value.object.get("count")) orelse return error.BadWorkerEvent);
                const encoded = value.object.get("tokens") orelse return error.BadWorkerEvent;
                if (encoded != .array or count > encoded.array.items.len) return error.BadWorkerEvent;
                for (encoded.array.items[0..count]) |item| try generated.append(a, @intCast(integer(item) orelse return error.BadWorkerEvent));
                const full = try text.decode(a, generated.items, false);
                const piece = if (std.mem.startsWith(u8, full, previous)) full[previous.len..] else full;
                previous = full;
                sendChunk(conn, a, self.config.profile, chat, piece) catch {
                    self.cancel();
                    return error.ClientGone;
                };
            } else if (std.mem.eql(u8, kind, "qualcomm-exl3-speculative-summary")) {
                summary.decode_ms = number(value.object.get("speculative_elapsed_ms"));
                summary.request_ms = number(value.object.get("request_elapsed_ms"));
                summary.rounds = integer(value.object.get("rounds"));
                summary.accepted_drafts = integer(value.object.get("accepted_drafts"));
                summary.serial_fallbacks = integer(value.object.get("serial_fallbacks"));
            } else if (std.mem.eql(u8, kind, "qualcomm-exl3-worker-done")) {
                if (integer(value.object.get("id")) != request_id) return error.BadWorkerEvent;
                summary.weight_uploads = integer(value.object.get("resident_weight_uploads"));
                summary.mtp_uploads = integer(value.object.get("mtp_projection_uploads"));
                done = true;
            }
        }
        if (generated.items.len != steps) return error.IncompleteGeneration;
        self.requests += 1;
        self.generated_tokens += generated.items.len;
        const decode_seconds = (summary.decode_ms orelse 0) / 1000.0;
        self.last_tokens_per_second = if (decode_seconds > 0) @as(f64, @floatFromInt(generated.items.len)) / decode_seconds else 0;
        self.last_weight_uploads = summary.weight_uploads orelse 0;
        self.last_mtp_uploads = summary.mtp_uploads orelse 0;
        try sendFinal(conn, a, self.config.profile, !std.mem.eql(u8, self.config.mtp, "none"), chat, tokens.len, generated.items, summary);
    }
};

fn readBody(conn: *Conn, a: Allocator) ![]u8 {
    const encoded = conn.header("Content-Length") orelse return error.MissingContentLength;
    const length = try std.fmt.parseInt(usize, encoded, 10);
    if (length > 1 << 20) return error.BodyTooLarge;
    var body: std.ArrayList(u8) = .empty;
    if (try conn.readExact(a, &body, length) != length) return error.ShortBody;
    return body.items;
}

fn freeModelConfig(a: Allocator, config: Config) void {
    a.free(config.engine);
    a.free(config.model);
    a.free(config.cache);
    a.free(config.model_sha);
    a.free(config.mtp);
    a.free(config.tokenizer);
    a.free(config.profile);
}

fn freeServerConfig(a: Allocator, config: Config) void {
    freeModelConfig(a, config);
    a.free(config.lib_dir);
    a.free(config.backend);
    a.free(config.power_mode);
    a.free(config.ambiguity_margin);
    if (config.auth_token) |token| a.free(token);
}

fn switchModel(worker: *Worker, text: *tokenizer_mod.Tokenizer, id: []const u8) !void {
    const a = worker.allocator;
    var plan = try model_registry.prepare(a, worker.io, id);
    errdefer plan.deinit();
    var next_tokenizer = try tokenizer_mod.loadTokenizer(worker.io, a, plan.tokenizer);
    errdefer next_tokenizer.deinit();
    const old = worker.config.*;
    worker.config.engine = plan.engine;
    worker.config.model = plan.model;
    worker.config.cache = plan.cache;
    worker.config.model_sha = plan.model_sha;
    worker.config.mtp = plan.mtp;
    worker.config.tokenizer = plan.tokenizer;
    worker.config.profile = plan.profile;
    worker.start() catch |err| {
        worker.config.* = old;
        worker.start() catch {};
        return err;
    };
    model_registry.persist(worker.io, a, &plan, old.power_mode, old.ambiguity_margin) catch |err| {
        worker.stop();
        worker.config.* = old;
        worker.start() catch {};
        return err;
    };
    var previous_tokenizer = text.*;
    text.* = next_tokenizer;
    previous_tokenizer.deinit();
    freeModelConfig(a, old);
    a.free(plan.id);
    a.free(plan.model_dir);
    plan = undefined;
}

fn handlePost(worker: *Worker, text: *tokenizer_mod.Tokenizer, conn: *Conn, a: Allocator) void {
    const body = readBody(conn, a) catch return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "invalid request body" } });
    const parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "invalid JSON" } });
    const value = parsed.value;
    if (std.mem.eql(u8, conn.path, "/tokenize") or std.mem.eql(u8, conn.path, "/v1/tokenize")) {
        const ids = encodeRequest(text, a, value, value == .object and value.object.get("messages") != null) catch
            return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "unsupported prompt" } });
        return sendJson(conn, a, 200, .{ .tokens = ids });
    }
    if (std.mem.eql(u8, conn.path, "/detokenize") or std.mem.eql(u8, conn.path, "/v1/detokenize")) {
        if (value != .object) return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "tokens must be an array" } });
        const encoded = value.object.get("tokens") orelse return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "tokens must be an array" } });
        if (encoded != .array) return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "tokens must be an array" } });
        const ids = a.alloc(u32, encoded.array.items.len) catch return;
        for (encoded.array.items, ids) |item, *slot| slot.* = @intCast(integer(item) orelse return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "invalid token id" } }));
        const decoded = text.decode(a, ids, false) catch return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "decode failed" } });
        return sendJson(conn, a, 200, .{ .content = decoded });
    }
    if (std.mem.eql(u8, conn.path, "/v1/local/models/select")) {
        if (value != .object) return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "model selection must be an object" } });
        const id = value.object.get("id") orelse return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "model id is required" } });
        if (id != .string) return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "model id must be a string" } });
        switchModel(worker, text, id.string) catch |err| {
            return sendJson(conn, a, 409, .{ .@"error" = .{ .message = @errorName(err) } });
        };
        return sendJson(conn, a, 200, .{
            .status = "loaded",
            .id = id.string,
            .profile = worker.config.profile,
            .worker_pid = worker.pid,
        });
    }
    if (std.mem.eql(u8, conn.path, "/v1/local/models/pull")) {
        if (value != .object) return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "pull request must be an object" } });
        const repo = value.object.get("repo_id") orelse return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "repo_id is required" } });
        if (repo != .string) return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "repo_id must be a string" } });
        const id = hf_catalog.pull(a, worker.io, repo.string) catch |err| {
            return sendJson(conn, a, 409, .{ .@"error" = .{ .message = @errorName(err) } });
        };
        return sendJson(conn, a, 200, .{ .status = "ready", .id = id, .repo_id = repo.string });
    }
    const chat = std.mem.eql(u8, conn.path, "/v1/chat/completions");
    if (!chat and !std.mem.eql(u8, conn.path, "/v1/completions")) return sendJson(conn, a, 404, .{ .@"error" = .{ .message = "not found" } });
    const options = request_mod.options(value, max_output_tokens) catch return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "greedy stream=true request required" } });
    if (!options.stream) return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "stream=true required" } });
    const ids = encodeRequest(text, a, value, chat) catch return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "unsupported prompt" } });
    if (ids.len == 0 or ids.len > max_prompt_tokens) return sendJson(conn, a, 400, .{ .@"error" = .{ .message = "prompt must contain 1..512 tokens" } });
    if (batteryTemperature(worker.io, a)) |temperature| {
        if (temperature >= 43.0) {
            return sendJson(conn, a, 503, .{ .@"error" = .{
                .message = "device is cooling; retry below 43 C",
                .temperature_c = temperature,
            } });
        }
    }
    startSse(conn) catch return;
    worker.request(a, conn, text, ids, options.steps, chat) catch |err| switch (err) {
        error.ClientGone => {},
        else => {
            worker.cancel();
            sse(conn, a, .{ .@"error" = .{ .message = @errorName(err) } }) catch {};
            conn.writeAll("data: [DONE]\n\n") catch {};
        },
    };
}

fn handleConnection(worker: *Worker, text: *tokenizer_mod.Tokenizer, fd: std.posix.socket_t) void {
    defer _ = std.posix.system.close(fd);
    var conn = Conn.init(worker.allocator, fd, "127.0.0.1") catch return;
    defer conn.deinit();
    var arena = std.heap.ArenaAllocator.init(worker.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if ((conn.readRequest(a, false) catch return) != .ready) return;
    if (worker.config.auth_token) |token| {
        const authorization = if (conn.headerCount("Authorization") == 1) conn.header("Authorization") else null;
        const expected = std.fmt.allocPrint(a, "Bearer {s}", .{token}) catch return;
        if (authorization == null or !std.mem.eql(u8, authorization.?, expected)) {
            return sendJson(&conn, a, 401, .{ .@"error" = .{ .message = "local authorization required" } });
        }
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/v1/local/models")) {
        const models = model_registry.list(a, worker.io, worker.config.model) catch |err| {
            return sendJson(&conn, a, 500, .{ .@"error" = .{ .message = @errorName(err) } });
        };
        return sendJson(&conn, a, 200, .{ .selected = worker.config.profile, .models = models });
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/v1/local/catalog")) {
        const catalog = hf_catalog.list(a, worker.io) catch |err| {
            return sendJson(&conn, a, 500, .{ .@"error" = .{ .message = @errorName(err) } });
        };
        return sendJson(&conn, a, 200, .{ .models = catalog });
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/manifest.webmanifest")) {
        return sendWebManifest(&conn, a);
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/sw.js")) {
        return sendServiceWorker(&conn);
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/")) {
        return sendHtml(&conn);
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/health")) {
        return sendJson(&conn, a, 200, .{
            .status = "ok",
            .worker_pid = worker.pid,
            .backend = worker.config.backend,
            .owned_parent = worker.config.exit_with_parent,
            .auth_required = worker.config.auth_token != null,
            .power_mode = worker.config.power_mode,
            .target_ambiguity_margin = worker.config.ambiguity_margin,
            .temperature_c = batteryTemperature(worker.io, a),
            .requests = worker.requests,
            .cancellations = worker.cancellations,
        });
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/metrics")) {
        return sendJson(&conn, a, 200, .{
            .worker_starts = worker.starts,
            .cancellations = worker.cancellations,
            .requests = worker.requests,
            .generated_tokens = worker.generated_tokens,
            .last_tokens_per_second = worker.last_tokens_per_second,
            .last_resident_weight_uploads = worker.last_weight_uploads,
            .last_mtp_projection_uploads = worker.last_mtp_uploads,
        });
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/v1/benchmark/manifest")) {
        return manifest.send(&conn, a, worker.io, worker.config.model, worker.config.tokenizer, worker.config.backend);
    }
    if (std.mem.eql(u8, conn.method, "GET") and std.mem.eql(u8, conn.path, "/v1/models")) {
        return sendJson(&conn, a, 200, .{ .object = "list", .data = [_]struct { id: []const u8, object: []const u8 }{.{ .id = worker.config.profile, .object = "model" }} });
    }
    if (!std.mem.eql(u8, conn.method, "POST")) return sendJson(&conn, a, 404, .{ .@"error" = .{ .message = "not found" } });
    handlePost(worker, text, &conn, a);
}

fn validPowerMode(value: []const u8) bool {
    for ([_][]const u8{
        "low_power_saver", "power_saver",      "high_power_saver",           "low_balanced",
        "balanced",        "high_performance", "sustained_high_performance", "burst",
    }) |mode| if (std.mem.eql(u8, value, mode)) return true;
    return false;
}

fn validBackend(value: []const u8) bool {
    return std.mem.eql(u8, value, "htp") or
        std.mem.eql(u8, value, "cpu") or
        std.mem.eql(u8, value, "vulkan");
}

fn validAuthToken(value: []const u8) bool {
    if (value.len != 64) return false;
    for (value) |byte| if (!(std.ascii.isDigit(byte) or (byte >= 'a' and byte <= 'f'))) return false;
    return true;
}

fn parseConfig(a: Allocator, args: []const []const u8) !Config {
    if (args.len < 9 or args.len > 14) return error.InvalidArguments;
    const power_mode = if (args.len >= 10) args[9] else "low_balanced";
    if (!validPowerMode(power_mode)) return error.InvalidPowerMode;
    const ambiguity_margin = if (args.len >= 11) args[10] else "0.125";
    if (!std.mem.eql(u8, ambiguity_margin, "none")) {
        const threshold = std.fmt.parseFloat(f32, ambiguity_margin) catch return error.InvalidAmbiguityMargin;
        if (!std.math.isFinite(threshold) or threshold < 0) return error.InvalidAmbiguityMargin;
    }
    const backend = if (args.len >= 12) args[11] else "htp";
    if (!validBackend(backend)) return error.InvalidBackend;
    const exit_with_parent = if (args.len >= 13)
        std.mem.eql(u8, args[12], "owned")
    else
        false;
    if (args.len >= 13 and !exit_with_parent) return error.InvalidOwnershipMode;
    const auth_token = if (args.len == 14) args[13] else null;
    if (auth_token) |token| if (!validAuthToken(token)) return error.InvalidAuthToken;
    return .{
        .engine = try a.dupe(u8, args[1]),
        .model = try a.dupe(u8, args[2]),
        .cache = try a.dupe(u8, args[3]),
        .model_sha = try a.dupe(u8, args[4]),
        .mtp = try a.dupe(u8, args[5]),
        .tokenizer = try a.dupe(u8, args[6]),
        .lib_dir = try a.dupe(u8, args[7]),
        .port = try std.fmt.parseInt(u16, args[8], 10),
        .power_mode = try a.dupe(u8, power_mode),
        .ambiguity_margin = try a.dupe(u8, ambiguity_margin),
        .backend = try a.dupe(u8, backend),
        .exit_with_parent = exit_with_parent,
        .auth_token = if (auth_token) |token| try a.dupe(u8, token) else null,
    };
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 3 and std.mem.eql(u8, args[1], "inspect")) {
        const info = model_admission.inspect(init.io, init.gpa, args[2]) catch |err| {
            std.debug.print("{{\"supported\":false,\"error\":\"{s}\"}}\n", .{@errorName(err)});
            return 65;
        };
        std.debug.print(
            "{{\"supported\":true,\"profile\":\"{s}\",\"hidden_size\":{d},\"intermediate_size\":{d},\"num_hidden_layers\":{d},\"vocab_size\":{d}}}\n",
            .{ info.profile, info.hidden, info.intermediate, info.layers, info.vocab },
        );
        return 0;
    }
    var config = parseConfig(init.gpa, args) catch {
        std.debug.print("usage: {s} inspect MODEL_DIR | ENGINE MODEL CACHE MODEL_SHA MTP TOKENIZER_DIR LIB_DIR PORT [POWER_MODE [TARGET_AMBIGUITY_MARGIN|none [BACKEND [owned [AUTH_TOKEN]]]]]\n", .{args[0]});
        return 64;
    };
    if (config.exit_with_parent and c.prctl(c.PR_SET_PDEATHSIG, c.SIGTERM, 0, 0, 0) != 0)
        return error.ParentDeathSignalFailed;
    const info = model_admission.inspect(init.io, init.gpa, config.model) catch |err| {
        std.debug.print("{{\"tensorfold\":\"qualcomm-model-refused\",\"error\":\"{s}\"}}\n", .{@errorName(err)});
        return 65;
    };
    config.profile = try init.gpa.dupe(u8, info.profile);
    defer freeServerConfig(init.gpa, config);
    const engine_name = std.fs.path.basename(config.engine);
    const engine_matches = if (std.mem.eql(u8, info.profile, "qwen3.5-4b"))
        std.mem.indexOf(u8, engine_name, "4b") != null
    else if (std.mem.eql(u8, info.profile, "qwen3.5-35b-a3b"))
        std.mem.indexOf(u8, engine_name, "35b-a3b") != null or std.mem.indexOf(u8, engine_name, "moe") != null
    else
        std.mem.indexOf(u8, engine_name, "4b") == null and std.mem.indexOf(u8, engine_name, "35b-a3b") == null;
    if (!engine_matches) {
        std.debug.print("{{\"tensorfold\":\"qualcomm-model-refused\",\"error\":\"EngineProfileMismatch\",\"profile\":\"{s}\",\"engine\":\"{s}\"}}\n", .{ info.profile, engine_name });
        return 65;
    }
    var text = try tokenizer_mod.loadTokenizer(init.io, init.gpa, config.tokenizer);
    defer text.deinit();
    var worker = Worker{ .allocator = init.gpa, .config = &config, .io = init.io };
    defer worker.stop();
    try worker.start();
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", config.port);
    const listener = try listener_mod.Listener.open(address);
    defer listener.close();
    std.debug.print("{{\"tensorfold\":\"qualcomm-native-http-ready\",\"port\":{d},\"worker_pid\":{d}}}\n", .{ config.port, worker.pid });
    var stop = std.atomic.Value(bool).init(false);
    while (listener.accept(&stop)) |accepted| handleConnection(&worker, &text, accepted.fd);
    return 0;
}
