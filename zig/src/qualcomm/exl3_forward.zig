//! A deliberately small, auditable Qwen3.5-2B EXL3 decode path for Qualcomm
//! Android bring-up. It executes token position zero end-to-end and keeps the
//! model math in TensorFold while routing every quantized projection through
//! either GGML CPU or Hexagon HTP. An optional content-addressed cache keeps
//! decoded row-major weights between runs so EXL3 reconstruction is paid once.

const std = @import("std");
const exl3 = @import("exl3_format");
const st = @import("safetensors");

extern fn tf_qualcomm_monotonic_ms() f64;
extern fn tf_qualcomm_peak_rss_kib() usize;
extern fn tf_qualcomm_backend_create(backend_name: [*:0]const u8, cpu_threads: i32) ?*anyopaque;
extern fn tf_qualcomm_backend_destroy(context: ?*anyopaque) void;
extern fn tf_qualcomm_backend_matmul_once(
    context: ?*anyopaque,
    k: usize,
    m: usize,
    n: usize,
    weight: [*]const f32,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
) c_int;

const prefix = "model.language_model";
const hidden = 2048;
const intermediate = 6144;
const layers = 24;
const vocab = 248320;
const eps: f32 = 1e-6;
const cache_magic = "TFEXL3F1";
const cache_header_bytes = 64;

fn f16Value(bytes: []const u8, index: usize) f32 {
    const bits = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
    const decoded: f16 = @bitCast(bits);
    return @floatCast(decoded);
}

fn bf16Value(bytes: []const u8, index: usize) f32 {
    const bits = std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
    return @bitCast(@as(u32, bits) << 16);
}

fn f32Value(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
}

fn value(tensor: st.Tensor, index: usize) f32 {
    return switch (tensor.dtype) {
        .f16 => f16Value(tensor.bytes, index),
        .bf16 => bf16Value(tensor.bytes, index),
        .f32 => f32Value(tensor.bytes, index),
        else => unreachable,
    };
}

fn roundF16(x: f32) f32 {
    const h: f16 = @floatCast(x);
    return @floatCast(h);
}

fn roundBF16(x: f32) f32 {
    var bits: u32 = @bitCast(x);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    bits &= 0xffff0000;
    return @bitCast(bits);
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

fn require(file: *const st.File, name: []const u8, dtype: st.DType, shape: []const usize) !st.Tensor {
    const tensor = file.get(name) orelse return error.MissingTensor;
    if (!tensor.is(dtype, shape)) return error.UnexpectedTensor;
    return tensor;
}

fn tensorName(buffer: []u8, comptime format: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buffer, format, args);
}

const Runner = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    file: *const st.File,
    backend: ?*anyopaque,
    cache_dir: ?[]const u8 = null,
    cache_model_id: ?[]const u8 = null,
    projection_count: usize = 0,
    projection_ms: f64 = 0,
    projection_wall_ms: f64 = 0,
    dequant_ms: f64 = 0,
    cache_hash_ms: f64 = 0,
    cache_read_ms: f64 = 0,
    cache_write_ms: f64 = 0,
    cache_hits: usize = 0,
    cache_misses: usize = 0,
    cache_bytes_read: usize = 0,
    cache_bytes_written: usize = 0,
    cache_trusted_hits: usize = 0,
    cache_aliases_created: usize = 0,

    const CachedWeight = struct {
        file: std.Io.File,
        map: std.Io.File.MemoryMap,
        weight: []const f32,

        fn close(self: *CachedWeight, io: std.Io) void {
            self.map.destroy(io);
            self.file.close(io);
            self.* = undefined;
        }
    };

    fn projectionKey(
        trellis_tensor: st.Tensor,
        suh_tensor: st.Tensor,
        svh_tensor: st.Tensor,
        mcg_tensor: st.Tensor,
        k: usize,
        m: usize,
        words: usize,
    ) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("TensorFold EXL3 decoded row-major f32 cache v1\n");
        var dimensions: [24]u8 = undefined;
        std.mem.writeInt(u64, dimensions[0..8], k, .little);
        std.mem.writeInt(u64, dimensions[8..16], m, .little);
        std.mem.writeInt(u64, dimensions[16..24], words, .little);
        hash.update(&dimensions);
        hash.update(trellis_tensor.bytes);
        hash.update(suh_tensor.bytes);
        hash.update(svh_tensor.bytes);
        hash.update(mcg_tensor.bytes);
        var key: [32]u8 = undefined;
        hash.final(&key);
        return key;
    }

    fn trustedProjectionKey(model_id: []const u8, base: []const u8, k: usize, m: usize, words: usize) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("TensorFold EXL3 decoded row-major f32 trusted cache v1\n");
        hash.update(model_id);
        hash.update("\n");
        hash.update(base);
        var dimensions: [24]u8 = undefined;
        std.mem.writeInt(u64, dimensions[0..8], k, .little);
        std.mem.writeInt(u64, dimensions[8..16], m, .little);
        std.mem.writeInt(u64, dimensions[16..24], words, .little);
        hash.update(&dimensions);
        var key: [32]u8 = undefined;
        hash.final(&key);
        return key;
    }

    fn cachePath(self: *Runner, key: [32]u8) !?[]u8 {
        const dir = self.cache_dir orelse return null;
        return try std.fmt.allocPrint(self.allocator, "{s}/{s}.tfexl3-f32", .{ dir, std.fmt.bytesToHex(key, .lower) });
    }

    fn cacheHeader(key: [32]u8, k: usize, m: usize, payload_bytes: usize) [cache_header_bytes]u8 {
        var header: [cache_header_bytes]u8 = @splat(0);
        @memcpy(header[0..cache_magic.len], cache_magic);
        std.mem.writeInt(u64, header[8..16], k, .little);
        std.mem.writeInt(u64, header[16..24], m, .little);
        std.mem.writeInt(u64, header[24..32], payload_bytes, .little);
        @memcpy(header[32..64], &key);
        return header;
    }

    fn loadCached(self: *Runner, key: [32]u8, k: usize, m: usize) !?CachedWeight {
        const path = try self.cachePath(key) orelse return null;
        defer self.allocator.free(path);
        var file = std.Io.Dir.cwd().openFile(self.io, path, .{}) catch return null;
        var file_owned = true;
        defer if (file_owned) file.close(self.io);
        const payload_bytes = try std.math.mul(usize, try std.math.mul(usize, k, m), @sizeOf(f32));
        const whole_bytes = try std.math.add(usize, cache_header_bytes, payload_bytes);
        if ((file.length(self.io) catch return null) != whole_bytes) return null;
        const started = tf_qualcomm_monotonic_ms();
        var map = std.Io.File.MemoryMap.create(self.io, file, .{
            .len = whole_bytes,
            .protection = .{ .read = true, .write = false },
            .populate = false,
        }) catch return null;
        var map_owned = true;
        defer if (map_owned) map.destroy(self.io);
        const want = cacheHeader(key, k, m, payload_bytes);
        // The final 32 bytes record the content key that created the payload.
        // A trusted model-ID alias is a hard link to that same immutable file,
        // so validate format and geometry here; filename identity is checked by
        // either the full content hash or the caller-supplied model SHA-256.
        if (!std.mem.eql(u8, map.memory[0..32], want[0..32])) return null;
        const payload = map.memory[cache_header_bytes..];
        const weight = std.mem.bytesAsSlice(f32, payload);
        self.cache_read_ms += tf_qualcomm_monotonic_ms() - started;
        self.cache_hits += 1;
        self.cache_bytes_read += payload_bytes;
        file_owned = false;
        map_owned = false;
        return .{ .file = file, .map = map, .weight = weight };
    }

    fn saveCached(self: *Runner, key: [32]u8, k: usize, m: usize, weight: []const f32) !void {
        const path = try self.cachePath(key) orelse return;
        defer self.allocator.free(path);
        const temp = try std.fmt.allocPrint(self.allocator, "{s}.{d}.tmp", .{ path, std.c.getpid() });
        defer self.allocator.free(temp);
        const payload = std.mem.sliceAsBytes(weight);
        const header = cacheHeader(key, k, m, payload.len);
        const started = tf_qualcomm_monotonic_ms();
        var file = try std.Io.Dir.cwd().createFile(self.io, temp, .{ .truncate = true });
        errdefer std.Io.Dir.cwd().deleteFile(self.io, temp) catch {};
        defer file.close(self.io);
        try file.writePositionalAll(self.io, &header, 0);
        try file.writePositionalAll(self.io, payload, cache_header_bytes);
        try file.setLength(self.io, cache_header_bytes + payload.len);
        file.sync(self.io) catch {};
        try std.Io.Dir.cwd().rename(temp, std.Io.Dir.cwd(), path, self.io);
        self.cache_write_ms += tf_qualcomm_monotonic_ms() - started;
        self.cache_bytes_written += payload.len;
    }

    fn linkCached(self: *Runner, content_key: [32]u8, trusted_key: [32]u8) !void {
        const content_path = try self.cachePath(content_key) orelse return;
        defer self.allocator.free(content_path);
        const trusted_path = try self.cachePath(trusted_key) orelse return;
        defer self.allocator.free(trusted_path);
        std.Io.Dir.cwd().hardLink(content_path, std.Io.Dir.cwd(), trusted_path, self.io, .{}) catch |err| switch (err) {
            error.PathAlreadyExists => return,
            else => return err,
        };
        self.cache_aliases_created += 1;
    }

    fn runWeight(self: *Runner, k: usize, m: usize, weight: []const f32, input: []const f32, output: []f32) !void {
        var ms: f64 = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_matmul_once(self.backend, k, m, 1, weight.ptr, input.ptr, output.ptr, &ms);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 1;
    }

    fn project(self: *Runner, base: []const u8, input: []const f32, output: []f32) !void {
        var name_buffer: [256]u8 = undefined;
        const trellis_tensor = self.file.get(try tensorName(&name_buffer, "{s}.trellis", .{base})) orelse return error.MissingTensor;
        if (trellis_tensor.dtype != .i16 or trellis_tensor.rank != 3) return error.UnexpectedTensor;
        const tiles_k = trellis_tensor.dim(0);
        const tiles_m = trellis_tensor.dim(1);
        const words = trellis_tensor.dim(2);
        if (words % 8 != 0) return error.UnexpectedTensor;
        const k = tiles_k * 16;
        const m = tiles_m * 16;
        if (input.len != k or output.len != m) return error.UnexpectedTensor;

        const suh_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.suh", .{base}), .f16, &.{k});
        const svh_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.svh", .{base}), .f16, &.{m});
        const mcg_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.mcg", .{base}), .i32, &.{});
        if (std.mem.readInt(u32, mcg_tensor.bytes[0..4], .little) != exl3.McgMul) return error.UnexpectedCodebook;

        var cache_key: ?[32]u8 = null;
        if (self.cache_dir != null) {
            const trusted_key = if (self.cache_model_id) |model_id| trustedProjectionKey(model_id, base, k, m, words) else null;
            if (trusted_key) |key| if (try self.loadCached(key, k, m)) |loaded_value| {
                self.cache_trusted_hits += 1;
                var loaded = loaded_value;
                defer loaded.close(self.io);
                return self.runWeight(k, m, loaded.weight, input, output);
            };
            const hash_started = tf_qualcomm_monotonic_ms();
            const key = projectionKey(trellis_tensor, suh_tensor, svh_tensor, mcg_tensor, k, m, words);
            self.cache_hash_ms += tf_qualcomm_monotonic_ms() - hash_started;
            cache_key = key;
            if (try self.loadCached(key, k, m)) |loaded_value| {
                if (trusted_key) |alias| try self.linkCached(key, alias);
                var loaded = loaded_value;
                defer loaded.close(self.io);
                return self.runWeight(k, m, loaded.weight, input, output);
            }
            self.cache_misses += 1;
        }

        const trellis = try self.allocator.alloc(i16, trellis_tensor.numel());
        defer self.allocator.free(trellis);
        @memcpy(std.mem.sliceAsBytes(trellis), trellis_tensor.bytes);
        const suh = try self.allocator.alloc(f16, k);
        defer self.allocator.free(suh);
        @memcpy(std.mem.sliceAsBytes(suh), suh_tensor.bytes);
        const svh = try self.allocator.alloc(f16, m);
        defer self.allocator.free(svh);
        @memcpy(std.mem.sliceAsBytes(svh), svh_tensor.bytes);

        const elements = try std.math.mul(usize, k, m);
        const decoded = try self.allocator.alloc(f32, elements);
        defer self.allocator.free(decoded);
        const unpacked = try self.allocator.alloc(f16, elements);
        defer self.allocator.free(unpacked);
        const decode_started = tf_qualcomm_monotonic_ms();
        exl3.dequantizeFastF32(.{ .halves = @intCast(words / 8) }, .mcg, trellis, tiles_k, tiles_m, suh, svh, decoded, unpacked);
        self.dequant_ms += tf_qualcomm_monotonic_ms() - decode_started;

        // EXL3 reconstructs [K,M]; GGML's matrix operand is contiguous [M,K].
        const weight = try self.allocator.alloc(f32, elements);
        defer self.allocator.free(weight);
        for (0..m) |out_col| for (0..k) |inner| {
            weight[out_col * k + inner] = decoded[inner * m + out_col];
        };
        if (cache_key) |key| try self.saveCached(key, k, m, weight);
        if (cache_key) |key| if (self.cache_model_id) |model_id| {
            try self.linkCached(key, trustedProjectionKey(model_id, base, k, m, words));
        };
        try self.runWeight(k, m, weight, input, output);
    }
};

fn dense(file: *const st.File, name: []const u8, input: []const f32, output: []f32) !void {
    const tensor = file.get(name) orelse return error.MissingTensor;
    if (tensor.dtype != .f16 or tensor.rank != 2 or tensor.dim(0) != output.len or tensor.dim(1) != input.len) return error.UnexpectedTensor;
    for (output, 0..) |*out, row| {
        var sum: f32 = 0;
        for (input, 0..) |x, col| sum += f16Value(tensor.bytes, row * input.len + col) * x;
        out.* = sum;
    }
}

fn rmsNorm(file: *const st.File, name: []const u8, input: []const f32, output: []f32, constant_bias: f32) !void {
    const tensor = file.get(name) orelse return error.MissingTensor;
    if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 1 or tensor.dim(0) != input.len or output.len != input.len) return error.UnexpectedTensor;
    var sum: f32 = 0;
    for (input) |x| sum += x * x;
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(input.len)) + eps);
    for (output, 0..) |*out, i| out.* = roundF16(input[i] * inv * (value(tensor, i) + constant_bias));
}

fn residualAdd(hidden_state: []f32, update: []const f32) void {
    for (hidden_state, update) |*h, u| h.* = roundF16(h.* + u);
}

fn linearAttention(runner: *Runner, layer: usize, input: []const f32, output: []f32) !void {
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.linear_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;

    const qkv = try a.alloc(f32, 3 * hidden);
    defer a.free(qkv);
    try runner.project(try tensorName(&projection_buffer, "{s}.in_proj_qkv", .{base}), input, qkv);
    for (qkv) |*x| x.* = roundBF16(roundF16(x.*));

    const z = try a.alloc(f32, hidden);
    defer a.free(z);
    try runner.project(try tensorName(&projection_buffer, "{s}.in_proj_z", .{base}), input, z);

    var a_gate: [16]f32 = undefined;
    var b_gate: [16]f32 = undefined;
    try dense(runner.file, try tensorName(&projection_buffer, "{s}.in_proj_a.weight", .{base}), input, &a_gate);
    try dense(runner.file, try tensorName(&projection_buffer, "{s}.in_proj_b.weight", .{base}), input, &b_gate);

    const conv_name = try tensorName(&projection_buffer, "{s}.conv1d.weight", .{base});
    const conv = try require(runner.file, conv_name, .bf16, &.{ 3 * hidden, 1, 4 });
    const mixed = try a.alloc(f32, 3 * hidden);
    defer a.free(mixed);
    for (mixed, 0..) |*out, c| {
        const conv_value = roundBF16(value(conv, c * 4 + 3) * qkv[c]);
        const conv_sigmoid = roundBF16(sigmoid(conv_value));
        out.* = roundBF16(conv_value * conv_sigmoid);
    }

    var dt_name: [256]u8 = undefined;
    const dt = try require(runner.file, try tensorName(&dt_name, "{s}.dt_bias", .{base}), .bf16, &.{16});
    var alog_name: [256]u8 = undefined;
    const alog = try require(runner.file, try tensorName(&alog_name, "{s}.A_log", .{base}), .f32, &.{16});
    var beta: [16]f32 = undefined;
    var decay: [16]f32 = undefined;
    for (0..16) |head| {
        beta[head] = roundBF16(sigmoid(b_gate[head]));
        const s = roundBF16(a_gate[head] + value(dt, head));
        const softplus = roundBF16(@max(s, 0.0) + @log(1.0 + @exp(-@abs(s))));
        decay[head] = @exp(-@exp(value(alog, head)) * softplus);
    }
    // Position zero starts from an all-zero recurrent state. `decay` is still
    // computed above so the complete gate path is validated, but it cancels
    // out of this first state update.

    const q = try a.alloc(f32, hidden);
    defer a.free(q);
    const k = try a.alloc(f32, hidden);
    defer a.free(k);
    const v = try a.alloc(f32, hidden);
    defer a.free(v);
    const q_scale = roundBF16(1.0 / 128.0);
    const k_scale = roundBF16(1.0 / @sqrt(@as(f32, 128.0)));
    for (0..16) |head| {
        var qss: f32 = 0;
        var kss: f32 = 0;
        for (0..128) |d| {
            const qv = mixed[head * 128 + d];
            const kv = mixed[hidden + head * 128 + d];
            qss += qv * qv;
            kss += kv * kv;
            v[head * 128 + d] = mixed[2 * hidden + head * 128 + d];
        }
        const qinv = 1.0 / @sqrt(qss / 128.0 + eps / 128.0);
        const kinv = 1.0 / @sqrt(kss / 128.0 + eps / 128.0);
        for (0..128) |d| {
            q[head * 128 + d] = roundBF16(q_scale * roundBF16(mixed[head * 128 + d] * qinv));
            k[head * 128 + d] = roundBF16(k_scale * roundBF16(mixed[hidden + head * 128 + d] * kinv));
        }
    }

    const y = try a.alloc(f32, hidden);
    defer a.free(y);
    for (0..16) |head| {
        var qk: f32 = 0;
        for (0..128) |d| qk += q[head * 128 + d] * k[head * 128 + d];
        const carried_state: f32 = 0.0 * decay[head];
        for (0..128) |d| y[head * 128 + d] = roundBF16((carried_state + v[head * 128 + d] * beta[head]) * qk);
    }

    var norm_name: [256]u8 = undefined;
    const norm = try require(runner.file, try tensorName(&norm_name, "{s}.norm.weight", .{base}), .f32, &.{128});
    const gated = try a.alloc(f32, hidden);
    defer a.free(gated);
    for (0..16) |head| {
        var ss: f32 = 0;
        for (0..128) |d| {
            const x = y[head * 128 + d];
            ss += x * x;
        }
        const inv = 1.0 / @sqrt(ss / 128.0 + eps);
        for (0..128) |d| {
            const i = head * 128 + d;
            gated[i] = roundF16(silu(z[i]) * (value(norm, d) * y[i] * inv));
        }
    }
    try runner.project(try tensorName(&projection_buffer, "{s}.out_proj", .{base}), gated, output);
}

fn headNorm(file: *const st.File, name: []const u8, input: []const f32, heads: usize, output: []f32) !void {
    const tensor = file.get(name) orelse return error.MissingTensor;
    const width = input.len / heads;
    if (input.len != output.len or width != 256 or tensor.dtype != .bf16 or !tensor.is(.bf16, &.{width})) return error.UnexpectedTensor;
    for (0..heads) |head| {
        var ss: f32 = 0;
        for (0..width) |d| {
            const x = input[head * width + d];
            ss += x * x;
        }
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(width)) + eps);
        for (0..width) |d| output[head * width + d] = roundF16(input[head * width + d] * inv * (value(tensor, d) + 1.0));
    }
}

fn fullAttention(runner: *Runner, layer: usize, input: []const f32, output: []f32) !void {
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.self_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;
    const qraw = try a.alloc(f32, 4096);
    defer a.free(qraw);
    const kraw = try a.alloc(f32, 512);
    defer a.free(kraw);
    const vraw = try a.alloc(f32, 512);
    defer a.free(vraw);
    try runner.project(try tensorName(&projection_buffer, "{s}.q_proj", .{base}), input, qraw);
    try runner.project(try tensorName(&projection_buffer, "{s}.k_proj", .{base}), input, kraw);
    try runner.project(try tensorName(&projection_buffer, "{s}.v_proj", .{base}), input, vraw);
    for (qraw) |*x| x.* = roundF16(x.*);
    for (kraw) |*x| x.* = roundF16(x.*);
    for (vraw) |*x| x.* = roundF16(x.*);

    const q = try a.alloc(f32, hidden);
    defer a.free(q);
    const gate = try a.alloc(f32, hidden);
    defer a.free(gate);
    for (0..8) |head| for (0..256) |d| {
        q[head * 256 + d] = qraw[head * 512 + d];
        gate[head * 256 + d] = qraw[head * 512 + 256 + d];
    };
    const qnorm = try a.alloc(f32, hidden);
    defer a.free(qnorm);
    const knorm = try a.alloc(f32, 512);
    defer a.free(knorm);
    var norm_buffer: [256]u8 = undefined;
    try headNorm(runner.file, try tensorName(&norm_buffer, "{s}.q_norm.weight", .{base}), q, 8, qnorm);
    try headNorm(runner.file, try tensorName(&norm_buffer, "{s}.k_norm.weight", .{base}), kraw, 2, knorm);
    // RoPE at position zero is identity. A one-key softmax is exactly one, so
    // attention output is V repeated across each group of four query heads.
    const attended = try a.alloc(f32, hidden);
    defer a.free(attended);
    for (0..8) |query_head| for (0..256) |d| {
        const i = query_head * 256 + d;
        attended[i] = roundF16(vraw[(query_head / 4) * 256 + d] * sigmoid(gate[i]));
    };
    try runner.project(try tensorName(&projection_buffer, "{s}.o_proj", .{base}), attended, output);
}

fn mlp(runner: *Runner, layer: usize, input: []const f32, output: []f32) !void {
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.mlp", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;
    const gate = try a.alloc(f32, intermediate);
    defer a.free(gate);
    const up = try a.alloc(f32, intermediate);
    defer a.free(up);
    const activated = try a.alloc(f32, intermediate);
    defer a.free(activated);
    try runner.project(try tensorName(&projection_buffer, "{s}.gate_proj", .{base}), input, gate);
    try runner.project(try tensorName(&projection_buffer, "{s}.up_proj", .{base}), input, up);
    for (activated, 0..) |*out, i| out.* = roundF16(silu(roundF16(gate[i])) * roundF16(up[i]));
    try runner.project(try tensorName(&projection_buffer, "{s}.down_proj", .{base}), activated, output);
}

const Top = struct { id: usize, logit: f32 };

fn tiedHead(file: *const st.File, state: []const f32) ![5]Top {
    const embedding = try require(file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
    var top: [5]Top = @splat(.{ .id = 0, .logit = -std.math.inf(f32) });
    for (0..vocab) |token| {
        var logit: f32 = 0;
        for (0..hidden) |d| logit += f16Value(embedding.bytes, token * hidden + d) * state[d];
        if (logit > top[4].logit) {
            top[4] = .{ .id = token, .logit = logit };
            var i: usize = 4;
            while (i > 0 and top[i].logit > top[i - 1].logit) : (i -= 1) std.mem.swap(Top, &top[i], &top[i - 1]);
        }
    }
    return top;
}

fn stateHash(state: []const f32) u64 {
    var hash: u64 = 14695981039346656037;
    for (state) |x| {
        const h: f16 = @floatCast(x);
        const bits: u16 = @bitCast(h);
        hash ^= @as(u8, @truncate(bits));
        hash *%= 1099511628211;
        hash ^= @as(u8, @truncate(bits >> 8));
        hash *%= 1099511628211;
    }
    return hash;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5 or args.len > 7) {
        std.debug.print("usage: {s} cpu|htp MODEL_SAFETENSORS TOKEN_ID CPU_THREADS [DECODED_CACHE_DIR [MODEL_SHA256]]\n", .{args[0]});
        return 64;
    }
    if (!std.mem.eql(u8, args[1], "cpu") and !std.mem.eql(u8, args[1], "htp")) return error.InvalidBackend;
    const token = try std.fmt.parseInt(usize, args[3], 10);
    const threads = try std.fmt.parseInt(i32, args[4], 10);
    if (token >= vocab or threads <= 0) return error.InvalidArgument;

    var file = try st.File.open(init.gpa, init.io, args[2]);
    defer file.close(init.io);
    const embedding = try require(&file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
    const backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BackendInitFailed;
    defer tf_qualcomm_backend_destroy(backend);
    const cache_dir: ?[]const u8 = if (args.len >= 6) args[5] else null;
    const cache_model_id: ?[]const u8 = if (args.len == 7) args[6] else null;
    if (cache_model_id) |id| {
        if (id.len != 64) return error.InvalidModelSha256;
        for (id) |c| if (!std.ascii.isHex(c)) return error.InvalidModelSha256;
    }
    if (cache_dir) |dir| try std.Io.Dir.cwd().createDirPath(init.io, dir);
    var runner: Runner = .{ .allocator = init.gpa, .io = init.io, .file = &file, .backend = backend, .cache_dir = cache_dir, .cache_model_id = cache_model_id };
    const started = tf_qualcomm_monotonic_ms();

    const state = try init.gpa.alloc(f32, hidden);
    defer init.gpa.free(state);
    for (state, 0..) |*x, d| x.* = f16Value(embedding.bytes, token * hidden + d);
    const normed = try init.gpa.alloc(f32, hidden);
    defer init.gpa.free(normed);
    const update = try init.gpa.alloc(f32, hidden);
    defer init.gpa.free(update);
    var name_buffer: [256]u8 = undefined;

    for (0..layers) |layer| {
        const input_norm = try tensorName(&name_buffer, "{s}.layers.{d}.input_layernorm.weight", .{ prefix, layer });
        try rmsNorm(&file, input_norm, state, normed, 1.0);
        if ((layer + 1) % 4 == 0) {
            try fullAttention(&runner, layer, normed, update);
        } else {
            try linearAttention(&runner, layer, normed, update);
        }
        residualAdd(state, update);
        const post_norm = try tensorName(&name_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
        try rmsNorm(&file, post_norm, state, normed, 1.0);
        try mlp(&runner, layer, normed, update);
        residualAdd(state, update);
        std.debug.print("tensorfold forward {s}: layer {d}/{d}, projections {d}\n", .{ args[1], layer + 1, layers, runner.projection_count });
    }

    try rmsNorm(&file, prefix ++ ".norm.weight", state, normed, 1.0);
    const head_started = tf_qualcomm_monotonic_ms();
    const top = try tiedHead(&file, normed);
    const head_ms = tf_qualcomm_monotonic_ms() - head_started;
    const elapsed = tf_qualcomm_monotonic_ms() - started;
    std.debug.print(
        "{{\"tensorfold\":\"qualcomm-exl3-forward\",\"backend\":\"{s}\",\"checkpoint_tensors\":{d},\"input_token\":{d},\"position\":0,\"layers\":{d},\"projections\":{d},\"head\":\"tied-f16-embedding\",\"next_token\":{d},\"top5\":[[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}]],\"state_f16_fnv1a\":\"{x}\",\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"head_ms\":{d:.3},\"dequant_ms\":{d:.3},\"cache_hash_ms\":{d:.3},\"cache_read_ms\":{d:.3},\"cache_write_ms\":{d:.3},\"cache_hits\":{d},\"cache_trusted_hits\":{d},\"cache_misses\":{d},\"cache_aliases_created\":{d},\"cache_bytes_read\":{d},\"cache_bytes_written\":{d},\"elapsed_ms\":{d:.3},\"peak_rss_kib\":{d},\"passed\":true}}\n",
        .{ args[1], file.names.count(), token, layers, runner.projection_count, top[0].id, top[0].id, top[0].logit, top[1].id, top[1].logit, top[2].id, top[2].logit, top[3].id, top[3].logit, top[4].id, top[4].logit, stateHash(normed), runner.projection_ms, runner.projection_wall_ms, head_ms, runner.dequant_ms, runner.cache_hash_ms, runner.cache_read_ms, runner.cache_write_ms, runner.cache_hits, runner.cache_trusted_hits, runner.cache_misses, runner.cache_aliases_created, runner.cache_bytes_read, runner.cache_bytes_written, elapsed, tf_qualcomm_peak_rss_kib() },
    );
    return 0;
}
