//! A deliberately small, auditable Qwen3.5-2B EXL3 decode path for Qualcomm
//! Android bring-up. It executes autoregressive decode end-to-end and keeps the
//! model math in TensorFold while routing every quantized projection through
//! either GGML CPU or Hexagon HTP. An optional content-addressed cache keeps
//! decoded row-major weights between runs so EXL3 reconstruction is paid once.

const std = @import("std");
const exl3 = @import("exl3_format");
const st = @import("safetensors");
const profile = @import("model_profile");

extern fn tf_qualcomm_monotonic_ms() f64;
extern fn tf_qualcomm_peak_rss_kib() usize;
extern fn tf_qualcomm_tied_head_f16(
    embedding: [*]const u16,
    state: [*]const f32,
    vocab_size: usize,
    hidden_size: usize,
    cpu_threads: i32,
    top_ids: *[5]usize,
    top_logits: *[5]f32,
) c_int;
extern fn tf_qualcomm_gdn_step_f32(
    recurrent: [*]f32,
    query: [*]const f32,
    key: [*]const f32,
    value: [*]const f32,
    beta: [*]const f32,
    decay: [*]const f32,
    output: [*]f32,
    key_heads: usize,
    value_heads: usize,
    key_width: usize,
    value_width: usize,
    thread_count: usize,
) void;
extern fn tf_qualcomm_dense_f16(weight: [*]const u16, input: [*]const f32, output: [*]f32, rows: usize, columns: usize) void;
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
extern fn tf_qualcomm_f32_to_f16(input: [*]const f32, output: [*]u16, count: usize) void;
extern fn tf_qualcomm_backend_matmul_f16_once(
    context: ?*anyopaque,
    k: usize,
    m: usize,
    n: usize,
    weight: [*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
) c_int;
extern fn tf_qualcomm_backend_matmul_f16_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    k: usize,
    m: usize,
    n: usize,
    weight: ?[*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weight_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_matmul_f16_resident_group_once(
    context: ?*anyopaque,
    slot_id: usize,
    matrix_count: usize,
    k: usize,
    m0: usize,
    m1: usize,
    m2: usize,
    n: usize,
    weight0: ?[*]const u16,
    weight1: ?[*]const u16,
    weight2: ?[*]const u16,
    input: [*]const f32,
    output0: [*]f32,
    output1: [*]f32,
    output2: ?[*]f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;
extern fn tf_qualcomm_backend_mlp_f16_resident_once(
    context: ?*anyopaque,
    slot_id: usize,
    input_width: usize,
    intermediate_width: usize,
    output_width: usize,
    n: usize,
    gate_weight: ?[*]const u16,
    up_weight: ?[*]const u16,
    down_weight: ?[*]const u16,
    input: [*]const f32,
    output: [*]f32,
    compute_ms: *f64,
    weights_uploaded: *c_int,
) c_int;

const prefix = "model.language_model";
const hidden = profile.hidden;
const intermediate = profile.intermediate;
const layers = profile.layers;
const vocab = profile.vocab;
const eps: f32 = 1e-6;
const linear_key_heads = profile.linear_key_heads;
const linear_value_heads = profile.linear_value_heads;
const linear_key_width = profile.linear_key_width;
const linear_value_width = profile.linear_value_width;
const linear_query_width = linear_key_heads * linear_key_width;
const linear_output_width = linear_value_heads * linear_value_width;
const linear_state_elems = linear_value_heads * linear_value_width * linear_key_width;
const conv_channels = 2 * linear_query_width + linear_output_width;
const conv_history_rows = 3;
const attention_heads = profile.attention_heads;
const kv_heads = profile.kv_heads;
const attention_width = profile.attention_width;
const attention_query_width = attention_heads * attention_width;
const kv_width = kv_heads * attention_width;
const rope_width = profile.rope_width;
const cache_magic = "TFEXL3F1";
const f16_cache_magic = "TFEXL3H1";
const cache_header_bytes = 64;
const projections_per_position = layers * 6 + layers / 4;

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
    cpu_threads: usize,
    use_f16_weights: bool = false,
    use_resident_weights: bool = false,
    use_resident_head: bool = false,
    cache_dir: ?[]const u8 = null,
    cache_model_id: ?[]const u8 = null,
    projection_count: usize = 0,
    projection_ms: f64 = 0,
    projection_wall_ms: f64 = 0,
    weight_convert_ms: f64 = 0,
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
    f16_cache_hits: usize = 0,
    f16_cache_misses: usize = 0,
    f16_cache_bytes_read: usize = 0,
    f16_cache_bytes_written: usize = 0,
    resident_loaded: [projections_per_position]bool = @splat(false),
    resident_slot_cursor: usize = 0,
    resident_weight_uploads: usize = 0,
    resident_weight_hits: usize = 0,
    resident_weight_fallbacks: usize = 0,
    resident_weight_bytes_uploaded: usize = 0,
    resident_head_loaded: bool = false,
    resident_head_uploads: usize = 0,
    resident_head_hits: usize = 0,
    resident_head_fallbacks: usize = 0,
    resident_head_compute_ms: f64 = 0,

    fn resetMetrics(self: *Runner) void {
        self.projection_count = 0;
        self.projection_ms = 0;
        self.projection_wall_ms = 0;
        self.weight_convert_ms = 0;
        self.dequant_ms = 0;
        self.cache_hash_ms = 0;
        self.cache_read_ms = 0;
        self.cache_write_ms = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.cache_bytes_read = 0;
        self.cache_bytes_written = 0;
        self.cache_trusted_hits = 0;
        self.cache_aliases_created = 0;
        self.f16_cache_hits = 0;
        self.f16_cache_misses = 0;
        self.f16_cache_bytes_read = 0;
        self.f16_cache_bytes_written = 0;
        self.resident_weight_uploads = 0;
        self.resident_weight_hits = 0;
        self.resident_weight_fallbacks = 0;
        self.resident_weight_bytes_uploaded = 0;
        self.resident_slot_cursor = 0;
        self.resident_head_uploads = 0;
        self.resident_head_hits = 0;
        self.resident_head_fallbacks = 0;
        self.resident_head_compute_ms = 0;
    }

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

    const CachedF16Weight = struct {
        file: std.Io.File,
        map: std.Io.File.MemoryMap,
        weight: []const u16,

        fn close(self: *CachedF16Weight, io: std.Io) void {
            self.map.destroy(io);
            self.file.close(io);
            self.* = undefined;
        }
    };

    const ResidentProjection = struct {
        k: usize,
        m: usize,
        key: [32]u8,
    };

    fn residentProjection(self: *Runner, base: []const u8) !ResidentProjection {
        var name_buffer: [256]u8 = undefined;
        const trellis_tensor = self.file.get(try tensorName(&name_buffer, "{s}.trellis", .{base})) orelse return error.MissingTensor;
        if (trellis_tensor.dtype != .i16 or trellis_tensor.rank != 3) return error.UnexpectedTensor;
        const tiles_k = trellis_tensor.dim(0);
        const tiles_m = trellis_tensor.dim(1);
        const words = trellis_tensor.dim(2);
        if (words % 8 != 0) return error.UnexpectedTensor;
        const k = tiles_k * 16;
        const m = tiles_m * 16;
        const model_id = self.cache_model_id orelse return error.InvalidModelSha256;
        return .{ .k = k, .m = m, .key = f16ProjectionKey(model_id, base, k, m, words) };
    }

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

    fn f16ProjectionKey(model_id: []const u8, base: []const u8, k: usize, m: usize, words: usize) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("TensorFold EXL3 decoded row-major f16 cache v1\n");
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

    fn f16CachePath(self: *Runner, key: [32]u8) !?[]u8 {
        const dir = self.cache_dir orelse return null;
        return try std.fmt.allocPrint(self.allocator, "{s}/{s}.tfexl3-f16", .{ dir, std.fmt.bytesToHex(key, .lower) });
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

    fn f16CacheHeader(key: [32]u8, k: usize, m: usize, payload_bytes: usize) [cache_header_bytes]u8 {
        var header = cacheHeader(key, k, m, payload_bytes);
        @memcpy(header[0..f16_cache_magic.len], f16_cache_magic);
        return header;
    }

    fn loadCachedF16(self: *Runner, key: [32]u8, k: usize, m: usize) !?CachedF16Weight {
        const path = try self.f16CachePath(key) orelse return null;
        defer self.allocator.free(path);
        var file = std.Io.Dir.cwd().openFile(self.io, path, .{}) catch return null;
        var file_owned = true;
        defer if (file_owned) file.close(self.io);
        const payload_bytes = try std.math.mul(usize, try std.math.mul(usize, k, m), @sizeOf(u16));
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
        const want = f16CacheHeader(key, k, m, payload_bytes);
        if (!std.mem.eql(u8, map.memory[0..cache_header_bytes], &want)) return null;
        const weight = std.mem.bytesAsSlice(u16, map.memory[cache_header_bytes..]);
        self.cache_read_ms += tf_qualcomm_monotonic_ms() - started;
        self.f16_cache_hits += 1;
        self.f16_cache_bytes_read += payload_bytes;
        file_owned = false;
        map_owned = false;
        return .{ .file = file, .map = map, .weight = weight };
    }

    fn saveCachedF16(self: *Runner, key: [32]u8, k: usize, m: usize, weight: []const u16) !void {
        const path = try self.f16CachePath(key) orelse return;
        defer self.allocator.free(path);
        const temp = try std.fmt.allocPrint(self.allocator, "{s}.{d}.tmp", .{ path, std.c.getpid() });
        defer self.allocator.free(temp);
        const payload = std.mem.sliceAsBytes(weight);
        const header = f16CacheHeader(key, k, m, payload.len);
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
        self.f16_cache_bytes_written += payload.len;
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
        if (input.len == 0 or input.len % k != 0) return error.UnexpectedTensor;
        const n = input.len / k;
        if (output.len != m * n) return error.UnexpectedTensor;
        var ms: f64 = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_matmul_once(self.backend, k, m, n, weight.ptr, input.ptr, output.ptr, &ms);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 1;
    }

    fn runResidentWeightF16(self: *Runner, slot_id: usize, k: usize, m: usize, weight: ?[]const u16, input: []const f32, output: []f32) bool {
        const n = input.len / k;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const weight_ptr: ?[*]const u16 = if (weight) |words| words.ptr else null;
        const status = tf_qualcomm_backend_matmul_f16_resident_once(self.backend, slot_id, k, m, n, weight_ptr, input.ptr, output.ptr, &ms, &uploaded);
        if (status != 0) {
            self.resident_weight_fallbacks += 1;
            self.resident_loaded[slot_id] = false;
            return false;
        }
        self.projection_ms += ms;
        self.projection_count += 1;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot_id] = true;
            self.resident_weight_uploads += 1;
            self.resident_weight_bytes_uploaded += k * m * @sizeOf(u16);
        } else {
            self.resident_weight_hits += 1;
        }
        return true;
    }

    fn runWeightF16(self: *Runner, k: usize, m: usize, weight: []const u16, input: []const f32, output: []f32) !void {
        if (input.len == 0 or input.len % k != 0) return error.UnexpectedTensor;
        const n = input.len / k;
        if (weight.len != k * m or output.len != m * n) return error.UnexpectedTensor;
        const started = tf_qualcomm_monotonic_ms();
        const slot_id = self.resident_slot_cursor;
        if (self.use_resident_weights and self.runResidentWeightF16(slot_id, k, m, weight, input, output)) {
            self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
            return;
        }
        var ms: f64 = 0;
        const status = tf_qualcomm_backend_matmul_f16_once(self.backend, k, m, n, weight.ptr, input.ptr, output.ptr, &ms);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) return error.BridgeFailed;
        self.projection_ms += ms;
        self.projection_count += 1;
        if (self.use_resident_weights) self.resident_slot_cursor += 1;
    }

    fn projectPairResident(self: *Runner, base0: []const u8, base1: []const u8, input: []const f32, output0: []f32, output1: []f32) !void {
        const p0 = try self.residentProjection(base0);
        const p1 = try self.residentProjection(base1);
        if (p0.k != p1.k or input.len == 0 or input.len % p0.k != 0) return error.UnexpectedTensor;
        const n = input.len / p0.k;
        if (output0.len != p0.m * n or output1.len != p1.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var loaded0: ?CachedF16Weight = null;
        defer if (loaded0) |*loaded| loaded.close(self.io);
        var loaded1: ?CachedF16Weight = null;
        defer if (loaded1) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            loaded0 = try self.loadCachedF16(p0.key, p0.k, p0.m) orelse return error.MissingResidentWeight;
            loaded1 = try self.loadCachedF16(p1.key, p1.k, p1.m) orelse return error.MissingResidentWeight;
        }
        const weight0: ?[*]const u16 = if (loaded0) |loaded| loaded.weight.ptr else null;
        const weight1: ?[*]const u16 = if (loaded1) |loaded| loaded.weight.ptr else null;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_matmul_f16_resident_group_once(self.backend, slot, 2, p0.k, p0.m, p1.m, 0, n, weight0, weight1, null, input.ptr, output0.ptr, output1.ptr, null, &ms, &uploaded);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("resident pair bridge failed: status {d}, slot {d}, k {d}, m {d}/{d}, n {d}\n", .{ status, slot, p0.k, p0.m, p1.m, n });
            return error.BridgeFailed;
        }
        self.projection_ms += ms;
        self.projection_count += 2;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 2;
            self.resident_weight_bytes_uploaded += (p0.k * p0.m + p1.k * p1.m) * @sizeOf(u16);
        } else {
            self.resident_weight_hits += 2;
        }
    }

    fn projectTripleResident(self: *Runner, base0: []const u8, base1: []const u8, base2: []const u8, input: []const f32, output0: []f32, output1: []f32, output2: []f32) !void {
        const p0 = try self.residentProjection(base0);
        const p1 = try self.residentProjection(base1);
        const p2 = try self.residentProjection(base2);
        if (p0.k != p1.k or p0.k != p2.k or input.len == 0 or input.len % p0.k != 0) return error.UnexpectedTensor;
        const n = input.len / p0.k;
        if (output0.len != p0.m * n or output1.len != p1.m * n or output2.len != p2.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var loaded0: ?CachedF16Weight = null;
        defer if (loaded0) |*loaded| loaded.close(self.io);
        var loaded1: ?CachedF16Weight = null;
        defer if (loaded1) |*loaded| loaded.close(self.io);
        var loaded2: ?CachedF16Weight = null;
        defer if (loaded2) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            loaded0 = try self.loadCachedF16(p0.key, p0.k, p0.m) orelse return error.MissingResidentWeight;
            loaded1 = try self.loadCachedF16(p1.key, p1.k, p1.m) orelse return error.MissingResidentWeight;
            loaded2 = try self.loadCachedF16(p2.key, p2.k, p2.m) orelse return error.MissingResidentWeight;
        }
        const weight0: ?[*]const u16 = if (loaded0) |loaded| loaded.weight.ptr else null;
        const weight1: ?[*]const u16 = if (loaded1) |loaded| loaded.weight.ptr else null;
        const weight2: ?[*]const u16 = if (loaded2) |loaded| loaded.weight.ptr else null;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_matmul_f16_resident_group_once(self.backend, slot, 3, p0.k, p0.m, p1.m, p2.m, n, weight0, weight1, weight2, input.ptr, output0.ptr, output1.ptr, output2.ptr, &ms, &uploaded);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("resident triple bridge failed: status {d}, slot {d}, k {d}, m {d}/{d}/{d}, n {d}\n", .{ status, slot, p0.k, p0.m, p1.m, p2.m, n });
            return error.BridgeFailed;
        }
        self.projection_ms += ms;
        self.projection_count += 3;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 3;
            self.resident_weight_bytes_uploaded += (p0.k * p0.m + p1.k * p1.m + p2.k * p2.m) * @sizeOf(u16);
        } else {
            self.resident_weight_hits += 3;
        }
    }

    fn mlpResident(self: *Runner, gate_base: []const u8, up_base: []const u8, down_base: []const u8, input: []const f32, output: []f32) !void {
        const gate = try self.residentProjection(gate_base);
        const up = try self.residentProjection(up_base);
        const down = try self.residentProjection(down_base);
        if (gate.k != up.k or gate.m != up.m or down.k != gate.m or input.len == 0 or input.len % gate.k != 0) return error.UnexpectedTensor;
        const n = input.len / gate.k;
        if (output.len != down.m * n) return error.UnexpectedTensor;
        const slot = self.resident_slot_cursor;
        var gate_loaded: ?CachedF16Weight = null;
        defer if (gate_loaded) |*loaded| loaded.close(self.io);
        var up_loaded: ?CachedF16Weight = null;
        defer if (up_loaded) |*loaded| loaded.close(self.io);
        var down_loaded: ?CachedF16Weight = null;
        defer if (down_loaded) |*loaded| loaded.close(self.io);
        if (!self.resident_loaded[slot]) {
            gate_loaded = try self.loadCachedF16(gate.key, gate.k, gate.m) orelse return error.MissingResidentWeight;
            up_loaded = try self.loadCachedF16(up.key, up.k, up.m) orelse return error.MissingResidentWeight;
            down_loaded = try self.loadCachedF16(down.key, down.k, down.m) orelse return error.MissingResidentWeight;
        }
        const gate_weight: ?[*]const u16 = if (gate_loaded) |loaded| loaded.weight.ptr else null;
        const up_weight: ?[*]const u16 = if (up_loaded) |loaded| loaded.weight.ptr else null;
        const down_weight: ?[*]const u16 = if (down_loaded) |loaded| loaded.weight.ptr else null;
        var ms: f64 = 0;
        var uploaded: c_int = 0;
        const started = tf_qualcomm_monotonic_ms();
        const status = tf_qualcomm_backend_mlp_f16_resident_once(self.backend, slot, gate.k, gate.m, down.m, n, gate_weight, up_weight, down_weight, input.ptr, output.ptr, &ms, &uploaded);
        self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
        if (status != 0) {
            std.debug.print("resident MLP bridge failed: status {d}, slot {d}, widths {d}/{d}/{d}, n {d}\n", .{ status, slot, gate.k, gate.m, down.m, n });
            return error.BridgeFailed;
        }
        self.projection_ms += ms;
        self.projection_count += 3;
        self.resident_slot_cursor += 1;
        if (uploaded != 0) {
            self.resident_loaded[slot] = true;
            self.resident_weight_uploads += 3;
            self.resident_weight_bytes_uploaded += (gate.k * gate.m + up.k * up.m + down.k * down.m) * @sizeOf(u16);
        } else {
            self.resident_weight_hits += 3;
        }
    }

    fn convertSaveRunF16(self: *Runner, key: [32]u8, k: usize, m: usize, weight: []const f32, input: []const f32, output: []f32) !void {
        const converted = try self.allocator.alloc(u16, weight.len);
        defer self.allocator.free(converted);
        const convert_started = tf_qualcomm_monotonic_ms();
        tf_qualcomm_f32_to_f16(weight.ptr, converted.ptr, weight.len);
        self.weight_convert_ms += tf_qualcomm_monotonic_ms() - convert_started;
        try self.saveCachedF16(key, k, m, converted);
        try self.runWeightF16(k, m, converted, input, output);
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
        if (input.len == 0 or input.len % k != 0 or output.len != m * (input.len / k)) return error.UnexpectedTensor;

        const suh_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.suh", .{base}), .f16, &.{k});
        const svh_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.svh", .{base}), .f16, &.{m});
        const mcg_tensor = try require(self.file, try tensorName(&name_buffer, "{s}.mcg", .{base}), .i32, &.{});
        if (std.mem.readInt(u32, mcg_tensor.bytes[0..4], .little) != exl3.McgMul) return error.UnexpectedCodebook;

        const f16_key: ?[32]u8 = if (self.use_f16_weights) if (self.cache_model_id) |model_id|
            f16ProjectionKey(model_id, base, k, m, words)
        else
            null else null;
        const resident_slot = self.resident_slot_cursor;
        if (self.use_resident_weights and self.resident_loaded[resident_slot]) {
            const started = tf_qualcomm_monotonic_ms();
            if (self.runResidentWeightF16(resident_slot, k, m, null, input, output)) {
                self.projection_wall_ms += tf_qualcomm_monotonic_ms() - started;
                return;
            }
        }
        if (f16_key) |key| {
            if (try self.loadCachedF16(key, k, m)) |loaded_value| {
                var loaded = loaded_value;
                defer loaded.close(self.io);
                return self.runWeightF16(k, m, loaded.weight, input, output);
            }
            self.f16_cache_misses += 1;
        }

        var cache_key: ?[32]u8 = null;
        if (self.cache_dir != null) {
            const trusted_key = if (self.cache_model_id) |model_id| trustedProjectionKey(model_id, base, k, m, words) else null;
            if (trusted_key) |key| if (try self.loadCached(key, k, m)) |loaded_value| {
                self.cache_trusted_hits += 1;
                var loaded = loaded_value;
                defer loaded.close(self.io);
                if (f16_key) |f16_key_value| return self.convertSaveRunF16(f16_key_value, k, m, loaded.weight, input, output);
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
                if (f16_key) |f16_key_value| return self.convertSaveRunF16(f16_key_value, k, m, loaded.weight, input, output);
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
        if (f16_key) |f16_key_value| {
            try self.convertSaveRunF16(f16_key_value, k, m, weight, input, output);
        } else {
            try self.runWeight(k, m, weight, input, output);
        }
    }
};

fn dense(runner: *Runner, name: []const u8, input: []const f32, output: []f32) !void {
    const tensor = runner.file.get(name) orelse return error.MissingTensor;
    if (tensor.dtype != .f16 or tensor.rank != 2 or tensor.dim(0) != output.len or tensor.dim(1) != input.len) return error.UnexpectedTensor;
    if (runner.use_resident_weights) {
        const words: [*]const u16 = @ptrCast(@alignCast(tensor.bytes.ptr));
        tf_qualcomm_dense_f16(words, input.ptr, output.ptr, output.len, input.len);
        return;
    }
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

const DecodeState = struct {
    gdn: []f32,
    conv: []f32,
    keys: []f32,
    values: []f32,
    max_positions: usize,

    fn init(allocator: std.mem.Allocator, max_positions: usize) !DecodeState {
        const gdn = try allocator.alloc(f32, layers * linear_state_elems);
        errdefer allocator.free(gdn);
        const conv = try allocator.alloc(f32, layers * conv_history_rows * conv_channels);
        errdefer allocator.free(conv);
        const keys = try allocator.alloc(f32, layers * max_positions * kv_width);
        errdefer allocator.free(keys);
        const values = try allocator.alloc(f32, layers * max_positions * kv_width);
        errdefer allocator.free(values);
        @memset(gdn, 0.0);
        @memset(conv, 0.0);
        @memset(keys, 0.0);
        @memset(values, 0.0);
        return .{ .gdn = gdn, .conv = conv, .keys = keys, .values = values, .max_positions = max_positions };
    }

    fn deinit(self: *DecodeState, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        allocator.free(self.keys);
        allocator.free(self.conv);
        allocator.free(self.gdn);
        self.* = undefined;
    }

    fn gdnLayer(self: *DecodeState, layer: usize) []f32 {
        const start = layer * linear_state_elems;
        return self.gdn[start .. start + linear_state_elems];
    }

    fn convLayer(self: *DecodeState, layer: usize) []f32 {
        const start = layer * conv_history_rows * conv_channels;
        return self.conv[start .. start + conv_history_rows * conv_channels];
    }

    fn keyPosition(self: *DecodeState, layer: usize, position: usize) []f32 {
        const start = (layer * self.max_positions + position) * kv_width;
        return self.keys[start .. start + kv_width];
    }

    fn valuePosition(self: *DecodeState, layer: usize, position: usize) []f32 {
        const start = (layer * self.max_positions + position) * kv_width;
        return self.values[start .. start + kv_width];
    }
};

fn linearAttention(runner: *Runner, decode: *DecodeState, layer: usize, input: []const f32, output: []f32) !void {
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.linear_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;

    const qkv = try a.alloc(f32, conv_channels);
    defer a.free(qkv);
    const z = try a.alloc(f32, linear_output_width);
    defer a.free(z);
    var qkv_name_buffer: [256]u8 = undefined;
    var z_name_buffer: [256]u8 = undefined;
    const qkv_name = try tensorName(&qkv_name_buffer, "{s}.in_proj_qkv", .{base});
    const z_name = try tensorName(&z_name_buffer, "{s}.in_proj_z", .{base});
    if (runner.use_resident_weights) {
        try runner.projectPairResident(qkv_name, z_name, input, qkv, z);
    } else {
        try runner.project(qkv_name, input, qkv);
        try runner.project(z_name, input, z);
    }
    for (qkv) |*x| x.* = roundBF16(roundF16(x.*));

    var a_gate: [linear_value_heads]f32 = undefined;
    var b_gate: [linear_value_heads]f32 = undefined;
    try dense(runner, try tensorName(&projection_buffer, "{s}.in_proj_a.weight", .{base}), input, &a_gate);
    try dense(runner, try tensorName(&projection_buffer, "{s}.in_proj_b.weight", .{base}), input, &b_gate);

    const conv_name = try tensorName(&projection_buffer, "{s}.conv1d.weight", .{base});
    const conv = try require(runner.file, conv_name, .bf16, &.{ conv_channels, 1, 4 });
    const mixed = try a.alloc(f32, conv_channels);
    defer a.free(mixed);
    const history = decode.convLayer(layer);
    for (mixed, 0..) |*out, c| {
        var sum: f32 = value(conv, c * 4 + 3) * qkv[c];
        for (0..conv_history_rows) |tap| sum += value(conv, c * 4 + tap) * history[tap * conv_channels + c];
        const conv_value = roundBF16(sum);
        const conv_sigmoid = roundBF16(sigmoid(conv_value));
        out.* = roundBF16(conv_value * conv_sigmoid);
    }
    std.mem.copyForwards(f32, history[0 * conv_channels .. 1 * conv_channels], history[1 * conv_channels .. 2 * conv_channels]);
    std.mem.copyForwards(f32, history[1 * conv_channels .. 2 * conv_channels], history[2 * conv_channels .. 3 * conv_channels]);
    @memcpy(history[2 * conv_channels .. 3 * conv_channels], qkv);

    var dt_name: [256]u8 = undefined;
    const dt = try require(runner.file, try tensorName(&dt_name, "{s}.dt_bias", .{base}), .bf16, &.{linear_value_heads});
    var alog_name: [256]u8 = undefined;
    const alog = try require(runner.file, try tensorName(&alog_name, "{s}.A_log", .{base}), .f32, &.{linear_value_heads});
    var beta: [linear_value_heads]f32 = undefined;
    var decay: [linear_value_heads]f32 = undefined;
    for (0..linear_value_heads) |head| {
        beta[head] = roundBF16(sigmoid(b_gate[head]));
        const s = roundBF16(a_gate[head] + value(dt, head));
        const softplus = roundBF16(@max(s, 0.0) + @log(1.0 + @exp(-@abs(s))));
        decay[head] = @exp(-@exp(value(alog, head)) * softplus);
    }
    const q = try a.alloc(f32, linear_query_width);
    defer a.free(q);
    const k = try a.alloc(f32, linear_query_width);
    defer a.free(k);
    const v = try a.alloc(f32, linear_output_width);
    defer a.free(v);
    const q_scale = roundBF16(1.0 / @as(f32, @floatFromInt(linear_key_width)));
    const k_scale = roundBF16(1.0 / @sqrt(@as(f32, @floatFromInt(linear_key_width))));
    for (0..linear_key_heads) |head| {
        var qss: f32 = 0;
        var kss: f32 = 0;
        for (0..linear_key_width) |d| {
            const qv = mixed[head * linear_key_width + d];
            const kv = mixed[linear_query_width + head * linear_key_width + d];
            qss += qv * qv;
            kss += kv * kv;
        }
        const width: f32 = @floatFromInt(linear_key_width);
        const qinv = 1.0 / @sqrt(qss / width + eps / width);
        const kinv = 1.0 / @sqrt(kss / width + eps / width);
        for (0..linear_key_width) |d| {
            q[head * linear_key_width + d] = roundBF16(q_scale * roundBF16(mixed[head * linear_key_width + d] * qinv));
            k[head * linear_key_width + d] = roundBF16(k_scale * roundBF16(mixed[linear_query_width + head * linear_key_width + d] * kinv));
        }
    }
    @memcpy(v, mixed[2 * linear_query_width .. conv_channels]);

    const y = try a.alloc(f32, linear_output_width);
    defer a.free(y);
    const recurrent = decode.gdnLayer(layer);
    if (runner.use_resident_weights) {
        tf_qualcomm_gdn_step_f32(recurrent.ptr, q.ptr, k.ptr, v.ptr, &beta, &decay, y.ptr, linear_key_heads, linear_value_heads, linear_key_width, linear_value_width, runner.cpu_threads);
        for (y) |*out| out.* = roundBF16(out.*);
    } else {
        for (0..linear_value_heads) |head| {
            const key_head = head / (linear_value_heads / linear_key_heads);
            for (0..linear_value_width) |dv| {
                const row = recurrent[(head * linear_value_width + dv) * linear_key_width ..][0..linear_key_width];
                var kv_mem: f32 = 0;
                for (0..linear_key_width) |dk| {
                    row[dk] *= decay[head];
                    kv_mem += row[dk] * k[key_head * linear_key_width + dk];
                }
                const delta = (v[head * linear_value_width + dv] - kv_mem) * beta[head];
                var out: f32 = 0;
                for (0..linear_key_width) |dk| {
                    row[dk] += k[key_head * linear_key_width + dk] * delta;
                    out += row[dk] * q[key_head * linear_key_width + dk];
                }
                y[head * linear_value_width + dv] = roundBF16(out);
            }
        }
    }

    var norm_name: [256]u8 = undefined;
    const norm = try require(runner.file, try tensorName(&norm_name, "{s}.norm.weight", .{base}), .f32, &.{linear_value_width});
    const gated = try a.alloc(f32, linear_output_width);
    defer a.free(gated);
    for (0..linear_value_heads) |head| {
        var ss: f32 = 0;
        for (0..linear_value_width) |d| {
            const x = y[head * linear_value_width + d];
            ss += x * x;
        }
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(linear_value_width)) + eps);
        for (0..linear_value_width) |d| {
            const i = head * linear_value_width + d;
            gated[i] = roundF16(silu(z[i]) * (value(norm, d) * y[i] * inv));
        }
    }
    try runner.project(try tensorName(&projection_buffer, "{s}.out_proj", .{base}), gated, output);
}

fn linearAttentionBatch(runner: *Runner, decode: *DecodeState, layer: usize, input: []const f32, output: []f32) !void {
    if (input.len == 0 or input.len % hidden != 0) return error.UnexpectedTensor;
    const count = input.len / hidden;
    if (output.len != count * hidden) return error.UnexpectedTensor;
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.linear_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;

    const qkv = try a.alloc(f32, count * conv_channels);
    defer a.free(qkv);
    const z = try a.alloc(f32, count * linear_output_width);
    defer a.free(z);
    var qkv_name_buffer: [256]u8 = undefined;
    var z_name_buffer: [256]u8 = undefined;
    const qkv_name = try tensorName(&qkv_name_buffer, "{s}.in_proj_qkv", .{base});
    const z_name = try tensorName(&z_name_buffer, "{s}.in_proj_z", .{base});
    if (runner.use_resident_weights) {
        try runner.projectPairResident(qkv_name, z_name, input, qkv, z);
    } else {
        try runner.project(qkv_name, input, qkv);
        try runner.project(z_name, input, z);
    }
    for (qkv) |*x| x.* = roundBF16(roundF16(x.*));

    const a_gate = try a.alloc(f32, count * linear_value_heads);
    defer a.free(a_gate);
    const b_gate = try a.alloc(f32, count * linear_value_heads);
    defer a.free(b_gate);
    const a_name = try tensorName(&projection_buffer, "{s}.in_proj_a.weight", .{base});
    var b_name_buffer: [256]u8 = undefined;
    const b_name = try tensorName(&b_name_buffer, "{s}.in_proj_b.weight", .{base});
    for (0..count) |position| {
        try dense(runner, a_name, input[position * hidden ..][0..hidden], a_gate[position * linear_value_heads ..][0..linear_value_heads]);
        try dense(runner, b_name, input[position * hidden ..][0..hidden], b_gate[position * linear_value_heads ..][0..linear_value_heads]);
    }

    const conv_name = try tensorName(&projection_buffer, "{s}.conv1d.weight", .{base});
    const conv = try require(runner.file, conv_name, .bf16, &.{ conv_channels, 1, 4 });
    const mixed = try a.alloc(f32, count * conv_channels);
    defer a.free(mixed);
    const history = decode.convLayer(layer);
    for (0..count) |position| {
        const qkv_row = qkv[position * conv_channels ..][0..conv_channels];
        const mixed_row = mixed[position * conv_channels ..][0..conv_channels];
        for (mixed_row, 0..) |*out, c| {
            var sum: f32 = value(conv, c * 4 + 3) * qkv_row[c];
            for (0..conv_history_rows) |tap| sum += value(conv, c * 4 + tap) * history[tap * conv_channels + c];
            const conv_value = roundBF16(sum);
            const conv_sigmoid = roundBF16(sigmoid(conv_value));
            out.* = roundBF16(conv_value * conv_sigmoid);
        }
        std.mem.copyForwards(f32, history[0 * conv_channels .. 1 * conv_channels], history[1 * conv_channels .. 2 * conv_channels]);
        std.mem.copyForwards(f32, history[1 * conv_channels .. 2 * conv_channels], history[2 * conv_channels .. 3 * conv_channels]);
        @memcpy(history[2 * conv_channels .. 3 * conv_channels], qkv_row);
    }

    var dt_name: [256]u8 = undefined;
    const dt = try require(runner.file, try tensorName(&dt_name, "{s}.dt_bias", .{base}), .bf16, &.{linear_value_heads});
    var alog_name: [256]u8 = undefined;
    const alog = try require(runner.file, try tensorName(&alog_name, "{s}.A_log", .{base}), .f32, &.{linear_value_heads});
    const q = try a.alloc(f32, count * linear_query_width);
    defer a.free(q);
    const k = try a.alloc(f32, count * linear_query_width);
    defer a.free(k);
    const v = try a.alloc(f32, count * linear_output_width);
    defer a.free(v);
    const y = try a.alloc(f32, count * linear_output_width);
    defer a.free(y);
    const recurrent = decode.gdnLayer(layer);
    const q_scale = roundBF16(1.0 / @as(f32, @floatFromInt(linear_key_width)));
    const k_scale = roundBF16(1.0 / @sqrt(@as(f32, @floatFromInt(linear_key_width))));
    for (0..count) |position| {
        const mixed_row = mixed[position * conv_channels ..][0..conv_channels];
        const q_row = q[position * linear_query_width ..][0..linear_query_width];
        const k_row = k[position * linear_query_width ..][0..linear_query_width];
        const v_row = v[position * linear_output_width ..][0..linear_output_width];
        const y_row = y[position * linear_output_width ..][0..linear_output_width];
        for (0..linear_key_heads) |head| {
            var qss: f32 = 0;
            var kss: f32 = 0;
            for (0..linear_key_width) |d| {
                const qv = mixed_row[head * linear_key_width + d];
                const kv = mixed_row[linear_query_width + head * linear_key_width + d];
                qss += qv * qv;
                kss += kv * kv;
            }
            const width: f32 = @floatFromInt(linear_key_width);
            const qinv = 1.0 / @sqrt(qss / width + eps / width);
            const kinv = 1.0 / @sqrt(kss / width + eps / width);
            for (0..linear_key_width) |d| {
                q_row[head * linear_key_width + d] = roundBF16(q_scale * roundBF16(mixed_row[head * linear_key_width + d] * qinv));
                k_row[head * linear_key_width + d] = roundBF16(k_scale * roundBF16(mixed_row[linear_query_width + head * linear_key_width + d] * kinv));
            }
        }
        @memcpy(v_row, mixed_row[2 * linear_query_width .. conv_channels]);
        for (0..linear_value_heads) |head| {
            const gate_index = position * linear_value_heads + head;
            const beta = roundBF16(sigmoid(b_gate[gate_index]));
            const s = roundBF16(a_gate[gate_index] + value(dt, head));
            const softplus = roundBF16(@max(s, 0.0) + @log(1.0 + @exp(-@abs(s))));
            const decay = @exp(-@exp(value(alog, head)) * softplus);
            const key_head = head / (linear_value_heads / linear_key_heads);
            for (0..linear_value_width) |dv| {
                const row = recurrent[(head * linear_value_width + dv) * linear_key_width ..][0..linear_key_width];
                var kv_mem: f32 = 0;
                for (0..linear_key_width) |dk| {
                    row[dk] *= decay;
                    kv_mem += row[dk] * k_row[key_head * linear_key_width + dk];
                }
                const delta = (v_row[head * linear_value_width + dv] - kv_mem) * beta;
                var result: f32 = 0;
                for (0..linear_key_width) |dk| {
                    row[dk] += k_row[key_head * linear_key_width + dk] * delta;
                    result += row[dk] * q_row[key_head * linear_key_width + dk];
                }
                y_row[head * linear_value_width + dv] = roundBF16(result);
            }
        }
    }

    var norm_name: [256]u8 = undefined;
    const norm = try require(runner.file, try tensorName(&norm_name, "{s}.norm.weight", .{base}), .f32, &.{linear_value_width});
    const gated = try a.alloc(f32, count * linear_output_width);
    defer a.free(gated);
    for (0..count) |position| {
        const y_row = y[position * linear_output_width ..][0..linear_output_width];
        const z_row = z[position * linear_output_width ..][0..linear_output_width];
        const gated_row = gated[position * linear_output_width ..][0..linear_output_width];
        for (0..linear_value_heads) |head| {
            var ss: f32 = 0;
            for (0..linear_value_width) |d| {
                const x = y_row[head * linear_value_width + d];
                ss += x * x;
            }
            const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(linear_value_width)) + eps);
            for (0..linear_value_width) |d| {
                const i = head * linear_value_width + d;
                gated_row[i] = roundF16(silu(z_row[i]) * (value(norm, d) * y_row[i] * inv));
            }
        }
    }
    try runner.project(try tensorName(&projection_buffer, "{s}.out_proj", .{base}), gated, output);
}

fn headNorm(file: *const st.File, name: []const u8, input: []const f32, heads: usize, output: []f32) !void {
    const tensor = file.get(name) orelse return error.MissingTensor;
    const width = input.len / heads;
    if (input.len != output.len or width != attention_width or tensor.dtype != .bf16 or !tensor.is(.bf16, &.{width})) return error.UnexpectedTensor;
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

fn rotateHalf(input: []const f32, output: []f32, position: usize) void {
    @memcpy(output, input);
    for (0..input.len / attention_width) |head| {
        const base = head * attention_width;
        for (0..rope_width / 2) |i| {
            const angle = @as(f32, @floatFromInt(position)) * std.math.pow(f32, 10_000_000.0, -@as(f32, @floatFromInt(i)) / @as(f32, rope_width / 2));
            const c = @cos(angle);
            const s = @sin(angle);
            const av = input[base + i];
            const bv = input[base + rope_width / 2 + i];
            output[base + i] = roundF16(av * c - bv * s);
            output[base + rope_width / 2 + i] = roundF16(bv * c + av * s);
        }
    }
}

fn fullAttention(runner: *Runner, decode: *DecodeState, layer: usize, position: usize, input: []const f32, output: []f32) !void {
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.self_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;
    const qraw = try a.alloc(f32, 2 * attention_query_width);
    defer a.free(qraw);
    const kraw = try a.alloc(f32, kv_width);
    defer a.free(kraw);
    const vraw = try a.alloc(f32, kv_width);
    defer a.free(vraw);
    var q_name_buffer: [256]u8 = undefined;
    var k_name_buffer: [256]u8 = undefined;
    var v_name_buffer: [256]u8 = undefined;
    const q_name = try tensorName(&q_name_buffer, "{s}.q_proj", .{base});
    const k_name = try tensorName(&k_name_buffer, "{s}.k_proj", .{base});
    const v_name = try tensorName(&v_name_buffer, "{s}.v_proj", .{base});
    if (runner.use_resident_weights) {
        try runner.projectTripleResident(q_name, k_name, v_name, input, qraw, kraw, vraw);
    } else {
        try runner.project(q_name, input, qraw);
        try runner.project(k_name, input, kraw);
        try runner.project(v_name, input, vraw);
    }
    for (qraw) |*x| x.* = roundF16(x.*);
    for (kraw) |*x| x.* = roundF16(x.*);
    for (vraw) |*x| x.* = roundF16(x.*);

    const q = try a.alloc(f32, attention_query_width);
    defer a.free(q);
    const gate = try a.alloc(f32, attention_query_width);
    defer a.free(gate);
    for (0..attention_heads) |head| for (0..attention_width) |d| {
        q[head * attention_width + d] = qraw[head * 2 * attention_width + d];
        gate[head * attention_width + d] = qraw[head * 2 * attention_width + attention_width + d];
    };
    const qnorm = try a.alloc(f32, attention_query_width);
    defer a.free(qnorm);
    const knorm = try a.alloc(f32, kv_width);
    defer a.free(knorm);
    var norm_buffer: [256]u8 = undefined;
    try headNorm(runner.file, try tensorName(&norm_buffer, "{s}.q_norm.weight", .{base}), q, attention_heads, qnorm);
    try headNorm(runner.file, try tensorName(&norm_buffer, "{s}.k_norm.weight", .{base}), kraw, kv_heads, knorm);
    const qrot = try a.alloc(f32, attention_query_width);
    defer a.free(qrot);
    const krot = try a.alloc(f32, kv_width);
    defer a.free(krot);
    rotateHalf(qnorm, qrot, position);
    rotateHalf(knorm, krot, position);
    @memcpy(decode.keyPosition(layer, position), krot);
    @memcpy(decode.valuePosition(layer, position), vraw);

    const attended = try a.alloc(f32, attention_query_width);
    defer a.free(attended);
    const scores = try a.alloc(f32, position + 1);
    defer a.free(scores);
    for (0..attention_heads) |query_head| {
        const kv_head = query_head / (attention_heads / kv_heads);
        var maximum = -std.math.inf(f32);
        for (0..position + 1) |token_position| {
            const cached_key = decode.keyPosition(layer, token_position);
            var score: f32 = 0;
            for (0..attention_width) |d| score += qrot[query_head * attention_width + d] * cached_key[kv_head * attention_width + d];
            scores[token_position] = score / 16.0;
            maximum = @max(maximum, scores[token_position]);
        }
        var denominator: f32 = 0;
        for (scores) |*score| {
            score.* = @exp(score.* - maximum);
            denominator += score.*;
        }
        for (0..attention_width) |d| {
            var sum: f32 = 0;
            for (0..position + 1) |token_position| {
                const cached_value = decode.valuePosition(layer, token_position);
                sum += scores[token_position] / denominator * cached_value[kv_head * attention_width + d];
            }
            const i = query_head * attention_width + d;
            attended[i] = roundF16(roundF16(sum) * sigmoid(gate[i]));
        }
    }
    try runner.project(try tensorName(&projection_buffer, "{s}.o_proj", .{base}), attended, output);
}

fn fullAttentionBatch(runner: *Runner, decode: *DecodeState, layer: usize, start_position: usize, input: []const f32, output: []f32) !void {
    if (input.len == 0 or input.len % hidden != 0) return error.UnexpectedTensor;
    const count = input.len / hidden;
    if (output.len != count * hidden or start_position + count > decode.max_positions) return error.UnexpectedTensor;
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.self_attn", .{ prefix, layer });
    var projection_buffer: [256]u8 = undefined;
    const qraw = try a.alloc(f32, count * 2 * attention_query_width);
    defer a.free(qraw);
    const kraw = try a.alloc(f32, count * kv_width);
    defer a.free(kraw);
    const vraw = try a.alloc(f32, count * kv_width);
    defer a.free(vraw);
    var q_name_buffer: [256]u8 = undefined;
    var k_name_buffer: [256]u8 = undefined;
    var v_name_buffer: [256]u8 = undefined;
    const q_name = try tensorName(&q_name_buffer, "{s}.q_proj", .{base});
    const k_name = try tensorName(&k_name_buffer, "{s}.k_proj", .{base});
    const v_name = try tensorName(&v_name_buffer, "{s}.v_proj", .{base});
    if (runner.use_resident_weights) {
        try runner.projectTripleResident(q_name, k_name, v_name, input, qraw, kraw, vraw);
    } else {
        try runner.project(q_name, input, qraw);
        try runner.project(k_name, input, kraw);
        try runner.project(v_name, input, vraw);
    }
    for (qraw) |*x| x.* = roundF16(x.*);
    for (kraw) |*x| x.* = roundF16(x.*);
    for (vraw) |*x| x.* = roundF16(x.*);

    const q = try a.alloc(f32, count * attention_query_width);
    defer a.free(q);
    const gate = try a.alloc(f32, count * attention_query_width);
    defer a.free(gate);
    const qnorm = try a.alloc(f32, count * attention_query_width);
    defer a.free(qnorm);
    const knorm = try a.alloc(f32, count * kv_width);
    defer a.free(knorm);
    const qrot = try a.alloc(f32, count * attention_query_width);
    defer a.free(qrot);
    const krot = try a.alloc(f32, count * kv_width);
    defer a.free(krot);
    var q_norm_name_buffer: [256]u8 = undefined;
    const q_norm_name = try tensorName(&q_norm_name_buffer, "{s}.q_norm.weight", .{base});
    var k_norm_name_buffer: [256]u8 = undefined;
    const k_norm_name = try tensorName(&k_norm_name_buffer, "{s}.k_norm.weight", .{base});
    for (0..count) |row| {
        const qraw_row = qraw[row * 2 * attention_query_width ..][0 .. 2 * attention_query_width];
        const q_row = q[row * attention_query_width ..][0..attention_query_width];
        const gate_row = gate[row * attention_query_width ..][0..attention_query_width];
        for (0..attention_heads) |head| for (0..attention_width) |d| {
            q_row[head * attention_width + d] = qraw_row[head * 2 * attention_width + d];
            gate_row[head * attention_width + d] = qraw_row[head * 2 * attention_width + attention_width + d];
        };
        const qnorm_row = qnorm[row * attention_query_width ..][0..attention_query_width];
        const knorm_row = knorm[row * kv_width ..][0..kv_width];
        try headNorm(runner.file, q_norm_name, q_row, attention_heads, qnorm_row);
        try headNorm(runner.file, k_norm_name, kraw[row * kv_width ..][0..kv_width], kv_heads, knorm_row);
        const position = start_position + row;
        const qrot_row = qrot[row * attention_query_width ..][0..attention_query_width];
        const krot_row = krot[row * kv_width ..][0..kv_width];
        rotateHalf(qnorm_row, qrot_row, position);
        rotateHalf(knorm_row, krot_row, position);
        @memcpy(decode.keyPosition(layer, position), krot_row);
        @memcpy(decode.valuePosition(layer, position), vraw[row * kv_width ..][0..kv_width]);
    }

    const attended = try a.alloc(f32, count * attention_query_width);
    defer a.free(attended);
    const scores = try a.alloc(f32, start_position + count);
    defer a.free(scores);
    for (0..count) |row| {
        const position = start_position + row;
        const qrot_row = qrot[row * attention_query_width ..][0..attention_query_width];
        const gate_row = gate[row * attention_query_width ..][0..attention_query_width];
        const attended_row = attended[row * attention_query_width ..][0..attention_query_width];
        for (0..attention_heads) |query_head| {
            const kv_head = query_head / (attention_heads / kv_heads);
            var maximum = -std.math.inf(f32);
            for (0..position + 1) |token_position| {
                const cached_key = decode.keyPosition(layer, token_position);
                var score: f32 = 0;
                for (0..attention_width) |d| score += qrot_row[query_head * attention_width + d] * cached_key[kv_head * attention_width + d];
                scores[token_position] = score / 16.0;
                maximum = @max(maximum, scores[token_position]);
            }
            var denominator: f32 = 0;
            for (scores[0 .. position + 1]) |*score| {
                score.* = @exp(score.* - maximum);
                denominator += score.*;
            }
            for (0..attention_width) |d| {
                var sum: f32 = 0;
                for (0..position + 1) |token_position| {
                    const cached_value = decode.valuePosition(layer, token_position);
                    sum += scores[token_position] / denominator * cached_value[kv_head * attention_width + d];
                }
                const i = query_head * attention_width + d;
                attended_row[i] = roundF16(roundF16(sum) * sigmoid(gate_row[i]));
            }
        }
    }
    try runner.project(try tensorName(&projection_buffer, "{s}.o_proj", .{base}), attended, output);
}

fn mlp(runner: *Runner, layer: usize, input: []const f32, output: []f32) !void {
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.mlp", .{ prefix, layer });
    var gate_name_buffer: [256]u8 = undefined;
    var up_name_buffer: [256]u8 = undefined;
    var down_name_buffer: [256]u8 = undefined;
    const gate_name = try tensorName(&gate_name_buffer, "{s}.gate_proj", .{base});
    const up_name = try tensorName(&up_name_buffer, "{s}.up_proj", .{base});
    const down_name = try tensorName(&down_name_buffer, "{s}.down_proj", .{base});
    if (runner.use_resident_weights) return runner.mlpResident(gate_name, up_name, down_name, input, output);
    const gate = try a.alloc(f32, intermediate);
    defer a.free(gate);
    const up = try a.alloc(f32, intermediate);
    defer a.free(up);
    const activated = try a.alloc(f32, intermediate);
    defer a.free(activated);
    try runner.project(gate_name, input, gate);
    try runner.project(up_name, input, up);
    for (activated, 0..) |*out, i| out.* = roundF16(silu(roundF16(gate[i])) * roundF16(up[i]));
    try runner.project(down_name, activated, output);
}

fn mlpBatch(runner: *Runner, layer: usize, input: []const f32, output: []f32) !void {
    if (input.len == 0 or input.len % hidden != 0) return error.UnexpectedTensor;
    const count = input.len / hidden;
    if (output.len != count * hidden) return error.UnexpectedTensor;
    const a = runner.allocator;
    var base_buffer: [256]u8 = undefined;
    const base = try tensorName(&base_buffer, "{s}.layers.{d}.mlp", .{ prefix, layer });
    var gate_name_buffer: [256]u8 = undefined;
    var up_name_buffer: [256]u8 = undefined;
    var down_name_buffer: [256]u8 = undefined;
    const gate_name = try tensorName(&gate_name_buffer, "{s}.gate_proj", .{base});
    const up_name = try tensorName(&up_name_buffer, "{s}.up_proj", .{base});
    const down_name = try tensorName(&down_name_buffer, "{s}.down_proj", .{base});
    if (runner.use_resident_weights) return runner.mlpResident(gate_name, up_name, down_name, input, output);
    const gate = try a.alloc(f32, count * intermediate);
    defer a.free(gate);
    const up = try a.alloc(f32, count * intermediate);
    defer a.free(up);
    const activated = try a.alloc(f32, count * intermediate);
    defer a.free(activated);
    try runner.project(gate_name, input, gate);
    try runner.project(up_name, input, up);
    for (activated, 0..) |*out, i| out.* = roundF16(silu(roundF16(gate[i])) * roundF16(up[i]));
    try runner.project(down_name, activated, output);
}

const Top = struct { id: usize, logit: f32 };

fn tiedHeadScalar(file: *const st.File, state: []const f32) ![5]Top {
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

fn tiedHead(runner: *Runner, state: []const f32, cpu_threads: i32) ![5]Top {
    const embedding = try require(runner.file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
    const embedding_words: [*]const u16 = @ptrCast(@alignCast(embedding.bytes.ptr));
    if (runner.use_resident_head) {
        const all_logits = try runner.allocator.alloc(f32, vocab);
        defer runner.allocator.free(all_logits);
        var compute_ms: f64 = 0;
        var uploaded: c_int = 0;
        const weight_ptr: ?[*]const u16 = if (runner.resident_head_loaded) null else embedding_words;
        const status = tf_qualcomm_backend_matmul_f16_resident_once(runner.backend, projections_per_position, hidden, vocab, 1, weight_ptr, state.ptr, all_logits.ptr, &compute_ms, &uploaded);
        if (status == 0) {
            runner.resident_head_compute_ms += compute_ms;
            if (uploaded != 0) {
                runner.resident_head_loaded = true;
                runner.resident_head_uploads += 1;
            } else {
                runner.resident_head_hits += 1;
            }
            var top: [5]Top = @splat(.{ .id = 0, .logit = -std.math.inf(f32) });
            for (all_logits, 0..) |logit, token| {
                if (logit > top[4].logit) {
                    top[4] = .{ .id = token, .logit = logit };
                    var i: usize = 4;
                    while (i > 0 and top[i].logit > top[i - 1].logit) : (i -= 1) std.mem.swap(Top, &top[i], &top[i - 1]);
                }
            }
            return top;
        }
        runner.resident_head_loaded = false;
        runner.resident_head_fallbacks += 1;
    }
    var ids: [5]usize = undefined;
    var logits: [5]f32 = undefined;
    if (tf_qualcomm_tied_head_f16(embedding_words, state.ptr, vocab, hidden, cpu_threads, &ids, &logits) != 0) {
        return tiedHeadScalar(runner.file, state);
    }
    var top: [5]Top = undefined;
    for (&top, ids, logits) |*candidate, id, logit| candidate.* = .{ .id = id, .logit = logit };
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
    if (args.len < 5 or args.len > 11) {
        std.debug.print("usage: {s} cpu|htp MODEL_SAFETENSORS TOKEN_ID CPU_THREADS [DECODED_CACHE_DIR [MODEL_SHA256 [STEPS [PROMPT_SUFFIX_TOKEN_IDS [serve|serve-resident [REQUESTS]]]]]]\n", .{args[0]});
        return 64;
    }
    if (!std.mem.eql(u8, args[1], "cpu") and !std.mem.eql(u8, args[1], "htp")) return error.InvalidBackend;
    const token = try std.fmt.parseInt(usize, args[3], 10);
    const threads = try std.fmt.parseInt(i32, args[4], 10);
    const steps = if (args.len >= 8) try std.fmt.parseInt(usize, args[7], 10) else 1;
    const suffix_count = if (args.len >= 9) std.mem.count(u8, args[8], ",") + 1 else 0;
    const serve_mode = args.len >= 10 and (std.mem.eql(u8, args[9], "serve") or std.mem.eql(u8, args[9], "serve-resident"));
    const resident_mode = args.len >= 10 and std.mem.eql(u8, args[9], "serve-resident");
    const requests = if (args.len == 11) try std.fmt.parseInt(usize, args[10], 10) else 1;
    if (args.len >= 10 and !serve_mode) return error.InvalidArgument;
    const prompt_len = 1 + suffix_count;
    const positions = prompt_len + steps - 1;
    if (token >= vocab or threads <= 0 or steps == 0 or steps > 32 or prompt_len > 128 or requests == 0 or requests > 16 or (requests > 1 and !resident_mode)) return error.InvalidArgument;

    var file = try st.File.open(init.gpa, init.io, args[2]);
    defer file.close(init.io);
    const embedding = try require(&file, prefix ++ ".embed_tokens.weight", .f16, &.{ vocab, hidden });
    const backend = tf_qualcomm_backend_create(args[1].ptr, threads) orelse return error.BackendInitFailed;
    defer tf_qualcomm_backend_destroy(backend);
    const cache_dir: ?[]const u8 = if (args.len >= 6) args[5] else null;
    const cache_model_id: ?[]const u8 = if (args.len >= 7) args[6] else null;
    if (cache_model_id) |id| {
        if (id.len != 64) return error.InvalidModelSha256;
        for (id) |c| if (!std.ascii.isHex(c)) return error.InvalidModelSha256;
    }
    if (cache_dir) |dir| try std.Io.Dir.cwd().createDirPath(init.io, dir);
    var runner: Runner = .{ .allocator = init.gpa, .io = init.io, .file = &file, .backend = backend, .cpu_threads = @intCast(threads), .use_f16_weights = serve_mode, .use_resident_weights = resident_mode, .use_resident_head = resident_mode and requests > 1, .cache_dir = cache_dir, .cache_model_id = cache_model_id };
    for (0..requests) |request_index| {
        runner.resetMetrics();
        const started = tf_qualcomm_monotonic_ms();
        var decode = try DecodeState.init(init.gpa, positions);

        const state = try init.gpa.alloc(f32, hidden);
        const normed = try init.gpa.alloc(f32, hidden);
        const update = try init.gpa.alloc(f32, hidden);
        const sequence = try init.gpa.alloc(usize, positions + 1);
        sequence[0] = token;
        if (args.len >= 9) {
            var suffix = std.mem.splitScalar(u8, args[8], ',');
            var i: usize = 1;
            while (suffix.next()) |encoded| : (i += 1) {
                const suffix_token = try std.fmt.parseInt(usize, encoded, 10);
                if (suffix_token >= vocab) return error.InvalidArgument;
                sequence[i] = suffix_token;
            }
        }
        var name_buffer: [256]u8 = undefined;
        var total_head_ms: f64 = 0;
        var skipped_heads: usize = 0;
        var decode_start: usize = 0;
        const head_kind = if (runner.use_resident_head) "tied-f16-htp-resident" else "tied-f16-neon";

        if (serve_mode and prompt_len > 1) {
            const prefill_started = tf_qualcomm_monotonic_ms();
            const projection_start = runner.projection_count;
            const compute_start = runner.projection_ms;
            const wall_start = runner.projection_wall_ms;
            const prompt_states = try init.gpa.alloc(f32, prompt_len * hidden);
            const prompt_normed = try init.gpa.alloc(f32, prompt_len * hidden);
            const prompt_update = try init.gpa.alloc(f32, prompt_len * hidden);
            for (0..prompt_len) |position| {
                const row = prompt_states[position * hidden ..][0..hidden];
                for (row, 0..) |*x, d| x.* = f16Value(embedding.bytes, sequence[position] * hidden + d);
            }
            for (0..layers) |layer| {
                const input_norm = try tensorName(&name_buffer, "{s}.layers.{d}.input_layernorm.weight", .{ prefix, layer });
                for (0..prompt_len) |position| {
                    try rmsNorm(&file, input_norm, prompt_states[position * hidden ..][0..hidden], prompt_normed[position * hidden ..][0..hidden], 1.0);
                }
                if ((layer + 1) % 4 == 0) {
                    try fullAttentionBatch(&runner, &decode, layer, 0, prompt_normed, prompt_update);
                } else {
                    try linearAttentionBatch(&runner, &decode, layer, prompt_normed, prompt_update);
                }
                residualAdd(prompt_states, prompt_update);
                const post_norm = try tensorName(&name_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
                for (0..prompt_len) |position| {
                    try rmsNorm(&file, post_norm, prompt_states[position * hidden ..][0..hidden], prompt_normed[position * hidden ..][0..hidden], 1.0);
                }
                try mlpBatch(&runner, layer, prompt_normed, prompt_update);
                residualAdd(prompt_states, prompt_update);
                std.debug.print("tensorfold prefill {s}: tokens {d}, layer {d}/{d}, projections {d}\n", .{ args[1], prompt_len, layer + 1, layers, runner.projection_count });
            }

            try rmsNorm(&file, prefix ++ ".norm.weight", prompt_states[(prompt_len - 1) * hidden ..][0..hidden], normed, 1.0);
            const head_started = tf_qualcomm_monotonic_ms();
            const top = try tiedHead(&runner, normed, threads);
            const head_ms = tf_qualcomm_monotonic_ms() - head_started;
            total_head_ms += head_ms;
            skipped_heads = prompt_len - 1;
            sequence[prompt_len] = top[0].id;
            const prefill_elapsed = tf_qualcomm_monotonic_ms() - prefill_started;
            std.debug.print(
                "{{\"tensorfold\":\"qualcomm-exl3-prefill\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"prompt_tokens\":{d},\"batch\":{d},\"layers\":{d},\"projections\":{d},\"head\":\"{s}\",\"heads_skipped\":{d},\"next_token\":{d},\"top5\":[[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}]],\"state_f16_fnv1a\":\"{x}\",\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"head_ms\":{d:.3},\"elapsed_ms\":{d:.3},\"passed\":true}}\n",
                .{ profile.name, args[1], prompt_len, prompt_len, layers, runner.projection_count - projection_start, head_kind, skipped_heads, top[0].id, top[0].id, top[0].logit, top[1].id, top[1].logit, top[2].id, top[2].logit, top[3].id, top[3].logit, top[4].id, top[4].logit, stateHash(normed), runner.projection_ms - compute_start, runner.projection_wall_ms - wall_start, head_ms, prefill_elapsed },
            );
            init.gpa.free(prompt_update);
            init.gpa.free(prompt_normed);
            init.gpa.free(prompt_states);
            decode_start = prompt_len;
        }

        for (decode_start..positions) |position| {
            const input_token = sequence[position];
            const step_started = tf_qualcomm_monotonic_ms();
            const projection_start = runner.projection_count;
            const compute_start = runner.projection_ms;
            const wall_start = runner.projection_wall_ms;
            for (state, 0..) |*x, d| x.* = f16Value(embedding.bytes, input_token * hidden + d);
            runner.resident_slot_cursor = 0;
            for (0..layers) |layer| {
                const input_norm = try tensorName(&name_buffer, "{s}.layers.{d}.input_layernorm.weight", .{ prefix, layer });
                try rmsNorm(&file, input_norm, state, normed, 1.0);
                if ((layer + 1) % 4 == 0) {
                    try fullAttention(&runner, &decode, layer, position, normed, update);
                } else {
                    try linearAttention(&runner, &decode, layer, normed, update);
                }
                residualAdd(state, update);
                const post_norm = try tensorName(&name_buffer, "{s}.layers.{d}.post_attention_layernorm.weight", .{ prefix, layer });
                try rmsNorm(&file, post_norm, state, normed, 1.0);
                try mlp(&runner, layer, normed, update);
                residualAdd(state, update);
                std.debug.print("tensorfold forward {s}: position {d}, layer {d}/{d}, projections {d}\n", .{ args[1], position, layer + 1, layers, runner.projection_count });
            }

            try rmsNorm(&file, prefix ++ ".norm.weight", state, normed, 1.0);
            const forced_next = position + 1 < prompt_len;
            if (forced_next and serve_mode) {
                skipped_heads += 1;
                const step_elapsed = tf_qualcomm_monotonic_ms() - step_started;
                std.debug.print(
                    "{{\"tensorfold\":\"qualcomm-exl3-forward-step\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"input_token\":{d},\"position\":{d},\"forced_next\":true,\"layers\":{d},\"projections\":{d},\"head\":\"skipped-teacher-forced\",\"state_f16_fnv1a\":\"{x}\",\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"head_ms\":0.000,\"elapsed_ms\":{d:.3},\"passed\":true}}\n",
                    .{ profile.name, args[1], input_token, position, layers, runner.projection_count - projection_start, stateHash(normed), runner.projection_ms - compute_start, runner.projection_wall_ms - wall_start, step_elapsed },
                );
                continue;
            }
            const head_started = tf_qualcomm_monotonic_ms();
            const top = try tiedHead(&runner, normed, threads);
            const head_ms = tf_qualcomm_monotonic_ms() - head_started;
            total_head_ms += head_ms;
            if (!forced_next) sequence[position + 1] = top[0].id;
            const step_elapsed = tf_qualcomm_monotonic_ms() - step_started;
            std.debug.print(
                "{{\"tensorfold\":\"qualcomm-exl3-forward-step\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"input_token\":{d},\"position\":{d},\"forced_next\":{},\"layers\":{d},\"projections\":{d},\"head\":\"{s}\",\"next_token\":{d},\"top5\":[[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}],[{d},{d}]],\"state_f16_fnv1a\":\"{x}\",\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"head_ms\":{d:.3},\"elapsed_ms\":{d:.3},\"passed\":true}}\n",
                .{ profile.name, args[1], input_token, position, forced_next, layers, runner.projection_count - projection_start, head_kind, top[0].id, top[0].id, top[0].logit, top[1].id, top[1].logit, top[2].id, top[2].logit, top[3].id, top[3].logit, top[4].id, top[4].logit, stateHash(normed), runner.projection_ms - compute_start, runner.projection_wall_ms - wall_start, head_ms, step_elapsed },
            );
        }
        const elapsed = tf_qualcomm_monotonic_ms() - started;
        std.debug.print("{{\"tensorfold\":\"qualcomm-exl3-generate\",\"model_profile\":\"{s}\",\"backend\":\"{s}\",\"request_index\":{d},\"requests\":{d},\"checkpoint_tensors\":{d},\"prompt_tokens\":{d},\"generated_tokens\":{d},\"positions\":{d},\"tokens\":[", .{ profile.name, args[1], request_index, requests, file.names.count(), prompt_len, steps, positions });
        for (sequence, 0..) |generated_token, i| {
            if (i != 0) std.debug.print(",", .{});
            std.debug.print("{d}", .{generated_token});
        }
        std.debug.print(
            "],\"projections\":{d},\"matmul_compute_ms\":{d:.3},\"projection_wall_ms\":{d:.3},\"weight_convert_ms\":{d:.3},\"head_ms\":{d:.3},\"heads_skipped\":{d},\"dequant_ms\":{d:.3},\"cache_hash_ms\":{d:.3},\"cache_read_ms\":{d:.3},\"cache_write_ms\":{d:.3},\"cache_hits\":{d},\"cache_trusted_hits\":{d},\"cache_misses\":{d},\"cache_aliases_created\":{d},\"cache_bytes_read\":{d},\"cache_bytes_written\":{d},\"f16_cache_hits\":{d},\"f16_cache_misses\":{d},\"f16_cache_bytes_read\":{d},\"f16_cache_bytes_written\":{d},\"resident_weight_uploads\":{d},\"resident_weight_hits\":{d},\"resident_weight_fallbacks\":{d},\"resident_weight_bytes_uploaded\":{d},\"resident_head_uploads\":{d},\"resident_head_hits\":{d},\"resident_head_fallbacks\":{d},\"resident_head_compute_ms\":{d:.3},\"elapsed_ms\":{d:.3},\"positions_per_second\":{d:.6},\"end_to_end_generated_tokens_per_second\":{d:.6},\"peak_rss_kib\":{d},\"passed\":true}}\n",
            .{ runner.projection_count, runner.projection_ms, runner.projection_wall_ms, runner.weight_convert_ms, total_head_ms, skipped_heads, runner.dequant_ms, runner.cache_hash_ms, runner.cache_read_ms, runner.cache_write_ms, runner.cache_hits, runner.cache_trusted_hits, runner.cache_misses, runner.cache_aliases_created, runner.cache_bytes_read, runner.cache_bytes_written, runner.f16_cache_hits, runner.f16_cache_misses, runner.f16_cache_bytes_read, runner.f16_cache_bytes_written, runner.resident_weight_uploads, runner.resident_weight_hits, runner.resident_weight_fallbacks, runner.resident_weight_bytes_uploaded, runner.resident_head_uploads, runner.resident_head_hits, runner.resident_head_fallbacks, runner.resident_head_compute_ms, elapsed, @as(f64, @floatFromInt(positions)) * 1000.0 / elapsed, @as(f64, @floatFromInt(steps)) * 1000.0 / elapsed, tf_qualcomm_peak_rss_kib() },
        );
        decode.deinit(init.gpa);
        init.gpa.free(sequence);
        init.gpa.free(update);
        init.gpa.free(normed);
        init.gpa.free(state);
    }
    return 0;
}
