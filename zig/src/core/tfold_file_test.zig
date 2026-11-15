//! Structural, integrity and round-trip tests for the native TFOLD v0.2 reader.
const std = @import("std");
const tf = @import("tfold_file.zig");
const st = @import("safetensors.zig");
const t = std.testing;
const a = t.allocator;

/// Build a v0.2 blob with the given tensors inlined in order. Each tensor is
/// placed at the next 64-byte-aligned offset after the body start. `tensors`
/// is a list of (name, dtype_str, shape, nbytes).
const TensorSpec = struct {
    name: []const u8,
    dtype: []const u8 = "u8",
    shape: []const i64 = &.{1},
    nbytes: usize,
    offset_override: ?usize = null, // for negative tests
};

fn buildHeaderAndLayout(
    gpa: std.mem.Allocator,
    tensors: []const TensorSpec,
) !struct { header: []u8, offsets: []usize, total_body: usize } {
    // First pass: produce a candidate header with offset placeholders, then
    // compute the true body_start from its length, then re-emit.
    var offsets = try gpa.alloc(usize, tensors.len);
    // Guess header length by rendering with offset=0; refine.
    var last_hlen: usize = 0;
    var placed_body: usize = 0;
    var header: []u8 = &.{};
    var pass: usize = 0;
    while (pass < 4) : (pass += 1) {
        const hdr_end_guess = tf.PREAMBLE_NBYTES + last_hlen;
        const pad_guess = (tf.ALIGN - hdr_end_guess % tf.ALIGN) % tf.ALIGN;
        const body_start = hdr_end_guess + pad_guess;
        // Lay out offsets.
        var cur: usize = body_start;
        for (tensors, 0..) |ts, i| {
            const o = ts.offset_override orelse cur;
            offsets[i] = o;
            cur = o + ts.nbytes;
            // Round up to next ALIGN for next tensor.
            if (cur % tf.ALIGN != 0) cur += tf.ALIGN - cur % tf.ALIGN;
        }
        placed_body = cur;
        // Build header JSON.
        if (header.len > 0) gpa.free(header);
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa,
            \\{"tfold":{"version":"0.2"},"quant":{"scheme":"exl3_mul1","bits_per_weight":8.0,"block_size":16},"model":{"source_sha256":"0000000000000000000000000000000000000000000000000000000000000000","vocab_size":1,"parameters":1},"tensors":[
        );
        for (tensors, 0..) |ts, i| {
            if (i > 0) try buf.append(gpa, ',');
            const prefix = try std.fmt.allocPrint(gpa,
                "{{\"name\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[",
                .{ ts.name, ts.dtype });
            defer gpa.free(prefix);
            try buf.appendSlice(gpa, prefix);
            for (ts.shape, 0..) |d, j| {
                if (j > 0) try buf.append(gpa, ',');
                const n = try std.fmt.allocPrint(gpa, "{d}", .{d});
                defer gpa.free(n);
                try buf.appendSlice(gpa, n);
            }
            const suffix = try std.fmt.allocPrint(gpa,
                "],\"offset\":{d},\"nbytes\":{d},\"sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\"}}",
                .{ offsets[i], ts.nbytes });
            defer gpa.free(suffix);
            try buf.appendSlice(gpa, suffix);
        }
        try buf.appendSlice(gpa, "]}");
        header = try gpa.dupe(u8, buf.items);
        if (header.len == last_hlen) break;
        last_hlen = header.len;
    }
    return .{ .header = header, .offsets = offsets, .total_body = placed_body };
}

fn makeValid(gpa: std.mem.Allocator, tensors: []const TensorSpec) ![]u8 {
    const h = try buildHeaderAndLayout(gpa, tensors);
    defer gpa.free(h.header);
    defer gpa.free(h.offsets);

    const hdr_end = tf.PREAMBLE_NBYTES + h.header.len;
    const pad = (tf.ALIGN - hdr_end % tf.ALIGN) % tf.ALIGN;
    const total = hdr_end + pad + (h.total_body - (hdr_end + pad)) + tf.TRAILER_NBYTES;

    const buf = try gpa.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..8], &tf.MAGIC);
    std.mem.writeInt(u32, buf[8..12], tf.VERSION_U32, .little);
    std.mem.writeInt(u32, buf[12..16], @intCast(h.header.len), .little);
    @memcpy(buf[tf.PREAMBLE_NBYTES..][0..h.header.len], h.header);
    // Body already zeroed.
    const fnv = tf.fnv1a64(tf.FNV_OFFSET_BASIS, buf[0 .. total - tf.TRAILER_NBYTES]);
    std.mem.writeInt(u64, buf[total - tf.TRAILER_NBYTES ..][0..8], fnv, .little);
    return buf;
}

fn writeFile(name: []const u8, bytes: []const u8) ![]const u8 {
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{name});
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = bytes });
    return path;
}

test "tfold_file: a valid 128-byte scalar file opens and returns one tensor" {
    const bytes = try makeValid(a, &.{.{ .name = "p", .nbytes = 128, .shape = &.{128} }});
    defer a.free(bytes);
    const path = try writeFile("tfold_valid.tfold", bytes);
    defer a.free(path);
    var f = try tf.openPrefixes(a, t.io, path, &.{});
    defer f.close(t.io);
    try t.expectEqual(@as(usize, 1), f.names.count());
    const got = f.get("p") orelse return error.MissingTensor;
    try t.expectEqual(@as(usize, 128), got.bytes.len);
    try t.expectEqual(st.DType.u8, got.dtype);
    for (got.bytes) |b| try t.expectEqual(@as(u8, 0), b);
}

test "tfold_file: bad magic is refused" {
    var bytes = try makeValid(a, &.{.{ .name = "p", .nbytes = 128, .shape = &.{128} }});
    defer a.free(bytes);
    bytes[0] = 'X';
    const path = try writeFile("tfold_badmagic.tfold", bytes);
    defer a.free(path);
    try t.expectError(error.TfoldBadMagic, tf.openPrefixes(a, t.io, path, &.{}));
}

test "tfold_file: a v0.1 major version is refused" {
    var bytes = try makeValid(a, &.{.{ .name = "p", .nbytes = 128, .shape = &.{128} }});
    defer a.free(bytes);
    std.mem.writeInt(u32, bytes[8..12], 0x00010000, .little);
    const tot = bytes.len;
    const fnv = tf.fnv1a64(tf.FNV_OFFSET_BASIS, bytes[0 .. tot - tf.TRAILER_NBYTES]);
    std.mem.writeInt(u64, bytes[tot - tf.TRAILER_NBYTES ..][0..8], fnv, .little);
    const path = try writeFile("tfold_badversion.tfold", bytes);
    defer a.free(path);
    try t.expectError(error.TfoldBadVersion, tf.openPrefixes(a, t.io, path, &.{}));
}

test "tfold_file: header overflow refused when hlen declares past-EOF" {
    var bytes = try makeValid(a, &.{.{ .name = "p", .nbytes = 128, .shape = &.{128} }});
    defer a.free(bytes);
    std.mem.writeInt(u32, bytes[12..16], 0x7FFFFFFF, .little);
    const path = try writeFile("tfold_hdroverflow.tfold", bytes);
    defer a.free(path);
    try t.expectError(error.TfoldHeaderOverflow, tf.openPrefixes(a, t.io, path, &.{}));
}

test "tfold_file: a truncated file (missing trailer) is refused" {
    const bytes = try makeValid(a, &.{.{ .name = "p", .nbytes = 128, .shape = &.{128} }});
    defer a.free(bytes);
    const path = try writeFile("tfold_trunc.tfold", bytes[0 .. bytes.len - 3]);
    defer a.free(path);
    const r = tf.openPrefixes(a, t.io, path, &.{});
    try t.expect(if (r) |f| blk: {
        var mut = f;
        mut.close(t.io);
        break :blk false;
    } else |err| switch (err) {
        error.TfoldIntegrity, error.TfoldTensorOverflow, error.TfoldHeaderOverflow => true,
        else => false,
    });
}

test "tfold_file: FNV trailer mismatch refused" {
    var bytes = try makeValid(a, &.{.{ .name = "p", .nbytes = 128, .shape = &.{128} }});
    defer a.free(bytes);
    // Corrupt one body byte at a known in-body offset.
    bytes[bytes.len - 20] ^= 0xFF;
    const path = try writeFile("tfold_fnv.tfold", bytes);
    defer a.free(path);
    try t.expectError(error.TfoldIntegrity, tf.openPrefixes(a, t.io, path, &.{}));
}

test "tfold_file: overlapping tensors refused" {
    // Build a custom blob where tensor `b` occupies the same bytes as `a`.
    const header = try std.fmt.allocPrint(a,
        \\{{"tfold":{{"version":"0.2"}},"quant":{{"scheme":"exl3_mul1","bits_per_weight":8.0,"block_size":16}},"model":{{"source_sha256":"0000000000000000000000000000000000000000000000000000000000000000","vocab_size":1,"parameters":1}},"tensors":[{{"name":"a","dtype":"u8","shape":[128],"offset":{d},"nbytes":128,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}},{{"name":"b","dtype":"u8","shape":[128],"offset":{d},"nbytes":128,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}]}}
    , .{ 1024, 1024 }); // pick an offset certain to be >= body_start when hlen ~450
    defer a.free(header);
    // Compute body_start from the header and ensure 320 is inside the body.
    const hdr_end = tf.PREAMBLE_NBYTES + header.len;
    const pad = (tf.ALIGN - hdr_end % tf.ALIGN) % tf.ALIGN;
    const body_start = hdr_end + pad;
    try t.expect(body_start <= 1024);
    const total = 1024 + 128 + tf.TRAILER_NBYTES;
    const buf = try a.alloc(u8, total);
    defer a.free(buf);
    @memset(buf, 0);
    @memcpy(buf[0..8], &tf.MAGIC);
    std.mem.writeInt(u32, buf[8..12], tf.VERSION_U32, .little);
    std.mem.writeInt(u32, buf[12..16], @intCast(header.len), .little);
    @memcpy(buf[tf.PREAMBLE_NBYTES..][0..header.len], header);
    const fnv = tf.fnv1a64(tf.FNV_OFFSET_BASIS, buf[0 .. total - tf.TRAILER_NBYTES]);
    std.mem.writeInt(u64, buf[total - tf.TRAILER_NBYTES ..][0..8], fnv, .little);
    const path = try writeFile("tfold_overlap.tfold", buf);
    defer a.free(path);
    try t.expectError(error.TfoldTensorOverlap, tf.openPrefixes(a, t.io, path, &.{}));
}

test "tfold_file: misaligned tensor offset refused" {
    // Place a tensor at body_start + 1 (not ALIGN=64 aligned).
    const header = try std.fmt.allocPrint(a,
        \\{{"tfold":{{"version":"0.2"}},"quant":{{"scheme":"exl3_mul1","bits_per_weight":8.0,"block_size":16}},"model":{{"source_sha256":"0000000000000000000000000000000000000000000000000000000000000000","vocab_size":1,"parameters":1}},"tensors":[{{"name":"p","dtype":"u8","shape":[128],"offset":{d},"nbytes":128,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}]}}
    , .{1025}); // 321 is not 64-aligned
    defer a.free(header);
    const hdr_end = tf.PREAMBLE_NBYTES + header.len;
    const pad = (tf.ALIGN - hdr_end % tf.ALIGN) % tf.ALIGN;
    const body_start = hdr_end + pad;
    try t.expect(body_start <= 1025);
    const total = 1025 + 128 + tf.TRAILER_NBYTES;
    const buf = try a.alloc(u8, total);
    defer a.free(buf);
    @memset(buf, 0);
    @memcpy(buf[0..8], &tf.MAGIC);
    std.mem.writeInt(u32, buf[8..12], tf.VERSION_U32, .little);
    std.mem.writeInt(u32, buf[12..16], @intCast(header.len), .little);
    @memcpy(buf[tf.PREAMBLE_NBYTES..][0..header.len], header);
    const fnv = tf.fnv1a64(tf.FNV_OFFSET_BASIS, buf[0 .. total - tf.TRAILER_NBYTES]);
    std.mem.writeInt(u64, buf[total - tf.TRAILER_NBYTES ..][0..8], fnv, .little);
    const path = try writeFile("tfold_misaligned.tfold", buf);
    defer a.free(path);
    try t.expectError(error.TfoldTensorMisaligned, tf.openPrefixes(a, t.io, path, &.{}));
}

test "tfold_file: tensor offset past body end refused" {
    const header = try std.fmt.allocPrint(a,
        \\{{"tfold":{{"version":"0.2"}},"quant":{{"scheme":"exl3_mul1","bits_per_weight":8.0,"block_size":16}},"model":{{"source_sha256":"0000000000000000000000000000000000000000000000000000000000000000","vocab_size":1,"parameters":1}},"tensors":[{{"name":"p","dtype":"u8","shape":[128],"offset":320,"nbytes":{d},"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}]}}
    , .{10000});
    defer a.free(header);
    const total = 1024 + 128 + tf.TRAILER_NBYTES;
    const buf = try a.alloc(u8, total);
    defer a.free(buf);
    @memset(buf, 0);
    @memcpy(buf[0..8], &tf.MAGIC);
    std.mem.writeInt(u32, buf[8..12], tf.VERSION_U32, .little);
    std.mem.writeInt(u32, buf[12..16], @intCast(header.len), .little);
    @memcpy(buf[tf.PREAMBLE_NBYTES..][0..header.len], header);
    const fnv = tf.fnv1a64(tf.FNV_OFFSET_BASIS, buf[0 .. total - tf.TRAILER_NBYTES]);
    std.mem.writeInt(u64, buf[total - tf.TRAILER_NBYTES ..][0..8], fnv, .little);
    const path = try writeFile("tfold_oob.tfold", buf);
    defer a.free(path);
    try t.expectError(error.TfoldTensorOverflow, tf.openPrefixes(a, t.io, path, &.{}));
}

test "tfold_file: unsupported dtype refused" {
    const bytes = try makeValid(a, &.{.{ .name = "p", .dtype = "unknown_dtype", .nbytes = 128, .shape = &.{128} }});
    defer a.free(bytes);
    const path = try writeFile("tfold_baddtype.tfold", bytes);
    defer a.free(path);
    try t.expectError(error.TfoldUnsupportedDtype, tf.openPrefixes(a, t.io, path, &.{}));
}

test "tfold_file: prefix filter keeps only matching tensors" {
    const bytes = try makeValid(a, &.{
        .{ .name = "keep.one", .nbytes = 64, .shape = &.{64} },
        .{ .name = "drop.two", .nbytes = 64, .shape = &.{64} },
    });
    defer a.free(bytes);
    const path = try writeFile("tfold_prefix.tfold", bytes);
    defer a.free(path);
    var f = try tf.openPrefixes(a, t.io, path, &.{"keep."});
    defer f.close(t.io);
    try t.expectEqual(@as(usize, 1), f.names.count());
    try t.expect(f.get("keep.one") != null);
    try t.expect(f.get("drop.two") == null);
}

test "fnv1a64: a known vector" {
    try t.expectEqual(tf.FNV_OFFSET_BASIS, tf.fnv1a64(tf.FNV_OFFSET_BASIS, &.{}));
    try t.expectEqual(@as(u64, 0x85944171f73967e8), tf.fnv1a64(tf.FNV_OFFSET_BASIS, "foobar"));
}

// ---------------------------------------------------------------------------
// Stage 6e: FNV optimization tests
// ---------------------------------------------------------------------------

test "fnv1a64_unrolled: bit-identical to scalar reference over random 16 MB" {
    const buf = try a.alloc(u8, 16 * 1024 * 1024);
    defer a.free(buf);
    var prng = std.Random.DefaultPrng.init(0xC0DED00D);
    prng.fill(buf);
    const h_scalar = tf.fnv1a64_scalar(tf.FNV_OFFSET_BASIS, buf);
    const h_unrolled = tf.fnv1a64_unrolled(tf.FNV_OFFSET_BASIS, buf);
    try t.expectEqual(h_scalar, h_unrolled);
}

test "fnv1a64_unrolled: identical for odd lengths exercising the tail" {
    const alpha = "The quick brown fox jumps over the lazy dog";
    inline for (.{ 0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025 }) |n| {
        if (n <= alpha.len) {
            const s = alpha[0..n];
            try t.expectEqual(tf.fnv1a64_scalar(tf.FNV_OFFSET_BASIS, s), tf.fnv1a64_unrolled(tf.FNV_OFFSET_BASIS, s));
        }
    }
    // A buffer specifically sized to spill into the tail path.
    var buf: [1025]u8 = undefined;
    for (&buf, 0..) |*b, i| b.* = @truncate(i);
    try t.expectEqual(tf.fnv1a64_scalar(tf.FNV_OFFSET_BASIS, &buf), tf.fnv1a64_unrolled(tf.FNV_OFFSET_BASIS, &buf));
}

test "fnv1a64_regression: a 1 MiB buffer hashes faster than 20 MB/s floor" {
    // Build/debug can be absurdly slow; this test only enforces a loose floor
    // meant to catch catastrophic regressions (algorithm swapped, CPU running
    // on the little cluster, something hijacked in startup).
    const N: usize = 1 * 1024 * 1024;
    const buf = try a.alloc(u8, N);
    defer a.free(buf);
    @memset(buf, 0xA5);
    const t0: i64 = @intCast(@divTrunc(std.Io.Clock.awake.now(t.io).toNanoseconds(), 1_000_000));
    _ = tf.fnv1a64(tf.FNV_OFFSET_BASIS, buf);
    const dt_ms: i64 = @as(i64, @intCast(@divTrunc(std.Io.Clock.awake.now(t.io).toNanoseconds(), 1_000_000))) - t0;
    // 1 MiB / dt_ms = MB/s (approximate: MiB vs MB difference ignored).
    const floor_bytes_per_ms: i64 = 20 * 1000; // 20 MB/s
    const bytes_per_ms: i64 = if (dt_ms <= 0) 1_000_000_000 else @divTrunc(@as(i64, @intCast(N)), dt_ms);
    try t.expect(bytes_per_ms >= floor_bytes_per_ms);
}
