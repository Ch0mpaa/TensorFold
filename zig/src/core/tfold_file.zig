//! TFOLD v0.2 reader that produces a `safetensors.File` pointing straight into
//! a memory-mapped .tfold file. Zero tensor-byte copies: the resulting `File`
//! aliases the same mmap as the original safetensors path would.
//!
//! Because the returned type IS `safetensors.File`, every existing `Checkpoint`
//! call site (`.get`, `.has`, `.release`, `.adviseRandom`, direct `.names`
//! iteration, direct `.map` access) works unchanged. The TFOLD-specific logic
//! lives in this file alone: preamble/header parsing, body-base alignment,
//! FNV-1a-64 integrity, and dtype mapping.
//!
//! Format details (short):
//!   bytes  0..8    MAGIC = "TFOLD\x02\x00\x00"  (v0.2)
//!   bytes  8..12   version = 0x00020000 little-endian u32
//!   bytes 12..16   header_length  little-endian u32
//!   bytes 16..16+header_length    header JSON (UTF-8, bounded)
//!   bytes padded to ALIGN (64) boundary
//!   body: tensor data at absolute file offsets recorded in header
//!   last 8 bytes: FNV-1a-64 of every byte before them, little-endian u64
//!
//! Header shape (relevant fields):
//!   {
//!     "tfold":   {"version": "0.2", ...},
//!     "quant":   {"scheme": "exl3_mcg", "bits_per_weight": 8.0, "groups": [...]},
//!     "model":   {"source_sha256": "...", "vocab_size": ..., "parameters": ...},
//!     "tensors": [{"name": "...", "dtype": "...", "shape": [...], "offset": <abs>, "nbytes": ..., "sha256": "..."}],
//!   }
//!
//! Tensor dtype mapping (lowercase v0.2 names -> safetensors.DType):
//!   bf16            -> .bf16
//!   fp16 | f16      -> .f16
//!   fp32 | f32      -> .f32
//!   u32             -> .u32   (synthetic __codebook__ scalar)
//!   i16             -> .i16   (EXL3 trellis tiles are stored as i16 words)
//!   i32             -> .i32   (per-group codebook marker scalars)
//!   exl3_mcg_tile   -> .i16
//!   exl3_mul1_tile  -> .i16
//!   exl3_3inst_tile -> .i16
//!
//! Integrity: `openPrefixes` verifies the FNV-1a-64 trailer over the body
//! (and the preamble + header JSON) before returning. Opt-out via
//! `ParseOptions.skip_fnv` exists only to let tests feed deliberately
//! malformed inputs through the structural checks; production loads never
//! disable it.
//!
//! Per-tensor SHA-256 hashes stored in the header are NOT re-verified on every
//! load. The file-wide FNV trailer gives O(file-size) full-coverage integrity;
//! `verifyTensorHashes` is provided for callers that want per-tensor hashes.

const std = @import("std");
const Io = std.Io;
const st = @import("safetensors.zig");

pub const MAGIC: [8]u8 = .{ 'T', 'F', 'O', 'L', 'D', 0x02, 0x00, 0x00 };
pub const VERSION_U32: u32 = 0x00020000;
pub const VERSION_MAJOR_MASK: u32 = 0xFFFF0000;
pub const PREAMBLE_NBYTES: usize = 16;
pub const TRAILER_NBYTES: usize = 8;
pub const ALIGN: usize = 64;

/// 1 GiB upper bound on JSON header size; stops a malformed preamble from
/// forcing us to parse the entire body as JSON.
pub const MAX_HEADER_BYTES: usize = 1 << 30;

pub const ParseOptions = struct {
    /// Index only tensors whose full name starts with one of these prefixes;
    /// empty keeps every tensor.
    prefixes: []const []const u8 = &.{},
    /// Default: verify the FNV-1a-64 trailer on open. Tests may disable this
    /// to feed deliberately-malformed inputs through the structural checks.
    skip_fnv: bool = false,
    /// After FNV verification succeeds, call madvise(DONTNEED) on the whole
    /// mapping. The page cache is preserved (the kernel keeps file pages
    /// separately from the process RSS), so tensor reads re-fault cheaply
    /// while the process RSS drops to near zero. Default: on, matching
    /// safetensors.File.release semantics.
    drop_pages_after_verify: bool = true,
    /// Return per-phase wall-clock stats in *stats_out when non-null.
    stats_out: ?*OpenStats = null,
    /// Print phase timings to stderr on every open; cheap diagnostic.
    log_stats: bool = false,
};

/// Per-phase wall-clock measurements captured during openOptions. Times are
/// monotonic nanoseconds; fnv_bytes gives the number of bytes the FNV scan
/// actually covered.
pub const OpenStats = struct {
    mmap_ns: u64 = 0,
    preamble_ns: u64 = 0,
    fnv_ns: u64 = 0,
    header_parse_ns: u64 = 0,
    drop_pages_ns: u64 = 0,
    fnv_bytes: usize = 0,
};

pub const Error = error{
    BadTfold,
    TfoldTruncated,
    TfoldBadMagic,
    TfoldBadVersion,
    TfoldHeaderOverflow,
    TfoldHeaderParse,
    TfoldBadAlignment,
    TfoldUnsupportedDtype,
    TfoldTensorOverflow,
    TfoldTensorMisaligned,
    TfoldTensorOverlap,
    TfoldTensorRankTooLarge,
    TfoldIntegrity,
    TfoldDuplicateName,
};

/// FNV-1a-64 over `bytes` starting from `state`. Reference implementation:
/// a byte-at-a-time scalar loop with the standard offset basis and prime.
/// Kept exported so cross-check tests can verify alternative implementations
/// produce the same hash.
pub fn fnv1a64_scalar(state: u64, bytes: []const u8) u64 {
    const PRIME: u64 = 0x100000001B3;
    var h: u64 = state;
    for (bytes) |b| {
        h ^= @as(u64, b);
        h = h *% PRIME;
    }
    return h;
}

/// Loop-unrolled FNV-1a-64. The hash is strictly dependent (each byte XOR
/// then multiply), so unrolling cannot shorten the data dependency chain;
/// what it DOES do is cut loop-overhead cycles (compare + branch) by a
/// factor of 16. In ReleaseFast that typically buys 5-15% throughput vs
/// the scalar loop on aarch64 and x86_64. Output is bit-identical to
/// fnv1a64_scalar for every input; a cross-check test enforces this.
pub fn fnv1a64_unrolled(state: u64, bytes: []const u8) u64 {
    const PRIME: u64 = 0x100000001B3;
    var h: u64 = state;
    var i: usize = 0;
    // Process 16 bytes per iteration. Each line is one FNV step; the
    // compiler cannot reorder these because of the data dependency.
    while (i + 16 <= bytes.len) : (i += 16) {
        h = (h ^ @as(u64, bytes[i + 0])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 1])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 2])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 3])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 4])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 5])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 6])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 7])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 8])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 9])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 10])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 11])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 12])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 13])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 14])) *% PRIME;
        h = (h ^ @as(u64, bytes[i + 15])) *% PRIME;
    }
    // Tail.
    while (i < bytes.len) : (i += 1) {
        h = (h ^ @as(u64, bytes[i])) *% PRIME;
    }
    return h;
}

/// The default FNV entry point. Routes to the unrolled implementation in
/// ReleaseFast/ReleaseSafe builds; stays with the scalar reference in
/// Debug so stack traces on test failures are easier to read. Both produce
/// the same hash; a cross-check test enforces it.
pub fn fnv1a64(state: u64, bytes: []const u8) u64 {
    const mode = @import("builtin").mode;
    return switch (mode) {
        .Debug, .ReleaseSmall => fnv1a64_scalar(state, bytes),
        .ReleaseFast, .ReleaseSafe => fnv1a64_unrolled(state, bytes),
    };
}
pub const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;

fn parseDtype(text: []const u8) ?st.DType {
    // Upstream uppercase names (BF16/F16/F32/I16/I32/U32/U8/...) via safetensors.DType.parse.
    if (text.len > 0 and text[0] >= 'A' and text[0] <= 'Z') {
        if (st.DType.parse(text)) |dt| return dt;
    }
    // v0.2 lowercase dtype names.
    const names = .{
        .{ "bf16", st.DType.bf16 },
        .{ "f16",  st.DType.f16 },  .{ "fp16", st.DType.f16 },
        .{ "f32",  st.DType.f32 },  .{ "fp32", st.DType.f32 },
        .{ "u8",   st.DType.u8 },   .{ "i8",   st.DType.i8 },
        .{ "u16",  st.DType.u16 },  .{ "i16",  st.DType.i16 },
        .{ "u32",  st.DType.u32 },  .{ "i32",  st.DType.i32 },
        .{ "u64",  st.DType.u64 },  .{ "i64",  st.DType.i64 },
        .{ "f64",  st.DType.f64 },
        // EXL3 trellis tiles are stored as i16 arrays; kernels read them as i16.
        .{ "exl3_mcg_tile",   st.DType.i16 },
        .{ "exl3_mul1_tile",  st.DType.i16 },
        .{ "exl3_3inst_tile", st.DType.i16 },
    };
    inline for (names) |n| if (std.mem.eql(u8, text, n[0])) return n[1];
    return null;
}

/// Open a .tfold v0.2 file and present it as a `safetensors.File`.
///
/// The returned `File` owns its mmap, arena, and path dup; close it exactly
/// like a safetensors.File (via `.close(io)`).
pub fn open(gpa: std.mem.Allocator, io: Io, path: []const u8) !st.File {
    return openOptions(gpa, io, path, .{});
}
pub fn openPrefix(gpa: std.mem.Allocator, io: Io, path: []const u8, prefix: ?[]const u8) !st.File {
    return openOptions(gpa, io, path, .{ .prefixes = if (prefix) |p| &.{p} else &.{} });
}
pub fn openPrefixes(gpa: std.mem.Allocator, io: Io, path: []const u8, prefixes: []const []const u8) !st.File {
    return openOptions(gpa, io, path, .{ .prefixes = prefixes });
}

pub fn openOptions(gpa: std.mem.Allocator, io: Io, path: []const u8, options: ParseOptions) !st.File {
    var stats: OpenStats = .{};
    const want_stats = options.stats_out != null;
    var tm0: i128 = @as(i128, std.Io.Clock.awake.now(io).toNanoseconds());

    var file = try Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);
    const len = std.math.cast(usize, try file.length(io)) orelse return error.TfoldTruncated;
    if (len < PREAMBLE_NBYTES + TRAILER_NBYTES + 2) return error.TfoldTruncated;

    var map = try Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
    errdefer map.destroy(io);
    {
        const t = @as(i128, std.Io.Clock.awake.now(io).toNanoseconds());
        stats.mmap_ns = @intCast(@as(i128, t) - tm0);
        tm0 = t;
    }

    // Preamble
    if (!std.mem.eql(u8, map.memory[0..8], &MAGIC)) return error.TfoldBadMagic;
    const version = std.mem.readInt(u32, map.memory[8..12], .little);
    if ((version & VERSION_MAJOR_MASK) != (VERSION_U32 & VERSION_MAJOR_MASK)) return error.TfoldBadVersion;
    const hlen = std.mem.readInt(u32, map.memory[12..16], .little);
    if (hlen == 0 or hlen > MAX_HEADER_BYTES) return error.TfoldHeaderOverflow;
    if (@as(usize, PREAMBLE_NBYTES) + @as(usize, hlen) + TRAILER_NBYTES > len) return error.TfoldHeaderOverflow;

    const hdr_end = PREAMBLE_NBYTES + @as(usize, hlen);
    const pad = (ALIGN - hdr_end % ALIGN) % ALIGN;
    const body_start = hdr_end + pad;
    const body_end = len - TRAILER_NBYTES;
    if (body_start > body_end) return error.TfoldBadAlignment;
    {
        const t = @as(i128, std.Io.Clock.awake.now(io).toNanoseconds());
        stats.preamble_ns = @intCast(@as(i128, t) - tm0);
        tm0 = t;
    }

    // FNV trailer (default on; opt-out is for malformed-input tests only).
    if (!options.skip_fnv) {
        const want = std.mem.readInt(u64, map.memory[body_end..][0..TRAILER_NBYTES], .little);
        const got = fnv1a64(FNV_OFFSET_BASIS, map.memory[0..body_end]);
        if (got != want) return error.TfoldIntegrity;
        stats.fnv_bytes = body_end;
    }
    {
        const t = @as(i128, std.Io.Clock.awake.now(io).toNanoseconds());
        stats.fnv_ns = @intCast(@as(i128, t) - tm0);
        tm0 = t;
    }

    // Parse the header into a safetensors.Header whose Entry.begin/end are
    // absolute file offsets. We set the returned File.data = 0 so that
    // safetensors.File.get() returns map.memory[e.begin..e.end] verbatim.
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const header_bytes = map.memory[PREAMBLE_NBYTES..hdr_end];
    const names = try parseHeader(arena.allocator(), header_bytes, body_start, body_end, options.prefixes);

    const own = try arena.allocator().dupeSentinel(u8, path, 0);
    {
        const t = @as(i128, std.Io.Clock.awake.now(io).toNanoseconds());
        stats.header_parse_ns = @intCast(@as(i128, t) - tm0);
        tm0 = t;
    }

    // Post-verify page reclamation: the FNV scan touched every page, so the
    // whole mapping is resident. The Qwen loader walks tensors in sequence
    // and cudaMemcpy'''s each to the GPU; it does not read any byte twice.
    // madvise(DONTNEED) tells the kernel it can drop the per-process PTEs
    // for this mapping. The page cache is preserved (madvise is not
    // munmap), so subsequent reads re-fault cheaply from the cache rather
    // than from disk. Net effect: the process RSS drops by ~3 GB on the
    // Qwen3.5-2B .tfold while tensor access stays correct.
    if (options.drop_pages_after_verify and !options.skip_fnv and @import("builtin").os.tag == .linux) {
        _ = std.os.linux.madvise(map.memory.ptr, map.memory.len, std.os.linux.MADV.DONTNEED);
    }
    {
        const t = @as(i128, std.Io.Clock.awake.now(io).toNanoseconds());
        stats.drop_pages_ns = @intCast(@as(i128, t) - tm0);
    }
    if (want_stats) options.stats_out.?.* = stats;

    // Opt-in diagnostic log: set log_stats = true in ParseOptions to print
    // the phase breakdown to stderr on every open.
    if (options.log_stats) {
        std.debug.print("tfold_file open: mmap={d}us preamble={d}us fnv={d}us header_parse={d}us drop_pages={d}us fnv_bytes={d}\n", .{
            @divTrunc(stats.mmap_ns, 1000),
            @divTrunc(stats.preamble_ns, 1000),
            @divTrunc(stats.fnv_ns, 1000),
            @divTrunc(stats.header_parse_ns, 1000),
            @divTrunc(stats.drop_pages_ns, 1000),
            stats.fnv_bytes,
        });
    }

    return .{
        .path = own,
        .file = file,
        .map = map,
        .data = 0, // entries store absolute file offsets
        .names = names,
        .arena = arena,
    };
}

/// Re-verify per-tensor SHA-256 hashes against a caller-provided map.
/// Caller builds the hash map by re-parsing the JSON header (we only keep the
/// entry begin/end + dtype on open).
pub fn verifyTensorHashes(file: *const st.File, hashes: *const std.StringHashMapUnmanaged([32]u8)) !void {
    var it = file.names.iterator();
    while (it.next()) |kv| {
        const want = hashes.get(kv.key_ptr.*) orelse continue;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(file.map.memory[kv.value_ptr.begin..kv.value_ptr.end]);
        var got: [32]u8 = undefined;
        h.final(&got);
        if (!std.mem.eql(u8, &got, &want)) return error.TfoldIntegrity;
    }
}

/// Parse the v0.2 JSON header into `st.Header` entries with absolute
/// file-offset begin/end. Rejects out-of-range offsets, misalignment,
/// overlap, and duplicates.
fn parseHeader(
    arena: std.mem.Allocator,
    header_bytes: []const u8,
    body_start: usize,
    body_end: usize,
    prefixes: []const []const u8,
) !st.Header {
    var parsed = std.json.parseFromSlice(std.json.Value, arena, header_bytes, .{}) catch return error.TfoldHeaderParse;
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.TfoldHeaderParse;

    const tensors_v = root.object.get("tensors") orelse return error.TfoldHeaderParse;
    if (tensors_v != .array) return error.TfoldHeaderParse;

    var header: st.Header = .empty;
    try header.ensureTotalCapacity(arena, tensors_v.array.items.len);

    for (tensors_v.array.items) |t_v| {
        if (t_v != .object) return error.TfoldHeaderParse;
        const name_v = t_v.object.get("name") orelse return error.TfoldHeaderParse;
        const dtype_v = t_v.object.get("dtype") orelse return error.TfoldHeaderParse;
        const shape_v = t_v.object.get("shape") orelse return error.TfoldHeaderParse;
        const offset_v = t_v.object.get("offset") orelse return error.TfoldHeaderParse;
        const nbytes_v = t_v.object.get("nbytes") orelse return error.TfoldHeaderParse;
        if (name_v != .string or dtype_v != .string or shape_v != .array or offset_v != .integer or nbytes_v != .integer) return error.TfoldHeaderParse;

        const name = name_v.string;
        if (prefixes.len > 0) {
            var matched = false;
            for (prefixes) |p| if (std.mem.startsWith(u8, name, p)) {
                matched = true;
                break;
            };
            if (!matched) continue;
        }

        const dtype = parseDtype(dtype_v.string) orelse return error.TfoldUnsupportedDtype;
        if (shape_v.array.items.len > st.max_rank) return error.TfoldTensorRankTooLarge;
        var shape: [st.max_rank]usize = @splat(0);
        for (shape_v.array.items, 0..) |dv, i| {
            if (dv != .integer or dv.integer < 0) return error.TfoldHeaderParse;
            shape[i] = @intCast(dv.integer);
        }

        const offset: i64 = offset_v.integer;
        const nbytes: i64 = nbytes_v.integer;
        if (offset < 0 or nbytes < 0) return error.TfoldHeaderParse;
        const off: usize = @intCast(offset);
        const nb: usize = @intCast(nbytes);

        if (off < body_start or off + nb > body_end) return error.TfoldTensorOverflow;
        if (off % ALIGN != 0) return error.TfoldTensorMisaligned;

        const name_owned = try arena.dupe(u8, name);
        const gop = try header.getOrPut(arena, name_owned);
        if (gop.found_existing) return error.TfoldDuplicateName;
        gop.value_ptr.* = .{
            .dtype = dtype,
            .rank = @intCast(shape_v.array.items.len),
            .shape = shape,
            .begin = off,
            .end = off + nb,
        };
    }

    try checkNoOverlap(arena, &header);
    return header;
}

fn checkNoOverlap(arena: std.mem.Allocator, header: *st.Header) !void {
    const Span = struct { begin: usize, end: usize };
    const n = header.count();
    if (n <= 1) return;
    const spans = try arena.alloc(Span, n);
    defer arena.free(spans);
    var i: usize = 0;
    var it = header.iterator();
    while (it.next()) |kv| : (i += 1) spans[i] = .{ .begin = kv.value_ptr.begin, .end = kv.value_ptr.end };
    std.mem.sort(Span, spans, {}, struct {
        fn lt(_: void, a: Span, b: Span) bool {
            return a.begin < b.begin;
        }
    }.lt);
    var prev_end: usize = 0;
    for (spans) |s| {
        if (s.begin < prev_end) return error.TfoldTensorOverlap;
        prev_end = s.end;
    }
}
