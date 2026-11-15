//! Native .tfold v0.1 reader.
//!
//! Implements the spec at docs/tfold-format-v0.md. Mirrors the Python
//! reference (tools/tfold_py/) byte-for-byte: same magic, same version,
//! same alignment, same FNV-1a-64 trailer, same SHA-256 per tensor, same
//! §4 fail-closed reject taxonomy.
//!
//! Dependencies: std only. No numpy, no BLAS, no network, no fork/exec.
//! Nothing in this module executes file-supplied data.

const std = @import("std");

pub const MAGIC: [8]u8 = .{ 'T', 'F', 'O', 'L', 'D', 0, 0, 0 };
pub const SPEC_VERSION_U32: u32 = 0x0001_0000;
pub const MAJOR_MASK: u32 = 0xFFFF_0000;
pub const ALIGN: usize = 64;
pub const TRAILER_NBYTES: usize = 8;
pub const PREAMBLE_NBYTES: usize = 16;

/// Spec §4 reject taxonomy. Keep in lockstep with
/// tools/tfold_py/_errors.py.
pub const SpecError = error{
    TfoldTooSmall,
    TfoldBadMagic,
    TfoldVersionUnsupported,
    TfoldHeaderOverflow,
    TfoldHeaderParse,
    TfoldHeaderSchema,
    TfoldIntegrity,
    TfoldTensorOob,
    TfoldTensorAlign,
    TfoldTensorOverlap,
    TfoldTensorDupName,
    TfoldUnknownArch,
    TfoldUnknownDtype,
    TfoldUnknownScheme,
    TfoldUnknownTokenizerType,
    TfoldUnknownReservedName,
    TfoldCheckpointMismatch,
    TfoldTensorIntegrity,
    TfoldTensorShapeNbytes,
};

pub const Dtype = enum {
    fp16,
    bf16,
    fp32,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    exl3_tile,
    exl3_mcg_tile,

    pub fn parse(s: []const u8) ?Dtype {
        const pairs = .{
            .{ "fp16", Dtype.fp16 },      .{ "bf16", Dtype.bf16 },
            .{ "fp32", Dtype.fp32 },      .{ "u8", Dtype.u8 },
            .{ "i8", Dtype.i8 },          .{ "u16", Dtype.u16 },
            .{ "i16", Dtype.i16 },        .{ "u32", Dtype.u32 },
            .{ "i32", Dtype.i32 },        .{ "exl3_tile", Dtype.exl3_tile },
            .{ "exl3_mcg_tile", Dtype.exl3_mcg_tile },
        };
        inline for (pairs) |p| {
            if (std.mem.eql(u8, s, p[0])) return p[1];
        }
        return null;
    }

    pub fn elementSize(self: Dtype) ?usize {
        return switch (self) {
            .fp16, .bf16, .u16, .i16 => 2,
            .fp32, .u32, .i32 => 4,
            .u8, .i8 => 1,
            .exl3_tile, .exl3_mcg_tile => null,
        };
    }
};

pub fn archIsKnown(s: []const u8) bool {
    const archs = .{ "qwen3", "qwen3_moe", "llama", "glm", "gemma3" };
    inline for (archs) |a| if (std.mem.eql(u8, s, a)) return true;
    return false;
}

pub fn schemeIsKnown(s: []const u8) bool {
    const sch = .{ "fp16", "bf16", "fp32", "exl3", "exl3_mcg", "gguf_q4_0", "gguf_q4_k_m", "gguf_q8_0" };
    inline for (sch) |x| if (std.mem.eql(u8, s, x)) return true;
    return false;
}

pub fn tokenizerTypeIsKnown(s: []const u8) bool {
    const types = .{ "tiktoken", "sentencepiece", "bpe", "unigram" };
    inline for (types) |t| if (std.mem.eql(u8, s, t)) return true;
    return false;
}

pub const Tensor = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const u64,
    offset: u64,
    nbytes: u64,
    sha256: [64]u8,
};

/// Parsed header; strings are slices into `arena`.
pub const Header = struct {
    arena: std.heap.ArenaAllocator,
    version_str: []const u8,
    model_name: []const u8,
    arch: []const u8,
    source_sha256: []const u8,
    tokenizer_type: []const u8,
    quant_scheme: []const u8,
    tensors: []Tensor,

    pub fn deinit(self: *Header) void {
        self.arena.deinit();
    }
};

pub const File = struct {
    gpa: std.mem.Allocator,
    bytes: []u8,
    header: Header,

    pub fn close(self: *File) void {
        self.header.deinit();
        self.gpa.free(self.bytes);
    }

    pub fn tensorByName(self: *const File, name: []const u8) ?*const Tensor {
        for (self.header.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    pub fn tensorSlice(self: *const File, t: *const Tensor) []const u8 {
        return self.bytes[t.offset .. t.offset + t.nbytes];
    }
};

pub fn fnv1a_64(data: []const u8) u64 {
    var h: u64 = 0xCBF2_9CE4_8422_2325;
    for (data) |b| {
        h ^= b;
        h, _ = @mulWithOverflow(h, 0x0000_0100_0000_01B3);
    }
    return h;
}

pub const OpenOptions = struct {
    pin_source_sha256: ?[]const u8 = null,
    verify_tensor_hashes: bool = true,
};

pub fn openPath(gpa: std.mem.Allocator, io: std.Io, path: []const u8, opts: OpenOptions) !File {
    const buf = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31));
    return openBytes(gpa, buf, opts);
}

/// On entry, ownership of `bytes` transfers in. On any error, `bytes` is
/// freed by this function. On success, ownership moves into the returned
/// File (freed on close).
pub fn openBytes(gpa: std.mem.Allocator, bytes: []u8, opts: OpenOptions) !File {
    // Single guard: if we return WITHOUT success, free everything.
    var owned_bytes: ?[]u8 = bytes;
    var owned_arena: ?std.heap.ArenaAllocator = null;
    defer {
        if (owned_bytes) |b| gpa.free(b);
        if (owned_arena) |*a| a.deinit();
    }

    // --- §4.1 TOO_SMALL
    const min_size: usize = PREAMBLE_NBYTES + 2 + TRAILER_NBYTES;
    if (bytes.len < min_size) return SpecError.TfoldTooSmall;

    // --- §4.2 BAD_MAGIC
    if (!std.mem.eql(u8, bytes[0..8], &MAGIC)) return SpecError.TfoldBadMagic;

    // --- §4.3 VERSION_UNSUPPORTED
    const version = std.mem.readInt(u32, bytes[8..12], .little);
    if ((version & MAJOR_MASK) != (SPEC_VERSION_U32 & MAJOR_MASK)) return SpecError.TfoldVersionUnsupported;

    // --- §4.4 HEADER_OVERFLOW (integer overflow guard)
    const header_length: u64 = @intCast(std.mem.readInt(u32, bytes[12..16], .little));
    const after_header = std.math.add(u64, @intCast(PREAMBLE_NBYTES), header_length) catch return SpecError.TfoldHeaderOverflow;
    const min_total = std.math.add(u64, after_header, @intCast(TRAILER_NBYTES)) catch return SpecError.TfoldHeaderOverflow;
    if (min_total > @as(u64, @intCast(bytes.len))) return SpecError.TfoldHeaderOverflow;

    // --- §4.5 HEADER_PARSE
    const header_bytes = bytes[PREAMBLE_NBYTES .. PREAMBLE_NBYTES + @as(usize, @intCast(header_length))];
    owned_arena = std.heap.ArenaAllocator.init(gpa);
    const arena_ref = &owned_arena.?;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena_ref.allocator(), header_bytes, .{}) catch {
        return SpecError.TfoldHeaderParse;
    };
    if (parsed != .object) return SpecError.TfoldHeaderParse;

    // --- §4.6 HEADER_SCHEMA
    var hdr = try parseAndValidate(arena_ref, parsed, bytes.len);

    // --- §4.7 INTEGRITY
    const body = bytes[0 .. bytes.len - TRAILER_NBYTES];
    const got = fnv1a_64(body);
    const want = std.mem.readInt(u64, bytes[bytes.len - TRAILER_NBYTES ..][0..8], .little);
    if (got != want) return SpecError.TfoldIntegrity;

    // --- §4.8..11 bounds + alignment + overlap + duplicates
    const file_end: u64 = @intCast(bytes.len - TRAILER_NBYTES);
    try checkTensorLayout(hdr.tensors, file_end);

    // --- §4.12 explicit registries
    if (!archIsKnown(hdr.arch)) return SpecError.TfoldUnknownArch;
    if (!schemeIsKnown(hdr.quant_scheme)) return SpecError.TfoldUnknownScheme;
    if (!tokenizerTypeIsKnown(hdr.tokenizer_type)) return SpecError.TfoldUnknownTokenizerType;

    // --- §4.13 CHECKPOINT_MISMATCH
    if (opts.pin_source_sha256) |pin| {
        if (!std.mem.eql(u8, pin, hdr.source_sha256)) return SpecError.TfoldCheckpointMismatch;
    }

    // --- §4.14 TENSOR_INTEGRITY
    if (opts.verify_tensor_hashes) {
        for (hdr.tensors) |t| {
            const blob = bytes[t.offset .. t.offset + t.nbytes];
            var sha = std.crypto.hash.sha2.Sha256.init(.{});
            sha.update(blob);
            var digest: [32]u8 = undefined;
            sha.final(&digest);
            var hex: [64]u8 = undefined;
            const hex_chars = "0123456789abcdef";
            for (digest, 0..) |byte, idx| {
                hex[idx * 2] = hex_chars[byte >> 4];
                hex[idx * 2 + 1] = hex_chars[byte & 0x0F];
            }
            if (!std.mem.eql(u8, &hex, &t.sha256)) return SpecError.TfoldTensorIntegrity;
        }
    }

    // SUCCESS: move ownership out of the guard.
    hdr.arena = owned_arena.?;
    owned_arena = null;
    const out_bytes = owned_bytes.?;
    owned_bytes = null;

    return File{
        .gpa = gpa,
        .bytes = out_bytes,
        .header = hdr,
    };
}

fn parseAndValidate(arena_ptr: *std.heap.ArenaAllocator, root: std.json.Value, file_size: usize) SpecError!Header {
    const arena = arena_ptr.allocator();
    _ = file_size;

    const o = root.object;

    const tfold_v = o.get("tfold") orelse return SpecError.TfoldHeaderSchema;
    if (tfold_v != .object) return SpecError.TfoldHeaderSchema;
    const version_v = tfold_v.object.get("version") orelse return SpecError.TfoldHeaderSchema;
    if (version_v != .string) return SpecError.TfoldHeaderSchema;
    if (!std.mem.eql(u8, version_v.string, "0.1")) return SpecError.TfoldHeaderSchema;

    const model_v = o.get("model") orelse return SpecError.TfoldHeaderSchema;
    if (model_v != .object) return SpecError.TfoldHeaderSchema;
    const model_name_v = model_v.object.get("name") orelse return SpecError.TfoldHeaderSchema;
    const arch_v = model_v.object.get("arch") orelse return SpecError.TfoldHeaderSchema;
    const source_sha_v = model_v.object.get("source_sha256") orelse return SpecError.TfoldHeaderSchema;
    if (model_name_v != .string or arch_v != .string or source_sha_v != .string) return SpecError.TfoldHeaderSchema;
    if (source_sha_v.string.len != 64) return SpecError.TfoldHeaderSchema;
    for (source_sha_v.string) |c| {
        if (!((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'))) return SpecError.TfoldHeaderSchema;
    }

    const tok_v = o.get("tokenizer") orelse return SpecError.TfoldHeaderSchema;
    if (tok_v != .object) return SpecError.TfoldHeaderSchema;
    const tok_type_v = tok_v.object.get("type") orelse return SpecError.TfoldHeaderSchema;
    if (tok_type_v != .string) return SpecError.TfoldHeaderSchema;

    const quant_v = o.get("quant") orelse return SpecError.TfoldHeaderSchema;
    if (quant_v != .object) return SpecError.TfoldHeaderSchema;
    const scheme_v = quant_v.object.get("scheme") orelse return SpecError.TfoldHeaderSchema;
    if (scheme_v != .string) return SpecError.TfoldHeaderSchema;

    const tensors_v = o.get("tensors") orelse return SpecError.TfoldHeaderSchema;
    if (tensors_v != .array) return SpecError.TfoldHeaderSchema;
    if (tensors_v.array.items.len == 0 or tensors_v.array.items.len > 65535) return SpecError.TfoldHeaderSchema;

    const list = arena.alloc(Tensor, tensors_v.array.items.len) catch return SpecError.TfoldHeaderSchema;
    for (tensors_v.array.items, 0..) |tv, i| {
        if (tv != .object) return SpecError.TfoldHeaderSchema;
        const to = tv.object;
        const name_v = to.get("name") orelse return SpecError.TfoldHeaderSchema;
        if (name_v != .string) return SpecError.TfoldHeaderSchema;
        if (name_v.string.len == 0 or name_v.string.len > 256) return SpecError.TfoldHeaderSchema;

        if (std.mem.startsWith(u8, name_v.string, "__") and !std.mem.eql(u8, name_v.string, "__codebook__")) {
            return SpecError.TfoldUnknownReservedName;
        }

        const dtype_v = to.get("dtype") orelse return SpecError.TfoldHeaderSchema;
        if (dtype_v != .string) return SpecError.TfoldHeaderSchema;
        const dtype = Dtype.parse(dtype_v.string) orelse return SpecError.TfoldUnknownDtype;

        const shape_v = to.get("shape") orelse return SpecError.TfoldHeaderSchema;
        if (shape_v != .array) return SpecError.TfoldHeaderSchema;
        if (shape_v.array.items.len == 0 or shape_v.array.items.len > 8) return SpecError.TfoldHeaderSchema;
        const shape = arena.alloc(u64, shape_v.array.items.len) catch return SpecError.TfoldHeaderSchema;
        for (shape_v.array.items, 0..) |dv, k| {
            if (dv != .integer or dv.integer <= 0) return SpecError.TfoldHeaderSchema;
            shape[k] = @intCast(dv.integer);
        }

        const off_v = to.get("offset") orelse return SpecError.TfoldHeaderSchema;
        if (off_v != .integer or off_v.integer < 0) return SpecError.TfoldHeaderSchema;
        const nbytes_v = to.get("nbytes") orelse return SpecError.TfoldHeaderSchema;
        if (nbytes_v != .integer or nbytes_v.integer <= 0) return SpecError.TfoldHeaderSchema;

        const sha_v = to.get("sha256") orelse return SpecError.TfoldHeaderSchema;
        if (sha_v != .string) return SpecError.TfoldHeaderSchema;
        if (sha_v.string.len != 64) return SpecError.TfoldHeaderSchema;

        if (dtype.elementSize()) |esize| {
            var product: u64 = esize;
            for (shape) |d| {
                const ov = @mulWithOverflow(product, d);
                if (ov[1] != 0) return SpecError.TfoldHeaderSchema;
                product = ov[0];
            }
            if (product != @as(u64, @intCast(nbytes_v.integer))) return SpecError.TfoldTensorShapeNbytes;
        }

        var sha_hex: [64]u8 = undefined;
        @memcpy(&sha_hex, sha_v.string);

        list[i] = .{
            .name = name_v.string,
            .dtype = dtype,
            .shape = shape,
            .offset = @intCast(off_v.integer),
            .nbytes = @intCast(nbytes_v.integer),
            .sha256 = sha_hex,
        };
    }

    return Header{
        .arena = undefined,
        .version_str = version_v.string,
        .model_name = model_name_v.string,
        .arch = arch_v.string,
        .source_sha256 = source_sha_v.string,
        .tokenizer_type = tok_type_v.string,
        .quant_scheme = scheme_v.string,
        .tensors = list,
    };
}

fn lessByOffset(_: void, a: Tensor, b: Tensor) bool {
    return a.offset < b.offset;
}

fn checkTensorLayout(tensors: []Tensor, file_end: u64) SpecError!void {
    var i: usize = 0;
    while (i < tensors.len) : (i += 1) {
        var j: usize = i + 1;
        while (j < tensors.len) : (j += 1) {
            if (std.mem.eql(u8, tensors[i].name, tensors[j].name)) return SpecError.TfoldTensorDupName;
        }
    }

    var tmp_buf: [256]Tensor = undefined;
    if (tensors.len > tmp_buf.len) return SpecError.TfoldHeaderSchema;
    const copy = tmp_buf[0..tensors.len];
    @memcpy(copy, tensors);
    std.mem.sort(Tensor, copy, {}, lessByOffset);

    var prev_end: u64 = 0;
    for (copy) |t| {
        if (t.offset % ALIGN != 0) return SpecError.TfoldTensorAlign;
        const end = std.math.add(u64, t.offset, t.nbytes) catch return SpecError.TfoldTensorOob;
        if (end > file_end) return SpecError.TfoldTensorOob;
        if (t.offset < prev_end) return SpecError.TfoldTensorOverlap;
        prev_end = end;
    }
}
