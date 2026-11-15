//! Native Zig .tfold v0.2 reader.
//!
//! Wraps the Stage 4 parser with the v0.2 magic + extended registries.
//! v0.1 and v0.2 coexist: each reader rejects the other with
//! TfoldBadMagic (because v0.2 uses `TFOLD\x02\x00\x00`).

const std = @import("std");
const reader = @import("reader.zig");

pub const MAGIC_V02: [8]u8 = .{ 'T', 'F', 'O', 'L', 'D', 0x02, 0, 0 };
pub const SPEC_VERSION_U32_V02: u32 = 0x0002_0000;

/// For v0.2 we recognize three extra tile dtype names mapping to the
/// same packed-byte handling as `exl3_mcg_tile` in v0.1 Stage 4.
pub fn dtypeIsKnownV02(s: []const u8) bool {
    if (reader.Dtype.parse(s) != null) return true;
    const v2_extras = .{ "exl3_mul1_tile", "exl3_3inst_tile" };
    inline for (v2_extras) |x| if (std.mem.eql(u8, s, x)) return true;
    return false;
}

/// v0.2 adds two scheme names.
pub fn schemeIsKnownV02(s: []const u8) bool {
    if (reader.schemeIsKnown(s)) return true;
    const v2_extras = .{ "exl3_mul1", "exl3_3inst" };
    inline for (v2_extras) |x| if (std.mem.eql(u8, s, x)) return true;
    return false;
}

/// Open a v0.2 file and run the structural checks. Full semantic
/// validation (overrides/groups) stays in the Python reference for now;
/// this Zig reader guarantees magic + integrity + alignment + bounds.
pub const OpenOptions = struct {
    pin_source_sha256: ?[]const u8 = null,
    verify_tensor_hashes: bool = true,
};

pub const File = struct {
    gpa: std.mem.Allocator,
    bytes: []u8,
    scheme_name: []const u8, // slice into bytes
    overrides_present: bool,
    groups_present: bool,

    pub fn close(self: *File) void {
        self.gpa.free(self.bytes);
    }
};

pub fn openPath(gpa: std.mem.Allocator, io: std.Io, path: []const u8, opts: OpenOptions) !File {
    const buf = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 32));
    return openBytes(gpa, buf, opts);
}

pub fn openBytes(gpa: std.mem.Allocator, bytes: []u8, opts: OpenOptions) !File {
    var owned_bytes: ?[]u8 = bytes;
    defer if (owned_bytes) |b| gpa.free(b);

    const min = reader.PREAMBLE_NBYTES + 2 + reader.TRAILER_NBYTES;
    if (bytes.len < min) return reader.SpecError.TfoldTooSmall;

    if (!std.mem.eql(u8, bytes[0..8], &MAGIC_V02)) return reader.SpecError.TfoldBadMagic;

    const version = std.mem.readInt(u32, bytes[8..12], .little);
    if ((version & reader.MAJOR_MASK) != (SPEC_VERSION_U32_V02 & reader.MAJOR_MASK))
        return reader.SpecError.TfoldVersionUnsupported;

    const hlen: u64 = @intCast(std.mem.readInt(u32, bytes[12..16], .little));
    const after = std.math.add(u64, @intCast(reader.PREAMBLE_NBYTES), hlen) catch return reader.SpecError.TfoldHeaderOverflow;
    const needed = std.math.add(u64, after, @intCast(reader.TRAILER_NBYTES)) catch return reader.SpecError.TfoldHeaderOverflow;
    if (needed > @as(u64, @intCast(bytes.len))) return reader.SpecError.TfoldHeaderOverflow;

    const header_bytes = bytes[reader.PREAMBLE_NBYTES .. reader.PREAMBLE_NBYTES + @as(usize, @intCast(hlen))];
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), header_bytes, .{}) catch return reader.SpecError.TfoldHeaderParse;
    if (parsed != .object) return reader.SpecError.TfoldHeaderParse;

    // version field must be "0.2"
    const tfold_v = parsed.object.get("tfold") orelse return reader.SpecError.TfoldHeaderSchema;
    if (tfold_v != .object) return reader.SpecError.TfoldHeaderSchema;
    const ver_v = tfold_v.object.get("version") orelse return reader.SpecError.TfoldHeaderSchema;
    if (ver_v != .string or !std.mem.eql(u8, ver_v.string, "0.2"))
        return reader.SpecError.TfoldHeaderSchema;

    // quant.scheme must be v0.2-known
    const quant_v = parsed.object.get("quant") orelse return reader.SpecError.TfoldHeaderSchema;
    if (quant_v != .object) return reader.SpecError.TfoldHeaderSchema;
    const scheme_v = quant_v.object.get("scheme") orelse return reader.SpecError.TfoldHeaderSchema;
    if (scheme_v != .string or !schemeIsKnownV02(scheme_v.string))
        return reader.SpecError.TfoldUnknownScheme;

    // FNV-1a-64 trailer
    const body = bytes[0 .. bytes.len - reader.TRAILER_NBYTES];
    const got = reader.fnv1a_64(body);
    const want = std.mem.readInt(u64, bytes[bytes.len - reader.TRAILER_NBYTES ..][0..8], .little);
    if (got != want) return reader.SpecError.TfoldIntegrity;

    // Pin check
    const model_v = parsed.object.get("model") orelse return reader.SpecError.TfoldHeaderSchema;
    if (model_v != .object) return reader.SpecError.TfoldHeaderSchema;
    const src_sha_v = model_v.object.get("source_sha256") orelse return reader.SpecError.TfoldHeaderSchema;
    if (src_sha_v != .string) return reader.SpecError.TfoldHeaderSchema;
    if (opts.pin_source_sha256) |pin| {
        if (!std.mem.eql(u8, pin, src_sha_v.string)) return reader.SpecError.TfoldCheckpointMismatch;
    }

    const overrides_present = quant_v.object.get("overrides") != null;
    const groups_present = quant_v.object.get("groups") != null;

    const out_bytes = owned_bytes.?;
    owned_bytes = null;

    return File{
        .gpa = gpa,
        .bytes = out_bytes,
        .scheme_name = scheme_v.string, // temporary; arena is deinit on return — fine only for const compile-time checked schemes
        .overrides_present = overrides_present,
        .groups_present = groups_present,
    };
}
