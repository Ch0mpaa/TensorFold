//! Bounded packed-tensor paging for Android MoE checkpoints.
//!
//! Safetensor headers are indexed without mapping tensor payloads. A lease pins
//! one packed EXL3 tensor; eviction can remove only unpinned least-recently-used
//! entries. Bytes become visible in the cache only after a complete positional
//! read, so cancellation and short reads never publish partial weights.
const std = @import("std");
const checkpoint = @import("checkpoint");

const Io = std.Io;
const Allocator = std.mem.Allocator;
const read_chunk = 1 << 20;
const header_limit = 64 << 20;

pub const Span = struct {
    shard: u16,
    offset: u64,
    bytes: usize,
    dtype: checkpoint.DType,
    rank: u8,
    shape: [checkpoint.max_rank]usize,
};

const Shard = struct {
    path: [:0]u8,
    file: Io.File,
};

pub const Catalog = struct {
    gpa: Allocator,
    io: Io,
    arena: std.heap.ArenaAllocator,
    shards: []Shard,
    spans: std.StringHashMapUnmanaged(Span),

    /// Index every routed-expert tensor under `dir` without mapping payloads.
    pub fn open(gpa: Allocator, io: Io, dir: []const u8) !Catalog {
        const paths = try checkpoint.shardFiles(gpa, io, dir);
        defer checkpoint.freeShardFiles(gpa, paths);
        if (paths.len > std.math.maxInt(u16)) return error.TooManyShards;

        var out: Catalog = .{
            .gpa = gpa,
            .io = io,
            .arena = .init(gpa),
            .shards = try gpa.alloc(Shard, paths.len),
            .spans = .empty,
        };
        errdefer out.deinitPartial(0);

        var opened: usize = 0;
        for (paths, 0..) |path, shard_index| {
            var file = try Io.Dir.cwd().openFile(io, path, .{});
            errdefer file.close(io);
            const size: usize = @intCast(try file.length(io));
            if (size < 8) return error.BadSafetensors;
            var prefix: [8]u8 = undefined;
            if (try file.readPositionalAll(io, &prefix, 0) != prefix.len) return error.BadSafetensors;
            const header_bytes: usize = @intCast(std.mem.readInt(u64, &prefix, .little));
            if (header_bytes > header_limit or header_bytes > size - 8) return error.BadSafetensors;
            const header = try gpa.alloc(u8, header_bytes);
            defer gpa.free(header);
            if (try file.readPositionalAll(io, header, 8) != header.len) return error.BadSafetensors;
            const data_at = 8 + header_bytes;
            const names = try checkpoint.parseHeader(out.arena.allocator(), header, size - data_at);

            const own_path = try gpa.dupeSentinel(u8, path, 0);
            out.shards[shard_index] = .{ .path = own_path, .file = file };
            opened += 1;

            var it = names.iterator();
            while (it.next()) |entry| {
                const name = entry.key_ptr.*;
                if (!expertTensor(name)) continue;
                const e = entry.value_ptr.*;
                const own_name = try out.arena.allocator().dupe(u8, name);
                const result = try out.spans.getOrPut(out.arena.allocator(), own_name);
                if (result.found_existing) return error.DuplicateTensor;
                result.value_ptr.* = .{
                    .shard = @intCast(shard_index),
                    .offset = @intCast(data_at + e.begin),
                    .bytes = e.end - e.begin,
                    .dtype = e.dtype,
                    .rank = e.rank,
                    .shape = e.shape,
                };
            }
        }
        if (out.spans.count() == 0) return error.NoExpertTensors;
        return out;
    }

    fn deinitPartial(self: *Catalog, opened: usize) void {
        for (self.shards[0..opened]) |*shard| {
            shard.file.close(self.io);
            self.gpa.free(shard.path);
        }
        self.gpa.free(self.shards);
        self.spans.deinit(self.arena.allocator());
        self.arena.deinit();
    }

    pub fn deinit(self: *Catalog) void {
        self.deinitPartial(self.shards.len);
        self.* = undefined;
    }

    pub fn get(self: *const Catalog, name: []const u8) ?Span {
        return self.spans.get(name);
    }

    pub fn count(self: *const Catalog) usize {
        return self.spans.count();
    }

    pub fn packedBytes(self: *const Catalog) u64 {
        var total: u64 = 0;
        var it = self.spans.valueIterator();
        while (it.next()) |span| total += span.bytes;
        return total;
    }

    pub fn shardCount(self: *const Catalog) usize {
        return self.shards.len;
    }

    fn read(self: *Catalog, span: Span, dest: []u8, cancelled: ?*const std.atomic.Value(bool)) !void {
        if (dest.len != span.bytes or span.shard >= self.shards.len) return error.InvalidSpan;
        var done: usize = 0;
        while (done < dest.len) {
            if (cancelled) |flag| if (flag.load(.acquire)) return error.Cancelled;
            const part = dest[done..][0..@min(read_chunk, dest.len - done)];
            const got = try self.shards[span.shard].file.readPositionalAll(self.io, part, span.offset + done);
            if (got != part.len) return error.ShortRead;
            done += got;
        }
        if (cancelled) |flag| if (flag.load(.acquire)) return error.Cancelled;
    }
};

fn expertTensor(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".experts.") != null or
        std.mem.indexOf(u8, name, ".routed_experts.") != null or
        std.mem.indexOf(u8, name, ".shared_expert.") != null;
}

const Node = struct {
    name: []const u8,
    bytes: []align(4096) u8,
    pins: u32 = 0,
    newer: ?*Node = null,
    older: ?*Node = null,
};

pub const Metrics = struct {
    hits: u64,
    misses: u64,
    evictions: u64,
    bytes_read: u64,
    resident_bytes: usize,
    resident_entries: usize,
};

pub const Pager = struct {
    gpa: Allocator,
    catalog: *Catalog,
    capacity: usize,
    resident: usize = 0,
    entries: std.StringHashMapUnmanaged(*Node) = .empty,
    newest: ?*Node = null,
    oldest: ?*Node = null,
    hits: u64 = 0,
    misses: u64 = 0,
    evictions: u64 = 0,
    bytes_read: u64 = 0,

    pub fn init(gpa: Allocator, catalog: *Catalog, capacity: usize) !Pager {
        if (capacity == 0) return error.ZeroCapacity;
        return .{ .gpa = gpa, .catalog = catalog, .capacity = capacity };
    }

    pub fn deinit(self: *Pager) void {
        var node = self.newest;
        while (node) |item| {
            const older = item.older;
            self.gpa.free(item.bytes);
            self.gpa.destroy(item);
            node = older;
        }
        self.entries.deinit(self.gpa);
        self.* = undefined;
    }

    pub const Lease = struct {
        pager: *Pager,
        node: *Node,
        released: bool = false,

        pub fn data(self: *const Lease) []const u8 {
            return self.node.bytes;
        }

        pub fn release(self: *Lease) void {
            if (self.released) return;
            std.debug.assert(self.node.pins > 0);
            self.node.pins -= 1;
            self.released = true;
        }
    };

    /// Pin one packed tensor. This serial pager intentionally holds no decoded
    /// matrix and performs no allocation on a hit.
    pub fn acquire(self: *Pager, name: []const u8, cancelled: ?*const std.atomic.Value(bool)) !Lease {
        if (self.entries.get(name)) |node| {
            node.pins += 1;
            self.touch(node);
            self.hits += 1;
            return .{ .pager = self, .node = node };
        }
        const span = self.catalog.get(name) orelse return error.MissingTensor;
        if (span.bytes > self.capacity) return error.TensorExceedsCapacity;
        try self.makeRoom(span.bytes);

        const node = try self.gpa.create(Node);
        errdefer self.gpa.destroy(node);
        const bytes = try self.gpa.alignedAlloc(u8, .fromByteUnits(4096), span.bytes);
        errdefer self.gpa.free(bytes);
        try self.catalog.read(span, bytes, cancelled);
        node.* = .{ .name = self.catalog.spans.getKey(name).?, .bytes = bytes, .pins = 1 };
        try self.entries.put(self.gpa, node.name, node);
        self.insertNewest(node);
        self.resident += bytes.len;
        self.misses += 1;
        self.bytes_read += bytes.len;
        return .{ .pager = self, .node = node };
    }

    fn makeRoom(self: *Pager, incoming: usize) !void {
        while (self.resident > self.capacity - incoming) {
            var candidate = self.oldest;
            while (candidate) |node| : (candidate = node.newer) {
                if (node.pins == 0) {
                    self.remove(node);
                    _ = self.entries.remove(node.name);
                    self.resident -= node.bytes.len;
                    self.gpa.free(node.bytes);
                    self.gpa.destroy(node);
                    self.evictions += 1;
                    break;
                }
            } else return error.AllEntriesPinned;
        }
    }

    fn insertNewest(self: *Pager, node: *Node) void {
        node.newer = null;
        node.older = self.newest;
        if (self.newest) |head| head.newer = node else self.oldest = node;
        self.newest = node;
    }

    fn remove(self: *Pager, node: *Node) void {
        if (node.newer) |newer| newer.older = node.older else self.newest = node.older;
        if (node.older) |older| older.newer = node.newer else self.oldest = node.newer;
        node.newer = null;
        node.older = null;
    }

    fn touch(self: *Pager, node: *Node) void {
        if (self.newest == node) return;
        self.remove(node);
        self.insertNewest(node);
    }

    pub fn metrics(self: *const Pager) Metrics {
        return .{
            .hits = self.hits,
            .misses = self.misses,
            .evictions = self.evictions,
            .bytes_read = self.bytes_read,
            .resident_bytes = self.resident,
            .resident_entries = self.entries.count(),
        };
    }
};

fn fixture(a: Allocator, tensors: []const struct { []const u8, []const u8 }) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    var at: usize = 0;
    for (tensors, 0..) |tensor, i| {
        if (i > 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ tensor[0], tensor[1].len, at, at + tensor[1].len });
        at += tensor[1].len;
    }
    try header.append(a, '}');
    const out = try a.alloc(u8, 8 + header.items.len + at);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    var data_at = 8 + header.items.len;
    for (tensors) |tensor| {
        @memcpy(out[data_at..][0..tensor[1].len], tensor[1]);
        data_at += tensor[1].len;
    }
    return out;
}

fn testCatalog(a: Allocator, io: Io, tmp: *std.testing.TmpDir) ![]u8 {
    const names = [_]struct { []const u8, []const u8 }{
        .{ "language_model.model.layers.0.mlp.experts.0.gate_proj.q_weight", "aaaa" },
        .{ "language_model.model.layers.0.mlp.experts.1.gate_proj.q_weight", "bbbbbb" },
        .{ "language_model.model.layers.0.mlp.shared_expert.gate_proj.q_weight", "ignored" },
    };
    const image = try fixture(a, &names);
    defer a.free(image);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = image });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"language_model.model.layers.0.mlp.experts.0.gate_proj.q_weight\":\"model-00001-of-00001.safetensors\"}}" });
    return std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
}

test "catalog indexes routed and shared experts" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try testCatalog(a, io, &tmp);
    defer a.free(dir);
    var catalog = try Catalog.open(a, io, dir);
    defer catalog.deinit();
    try std.testing.expectEqual(@as(usize, 3), catalog.count());
    try std.testing.expectEqual(@as(usize, 4), catalog.get("language_model.model.layers.0.mlp.experts.0.gate_proj.q_weight").?.bytes);
    try std.testing.expectEqual(@as(usize, 7), catalog.get("language_model.model.layers.0.mlp.shared_expert.gate_proj.q_weight").?.bytes);
}

test "bounded pager hits and evicts only unpinned packed tensors" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try testCatalog(a, io, &tmp);
    defer a.free(dir);
    var catalog = try Catalog.open(a, io, dir);
    defer catalog.deinit();
    var pager = try Pager.init(a, &catalog, 9);
    defer pager.deinit();
    const zero = "language_model.model.layers.0.mlp.experts.0.gate_proj.q_weight";
    const one = "language_model.model.layers.0.mlp.experts.1.gate_proj.q_weight";
    var first = try pager.acquire(zero, null);
    try std.testing.expectEqualStrings("aaaa", first.data());
    var hit = try pager.acquire(zero, null);
    hit.release();
    try std.testing.expectError(error.AllEntriesPinned, pager.acquire(one, null));
    first.release();
    var second = try pager.acquire(one, null);
    defer second.release();
    try std.testing.expectEqualStrings("bbbbbb", second.data());
    const m = pager.metrics();
    try std.testing.expectEqual(@as(u64, 1), m.hits);
    try std.testing.expectEqual(@as(u64, 2), m.misses);
    try std.testing.expectEqual(@as(u64, 1), m.evictions);
    try std.testing.expectEqual(@as(usize, 6), m.resident_bytes);
}

test "cancelled read publishes no cache entry" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try testCatalog(a, io, &tmp);
    defer a.free(dir);
    var catalog = try Catalog.open(a, io, dir);
    defer catalog.deinit();
    var pager = try Pager.init(a, &catalog, 10);
    defer pager.deinit();
    var cancelled: std.atomic.Value(bool) = .init(true);
    const name = "language_model.model.layers.0.mlp.experts.0.gate_proj.q_weight";
    try std.testing.expectError(error.Cancelled, pager.acquire(name, &cancelled));
    try std.testing.expectEqual(@as(usize, 0), pager.metrics().resident_entries);
    try std.testing.expectEqual(@as(u64, 0), pager.metrics().misses);
}
