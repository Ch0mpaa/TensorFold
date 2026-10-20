const std = @import("std");
const pager = @import("expert_pager");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) {
        std.debug.print("usage: {s} MODEL_DIR\n", .{args[0]});
        return 64;
    }
    var catalog = pager.Catalog.open(init.gpa, init.io, args[1]) catch |err| {
        std.debug.print("{{\"supported\":false,\"error\":\"{s}\"}}\n", .{@errorName(err)});
        return 65;
    };
    defer catalog.deinit();
    std.debug.print("{{\"supported\":true,\"expert_tensors\":{d},\"packed_expert_bytes\":{d},\"shards\":{d}}}\n", .{
        catalog.count(),
        catalog.packedBytes(),
        catalog.shardCount(),
    });
    return 0;
}
