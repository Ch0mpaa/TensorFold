const std = @import("std");
const builtin = @import("builtin");

const required_symbols = [_][:0]const u8{
    "remote_handle64_open",
    "remote_handle64_invoke",
    "remote_handle64_close",
    "rpcmem_alloc",
    "rpcmem_to_fd",
    "rpcmem_free",
};

pub fn main(init: std.process.Init) !u8 {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux or builtin.abi != .android)
        return error.UnsupportedTarget;

    var lib = std.DynLib.open("/vendor/lib64/libcdsprpc.so") catch {
        try std.Io.File.stderr().writeStreamingAll(init.io, "qualcomm-probe: cannot load libcdsprpc.so\n");
        return 2;
    };
    defer lib.close();

    for (required_symbols) |symbol| {
        if (lib.lookup(*const anyopaque, symbol) == null) {
            try std.Io.File.stderr().writeStreamingAll(init.io, "qualcomm-probe: required FastRPC symbol missing\n");
            return 3;
        }
    }

    try std.Io.File.stdout().writeStreamingAll(
        init.io,
        "{\"tensorfold\":\"qualcomm-probe\",\"libcdsprpc\":true,\"fastrpc_symbols\":true,\"dsp_session_opened\":false,\"stage\":1}\n",
    );
    return 0;
}
