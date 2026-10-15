//! Minimal Android/Qualcomm bring-up binary.
//! This deliberately has no libc or vendor SDK dependency. The next gate adds
//! the host-side FastRPC boundary after this executable is proven on-device.

const std = @import("std");
const builtin = @import("builtin");

pub fn main(init: std.process.Init) !u8 {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux or builtin.abi != .android)
        return error.UnsupportedTarget;

    try std.Io.File.stdout().writeStreamingAll(
        init.io,
        "{\"tensorfold\":\"qualcomm-smoke\",\"arch\":\"aarch64\",\"abi\":\"android\",\"stage\":0}\n",
    );
    return 0;
}
