//! Standalone build.zig for the .tfold v0.1 native loader.
//!
//! NOT connected to TensorFold's monolith build.zig to keep stage-4 work
//! isolated. Invoke from the repo root:
//!
//!   zig build -Dtarget=native --build-file zig/src/tfold/build.zig run-demo
//!   zig build -Dtarget=native --build-file zig/src/tfold/build.zig test
//!   zig build -Dtarget=aarch64-linux-gnu --build-file zig/src/tfold/build.zig
//!     (ARM64 cross-compile check; no Android runtime needed)

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Reusable module
    const tfold_mod = b.addModule("tfold", .{
        .root_source_file = b.path("tfold.zig"),
    });
    _ = tfold_mod;

    // Demo executable
    const demo_exe = b.addExecutable(.{
        .name = "tfold-cpu-demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("demo.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(demo_exe);

    const run_demo = b.addRunArtifact(demo_exe);
    // (Build.args passthrough is Zig 0.18+; intentionally omitted here.)
    const run_step = b.step("run-demo", "Run tfold-cpu-demo");
    run_step.dependOn(&run_demo.step);

    // Tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tfold native loader tests");
    test_step.dependOn(&run_tests.step);
}
