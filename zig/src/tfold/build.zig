//! Standalone build.zig for the .tfold v0.1 native loader + Stage 5
//! EXL3 conversion tests.
//!
//! Invoke from the repo root:
//!
//!   zig build -Dtarget=native --build-file zig/src/tfold/build.zig run-demo
//!   zig build -Dtarget=native --build-file zig/src/tfold/build.zig test
//!   zig build -Dtarget=aarch64-linux-gnu --build-file zig/src/tfold/build.zig

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("tfold", .{
        .root_source_file = b.path("tfold.zig"),
    });

    // Demo exe
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
    const run_step = b.step("run-demo", "Run tfold-cpu-demo");
    run_step.dependOn(&run_demo.step);

    // Core tests (reader + ops + demo fixture round-trip)
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(tests);

    // Stage 5 tests (converted EXL3 fixture)
    // Stage 5 tests (converted EXL3 fixture) — needs core/exl3_format.zig
    // for the production decoder.
    const exl3_format_mod = b.createModule(.{
        .root_source_file = b.path("../core/exl3_format.zig"),
        .target = target,
        .optimize = optimize,
    });
    const exl3_tests_mod = b.createModule(.{
        .root_source_file = b.path("exl3_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    exl3_tests_mod.addImport("exl3_format", exl3_format_mod);
    const exl3_tests = b.addTest(.{ .root_module = exl3_tests_mod });
    const run_exl3 = b.addRunArtifact(exl3_tests);
    const test_step = b.step("test", "Run tfold native loader + EXL3 conversion tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_exl3.step);

    // Stage 6b Gate 3 — v0.2 tests
    const v02_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("v02_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_v02 = b.addRunArtifact(v02_tests);
    test_step.dependOn(&run_v02.step);
}
