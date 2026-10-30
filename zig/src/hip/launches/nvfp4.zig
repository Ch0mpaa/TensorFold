//! NVFP4 routed-expert launches: reference decode + prompt (ops/nvfp4.hpp).

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const C = util.C;
const CI = util.CI;
const F = util.F;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;

/// Row BLOCK_ROWS of ops/nvfp4.hpp's prompt kernel.
pub const prompt_block_rows: c_int = 128;

/// tf_nvfp4_expert_rows_f32: one block per (item, output-column-group), 32 threads, one output column a thread.
/// `x` is bf16 input rows laid out `[*, x_stride]`. `out` is fp32 `[pair_count, N]`.
pub fn expertRowsF32(
    l: *const Launcher,
    x: C,
    x_stride: c_int,
    slots: c_int,
    codes: C,
    scales: C,
    globals: C,
    K: c_int,
    N: c_int,
    items: CI,
    counts: CI,
    members: CI,
    out: F,
    limit: f32,
    skip: c_int,
    item_count: c_int,
    s: S,
) Error!void {
    const nb = @divFloor(N, 32);
    if (item_count == 0 or nb == 0) return;
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(x_stride);
    try a.add(slots);
    try a.add(ad(codes));
    try a.add(ad(scales));
    try a.add(ad(globals));
    try a.add(K);
    try a.add(N);
    try a.add(ad(items));
    try a.add(ad(counts));
    try a.add(ad(members));
    try a.add(ad(out));
    try a.add(limit);
    try a.add(skip);
    try l.go(
        l.op.nvfp4_rows_f32,
        dim(item_count * nb, 1, 1),
        dim(32, 1, 1),
        0,
        s,
        &a,
    );
}

/// tf_nvfp4_expert_prompt_f32: same arithmetic for prompt tiles; adds a BLOCK_ROWS row-tile dimension.
/// `rows_per_item` is the largest `count` across items in this launch so the grid covers every tile.
pub fn expertPromptF32(
    l: *const Launcher,
    x: C,
    x_stride: c_int,
    slots: c_int,
    codes: C,
    scales: C,
    globals: C,
    K: c_int,
    N: c_int,
    items: CI,
    counts: CI,
    members: CI,
    out: F,
    limit: f32,
    skip: c_int,
    item_count: c_int,
    rows_per_item: c_int,
    s: S,
) Error!void {
    const nb = @divFloor(N, 32);
    if (item_count == 0 or nb == 0 or rows_per_item == 0) return;
    const row_tiles: c_int = @divFloor(rows_per_item + prompt_block_rows - 1, prompt_block_rows);
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(x_stride);
    try a.add(slots);
    try a.add(ad(codes));
    try a.add(ad(scales));
    try a.add(ad(globals));
    try a.add(K);
    try a.add(N);
    try a.add(ad(items));
    try a.add(ad(counts));
    try a.add(ad(members));
    try a.add(ad(out));
    try a.add(limit);
    try a.add(skip);
    try l.go(
        l.op.nvfp4_prompt_f32,
        dim(item_count * nb, row_tiles, 1),
        dim(32, 1, 1),
        0,
        s,
        &a,
    );
}

/// tf_nvfp4_rows_f32: non-routed [m, K] bf16 x [N, K] NVFP4 projection, fp32 out [m, N].
/// Grid: (N/32, m, 1); 32 threads/block.
pub fn rowsF32(
    l: *const Launcher,
    x: C,
    x_stride: c_int,
    codes: C,
    scales: C,
    globals: C,
    K: c_int,
    N: c_int,
    out: F,
    m: c_int,
    s: S,
) Error!void {
    const nb = @divFloor(N, 32);
    if (m == 0 or nb == 0) return;
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(x_stride);
    try a.add(ad(codes));
    try a.add(ad(scales));
    try a.add(ad(globals));
    try a.add(K);
    try a.add(N);
    try a.add(ad(out));
    try a.add(m);
    try l.go(
        l.op.nvfp4_rows_f32_nr,
        dim(nb, m, 1),
        dim(32, 1, 1),
        0,
        s,
        &a,
    );
}
