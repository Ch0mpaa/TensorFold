//! Public entry point for the native .tfold v0.1 loader + CPU reference
//! ops. Intended to be imported from TensorFold's main tree once the
//! format reaches Stage 5 (EXL3 converter).

const std = @import("std");
pub const reader = @import("reader.zig");
pub const ops = @import("ops.zig");

pub const Dtype = reader.Dtype;
pub const File = reader.File;
pub const Header = reader.Header;
pub const Tensor = reader.Tensor;
pub const SpecError = reader.SpecError;
pub const OpError = ops.OpError;
pub const OpenOptions = reader.OpenOptions;
pub const openPath = reader.openPath;
pub const openBytes = reader.openBytes;
