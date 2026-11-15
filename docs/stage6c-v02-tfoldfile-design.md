# Stage 6c v0.2 integration — TfoldFile adapter design

## Goal

Teach the upstream TensorFold checkpoint loader (`zig/src/core/checkpoint.zig::Checkpoint`) to open a directory containing a `model.tfold` v0.2 file and serve tensors through the same `get/has` surface the CUDA EXL3 admit path already uses — so that `tensorfold run MODEL.tfold-dir --tokens ...` produces inference tokens with no changes to the Qwen3.5 forward.

## Why zero-copy is possible

TFOLD v0.2 preserves the EXL3 tile byte layout verbatim: the v0.2 converter
(`tools/tfold_convert/convert_full.py`) stages each tensor contiguously in the
body in its original stored form, and the v0.2 header records
`(name, dtype, shape, offset, nbytes, sha256)` for every tensor. Our `__codebook__`
synthetic tensor mirrors the per-group `.mcg` or `.mul1` scalar magic marker
that the qwen3_5 loader expects.

That means a `TfoldFile` can expose exactly the same `(bytes: []const u8, dtype, rank, shape)` shape `safetensors.File.get` returns, pointing straight into
one mmap of the `.tfold` body — no re-emit, no repacking.

## Surface to mirror

From `zig/src/core/safetensors.zig`:

```
pub const File = struct {
    path: [:0]const u8,
    file: Io.File,
    map: Io.File.MemoryMap,
    data: usize,
    names: Header,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa, io, path) !File;
    pub fn openPrefixes(gpa, io, path, prefixes) !File;
    pub fn close(self, io) void;
    pub fn closeCopied(self, io) void;
    pub fn release(self, scratch) void;
    pub fn releaseExcept(self, keep, scratch) void;
    pub fn adviseRandom(self, bytes) void;
    pub fn get(self, name) ?Tensor;
};
```

Tensor is the same `{ dtype, rank, shape, bytes }` record both paths return.

## Design — `zig/src/core/tfold_file.zig`

A sibling struct with the same surface:

```
pub const File = struct {
    path: [:0]const u8,
    file: Io.File,
    map: Io.File.MemoryMap,
    data: usize,             // PREAMBLE + header_len (tfold body offset)
    names: Header,           // reuses safetensors.Header for API parity

    // v0.2 header JSON once parsed, retained in arena
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa, io, path) !File;
    pub fn openPrefixes(gpa, io, path, prefixes) !File;
    pub fn close(self, io) void;
    pub fn release(self, scratch) void;
    pub fn adviseRandom(self, bytes) void;
    pub fn get(self, name) ?Tensor;
};
```

### Tensor name expansion

v0.2 stores a `quant.groups` list where each group has a base name
(`model.layers.0.linear_attn.in_proj_qkv`) and child tensors with suffixes
(`.trellis`, `.suh`, `.svh`, `.mcg`/`.mul1`). The Qwen3.5 CUDA loader requests
`<base>.trellis`, `<base>.mcg`, etc. as full names via `ck.get`.

`TfoldFile.open` enumerates every tensor from the v0.2 header AND synthesizes
`<base>.<codebook>` marker tensors from each group's codebook field. The
marker tensor's bytes point at a group's dedicated 4-byte magic slot in the
body; the converter writes one such scalar per group.

### `Checkpoint` wiring

`Checkpoint.files` is currently `std.ArrayList(safetensors.File)`. Two options:

**Option A — tagged union field:**

```
const BackedFile = union(enum) {
    safetensors: safetensors.File,
    tfold: tfold_file.File,

    pub fn get(self, name) ?safetensors.Tensor { ... }
    pub fn close(self, io) void { ... }
    pub fn release(self, scratch) void { ... }
    ...
};
```

Change `Checkpoint.files: std.ArrayList(BackedFile)` and dispatch every call.

**Option B — same-shape duck-typing:** Give `TfoldFile` the exact same field
and method layout as `safetensors.File`, so existing `Checkpoint` code
compiles unchanged if we swap the type parameter. Requires making `files`
generic over the file type.

Option A is honest and more maintainable; Option B is cute but brittle.
**Choose A.**

### `openModelPrefixes` detection

```zig
pub fn openModelPrefixes(gpa, io, dir, prefixes) !Checkpoint {
    try save_guard.check(io, dir);
    // Try TFOLD v0.2 first by canonical filename.
    const tfold_path = try std.fs.path.join(gpa, &.{ dir, "model.tfold" });
    defer gpa.free(tfold_path);
    if (fileExists(io, tfold_path)) {
        var ck: Checkpoint = .{ .gpa = gpa, .io = io };
        errdefer ck.close();
        try ck.files.ensureTotalCapacityPrecise(gpa, 1);
        ck.files.appendAssumeCapacity(.{ .tfold = try tfold_file.File.openPrefixes(gpa, io, tfold_path, prefixes) });
        return ck;
    }
    // Existing safetensors path.
    ...
}
```

The directory must still contain `config.json`, tokenizer files etc., exactly
like a safetensors checkpoint directory. Only `model.safetensors*` is
replaced by `model.tfold`.

## Scope

Three files touched in upstream fork:

| | File | Change |
|---|---|---|
| **new** | `zig/src/core/tfold_file.zig` | v0.2 reader, `safetensors.File` surface |
| mod | `zig/src/core/checkpoint.zig` | `BackedFile` union + `openModelPrefixes` TFOLD-first detection |
| mod | `zig/build/core.zig` (or wherever) | wire the new file into the `core` module |

One new test file:

| | File | Change |
|---|---|---|
| **new** | `zig/src/core/tfold_file_test.zig` | synthetic .tfold with one group, admit path round-trip |

## Risks and non-goals

- **Not a general-purpose format shim:** the adapter assumes EXL3 tile layout;
  non-EXL3 TFOLD files (plain bf16, GGUF-style, ModelOpt) would need their
  own byte-identity contracts.
- **No v0.1 support here:** v0.1 lacks `quant.groups`, so the Qwen3.5 EXL3
  admit path has no base→suffix map. Only v0.2 adapts.
- **Vocab/embed tables:** v0.2 serializes these as ordinary tensors with
  their HF names (`model.language_model.embed_tokens.weight`), so the
  dispatch is identical to safetensors — no synthesis needed.
- **Prefetch/release:** `adviseRandom`, `release`, `releaseExcept` just
  pass through to the TFOLD body mmap; semantics are identical.

## Testing

1. Synthetic .tfold with one `.mcg` group of 128×128 trellis + suh + svh; admit
   must succeed and return `Codebook.mcg`.
2. Real Qwen3.5-2B-exl3-mcg .tfold (3.2 GiB) opened by `tensorfold run`:
   `--tokens 760,4087,310,2199,11,279,15039,11,321,4156,369 --max-tokens 16`
   must emit the same 16 tokens as the same prompt through the original
   safetensors checkpoint (Gate A receipt).

## Status

Design only — implementation is the Stage 6c v0.2 integration phase.
Blockers downstream:
1. `BackedFile` union is invasive to upstream (6+ call sites in `checkpoint.zig`).
2. Needs to land as a second PR against `ashhart/TensorFold` after #3412.

If upstream rejects or delays, we vendor into our fork as a Stage 6c follow-up.
