# Stage 6d — teaching TensorFold to read `.tfold` natively

> Beginner-friendly companion to the Stage 6d upstream work on
> `ashhart/TensorFold:stage6d-tfold-reader`. The goal was to make the
> CUDA runtime load a complete TFOLD v0.2 checkpoint **directly**, instead
> of first expanding it into a safetensors directory.

## Why memory mapping avoids an extra CPU-side copy

When a program wants to use a 3 GB file, it has two main choices:

1. **Read into a buffer.** Call `read()` repeatedly, filling heap memory
   that you own. The kernel copies pages from disk into its page cache,
   then copies again into your buffer. Total: at least one CPU-side copy
   per byte you want to touch.
2. **Memory-map the file.** Call `mmap()`. The kernel exposes the file's
   page cache directly in your process's address space. There is no
   CPU-side copy: reading `mem[i]` triggers a page fault that the kernel
   services by reading the disk page once into the page cache, and your
   program sees that same page.

For large read-only checkpoints, (2) wins: zero copies, instant startup,
page-cache sharing between processes. That is what Stage 6d uses.

### Important caveat

"Zero-copy CPU-side" is **not** the same as "zero-copy to GPU". To put
weights on the GPU, TensorFold still issues `cudaMemcpy` from host memory
to device memory. The only thing mmap removed is the *extra* host-side
copy that a read-into-buffer approach would add. Everything downstream
of host memory is unchanged.

## How Zig reads tensors stored inside a binary file

Zig's standard library gives us `std.Io.File.MemoryMap`. In Stage 6d's
`tfold_file.zig`:

```zig
var map = try Io.File.MemoryMap.create(io, file, .{
    .len = len,
    .protection = .{ .read = true, .write = false },
    .populate = false,
});
```

After this call, `map.memory` is a `[*]const u8` that behaves like an
array of every byte in the file. We then validate the preamble, parse
the JSON header into a `safetensors.Header`, and record each tensor's
`begin` and `end` as **absolute file offsets**.

When the loader asks for a tensor by name, we return a slice of the mmap:

```zig
pub fn get(self: *const File, name: []const u8) ?st.Tensor {
    const e = self.names.get(name) orelse return null;
    return .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape,
              .bytes = self.map.memory[e.begin..e.end] };
}
```

Those `bytes` point **into the mmap**. No copy. When the CUDA uploader
asks for `t.bytes.ptr`, it gets a pointer that `cudaMemcpy` can send
straight to the GPU.

## Why the "checkpoint union" ended up unnecessary

The original design (see `docs/stage6c-v02-tfoldfile-design.md`) proposed
a tagged union:

```zig
pub const BackedFile = union(enum) {
    safetensors: safetensors.File,
    tfold: tfold_file.File,
};
```

That would have required every `Checkpoint` method (`.get`, `.has`,
`.release`, `.adviseRandom`, tensor-name iteration in six downstream
files) to dispatch through the union. Six to ten files worth of mechanical
edits.

A simpler option emerged during implementation: **make the TFOLD reader
*return* a `safetensors.File` directly**. The safetensors.File struct is
just `{ path, file, map, data, names, arena }`. If we:

- parse the TFOLD header into a `safetensors.Header`,
- store each tensor's absolute file offset in `Entry.begin/.end`,
- set the returned `File.data = 0`

then `safetensors.File.get(name)` returns
`map.memory[0 + e.begin .. 0 + e.end]` — exactly the TFOLD body bytes.
The returned type IS `safetensors.File`, so every existing checkpoint
caller works unchanged. The only edit to `checkpoint.zig` is a four-line
routing change in `openModelPrefixes` that detects `dir/model.tfold`
and routes through `tfold_file.openPrefixes` instead of
`safetensors.File.openPrefixes`.

The union would have been "correct" but 10× the diff. Pick the smaller
change that keeps the contract.

## How the GPU receives model weights

Once the native reader has an mmap and a tensor layout:

1. The Qwen3.5 CUDA loader calls `ck.get("model.layers.0.linear_attn.in_proj_qkv.trellis")`.
2. `safetensors.File.get` returns `{ dtype = .i16, shape = [...], bytes = &map.memory[...] }`.
3. The upstream dense/weights path takes `bytes.ptr`, allocates a CUDA
   device buffer of `bytes.len` bytes, and issues one `cudaMemcpy`
   (host → device).
4. The kernel reads from device memory. CPU mmap and GPU memory are two
   independent copies of the bytes.

So the Stage 6d flow is:

```
.tfold file on NVMe
       │
       │  ← one page fault per touched page, into the kernel page cache
       ▼
process address space (map.memory)
       │
       │  ← one cudaMemcpy per tensor, host → device
       ▼
GPU HBM (what the kernels read)
```

The CPU-side mmap is zero-copy. The CPU-to-GPU hop is not.

## Memory ownership, alignment, verification, cleanup

**Ownership.** `tfold_file.open` owns three resources:

1. An `Io.File` (file descriptor).
2. An `Io.File.MemoryMap` (the mmap).
3. An `ArenaAllocator` that holds the parsed JSON, duplicated tensor
   names (keys for `names`), and a NUL-terminated path.

`close(io)` tears down all three in order (mmap, file, arena). If any
step during open fails after creation, `errdefer` drops the same
resources in reverse — no leaks.

**Alignment.** TFOLD v0.2 declares a body alignment of 64 bytes. The
reader checks this twice: once when it computes `body_start` (padding
the header to a 64-byte boundary) and again on every tensor's
`offset % ALIGN`. A checkpoint with a misaligned tensor is refused with
`TfoldTensorMisaligned` before any CUDA call.

**Verification.** On open, the reader runs FNV-1a-64 over every byte
from offset 0 through `file_size - 8`, and checks it against the last
8 bytes. A mismatch is `TfoldIntegrity`. This is the default policy; an
opt-out `skip_fnv` exists only so test cases can feed deliberately
malformed inputs. No production code path turns it off. (We measured
~1.5 s of FNV time on a 3.2 GB file with the naive pure-Zig loop on a
GB10.)

**Cleanup.** `close` is idempotent in intent but writes `self.* =
undefined` so a second call crashes loudly. This matches
`safetensors.File`'s own convention. When callers use `BackedFile`
(wait, we don't — see "Why the checkpoint union..." above), they still
get the same semantics.

## What was implemented

- **`zig/src/core/tfold_file.zig`** (new, ~280 LoC)
  Native v0.2 reader. Parses preamble, validates magic + major version,
  bounded header size (1 GiB cap), 64-byte body alignment, per-tensor
  offset/size/overlap/misalignment checks, FNV trailer, and dtype
  mapping (including lowercase v0.2 names, `exl3_mcg_tile` etc. → `.i16`).

- **`zig/src/core/checkpoint.zig`** (modified, +17 LoC)
  `openModelPrefixes` now checks for `dir/model.tfold` first. If
  present, the Checkpoint holds exactly one safetensors.File — but
  that File's bytes point into the TFOLD mmap, with absolute file
  offsets in each tensor's `Entry.begin/.end`. If absent, the loader
  falls back to safetensors shards exactly as before.

- **`zig/src/core/tfold_file_test.zig`** (new, 13 tests)
  Bad magic, bad version, header overflow, truncation, FNV mismatch,
  tensor overlap, tensor misalignment, tensor overflow, unsupported
  dtype, prefix filter, a known FNV vector.

## Design alternatives considered

1. **Tagged `BackedFile` union** (above). Rejected: 10× the diff for
   zero semantic gain. The returned `safetensors.File` already captures
   the only abstraction the Checkpoint needs.
2. **Convert `.tfold` to safetensors on load.** That is literally Stage
   6c's materializer and is now the fallback / A/B baseline in Gate C.
   Downside: writes 3 GB of bytes to disk for no reason.
3. **Deferred FNV verification.** Trailer computed on first page-touch
   fault, amortized. Rejected: silently weakens the default integrity
   policy, which the brief explicitly forbids.
4. **mmap `MAP_POPULATE`.** Pre-fault every page, so first-touch latency
   is paid up front. Rejected as the default (the kernel is better at
   deciding this), left as an option for a dedicated `warmup` command.
5. **Per-tensor SHA-256 verification on load.** Rejected: FNV is
   O(file-size) and already catches disk corruption / bit flips. Per-tensor
   SHA verification is still available as `verifyTensorHashes` for
   callers that want it; the default load path doesn't run it.

## Tests and failures

### Pass

| Suite | Count | Delta |
|---|---|---|
| `test-checkpoint-io-host` (includes new `tfold_file_*` + `fnv1a64`) | 63/63 | +12 new TFOLD tests |
| `test-qwen35-exl3-admission-host` | 5/5 | unchanged from Stage 6c |
| `test-exl3-cuda-host` | 1017/1027 pass, 10 skip | **+2 new tests** (both new tfold_file tests get swept in); unchanged behavior for every pre-existing test |

### Failures we hit and fixed during development

1. `std.meta.FieldType` — doesn't exist in Zig 0.17. Replaced with
   explicit `safetensors.DType.u8`.
2. `std.fmt.format` over `ArrayList.writer()` — ArrayList.writer() isn't
   defined in Zig 0.17 either. Replaced with `allocPrint` + `appendSlice`
   pairs, which is pointlessly verbose but compiles.
3. Test blobs hardcoded offset=64 but body_start depends on hlen. First
   run produced four `TfoldTensorOverflow` failures that looked like
   bugs in the reader; they were bugs in the test harness. The reader
   was right to refuse. Fixed by picking offsets like 1024 that always
   land past body_start regardless of hlen.
4. Dead `makeValid` call in the overlap test, with `.offset_override = 0`,
   caused an integer overflow because 0 is before body_start. Deleted.

## Hands-on exercise

Clone the upstream fork branch, build on a CUDA host, and reproduce
Gate C's determinism:

```sh
# 1. Clone the branch (base is upstream main + stage6c-cuda-mcg + Stage 6d).
git clone -b stage6d-tfold-reader https://github.com/Ch0mpaa/TensorFold
cd TensorFold

# 2. Install Zig 0.17.0 (your platform may differ).
curl -L https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz | tar -xJ

# 3. Build the CUDA engine + kernels.
./zig-aarch64-linux-0.17.0/zig build -Dnvcc=/usr/local/cuda-13.0/bin/nvcc install

# 4. Grab a real mcg checkpoint and convert it with the Stage 6b tool.
huggingface-cli download turboderp/Qwen3.5-2B-exl3-8.00bpw --local-dir ~/Q35
python tools/tfold_convert/convert_full.py \
    --model-dir ~/Q35 --out ~/Q35.tfold

# 5. Assemble a tfold directory and run inference.
mkdir ~/Q35-native
cp ~/Q35.tfold ~/Q35-native/model.tfold
cp ~/Q35/{config,tokenizer,tokenizer_config,chat_template,merges,vocab,preprocessor_config,quantization_config}.* ~/Q35-native/
zig-out/bin/tf-qwen35-cuda ~/Q35-native prompt \
    '760,4087,310,2199,11,279,15039,11,321,4156,369' /tmp/out-native.bin
sha256sum /tmp/out-native.bin
```

The expected SHA is
`6b198ea823135943d8d662837ca0c42e7ab4165addfa2064a23eb0be3824c1e0`
(the same SHA Gate A and Gate B reported). If you see a different hash,
either your checkpoint isn't bit-identical to turboderp's release, or
something in your stack changed.

## Comprehension questions

**Q1.** Suppose you mmap a 4 GB file on a host with 8 GB of RAM, touch
every byte once, then do it again. The first pass takes ~4 s. How long
should the second pass take, and why? What specifically makes that
number small, and what change to the system would stop it from being
small?

**Q2.** Stage 6d sets `safetensors.File.data = 0` and stores absolute
file offsets in each tensor's `Entry.begin/.end`. The original
safetensors code sets `data = 8 + header_len` and stores *body-relative*
offsets. Both approaches present the same `get()` surface. Why did the
TFOLD reader pick the absolute form, and when would the choice actually
matter?

---

*These questions are deliberately left unanswered. Answer them for
yourself; a maintainer may discuss your answer in review.*
