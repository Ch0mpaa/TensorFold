# Stage 6e — measuring and tuning the native TFOLD open path

> Beginner-friendly companion to Stage 6e. The milestone was not "make
> it faster"; it was "**measure** where the time goes and change what the
> numbers justify, without touching correctness or integrity."

## Why full-file hashing costs time

A model.tfold v0.2 ends with an 8-byte FNV-1a-64 trailer that is computed
over every byte before it (preamble + header + 64-byte pad + body). The
reader recomputes that hash on open and refuses the file on any
mismatch. That's the file's whole-of-file integrity check.

For a 3.2 GB checkpoint, FNV-1a-64 looks like this inner loop:

```zig
for (bytes) |b| {
    h ^= @as(u64, b);
    h = h *% PRIME;
}
```

Two instructions per byte. **There's no way to skip bytes without
breaking the hash** — every byte XORs into `h`, and the next byte's
XOR depends on the previous byte's multiply result. The multiply is the
dependency chain's limiting factor: ~4 cycles on aarch64, ~3 on modern
x86, so you can reach ~1-2 GB/s on a single core.

At 1-2 GB/s, a 3.2 GB file takes 1.5-3 s to verify. That matches what we
observed in Stage 6d (`fnv_ns` ≈ 1.3-1.6 s on the GB10 Spark).

Takeaway: FNV cost is real and proportional to file size. It is the
single biggest knob on native-TFOLD open time.

## How memory mapping affects resident memory

When you `mmap(3.2 GB file, PROT_READ)`:

- **Virtual size (`VmSize`) grows by 3.2 GB.** Virtual size costs ~nothing
  on a 64-bit system; it's just reservations in your page table.
- **Resident Set Size (`VmRSS`) grows only as you touch pages.** Each
  page you read for the first time triggers a fault that the kernel
  services by paging it into the per-process address space AND the
  page cache. That counts toward RSS.
- **`RssFile` grows for file-backed pages; `RssAnon` grows for private
  dirty pages.** Our mapping is read-only, so dirty pages are rare;
  but Linux still reports the mmap pages in `RssFile`.

Stage 6d's FNV scan touches every page of the 3.2 GB file. After the
scan, `RssFile` is ~3.2 GB high. If we then `cudaMemcpy` the weights to
the GPU, the host-side mapping is no longer logically needed. But
nobody told the kernel.

`madvise(ptr, len, DONTNEED)` is how we tell it: "I don't plan to look
at these bytes again; feel free to reclaim the PTEs." Importantly:

- It does **not** unmap the region. The virtual address space is still
  there; a later read re-faults the page back in.
- It does **not** evict the page cache. Those bytes stay available for
  other processes (or ourselves, on the next open) to reuse cheaply.
- It **does** drop our per-process PTEs, so `VmRSS` and `RssFile`
  decrease. The kernel accounting now shows our process holding fewer
  pages.

That's the Stage 6e win on the residency axis: after FNV verification
and before handing out any `.get(name)` slice, call DONTNEED on the
whole mapping. The Qwen3.5 loader walks tensors in order and pages
them back in cheaply from the page cache during `cudaMemcpy`.

### What could go wrong

If DONTNEED races with a live `cudaMemcpy` (another thread reading the
same bytes), the memory read **does not fail** — DONTNEED is advisory.
The page faults back in and the copy completes. The only observable
cost is one disk/page-cache fault.

If the mapping were `MAP_PRIVATE | MAP_WRITABLE` and dirty, DONTNEED
would **zero** those pages. Our mapping is PROT_READ only, so dirty
pages cannot exist. Safe.

## Which optimizations we tried

### 1. Loop-unrolled FNV (`fnv1a64_unrolled`)

Stage 6d used a byte-at-a-time for-loop. Stage 6e adds a second variant
that unrolls 16 iterations per loop body. The hash is still strictly
data-dependent — the unroll only cuts loop overhead (compare + branch)
by 16×. In Debug/ReleaseSmall the reference `fnv1a64_scalar` is kept;
ReleaseFast/ReleaseSafe uses the unrolled form.

A cross-check test hashes a 16 MB random buffer with both variants and
asserts they return the identical u64. Tail-length edge cases (lengths
1, 7, 8, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025) are checked too.

### 2. Post-verify `madvise(DONTNEED)`

After FNV succeeds, the reader calls `madvise(DONTNEED)` on the whole
mapping. This is on by default; the policy option `drop_pages_after_
verify` lets a caller turn it off if they want to keep the first-touch
read amortization.

### 3. Phase-timed `OpenStats`

`openOptions` now unconditionally populates a per-call `OpenStats`
struct with nanosecond timings of mmap/preamble-check/FNV/header-parse/
drop-pages phases, plus the number of bytes FNV covered. Callers pass
`stats_out: ?*OpenStats` to receive it. The environment variable
`TF_TFOLD_LOG_STATS=1` wired through `core.Checkpoint.openModelPrefixes`
logs the stats to stderr on every open.

### 4. Opt-out `skip_fnv` only for malformed-input tests

Stage 6d already had `skip_fnv`. Stage 6e preserves the policy: no
production code path turns it off. The integrity policy is not weakened
to report a speedup.

## Why some things did not work

- **Processing 8 bytes at a time via u64 word XOR.** FNV-1a-64's step is
  "XOR one byte, multiply." Swapping in a full u64 word changes the
  hash — now you'd XOR 8 bytes at once and multiply only once, producing
  a different u64 output. Rejected: hash bytes must stay identical.
- **Parallelizing FNV across cores.** Dependent hash chain; the final
  result requires sequential folding. Possible with careful tree-hash
  reductions, but changes the hash. Rejected.
- **Switching to SHA256 (harder-coded natively).** Changes the trailer
  bytes. Rejected: spec is FNV-1a-64.
- **Eagerly calling `MAP_POPULATE`.** Moves all faults up-front, but
  doesn't change the total page cost. No net win in cold-case; costs
  startup in warm-case where Linux's demand-paging is already optimal.
  Rejected as the default.
- **Processing u64 words through the dependent chain:** reducing loads by
  8× but keeping the byte-wise FNV step. Investigated: measured within
  noise of byte-wise code, because the byte loop was already memory-
  bandwidth-limited on this workload. Not adopted.

## What the actual benchmarks showed

See `docs/stage6e-perf-receipt.json` for the machine-readable
measurements. Summary:

| Phase | Stage 6d (unrolled OFF) | Stage 6e (unrolled ON, drop_pages ON) |
|---|---|---|
| mmap create | <1 ms | <1 ms |
| preamble parse | <1 ms | <1 ms |
| FNV scan (3.2 GB) | *filled from receipt* | *filled from receipt* |
| header parse | *filled from receipt* | *filled from receipt* |
| drop_pages | — | *filled from receipt* |
| total native open | *filled* | *filled* |
| native_warm inference wall | 5.523 s | *filled from receipt* |
| safetensors warm | 4.210 s | 4.210 s (unchanged baseline) |
| peak RSS (native) | 7.3 GB | *filled from receipt* |

**We will not claim TFOLD outperforms safetensors unless the measured
warm walls show it.** The brief forbids that claim without controlled
evidence.

### Equivalent verification work

To be fair to TFOLD, Stage 6e also measures safetensors through an
equivalent-work mode: the whole safetensors file is scanned with the
same FNV-1a-64 implementation before inference, matching the TFOLD
trailer cost. This isolates the integrity cost from the file-format
cost. See the receipt for `safetensors_fnv_scanned` numbers.

## Hands-on exercise

Rebuild the Stage 6e binary on a CUDA host and reproduce the phase
breakdown:

```sh
git clone -b stage6e-tfold-perf https://github.com/Ch0mpaa/TensorFold
cd TensorFold
curl -L https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz | tar -xJ
./zig-aarch64-linux-0.17.0/zig build -Dnvcc=/usr/local/cuda-13.0/bin/nvcc install

# Run with per-phase logging on. Values are microseconds.
TF_TFOLD_LOG_STATS=1 zig-out/bin/tf-qwen35-cuda ~/Q35-native prompt \
    '760,4087,310,2199,11,279,15039,11,321,4156,369' /tmp/out.bin
```

Expected stderr line format:

```
tfold_file open: mmap=<Nus> preamble=<Nus> fnv=<Nus> header_parse=<Nus> drop_pages=<Nus> fnv_bytes=<N>
```

`fnv` divided by `fnv_bytes` tells you your single-core FNV throughput.

## Comprehension questions

**Q1.** We observed FNV-1a-64 at ~1.5-2 GB/s single core. FNV-1a-64 is
strictly dependent (each byte's step depends on the previous byte's
multiply result). Suggest one legitimate way to reach ≥5 GB/s on the
same file without changing the final 8-byte digest value stored in the
trailer. What is one practical cost of your approach? (Hint: think
about separating the *verification-time computation* from the
*on-disk representation*.)

**Q2.** Stage 6e's DONTNEED call drops the ~3 GB of per-process PTEs
after FNV but keeps the page cache. On a hot-cache rerun, RSS stays
low AND the Qwen loader's first-touch reads are cheap. On a cold-cache
run, RSS also stays low — but what happens to wall time, and why? If
the wall-time delta is bad, sketch one policy change that keeps the
RSS win on warm runs without making cold runs slower.

---

*These questions are deliberately left unanswered. Write out your own
answers and keep them for a review discussion.*
