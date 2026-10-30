# AMD gfx1151 port notes — rolling log

Working journal for the AMD Ryzen AI MAX+ 395 (Radeon 8060S, `gfx1151`) port
effort. Each session appends to `log.md`; verified evidence graduates to a
proper `docs/receipts/gfx1151-*` folder once a milestone is reached.

## Target box

- Host: `dk10-hathaway27.degima.ai` (container)
- CPU: AMD Ryzen AI MAX+ 395 (16c/32t)
- GPU: AMD Radeon 8060S iGPU, `gfx1151` (RDNA 3.5), 40 CUs, 128 GiB unified
- OS: Ubuntu 24.04.5 LTS, kernel 6.14.0-1020-oem
- Device: `/dev/kfd`, `/dev/dri/renderD128`
- Disk: 1.4 TiB free on `/`

## Toolchain (snapshot 2026-10-30)

| Tool | Version | Path |
| --- | --- | --- |
| ROCm | 10.1.0-3 (`amdrocm-*10.1`) | `/opt/rocm` |
| HIP (hipcc) | 7.16.26385-0000000 | `/usr/bin/hipcc` |
| AMD clang | 24.0.0git (ROCm fork d6f6cb69) | `/usr/bin/amdclang++` |
| Zig | 0.17.0 | `/usr/local/bin/zig` (symlink → `/opt/zig-0.17.0/zig`) |
| Python | 3.x | `/usr/bin/python3` |
| rocBLAS gfx1151 | 10.1.0-3 prebuilt | `amdrocm-blas10.1-gfx1151` |
| gcc | present | `/usr/bin/gcc` |
| cmake | missing | — |
| rocm-smi | missing | — |
| HIP device bitcode | present | `/opt/rocm/core-10.1/lib/llvm/amdgcn/bitcode` (`oclc_isa_version_1151.bc` ✓) |
| HIP headers | present | `/opt/rocm/include/hip/hip_runtime.h` ✓ |

**Toolchain drift vs the project's published receipts** (which used HIP 7.1,
clang 21, llvm-21 bitcode): our ROCm is newer (10.1 vs 7.x generation) and
clang is 24. The `-Dhip-relax-branches` flag was added for a clang-21 workaround
on `gfx1100` (see `zig/build/hip.zig:17`, `zig/build/hip.zig:374`); whether a
similar workaround is needed on clang 24 for `gfx1151` is open — tracked below.

## Build command (first attempt, with corrected paths)

```sh
zig build -Dhip=true native-hip \
  -Dhipcc=/usr/bin/hipcc \
  -Dhip-include=/opt/rocm/include \
  -Dhip-arch=gfx1151 \
  -Dhip-device-lib=/opt/rocm/core-10.1/lib/llvm/amdgcn/bitcode \
  -Doptimize=ReleaseFast
```

Differences from `docs/receipts/gfx1151-qwen-833bbfcb/` reproducer:

- `-Dhip-include` was `/usr/include`; our box only has HIP headers under
  `/opt/rocm/include/hip`, so pointed to `/opt/rocm/include`.
- `-Dhip-device-lib` was `/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode`; our
  LLVM is bundled with ROCm, so pointed to
  `/opt/rocm/core-10.1/lib/llvm/amdgcn/bitcode`.

## Phase plan

1. **Phase 0** — bootstrap the box: install Zig 0.17.0 ✓, locate HIP paths ✓,
   wire the fork `Ch0mpaa/TensorFold` and branch `amd-port-gfx1151` ✓, run the
   first `native-hip` build and `zig build test` host suite.
2. **Phase 1** — re-qualify the Qwen3.8-27B baseline on current `main` with
   this toolchain; produce `docs/receipts/gfx1151-qwen-<sha>/` matching the
   v1.0.12 shape (model-check + serve-check + serve-cancel).
3. **Phase 2** — perf baseline + first wins: run `hip-affine-tune`, record
   prefill and decode tokens/s at 2k/8k/16k; compare to published CUDA numbers;
   land tuned tiles.
4. **Phase 3** — close the explicit feature gap: implement DFlash2 drafter on
   HIP so `serve --drafter` stops refusing on `rocm` (see
   `docs/receipts/hip-refuses-drafter/`).
5. **Phase 4** — extend families on HIP (Gemma 4, Nemotron-H, or GLM-5.3) per
   `docs/recipes/adding-a-zig-family.md`.

See [log.md](log.md) for the rolling session journal.
