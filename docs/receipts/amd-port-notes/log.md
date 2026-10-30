# Session log

## 2026-10-30 — Phase 0 bootstrap

- Confirmed box reachable via `ssh -i ~/.ssh/id_itx -p 10104 root@dk10-hathaway27.degima.ai`.
- Captured toolchain (see `README.md`): ROCm 10.1, HIP 7.16, amdclang 24, no Zig.
- Installed Zig 0.17.0 to `/opt/zig-0.17.0`, symlinked `/usr/local/bin/zig`.
- Forked `ashhart/TensorFold` → `Ch0mpaa/TensorFold`; wired remotes on both
  the workstation checkout and `~/TensorFold` on the box; cut
  `amd-port-gfx1151` on each.
- About to attempt the first HIP build with path overrides (box has ROCm-bundled
  LLVM, not system `llvm-21`); any build errors are captured below.

### Build attempts

**Attempt 1** — `-Dhipcc=/usr/bin/hipcc`:

```
sh: 1: /usr/lib/llvm/bin/clang++: not found
failed command: /usr/bin/hipcc --genco ... --rocm-path=/usr ...
```

Cause: `/usr/bin/hipcc` is a symlink into `/etc/alternatives` and derives
`--rocm-path=/usr` from its own location, then looks for
`/usr/lib/llvm/bin/clang++` which does not exist on this box. The real
ROCm toolchain lives under `/opt/rocm/`.

Fix: point the build at the ROCm-rooted hipcc with
`-Dhipcc=/opt/rocm/bin/hipcc`; its derived `--rocm-path=/opt/rocm` finds
`/opt/rocm/llvm/bin/clang++` (symlink to `clang`).

**Attempt 2** — `-Dhipcc=/opt/rocm/bin/hipcc`:

```
/opt/rocm/core-10.1/lib/llvm/lib/clang/24/include/cuda_wrappers/cmath:30:2:
  error: "Could not find standard C++ header 'cmath'. ..."
/opt/rocm/core-10.1/lib/llvm/lib/clang/24/include/__clang_hip_runtime_wrapper.h:121:10:
  fatal error: 'cstdlib' file not found
```

Cause: amdclang 24 picks `/usr/lib/gcc/x86_64-linux-gnu/14` as its GCC
toolchain root but `libstdc++-14-dev` was not installed, so no
`/usr/include/c++/14/{cmath,cstdlib}`.

Fix: `apt-get install -y g++ libstdc++-14-dev` → provides the C++20
standard library headers under `/usr/include/c++/14/`. Both headers
confirmed present after install.

**Attempt 3** — same flags, with `libstdc++-14-dev` installed: in progress.

### Phase 0 results

**Attempt 3 succeeded.** `zig build native-hip` with
`-Dhipcc=/opt/rocm/bin/hipcc -Dhip-include=/opt/rocm/include
-Dhip-arch=gfx1151 -Dhip-device-lib=/opt/rocm/core-10.1/lib/llvm/amdgcn/bitcode
-Doptimize=ReleaseFast` built in 182 s (binary 47,467,344 bytes, sha256
`5349aeb322f44e314e76d7c7844a554e289af501d2ca62c74adb3e39aec4b7b8`,
`version 1.1.9`).

### Caps probe failed silently without the loader entry

First GPU smoke returned:

```
tensorfold: HIP device 0 is not a GPU this engine supports
```

Cause: the ROCm 10.1 packages shipped on this box do not install
`/etc/ld.so.conf.d/rocm.conf`, so `libamdhip64.so` is only in
`/opt/rocm/lib`, not in `ldconfig`'s cache. The engine's caps probe at
`zig/src/hip/device.zig:46-52` opens the runtime via `platform.DynLib.open`
and `catch return null`s silently on `DriverUnavailable`, so the engine
reports the chip as unsupported instead of as unreachable.

Fix on the box: `echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig`.
After that, `tensorfold-native capabilities --json` reports
`{"backends":["hip"],"chip":"rdna3","version":"1.1.9"}` with no env
overrides.

**Known cosmetic bug**: the probe reports `chip: "rdna3"` on gfx1151,
which is RDNA 3.5 (`Generation.rdna3_5` in `zig/src/hip/caps.zig`). Tracked
as "fix chip label for rdna3_5" below; does not affect kernel selection
because WMMA codegen keys off `Caps.matrix` and `Caps.generation`, both of
which are correct.

### End-to-end serve of Qwen3.5-9B on gfx1151

```
[tensorfold] HIP rank 0 of 1: policy matrix=auto,attention=auto,kernels=auto,graphs=auto,prefill_step=1024,fill_rows=512; mtp drafts 3 under 0.3, 8 snapshots
[tensorfold] HIP gfx1151 device 0, rank 0 of 1: model 4.69 GiB; 2 streams at once, 0.18 GiB each at a 4096-token window, of 52.7 GiB left after a 6.2 GiB reserve; prompt cache 1.00 GiB (311 pages, 8 snapshots); prompts in 1024-row chunks
[tensorfold] serving amd-9b at http://127.0.0.1:8091/v1 (sampling: greedy; drafts: copies only; context: 4096; loaded in 8.9s)
```

Clean SIGTERM drain, 146 of 152 captured graphs replayed. First proof the
HIP engine runs on this box.

### Host test suite

`zig build test` with HIP enabled: 258 s, **Build Summary: 465/468 steps
succeeded (1 failed); 5843/5876 tests passed (31 skipped, 2 failed)**.

Both failures are in `drama_voice_tool` cleanup checks that assume a non-root
account whose directory permissions actually deny writes; this container runs
as `uid=0`, so the restrict-and-fail path is unreachable. Not AMD-port
regressions. The two cases are:

- `drama_voice_tool.test.voice tool: cleanup restores private directory permissions after encoder exit`
- `drama_voice_tool.test.voice tool: cleanup failure warns after publication and accompanies encoder failure`

### Minor warning on serve

```
warning: libtfgrammar.so is required for structured output and required tool
calls (HTTP 400 when missing); restore the matching release archive's lib/
directory or run zig build grammar -Dxgrammar=DIR; prepare DIR with
git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar
```

Expected on a from-source build (grammar is produced by a separate release
step). Blocks only structured output / forced tool calls. Deferred.

### Phase 0 close — gate passed

- `native-hip` builds green on HIP 7.16 + ROCm 10.1 + amdclang 24 for gfx1151.
- HIP engine opens the Radeon 8060S, serves Qwen3.5-9B on loopback, drains.
- Host test suite passes except two unrelated root-env Mac voice-tool checks.

Open follow-ups tracked for Phase 1:

- Teach `Device.capsOf` to tell "driver not loadable" from "chip outside the
  table" instead of returning `null` for both.
- Teach `capabilities` to render `rdna3_5` instead of `rdna3` for gfx1150/1/2/3.
- Document the ldconfig step in the HIP build recipe (receipt + a note in
  `docs/receipts/recipes/*` or the top-level HIP build doc).

### Phase 1 kickoff

- 9B at pinned revision cached: `/root/models/qwen3.5-9b-mlx4` (5.6 GiB).
- 27B at pinned revision `70ae7fac63274ff2eac54152031433374cb80f2f`
  downloading in background.
- `tf-qwen35-test check` running on 9B in background to re-produce the
  invariants panel from `docs/receipts/gfx1151-qwen-833bbfcb/9b-model-check.log`.
