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
