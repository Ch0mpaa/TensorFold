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
