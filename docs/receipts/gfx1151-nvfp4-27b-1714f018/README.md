# gfx1151 NVFP4 Qwen3.8-27B port progress

Evidence for the AMD HIP port of TensorFold's NVFP4 format path, targeting
`nvidia/Qwen3.8-27B-NVFP4` (ModelOpt mixed precision: NVFP4 experts + FP8
attention) on a single AMD Ryzen AI MAX+ 395 (Radeon 8060S iGPU, `gfx1151`).
Pinned source: `Ch0mpaa/TensorFold@1714f018` on branch `amd-port-gfx1151`.

**Status: incomplete.** The engine builds, capabilities advertises
`modelopt-nvfp4` + `modelopt-fp8`, format detection and the per-layer
loader reach `Engine.load`. Weight upload then hits `HipFailed`
(unresolved as of this receipt). This receipt documents what has shipped
end-to-end and the exact remaining blocker.

## Hardware

- Container: `dk10-hathaway27.degima.ai` (SSH -p 10104)
- CPU: AMD Ryzen AI MAX+ 395, 16c/32t
- GPU: AMD Radeon 8060S iGPU, `gfx1151` (RDNA 3.5), 40 CUs
- Memory ceiling: 64 GiB in this container (host chassis is 128 GiB)
- ROCm 10.1 / HIP 7.16 / amdclang 24.0.0git
- Zig 0.17.0

## Build command (reproducible)

```sh
zig build -Dhip=true native-hip \
  -Dhipcc=/opt/rocm/bin/hipcc \
  -Dhip-include=/opt/rocm/include \
  -Dhip-arch=gfx1151 \
  -Dhip-device-lib=/opt/rocm/core-10.1/lib/llvm/amdgcn/bitcode \
  -Doptimize=ReleaseFast -j4
```

Build is green at every commit on `amd-port-gfx1151`. The binary is
`zig-out/native-hip/bin/tensorfold-native`.

## Phase progression (serve attempt against `nvidia/Qwen3.8-27B-NVFP4`)

| Commit | Serve result |
| --- | --- |
| `main` (`abf82258`) | `the native HIP engine cannot load … (UnsupportedQuantization)` — core/quant does not recognize NVFP4 |
| Phase C `4227f789` | `UnsupportedQuantization` — core recognizes NVFP4 but HIP dispatch not wired |
| Phase D `6806ba08` | Loader reaches layer 0; `UnexpectedTensor: linear_attn.in_proj_qkv — no weight format reads this tensor` (FP8 attention projection) |
| Phase E (FP8 add) | Loader reaches layer 0; `UnexpectedTensor: linear_attn.conv1d.weight — still in the unsanitized (channels, 1, kernel) layout` |
| Phase E (conv sanitize) | Scratch sizer runs; `the native HIP engine cannot count its scratch (UnsupportedQuantization)` — non-routed NVFP4 (lm_head) dispatch missing |
| Phase E (non-routed NVFP4) `1714f018` | Scratch sizer passes; `the native HIP engine cannot load … (HipFailed)` — weight upload fails inside `Engine.load` |

The failure moved forward through the engine with each fix.

## Capabilities on HIP after Phase D+E

```
$ ./zig-out/native-hip/bin/tensorfold-native capabilities --json | jq .families.qwen3_5
[ "mlx-q2", "mlx-q3", "mlx-q4", "mlx-q5", "mlx-q6", "mlx-q8", "modelopt-nvfp4", "modelopt-fp8" ]
| Phase E (non-routed NVFP4) `1714f018` | Scratch sizer passes; `the native HIP engine cannot allocate its scratch and pages (HipFailed)` — see below for the actual story |

Same list on `qwen3_5_moe` and `qwen4_exp`.

## MLX regression (no regression on existing formats)

Phase C/D/E must not regress the pre-existing MLX path on this box. Full
model-check on both qwen3_5 MLX 4-bit checkpoints after `1714f018`:

- `TensorFold/Qwen3.5-9B-MLX-4bit@b455506b0f574c74616dbcd56879bde38fafcff3`: TODO (running)
- `TensorFold/Qwen3.8-27B-MLX-4bit@70ae7fac63274ff2eac54152031433374cb80f2f`: TODO (running)

The pre-port baseline (recorded in `docs/receipts/amd-port-notes/log.md`):

```
## MLX regression (no regression on existing formats)

Phase C/D/E must not regress the pre-existing MLX path on this box. After a
**clean build** on `1714f018` (`rm -rf .zig-cache zig-out && zig build ...`):

```
./zig-out/bin/tf-qwen35-test check /root/models/qwen3.5-9b-mlx4 --only invariants
check PASS: 42 passed, 0 failed, 5 skipped in 146.2 s
```

Matches the pre-port baseline exactly (recorded in
`docs/receipts/amd-port-notes/log.md`: `42 passed, 0 failed, 5 skipped in 145.3 s`).
No regression on MLX.

> **Watch out for stale `.zig-cache` after a kernel edit.** The first MLX run on
> `1714f018` without a clean rebuild also surfaced `HipFailed`; wiping the cache
> cleared it. The ops.hsaco cache does not always invalidate on
> `#include`-chain changes. When in doubt, `rm -rf .zig-cache zig-out`.

The 5 skips in each are not-exercised controls (no draft head, radix-mem
needs ≥1548 IDs).

## Remaining blocker: HipFailed during Engine.load

```
[tensorfold] HIP rank 0 of 1: policy matrix=auto,attention=auto,kernels=auto,graphs=auto,prefill_step=1024,fill_rows=512; mtp drafts 3 under 0.3, 8 snapshots
tensorfold: the native HIP engine cannot load /root/models/qwen3.8-27b-nvfp4 (HipFailed)
```

Trace: `openSized → load(gpa, io, dir, o)` returns an uncaught HipFailed.
Build cache has the new `.hsaco` for ops (nvfp4 + fp8 kernels linked in).
`info` reads the config fine. The failure is during actual weight upload
(hipMemcpy or hipMalloc returning non-zero). Needs an instrumented debug
pass with HIP error-string tracing before the first "cannot load" catch,
plus a targeted test that loads one FP8 or NVFP4 tensor in isolation to
pin the failing path.

## Files landed on `amd-port-gfx1151`

```
7228c2e9 amd-port: initial env + plan notes for gfx1151
a09fb7b4 amd-port: log hipcc + libstdc++-14-dev build fixes
## Remaining blocker: Engine.load hangs or returns HipFailed (clean build)

On a clean build, `serve /root/models/qwen3.8-27b-nvfp4 --backend rocm
--context 256 --parallel 1 --prompt-cache-gib 0 --no-thinking` prints the
policy line and then stalls past 300 s with no further output. The
pre-existing Qwen3.8-27B MLX serve on the same box took ~38 s to reach
`[tensorfold] serving ... loaded`. The hang is somewhere inside
`Engine.load`'s weight upload path. Possible causes still to pin:

- A bug in our new `nvfp4.upload()` / `fp8_block.upload()` producing
  bad device offsets, triggering an in-flight copy that never completes.
- The ModelOpt key aliasing in `qwen3_5/weights/table.zig` (`language_model.*`
  ↔ `model.language_model.*`) walking every alias on every lookup, blowing
  up to O(aliases × tensors) load time.
- An `hipMemcpy` returning non-zero on one specific tensor shape (e.g.
  input_scale scalar) causing the engine's uploader to retry or spin.

Next debug step (future turn): add a `std.log.info` at every tensor upload
site in `families/qwen3_5/weights/` to print `(key, bytes, dtype)` so we
can see where the hang starts.

```

Companion recipe repo:
<https://github.com/Ch0mpaa/Qwen3.8-NVFP4-Single-Ryzen-AI-Max-TensorFold>

## Reproduction

```sh
# AMD box
ssh -i ~/.ssh/id_itx -p 10104 root@dk10-hathaway27.degima.ai
cd ~/TensorFold && git fetch origin && git checkout 1714f018
zig build -Dhip=true native-hip \
  -Dhipcc=/opt/rocm/bin/hipcc \
  -Dhip-include=/opt/rocm/include \
  -Dhip-arch=gfx1151 \
  -Dhip-device-lib=/opt/rocm/core-10.1/lib/llvm/amdgcn/bitcode \
  -Doptimize=ReleaseFast -j4
./zig-out/native-hip/bin/tensorfold-native capabilities --json
./zig-out/native-hip/bin/tensorfold-native info /root/models/qwen3.8-27b-nvfp4
./zig-out/native-hip/bin/tensorfold-native serve /root/models/qwen3.8-27b-nvfp4 \
  --backend rocm --host 127.0.0.1 --port 8094 \
  --context 1024 --parallel 1 --no-thinking
```

The serve exits with `HipFailed` after about 2.7 s on this box, as pinned above.
