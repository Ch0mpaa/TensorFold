# Qualcomm packed-direct MoE

## Selected checkpoints

Two revisions serve different purposes; their tokens and measurements must not
be combined. The storage-fit CPU/app paging reference is
`turboderp/Qwen3.5-35B-A3B-exl3@2.54bpw`, revision
`af18e8ad45f4d529925bab859d79e163dc5abe60`. The privileged HTP engineering
candidate is `turboderp/Qwen3.5-35B-A3B-exl3@4.00bpw`, revision
`8a68a1aca688b768853bad4d926c0136ff62286c`; its evidence is recorded below.
Neither revision has completed the release admission gates.

The 2.54-bpw revision is the fit-first reference:

- Apache-2.0 Qwen3.5 model, 35B total parameters and about 3B active per token.
- `qwen3_5_moe`: 40 text layers, hidden size 2,048, 256 routed experts,
  top-8 routing, 512-wide routed and shared experts, three linear-attention
  layers for each full-attention layer.
- EXL3 2.54 bpw, `mcg` codebook, 6-bit head. The two weight shards total
  12,987,659,758 bytes. The checkpoint remains packed on storage.
- Native tool-call chat template and coding/agentic base make it a better
  bounded PwnHarness candidate than a generic chat-only model.

The 2.00/2.08/2.13 bpw revisions are smaller but are not the first storage-fit
target. The 3.00 bpw revision is a fallback if 2.54 bpw fails the quality gate,
not a qualified result. The 4.00-bpw HTP candidate is a separate kernel-fit
choice, not evidence that any revision has passed the quality gate.

## Execution contract

TensorFold must not build a decoded full-model cache. The storage-fit CPU path
keeps embeddings, normalization, routers, attention/DeltaNet projections, the
shared expert and output head in their packed representation. Routed expert
trellises remain in the checkpoint; a bounded cache admits selected expert
pages. The 4.00-bpw HTP candidate separately permits a content-bound decoded
FP16 cache for only its 6-bit output head, not the full model.

For each MoE layer:

1. Compute router logits and stable top-8 selection.
2. Group rows by expert without changing token order.
3. Resolve the selected gate/up/down trellis spans through the expert-page
   cache; one key is checkpoint digest, layer, expert, projection and EXL3
   descriptor.
4. Decode directly from packed EXL3 bytes into the projection operation.
   Never materialize a whole expert matrix.
5. Apply SwiGLU and combine weighted expert outputs in deterministic router
   order, then add the resident shared expert.
6. Release pins after the layer. Eviction is LRU among unpinned pages only.

Cold misses use bounded `pread` into aligned app-owned buffers. Warm hits do no
filesystem lookup or allocation. Cancellation ends pending reads and cannot
publish a partial page. A page is usable only after its source span and
checkpoint digest have been validated.

## Admission gates

A phone is admitted only when all gates pass:

| Gate | Required result |
| --- | --- |
| Architecture | Exact Qwen3.5-MoE geometry above; text-only execution; unknown tensors or layouts refuse before inference. |
| Checkpoint | Public revision, every artifact byte count and SHA-256 pinned; HTTPS-only resumable download; no local path publication. |
| Storage | Packed artifacts plus 2 GiB reserve fit app-private storage; no decoded full-model cache. |
| Memory | Peak app RSS at or below 12 GiB on the 16 GiB SM8750 reference phone; expert cache is explicitly bounded and reported. |
| Exactness | Tokenizer parity; selected-expert IDs and weights match the reference; each EXL3 projection matches reconstruction bounds; frozen greedy token sequence matches ExLlamaV3 for 256 tokens on code, chat and tool prompts. |
| Quality | 2.54 bpw must not regress below the existing 4B baseline on the versioned authorized-pentest, coding and tool-schema fixtures. If it does, use 3.00 bpw. |
| Tools | Tool calls must parse under the existing strict schema; malformed, unknown or out-of-scope calls are refused. |
| Safety | PwnHarness owns scope confirmation, approvals, evidence and STOP. TensorFold exposes no unrestricted Accessibility or shell automation. |
| Thermal | Refuse new work at 43 C; sustained runs pause at the existing Qualcomm high/low bands and resume without changing tokens. |
| Lifecycle | Cancellation, activity stop and force-stop leave no worker, open page pin or partial cache publication. |
| Performance | Report prompt processing, delivery tok/s, TTFT, page-hit rate, storage reads, RSS and temperature. No speed claim until protocol-complete runs pass. |

## Qualification order

1. Parse and inspect both safetensor shards without loading tensor payloads.
2. Qualify packed expert span lookup and deterministic page eviction on host
   fixtures and the real checkpoint index.
3. Qualify one real routed layer against ExLlamaV3.
4. Add the complete text-only Qwen3.5-MoE forward path and frozen-token check.
5. Add the checkpoint to the Android catalog only after fit, exactness,
   quality, lifecycle and thermal gates pass on Phone B.
6. Expose it to PwnHarness only after strict tool-call and STOP tests pass.

The experimental 2.54-bpw Android catalog entry described below is a
qualification path, not proof that these release gates have passed.

## Pager qualification evidence

`zig/src/qualcomm/exl3_expert_pager.zig` now indexes safetensor headers
without mapping payloads and provides a bounded packed-tensor LRU with pinned
leases, cancellation checks between 1 MiB positional reads, hit/miss/eviction
accounting, and atomic publication after a complete read. Permanent tests cover
real safetensor headers, no eviction of pinned entries, deterministic LRU
eviction, hit reuse, and cancellation without partial publication.

The Android inspector parsed the real 2.54 bpw revision's two shard headers on
Phone B and admitted 123,360 routed and shared expert tensors totaling
9,985,278,432 packed bytes. The inspector reads only headers; this result proves
the real checkpoint's expert namespace and spans fit the pager contract, not
model exactness or quality.

## Full-forward and short-sequence evidence

The packed-direct CPU path executes all 40 Qwen3.5-MoE layers from the two
checkpoint shards on Phone B without a decoded full-model cache. Real
checkpoint comparison against ExLlamaV3 exposed a routing defect: the first
selected logit was overwritten before the remaining softmax numerators were
computed. `moe_routing.normalizeTopK` now preserves the original maximum,
rounds the published weights to FP16 like the reference, and has a permanent
regression test.

For the 2.54-bpw revision and tokenizer input `hi` (ID `5834`), TensorFold and
ExLlamaV3 return the same eight generated tokens. The stream below includes
the input ID first:

```text
5834, 11, 353, 1144, 310, 1791, 264, 3779, 364
```

Seven of eight ordered top-five lists match exactly. At the final observed step,
the same five IDs appear with reference-tied IDs `13` and `440` reversed at
ranks four and five. The first step was:

| Rank | ExLlamaV3 token | ExLlamaV3 logit | TensorFold token | TensorFold logit |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11 | 12.6953125 | 11 | 12.693305 |
| 2 | 1017 | 12.3671875 | 1017 | 12.356280 |
| 3 | 585 | 11.7421875 | 585 | 11.748819 |
| 4 | 0 | 11.6796875 | 0 | 11.671944 |
| 5 | 13 | 11.4687500 | 13 | 11.471230 |

Feeding ExLlamaV3's final normalized hidden state directly through TensorFold's
packed `lm_head` reproduced the same ordered top five and near-identical logits.
This isolates the output head from model-state debugging.

Partitioning each packed projection by output tiles across 16 CPU workers keeps
the same tokens and reduced the one-token Phone B elapsed time from 47,557.515
ms to 13,872.030 ms. The packed projection portion fell from 47,212.067 ms to
13,450.176 ms and the output head from 8,215.107 ms to 1,786.323 ms. This is a
3.43x bring-up improvement, not a usable decode speed.

The eight-token result is not the 256-token exactness gate. Full code, chat and
tool fixtures, RSS, thermal/lifecycle qualification, quality evaluation and the
standard community benchmark still must pass before support or speed claims.

The Android app packages this path as `libtensorfold_engine_moe.so`, admits only
the pinned Qwen3.5-35B-A3B geometry, keeps the model in app-private storage, and
starts an authenticated loopback server under the app UID. PwnHarness reads the
active endpoint, one-process bearer token, model ID and ready state only through
an explicit Android runtime permission. It cannot select another checkpoint or
bypass PwnHarness scope, approval, evidence or STOP controls.

## HTP 4.00-bpw engineering evidence

The 2.54-bpw checkpoint remains the 12.99 GB storage-fit and Android paging
reference, but its predominantly 2- and 3-bit trellises do not match the
qualified single-row HTP kernel. A private generic low-bit HTP decoder matched
the first token but remained slower than the 16-worker CPU path. That failed
performance candidate is not the source of the 4.00-bpw figures below.

The separate acceleration candidate is
`turboderp/Qwen3.5-35B-A3B-exl3@4.00bpw`, revision
`8a68a1aca688b768853bad4d926c0136ff62286c`. Its three weight shards total
19,329,446,010 bytes and matched their published SHA-256 values. All 30,970
non-head EXL3 projections use 4-bit trellises; the output head uses 6 bits.
The warm HTP path uses a content-bound decoded FP16 cache for only that head.

CPU and HTP produced the same observed eight generated tokens from `hi`, with
the same ordered top-five IDs at each observed position. This 4.00-bpw stream
includes input ID `5834` first and differs from the 2.54-bpw stream above:

```text
5834, 11, 585, 1318, 310, 1791, 264, 3779, 364
```

| Phone B engineering run | Decode tok/s | End-to-end tok/s | Peak RSS |
| --- | ---: | ---: | ---: |
| 4.00-bpw CPU, 8 tokens | not separately reported here | 0.068943 | 2,040,216 KiB |
| 4.00-bpw initial HTP, 8 tokens, packed head | not separately reported here | 0.343639 | 2,045,704 KiB |
| 4.00-bpw HTP, 64 tokens, warm decoded head | 0.753628 | 0.753648 | 2,675,068 KiB |

The 64-token HTP run moved battery temperature from 31.2 C to 31.7 C. The
retained stream repeats a short pattern after the initial continuation; a run
on the one-token `hi` input does not establish useful chat, coding or tool
quality. The 64-token run also does not extend the eight-token CPU/HTP
comparison into a 64-token reference-parity claim.

These are rooted Phone B userspace/model-runtime engineering measurements,
not full NPU-only execution, ordinary-app FastRPC access, a production release,
or publication-grade foreground-app results. CPU orchestration and non-HTP
model work remain part of this mixed execution lane. The public APK does not
bundle privileged HTP. The release speed gate, 256-token code/chat/tool parity,
quality evaluation and protocol-complete community-v2 benchmark remain unmet.
Do not combine these results with the different-checkpoint 2.54-bpw CPU
evidence or the existing dense 2B/4B benchmarks.

Retained evidence: [4.00-bpw engineering record](../recordings/20261020-qualcomm-35b-a3b-4bpw/README.md)
and its CPU, initial HTP and warm 64-token summary files.
