# Qwen3.5-35B-A3B 4.00-bpw HTP qualification

Checkpoint: `turboderp/Qwen3.5-35B-A3B-exl3@4.00bpw`, revision `8a68a1aca688b768853bad4d926c0136ff62286c`.

The three weight shards total 19,329,446,010 bytes and matched their published SHA-256 values:

- `ae09fd598b8c0bd651904de8ec44cfbd8d7832ab0ae9de66df131cda40959be9`
- `e47ffb46fe8dd9a97b97ce7e6e0a552bda20ac6eb9aae6ac8a19cb3c6549b92b`
- `4202ac8815e0c91c183f8a2948c1abbe2461c01fbff6f8d73f5602857178e45a`

All 30,970 non-head EXL3 projections use 4-bit trellises. The untied output head uses 6 bits.

## Results

The CPU and Qualcomm HTP paths produced the same observed eight-token sequence from tokenizer ID `5834`:

```text
5834, 11, 585, 1318, 310, 1791, 264, 3779, 364
```

Their ordered top-five token IDs matched at every observed position. The initial HTP path produced 0.343639 end-to-end tok/s, 4.98x the CPU path's 0.068943 tok/s, with about 2.0 GiB peak RSS on each path.

A content-bound decoded FP16 cache for only the 6-bit output head reduced warm head time from about 1.6 seconds to 0.11-0.19 seconds. A sustained 64-token HTP run then delivered 0.753628 decode tok/s and 0.753648 end-to-end tok/s at 2,675,068 KiB peak RSS. Battery temperature moved from 31.2 C to 31.7 C.

This is an engineering HTP userspace/model-runtime result on rooted Phone B. It does not establish ordinary app access to FastRPC, 256-token parity, quality, or the release speed gate. It is still far below the pocket-rocket target.

The private Hexagon fork also gained bounded EXL3 trellis admission from 0.5 to 8 bits. Its generic low-bit decoder produced the same first token but was slower than the 16-worker CPU path for the 2.54-bpw checkpoint. The 4-bit checkpoint therefore remains the current HTP optimization candidate.
