# Qwen3.5-35B-A3B Qualcomm bring-up

Device: Phone B, OnePlus CPH2655, Snapdragon 8 Elite (`SM8750`).

Checkpoint: `turboderp/Qwen3.5-35B-A3B-exl3@2.54bpw`, revision `af18e8ad45f4d529925bab859d79e163dc5abe60`, identity `1539df4e836d11bceba0e54a57696eedc30f1b5df040b20fc3241a0fed1c7199`.

## Correctness

`exllamav3-hi-8.json` and `tensorfold-hi-8.jsonl` contain the independently observed reference and Phone B results for tokenizer input `hi` (`5834`). Both produce:

```text
5834, 11, 353, 1144, 310, 1791, 264, 3779, 364
```

All eight observed ordered top-five lists match. This is short-sequence evidence, not the required 256-token code, chat and tool gate.

The defect was a mutable-maximum softmax loop in routed-expert normalization. The first logit was changed to `1` before the remaining numerators used it. `zig/src/qualcomm/moe_routing.zig` fixes the invariant and retains a regression test.

## Performance

The eight-token baseline used the single-threaded packed projection loop and measured 387,261.877 ms end to end, 0.020658 generated tok/s and 2,064,564 KiB peak RSS.

`tensorfold-hi-1-parallel16.jsonl` partitions each packed projection by output tiles across 16 workers. It retains the same token and ordered top five. One-token elapsed time fell from 47,557.515 ms to 13,872.030 ms, a 3.43x improvement. Packed projection time was 13,450.176 ms, output-head time 1,786.323 ms and peak RSS 1,893,372 KiB.

This remains far below the release performance gate. No general speed or support claim is justified.

A battery snapshot after the recorded development runs reported 31.3 C, level 81 and status 2. It is not a bounded before/after thermal qualification.

Private qualification APK SHA-256: `bfb4a34e11047e6c525e4848a67421451412cabf63a271504a8fc2e8820799b2`.
