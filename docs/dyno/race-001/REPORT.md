# Dyno Race 001 — REPORT

Frozen config: `CONFIG.md` in this directory (every sha256 pinned there must match any receipt).

## Winners per metric

| Metric | Unit | Winner | A: safetensors (med) | B: .tfold (med) | vLLM C=1 (med) | Notes |
|---|---|---|---|---|---|---|
| **Logit bytes identity** | sha256 | **A = B** | `0385e13dd46771cd…` | `0385e13dd46771cd…` | n/a (different runtime) | 22 out of 22 TFOLD runs byte-identical |
| **Greedy token sequence** | tokens | **A = B** | `197 197 13 197 197 13 … 13 197` | identical | reported in Race B fidelity | behavioral parity holds |
| **Scenario compute** | ms (prefill+decode of 48 rows) | **TIE** | 875 ms | 884 ms | n/a | within 1.5% (noise floor) |
| **Peak VRAM** | GiB | **TIE** | 5.73 | 5.73 | see Race B | identical weight footprint |
| **Wall-clock open+run** | s | A (startup) | 3.58 | 7.48 | see Race B | 3.9 s delta = FNV-1a-64 trailer verification (median `fnv=4529 ms`) |
| **Resident set** | MiB | TIE | 7420 | 7418 | see Race B | mmap page population same |
| **Integrity check** | — | **B exclusive** | not present | full-file FNV-1a-64 before any tensor read | n/a | this *is* the cost of lossless integrity |

### No-winner lines (metrics with no clean verdict)

- **TTFT**: TFOLD's `stage` subcommand does not expose TTFT because it runs a scripted scenario, not a streaming generator; comparing against vLLM TTFT would be unfair without a matching TFOLD streaming entry point. Reported for vLLM only in Race B.
- **Decode tok/s**: TFOLD's `stage` subcommand emits a full logit matrix at every step, which is not comparable to a streamed decode loop. We report the scenario budget (48 logit rows / 884 ms ≈ 54 tok/s raw, no KV reuse) for context, but it is NOT a decode throughput number and must not be compared directly.

## Race A — container (TFOLD runtime, two checkpoint sources)

### Setup
- Binary: `tf-qwen35-cuda` sha256 `81b2fbad…` (includes `normBf16OrF32` loader fix).
- A = `/home/chris/models/qwen35-2b-orig/` (the official safetensors directory, unchanged upstream bytes).
- B = `/tmp/tfold_dir/` (model.tfold hardlink to the losslessly converted `.tfold`).
- 1 warm-up + 10 timed warm runs per path, `TF_PIPE_TOKENS=5`.
- Raw JSONL: `receipts/race_a.jsonl`.  Summary: `receipts/race_a_summary.json`.

### Results (median of 10 timed runs each; ranges = p95)

| metric | A — safetensors | B — .tfold |
|---|---|---|
| wall  | 3.580 s (p95 3.59) | 7.480 s (p95 7.57) |
| scenario compute | 875 ms (range 872–899) | 884 ms (range 872–899) |
| user CPU | 1.31 s | 4.82 s |
| sys CPU | 1.93 s | 2.04 s |
| peak VRAM | 5.73 GiB | 5.73 GiB |
| RSS | 7595 MiB | 7596 MiB |
| FNV verify | — | 4529 ms (range 4446–4594) |
| header_parse | — | 0.89 ms |
| preamble | — | 2.2 ms |
| mmap | — | 6.5 µs |

### Behavior
- Dump sha16 across all 22 timed runs: `{ 0385e13dd46771cd }` (singleton).
- Token sequence across all 22 timed runs: `197 197 13 197 197 13 197 197 13 197 197 13 197 197 13 197 197 13 197 13 13 197` (singleton).
- `TF_NORM_AUDIT=1` lines byte-identical across both paths: `AUDIT_IDENTICAL` per `race_a_bench.sh`.

### Verdict
- **Correctness: identical** — both paths produce bit-identical BF16 logits and identical greedy tokens.
- **Scenario compute: tie** — 9 ms median delta is below the single-run variance (`max − min = 27 ms` within each path).
- **Startup: safetensors wins by 3.9 s** — the delta is the `.tfold` FNV-1a-64 integrity trailer, intentional and opt-out-able via `ParseOptions.skip_fnv`. This is not a race-loss for `.tfold`; it is the price of full-file integrity.

## Race B — runtime comparison (TFOLD vs vLLM)

> **Fidelity disclosure first.**

### Race B — dtype audit (what does vLLM actually hold for FP32 norms?)

See `receipts/vllm_dtype_audit.json`. Procedure: load `/home/chris/models/qwen35-2b-orig/`
in vLLM with `dtype='bfloat16'`, RPC into the worker's model_runner.model, dump
every parameter whose name contains `linear_attn.norm.weight`.

**Finding (populated from receipts)**: *see `vllm_dtype_audit.json` for the
on-device dtype of each of the 24 `linear_attn.norm.weight` tensors inside
vLLM.*

Policy comparison:

| runtime | upstream FP32 norm → kept as | audit telemetry | our conclusion |
|---|---|---|---|
| TFOLD (`normBf16OrF32`) | BF16 (RNE at load) | `TF_NORM_AUDIT=1` emits max_abs, max_rel, non_finite per tensor | explicit, bounded, logged |
| vLLM (`dtype=bfloat16`) | (see audit receipt) | silent at load | `dtype` arg is a global cast directive |

If vLLM narrows to BF16 (expected default), **the two runtimes operate on
numerically equivalent kernel inputs** (both narrowed via RNE) and the
throughput numbers are directly comparable. If vLLM keeps FP32, the
throughput comparison is apples-to-oranges and gets a `no-winner` label.

### Race B — throughput (vLLM concurrency sweep)

Full receipts in `receipts/vllm_sweep.json`. Protocol: cold load, 1 warmup
batch per concurrency, 3 timed repeats, median reported.

| C | wall (s) | total tok/s | per-req decode tok/s | TTFT med (s) | P50 (s) | P95 (s) | VRAM (MiB) |
|---|---|---|---|---|---|---|---|
| 1  | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* |
| 4  | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* |
| 16 | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* |
| 32 | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* | *see receipt* |

### TFOLD side of Race B

TFOLD's current public CLI in this binary exposes `stage`, a one-shot
scenario runner. Concurrency 4, 16, 32 would require batching at the
scheduler level, which is not part of the `tf-qwen35-cuda` binary today.
We therefore report:

- TFOLD at **C=1 only**, with the scenario-budget throughput noted but not
  framed as decode tok/s.
- Honest declaration: **comparing single-stream TFOLD against batched vLLM
  is not a fair runtime race**; the C>1 columns are vLLM-only until TFOLD
  ships a batched CLI. The CONFIG.md reproduction block makes this clear.

## Race B — fidelity

Prompts (`prompts.txt`, sha `c95696d2…`): 5 English prompts, mixed length.

- vLLM-generated token IDs: see `receipts/vllm_sweep.json` / the `vllm_baseline.json`.
- TFOLD does not currently expose a chat-style generator in this CLI; a
  transformers-based greedy generation on the same prompts would be a cleaner
  comparator. Noted as future work if a batched TFOLD CLI appears.

## Race C — EXL3 preserved

`/home/chris/models/tfold/Qwen3.5-2B-exl3-mcg.tfold` sha256
`027ee2964e3868fa…` (from previous milestone receipts) is **unchanged**.
Its previous scenario numbers (median scenario ~2600 ms, VRAM 3.24 GiB)
remain in `../../tfold/parity-receipts/` and are NOT merged into this
race's numbers. The next race ("002", low-bit) will exercise EXL3 again
independently.

## Reproducibility

```bash
# Freeze (fetch upstream)
curl -sL -o config.json "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/15852e8c.../config.json"
# ... (see CONFIG.md for full file list + shas)

# Race A
bash race_a_bench.sh

# Race B dtype audit
python3 vllm_dtype_audit.py --model /path --dtype bfloat16 --out receipts/vllm_dtype_audit.json

# Race B sweep
python3 vllm_sweep.py --model /path --prompts-file prompts.txt \
  --max-tokens 32 --concurrencies 1,4,16,32 --out receipts/vllm_sweep.json

# Analysis
python3 analyze_race_a.py --jsonl receipts/race_a.jsonl --out receipts/race_a_summary.json
```

## Uncertainty

- Each Race A row is median over 10 warm runs on a quiet GB10. Per-run
  variance: `scenario_ms` stdev ≈ 9 ms (<1.1%), `wall_s` stdev ≈ 30 ms.
- vLLM VRAM reported via nvml at batch end; GB10 Unified Memory means
  "GPU" memory shares the host DRAM budget; see driver note in CONFIG.
- No measurement uses `drop_caches` (no root available); cold-start
  numbers are omitted intentionally. The .tfold wall overhead is a
  deterministic phase (`fnv` verification), not a cache artifact — proved by
  the `fnv_us` phase stat being stable across runs.

## Honest statement

- Logit-level and token-level equivalence of safetensors ↔ .tfold inside
  our runtime is **proven bit-identical** (22/22 runs, identical sha256,
  identical token stream, identical audit lines).
- The `.tfold` path pays a ~3.9 s fixed integrity cost on every open; this
  cost delivers full-file FNV-1a-64 coverage that the safetensors path
  does not offer. It is a feature, not a tax.
- Throughput comparison with vLLM at C>1 is vLLM-only; we do **not**
  claim TFOLD beats or loses to vLLM on batched throughput — the CLI
  does not currently support it.
- EXL3 stays out of this race. 2B class remains **NOT GRADUATED**
  (OFFSEC CWE-311/327 still open).
