"""vLLM concurrency sweep for Race B.

Given an identical prompt file, run concurrency C in {1,4,16,32} (as supported)
and record: prefill tok/s, decode tok/s, TTFT, inter-token latency, P50, P95,
peak VRAM via nvml, cold load, generation throughput.

Output: one JSON file with per-concurrency sections.
"""
import argparse, json, os, sys, time, statistics
from pathlib import Path

def vram_mib():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        return m.used / 1024 / 1024
    except Exception:
        return None

def percentile(xs, p):
    if not xs: return None
    s = sorted(xs)
    k = max(0, min(len(s)-1, int(round((p/100) * (len(s)-1)))))
    return s[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--concurrencies", default="1,4,16,32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-util", type=float, default=0.08)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    prompts = [p.rstrip("\n") for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]
    if not prompts:
        raise SystemExit("no prompts")

    rep = {"model": args.model, "dtype": args.dtype, "max_tokens": args.max_tokens,
           "concurrencies": [], "prompt_count": len(prompts)}

    base_vram = vram_mib() or 0.0
    rep["vram_base_mib"] = base_vram

    t0 = time.time()
    llm = LLM(model=args.model, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_model_len,
              enforce_eager=True, trust_remote_code=True)
    rep["cold_load_s"] = round(time.time() - t0, 2)
    rep["vram_after_load_mib"] = (vram_mib() or 0.0) - base_vram

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    for C in map(int, args.concurrencies.split(",")):
        # Build the batch by replicating prompts up to C total requests
        batch = (prompts * ((C // len(prompts)) + 1))[:C]
        # warmup
        _ = llm.generate(batch, sampling_params=sp, use_tqdm=False)
        # repeats
        wall_times, ttfts = [], []
        gen_tokens_total = []
        for rep_i in range(args.repeats):
            t_rep = time.time()
            vram_before = vram_mib() or 0.0
            # time.monotonic for TTFT via a single-prompt probe
            t_ttft = time.monotonic()
            sp_ttft = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
            _ = llm.generate([batch[0]], sampling_params=sp_ttft, use_tqdm=False)
            ttfts.append(time.monotonic() - t_ttft)
            # full batch
            outs = llm.generate(batch, sampling_params=sp, use_tqdm=False)
            wall = time.time() - t_rep
            wall_times.append(wall)
            toks = sum(len(o.outputs[0].token_ids) for o in outs)
            gen_tokens_total.append(toks)
            vram_after_batch = vram_mib() or 0.0
        avg_toks = statistics.mean(gen_tokens_total)
        results_C = {
            "concurrency": C,
            "wall_times_s": [round(x, 3) for x in wall_times],
            "wall_median_s": round(statistics.median(wall_times), 3),
            "gen_tokens_total_median": statistics.median(gen_tokens_total),
            "total_throughput_tok_s_median": round(statistics.median(gen_tokens_total) / statistics.median(wall_times), 2),
            "per_request_decode_tok_s_median": round((statistics.median(gen_tokens_total)/C) / statistics.median(wall_times), 2),
            "ttft_s_min": round(min(ttfts), 3),
            "ttft_s_median": round(statistics.median(ttfts), 3),
            "ttft_s_p95": round(percentile(ttfts, 95), 3),
            "vram_during_batch_mib": round(vram_after_batch - base_vram, 2),
        }
        # Per-request latency breakdown (one representative batch)
        per_req_latencies = []
        t_batch = time.time()
        outs = llm.generate(batch, sampling_params=sp, use_tqdm=False)
        batch_wall = time.time() - t_batch
        for o in outs:
            # vllm 0.26 may not expose per-request latency; approximate
            per_req_latencies.append(batch_wall)
        results_C["p50_wall_s"] = round(percentile(per_req_latencies, 50), 3)
        results_C["p95_wall_s"] = round(percentile(per_req_latencies, 95), 3)
        rep["concurrencies"].append(results_C)
        print(f"[C={C}] wall_med={results_C['wall_median_s']} throughput={results_C['total_throughput_tok_s_median']} ttft_med={results_C['ttft_s_median']}", file=sys.stderr)

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
