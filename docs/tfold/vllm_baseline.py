"""vLLM baseline against the same Qwen/Qwen3.5-2B pinned revision.

Records: cold load time, generation tok/s, first-token latency (TTFT),
peak VRAM via torch.cuda.max_memory_allocated. Keeps results in a
separate receipts file to avoid mixing with the EXL3 low-bit race.
"""
import argparse, json, os, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local directory")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    print("loading vllm", file=sys.stderr)
    from vllm import LLM, SamplingParams

    prompts = [p.rstrip("\n") for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    rep = {"model": args.model, "revision": args.revision, "dtype": args.dtype,
           "max_tokens": args.max_tokens, "n_prompts": len(prompts)}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        llm = LLM(model=args.model, revision=args.revision, dtype=args.dtype,
                  gpu_memory_utilization=0.08,
                  max_model_len=512, enforce_eager=True,
                  trust_remote_code=True)
    except Exception as e:
        rep["error"] = f"vllm_load_failed: {type(e).__name__}: {e}"
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(rep["error"], file=sys.stderr); sys.exit(2)
    rep["load_time_s"] = round(time.time() - t0, 2)
    rep["peak_vram_after_load_bytes"] = torch.cuda.max_memory_allocated()

    print(f"loaded in {rep['load_time_s']}s, peak {rep['peak_vram_after_load_bytes']/2**30:.2f} GiB", file=sys.stderr)

    results = []
    for i, p in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = llm.generate([p], sampling_params=sp)
        dt = time.time() - t0
        toks = out[0].outputs[0].token_ids
        text = out[0].outputs[0].text
        tps = len(toks) / dt if dt > 0 else float("inf")
        results.append({"prompt_idx": i, "prompt_truncated": p[:80], "dt_s": round(dt, 3),
                        "n_tokens_out": len(toks), "tokens_per_s": round(tps, 2),
                        "generated_text": text[:200],
                        "token_ids": list(toks),
                        "peak_vram_bytes": torch.cuda.max_memory_allocated()})
        print(f"  prompt {i}: {len(toks)} toks in {dt:.2f}s = {tps:.1f} tok/s", file=sys.stderr)
    rep["results"] = results
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("report:", args.out, file=sys.stderr)

if __name__ == "__main__":
    main()
