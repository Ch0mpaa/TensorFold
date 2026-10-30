#!/usr/bin/env python3
"""
A/B benchmark harness for TensorFold HIP (AMD) vs TensorFold CUDA (NVIDIA DGX Spark).

Measures per-request: TTFT (ms), decode tok/s, total latency.
Aggregates: p50/p90/p99 for each metric across a prompt set.

Usage:
  python3 bench.py --base http://192.168.60.228:8888/v1 --model qwen-flash-uncensored --tag nvidia-dgx
  python3 bench.py --base http://192.168.60.228:8081/v1 --model amd-27b-nvfp4         --tag amd-gfx1151

Writes JSON to <tag>.json with per-request + aggregate stats.
"""
import argparse, json, time, sys, statistics
import urllib.request, urllib.error

PROMPTS = [
    # short prose
    ("prose-short", "Write a two-sentence summary of what makes a transformer model different from an RNN.", 64),
    # medium prose
    ("prose-med", "Explain how KV cache paging works in a long-context LLM server, for a reader familiar with attention. Three paragraphs.", 256),
    # code gen
    ("code-short", "Write a Python function `longest_run(xs: list[int]) -> int` that returns the length of the longest run of equal elements. Include one example test.", 128),
    ("code-med", "Implement a Zig function that reads a CSV file and returns the mean of a named column, using std.io buffered reader. Handle missing values. 20-40 lines.", 384),
    # reasoning
    ("reason", "A train leaves A at 60 km/h. Another leaves B at 90 km/h toward A. A and B are 300 km apart. When do they meet? Show the arithmetic.", 128),
    # long prefill, short decode
    ("prefill-long", "You are a careful technical reviewer. Below is a long context. " + ("Lorem ipsum dolor sit amet. " * 400) + " Based on the above, write exactly one sentence summarizing what was asked.", 32),
]

def request(base, model, prompt, max_tokens, stream=True):
    body = {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 7,
        "messages": [{"role": "user", "content": prompt}],
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    t_first = None
    out_text = []
    prompt_tokens = None
    completion_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if stream:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        prompt_tokens = chunk["usage"].get("prompt_tokens")
                        completion_tokens = chunk["usage"].get("completion_tokens")
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content and t_first is None:
                        t_first = time.monotonic()
                    out_text.append(content)
            else:
                payload = json.loads(resp.read())
                t_first = time.monotonic()
                prompt_tokens = payload.get("usage", {}).get("prompt_tokens")
                completion_tokens = payload.get("usage", {}).get("completion_tokens")
                out_text.append(payload["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read()[:200].decode(errors="replace")}
    except Exception as e:
        return {"error": str(e)}
    t_end = time.monotonic()
    text = "".join(out_text)
    return {
        "ttft_ms": (t_first - t0) * 1000 if t_first else None,
        "total_ms": (t_end - t0) * 1000,
        "decode_ms": (t_end - t_first) * 1000 if t_first else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tok_s": (completion_tokens * 1000 / ((t_end - t_first) * 1000)) if (t_first and completion_tokens) else None,
        "chars": len(text),
        "sample_head": text[:80],
    }

def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="OpenAI-compatible base URL (ends in /v1)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="output stem, e.g. nvidia-dgx-27b-nvfp4")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3, help="per-prompt repeats")
    args = ap.parse_args()

    # warm the server
    for _ in range(args.warmup):
        request(args.base, args.model, "Say ok.", 4)

    results = []
    for name, prompt, max_tok in PROMPTS:
        for rep in range(args.repeats):
            print(f"  {name}#{rep} max={max_tok}...", flush=True)
            r = request(args.base, args.model, prompt, max_tok)
            r.update({"name": name, "rep": rep, "max_tokens_asked": max_tok})
            results.append(r)
            if "error" in r:
                print(f"    ERROR: {r['error']}", flush=True)
            else:
                print(f"    ttft={r['ttft_ms']:.0f}ms decode={r['decode_tok_s']:.1f} tok/s "
                      f"({r['completion_tokens']}/{r['prompt_tokens']} tokens, head: {r['sample_head'][:60]!r})",
                      flush=True)

    def summarise(name):
        subset = [r for r in results if r.get("name") == name and "error" not in r]
        if not subset:
            return None
        ttfts = [r["ttft_ms"] for r in subset]
        decodes = [r["decode_tok_s"] for r in subset]
        prompts = [r["prompt_tokens"] for r in subset]
        completions = [r["completion_tokens"] for r in subset]
        return {
            "n": len(subset),
            "prompt_tokens": statistics.median(prompts) if prompts else None,
            "completion_tokens": statistics.median(completions) if completions else None,
            "ttft_ms_p50": pct(ttfts, 0.5),
            "ttft_ms_p90": pct(ttfts, 0.9),
            "decode_tok_s_p50": pct(decodes, 0.5),
            "decode_tok_s_p90": pct(decodes, 0.9),
        }
    summary = {name: summarise(name) for name, _, _ in PROMPTS}
    out = {"base": args.base, "model": args.model, "tag": args.tag, "summary": summary, "runs": results}
    path = args.tag + ".json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
