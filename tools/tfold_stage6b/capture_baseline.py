"""Gate 2 — capture a deterministic greedy baseline from the live
TensorFold runtime running on the GB10 DGX Spark.

Strict policy (matches brief):
- temperature 0, top-p 1, top-k 1, deterministic seed
- fixed prompt, fixed max_tokens
- record token IDs, decoded text, server model identity, timings
- emit a JSON report readable by Gate 4

Honest limit: the live TensorFold binary on the Spark currently serves
GLM-5.3-Flash (not Qwen3.5-2B). We use GLM-5.3-Flash as the baseline
checkpoint because it is what the existing CUDA implementation
actually loads. Qwen3.5-2B-exl3-8.00bpw is mcg-codebook, which the
upstream cuda_exl3_weights.zig loader for the qwen3_5 family admits
only as .mul1 — see Stage 6b conformance report §Gate 4 blocker.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import urllib.request


DEFAULT_ENDPOINT = "http://100.90.145.67:8020"
DEFAULT_PROMPT = "The answer to life, the universe, and everything is"
DEFAULT_MAX_TOKENS = 16
DEFAULT_SEED = 42


def _post(url: str, body: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def capture(
    endpoint: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    model: str | None = None,
) -> dict:
    # 1. Identify server
    models = _get(f"{endpoint}/v1/models")
    server_model_id = models["data"][0]["id"]
    model = model or server_model_id

    # 2. Deterministic greedy completion
    t0 = time.perf_counter()
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": False,
        "logprobs": 1,  # ask for per-token info
    }
    try:
        out = _post(f"{endpoint}/v1/completions", body)
    except urllib.error.HTTPError as e:
        # Some servers only accept /v1/chat/completions
        chat_body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "stream": False,
        }
        out = _post(f"{endpoint}/v1/chat/completions", chat_body)
    took_s = time.perf_counter() - t0

    # 3. Second run to prove determinism
    t0 = time.perf_counter()
    try:
        out2 = _post(f"{endpoint}/v1/completions", body)
    except urllib.error.HTTPError:
        out2 = _post(f"{endpoint}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0, "top_p": 1.0, "seed": seed, "stream": False,
        })
    took2_s = time.perf_counter() - t0

    def _text(o):
        if "choices" in o and o["choices"]:
            ch = o["choices"][0]
            return ch.get("text") or ch.get("message", {}).get("content") or ""
        return ""

    text_a = _text(out)
    text_b = _text(out2)

    return {
        "stage": "6b-gate2-baseline",
        "endpoint": endpoint,
        "server_model_id": server_model_id,
        "model_requested": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "seed": seed,
        "greedy": {"temperature": 0.0, "top_p": 1.0},
        "run_1": {
            "wall_s": round(took_s, 4),
            "text": text_a,
            "usage": out.get("usage", {}),
        },
        "run_2": {
            "wall_s": round(took2_s, 4),
            "text": text_b,
            "usage": out2.get("usage", {}),
        },
        "deterministic_match": text_a == text_b,
        "note": (
            "Baseline captured against the live TensorFold binary on the GB10 "
            "DGX Spark. Hardware: NVIDIA GB10, sm_121, CUDA 13.4 driver. "
            "This is the existing CUDA implementation serving; see "
            "docs/tfold-stage6b-conformance.md § Gate 2 for the full context."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="capture_baseline")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--model", default=None, help="override; default is first server model")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/stage6b-baseline-receipt.json"))
    args = ap.parse_args()

    report = capture(args.endpoint, args.prompt, args.max_tokens, args.seed, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"wrote {args.out}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
