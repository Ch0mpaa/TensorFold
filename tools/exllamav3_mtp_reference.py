#!/usr/bin/env python3
"""Measure Qwen3.5 native-MTP acceptance against an EXL3 target.

This is a development oracle for the Qualcomm port.  It runs greedy decoding
with and without the checkpoint's MTP component, verifies that emitted token
IDs are unchanged, and reports draft acceptance plus wall-clock throughput.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# torch's JIT extension loader searches PATH for ninja even when Python is
# launched directly from an unactivated virtual environment.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

import torch

from exllamav3 import ArgmaxSampler, Cache, Config, Generator, Job, Model, Tokenizer


def parse_ids(encoded: str) -> list[int]:
    ids = [int(value) for value in encoded.split(",") if value.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("at least one token ID is required")
    return ids


@torch.inference_mode()
def generate(
    model: Model,
    cache: Cache,
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    new_tokens: int,
    draft_model: Model | None,
    draft_cache: Cache | None,
    depth: int,
) -> dict:
    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        draft_model=draft_model,
        draft_cache=draft_cache,
        num_draft_tokens=depth,
        max_batch_size=1,
    )
    job = Job(
        input_ids=input_ids,
        max_new_tokens=new_tokens,
        sampler=ArgmaxSampler(),
        stop_conditions=[],
    )
    generator.enqueue(job)
    output: list[int] = []
    final: dict = {}
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                token_ids = result.get("token_ids")
                if token_ids is not None:
                    output.extend(token_ids.view(-1).tolist())
                if result.get("eos"):
                    final = result
    finally:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generator.close()

    accepted = int(final.get("accepted_draft_tokens", 0))
    rejected = int(final.get("rejected_draft_tokens", 0))
    offered = accepted + rejected
    return {
        "depth": depth,
        "tokens": output,
        "generated_tokens": len(output),
        "elapsed_ms": elapsed * 1000.0,
        "tokens_per_second": len(output) / elapsed,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "draft_acceptance": accepted / offered if offered else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="EXL3 directory containing the mtp.safetensors sidecar")
    parser.add_argument("--prompt-token-ids", required=True, type=parse_ids)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--depths", type=parse_ids, default=parse_ids("1,2,4"))
    parser.add_argument("--cache-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.new_tokens < 1 or any(depth < 1 for depth in args.depths):
        parser.error("new-tokens and every depth must be positive")

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    mtp_model = Model.from_config(config, component="mtp")
    history = max(args.depths)
    cache = Cache(
        model,
        max_num_tokens=args.cache_tokens,
        max_batch_size=1,
        max_history=history,
    )
    mtp_cache = Cache(
        mtp_model,
        max_num_tokens=args.cache_tokens,
        max_batch_size=1,
        max_history=history,
    )
    mtp_model.load(device=args.device, progressbar=True, max_batch_size=1)
    model.load(device=args.device, progressbar=True, max_batch_size=1)
    tokenizer = Tokenizer.from_config(config)
    input_ids = torch.tensor([args.prompt_token_ids], dtype=torch.long)

    try:
        plain = generate(model, cache, tokenizer, input_ids, args.new_tokens, None, None, 0)
        rows = [plain]
        for depth in args.depths:
            drafted = generate(
                model,
                cache,
                tokenizer,
                input_ids,
                args.new_tokens,
                mtp_model,
                mtp_cache,
                depth,
            )
            drafted["matches_plain"] = drafted["tokens"] == plain["tokens"]
            rows.append(drafted)
        plain_repeat = generate(model, cache, tokenizer, input_ids, args.new_tokens, None, None, 0)
        plain_repeat["run"] = "plain-repeat-after-draft-shapes"
        plain_repeat["matches_plain"] = plain_repeat["tokens"] == plain["tokens"]
        rows.append(plain_repeat)
        print(
            json.dumps(
                {
                    "reference": "exllamav3-qwen35-native-mtp",
                    "model": args.model,
                    "prompt_token_ids": args.prompt_token_ids,
                    "new_tokens": args.new_tokens,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(torch.device(args.device)),
                    "runs": rows,
                    "passed": all(row.get("matches_plain", True) for row in rows),
                },
                separators=(",", ":"),
            )
        )
    finally:
        mtp_model.unload()
        model.unload()


if __name__ == "__main__":
    main()
