#!/usr/bin/env python3
"""Emit a deterministic multi-token ExLlamaV3 reference for Qualcomm parity."""

import argparse
import json

import torch
from exllamav3 import Cache, Config, Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("token_id", type=int)
    parser.add_argument("steps", type=int)
    parser.add_argument("--cache-tokens", type=int, default=2048)
    parser.add_argument("--prompt-suffix-token-ids", default="")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_tokens)
    model.load(progressbar=True)

    prompt = [args.token_id]
    if args.prompt_suffix_token_ids:
        prompt.extend(int(token) for token in args.prompt_suffix_token_ids.split(","))
    tokens = list(prompt)
    positions = len(prompt) + args.steps - 1
    if positions > args.cache_tokens:
        parser.error("prompt plus generation exceeds cache-tokens")
    records = []
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, args.cache_tokens),
    }
    with torch.inference_mode():
        for position in range(positions):
            token = tokens[position]
            input_ids = torch.tensor([[token]], dtype=torch.long)
            params["past_len"] = position
            logits = model.forward(input_ids=input_ids, params=params).float().view(-1)
            top_logits, top_ids = torch.topk(logits, k=5)
            next_token = top_ids[0].item()
            forced_next = position + 1 < len(prompt)
            if not forced_next:
                tokens.append(next_token)
            records.append(
                {
                    "position": position,
                    "input_token": tokens[position],
                    "forced_next": forced_next,
                    "next_token": next_token,
                    "top5": [
                        [candidate.item(), score.item()]
                        for candidate, score in zip(top_ids.cpu(), top_logits.cpu())
                    ],
                }
            )
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "reference": "exllamav3-greedy",
                "head": "checkpoint-exl3-lm_head",
                "steps": args.steps,
                "tokens": tokens,
                "records": records,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
