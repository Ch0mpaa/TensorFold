#!/usr/bin/env python3
"""Emit an upstream ExLlamaV3 position-zero reference for Qualcomm parity."""

import argparse
import json

import torch
from exllamav3 import Cache, Config, Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("token_id", type=int)
    parser.add_argument("--cache-tokens", type=int, default=2048)
    args = parser.parse_args()

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_tokens)
    model.load(progressbar=True)

    input_ids = torch.tensor([[args.token_id]], dtype=torch.long)
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, args.cache_tokens),
    }
    with torch.inference_mode():
        logits = model.forward(input_ids=input_ids, params=params).float().view(-1)
        top_logits, top_ids = torch.topk(logits, k=5)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "reference": "exllamav3",
                "input_token": args.token_id,
                "position": 0,
                "head": "checkpoint-exl3-lm_head",
                "next_token": top_ids[0].item(),
                "top5": [
                    [token.item(), score.item()]
                    for token, score in zip(top_ids.cpu(), top_logits.cpu())
                ],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
