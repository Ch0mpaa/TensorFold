#!/usr/bin/env python3
"""Emit a deterministic multi-token ExLlamaV3 reference for Qualcomm parity."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

if os.name == "nt":
    dll_directories = [Path(torch.__file__).resolve().parent / "lib", Path(sys.base_prefix)]
    if cuda_path := os.environ.get("CUDA_PATH"):
        dll_directories.extend((Path(cuda_path) / "bin", Path(cuda_path) / "bin" / "x64"))
    for dll_directory in dll_directories:
        if dll_directory.is_dir():
            os.add_dll_directory(str(dll_directory))

from exllamav3 import Cache, Config, Model
from exllamav3.modules.quant.fp16 import LinearFP16
from exllamav3.modules.transformer import TransformerBlock


def install_layer_state_snapshots() -> None:
    original_forward = TransformerBlock.forward

    def forward(self, x, params, out_dtype=None):
        before = len(params.get("export_states", ()))
        result = original_forward(self, x, params, out_dtype)
        states = params.get("export_states")
        if states is not None and len(states) > before:
            states[-1] = states[-1].clone()
        return result

    TransformerBlock.forward = forward




def install_torch_fp16_linear() -> None:
    """Avoid the extension's direct cuBLAS path on mismatched Windows toolkits."""

    def forward(self, x, params, out_dtype=None):
        del params
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        dtype = out_dtype or self.out_dtype or torch.half
        weight = self.weight
        if self._pinned_store is not None:
            raise RuntimeError("Torch FP16 oracle fallback does not support pinned vision weights")
        if dtype == torch.float:
            y = torch.matmul(x.float(), weight.float())
        else:
            y = torch.matmul(x, weight).to(dtype)
        if self.bias is not None:
            y += self.bias
        return y.view(out_shape)

    LinearFP16.forward = forward



    from exllamav3.ext import exllamav3_ext

    def hgemm_batched(x, weight, output):
        if output.dtype == torch.float:
            output.copy_(torch.bmm(x.float(), weight.float()))
        else:
            torch.bmm(x, weight, out=output)

    exllamav3_ext.hgemm_batched = hgemm_batched

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("token_id", type=int)
    parser.add_argument("steps", type=int)
    parser.add_argument("--cache-tokens", type=int, default=2048)
    parser.add_argument("--prompt-suffix-token-ids", default="")
    parser.add_argument(
        "--torch-fp16-linear",
        action="store_true",
        help="Use torch.matmul for unquantized FP16 projections",
    )
    parser.add_argument("--hidden-dump")
    parser.add_argument("--list-module-keys", action="store_true")
    parser.add_argument("--layer-dump")
    parser.add_argument("--norm-dump")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")

    if args.torch_fp16_linear:
        install_torch_fp16_linear()
    if args.layer_dump:
        install_layer_state_snapshots()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_tokens)
    model.load(progressbar=True)

    prompt = [args.token_id]
    if args.prompt_suffix_token_ids:
        prompt.extend(int(token) for token in args.prompt_suffix_token_ids.split(","))
    tokens = list(prompt)
    positions = len(prompt) + args.steps - 1
    if args.list_module_keys:
        print(json.dumps([
            [module.key, type(module).__name__, getattr(module, "layer_idx", None)]
            for module in model.modules
        ]))
        return
    if positions > args.cache_tokens:
        parser.error("prompt plus generation exceeds cache-tokens")
    records = []
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, args.cache_tokens),
    }
    exported_states = []
    if args.hidden_dump:
        params["export_state_norm_keys"] = ("model.language_model.norm",)
        params["export_states"] = exported_states
    if args.layer_dump:
        if args.hidden_dump:
            parser.error("--hidden-dump and --layer-dump are mutually exclusive")
        params["export_state_layers"] = tuple(range(40))
        params["export_states"] = exported_states
    if args.norm_dump:
        if args.hidden_dump or args.layer_dump:
            parser.error("state dump options are mutually exclusive")
        params["export_state_norm_keys"] = tuple(
            key
            for layer in range(40)
            for key in (
                f"model.language_model.layers.{layer}.input_layernorm",
                f"model.language_model.layers.{layer}.post_attention_layernorm",
            )
        )
        params["export_states"] = exported_states
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
    captured_states = params.get("export_states", exported_states)
    if args.hidden_dump:
        if len(captured_states) != positions:
            raise RuntimeError(
                f"expected {positions} exported final-norm states, got {len(captured_states)}"
            )
        torch.stack([state.float().cpu() for state in captured_states]).numpy().tofile(
            args.hidden_dump
        )
    if args.layer_dump:
        expected = positions * 40
        if len(captured_states) != expected:
            raise RuntimeError(
                f"expected {expected} exported layer states, got {len(captured_states)}"
            )
        torch.stack([state.float().cpu() for state in captured_states]).numpy().tofile(
            args.layer_dump
        )
    if args.norm_dump:
        expected = positions * 80
        if len(captured_states) != expected:
            raise RuntimeError(
                f"expected {expected} exported norm states, got {len(captured_states)}"
            )
        torch.stack([state.float().cpu() for state in captured_states]).numpy().tofile(
            args.norm_dump
        )
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
