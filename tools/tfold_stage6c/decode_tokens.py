#!/usr/bin/env python3
"""Decode a raw token-id file from `tf-qwen35-cuda prompt` back to text.

The tf-qwen35-cuda tool writes little-endian u32 token IDs in sequence to
the output path. We read them, pass them to the model's tokenizer, and
print both the ids and the decoded string.
"""
import argparse
import json
import pathlib
import struct
import sys

try:
    from tokenizers import Tokenizer
except ImportError:
    print("pip install tokenizers", file=sys.stderr)
    sys.exit(1)


def read_ids(path: pathlib.Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        print(f"warn: output is {len(data)} bytes, not a multiple of 4", file=sys.stderr)
    return [struct.unpack("<I", data[i : i + 4])[0] for i in range(0, len(data) - len(data) % 4, 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_path", type=pathlib.Path)
    ap.add_argument("--tokenizer", type=pathlib.Path, required=True)
    ap.add_argument("--prompt-ids", help="CSV prompt ids to strip from the output")
    ap.add_argument("--json", action="store_true", help="dump JSON receipt")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    ids = read_ids(args.out_path)
    text = tok.decode(ids, skip_special_tokens=False)
    prompt_ids = [int(x) for x in args.prompt_ids.split(",")] if args.prompt_ids else []
    continuation = ids[len(prompt_ids):] if prompt_ids == ids[: len(prompt_ids)] else ids

    if args.json:
        print(json.dumps({
            "output_bytes": args.out_path.stat().st_size,
            "n_ids": len(ids),
            "ids": ids,
            "prompt_ids": prompt_ids,
            "continuation_ids": continuation,
            "text": text,
            "continuation_text": tok.decode(continuation, skip_special_tokens=False),
        }, indent=2))
    else:
        print(f"n_ids = {len(ids)}")
        print(f"ids   = {ids}")
        print(f"text  = {text!r}")
        if prompt_ids:
            print(f"continuation_ids  = {continuation}")
            print(f"continuation_text = {tok.decode(continuation, skip_special_tokens=False)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
