#!/usr/bin/env python3
"""Compare TensorFold Qualcomm generation logs with an ExLlamaV3 reference."""

import argparse
import json
from pathlib import Path


def json_objects(path: Path) -> list[dict]:
    objects = []
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    for line in raw.decode(encoding).splitlines():
        line = line.strip()
        if line.startswith("{"):
            objects.append(json.loads(line))
    return objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("tensorfold_log", type=Path)
    args = parser.parse_args()

    reference_objects = json_objects(args.reference)
    if len(reference_objects) != 1:
        raise SystemExit(f"expected one reference object, found {len(reference_objects)}")
    reference = reference_objects[0]
    actual_objects = json_objects(args.tensorfold_log)
    steps = [obj for obj in actual_objects if obj.get("tensorfold") == "qualcomm-exl3-forward-step"]
    summaries = [obj for obj in actual_objects if obj.get("tensorfold") == "qualcomm-exl3-generate"]
    if len(summaries) != 1:
        raise SystemExit(f"expected one TensorFold summary, found {len(summaries)}")
    summary = summaries[0]

    if summary["tokens"] != reference["tokens"]:
        raise SystemExit("generated token sequence mismatch")
    if len(steps) != len(reference["records"]):
        raise SystemExit(f"step count mismatch: {len(steps)} != {len(reference['records'])}")

    top1_mismatches = []
    top5_mismatches = []
    max_logit_delta = 0.0
    for expected, actual in zip(reference["records"], steps):
        expected_ids = [item[0] for item in expected["top5"]]
        actual_ids = [item[0] for item in actual["top5"]]
        if expected_ids[0] != actual_ids[0]:
            top1_mismatches.append(
                {
                    "position": expected["position"],
                    "expected": expected_ids[0],
                    "actual": actual_ids[0],
                }
            )
        if expected_ids != actual_ids:
            top5_mismatches.append(
                {
                    "position": expected["position"],
                    "expected": expected_ids,
                    "actual": actual_ids,
                }
            )
        for expected_item, actual_item in zip(expected["top5"], actual["top5"]):
            if expected_item[0] == actual_item[0]:
                max_logit_delta = max(max_logit_delta, abs(expected_item[1] - actual_item[1]))

    result = {
        "comparison": "tensorfold-qualcomm-vs-exllamav3",
        "backend": summary["backend"],
        "positions": len(steps),
        "generated_tokens": reference["steps"],
        "sequence_match": True,
        "top1_exact_positions": len(steps) - len(top1_mismatches),
        "top1_mismatch_positions": top1_mismatches,
        "top5_exact_positions": len(steps) - len(top5_mismatches),
        "top5_mismatch_positions": top5_mismatches,
        "max_matched_logit_abs_delta": max_logit_delta,
        "passed": not top1_mismatches,
    }
    print(json.dumps(result, separators=(",", ":")))
    if top1_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
