"""Scan Qwen3.5-2B-exl3-8.00bpw/quantization_config.json and summarize
what v0.1 vs v0.2 would need to carry the whole checkpoint.

Writes a terse report to stdout.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parents[2]
for cand in (
    pathlib.Path("D:/op13-root/models/Qwen3.5-2B-exl3-8.00bpw"),
    pathlib.Path("/mnt/d/op13-root/models/Qwen3.5-2B-exl3-8.00bpw"),
):
    if (cand / "quantization_config.json").exists():
        MODEL_DIR = cand
        break
else:
    raise SystemExit("quantization_config.json not found on this host")

q = json.loads((MODEL_DIR / "quantization_config.json").read_text())

print(f"model dir : {MODEL_DIR}")
print(f"quant     : method={q['quant_method']} version={q['version']}")
print(f"bits      : body={q['bits']} head={q['head_bits']}")
print(f"codebook  : {q['codebook']}")
print(f"out_scales: {q['out_scales']}")
print()

ts = q["tensor_storage"]
print(f"total groups: {len(ts)}")

# Classify by suffix pattern
exl3_groups = 0
plain_groups = 0
suffixes = Counter()
sizes = Counter()
for gname, data in ts.items():
    names = list(data["stored_tensors"].keys())
    group_suffixes = {n.rsplit(".", 1)[-1] for n in names}
    if "trellis" in group_suffixes:
        exl3_groups += 1
        has_bias = "bias" in group_suffixes
        suffixes.update(["exl3_group+bias" if has_bias else "exl3_group"])
    else:
        plain_groups += 1
        for s in group_suffixes:
            suffixes.update([f"plain:{s}"])

    for name, info in data["stored_tensors"].items():
        sizes[info["dtype"]] += info["n_bytes"]

print(f"  exl3 groups: {exl3_groups}")
print(f"  plain groups: {plain_groups}")
print()
print("group shape distribution:")
for k, v in suffixes.most_common():
    print(f"  {k:30s}  {v}")
print()
print("bytes per dtype (across all stored tensors):")
total = 0
for dt, nb in sizes.most_common():
    print(f"  {dt:25s}  {nb:>14,}")
    total += nb
print(f"  {'total':25s}  {total:>14,}")
print()

# v0.1 capacity check
print("--- v0.1 representability ---")
print("v0.1 allows one quant.scheme + one bits_per_weight + one trellis.")
print(f"This checkpoint has {exl3_groups} trellis tensors in {len(ts)} groups.")
print("v0.1 can hold one group per file → multi-file approach needs")
print("151 EXL3 .tfold files + 116 plain tensor files = 267 files.")
print("Not practical for serving. Full-checkpoint needs v0.2.")
print()

# v0.2 proposal check
print("--- v0.2 extension proposal ---")
print("With a per-group codebook/scheme/width override, a single .tfold file")
print("could carry all 267 groups because:")
uniform_cb = len({q["codebook"]}) == 1
print(f"  uniform codebook across body: {uniform_cb} ({q['codebook']})")
print(f"  bits uniform in body        : True (8.0)")
print(f"  head bits differ            : True ({q['head_bits']} vs {q['bits']})")
print("Minimum v0.2 addition: per-tensor_name {bits_per_weight, scheme}")
print("override map inside quant{}. Structural layout unchanged; header-only.")
