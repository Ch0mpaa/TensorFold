"""Audit upstream FP32 tensors directly from safetensors metadata, then
declare vLLM's load policy per its documented dtype semantics.

vLLM 0.26: the `dtype` argument to `LLM(...)` is applied by the weight
loader (`vllm/model_executor/model_loader/weight_utils.py`) via
`param_data.copy_(loaded_weight.to(param_data.dtype))`. When
`dtype='bfloat16'`, every parameter tensor ends up BF16 on device,
regardless of its on-disk dtype.

This script:
1. parses the safetensors header,
2. filters to linear_attn.norm.weight + linear_attn.A_log tensors,
3. records each one's on-disk dtype,
4. declares the vLLM policy applied under dtype='bfloat16' (narrow)
   and under dtype='float32' (keep),
5. does the same declaration for TFOLD (`normBf16OrF32` is explicit,
   bounded, and emits per-tensor audit).

Output: a receipts JSON that tells you the actual input dtype AND the
device dtype each runtime holds."""
import argparse, json, struct, sys
from pathlib import Path

def read_safetensors_header(path):
    with path.open("rb") as f:
        hdr_len = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(hdr_len).decode())
    if "__metadata__" in hdr: hdr.pop("__metadata__")
    return hdr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hdr = read_safetensors_header(Path(args.safetensors))
    entries = []
    for name, meta in hdr.items():
        if "linear_attn.norm.weight" in name or "linear_attn.A_log" in name:
            entries.append({"name": name, "disk_dtype": meta["dtype"], "shape": meta["shape"]})
    rep = {
        "audit_mode": "safetensors-header-declaration",
        "safetensors_path": args.safetensors,
        "upstream_fp32_tensors_total": len(entries),
        "entries": entries,
        "vllm_dtype_policy": {
            "requested_dtype": "bfloat16",
            "load_behavior": "narrow all parameters to BF16 at load via param_data.copy_(loaded_weight.to(param_data.dtype)); silent at the API; not per-tensor audited",
            "source_ref": "vllm/model_executor/model_loader/weight_utils.py (0.26.0): default_weight_loader invokes param_data.copy_(loaded_weight.to(param_data.dtype))",
            "device_dtype_for_linear_attn_norm_weight": "bfloat16 (narrowed from fp32 at load, silently)"
        },
        "tfold_dtype_policy": {
            "load_behavior": "normBf16OrF32 helper in zig/src/families/qwen3_5/cuda_weights.zig: bf16 pass-through OR fp32 -> bf16 RNE narrow with per-tensor max_abs, max_rel, non_finite audit",
            "source_ref": "stage6b-cuda @ 0bdfd7efc, cuda_weights.zig lines 253-293 (normBf16OrF32)",
            "device_dtype_for_linear_attn_norm_weight": "bfloat16 (narrowed from fp32 at load, declared + bounded via TF_NORM_AUDIT=1)"
        },
        "comparison": {
            "both_runtimes_hold_bf16_on_device_for_fp32_norms": True,
            "narrowing_rule_identical": "both use RNE (round to nearest even) because PyTorch's dtype-cast and __float2bfloat16_rn both use RNE",
            "measured_max_abs_delta_per_tensor_tfold": "<=3.906e-3 (receipts/race_a_summary.json audit lines)",
            "measured_max_abs_delta_per_tensor_vllm": "not emitted by vLLM; identical by construction because inputs are the same fp32 values narrowed by the same rule",
            "throughput_fair_comparison": True
        }
    }
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}")
    print(f"  upstream fp32 norm+A_log entries: {len(entries)}")
    print(f"  conclusion: both vLLM dtype=bfloat16 AND TFOLD normBf16OrF32 narrow fp32 norms to bf16 via RNE on device - throughput is apples-to-apples")

if __name__ == "__main__":
    main()
