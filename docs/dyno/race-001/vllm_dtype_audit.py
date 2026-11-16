"""Inspect which dtype vLLM actually keeps on-device for linear_attn.norm.weight.

Strategy: load with vllm.LLM(dtype='bfloat16' | 'float32'), then reach into the
underlying model parameters and print each linear_attn.norm.weight's dtype,
shape, device, mean, min, max. If vLLM narrows FP32 upstream weights to BF16
at load (the default), we document it; if it keeps FP32, we note that.
"""
import argparse, json, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from vllm import LLM
    print(f"loading vllm (dtype={args.dtype})", file=sys.stderr)
    llm = LLM(model=args.model, dtype=args.dtype,
              gpu_memory_utilization=0.08,
              max_model_len=256, enforce_eager=True,
              trust_remote_code=True)

    rep = {"vllm_requested_dtype": args.dtype, "entries": []}
    # vllm.LLM wraps an engine_core; reach into model_runner.model.
    try:
        engine = llm.llm_engine
        # vllm v1: engine.engine_core._exec via a worker; simpler: ask for a dumped param list
        # Use collect_model_execution_logs - but easier to walk .named_parameters()
        mr = None
        # Try v1 path
        try:
            mr = engine.engine_core.executor.collective_rpc("get_model_runner", args=())[0]
        except Exception as e_v1:
            rep["v1_rpc_error"] = str(e_v1)
        if mr is None:
            # Fallback: query any attribute that exposes params
            mr = engine
        # fallback: send a request to introspect via a worker function
        # Simplest path: write a worker side function
        pass
    except Exception as e:
        rep["engine_introspection_error"] = str(e)

    # Fallback: dump all named parameters via a RPC to the worker.
    try:
        def _dump_norms(worker, pattern="linear_attn.norm.weight"):
            out = []
            model = worker.model_runner.model
            for name, p in model.named_parameters():
                if pattern in name:
                    out.append({
                        "name": name,
                        "dtype": str(p.dtype),
                        "shape": list(p.shape),
                        "device": str(p.device),
                        "mean": float(p.float().mean().item()),
                        "min":  float(p.float().min().item()),
                        "max":  float(p.float().max().item()),
                    })
            return out
        results = llm.llm_engine.engine_core.executor.collective_rpc(_dump_norms, args=())
        for r in results:
            rep["entries"].extend(r)
    except Exception as e:
        rep["collective_rpc_error"] = repr(e)
        # Last resort: also enumerate A_log dtype
        try:
            def _dump_a_log(worker):
                out = []
                model = worker.model_runner.model
                for name, p in model.named_parameters():
                    if "linear_attn.A_log" in name:
                        out.append({"name": name, "dtype": str(p.dtype),
                                    "shape": list(p.shape)})
                return out
            rep["a_log_entries"] = llm.llm_engine.engine_core.executor.collective_rpc(_dump_a_log, args=())
        except Exception as e2:
            rep["collective_rpc_error_2"] = repr(e2)

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)
    if rep.get("entries"):
        print(f"found {len(rep['entries'])} linear_attn.norm.weight tensors", file=sys.stderr)
        for e in rep["entries"][:3]:
            print(" ", e, file=sys.stderr)

if __name__ == "__main__":
    main()
