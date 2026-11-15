"""Build examples/demo.tfold: a small FP32 fixture that round-trips and
executes a verified matrix-vector operation.

Fixture contents:
    weight.linear_0 : fp32 [8, 4]  deterministic from a numpy RNG seed
    bias.linear_0   : fp32 [4]     deterministic from the same seed
    input.x         : fp32 [8]     deterministic from the same seed

The companion reference output is y = x @ weight + bias, computed in
pure numpy with the same deterministic inputs. Stored alongside the
.tfold for cross-verification in examples/demo.reference.npz.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import tfold_py


HERE = pathlib.Path(__file__).resolve().parent
OUT_FILE = HERE / "demo.tfold"
OUT_REF = HERE / "demo.reference.npz"


def deterministic_source_sha256() -> str:
    """Pretend we have a 'source checkpoint' to pin the file to.
    Use a hash of a fixed byte string so anyone rerunning this gets
    the same digest."""
    return hashlib.sha256(b"tfold-v0.1-demo-fp32-fixture").hexdigest()


def build() -> dict:
    rng = np.random.default_rng(seed=0x7F01D)  # deterministic
    w = rng.standard_normal((8, 4), dtype=np.float32)
    b = rng.standard_normal((4,), dtype=np.float32)
    x = rng.standard_normal((8,), dtype=np.float32)

    # Reference computation
    y = x @ w + b  # shape (4,)

    # Save numpy reference
    np.savez(OUT_REF, weight=w, bias=b, input=x, output=y)

    header = {
        "tfold": {"version": "0.1", "writer": "tfold-py@0.1.0"},
        "model": {
            "name": "tfold-demo-fp32",
            "arch": "qwen3",  # arbitrary; the demo isn't a Qwen model, just reuses a registered arch tag
            "parameters": int(w.size + b.size),
            "vocab_size": 1,
            "context_length": 1,
            "source_sha256": deterministic_source_sha256(),
            "source_url": "builtin://tfold-v0.1-demo-fp32-fixture",
        },
        "tokenizer": {
            "type": "tiktoken",
            # Tokenizer file does not exist for the demo; use a deterministic
            # placeholder SHA so the header is schema-conforming. A real model
            # would point to a real tokenizer file alongside the .tfold.
            "sha256": hashlib.sha256(b"tfold-v0.1-demo-no-tokenizer").hexdigest(),
        },
        "quant": {
            "scheme": "fp32",
            "bits_per_weight": 32.0,
            "block_size": 0,
        },
        "tensors": [
            {"name": "weight.linear_0", "dtype": "fp32", "shape": [8, 4], "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "bias.linear_0",   "dtype": "fp32", "shape": [4],    "offset": 0, "nbytes": 0, "sha256": "0" * 64},
            {"name": "input.x",         "dtype": "fp32", "shape": [8],    "offset": 0, "nbytes": 0, "sha256": "0" * 64},
        ],
    }

    final = tfold_py.write(
        OUT_FILE,
        header,
        {"weight.linear_0": w, "bias.linear_0": b, "input.x": x},
    )
    return final


def main() -> int:
    final_header = build()
    print(f"wrote {OUT_FILE}  size={OUT_FILE.stat().st_size} bytes")
    print(f"wrote {OUT_REF}   (numpy reference)")
    print(f"source_sha256 (pin value): {final_header['model']['source_sha256']}")
    for t in final_header["tensors"]:
        print(f"  tensor {t['name']:20s} dtype={t['dtype']:6s} shape={t['shape']} "
              f"offset={t['offset']:6d} nbytes={t['nbytes']:6d} sha={t['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
