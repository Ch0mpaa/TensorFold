"""Convert one EXL3 group into a .tfold v0.1 container.

Layout of the output tensors (names are stable; readers can look them up):

    trellis        dtype=exl3_mcg_tile    shape=[kt, nt, tile_words]   (int16 bytes preserved verbatim)
    suh            dtype=fp16             shape=[K]
    svh            dtype=fp16             shape=[N]
    (bias)         dtype=fp16             shape=[N]   (optional)

Packed trellis bytes are NEVER dequantized and requantized. This is a
pure container conversion: the same bits land in the new file.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np

import tfold_py

from .oracle import OracleBlock, MODEL_SHA256, MODEL_URL


def convert_group_to_tfold(
    group: OracleBlock,
    out_path: str | pathlib.Path,
    *,
    tokenizer_sha256: Optional[str] = None,
    bias_fp16: Optional[bytes] = None,
    model_name: str = "Qwen3.5-2B-exl3-mcg-128x128-oracle",
    arch: str = "qwen3",
) -> dict:
    """Write one EXL3 group to a .tfold v0.1 file.

    The packed ``trellis`` bytes are preserved exactly. The returned
    header carries the final per-tensor offsets and SHA-256 values.
    """
    out_path = pathlib.Path(out_path)

    if group.codebook != "mcg":
        raise ValueError(
            f"Stage 5 supports codebook 'mcg' only; got {group.codebook!r}. "
            "Other codebooks require inst3 or mul1 scheme tag + matching Zig decoder."
        )

    if tokenizer_sha256 is None:
        tokenizer_sha256 = "0" * 64

    # The mcg "codebook" is procedural: value(s) = f(s * 0xCBAC1FED).
    # v0.1 schema requires an embedded __codebook__ tensor for exl3_* schemes,
    # so we pin the multiplier bytes here. The marker is 4 bytes of the mcg
    # u32 constant (little-endian), enough to disambiguate from inst3 / mul1.
    import struct as _struct
    MCG_MUL_LE = _struct.pack("<I", 0xCBAC1FED)

    tensors_list = [
        {"name": "__codebook__", "dtype": "u32",
         "shape": [1], "offset": 0,
         "nbytes": 4, "sha256": "0" * 64},
        {"name": "trellis", "dtype": "exl3_mcg_tile",
         "shape": group.trellis_shape, "offset": 0,
         "nbytes": len(group.trellis_i16), "sha256": "0" * 64},
        {"name": "suh", "dtype": "fp16",
         "shape": [group.K], "offset": 0,
         "nbytes": len(group.suh_fp16), "sha256": "0" * 64},
        {"name": "svh", "dtype": "fp16",
         "shape": [group.N], "offset": 0,
         "nbytes": len(group.svh_fp16), "sha256": "0" * 64},
    ]
    buffers: dict[str, object] = {
        "__codebook__": np.frombuffer(MCG_MUL_LE, dtype=np.dtype("<u4")).reshape(1),
        "trellis": group.trellis_i16,
        "suh": np.frombuffer(group.suh_fp16, dtype=np.dtype("<f2")).reshape(group.K),
        "svh": np.frombuffer(group.svh_fp16, dtype=np.dtype("<f2")).reshape(group.N),
    }

    if bias_fp16 is not None:
        if len(bias_fp16) != group.N * 2:
            raise ValueError(f"bias_fp16 must be {group.N*2} bytes; got {len(bias_fp16)}")
        tensors_list.append({
            "name": "bias", "dtype": "fp16",
            "shape": [group.N], "offset": 0,
            "nbytes": len(bias_fp16), "sha256": "0" * 64,
        })
        buffers["bias"] = np.frombuffer(bias_fp16, dtype=np.dtype("<f2")).reshape(group.N)

    parameters = group.K * group.N

    header = {
        "tfold": {"version": "0.1", "writer": "tfold-convert-exl3@0.1.0"},
        "model": {
            "name": model_name,
            "arch": arch,
            "parameters": parameters,
            "vocab_size": 151936,       # inherited from the source model family (Qwen3)
            "context_length": 32768,
            "source_sha256": MODEL_SHA256,
            "source_url": MODEL_URL,
        },
        "tokenizer": {"type": "tiktoken", "sha256": tokenizer_sha256},
        "quant": {
            "scheme": "exl3_mcg",
            "bits_per_weight": group.bits_per_weight,
            "block_size": 256,  # 256 weights per EXL3 tile
        },
        "tensors": tensors_list,
    }

    final_header = tfold_py.write(out_path, header, buffers)
    return final_header


def read_converted_group_bytes(path: str | pathlib.Path) -> dict[str, bytes]:
    """Open the converted file via the Stage 2 reader and return the
    raw tensor byte slices (as bytes), ready for byte-for-byte compare."""
    header, tensors = tfold_py.read(path, pin_source_sha256=MODEL_SHA256)
    out: dict[str, bytes] = {}
    for spec in header["tensors"]:
        name = spec["name"]
        arr = tensors[name]
        out[name] = bytes(np.ascontiguousarray(arr).view(np.uint8))
    return out
