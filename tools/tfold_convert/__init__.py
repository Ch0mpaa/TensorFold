"""tfold_convert — real EXL3 checkpoint ingestion into .tfold v0.1.

Stage 5 scope:
    One EXL3 group → one .tfold v0.1 container.

    A "group" is the quantized-linear bundle described in
    docs/recipes/exl3.md: one `trellis` int16 tensor, plus per-side
    `suh`/`svh` fp16 scale vectors, an optional fp16 `bias`, and a
    named codebook (3inst/mcg/mul1).

    Packed trellis bytes are preserved verbatim (no dequantize-then-
    requantize). Writer + Stage 2 reader + Stage 4 Zig reader all
    verify the identical bytes.

v0.1 limits (see docs/tfold-stage5-conformance.md § v0.1 limits):
    - exactly one codebook per file (quant.scheme is file-wide)
    - exactly one width (bits-per-weight is file-wide)
    - a full multi-layer checkpoint with mixed widths needs v0.2

Public API:
    load_oracle_mcg_block(path) -> OracleBlock
        Read the Stage 4 oracle fixture and return the single public
        mcg 8-bpw block described by docs/recipes/exl3.md:
        trellis int16 [K/16, N/16, 16*bits], suh fp16 [K], svh fp16 [N],
        and the FP64 reference weight matrix.

    convert_group_to_tfold(group, out_path, pin_source_sha256) -> header
        Write the EXL3 group as a .tfold v0.1 file. Returns the final
        header (with offsets + SHA-256 filled in).
"""

from .oracle import OracleBlock, load_oracle_mcg_block
from .convert import convert_group_to_tfold, read_converted_group_bytes

__all__ = [
    "OracleBlock",
    "load_oracle_mcg_block",
    "convert_group_to_tfold",
    "read_converted_group_bytes",
]
