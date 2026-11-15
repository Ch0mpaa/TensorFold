"""Dtype registry (spec §2.6).

Every dtype has:
  - element_size: bytes per logical element
  - numpy_dtype:  little-endian numpy dtype, or None if numpy has no
                  native representation (bf16 and packed quant dtypes)

Packed quant dtypes have no simple product(shape)*element_size formula;
on-disk nbytes is scheme-defined. For v0.1 Python tooling only supports
FP32/FP16 execution; everything else can be read as opaque bytes.
"""

from __future__ import annotations

import numpy as np

# Dtype registry: name -> (element_size_bytes, numpy_dtype_or_None, is_packed)
# element_size for packed dtypes is 0 (writer MUST pass explicit nbytes).
DTYPES: dict[str, tuple[int, "np.dtype | None", bool]] = {
    "fp16":            (2, np.dtype("<f2"), False),
    "bf16":            (2, None,            False),  # no numpy bf16
    "fp32":            (4, np.dtype("<f4"), False),
    "u8":              (1, np.dtype("<u1"), False),
    "i8":              (1, np.dtype("<i1"), False),
    "u16":             (2, np.dtype("<u2"), False),
    "i16":             (2, np.dtype("<i2"), False),
    "u32":             (4, np.dtype("<u4"), False),
    "i32":             (4, np.dtype("<i4"), False),
    "exl3_tile":       (0, None,            True),
    "exl3_mcg_tile":   (0, None,            True),
}


def expected_nbytes(dtype: str, shape: list[int]) -> int | None:
    """Return the expected nbytes for an unpacked dtype, else None."""
    entry = DTYPES.get(dtype)
    if entry is None:
        return None
    esize, _, is_packed = entry
    if is_packed:
        return None
    n = esize
    for d in shape:
        n *= d
    return n


def numpy_dtype_of(dtype: str) -> "np.dtype | None":
    entry = DTYPES.get(dtype)
    return None if entry is None else entry[1]


def is_known(dtype: str) -> bool:
    return dtype in DTYPES
