"""SHA-256 helpers and FNV-1a-64 trailer (spec §1.6)."""

from __future__ import annotations

import hashlib
from typing import Iterable


FNV_OFFSET_64 = 0xCBF29CE484222325
FNV_PRIME_64 = 0x100000001B3
FNV_MASK_64 = (1 << 64) - 1


def fnv1a_64(data: bytes) -> int:
    """Reference FNV-1a-64 over a single bytes buffer."""
    h = FNV_OFFSET_64
    for b in data:
        h ^= b
        h = (h * FNV_PRIME_64) & FNV_MASK_64
    return h


def fnv1a_64_chunks(chunks: Iterable[bytes]) -> int:
    """Streaming variant: incrementally digest multiple chunks."""
    h = FNV_OFFSET_64
    for chunk in chunks:
        for b in chunk:
            h ^= b
            h = (h * FNV_PRIME_64) & FNV_MASK_64
    return h


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
