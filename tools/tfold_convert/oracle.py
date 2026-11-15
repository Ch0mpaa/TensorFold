"""Read the committed EXL3 oracle fixture at
zig/tests/fixtures/exl3_oracle.bin and return the single public
128x128 mcg 8.00-bpw block + per-side scales + FP64 reference weights.

The fixture format is defined by zig/src/core/exl3_format.zig and
documented in docs/recipes/exl3.md. Field layout (little-endian):

    [ 0.. 8)   magic           b"EXL3ORAC"
    [ 8..16)   records_count   u64
    then records_count records of:
        [ 0.. 2)  halves       u16
        [ 2.. 3)  codebook     u8   (0=inst3, 1=mcg, 2=mul1)
        [ 4.. 8)  kt           u32   (K tile count)
        [ 8..12)  nt           u32   (N tile count)
        then trellis words (kt * nt * tileWords * 2 bytes)
        then states (kt * nt * 256 * 4 bytes)
    then a "spots" header (u32) and that many 6-byte probes
    then one single publicly-verifiable block:
        [ 0.. 2)  halves       u16
        [ 2.. 6)  kt           u32
        [ 6..10)  nt           u32
        [10..14)  trellis_len  u32
        [14..14+trellis_len)   trellis bytes
        u32 suh_len; suh bytes (K * 2)
        u32 svh_len; svh bytes (N * 2)
        u32 wq_len;  wq  bytes (K * N, int8 quantized dequant reference)
        u32 dq_len;  dq  bytes (K * N * 8, FP64 reference weights)

We care only about the final publicly-verifiable block in Stage 5.
"""

from __future__ import annotations

import hashlib
import pathlib
import struct
from dataclasses import dataclass
from typing import Literal


# Committed provenance (also referenced from tools/generate_qnn_full_projection.py).
FIXTURE_SHA256 = "8eecb6ac56325a50634eaf75cd2c2ff4aed74e4341b5f2af6c05a3bf0985c810"
MODEL_SHA256 = "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f"
MODEL_URL = "turboderp/Qwen3.5-2B-exl3@8.00bpw/model.safetensors"
MODEL_LICENSE = "Apache-2.0 (base weights), EXL3 encoding by turboderp"
BLOCK_SOURCE_NOTE = (
    "Public 8.00-bpw mcg block from layers.0.linear_attn.in_proj_qkv, "
    "128x128 window chosen by docs/recipes/exl3.md as an oracle example."
)

Codebook = Literal["inst3", "mcg", "mul1"]
_CODEBOOK_BY_ID: dict[int, Codebook] = {0: "inst3", 1: "mcg", 2: "mul1"}


@dataclass(frozen=True)
class OracleBlock:
    """The single publicly-verifiable EXL3 group in the oracle fixture."""
    codebook: Codebook           # "mcg"
    halves: int                  # 16 (8 bits per weight)
    bits_per_weight: float       # 8.0
    kt: int                      # K-tile count (8)
    nt: int                      # N-tile count (8)
    K: int                       # kt * 16 (128)
    N: int                       # nt * 16 (128)
    trellis_i16: bytes           # raw little-endian int16 payload, shape [kt*nt*tileWords]
    suh_fp16: bytes              # raw little-endian fp16 bytes, shape [K]
    svh_fp16: bytes              # raw little-endian fp16 bytes, shape [N]
    dequant_reference_fp64: bytes  # raw little-endian fp64 bytes, shape [K*N]

    @property
    def tile_words(self) -> int:
        # halves * 8 int16 words per tile
        return self.halves * 8

    @property
    def trellis_shape(self) -> list[int]:
        # Spec: trellis int16 [K/16, N/16, 16 * bits] where bits = halves/2 so 16*bits = halves*8 = tile_words
        return [self.kt, self.nt, self.tile_words]


def _u16(data: bytes, at: int) -> int:
    return struct.unpack_from("<H", data, at)[0]


def _u32(data: bytes, at: int) -> int:
    return struct.unpack_from("<I", data, at)[0]


def _u64(data: bytes, at: int) -> int:
    return struct.unpack_from("<Q", data, at)[0]


def _records_end(data: bytes) -> int:
    """Return the byte offset immediately after the fixture's record list."""
    if data[:8] != b"EXL3ORAC":
        raise ValueError("fixture magic is not EXL3ORAC")
    at = 16
    for _ in range(_u64(data, 8)):
        halves = _u16(data, at)
        if halves == 0:
            raise ValueError("invalid fixture record width")
        kt = _u32(data, at + 4)
        nt = _u32(data, at + 8)
        tile_words = halves * 8
        at += 12 + kt * nt * (tile_words * 2 + 256 * 4)
    return at


def load_oracle_mcg_block(path: str | pathlib.Path | None = None) -> OracleBlock:
    """Load the committed EXL3 oracle fixture and return its publicly-
    verifiable 128x128 mcg 8.00-bpw block.

    If ``path`` is None, defaults to the committed location
    ``zig/tests/fixtures/exl3_oracle.bin`` relative to the repository
    root (identified by walking upward until ``.zig-version`` is found).
    """
    if path is None:
        here = pathlib.Path(__file__).resolve()
        for cand in here.parents:
            if (cand / ".zig-version").exists():
                path = cand / "zig/tests/fixtures/exl3_oracle.bin"
                break
        if path is None:
            raise FileNotFoundError("could not locate .zig-version to resolve default fixture path")
    path = pathlib.Path(path)
    data = path.read_bytes()

    digest = hashlib.sha256(data).hexdigest()
    if digest != FIXTURE_SHA256:
        raise ValueError(
            f"fixture SHA-256 mismatch: got {digest}, expected {FIXTURE_SHA256}. "
            "Refusing to parse an unverified fixture."
        )

    at = _records_end(data)
    spots = _u32(data, at)
    at += 4 + spots * 6

    halves = _u16(data, at)
    kt = _u32(data, at + 2)
    nt = _u32(data, at + 6)
    trellis_len = _u32(data, at + 10)
    at += 14
    trellis_bytes = bytes(data[at:at + trellis_len])
    at += trellis_len

    suh_len = _u32(data, at)
    at += 4
    suh_bytes = bytes(data[at:at + suh_len])
    at += suh_len

    svh_len = _u32(data, at)
    at += 4
    svh_bytes = bytes(data[at:at + svh_len])
    at += svh_len

    wq_len = _u32(data, at)
    at += 4
    # wq is the integer-domain CPU dot reference; not emitted into .tfold v0.1.
    at += wq_len

    dq_len = _u32(data, at)
    at += 4
    dq_bytes = bytes(data[at:at + dq_len])
    at += dq_len

    if (halves, kt, nt) != (16, 8, 8):
        raise ValueError(f"expected (halves=16,kt=8,nt=8), got ({halves},{kt},{nt})")
    K = kt * 16
    N = nt * 16
    tile_words = halves * 8
    expected_trellis = kt * nt * tile_words * 2
    if trellis_len != expected_trellis:
        raise ValueError(f"trellis_len {trellis_len} != expected {expected_trellis}")
    if suh_len != K * 2:
        raise ValueError(f"suh_len {suh_len} != expected {K*2}")
    if svh_len != N * 2:
        raise ValueError(f"svh_len {svh_len} != expected {N*2}")
    if dq_len != K * N * 8:
        raise ValueError(f"dq_len {dq_len} != expected {K*N*8}")
    if at != len(data):
        raise ValueError(f"trailing bytes after dq: {len(data) - at}")

    # bits_per_weight for mcg at halves=16 is 8 (halves/2 == bits).
    bits_per_weight = halves / 2.0

    return OracleBlock(
        codebook="mcg",
        halves=halves,
        bits_per_weight=bits_per_weight,
        kt=kt,
        nt=nt,
        K=K,
        N=N,
        trellis_i16=trellis_bytes,
        suh_fp16=suh_bytes,
        svh_fp16=svh_bytes,
        dequant_reference_fp64=dq_bytes,
    )
