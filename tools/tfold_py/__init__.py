"""tfold v0.1 reference writer / reader.

Public API:
    write(path, header, tensors, pin_source_sha256=None) -> dict
        Produce a .tfold file. Returns the exact header that was written
        (with per-tensor offsets/nbytes/sha256 computed).

    read(path, pin_source_sha256=None) -> (header, tensor_dict)
        Validate + mmap-load a .tfold file. Returns the header and a
        dict mapping tensor name -> numpy array (view into the mmap,
        little-endian, correct dtype and shape).

    SpecError
        Base class for every spec-defined fail-closed rejection.

The reference implementation is stdlib + numpy only. It is intended to
be bit-identical with the Zig loader that lands in Stage 4. Spec and
JSON schema live in docs/tfold-format-v0.md and schemas/tfold-v0.json.
"""

from ._errors import SpecError, TfoldError
from ._io import read, write
from ._v02 import read_v02, write_v02, resolve_tensor_scheme
from ._version import WRITER_ID, SPEC_VERSION_STR, SPEC_VERSION_U32

__all__ = [
    "read",
    "write",
    "read_v02",
    "write_v02",
    "resolve_tensor_scheme",
    "SpecError",
    "TfoldError",
    "WRITER_ID",
    "SPEC_VERSION_STR",
    "SPEC_VERSION_U32",
]
