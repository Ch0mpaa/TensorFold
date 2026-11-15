"""Exception taxonomy matching spec §4.

Every reader rejection that the spec enumerates maps to one of these.
The exception message carries the stable reject code as the first token,
so tests + logs can grep for the code.
"""


class TfoldError(Exception):
    """Base class."""


class SpecError(TfoldError):
    """Spec-defined fail-closed rejection. Carries a stable reject code."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code} {detail}".strip())


# Reject codes (spec §4 table). Keep these string constants stable.
TOO_SMALL              = "TFOLD_TOO_SMALL"
BAD_MAGIC              = "TFOLD_BAD_MAGIC"
VERSION_UNSUPPORTED    = "TFOLD_VERSION_UNSUPPORTED"
HEADER_OVERFLOW        = "TFOLD_HEADER_OVERFLOW"
HEADER_PARSE           = "TFOLD_HEADER_PARSE"
HEADER_SCHEMA          = "TFOLD_HEADER_SCHEMA"
INTEGRITY              = "TFOLD_INTEGRITY"
TENSOR_OOB             = "TFOLD_TENSOR_OOB"
TENSOR_ALIGN           = "TFOLD_TENSOR_ALIGN"
TENSOR_OVERLAP         = "TFOLD_TENSOR_OVERLAP"
TENSOR_DUP_NAME        = "TFOLD_TENSOR_DUP_NAME"
UNKNOWN_ARCH           = "TFOLD_UNKNOWN_ARCH"
UNKNOWN_DTYPE          = "TFOLD_UNKNOWN_DTYPE"
UNKNOWN_SCHEME         = "TFOLD_UNKNOWN_SCHEME"
UNKNOWN_TOKENIZER_TYPE = "TFOLD_UNKNOWN_TOKENIZER_TYPE"
UNKNOWN_RESERVED_NAME  = "TFOLD_UNKNOWN_RESERVED_NAME"
CHECKPOINT_MISMATCH    = "TFOLD_CHECKPOINT_MISMATCH"
TENSOR_INTEGRITY       = "TFOLD_TENSOR_INTEGRITY"
TENSOR_SHAPE_NBYTES    = "TFOLD_TENSOR_SHAPE_NBYTES"
TRAILING_BYTES         = "TFOLD_TRAILING_BYTES"
