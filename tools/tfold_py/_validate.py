"""Header schema validation (spec §4 steps 5-6, 12).

Lives separately from the reader's bounds/integrity checks: this
module answers "is the HEADER valid JSON that satisfies tfold-v0.json";
the reader additionally enforces §4 steps 7-11 and 13-14 against the
actual file bytes.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import jsonschema

from . import _errors as E

_SCHEMA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "schemas" / "tfold-v0.json"


def load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


_SCHEMA_SINGLETON: dict[str, Any] | None = None
_VALIDATOR_SINGLETON: jsonschema.protocols.Validator | None = None


def _validator() -> jsonschema.protocols.Validator:
    global _SCHEMA_SINGLETON, _VALIDATOR_SINGLETON
    if _VALIDATOR_SINGLETON is None:
        _SCHEMA_SINGLETON = load_schema()
        cls = jsonschema.validators.validator_for(_SCHEMA_SINGLETON)
        cls.check_schema(_SCHEMA_SINGLETON)
        _VALIDATOR_SINGLETON = cls(_SCHEMA_SINGLETON)
    return _VALIDATOR_SINGLETON


def validate_header(header: dict[str, Any]) -> None:
    """Raise SpecError(HEADER_SCHEMA) if the header doesn't conform."""
    errors = sorted(_validator().iter_errors(header), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise E.SpecError(E.HEADER_SCHEMA, f"at {path}: {e.message}")
