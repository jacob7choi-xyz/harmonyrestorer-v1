"""The one JSON serialisation for result files whose digests are recorded.

A result document is written with these bytes, so a given document is written as the
same bytes on every run, and its digest is stable. The layout is indented, with sorted keys,
the default ASCII escaping, and one trailing newline. Non-finite numbers are refused, since JSON
cannot carry them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

__all__ = ["canonical_json", "canonical_json_bytes"]


def canonical_json(document: Mapping[str, Any]) -> str:
    """The document as canonical JSON text.

    Args:
        document: A JSON-compatible mapping.

    Returns:
        Indented JSON with sorted keys and one trailing newline.

    Raises:
        ValueError: If a number in the document is NaN or infinite.
    """
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    """The document as canonical JSON bytes, UTF-8 encoded.

    Args:
        document: A JSON-compatible mapping.

    Returns:
        The bytes of `canonical_json(document)`.

    Raises:
        ValueError: If a number in the document is NaN or infinite.
    """
    return canonical_json(document).encode("utf-8")
