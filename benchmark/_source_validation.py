"""Construction primitives shared by the source-outcome modules.

Internal to the package, which is what the leading underscore says. It exists so that
`source_outcome` does not have to reach past `source_evidence.__all__` into its private
helpers: a module that advertises an explicit public surface and is then bypassed through
underscored imports has a decorative boundary rather than a real one.

These are validators for construction, not domain types, so they do not belong in
`source_evidence` alongside identity and metadata. The dependency runs one way:

    _source_validation -> source_evidence -> source_outcome -> source_outcome_contract

`SourceOutcomeError` is the base every caller raises, and it keeps that name rather than
being renamed to match this module, because the class name is what appears in tracebacks
and logs.
"""

from __future__ import annotations

import re

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


class SourceOutcomeError(Exception):
    """Raised when a value would make a record assert something that did not happen.

    Named for the package rather than for this module on purpose. Defining it here as
    `SourceOutcomeError` and aliasing the historical name would preserve what a caller
    can catch while changing what a traceback, a log line, and pytest all print, since
    `type(error).__name__` follows the class rather than the alias. The defining module
    moves in this refactor; the observable identity does not.
    """


def checked_digest(value: str, field: str) -> str:
    """Require exactly 64 lowercase hexadecimal characters.

    Args:
        value: The candidate digest.
        field: Field name, for the error message.

    Returns:
        The value unchanged.

    Raises:
        SourceOutcomeError: If it is not a SHA-256 digest in canonical form. Case and
            surrounding whitespace are rejected rather than normalised, because a record
            that accepts two spellings of one digest breaks content addressing.
    """
    if not _SHA256.fullmatch(value):
        raise SourceOutcomeError(
            f"{field} must be 64 lowercase hexadecimal characters, got {value!r}"
        )
    return value


def checked_identifier(value: str, field: str) -> str:
    """Require a non-blank identifier with no surrounding whitespace.

    Args:
        value: The candidate identifier.
        field: Field name, for the error message.

    Returns:
        The value unchanged.

    Raises:
        SourceOutcomeError: If it is empty, blank, or padded. Blank-as-present is how a
            record ends up naming nothing while appearing complete.
    """
    if not value or not value.strip() or value.strip() != value:
        raise SourceOutcomeError(f"{field} must be non-blank and unpadded, got {value!r}")
    return value


def checked_int(value: int, field: str, *, minimum: int) -> int:
    """Require a plain integer at or above a floor.

    Args:
        value: The candidate.
        field: Field name, for the error message.
        minimum: Smallest permitted value.

    Returns:
        The value unchanged.

    Raises:
        SourceOutcomeError: If it is a `bool` or below the floor. `bool` is rejected
            explicitly because `isinstance(True, int)` is true, so a channel count of
            `True` would otherwise be recorded as one channel.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceOutcomeError(f"{field} must be a plain integer, got {value!r}")
    if value < minimum:
        raise SourceOutcomeError(f"{field} must be at least {minimum}, got {value}")
    return value
