"""Writing a file so that it is either complete or absent.

The project rule for file output is a temporary file beside the target, flushed and fsynced,
then moved over the target with `Path.replace`, so a crash or a full disk leaves the old file
or nothing rather than a partial one. `replace` rather than `rename`, because `rename` fails on
Windows when the target exists.

> **Note:** The parent directory is not fsynced after the replace, so on some filesystems a
> crash just after it can lose the rename itself, leaving the old file in place. The data in
> the temporary file is durable before the rename; whether the rename survives a crash is not.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["write_atomically"]


def write_atomically(target: Path, data: bytes | memoryview) -> None:
    """Write *data* to a temporary file beside *target*, fsync it, then replace *target*.

    Args:
        target: The file to create or replace. Its directory must exist.
        data: The complete contents.

    Raises:
        OSError: If the write, the fsync, or the replace fails. The temporary file is removed
            and *target* is left as it was.
    """
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    replaced = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(target)
        replaced = True
    finally:
        if not replaced:
            Path(temporary).unlink(missing_ok=True)
