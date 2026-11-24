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
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

__all__ = ["write_atomically", "write_directory_atomically"]


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


def write_directory_atomically(directory: Path, files: Mapping[str, bytes]) -> None:
    """Create a new directory holding every file, or leave nothing behind.

    Each file is written with `write_atomically` into a staging directory beside `directory`.
    The staging directory is then moved to `directory` with `Path.replace`. A failure at any step
    removes the staging directory, so a partial output is never left under the target name. The
    callers check that `directory` does not exist first, since an existing empty directory would
    be replaced by the move. As in `write_atomically`, the parent directory is not fsynced after
    the move.

    Args:
        directory: The directory to create. Its parent is created if it does not exist.
        files: Each file's name inside the directory, mapped to its complete contents.

    Raises:
        OSError: If a file cannot be written, or the staging directory cannot be moved into place.
            The staging directory is removed first.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.partial")
    staging.mkdir()
    published = False
    try:
        for name, data in files.items():
            write_atomically(staging / name, data)
        staging.replace(directory)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
