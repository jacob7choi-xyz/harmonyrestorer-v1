"""The test-page command: one listener's self-contained HTML page, written beside the build.

Run it from the repository root, once per listener in order of consent:

    uv run python -m benchmark.listening.run_test_page --build-dir DIR --pages-dir DIR
        --listener-index 0 --listener-id L000

The page refers to clips by a path relative to the pages directory. The page for a listener is
written once, and a second attempt for the same index is refused.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmark.listening.protocol import ListeningProtocolError, load_listening_protocol
from benchmark.listening.test_page import PageError, write_test_page

__all__ = ["main"]

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command-line arguments.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        The parsed namespace. The listener index is a non-negative integer.
    """
    parser = argparse.ArgumentParser(description="Write one listener's test page.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--listener-index", type=_non_negative, required=True)
    parser.add_argument("--listener-id", required=True)
    return parser.parse_args(argv)


def _non_negative(text: str) -> int:
    """An argparse type: a non-negative integer."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("the listener index must be non-negative")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Write the page. Returns 0 on success and 1 on a refusal.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        0 when the page is written, 1 when the protocol, the key or a clip refuses.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        protocol = load_listening_protocol()
        path = write_test_page(
            args.build_dir, args.pages_dir, protocol, args.listener_index, args.listener_id
        )
    except (ListeningProtocolError, PageError) as error:
        logger.error("Test page refused: %s", error)
        print(f"run_test_page: {error}", file=sys.stderr)
        return 1
    print(f"run_test_page: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
