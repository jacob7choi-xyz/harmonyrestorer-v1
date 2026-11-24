"""The analysis command: descriptive results from the saved answers and the sealed key.

Run it from the repository root:

    uv run python -m benchmark.listening.run_analysis --build-dir DIR
        --answers-dir DIR --out-dir NEW_DIR

The output directory must not exist. It receives `analysis.json` and `analysis.md`, and both are
written or neither is.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmark.listening.analysis import AnalysisError, analyse, write_analysis
from benchmark.listening.protocol import ListeningProtocolError, load_listening_protocol

__all__ = ["main"]

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command-line arguments.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Analyse the saved listening answers.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--answers-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analysis. Returns 0 on success and 1 on a refusal.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        0 when the result is written, 1 when the protocol, the key or an answers file refuses.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        protocol = load_listening_protocol()
        document = analyse(protocol, args.build_dir, args.answers_dir)
        out = write_analysis(document, args.out_dir)
    except (ListeningProtocolError, AnalysisError) as error:
        logger.error("Analysis refused: %s", error)
        print(f"run_analysis: {error}", file=sys.stderr)
        return 1
    print(f"run_analysis: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
