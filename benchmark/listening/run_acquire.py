"""The acquisition command: download, decode and window the listening study's recordings.

Run it from the repository root, after the universe file is in place:

    uv run python -m benchmark.listening.run_acquire --out-dir NEW_DIR

The output directory must not exist. It is published whole or not at all. This command downloads
audio, so it runs only after the protocol is merged.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from benchmark.listening.acquire import AcquisitionError, acquire
from benchmark.listening.manifest import UniverseError, read_universe
from benchmark.listening.protocol import ListeningProtocolError, load_listening_protocol
from benchmark.protocol import ProtocolError, load_protocol
from benchmark.training_corpus import TrainingCorpusError, load_training_identities

__all__ = ["main"]

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command-line arguments.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        The parsed namespace, with `out_dir` as a Path.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="a directory that does not exist"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the acquisition and report a refusal on stderr.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        0 on success, 1 when the protocol, the universe, the training corpus or the acquisition
        refuses.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    try:
        protocol = load_listening_protocol()
        benchmark = load_protocol()
        universe = read_universe(protocol)
        training = load_training_identities()
        out = acquire(protocol, benchmark, universe, training, args.out_dir)
    except (
        ListeningProtocolError,
        UniverseError,
        ProtocolError,
        TrainingCorpusError,
        AcquisitionError,
    ) as error:
        logger.error("Acquisition refused: %s", error)
        print(f"run_acquire: {error}")
        return 1
    print(f"run_acquire: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
