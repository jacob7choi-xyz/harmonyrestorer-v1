"""The build command: blinded listening clips and a sealed key, from an acquisition directory.

Run it from the repository root, with the OpGAN checkpoint and the UVR model directory in place:

    uv run python -m benchmark.listening.run_build --acquisition-dir DIR --out-dir NEW_DIR
        --blind-seed SEED --uvr-model-dir DIR

The blinding seed is an operator secret. It is written only into the sealed key, and it should not
be stored in the repository. The output directory must not exist, and it is published whole or not
at all. The engines run in a scratch directory that is removed when the command ends.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from benchmark.golden_fixture import platform_token
from benchmark.listening.build import BuildError, build
from benchmark.listening.protocol import ListeningProtocolError, load_listening_protocol
from benchmark.protocol import ProtocolError, load_protocol
from benchmark.run_preconditions import verify_run_preconditions
from benchmark.run_systems import OPGAN_CHECKPOINT, build_production_systems, check_preflight
from benchmark.runner import CanonicalFileError
from benchmark.systems_stage import HarnessFailure

__all__ = ["main"]

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command-line arguments.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        The parsed namespace. The blinding seed is a non-negative integer.
    """
    parser = argparse.ArgumentParser(description="Build the blinded listening clips.")
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--blind-seed", type=_non_negative, required=True)
    parser.add_argument("--uvr-model-dir", type=Path, required=True)
    parser.add_argument("--opgan-checkpoint", type=Path, default=OPGAN_CHECKPOINT)
    return parser.parse_args(argv)


def _non_negative(text: str) -> int:
    """An argparse type: a non-negative integer."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("the blinding seed must be non-negative")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Build the clips. Returns 0 on success and 1 on a refusal.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        0 when the build is written, 1 when a check, an engine output or a precondition refuses.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        protocol = load_listening_protocol()
        benchmark = load_protocol()
        check_preflight(benchmark)
        abort = verify_run_preconditions(
            benchmark,
            runtime_platform=platform_token(),
            opgan_checkpoint=args.opgan_checkpoint,
            uvr_model_dir=args.uvr_model_dir,
        )
        if abort is not None:
            raise BuildError(f"run preconditions failed: {abort.reason}: {abort.detail}")
        with tempfile.TemporaryDirectory(prefix="listening-build-") as scratch_name:
            scratch = Path(scratch_name)
            opgan, uvr = build_production_systems(
                benchmark, scratch, args.opgan_checkpoint, args.uvr_model_dir
            )
            build(
                protocol,
                benchmark,
                args.acquisition_dir,
                args.out_dir,
                seed=args.blind_seed,
                opgan=opgan,
                uvr=uvr,
                scratch=scratch,
            )
    except (
        ListeningProtocolError,
        ProtocolError,
        HarnessFailure,
        BuildError,
        CanonicalFileError,
    ) as error:
        logger.error("Build refused: %s", error)
        print(f"run_build: {error}", file=sys.stderr)
        return 1
    print(f"run_build: wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
