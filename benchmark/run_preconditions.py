"""The run-level preconditions a run must pass before it opens any candidate.

Protocol semantics and runtime conformance are checked when the protocol loads. What
remains is checked here, in this order:

0. **Torch loading override.** A process with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` set in its
   environment aborts with `ENVIRONMENT_MISMATCH`, because that variable changes how
   `torch.load` treats the UVR weights (see below). The check is by presence, not value.
1. **Golden fixtures.** A runtime other than the reference platform aborts with
   `ENVIRONMENT_MISMATCH` before any fixture is tried; a fixture that does not reproduce
   its frozen digest aborts with `GOLDEN_FIXTURE_FAILURE` (protocol v5).
2. **The degradation golden.** Recomputed and compared with its frozen digest; a mismatch
   aborts with `GOLDEN_FIXTURE_FAILURE` (protocol v7).
3. **Model files.** The OpGAN checkpoint, the UVR model, and the two UVR parameter files are
   hashed and compared with their frozen digests (protocol v7); a missing or different file
   aborts with the code `runner.model_digest_mismatch_abort` names (protocol v8). Every
   mismatching file is reported, not only the first. The systems stage compares the UVR files
   again after each warm-up (`benchmark.run_systems`), through `uvr_file_failures`.

None of these aborts names a source, because no candidate has been opened yet. Three callers run
these checks. The canonicalisation runner (`benchmark/runner.py`) calls this module before any
worker starts. The selection walk (`benchmark/run_walk.py`) calls it before it opens a candidate.
The systems stage (`benchmark/run_systems.py`) calls it before it builds any engine.

**Check before the engines load, and never download mid-run.** The caller supplies the file
locations: audio-separator reads its model directory from `AUDIO_SEPARATOR_MODEL_DIR` when
that is set, so the directory to check is the one the engine will use, not a default path.
The check has to run before the UVR separator is built, because `load_model` downloads any
missing file and loads the model in one call, so a check made afterwards would vouch for a
file already loaded. A missing file therefore aborts here rather than being fetched; the
files are provisioned and pinned in a separate, explicit step. audio-separator 0.44.2 loads
the UVR model with `torch.load` and no `weights_only` argument, which torch 2.14.1 treats as
`weights_only=True` unless `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is set. Its two parameter
files are JSON data.

The OpGAN engine already verifies its checkpoint against the backend's pin before loading
it (`read_verified_checkpoint` in `backend/app/services/opgan_denoiser.py`), so the OpGAN
check here repeats that by path, as protocol v7 requires of the runner. Nothing equivalent
exists for UVR: this check and audio-separator's load read the UVR files separately.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from benchmark.degradation import golden_degraded_sha256
from benchmark.golden_fixture import verify_golden_fixtures
from benchmark.protocol import Protocol
from benchmark.source_outcome import AbortReason, RunAbortOutcome

__all__ = [
    "degradation_golden_failure",
    "file_sha256",
    "model_file_failures",
    "uvr_file_failures",
    "verify_run_preconditions",
]

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 20
_VR_MODEL_DATA = "vr_model_data.json"
_MDX_MODEL_DATA = "mdx_model_data.json"
_TORCH_FORCE_ENV = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


def file_sha256(path: Path, chunk_bytes: int = _CHUNK_BYTES) -> str | None:
    """The SHA-256 of a regular file read in chunks, or None if it is not one.

    Args:
        path: The file to hash.
        chunk_bytes: How much to read at a time, so a large model is never held whole.

    Returns:
        The lowercase hex digest, or None when the path is missing or not a regular file.
    """
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def degradation_golden_failure(protocol: Protocol) -> str | None:
    """Why the degradation golden did not reproduce its frozen digest, or None if it did.

    Run after the golden fixtures have passed, so the fixture it degrades canonicalises.
    """
    frozen = protocol.degradation.golden.degraded_sha256
    actual = golden_degraded_sha256(protocol)
    if actual != frozen:
        return f"degradation golden produced {actual}, expected {frozen}"
    return None


def _digest_failures(files: tuple[tuple[Path, str], ...]) -> list[str]:
    """Each (path, frozen digest) pair whose file is missing or differs from the digest."""
    failures = []
    for path, frozen in files:
        actual = file_sha256(path)
        if actual is None:
            failures.append(f"{path} is missing or not a regular file")
        elif actual != frozen:
            failures.append(f"{path} has digest {actual}, expected {frozen}")
    return failures


def uvr_file_failures(protocol: Protocol, uvr_model_dir: Path) -> list[str]:
    """Each UVR model file that is missing or differs from its frozen digest.

    The UVR model and its two parameter files are read from `uvr_model_dir`, the directory
    audio-separator loads them from. The systems stage calls this after each warm-up, since the
    warm-up is the call that loads the model.

    Args:
        protocol: The loaded protocol.
        uvr_model_dir: The directory audio-separator loads the UVR files from.

    Returns:
        One description per failing file, empty when all three match.
    """
    execution = protocol.system_execution
    return _digest_failures(
        (
            (uvr_model_dir / execution.uvr_model, execution.uvr_model_sha256),
            (uvr_model_dir / _VR_MODEL_DATA, execution.uvr_vr_model_data_sha256),
            (uvr_model_dir / _MDX_MODEL_DATA, execution.uvr_mdx_model_data_sha256),
        )
    )


def model_file_failures(
    protocol: Protocol, *, opgan_checkpoint: Path, uvr_model_dir: Path
) -> list[str]:
    """Each model file that is missing or differs from its frozen digest.

    Args:
        protocol: The loaded protocol.
        opgan_checkpoint: The checkpoint the OpGAN engine loads.
        uvr_model_dir: The directory audio-separator loads the UVR files from.

    Returns:
        One description per failing file, empty when all four match.
    """
    opgan = ((opgan_checkpoint, protocol.system_execution.opgan_checkpoint_sha256),)
    return _digest_failures(opgan) + uvr_file_failures(protocol, uvr_model_dir)


def _weights_only_override() -> RunAbortOutcome | None:
    """An abort when `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is set, or None when it is not.

    audio-separator loads the UVR model with `torch.load` and no `weights_only` argument, so this
    variable decides how the UVR weights are unpickled. A run that sets it would load weights
    the protocol's loading assumptions do not cover, so the run is refused before anything else.
    """
    if _TORCH_FORCE_ENV not in os.environ:
        return None
    detail = f"{_TORCH_FORCE_ENV} is set, which changes how torch.load treats the UVR weights"
    logger.error("Run refused: %s", detail)
    return RunAbortOutcome(reason=AbortReason.ENVIRONMENT_MISMATCH, detail=detail)


def verify_run_preconditions(
    protocol: Protocol,
    *,
    runtime_platform: str,
    opgan_checkpoint: Path,
    uvr_model_dir: Path,
) -> RunAbortOutcome | None:
    """Run the override, golden, degradation, and model checks, in that order.

    Args:
        protocol: The loaded protocol.
        runtime_platform: This runtime's platform, normally `platform_token()`.
        opgan_checkpoint: The checkpoint the OpGAN engine loads.
        uvr_model_dir: The directory audio-separator loads the UVR files from.

    Returns:
        None when every precondition holds, otherwise the first abort, which names no source.
    """
    override = _weights_only_override()
    if override is not None:
        return override
    golden = verify_golden_fixtures(protocol, runtime_platform=runtime_platform)
    if golden is not None:
        return golden
    degradation = degradation_golden_failure(protocol)
    if degradation is not None:
        logger.error("Degradation golden failed: %s", degradation)
        return RunAbortOutcome(reason=AbortReason.GOLDEN_FIXTURE_FAILURE, detail=degradation)
    failures = model_file_failures(
        protocol, opgan_checkpoint=opgan_checkpoint, uvr_model_dir=uvr_model_dir
    )
    if failures:
        detail = "; ".join(failures)
        logger.error("Model files do not match their frozen digests: %s", detail)
        reason = AbortReason(protocol.runner.model_digest_mismatch_abort)
        return RunAbortOutcome(reason=reason, detail=detail)
    logger.info("Run preconditions hold")
    return None
