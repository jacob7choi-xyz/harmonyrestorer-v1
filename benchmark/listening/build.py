"""Build the blinded clips for the listening study from an acquisition directory.

For each recording the build makes three conditions, all cut from the same eight-second window:

- `original`: the protocol v14 canonical waveform, 16 kHz mono, from the whole side.
- `opgan`: the OpGAN engine run on the whole canonical side at 16 kHz.
- `uvr`: the whole side decoded from the staged FLAC, downmixed to mono and resampled to 44100 Hz
  with librosa soxr_hq, and run through the UVR engine. The engine returns its own 16 kHz mono
  conversion (protocol v2, `conditions.uvr`), which the build reads without resampling it again.

The window is the one the acquisition manifest records, and it is cut at the same sample index in
every condition. Each condition is loudness matched with the protocol's rule and written as 16-bit
PCM, with blinded file names. The mapping from blinded names to recordings and conditions is written
to a sealed key, and the key's SHA-256 goes into the public build manifest.

> **Note:** Every engine output is read with `read_output` at the rate the protocol names for it,
> and an output at any other rate or with any other channel count is refused.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

from benchmark.atomic_write import write_atomically
from benchmark.canonical_json import canonical_json_bytes
from benchmark.canonicalisation import downmix
from benchmark.listening.acquire import ACQUISITION_NAME
from benchmark.listening.loudness import ClipMatch, match_clip
from benchmark.listening.protocol import ListeningProtocol
from benchmark.protocol import Protocol
from benchmark.runner import CanonicalFile
from benchmark.source_acquisition import ArtifactBuffered, acquire_local_artifact
from benchmark.source_decode import decode_source
from benchmark.source_inspection import inspect_source
from benchmark.source_outcome import SourceIneligibleOutcome
from benchmark.systems_stage import (
    HarnessFailure,
    System,
    SystemFailed,
    read_output,
    write_frame_input,
)

__all__ = [
    "BUILD_NAME",
    "CLIPS_DIR",
    "KEY_NAME",
    "SEALED_DIR",
    "BuildError",
    "build",
    "to_pcm16",
]

logger = logging.getLogger(__name__)

BUILD_NAME = "build.json"
CLIPS_DIR = "clips"
SEALED_DIR = "sealed"
KEY_NAME = "key.json"
CONDITIONS = ("original", "opgan", "uvr")
# The UVR rate the protocol names in conditions.uvr. It is not a field of the frozen JSON.
UVR_INPUT_RATE_HZ = 44_100
# The rate the UVR service returns, which protocol v2 names for conditions.uvr.
UVR_OUTPUT_RATE_HZ = 16_000
_INT16_SCALE = 2**15
_INT16_MIN = -(2**15)
_INT16_MAX = 2**15 - 1
_BLIND_ID_HEX_DIGITS = 12
_BLIND_ID_BITS = 4 * _BLIND_ID_HEX_DIGITS


class BuildError(RuntimeError):
    """Raised when the build must stop: a mismatch, a refused engine output, or a clip."""


@dataclass(frozen=True)
class _Recording:
    """One acquired recording, as the manifest and the files on disk describe it."""

    identifier: str
    flac_path: Path
    raw_sha256: str
    canonical: CanonicalFile
    start_s: int


def to_pcm16(samples: npt.NDArray[np.float64], label: str) -> npt.NDArray[np.int16]:
    """Scale by 2**15, round to the nearest integer, and refuse any value outside int16.

    Ties are rounded to the even integer by `numpy.rint`. The protocol says only "nearest".

    Args:
        samples: Float samples, nominally in [-1, 1].
        label: The clip's name, for the error message.

    Returns:
        The 16-bit samples.

    Raises:
        BuildError: If a rounded sample is outside the int16 range.
    """
    scaled = np.rint(np.asarray(samples, dtype=np.float64) * _INT16_SCALE)
    if scaled.size and (scaled.max() > _INT16_MAX or scaled.min() < _INT16_MIN):
        logger.error("Clip %s has a sample outside the 16-bit range after scaling", label)
        raise BuildError(f"clip {label} would clip at 16 bits")
    return scaled.astype(np.int16)


def _read_manifest(protocol: ListeningProtocol, acquisition: Path) -> dict[str, Any]:
    """The acquisition manifest, checked against the protocol's digests.

    Raises:
        BuildError: If the manifest is missing, malformed, or made for another protocol.
    """
    try:
        data: dict[str, Any] = json.loads((acquisition / ACQUISITION_NAME).read_text())
    except (OSError, json.JSONDecodeError) as error:
        logger.error("Cannot read the acquisition manifest in %s: %s", acquisition, error)
        raise BuildError(f"cannot read the acquisition manifest: {error}") from error
    if data.get("protocol_sha256") != protocol.sha256:
        raise BuildError("the acquisition was made for a different protocol")
    if data.get("universe_sha256") != protocol.universe.sha256:
        raise BuildError("the acquisition was made for a different universe")
    if len(data.get("items", [])) != protocol.selection.sample_size:
        raise BuildError("the acquisition does not hold the protocol's sample size")
    return data


def _recordings(acquisition: Path, manifest: dict[str, Any]) -> tuple[_Recording, ...]:
    """Each item as a recording, with its staged FLAC checked against its raw digest.

    Raises:
        BuildError: If a staged FLAC or a canonical file does not match its recorded digest.
    """
    recordings = []
    for entry in manifest["items"]:
        flac = acquisition / entry["flac"]["path"]
        if hashlib.sha256(flac.read_bytes()).hexdigest() != entry["flac"]["sha256"]:
            logger.error("Staged FLAC %s does not match its recorded digest", flac)
            raise BuildError(f"staged FLAC for {entry['identifier']} does not match its digest")
        canonical = CanonicalFile(
            path=acquisition / entry["canonical"]["path"],
            sha256=entry["canonical"]["sha256"],
            samples=entry["canonical"]["samples"],
            raw_sha256=entry["flac"]["sha256"],
            decoded_pcm_sha256=entry["decoded_pcm_sha256"],
        )
        recordings.append(
            _Recording(
                entry["identifier"],
                flac,
                entry["flac"]["sha256"],
                canonical,
                entry["window"]["start_s"],
            )
        )
    return tuple(recordings)


def _uvr_input(benchmark: Protocol, recording: _Recording) -> npt.NDArray[np.float64]:
    """The whole side decoded from the staged FLAC, downmixed and resampled to 44100 Hz.

    The decode runs in this process on a file the worker has already accepted, and its raw digest
    is checked again here.

    Raises:
        BuildError: If the artifact cannot be buffered or decoded.
    """
    identifier = recording.identifier
    config = benchmark.canonicalisation
    artifact = acquire_local_artifact(
        recording.flac_path,
        logical_source_id=identifier,
        config=config.decode,
        expected_sha256=recording.raw_sha256,
    )
    if not isinstance(artifact, ArtifactBuffered):
        logger.error("UVR input for %s could not be buffered: %s", identifier, artifact)
        raise BuildError(f"the source for {identifier} could not be buffered for UVR")
    inspected = inspect_source(
        artifact,
        logical_source_id=identifier,
        protocol_sha256=benchmark.source_sha256,
        config=config,
    )
    if isinstance(inspected, SourceIneligibleOutcome):
        raise BuildError(f"the source for {identifier} was refused for UVR")
    decoded = decode_source(
        artifact,
        inspected,
        logical_source_id=identifier,
        protocol_sha256=benchmark.source_sha256,
        config=config,
    )
    if isinstance(decoded, SourceIneligibleOutcome):
        raise BuildError(f"the source for {identifier} was refused for UVR")
    resample = config.resample
    return np.asarray(
        librosa.resample(
            downmix(decoded),
            orig_sr=inspected.metadata.sample_rate,
            target_sr=UVR_INPUT_RATE_HZ,
            res_type=resample.res_type,
            fix=resample.fix,
            scale=resample.scale,
            axis=resample.axis,
        ),
        dtype=np.float64,
    )


def _engine_output(
    system: System,
    input_samples: npt.NDArray[np.float64],
    input_rate: int,
    output_rate: int,
    path: Path,
    label: str,
) -> npt.NDArray[np.float64]:
    """Write one input at *input_rate*, run one engine, and read its output at *output_rate*.

    The input is cast from float64 to float32, because `write_frame_input` takes float32.

    Raises:
        BuildError: If the engine's output is unreadable, or it is not mono at *output_rate*.
    """
    write_frame_input(input_samples.astype(np.float32), path, input_rate)
    output = system.denoise(path)
    try:
        result = read_output(output, output_rate)
    except HarnessFailure as error:
        logger.error("Engine %s output for %s cannot be read: %s", label, path.name, error)
        raise BuildError(f"engine {label} output cannot be read: {error}") from error
    if isinstance(result, SystemFailed):
        logger.error("Engine %s output rejected for %s: %s", label, path.name, result.reason)
        raise BuildError(f"engine {label} output is not {output_rate} Hz mono: {result.reason}")
    return result.samples


def _whole_sides(
    benchmark: Protocol,
    recording: _Recording,
    opgan: System,
    uvr: System,
    scratch: Path,
    rate: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """The three conditions for the whole side, before any window is cut."""
    name = _stem(recording.identifier)
    original = recording.canonical.load()
    opgan_side = _engine_output(
        opgan, original, rate, rate, scratch / f"{name}_opgan_in.wav", "OpGAN"
    )
    uvr_mono = _uvr_input(benchmark, recording)
    uvr_side = _engine_output(
        uvr,
        uvr_mono,
        UVR_INPUT_RATE_HZ,
        UVR_OUTPUT_RATE_HZ,
        scratch / f"{name}_uvr_in.wav",
        "UVR",
    )
    return {"original": original, "opgan": opgan_side, "uvr": uvr_side}


def _stem(identifier: str) -> str:
    """A file-name-safe stem for an identifier, used only for scratch files."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]


def _cut(
    sides: Mapping[str, npt.NDArray[np.float64]], start_s: int, clip_samples: int, rate: int
) -> dict[str, npt.NDArray[np.float64]]:
    """The same window from every condition, at the same sample index.

    Raises:
        BuildError: If a condition is too short for the window, or the cut lengths differ.
    """
    first = start_s * rate
    end = first + clip_samples
    short = [name for name, side in sides.items() if side.size < end]
    if short:
        logger.error("Conditions %s are too short for the window ending at sample %d", short, end)
        raise BuildError(f"conditions {short} are too short for the window")
    cut = {name: side[first:end] for name, side in sides.items()}
    if len({window.size for window in cut.values()}) != 1:
        raise BuildError("the conditions were cut to different lengths")
    return cut


def _blind_names(seed: int, count: int) -> list[str]:
    """Distinct random-looking identifiers from a generator seeded by the operator's seed.

    Raises:
        BuildError: If the generator cannot produce *count* distinct identifiers.
    """
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < count:
        value = int(generator.integers(0, 2**_BLIND_ID_BITS, dtype=np.uint64))
        name = f"{value:0{_BLIND_ID_HEX_DIGITS}x}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _assign(seed: int, pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Each (recording, condition) pair with a blinded name, in a seeded random order.

    The permutation is drawn from a generator separate from the names' generator, so the
    names do not follow the pair order.
    """
    names = _blind_names(seed, len(pairs))
    permutation = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, 1]))
    ).permutation(len(pairs))
    return [
        (names[slot], pairs[int(source)][0], pairs[int(source)][1])
        for slot, source in enumerate(permutation)
    ]


def _clip_bytes(pcm: npt.NDArray[np.int16], rate: int) -> bytes:
    """A 16-bit PCM WAV with no metadata chunks written by this module."""
    buffer = io.BytesIO()
    sf.write(buffer, pcm, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _render_recording(
    benchmark: Protocol,
    protocol: ListeningProtocol,
    recording: _Recording,
    opgan: System,
    uvr: System,
    scratch: Path,
) -> tuple[dict[str, npt.NDArray[np.int16]], ClipMatch]:
    """The three 16-bit windows for one recording, and the loudness match that produced them."""
    rate = protocol.clips.sample_rate_hz
    clip_samples = round(protocol.clips.duration_s * rate)
    sides = _whole_sides(benchmark, recording, opgan, uvr, scratch, rate)
    window = _cut(sides, recording.start_s, clip_samples, rate)
    match = match_clip(window, rate, protocol.loudness)
    pcm = {
        name: to_pcm16(window[name] * match.gains[name], f"{recording.identifier}/{name}")
        for name in CONDITIONS
    }
    return pcm, match


def _stage(
    protocol: ListeningProtocol,
    benchmark: Protocol,
    recordings: Sequence[_Recording],
    opgan: System,
    uvr: System,
    scratch: Path,
    seed: int,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Every clip's file bytes by file name, and the sealed key that maps names to conditions."""
    rate = protocol.clips.sample_rate_hz
    pairs = [(r.identifier, c) for r in recordings for c in CONDITIONS]
    renders = {
        r.identifier: _render_recording(benchmark, protocol, r, opgan, uvr, scratch)
        for r in recordings
    }
    files: dict[str, bytes] = {}
    clips: list[dict[str, Any]] = []
    for blind, identifier, condition in _assign(seed, pairs):
        pcm, match = renders[identifier]
        data = _clip_bytes(pcm[condition], rate)
        files[f"{CLIPS_DIR}/{blind}.wav"] = data
        clips.append(
            {
                "blinded_id": blind,
                "identifier": identifier,
                "condition": condition,
                "gain": match.gains[condition],
                "loudness_lkfs": match.loudness[condition],
                "target_lkfs": match.target_lkfs,
                "lowered_steps": match.lowered_steps,
            }
        )
    key = {
        "study": protocol.study,
        "protocol_sha256": protocol.sha256,
        "blind_seed": seed,
        "clips": sorted(clips, key=lambda clip: clip["blinded_id"]),
    }
    return files, key


def _build_manifest(
    protocol: ListeningProtocol, files: Mapping[str, bytes], key_bytes: bytes, acquisition_sha: str
) -> bytes:
    """The public build manifest: digests of every clip file and of the sealed key. No mapping."""
    clips = [
        {
            "blinded_id": Path(name).stem,
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in sorted(files.items())
    ]
    manifest = {
        "study": protocol.study,
        "protocol_version": protocol.protocol_version,
        "protocol_sha256": protocol.sha256,
        "acquisition_sha256": acquisition_sha,
        "sample_rate_hz": protocol.clips.sample_rate_hz,
        "clip_duration_s": protocol.clips.duration_s,
        "bits": protocol.loudness.presentation_bits,
        "key_path": f"{SEALED_DIR}/{KEY_NAME}",
        "key_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "clips": clips,
    }
    return canonical_json_bytes(manifest)


def _publish(out_dir: Path, partial: Path, files: Mapping[str, bytes], build_bytes: bytes) -> None:
    """Write every file into the partial directory, the build manifest last, then rename it."""
    for name, data in files.items():
        write_atomically(partial / name, data)
    write_atomically(partial / BUILD_NAME, build_bytes)
    partial.replace(out_dir)


def _package(
    protocol: ListeningProtocol,
    benchmark: Protocol,
    acquisition: Path,
    opgan: System,
    uvr: System,
    scratch: Path,
    seed: int,
) -> tuple[dict[str, bytes], bytes]:
    """Every file to publish, by relative path, and the public build manifest that names them.

    Raises:
        BuildError: If the acquisition does not match the protocol, a staged file does not match
            its digest, an engine output is refused, or a clip would clip.
    """
    manifest = _read_manifest(protocol, acquisition)
    recordings = _recordings(acquisition, manifest)
    files, key = _stage(protocol, benchmark, recordings, opgan, uvr, scratch, seed)
    key_bytes = canonical_json_bytes(key)
    files[f"{SEALED_DIR}/{KEY_NAME}"] = key_bytes
    acquisition_sha = hashlib.sha256((acquisition / ACQUISITION_NAME).read_bytes()).hexdigest()
    return files, _build_manifest(protocol, files, key_bytes, acquisition_sha)


def build(
    protocol: ListeningProtocol,
    benchmark: Protocol,
    acquisition: Path,
    out_dir: Path,
    *,
    seed: int,
    opgan: System,
    uvr: System,
    scratch: Path,
) -> Path:
    """Build the blinded clips and the sealed key into a new directory, or nothing.

    Args:
        protocol: The listening protocol.
        benchmark: The benchmark protocol, for the decoder and resampler arguments.
        acquisition: The acquisition directory, as `run_acquire` wrote it.
        out_dir: A directory that does not exist yet.
        seed: The operator's blinding seed. It is written only into the sealed key.
        opgan: The OpGAN engine.
        uvr: The UVR engine.
        scratch: A directory outside *out_dir* for the engine inputs and outputs.

    Returns:
        The output directory.

    Raises:
        BuildError: If the directory exists, a check fails, an engine output is refused, or a
            clip would clip.
    """
    if out_dir.exists():
        raise BuildError(f"output directory {out_dir} already exists")
    if out_dir.resolve().is_relative_to(scratch.resolve()):
        raise BuildError("the output directory must not be inside the scratch directory")
    files, build_bytes = _package(protocol, benchmark, acquisition, opgan, uvr, scratch, seed)
    partial = out_dir.with_name(f".{out_dir.name}.{uuid.uuid4().hex}.partial")
    try:
        (partial / CLIPS_DIR).mkdir(parents=True)
        (partial / SEALED_DIR).mkdir()
        _publish(out_dir, partial, files, build_bytes)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    logger.info("Built %d clips in %s", len(files) - 1, out_dir)
    return out_dir
