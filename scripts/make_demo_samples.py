r"""Generate the demo samples from canonical recordings, with OpGAN and UVR restorations.

Each clip is a slice of one canonical recording (raw little-endian float64 at 16 kHz, no
header). The slice is degraded at training level by benchmark.degradation, restored by the
product's OpGAN and UVR engines, trimmed to the shortest of the four streams, and written as
16-bit mono WAV files. samples/MANIFEST.json records each clip's source, seed, gain, trimming,
and the SHA-256 of every file written.

Every clip is generated and checked before any file is written, and the manifest is written
last. A refused clip therefore leaves no sample and no manifest behind. Each file is written
atomically through benchmark.atomic_write. Every stream is quantised to 16 bits, including the
restored streams. The engines already wrote those at 16 bits, and requantising them changes
32,767 of the 65,536 possible 16-bit values, each by one least significant bit (measured on
every value).

Every clip is degraded with the same declared noise profile, DEMO_PROFILE, which is checked
against the protocol's training ranges before any clip is generated. The profile is fixed rather
than drawn, so the demo sounds the same on every clip. The seed drives only the noise samples
(hiss and crackle positions), through one np.random.Generator per clip seeded from
np.random.SeedSequence(seed). These are demo clips, not benchmark frames, so they do not use the
benchmark's frame_seed policy.

Example:
    PYTHONPATH=backend uv run --frozen python -m scripts.make_demo_samples \\
        --clips clips.json --uvr-model-dir /path/to/uvr/models
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
from app.config import settings

from benchmark.atomic_write import write_atomically
from benchmark.canonical_worker import SAMPLE_FORMAT
from benchmark.degradation import DegradationParams, DegradedFrame, apply_degradation
from benchmark.frame_scoring import reference_frame
from benchmark.protocol import Protocol, TrainingNoiseRanges, load_protocol
from benchmark.run_preconditions import uvr_file_failures
from benchmark.run_systems import build_production_systems
from benchmark.systems_stage import System

__all__ = ["ClipSpec", "DemoSampleError", "generate", "load_clip_specs", "main"]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "samples"
SAMPLE_RATE = 16_000
PEAK_LIMIT = 1.0
PCM16_SCALE = 32767
# The benchmark writes its engine inputs as 32-bit float. A 16-bit value k is written as k / 2**15,
# which is exact in float32, so the engine reads the same samples that are shipped.
_FLOAT_INPUT_SCALE = 2**15
MANIFEST_NAME = "MANIFEST.json"
OPGAN_DIR = "opgan"
UVR_DIR = "uvr"
STREAMS = ("clean", "noisy", "opgan", "uvr")
SLUG_PATTERN = re.compile(r"[a-z0-9_]+")
CLIP_KEYS = frozenset(
    {"canonical_path", "identifier", "performer", "slug", "start_s", "duration_s", "seed"}
)
NOISE_POLICY = "fixed demo profile within the training ranges; per-clip seed for noise samples only"
# The noise every clip receives. Each value lies inside the protocol's training ranges, which
# validate_profile checks before generation starts.
DEMO_PROFILE = DegradationParams(
    hiss_snr_db=10.0,
    hiss_color="pink",
    rolloff_hz=5000.0,
    enable_crackle=True,
    crackle_density=0.001,
    crackle_amplitude=0.05,
    enable_hum=True,
    hum_amplitude=0.02,
    hum_freq=60.0,
    enable_saturation=False,
    saturation_drive=0.1,
)
_FLOAT64_BYTES = np.dtype(SAMPLE_FORMAT).itemsize


class DemoSampleError(Exception):
    """Raised when a clip, a generated stream, or an engine output cannot be used."""


@dataclass(frozen=True)
class ClipSpec:
    """One clip to generate, as listed in the clips file."""

    canonical_path: Path
    identifier: str
    performer: str
    slug: str
    start_s: float
    duration_s: float
    seed: int


@dataclass(frozen=True)
class Recording:
    """A canonical recording read once, with the facts its clips need."""

    samples: npt.NDArray[np.float64]
    sha256: str
    peak: float


@dataclass(frozen=True)
class ClipOutput:
    """One generated clip: its manifest facts and the bytes of each file it writes."""

    spec: ClipSpec
    canonical_sha256: str
    gain: float
    samples_written: int
    samples_trimmed: dict[str, int]
    files: dict[str, bytes]


def validate_slug(slug: str) -> None:
    """Refuse a slug that is not lowercase letters, digits, and underscores in full.

    Args:
        slug: The clip's output name.

    Raises:
        DemoSampleError: If the slug does not match `[a-z0-9_]+` as a whole string.
    """
    if SLUG_PATTERN.fullmatch(slug) is None:
        logger.error("Refusing slug %r: it must match [a-z0-9_]+", slug)
        raise DemoSampleError(f"slug {slug!r} must match [a-z0-9_]+")


def seconds_to_samples(seconds: float, name: str) -> int:
    """Convert seconds to a whole number of 16 kHz samples.

    Args:
        seconds: A finite duration or offset in seconds.
        name: The clip key the value came from, for the error message.

    Returns:
        The number of samples.

    Raises:
        DemoSampleError: If the value is not finite or is not a whole number of samples.
    """
    if not math.isfinite(seconds):
        logger.error("Refusing %s: %r is not finite", name, seconds)
        raise DemoSampleError(f"{name} must be finite, got {seconds!r}")
    exact = seconds * SAMPLE_RATE
    whole = round(exact)
    if not math.isclose(whole, exact, abs_tol=1e-6):
        logger.error("Refusing %s: %r is not a whole number of samples", name, seconds)
        raise DemoSampleError(f"{name} must be a whole number of samples, got {seconds!r}")
    return whole


def validate_clip(spec: ClipSpec) -> None:
    """Check a clip's own fields, independently of any recording.

    Args:
        spec: The clip to check.

    Raises:
        DemoSampleError: If the slug, seed, or durations are outside their domains.
    """
    validate_slug(spec.slug)
    if spec.seed < 0:
        logger.error("Refusing clip %s: seed %d is negative", spec.slug, spec.seed)
        raise DemoSampleError(f"clip {spec.slug} has a negative seed")
    if seconds_to_samples(spec.start_s, "start_s") < 0:
        logger.error("Refusing clip %s: start_s %r is negative", spec.slug, spec.start_s)
        raise DemoSampleError(f"clip {spec.slug} has a negative start_s")
    if seconds_to_samples(spec.duration_s, "duration_s") <= 0:
        logger.error("Refusing clip %s: duration_s %r is not positive", spec.slug, spec.duration_s)
        raise DemoSampleError(f"clip {spec.slug} has a non-positive duration_s")


def _parse_clip(item: object, index: int) -> ClipSpec:
    """Build one clip from its JSON object, refusing a missing or extra key."""
    if not isinstance(item, dict) or set(item) != CLIP_KEYS:
        logger.error("Clip %d must have exactly the keys %s", index, sorted(CLIP_KEYS))
        raise DemoSampleError(f"clip {index} must have exactly the keys {sorted(CLIP_KEYS)}")
    try:
        spec = ClipSpec(
            canonical_path=Path(str(item["canonical_path"])),
            identifier=str(item["identifier"]),
            performer=str(item["performer"]),
            slug=str(item["slug"]),
            start_s=float(item["start_s"]),
            duration_s=float(item["duration_s"]),
            seed=int(item["seed"]),
        )
    except (TypeError, ValueError) as error:
        logger.error("Clip %d has a value of the wrong type: %s", index, error)
        raise DemoSampleError(f"clip {index} has a value of the wrong type") from error
    validate_clip(spec)
    return spec


def load_clip_specs(path: Path) -> list[ClipSpec]:
    """Read the clips file: a JSON list of clip objects.

    Args:
        path: The clips file.

    Returns:
        The clips, in file order.

    Raises:
        DemoSampleError: If the file is unreadable, is not a JSON list, or a clip is invalid.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.error("Clips file %s cannot be read as JSON: %s", path, error)
        raise DemoSampleError(f"clips file {path} cannot be read as JSON") from error
    if not isinstance(raw, list):
        logger.error("Clips file %s must hold a JSON list", path)
        raise DemoSampleError(f"clips file {path} must hold a JSON list")
    return [_parse_clip(item, index) for index, item in enumerate(raw)]


def check_clip_list(specs: Sequence[ClipSpec]) -> None:
    """Refuse an empty clip list, an invalid clip, or two clips with the same slug.

    Args:
        specs: The clips to generate.

    Raises:
        DemoSampleError: If the list is empty, a clip is invalid, or a slug repeats.
    """
    if not specs:
        logger.error("No clips to generate")
        raise DemoSampleError("no clips to generate")
    seen: set[str] = set()
    for spec in specs:
        validate_clip(spec)
        if spec.slug in seen:
            logger.error("Slug %s is used by more than one clip", spec.slug)
            raise DemoSampleError(f"slug {spec.slug} is used by more than one clip")
        seen.add(spec.slug)


def read_recording(path: Path) -> Recording:
    """Read a canonical recording's samples, its SHA-256, and its whole-recording peak.

    Args:
        path: A raw little-endian float64 file with no header.

    Returns:
        The recording.

    Raises:
        DemoSampleError: If the file is empty, is not a whole number of float64 values, or has
            no finite, positive peak.
    """
    data = path.read_bytes()
    if not data or len(data) % _FLOAT64_BYTES:
        logger.error("Canonical file %s is not a whole number of float64 values", path)
        raise DemoSampleError(f"canonical file {path} is not a whole number of float64 values")
    samples = np.asarray(np.frombuffer(data, dtype=SAMPLE_FORMAT), dtype=np.float64)
    peak = float(np.max(np.abs(samples)))
    if not (math.isfinite(peak) and peak > 0):
        logger.error("Canonical file %s has no finite positive peak", path)
        raise DemoSampleError(f"canonical file {path} has no finite positive peak")
    return Recording(samples=samples, sha256=hashlib.sha256(data).hexdigest(), peak=peak)


def clip_frame(samples: npt.NDArray[np.float64], spec: ClipSpec) -> npt.NDArray[np.float64]:
    """Cut the clip's samples from its recording.

    Args:
        samples: The whole recording.
        spec: The clip, whose start and duration are in seconds.

    Returns:
        The samples in [start, start + duration).

    Raises:
        DemoSampleError: If the clip starts before the recording or runs past its end.
    """
    start = seconds_to_samples(spec.start_s, "start_s")
    length = seconds_to_samples(spec.duration_s, "duration_s")
    if start < 0 or start + length > samples.size:
        logger.error(
            "Clip %s covers samples [%d, %d) of a recording of %d samples",
            spec.slug,
            start,
            start + length,
            samples.size,
        )
        raise DemoSampleError(f"clip {spec.slug} runs outside its recording")
    return samples[start : start + length]


def validate_profile(profile: DegradationParams, ranges: TrainingNoiseRanges) -> None:
    """Refuse a noise profile with any value outside the protocol's training ranges.

    Args:
        profile: The noise profile to check.
        ranges: The protocol's training noise ranges.

    Raises:
        DemoSampleError: If a numeric value is outside its inclusive range, the colour is not
            one of the allowed colours, or the hum frequency is not one of the allowed ones.
    """
    bounded = {
        "hiss_snr_db": (profile.hiss_snr_db, ranges.hiss_snr_db),
        "rolloff_hz": (profile.rolloff_hz, ranges.rolloff_hz),
        "crackle_density": (profile.crackle_density, ranges.crackle_density),
        "crackle_amplitude": (profile.crackle_amplitude, ranges.crackle_amplitude),
        "hum_amplitude": (profile.hum_amplitude, ranges.hum_amplitude),
        "saturation_drive": (profile.saturation_drive, ranges.saturation_drive),
    }
    for name, (value, (low, high)) in bounded.items():
        if not low <= value <= high:
            logger.error(
                "Noise profile %s=%r is outside the training range [%r, %r]", name, value, low, high
            )
            raise DemoSampleError(f"noise profile {name}={value!r} is outside the training range")
    if profile.hiss_color not in ranges.hiss_colors:
        logger.error("Noise profile colour %r is not a training colour", profile.hiss_color)
        raise DemoSampleError(
            f"noise profile colour {profile.hiss_color!r} is not a training colour"
        )
    if profile.hum_freq not in ranges.hum_freqs:
        logger.error("Noise profile hum_freq %r is not a training frequency", profile.hum_freq)
        raise DemoSampleError(
            f"noise profile hum_freq {profile.hum_freq!r} is not a training frequency"
        )


def degrade_clip(
    frame: npt.NDArray[np.float64],
    recording_peak: float,
    seed: int,
    ranges: TrainingNoiseRanges,
    profile: DegradationParams = DEMO_PROFILE,
) -> DegradedFrame:
    """Degrade a clip at training level with the declared profile.

    The gain is headroom over the whole recording's peak, as the benchmark sets training
    level. The seed drives only the noise samples, through one generator per clip.

    Args:
        frame: The clip's canonical samples.
        recording_peak: The largest absolute sample of the whole recording.
        seed: The clip's seed, for the noise samples.
        ranges: The protocol's training noise ranges.
        profile: The noise profile, the same for every clip.

    Returns:
        The degraded clip, the gain that set its level, and the profile used.
    """
    gain = ranges.headroom / recording_peak
    scaled = (frame * gain).astype(np.float32)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    degraded = apply_degradation(scaled, SAMPLE_RATE, profile, rng)
    return DegradedFrame(degraded=degraded, gain=gain, params=profile)


def to_pcm16(samples: npt.NDArray[np.float64], label: str) -> npt.NDArray[np.int16]:
    """Quantise samples to 16-bit integers, refusing any sample outside [-1, 1].

    Args:
        samples: The stream's samples.
        label: The stream's name, for the error message.

    Returns:
        The 16-bit samples.

    Raises:
        DemoSampleError: If a sample is not finite or its absolute value exceeds 1.0. The
            stream is refused rather than clipped.
    """
    if not np.all(np.isfinite(samples)):
        logger.error("Stream %s holds a non-finite sample", label)
        raise DemoSampleError(f"stream {label} holds a non-finite sample")
    peak = float(np.max(np.abs(samples), initial=0.0))
    if peak > PEAK_LIMIT:
        logger.error(
            "Stream %s peaks at %.6f, above %.1f; refusing rather than clipping",
            label,
            peak,
            PEAK_LIMIT,
        )
        raise DemoSampleError(f"stream {label} peaks above {PEAK_LIMIT}")
    return np.round(samples * PCM16_SCALE).astype(np.int16)


def wav_bytes(pcm: npt.NDArray[np.int16]) -> bytes:
    """Encode 16-bit mono samples at the canonical rate as a PCM_16 WAV file."""
    buffer = io.BytesIO()
    sf.write(buffer, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def float_input_bytes(pcm: npt.NDArray[np.int16]) -> bytes:
    """Encode the noisy stream as the benchmark's FLOAT input, with the shipped values."""
    buffer = io.BytesIO()
    values = pcm.astype(np.float32) / _FLOAT_INPUT_SCALE
    sf.write(buffer, values, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def read_engine_output(path: Path, label: str) -> npt.NDArray[np.float64]:
    """Read an engine's output as float64 mono at the canonical rate.

    Args:
        path: The engine's output WAV.
        label: The engine and clip, for the error message.

    Returns:
        One channel of samples.

    Raises:
        DemoSampleError: If the output is not mono at the canonical rate.
    """
    data, rate = sf.read(str(path), dtype="float64", always_2d=True)
    if rate != SAMPLE_RATE or data.shape[1] != 1:
        logger.error(
            "Output %s for %s is %d Hz with %d channels; expected %d Hz mono",
            path.name,
            label,
            rate,
            data.shape[1],
            SAMPLE_RATE,
        )
        raise DemoSampleError(f"output for {label} is not {SAMPLE_RATE} Hz mono")
    return np.asarray(data[:, 0], dtype=np.float64)


def run_denoisers(
    slug: str,
    noisy_pcm: npt.NDArray[np.int16],
    workdir: Path,
    opgan: System,
    uvr: System,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Give the noisy stream to both engines and read back what each returned.

    Args:
        slug: The clip's name, used for the input file.
        noisy_pcm: The full noisy stream as 16-bit samples.
        workdir: A scratch directory for the input file.
        opgan: The OpGAN system.
        uvr: The UVR system.

    Returns:
        The OpGAN output and the UVR output, as float64 mono samples.
    """
    noisy_input = workdir / f"{slug}_noisy_input.wav"
    write_atomically(noisy_input, float_input_bytes(noisy_pcm))
    opgan_out = read_engine_output(opgan.denoise(noisy_input), f"{slug} OpGAN")
    uvr_out = read_engine_output(uvr.denoise(noisy_input), f"{slug} UVR")
    return opgan_out, uvr_out


def trim_streams(
    streams: dict[str, npt.NDArray[np.float64]],
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, int]]:
    """Trim every stream to the shortest one, from the end.

    Args:
        streams: The four streams, keyed by name.

    Returns:
        The trimmed streams and the number of samples removed from each.

    Raises:
        DemoSampleError: If the shortest stream is empty.
    """
    shortest = min(stream.size for stream in streams.values())
    if shortest == 0:
        logger.error("A stream is empty; nothing to write")
        raise DemoSampleError("a stream is empty after an engine returned it")
    trimmed = {name: stream[:shortest] for name, stream in streams.items()}
    removed = {name: stream.size - shortest for name, stream in streams.items()}
    return trimmed, removed


def encode_files(slug: str, pcm: dict[str, npt.NDArray[np.int16]]) -> dict[str, bytes]:
    """Name and encode one clip's six WAV files, keyed by their path under the output directory."""
    files: dict[str, bytes] = {}
    for directory in (OPGAN_DIR, UVR_DIR):
        files[f"{directory}/{slug}_clean.wav"] = wav_bytes(pcm["clean"])
        files[f"{directory}/{slug}_noisy.wav"] = wav_bytes(pcm["noisy"])
    files[f"{OPGAN_DIR}/{slug}_restored.wav"] = wav_bytes(pcm["opgan"])
    files[f"{UVR_DIR}/{slug}_restored.wav"] = wav_bytes(pcm["uvr"])
    return files


def generate_clip(
    spec: ClipSpec,
    recording: Recording,
    ranges: TrainingNoiseRanges,
    engines: tuple[System, System],
    workdir: Path,
    profile: DegradationParams = DEMO_PROFILE,
) -> ClipOutput:
    """Generate one clip: degrade it, restore it with both engines, trim, and encode it.

    Args:
        spec: The clip.
        recording: The canonical recording the clip is cut from.
        ranges: The protocol's training noise ranges.
        engines: The OpGAN system and the UVR system, in that order.
        workdir: A scratch directory for the engines' input and output.
        profile: The noise profile the clip is degraded with.

    Returns:
        The clip's manifest facts and the bytes of its files.

    Raises:
        DemoSampleError: If the clip is out of range or any stream is refused.
    """
    opgan, uvr = engines
    frame = clip_frame(recording.samples, spec)
    degraded = degrade_clip(frame, recording.peak, spec.seed, ranges, profile)
    noisy = np.asarray(degraded.degraded, dtype=np.float64)
    noisy_pcm = to_pcm16(noisy, f"{spec.slug} noisy")
    opgan_out, uvr_out = run_denoisers(spec.slug, noisy_pcm, workdir, opgan, uvr)
    streams = {
        "clean": reference_frame(frame, degraded.gain),
        "noisy": noisy,
        "opgan": opgan_out,
        "uvr": uvr_out,
    }
    trimmed, removed = trim_streams(streams)
    for name in STREAMS:
        logger.info("Trimmed %d samples from %s stream of clip %s", removed[name], name, spec.slug)
    pcm = {name: to_pcm16(stream, f"{spec.slug} {name}") for name, stream in trimmed.items()}
    return ClipOutput(
        spec=spec,
        canonical_sha256=recording.sha256,
        gain=degraded.gain,
        samples_written=int(pcm["clean"].size),
        samples_trimmed=removed,
        files=encode_files(spec.slug, pcm),
    )


def write_outputs(clips: Sequence[ClipOutput], out_dir: Path) -> None:
    """Write every clip's files, each atomically, under the output directory.

    Args:
        clips: The generated clips.
        out_dir: The output directory.
    """
    for clip in clips:
        for relative, data in clip.files.items():
            target = out_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            write_atomically(target, data)
            logger.info("Wrote %s", target)


def _clip_record(clip: ClipOutput) -> dict[str, object]:
    """One clip's manifest entry, with the SHA-256 of each file it wrote."""
    spec = clip.spec
    files = [
        {"path": path, "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in sorted(clip.files.items())
    ]
    return {
        "canonical_sha256": clip.canonical_sha256,
        "duration_s": spec.duration_s,
        "files": files,
        "gain": clip.gain,
        "identifier": spec.identifier,
        "performer": spec.performer,
        "samples_trimmed": clip.samples_trimmed,
        "samples_written": clip.samples_written,
        "seed": spec.seed,
        "slug": spec.slug,
        "start_s": spec.start_s,
    }


def manifest_bytes(
    clips: Sequence[ClipOutput], protocol_version: int, profile: DegradationParams
) -> bytes:
    """Encode the manifest as sorted, indented JSON with no timestamps or absolute paths.

    Args:
        clips: The generated clips.
        protocol_version: The benchmark protocol version the training ranges came from.
        profile: The noise profile every clip used.

    Returns:
        The manifest's bytes.
    """
    body = {
        "clips": [_clip_record(clip) for clip in clips],
        "noise_policy": NOISE_POLICY,
        "noise_profile": asdict(profile),
        "protocol_version": protocol_version,
        "sample_rate": SAMPLE_RATE,
    }
    return (json.dumps(body, sort_keys=True, indent=2) + "\n").encode("utf-8")


def generate(
    specs: Sequence[ClipSpec],
    opgan: System,
    uvr: System,
    ranges: TrainingNoiseRanges,
    protocol_version: int,
    workdir: Path,
    out_dir: Path,
    profile: DegradationParams = DEMO_PROFILE,
) -> Path:
    """Generate every clip, then write the sample files and, last, the manifest.

    Args:
        specs: The clips to generate.
        opgan: The OpGAN system.
        uvr: The UVR system.
        ranges: The protocol's training noise ranges.
        protocol_version: The protocol version recorded in the manifest.
        workdir: A scratch directory for the engines.
        out_dir: The output directory, which is created if needed.
        profile: The noise profile every clip uses, checked against the training ranges.

    Returns:
        The manifest's path.

    Raises:
        DemoSampleError: If the profile or a clip is invalid, or any clip is refused. Nothing is
            written then.
    """
    check_clip_list(specs)
    validate_profile(profile, ranges)
    logger.info("Noise profile for every clip: %s", profile)
    recordings = {path: read_recording(path) for path in {s.canonical_path for s in specs}}
    clips = [
        generate_clip(spec, recordings[spec.canonical_path], ranges, (opgan, uvr), workdir, profile)
        for spec in specs
    ]
    write_outputs(clips, out_dir)
    manifest = out_dir / MANIFEST_NAME
    write_atomically(manifest, manifest_bytes(clips, protocol_version, profile))
    logger.info("Wrote %d clips and %s", len(clips), manifest)
    return manifest


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description="Generate the demo samples.")
    parser.add_argument("--clips", type=Path, required=True, help="JSON list of clip specs")
    parser.add_argument("--uvr-model-dir", type=Path, required=True)
    parser.add_argument("--opgan-checkpoint", type=Path, default=settings.opgan_checkpoint)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def _check_uvr_models(protocol: Protocol, uvr_model_dir: Path) -> None:
    """Refuse a UVR model directory whose files fail their pinned digests."""
    failures = uvr_file_failures(protocol, uvr_model_dir)
    if failures:
        logger.error("UVR model files in %s fail their digests: %s", uvr_model_dir, failures)
        raise DemoSampleError(f"UVR model files in {uvr_model_dir} fail their digests")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generator from the command line.

    Args:
        argv: The arguments, or None for the process's own.

    Returns:
        0 on success, 1 when a clip or stream is refused.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        specs = load_clip_specs(args.clips)
        protocol = load_protocol()
        _check_uvr_models(protocol, args.uvr_model_dir)
        with tempfile.TemporaryDirectory(prefix="demo-samples-") as tmp:
            workdir = Path(tmp)
            opgan, uvr = build_production_systems(
                protocol, workdir, args.opgan_checkpoint, args.uvr_model_dir
            )
            generate(
                specs,
                opgan,
                uvr,
                protocol.degradation.ranges,
                protocol.protocol_version,
                workdir,
                args.out_dir,
            )
    except DemoSampleError as error:
        print(f"make_demo_samples: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
