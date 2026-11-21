"""The duplicate screen recalibrated on the training recordings, before the screen is frozen.

Protocol v1 calibrated the aligned-MFCC screen on training recordings only, reporting a
positives minimum of 0.8471 (n=15) and a negatives maximum of 0.0995 (n=28), with
same-session negatives up to 0.1033 (n=21). Neither the recordings in those sets nor the
code that scored them is in the repository, so an implementation of the screen cannot be
checked against those figures. This measurement recalibrates on the same kind of evidence,
the training recordings only, so that no benchmark candidate is involved.

Choices the protocol leaves open are made here explicitly, so the amendment that freezes
the screen can state them:

- **Input.** Each recording's canonical waveform from the frozen canonicalisation, already
  mono at 16 kHz, so no second resampling path is introduced.
- **MFCC.** `librosa.feature.mfcc` with the frozen `n_mfcc` and `hop_length` and every other
  argument at the pinned librosa version's defaults.
- **Normalisation.** Each coefficient z-normalised over the recording's frames, with the
  population standard deviation. Whether the population or the sample standard deviation is
  used does not change any score: it scales every coefficient by the same factor, and each
  frame is then scaled to unit length. A coefficient with zero variance and a frame vector
  with zero norm are counted and reported, not resolved by a chosen rule.
- **Offsets.** Every multiple of the frozen stride whose absolute value is at most
  `floor(alignment_search_s * sample_rate / hop_length)` frames, which includes 0.
- **Similarity.** For each offset with at least `min_overlap_frames` overlapping frames, the
  mean over the overlap of the per-frame cosine; the best over offsets. It is symmetric in
  its two recordings because the offsets are.
- **Positives.** Every training recording against three transforms of itself, applied to
  its canonical waveform: the first 3 s trimmed, a gain of -6 dB, and a round trip through
  8 kHz with the frozen resampler.
- **Negatives.** Every pair of different training recordings. The ten most similar pairs
  are printed by name rather than grouping "same session" pairs by filename.

Each training file is acquired with its manifest digest as the expected digest, so a file
that differs from the manifest stops at acquisition. The decoded-PCM digest (the third
automatic exclusion rule) is computed on the way and checked for collisions.

**Measured on the reference platform**, before the screen was frozen, on two runs with
identical output: all 145 files matched the manifest and canonicalised; no two shared a
decoded-PCM digest; no zero-variance coefficient or zero-norm frame occurred. Positives:
trim minimum 0.9856, gain minimum 1.0000, 8 kHz round trip minimum 0.7550 (the Peer Gynt
"In the Hall of the Mountain King" recording), n=145 each. Negatives: 10,440 pairs, median
0.0239, 99.9th percentile 0.1112, maximum 0.4832, none at or above the 0.50 threshold. That
maximum is a pair of files whose names identify one Brahms Symphony No. 2 movement and end
`take1` and `take2`; the next highest pair is 0.1435. The identification is from the file
names only, and the files were not listened to. If they are two takes, as the names indicate,
the corpus holds more than one performance of that movement, which the protocol's statement
that it contains one performance per work does not allow for. The protocol's same-session
and same-composer rows were not recomputed: only the ten highest pairs are printed.

Run from the repository root: `uv run --frozen python -m benchmark.measurements.training_screen`.
It reads `data/raw/`, which is not in the repository, holds one recording's arrays at a time
plus about 300 MB of features, and takes tens of minutes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt

from benchmark.canonicalisation import CanonicalWaveform, canonicalise
from benchmark.protocol import MfccConfig, Protocol, load_protocol
from benchmark.source_acquisition import ArtifactBuffered, acquire_local_artifact
from benchmark.source_decode import decode_source
from benchmark.source_inspection import InspectedSource, inspect_source

MANIFEST = Path("benchmark/manifests/training_corpus.json")
RAW = Path("data/raw")
TRIM_S = 3.0
GAIN_DB = -6.0
ROUND_TRIP_RATE = 8000
TRANSFORMS = ("trim_3_s", "gain_minus_6_db", "round_trip_8_khz")
TOP_PAIRS = 10
LOWEST_POSITIVES = 3
# The byte layout the protocol hashes decoded samples in: float64, little-endian, interleaved.
_DECODED_FORMAT = "<f8"

Features = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Degenerate:
    """Counts of the cases the normalisation leaves undefined."""

    zero_variance_coefficients: int
    zero_norm_frames: int


def offsets(config: MfccConfig) -> list[int]:
    """Every multiple of the stride within the alignment search, in frames, including 0."""
    reach = math.floor(config.alignment_search_s * config.sample_rate / config.hop_length)
    steps = reach // config.alignment_stride_frames
    return [k * config.alignment_stride_frames for k in range(-steps, steps + 1)]


def unit_features(
    waveform: npt.NDArray[np.float64], config: MfccConfig
) -> tuple[Features, Degenerate]:
    """Z-normalised MFCC frames scaled to unit length, shaped (frames, coefficients).

    Scaling each frame to unit length makes the per-frame cosine a dot product. A zero
    variance coefficient is left at zero and a zero-norm frame stays zero, and both are
    counted, so the measurement reports whether either occurs.
    """
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=config.sample_rate, n_mfcc=config.n_mfcc, hop_length=config.hop_length
    )
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True)
    flat = std[:, 0] == 0.0
    z = np.divide(mfcc - mean, std, out=np.zeros_like(mfcc), where=~flat[:, None]).T
    norms = np.linalg.norm(z, axis=1)
    zero = norms == 0.0
    unit = np.divide(z, norms[:, None], out=np.zeros_like(z), where=~zero[:, None])
    return np.ascontiguousarray(unit), Degenerate(int(flat.sum()), int(zero.sum()))


def similarity(a: Features, b: Features, lags: list[int], min_overlap: int) -> float | None:
    """The best mean per-frame cosine over the offsets, or None when no offset overlaps enough.

    At offset `o`, frame `t + o` of `a` is paired with frame `t` of `b`.
    """
    best: float | None = None
    for lag in lags:
        start_a, start_b = max(lag, 0), max(-lag, 0)
        overlap = min(len(a) - start_a, len(b) - start_b)
        if overlap < min_overlap:
            continue
        x = a[start_a : start_a + overlap]
        y = b[start_b : start_b + overlap]
        score = float(np.einsum("ij,ij->", x, y)) / overlap
        if best is None or score > best:
            best = score
    return best


def transformed(
    waveform: npt.NDArray[np.float64], name: str, protocol: Protocol
) -> npt.NDArray[np.float64]:
    """One of the three positive-set transforms of a canonical waveform."""
    rate = protocol.duplicate_detection.mfcc.sample_rate
    if name == "trim_3_s":
        return waveform[int(TRIM_S * rate) :]
    if name == "gain_minus_6_db":
        scaled: npt.NDArray[np.float64] = waveform * 10.0 ** (GAIN_DB / 20.0)
        return scaled
    down = _resampled(waveform, rate, ROUND_TRIP_RATE, protocol)
    return _resampled(down, ROUND_TRIP_RATE, rate, protocol)


def _resampled(
    waveform: npt.NDArray[np.float64], orig_sr: int, target_sr: int, protocol: Protocol
) -> npt.NDArray[np.float64]:
    """Resample with the frozen resampler's settings, the ones canonicalisation uses."""
    resample = protocol.canonicalisation.resample
    return np.asarray(
        librosa.resample(
            waveform, orig_sr=orig_sr, target_sr=target_sr, res_type=resample.res_type,
            fix=resample.fix, scale=resample.scale, axis=resample.axis,
        ),
        dtype=np.float64,
    )  # fmt: skip


def load_recording(
    path: Path, *, name: str, expected_sha256: str, protocol: Protocol
) -> tuple[str, npt.NDArray[np.float64]] | str:
    """The decoded-PCM digest and canonical waveform of one training file, or why not."""
    config = protocol.canonicalisation
    artifact = acquire_local_artifact(
        path, logical_source_id=name, config=config.decode, expected_sha256=expected_sha256
    )
    if not isinstance(artifact, ArtifactBuffered):
        return f"acquisition gave {type(artifact).__name__}: {getattr(artifact, 'reason', '')}"
    inspected = inspect_source(
        artifact, logical_source_id=name, protocol_sha256=protocol.source_sha256, config=config
    )
    if not isinstance(inspected, InspectedSource):
        return f"inspection gave {inspected.primary_reason_code}"
    decoded = decode_source(
        artifact, inspected, logical_source_id=name, protocol_sha256=protocol.source_sha256,
        config=config,
    )  # fmt: skip
    if not isinstance(decoded, np.ndarray):
        return f"decode gave {decoded.primary_reason_code}"
    decoded_sha256 = hashlib.sha256(
        np.ascontiguousarray(decoded, dtype=_DECODED_FORMAT).data
    ).hexdigest()
    canonical = canonicalise(decoded, inspected, logical_source_id=name, config=config)
    if not isinstance(canonical, CanonicalWaveform):
        return f"canonicalisation gave {canonical.reason}"
    return decoded_sha256, canonical.samples


@dataclass
class Corpus:
    """What the measurement keeps per recording: names, digests, features, positive scores."""

    names: list[str]
    decoded: dict[str, str]
    features: list[Features]
    positives: dict[str, list[float | None]]
    degenerate: Counter[str]
    failures: dict[str, str]


def measure_corpus(protocol: Protocol) -> Corpus:
    """Load every training recording, keep its features, and score its three positives."""
    config = protocol.duplicate_detection.mfcc
    lags = offsets(config)
    recordings = json.loads(MANIFEST.read_text())["recordings"]
    corpus = Corpus([], {}, [], {t: [] for t in TRANSFORMS}, Counter(), {})
    for index, entry in enumerate(recordings):
        name = entry["filename"]
        loaded = load_recording(
            RAW / name, name=name, expected_sha256=entry["sha256"], protocol=protocol
        )
        if isinstance(loaded, str):
            corpus.failures[name] = loaded
            continue
        decoded_sha256, waveform = loaded
        features, degenerate = unit_features(waveform, config)
        corpus.degenerate.update(degenerate.__dict__)
        for transform in TRANSFORMS:
            other, other_degenerate = unit_features(
                transformed(waveform, transform, protocol), config
            )
            corpus.degenerate.update(other_degenerate.__dict__)
            score = similarity(features, other, lags, config.min_overlap_frames)
            corpus.positives[transform].append(score)
        corpus.names.append(name)
        corpus.decoded[name] = decoded_sha256
        corpus.features.append(features)
        print(f"{index + 1}/{len(recordings)} {name}: {len(features)} frames", flush=True)
    return corpus


def negatives(corpus: Corpus, protocol: Protocol) -> list[tuple[float | None, str, str]]:
    """The similarity of every pair of different training recordings."""
    config = protocol.duplicate_detection.mfcc
    lags = offsets(config)
    scores = []
    for i, j in combinations(range(len(corpus.names)), 2):
        score = similarity(corpus.features[i], corpus.features[j], lags, config.min_overlap_frames)
        scores.append((score, corpus.names[i], corpus.names[j]))
    return scores


def report(corpus: Corpus, pairs: list[tuple[float | None, str, str]], protocol: Protocol) -> None:
    """Print the calibration beside protocol v1's, and every fact the amendment will need."""
    screen = protocol.duplicate_detection
    print(f"\nrecordings loaded: {len(corpus.names)}; failed: {len(corpus.failures)}")
    for name, why in corpus.failures.items():
        print(f"  failed {name}: {why}")
    collisions = [d for d, n in Counter(corpus.decoded.values()).items() if n > 1]
    print(f"decoded-PCM digest collisions: {len(collisions)}")
    print(f"degenerate cases: {dict(corpus.degenerate)}")
    for transform, scores in corpus.positives.items():
        known = [s for s in scores if s is not None]
        unscored = len(scores) - len(known)
        print(f"positives {transform}: n={len(scores)}, unscored={unscored}, min={min(known):.4f}")
        lowest = sorted((s, n) for s, n in zip(scores, corpus.names, strict=True) if s is not None)
        for score, name in lowest[:LOWEST_POSITIVES]:
            print(f"  {score:.4f}  {name}")
    scored = sorted(((s, a, b) for s, a, b in pairs if s is not None), reverse=True)
    values = np.array([p[0] for p in scored])
    print(f"negatives: n={len(pairs)}, unscored={len(pairs) - len(scored)}, max={values.max():.4f}")
    print(
        "negatives quantiles 0.5/0.99/0.999: "
        + " ".join(f"{q:.4f}" for q in np.quantile(values, [0.5, 0.99, 0.999]))
    )
    print(
        f"negatives at or above the {screen.flag_threshold} threshold: {int((values >= screen.flag_threshold).sum())}"
    )
    print(f"top {TOP_PAIRS} negative pairs:")
    for score, first, second in scored[:TOP_PAIRS]:
        print(f"  {score:.4f}  {first}  |  {second}")
    cal = screen.calibration
    print(
        f"protocol v1: positives min {cal.positives_min} (n={cal.positives_n}), negatives max "
        f"{cal.negatives_max} (n={cal.negatives_n}), same-session max "
        f"{cal.same_session_negatives_max} (n={cal.same_session_negatives_n})"
    )


def main() -> None:
    """Load and score the training corpus, then every negative pair, and print the result."""
    protocol = load_protocol()
    start = time.perf_counter()
    corpus = measure_corpus(protocol)
    print(f"corpus measured in {time.perf_counter() - start:.0f} s", flush=True)
    pairs = negatives(corpus, protocol)
    print(f"negatives measured in {time.perf_counter() - start:.0f} s", flush=True)
    report(corpus, pairs, protocol)


if __name__ == "__main__":
    main()
