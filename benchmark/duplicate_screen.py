"""The duplicate screen's features and similarity, as protocol v11 freezes them.

The screen compares a candidate's canonical waveform, mono at 16 kHz, with every training
recording and then with every candidate admitted earlier in the walk
(`duplicate_detection.screen_scope`). Every argument the features depend on is read from
`duplicate_detection.mfcc`:

- **Features.** MFCCs per frame, each coefficient z-normalised over the recording's frames,
  then each frame scaled to unit length, so the per-frame cosine is a dot product.
- **Offsets.** Every multiple of `alignment_stride_frames` whose absolute value is at most
  `floor(alignment_search_s * sample_rate / hop_length)` frames, including 0.
- **Similarity.** For each offset with at least `min_overlap_frames` overlapping frames, the
  mean per-frame cosine over the overlap; the best over the offsets. It is symmetric in its
  two recordings because the offsets are.

A coefficient with zero variance and a frame with zero norm are left at zero and counted in
`Degenerate`; the protocol makes a candidate with either uncertain
(`duplicate_detection.degenerate_scope`). These functions were first written for
`benchmark.measurements.training_screen`, which calibrated the screen and now imports them.

`screen` compares one candidate with every reference and reports what the walk needs: whether
the answer is uncertain, the references at or above `flag_threshold`, and the maximum
similarity against each group for the audit trail. It reads no files: the caller computes the
features and names each reference by the three values an adjudication record names it by.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import librosa
import numpy as np
import numpy.typing as npt

from benchmark.protocol import DuplicateDetection, MfccConfig

logger = logging.getLogger(__name__)

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


# The protocol closes pad_mode to "constant"; the table gives it librosa's literal type.
_PAD_MODES: dict[str, Literal["constant"]] = {"constant": "constant"}


def coefficients(waveform: npt.NDArray[np.float64], config: MfccConfig) -> npt.NDArray[np.float64]:
    """The MFCCs with every argument taken from the protocol, shaped (coefficients, frames)."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length,
        win_length=config.win_length, window=config.window, center=config.center,
        pad_mode=_PAD_MODES[config.pad_mode], power=config.power, n_mels=config.n_mels,
        fmin=config.fmin, fmax=config.fmax, htk=config.htk, norm=config.mel_norm,
        dtype=np.dtype(config.mel_basis_dtype),
    )  # fmt: skip
    decibels = librosa.power_to_db(
        mel, ref=config.db_ref, amin=config.db_amin, top_db=config.db_top_db
    )
    return np.asarray(
        librosa.feature.mfcc(
            S=decibels, n_mfcc=config.n_mfcc, dct_type=config.dct_type, norm=config.dct_norm,
            lifter=config.lifter,
        ),
        dtype=np.float64,
    )  # fmt: skip


def unit_features(
    waveform: npt.NDArray[np.float64], config: MfccConfig
) -> tuple[Features, Degenerate]:
    """Z-normalised MFCC frames scaled to unit length, shaped (frames, coefficients).

    Scaling each frame to unit length makes the per-frame cosine a dot product. A zero
    variance coefficient is left at zero and a zero-norm frame stays zero, and both are
    counted rather than divided by.
    """
    mfcc = coefficients(waveform, config)
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True, ddof=config.zscore_ddof)
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


class ScreenError(ValueError):
    """Raised when the screen is given inputs it cannot answer for."""


class Group(StrEnum):
    """The two sets a candidate is screened against, in the order the protocol considers them."""

    TRAINING = "training"
    ADMITTED = "admitted"


@dataclass(frozen=True)
class RecordingIdentity:
    """A recording as an adjudication record names it.

    `identifier` is a training recording's manifest file name or a candidate's source
    identifier (`duplicate_detection.adjudication_record_match`).
    """

    identifier: str
    raw_sha256: str
    decoded_pcm_sha256: str


@dataclass(frozen=True)
class Reference:
    """A recording the candidate is compared with, and its features."""

    identity: RecordingIdentity
    features: Features


@dataclass(frozen=True)
class Flag:
    """A reference the candidate scored at or above the flag threshold against."""

    group: Group
    identity: RecordingIdentity
    score: float


@dataclass(frozen=True)
class ScreenResult:
    """What the screen found for one candidate.

    `max_training` and `max_admitted` are the highest similarities scored against each group,
    for the audit trail; each is None when no comparison in that group was scored, which for
    `max_admitted` includes a walk that has admitted nothing yet. `unscored` names the
    references with too little overlap to score.
    """

    degenerate: Degenerate
    max_training: float | None
    max_admitted: float | None
    flagged: tuple[Flag, ...]
    unscored: tuple[RecordingIdentity, ...]

    @property
    def uncertain(self) -> bool:
        """Degenerate candidate features, or a comparison with too little overlap."""
        degenerate = self.degenerate.zero_variance_coefficients or self.degenerate.zero_norm_frames
        return bool(degenerate or self.unscored)


def _check_shape(features: Features, n_mfcc: int, what: str) -> None:
    """Refuse features that are not (frames, n_mfcc) float64."""
    if features.ndim != 2 or features.shape[1] != n_mfcc or features.dtype != np.float64:
        logger.error("Screen refused %s features shaped %s", what, features.shape)
        raise ScreenError(f"{what} features must be (frames, {n_mfcc}) float64")


def _check_references(
    training: Sequence[Reference], admitted: Sequence[Reference], n_mfcc: int
) -> None:
    """At least one training reference, every identity distinct, every shape usable.

    With no training reference the screen would flag nothing and admit every candidate, so an
    empty training set is refused rather than screened.
    """
    if not training:
        logger.error("Screen refused an empty training set")
        raise ScreenError("the screen needs the training references; none were given")
    identities = [reference.identity for reference in (*training, *admitted)]
    if len(set(identities)) != len(identities):
        logger.error("Screen refused references that repeat an identity")
        raise ScreenError("each reference must be a distinct recording")
    for reference in (*training, *admitted):
        _check_shape(reference.features, n_mfcc, reference.identity.identifier)


def screen(
    features: Features,
    degenerate: Degenerate,
    training: Sequence[Reference],
    admitted: Sequence[Reference],
    config: DuplicateDetection,
) -> ScreenResult:
    """Compare a candidate's features with every training reference, then every admission.

    Every comparison is made even when the candidate's features are degenerate, so the audit
    trail has both maxima. A score at or above `flag_threshold` flags the reference.

    Raises:
        ScreenError: If no training reference is given, two references are the same
            recording, or any features are not (frames, n_mfcc) float64.
    """
    mfcc = config.mfcc
    _check_shape(features, mfcc.n_mfcc, "candidate")
    _check_references(training, admitted, mfcc.n_mfcc)
    lags = offsets(mfcc)
    flagged: list[Flag] = []
    unscored: list[RecordingIdentity] = []
    maxima: dict[Group, float | None] = {}
    for group, references in ((Group.TRAINING, training), (Group.ADMITTED, admitted)):
        best: float | None = None
        for reference in references:
            score = similarity(features, reference.features, lags, mfcc.min_overlap_frames)
            if score is None:
                unscored.append(reference.identity)
                continue
            best = score if best is None else max(best, score)
            if score >= config.flag_threshold:
                flagged.append(Flag(group, reference.identity, score))
        maxima[group] = best
    return ScreenResult(
        degenerate=degenerate,
        max_training=maxima[Group.TRAINING],
        max_admitted=maxima[Group.ADMITTED],
        flagged=tuple(flagged),
        unscored=tuple(unscored),
    )
