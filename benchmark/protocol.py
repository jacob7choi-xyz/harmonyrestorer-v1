"""Immutable, validated access to the frozen benchmark protocol.

Two things share the authority here, with different jobs. `benchmark/protocols/v3.json`
holds the frozen values a run executes against; its digest is what establishes *which*
values those are. This module defines what counts as a *well-formed* protocol, which is
why the gate set, the supported version, the transform vocabulary, and the
direction-to-effect mapping live in code rather than in the file they validate: a
config cannot be the sole judge of its own validity.

The division is deliberate, and the line is whether a value expresses a kind or a
magnitude. Kinds are pinned in code: periodic rather than symmetric, additive rather
than clamped, backward rather than orthonormal, RMS before mean, amplitude dB rather
than power dB. Magnitudes stay in the config alone: the seed, the partition sizes, the
transform lengths, the offset. Copying those into code would create a second authority
for the same fact, and the file digest already pins them. Where a factor follows from a
kind, such as the 20 in amplitude dB, it is carried by the operation's name and is not a
setting at all. Validation therefore answers "is this well formed", never "is this
the approved experiment", which is the question `load_protocol` answers by refusing to
take a path at all.

Superseded versions stay in the tree rather than being replaced, so an artifact
recording an older `protocol_version` remains interpretable without archaeology
through git history. The current version names its predecessor by digest, and the
loader recomputes those digests, so the chain is checked rather than asserted.

`load_protocol` is the only supported way for production code to reach either. The
config path is private and a test flags direct reads, so a bypass has to be written
deliberately rather than reached for by accident. Python cannot make it impossible,
so code review remains part of this boundary.

Three properties make the loaded object load-bearing rather than descriptive:

Unknown and missing keys are rejected, so a typo becomes a load failure instead of a
silently ignored setting. Semantic relationships are validated, so a config that
parses but states an impossible protocol cannot be loaded: a duplicate threshold
sitting outside its own calibration evidence, a paired effect whose sign contradicts
its metric direction, a bootstrap resampling frames rather than recordings. The
runtime is checked against the pinned NumPy version and BitGenerator, because a
reproducibility guarantee recorded in JSON but never enforced is documentation.

The returned `Protocol` is deeply immutable and carries the SHA-256 of the exact
bytes it was parsed from. That digest is a file-identity claim and nothing more:
these bytes governed this execution. It does not assert that two byte-distinct but
whitespace-equivalent configs differ in meaning.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

# Private so that production code has no supported affordance for reading the file
# itself. `load_protocol` is the way in.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL_PATH = _REPO_ROOT / "benchmark" / "protocols" / "v3.json"

SUPPORTED_PROTOCOL_VERSION = 3

# Which protocol v2 supersedes, and the artifacts it supersedes, as historical facts
# rather than as arithmetic. `SUPPORTED_PROTOCOL_VERSION - 1` would encode a universal
# rule that every version amends its immediate predecessor, which is not something the
# governance model promises. The paths are frozen here so the config cannot redirect
# the amendment chain at another file whose digest it also records, and so a relative
# path out of the tree is not reachable at all.
_PREDECESSOR_VERSION = 2
_PREDECESSOR_PROTOCOL_PATH = "benchmark/protocols/v2.json"
_PREDECESSOR_DOCUMENT_PATH = "docs/benchmark-protocol-v2.md"

# Closed vocabularies for the frozen LSD transform. Each admits exactly one value. The
# fields exist so an independent implementer reads each convention from the
# config instead of inheriting a library default, and so code branches on an enum
# rather than parsing an equation string.
_WINDOW_KIND = "hann"
_WINDOW_SYMMETRY = "periodic"
_FFT_KIND = "rfft"
_FFT_NORM = "backward"
_PAD_MODE = "constant"
_FRAME_STARTS = "multiples_of_hop_from_zero"
# Names the amplitude convention rather than exposing its factor as a setting. The 20
# in 20*log10 distinguishes amplitude spectra from power spectra, so it belongs to the
# kind of measurement, not to the list of numbers a config may choose. Storing it
# separately would put the same authoritative value in two places.
_LOG_MAGNITUDE_OPERATION = "amplitude_db_additive_offset"

# What SI-SNR does with a constant reference, which is exactly the case where the centred
# reference has no energy in exact arithmetic. The only value v3 permits is to invalidate
# the item, because the alternative is letting epsilon manufacture a score for an
# undefined metric. Declared rather than implied so an independent implementation reading
# the config alone reaches the same population.
_CONSTANT_REFERENCE = "invalid_item"
_BIN_REDUCTION = "rms"
_FRAME_REDUCTION = "arithmetic_mean"

# A positive paired effect must favour OpGAN for every metric, so the effect
# expression follows from the metric's direction instead of being chosen per metric.
# Decoupling them is how a sign error inverts a published conclusion.
EFFECT_FOR_DIRECTION = MappingProxyType(
    {"higher_better": "opgan_minus_uvr", "lower_better": "uvr_minus_opgan"}
)

# The exact publication gates of the current protocol. Neither removals nor additions are
# permitted without a version bump. Dropping a gate obviously weakens the run, but
# adding one is not automatically safer: a gate invented once results exist decides
# which results get published, which is the preregistration failure this whole
# protocol exists to prevent.
FROZEN_GATES = frozenset(
    {
        "loaded_protocol_hash_matches_artifact",
        "candidate_universe_hash_verified",
        "training_performance_screen_complete",
        "population_ids_equal_across_systems",
        "frame_population_ids_equal_across_systems",
        "coverage_opgan_complete",
        "coverage_uvr_complete",
    }
)

_ROOT_KEYS = frozenset(
    {
        "protocol_version",
        "protocol_document",
        "note",
        "amends",
        "environment",
        "selection",
        "eligibility",
        "duplicate_detection",
        "conditioning",
        "metrics",
        "reporting",
        "aggregation",
        "publication_gates",
    }
)


class ProtocolError(Exception):
    """Base exception for protocol loading failures."""


class ProtocolSchemaError(ProtocolError):
    """Raised when the config's shape does not match the expected schema."""


class ProtocolValueError(ProtocolError):
    """Raised when the config parses but states an invalid protocol."""


class ProtocolEnvironmentError(ProtocolError):
    """Raised when the runtime differs from the environment the protocol pins."""


@dataclass(frozen=True)
class Environment:
    """The runtime the protocol's reproducibility guarantee is scoped to."""

    numpy_version: str
    bit_generator: str


@dataclass(frozen=True)
class Selection:
    """Deterministic admission of candidate recordings into partitions."""

    seed: int
    partitions: Mapping[str, int]
    acquisition_retries: int
    permanent_failure_codes: tuple[int, ...]


@dataclass(frozen=True)
class Eligibility:
    """Floors a recording and its frames must clear to be scored."""

    recording_min_duration_s: float
    recording_rms_dbfs_min: float
    frame_rms_dbfs_min: float
    rms_definition: str


@dataclass(frozen=True)
class MfccConfig:
    """Feature configuration for the performance-identity screen."""

    sample_rate: int
    n_mfcc: int
    hop_length: int
    normalization: str
    alignment_search_s: float
    alignment_stride_frames: int
    min_overlap_frames: int
    similarity: str


@dataclass(frozen=True)
class MfccCalibration:
    """Observed separation between known-duplicate and known-distinct pairs."""

    positives_min: float
    positives_n: int
    negatives_max: float
    negatives_n: int
    same_session_negatives_max: float
    same_session_negatives_n: int
    untested_case: str


@dataclass(frozen=True)
class DuplicateDetection:
    """Screening that keeps a training performance out of the evaluation set."""

    flag_threshold: float
    action_on_flag: str
    action_on_uncertain: str
    mfcc: MfccConfig
    calibration: MfccCalibration


@dataclass(frozen=True)
class Conditioning:
    """Rules for making a system's output comparable to the clean reference."""

    sample_rate: int
    channels: str
    max_length_difference_samples: int
    length_difference_provenance: str
    alignment_search: bool


@dataclass(frozen=True)
class ReconstructionSnr:
    """Sample-wise fidelity; penalises gain, timing, and distortion alike."""

    direction: str
    residual_floor: float
    cap_db: float
    paired_effect: str


@dataclass(frozen=True)
class SiSnr:
    """Fidelity with global gain projected out."""

    direction: str
    zero_mean: bool
    epsilon: float
    cap_db: float
    paired_effect: str
    constant_reference: str
    domain_note: str


@dataclass(frozen=True)
class WindowSpec:
    """Analysis window, defined by equation rather than by library function name."""

    kind: str
    symmetry: str
    note: str


@dataclass(frozen=True)
class FftSpec:
    """Transform and its normalisation convention.

    Normalisation is result-sensitive for this metric: the additive log-magnitude
    offset stops a uniform scaling from cancelling between the two spectra.
    """

    kind: str
    norm: str
    note: str


@dataclass(frozen=True)
class FramingSpec:
    """Padding and frame placement, stated per side so 'width' cannot mean total."""

    center: bool
    pad_mode: str
    pad_value: float
    pad_left_samples: int
    pad_right_samples: int
    frame_starts: str
    partial_trailing_frame: bool


@dataclass(frozen=True)
class LogMagnitudeSpec:
    """The log operation, as a named operation rather than a parsable equation.

    The dB factor is not a field. It follows from the named operation, so amplitude
    and power conventions are distinguished by kind instead of by a number a config
    could set to either.
    """

    operation: str
    offset: float
    note: str


@dataclass(frozen=True)
class ReductionSpec:
    """Order and kind of the two aggregation steps, bins first then frames."""

    bins: str
    frames: str


@dataclass(frozen=True)
class LogSpectralDistance:
    """Spectral magnitude fidelity; blind to phase."""

    direction: str
    paired_effect: str
    n_fft: int
    win_length: int
    hop_length: int
    bins: str
    window: WindowSpec
    fft: FftSpec
    framing: FramingSpec
    log_magnitude: LogMagnitudeSpec
    reduction: ReductionSpec


MetricConfig = ReconstructionSnr | SiSnr | LogSpectralDistance


@dataclass(frozen=True)
class Metrics:
    """The three co-primary estimands and the legacy diagnostics."""

    primary: tuple[str, ...]
    legacy: tuple[str, ...]
    legacy_in_headline: bool
    composite_score: bool
    reconstruction_snr: ReconstructionSnr
    si_snr: SiSnr
    log_spectral_distance: LogSpectralDistance

    def metric_config(self, name: str) -> MetricConfig:
        """Return the frozen configuration for a named primary metric.

        Args:
            name: Primary metric name, as listed in `primary`.

        Returns:
            The frozen config for that metric.

        Raises:
            ProtocolValueError: If the name is not a configured primary metric.
        """
        config = getattr(self, name, None)
        if not isinstance(config, ReconstructionSnr | SiSnr | LogSpectralDistance):
            raise ProtocolValueError(f"metrics.{name} is not a configured primary metric")
        return config


@dataclass(frozen=True)
class Bootstrap:
    """Resampling contract for the paired comparison."""

    unit: str
    iterations: int
    method: str
    confidence: float
    seed: int
    paired: bool


@dataclass(frozen=True)
class Aggregation:
    """How frame metrics become recording metrics and then intervals."""

    within_track: str
    across_tracks: tuple[str, ...]
    bootstrap: Bootstrap


@dataclass(frozen=True)
class Reporting:
    """Presentation-integrity rules binding how results may be stated."""

    note: str
    co_primary_reported_together: bool
    omnibus_winner: bool
    significance_language: bool


@dataclass(frozen=True)
class PublicationGates:
    """Conditions every one of which must hold before numbers may be published."""

    note: str
    required: tuple[str, ...]


@dataclass(frozen=True)
class Amends:
    """Which protocol this one replaces, identified by digest rather than by number.

    The loader recomputes both digests from the archived files, so the chain is
    checked rather than merely recorded. That establishes that the predecessor
    present in the tree is the one this version claims to amend. It does not
    establish that the predecessor was externally approved; git history and review
    carry that, and no amount of repo-local hashing can substitute for them.
    """

    protocol_version: int
    protocol_path: str
    protocol_sha256: str
    document_path: str
    document_sha256: str
    reason: str


@dataclass(frozen=True)
class Protocol:
    """The validated, immutable benchmark protocol."""

    protocol_version: int
    protocol_document: str
    note: str
    amends: Amends
    environment: Environment
    selection: Selection
    eligibility: Eligibility
    duplicate_detection: DuplicateDetection
    conditioning: Conditioning
    metrics: Metrics
    reporting: Reporting
    aggregation: Aggregation
    publication_gates: PublicationGates
    source_sha256: str
    source_path: Path


T = TypeVar("T")


def _check_keys(provided: set[str], expected: set[str], path: str) -> None:
    """Reject unknown and missing keys so a typo fails loudly rather than defaulting."""
    unknown = provided - expected
    if unknown:
        raise ProtocolSchemaError(f"{path}: unknown key(s) {sorted(unknown)}")
    missing = expected - provided
    if missing:
        raise ProtocolSchemaError(f"{path}: missing key(s) {sorted(missing)}")


def _build(cls: type[T], data: Mapping[str, Any], path: str, **nested: Any) -> T:
    """Construct a frozen section from JSON, with pre-built values passed as `nested`.

    Args:
        cls: The frozen dataclass to construct.
        data: The JSON object for this section.
        path: Dotted location, used in error messages.
        **nested: Already-converted values that override their raw JSON counterparts,
            used for sub-objects and for lists that must become tuples.

    Returns:
        The constructed frozen instance.

    Raises:
        ProtocolSchemaError: If any key is unknown or missing.
    """
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    _check_keys(set(data) | set(nested), expected, path)
    plain = {k: v for k, v in data.items() if k not in nested}
    return cls(**plain, **nested)


def _parse_selection(data: Mapping[str, Any]) -> Selection:
    """Build the selection section with immutable collections."""
    return _build(
        Selection,
        data,
        "selection",
        partitions=MappingProxyType(dict(data["partitions"])),
        permanent_failure_codes=tuple(data["permanent_failure_codes"]),
    )


def _parse_duplicate_detection(data: Mapping[str, Any]) -> DuplicateDetection:
    """Build the duplicate-detection section and its two sub-objects."""
    return _build(
        DuplicateDetection,
        data,
        "duplicate_detection",
        mfcc=_build(MfccConfig, data["mfcc"], "duplicate_detection.mfcc"),
        calibration=_build(MfccCalibration, data["calibration"], "duplicate_detection.calibration"),
    )


def _parse_log_spectral_distance(data: Mapping[str, Any]) -> LogSpectralDistance:
    """Build the LSD section and its five transform sub-objects."""
    at = "metrics.log_spectral_distance"
    return _build(
        LogSpectralDistance,
        data,
        at,
        window=_build(WindowSpec, data["window"], f"{at}.window"),
        fft=_build(FftSpec, data["fft"], f"{at}.fft"),
        framing=_build(FramingSpec, data["framing"], f"{at}.framing"),
        log_magnitude=_build(LogMagnitudeSpec, data["log_magnitude"], f"{at}.log_magnitude"),
        reduction=_build(ReductionSpec, data["reduction"], f"{at}.reduction"),
    )


def _parse_metrics(data: Mapping[str, Any]) -> Metrics:
    """Build the metrics section and its three per-metric configs."""
    return _build(
        Metrics,
        data,
        "metrics",
        primary=tuple(data["primary"]),
        legacy=tuple(data["legacy"]),
        reconstruction_snr=_build(
            ReconstructionSnr, data["reconstruction_snr"], "metrics.reconstruction_snr"
        ),
        si_snr=_build(SiSnr, data["si_snr"], "metrics.si_snr"),
        log_spectral_distance=_parse_log_spectral_distance(data["log_spectral_distance"]),
    )


def _parse_aggregation(data: Mapping[str, Any]) -> Aggregation:
    """Build the aggregation section and its bootstrap sub-object."""
    return _build(
        Aggregation,
        data,
        "aggregation",
        across_tracks=tuple(data["across_tracks"]),
        bootstrap=_build(Bootstrap, data["bootstrap"], "aggregation.bootstrap"),
    )


def _parse(data: Mapping[str, Any], digest: str, source: Path) -> Protocol:
    """Build the whole frozen object graph from the parsed JSON."""
    _check_keys(set(data), set(_ROOT_KEYS), "(root)")
    return _build(
        Protocol,
        data,
        "(root)",
        amends=_build(Amends, data["amends"], "amends"),
        environment=_build(Environment, data["environment"], "environment"),
        selection=_parse_selection(data["selection"]),
        eligibility=_build(Eligibility, data["eligibility"], "eligibility"),
        duplicate_detection=_parse_duplicate_detection(data["duplicate_detection"]),
        conditioning=_build(Conditioning, data["conditioning"], "conditioning"),
        metrics=_parse_metrics(data["metrics"]),
        reporting=_build(Reporting, data["reporting"], "reporting"),
        aggregation=_parse_aggregation(data["aggregation"]),
        publication_gates=_build(
            PublicationGates,
            data["publication_gates"],
            "publication_gates",
            required=tuple(data["publication_gates"]["required"]),
        ),
        source_sha256=digest,
        source_path=source,
    )


def _validate_environment(env: Environment) -> None:
    """Refuse to run outside the environment the reproducibility claim is scoped to."""
    if env.bit_generator != "PCG64":
        raise ProtocolValueError(
            f"environment.bit_generator must be 'PCG64' in v1, got {env.bit_generator!r}; "
            "the seeded stream guarantee is specific to the named generator"
        )
    if not hasattr(np.random, env.bit_generator):
        raise ProtocolEnvironmentError(
            f"NumPy provides no BitGenerator named {env.bit_generator!r}"
        )
    if np.__version__ != env.numpy_version:
        raise ProtocolEnvironmentError(
            f"protocol pins NumPy {env.numpy_version} but the runtime has {np.__version__}; "
            "bootstrap results are reproducible only under the pinned version, so this "
            "fails rather than running with a silently weakened guarantee"
        )


def _validate_selection(sel: Selection) -> None:
    """Every partition must claim at least one slot and the walk must terminate."""
    if sel.seed < 0:
        raise ProtocolValueError(f"selection.seed must be non-negative, got {sel.seed}")
    if not sel.partitions:
        raise ProtocolValueError("selection.partitions must not be empty")
    for name, count in sel.partitions.items():
        if count <= 0:
            raise ProtocolValueError(f"selection.partitions.{name} must be positive, got {count}")
    if sel.acquisition_retries < 0:
        raise ProtocolValueError("selection.acquisition_retries must be non-negative")
    if not sel.permanent_failure_codes:
        raise ProtocolValueError("selection.permanent_failure_codes must not be empty")


def _validate_eligibility(el: Eligibility) -> None:
    """Loudness floors are dBFS below full scale, so they cannot be positive."""
    if el.recording_min_duration_s <= 0:
        raise ProtocolValueError("eligibility.recording_min_duration_s must be positive")
    for name, value in (
        ("recording_rms_dbfs_min", el.recording_rms_dbfs_min),
        ("frame_rms_dbfs_min", el.frame_rms_dbfs_min),
    ):
        if value > 0:
            raise ProtocolValueError(
                f"eligibility.{name} is dBFS relative to full scale and must be <= 0, got {value}"
            )


def _validate_calibration(dd: DuplicateDetection) -> None:
    """The threshold must sit inside the evidence that is offered to justify it."""
    cal = dd.calibration
    for name, count in (
        ("positives_n", cal.positives_n),
        ("negatives_n", cal.negatives_n),
        ("same_session_negatives_n", cal.same_session_negatives_n),
    ):
        if count <= 0:
            raise ProtocolValueError(f"duplicate_detection.calibration.{name} must be positive")
    ceiling = max(cal.negatives_max, cal.same_session_negatives_max)
    if not ceiling < dd.flag_threshold < cal.positives_min:
        raise ProtocolValueError(
            f"duplicate_detection.flag_threshold {dd.flag_threshold} must lie strictly between "
            f"the calibration negatives ceiling {ceiling} and the positives floor "
            f"{cal.positives_min}; a threshold outside its own evidence is unjustified"
        )


def _validate_duplicate_detection(dd: DuplicateDetection) -> None:
    """Screening must be a cosine threshold with a fail-closed uncertain verdict."""
    if not -1.0 <= dd.flag_threshold <= 1.0:
        raise ProtocolValueError(
            f"duplicate_detection.flag_threshold is a cosine similarity and must lie in "
            f"[-1, 1], got {dd.flag_threshold}"
        )
    if dd.action_on_flag != "adjudicate":
        raise ProtocolValueError("duplicate_detection.action_on_flag must be 'adjudicate'")
    if dd.action_on_uncertain != "exclude":
        raise ProtocolValueError(
            "duplicate_detection.action_on_uncertain must be 'exclude'; an uncertain verdict "
            "that admits the candidate is a fail-open duplicate policy"
        )
    mfcc = dd.mfcc
    for name, value in (
        ("sample_rate", mfcc.sample_rate),
        ("n_mfcc", mfcc.n_mfcc),
        ("hop_length", mfcc.hop_length),
        ("alignment_stride_frames", mfcc.alignment_stride_frames),
        ("min_overlap_frames", mfcc.min_overlap_frames),
    ):
        if value <= 0:
            raise ProtocolValueError(f"duplicate_detection.mfcc.{name} must be positive")
    if mfcc.alignment_search_s < 0:
        raise ProtocolValueError("duplicate_detection.mfcc.alignment_search_s must be >= 0")
    _validate_calibration(dd)


def _validate_conditioning(cond: Conditioning) -> None:
    """Alignment search is forbidden because it is indistinguishable from metric fitting."""
    if cond.sample_rate <= 0:
        raise ProtocolValueError("conditioning.sample_rate must be positive")
    if cond.channels != "mono":
        raise ProtocolValueError(f"conditioning.channels must be 'mono', got {cond.channels!r}")
    if cond.max_length_difference_samples < 0:
        raise ProtocolValueError("conditioning.max_length_difference_samples must be >= 0")
    if cond.alignment_search is not False:
        raise ProtocolValueError(
            "conditioning.alignment_search must be false; searching for the offset that "
            "maximises a score is indistinguishable from fitting the metric"
        )


def _validate_metric_parameters(metrics: Metrics) -> None:
    """Frozen formula constants must be usable: positive floors, coherent STFT sizes."""
    if metrics.reconstruction_snr.residual_floor <= 0:
        raise ProtocolValueError("metrics.reconstruction_snr.residual_floor must be positive")
    if metrics.si_snr.epsilon <= 0:
        raise ProtocolValueError("metrics.si_snr.epsilon must be positive")
    if metrics.si_snr.constant_reference != _CONSTANT_REFERENCE:
        raise ProtocolValueError(
            f"metrics.si_snr.constant_reference must be {_CONSTANT_REFERENCE!r}, got "
            f"{metrics.si_snr.constant_reference!r}; the alternative is letting epsilon "
            "return a finite score for a metric with no projection direction"
        )
    if metrics.si_snr.zero_mean is not True:
        raise ProtocolValueError(
            "metrics.si_snr.zero_mean must be true; the protocol freezes the zero-mean "
            "formulation because 'standard SI-SNR' is not one thing"
        )
    for name, cap in (
        ("reconstruction_snr", metrics.reconstruction_snr.cap_db),
        ("si_snr", metrics.si_snr.cap_db),
    ):
        if cap <= 0:
            raise ProtocolValueError(f"metrics.{name}.cap_db must be positive, got {cap}")
    lsd = metrics.log_spectral_distance
    for name, value in (
        ("n_fft", lsd.n_fft),
        ("win_length", lsd.win_length),
        ("hop_length", lsd.hop_length),
    ):
        if value <= 0:
            raise ProtocolValueError(f"metrics.log_spectral_distance.{name} must be positive")
    if lsd.win_length > lsd.n_fft:
        raise ProtocolValueError(
            f"metrics.log_spectral_distance.win_length {lsd.win_length} exceeds n_fft {lsd.n_fft}"
        )
    if lsd.hop_length > lsd.win_length:
        raise ProtocolValueError(
            f"metrics.log_spectral_distance.hop_length {lsd.hop_length} exceeds win_length "
            f"{lsd.win_length}, which would leave samples unanalysed"
        )
    _validate_lsd_transform(lsd)


def _validate_lsd_transform(lsd: LogSpectralDistance) -> None:
    """Every convention that moves the number is pinned to one value.

    v1 named a Hann window without fixing periodic against symmetric and never
    stated an FFT normalisation. Both change the result, so both are frozen here
    rather than inherited from whichever library happens to be installed.
    """
    at = "metrics.log_spectral_distance"
    for field, actual, expected in (
        ("window.kind", lsd.window.kind, _WINDOW_KIND),
        ("window.symmetry", lsd.window.symmetry, _WINDOW_SYMMETRY),
        ("fft.kind", lsd.fft.kind, _FFT_KIND),
        ("fft.norm", lsd.fft.norm, _FFT_NORM),
        ("framing.pad_mode", lsd.framing.pad_mode, _PAD_MODE),
        ("framing.frame_starts", lsd.framing.frame_starts, _FRAME_STARTS),
        ("log_magnitude.operation", lsd.log_magnitude.operation, _LOG_MAGNITUDE_OPERATION),
        ("reduction.bins", lsd.reduction.bins, _BIN_REDUCTION),
        ("reduction.frames", lsd.reduction.frames, _FRAME_REDUCTION),
    ):
        if actual != expected:
            raise ProtocolValueError(f"{at}.{field} must be {expected!r} in v2, got {actual!r}")

    framing = lsd.framing
    if framing.center is not True:
        raise ProtocolValueError(f"{at}.framing.center must be true")
    if framing.partial_trailing_frame is not False:
        raise ProtocolValueError(
            f"{at}.framing.partial_trailing_frame must be false; a partial final frame "
            "would be analysed against a differently shaped window"
        )
    if framing.pad_value != 0.0:
        raise ProtocolValueError(
            f"{at}.framing.pad_value must be 0.0, got {framing.pad_value}; constant padding "
            "with an unstated value inherits a library default"
        )
    expected_pad = lsd.n_fft // 2
    for side, value in (
        ("pad_left_samples", framing.pad_left_samples),
        ("pad_right_samples", framing.pad_right_samples),
    ):
        if value != expected_pad:
            raise ProtocolValueError(
                f"{at}.framing.{side} must be n_fft // 2 = {expected_pad}, got {value}"
            )

    if lsd.log_magnitude.offset <= 0:
        raise ProtocolValueError(f"{at}.log_magnitude.offset must be positive")


def _validate_amends(amends: Amends) -> None:
    """The predecessor named here must be the frozen predecessor present in the tree.

    Which artifacts are amended is checked as well as their digests. Verifying only the
    digests would establish that the named file matches the named hash, a claim any
    config naming some other file together with that file's hash satisfies equally, and
    it would read whatever path the config supplied.
    """
    if amends.protocol_version != _PREDECESSOR_VERSION:
        raise ProtocolValueError(
            f"amends.protocol_version must be {_PREDECESSOR_VERSION}, got {amends.protocol_version}"
        )
    if not amends.reason.strip():
        raise ProtocolValueError(
            "amends.reason must not be empty; an amendment without a recorded reason is "
            "indistinguishable from an edit"
        )
    for label, stated, frozen in (
        ("protocol", amends.protocol_path, _PREDECESSOR_PROTOCOL_PATH),
        ("document", amends.document_path, _PREDECESSOR_DOCUMENT_PATH),
    ):
        if stated != frozen:
            raise ProtocolValueError(
                f"amends.{label}_path must be {frozen!r}, got {stated!r}; the amendment "
                "chain is not redirectable by configuration"
            )
    for label, rel_path, recorded in (
        ("protocol", amends.protocol_path, amends.protocol_sha256),
        ("document", amends.document_path, amends.document_sha256),
    ):
        if len(recorded) != 64 or not all(c in "0123456789abcdef" for c in recorded):
            raise ProtocolValueError(f"amends.{label}_sha256 must be 64 lowercase hex digits")
        archived = _REPO_ROOT / rel_path
        if not archived.is_file():
            raise ProtocolValueError(
                f"amends.{label}_path {rel_path} is missing; the predecessor artifact must "
                "stay in the tree so an artifact naming the older version stays interpretable"
            )
        actual = hashlib.sha256(archived.read_bytes()).hexdigest()
        if actual != recorded:
            raise ProtocolValueError(
                f"amends.{label}_sha256 records {recorded[:12]} but {rel_path} hashes to "
                f"{actual[:12]}; the archived predecessor has been modified"
            )
    logger.debug("Amendment chain verified against predecessor v%d", amends.protocol_version)


def _validate_metrics(metrics: Metrics) -> None:
    """A positive effect must favour OpGAN for every metric, and no metric may be collapsed."""
    if len(set(metrics.primary)) != len(metrics.primary):
        raise ProtocolValueError(f"metrics.primary contains duplicates: {list(metrics.primary)}")
    if not metrics.primary:
        raise ProtocolValueError("metrics.primary must not be empty")
    overlap = set(metrics.primary) & set(metrics.legacy)
    if overlap:
        raise ProtocolValueError(f"metrics.legacy overlaps metrics.primary: {sorted(overlap)}")
    if metrics.composite_score is not False:
        raise ProtocolValueError(
            "metrics.composite_score must be false; collapsing correlated views into one "
            "score invites a championship metric chosen after seeing which one separates best"
        )
    if metrics.legacy_in_headline is not False:
        raise ProtocolValueError("metrics.legacy_in_headline must be false")
    for name in metrics.primary:
        config = metrics.metric_config(name)
        expected = EFFECT_FOR_DIRECTION.get(config.direction)
        if expected is None:
            raise ProtocolValueError(
                f"metrics.{name}.direction must be one of "
                f"{sorted(EFFECT_FOR_DIRECTION)}, got {config.direction!r}"
            )
        if config.paired_effect != expected:
            raise ProtocolValueError(
                f"metrics.{name} is {config.direction} so its paired_effect must be "
                f"{expected!r} for a positive delta to favour OpGAN, got {config.paired_effect!r}"
            )
    _validate_metric_parameters(metrics)


def _validate_aggregation(agg: Aggregation) -> None:
    """Resampling frames instead of recordings is the pseudo-replication this forbids."""
    boot = agg.bootstrap
    if boot.unit != "recording":
        raise ProtocolValueError(
            f"aggregation.bootstrap.unit must be 'recording', got {boot.unit!r}; frames within "
            "a recording are not independent and resampling them overstates confidence"
        )
    if boot.paired is not True:
        raise ProtocolValueError(
            "aggregation.bootstrap.paired must be true; resampling the two systems "
            "independently discards the pairing the design exists to exploit"
        )
    if boot.method != "percentile":
        raise ProtocolValueError(
            f"aggregation.bootstrap.method must be 'percentile', got {boot.method!r}"
        )
    if boot.iterations <= 0:
        raise ProtocolValueError("aggregation.bootstrap.iterations must be positive")
    if not 0.0 < boot.confidence < 1.0:
        raise ProtocolValueError(
            f"aggregation.bootstrap.confidence must lie in (0, 1), got {boot.confidence}"
        )
    if boot.seed < 0:
        raise ProtocolValueError("aggregation.bootstrap.seed must be non-negative")
    if not agg.across_tracks:
        raise ProtocolValueError("aggregation.across_tracks must not be empty")


def _validate_reporting(rep: Reporting) -> None:
    """The anti-cherry-picking rules are structural, not editorial guidance."""
    if rep.co_primary_reported_together is not True:
        raise ProtocolValueError(
            "reporting.co_primary_reported_together must be true; emitting one primary effect "
            "without the other two permits selective emphasis after seeing results"
        )
    if rep.omnibus_winner is not False:
        raise ProtocolValueError("reporting.omnibus_winner must be false")
    if rep.significance_language is not False:
        raise ProtocolValueError(
            "reporting.significance_language must be false; each interval is marginal for its "
            "named metric and the family-wise rate across three correlated metrics is unquantified"
        )


def _validate_gates(gates: PublicationGates) -> None:
    """v1 freezes the exact gate set, in both directions."""
    if len(set(gates.required)) != len(gates.required):
        raise ProtocolValueError("publication_gates.required contains duplicates")
    configured = set(gates.required)
    if configured != FROZEN_GATES:
        raise ProtocolValueError(
            "publication_gates.required must be exactly the frozen v1 set; "
            f"added {sorted(configured - FROZEN_GATES)}, "
            f"removed {sorted(FROZEN_GATES - configured)}. Changing what makes a run "
            "publishable after the protocol is frozen selects which runs get published, "
            "so additions require a version bump just as removals do"
        )


def _validate(protocol: Protocol) -> None:
    """Run every section validator against the parsed protocol."""
    if protocol.protocol_version != SUPPORTED_PROTOCOL_VERSION:
        raise ProtocolValueError(
            f"unsupported protocol_version {protocol.protocol_version}; this module implements "
            f"v{SUPPORTED_PROTOCOL_VERSION}"
        )
    _validate_amends(protocol.amends)
    _validate_environment(protocol.environment)
    _validate_selection(protocol.selection)
    _validate_eligibility(protocol.eligibility)
    _validate_duplicate_detection(protocol.duplicate_detection)
    _validate_conditioning(protocol.conditioning)
    _validate_metrics(protocol.metrics)
    _validate_aggregation(protocol.aggregation)
    _validate_reporting(protocol.reporting)
    _validate_gates(protocol.publication_gates)


def load_protocol() -> Protocol:
    """Load, validate, and freeze the one approved v1 configuration.

    Takes no arguments on purpose. The validators below establish that a config is
    *shaped* like v1, which is a weaker property than *being* v1: a file with a
    different selection seed, partition size, or STFT length satisfies every one of
    them while describing a different benchmark. If this function accepted a path,
    the supported API could bless an unregistered alternative as protocol v1, which
    is a wider hole than any direct file read. There is one approved configuration,
    and this is the only way production code reaches it.

    Returns:
        The validated, deeply immutable protocol, carrying the SHA-256 of the exact
        bytes it was parsed from.

    Raises:
        ProtocolSchemaError: If the file is not valid JSON, or a key is unknown,
            missing, or holds a value of the wrong JSON type.
        ProtocolValueError: If the config parses but states an invalid protocol.
        ProtocolEnvironmentError: If the runtime differs from the pinned environment.

    Example:
        >>> protocol = load_protocol()
        >>> protocol.aggregation.bootstrap.unit
        'recording'
    """
    return _load_from_path(_PROTOCOL_PATH)


def _load_from_path(source: Path) -> Protocol:
    """Parse and validate a protocol config from an arbitrary path.

    Private, and production code must not call it. Its only caller outside this
    module is the test suite, which needs to feed deliberately broken copies through
    the real validators. Exposing it publicly would reintroduce exactly the hole
    `load_protocol` closes by taking no arguments.
    """
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProtocolSchemaError(f"{source}: not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ProtocolSchemaError(f"{source}: top level must be a JSON object")

    # Before anything else, because a superseded config fails on whichever key changed
    # since, and "missing key(s) ['amends']" is a far less useful answer than the
    # version it actually declares.
    declared = data.get("protocol_version")
    if declared != SUPPORTED_PROTOCOL_VERSION:
        raise ProtocolValueError(
            f"{source}: unsupported protocol_version {declared!r}; this module implements "
            f"v{SUPPORTED_PROTOCOL_VERSION}"
        )

    try:
        protocol = _parse(data, digest, source)
    except (TypeError, KeyError) as e:
        raise ProtocolSchemaError(f"{source}: malformed protocol section: {e}") from e

    try:
        _validate(protocol)
    except TypeError as e:
        raise ProtocolSchemaError(f"{source}: value has the wrong JSON type: {e}") from e

    logger.info(
        "Loaded benchmark protocol v%d from %s (sha256 %s)",
        protocol.protocol_version,
        source,
        digest[:12],
    )
    return protocol
