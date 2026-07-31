"""Immutable, validated access to the frozen benchmark protocol.

Two things share the authority here, with different jobs. `benchmark/protocols/v4.json`
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
import importlib.metadata
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
_PROTOCOL_PATH = _REPO_ROOT / "benchmark" / "protocols" / "v4.json"

SUPPORTED_PROTOCOL_VERSION = 4

# Which protocol v2 supersedes, and the artifacts it supersedes, as historical facts
# rather than as arithmetic. `SUPPORTED_PROTOCOL_VERSION - 1` would encode a universal
# rule that every version amends its immediate predecessor, which is not something the
# governance model promises. The paths are frozen here so the config cannot redirect
# the amendment chain at another file whose digest it also records, and so a relative
# path out of the tree is not reachable at all.
_PREDECESSOR_VERSION = 3
_PREDECESSOR_PROTOCOL_PATH = "benchmark/protocols/v3.json"
_PREDECESSOR_DOCUMENT_PATH = "docs/benchmark-protocol-v3.md"

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

# The source-to-canonical transformation, pinned as kinds. Every one of these was
# implementation-defined in v3 and each changes the canonical bytes, so each changes every
# content hash downstream.
_CANONICALISATION_ORDER = (
    "decode",
    "validate_source",
    "downmix",
    "resample",
    "normalise_signed_zero",
    "validate_canonical",
)
_DOWNMIX_OPERATION = "mono_passthrough_or_stereo_arithmetic_mean_float64"
_DECODE_DTYPE = "float64"
_METADATA_AUTHORITY = "decoder_reported"
_BYTE_IDENTITY = "bounded_single_buffer_read_hash_inspect_decode"
_READ_BOUND_ENFORCEMENT = "before_allocation"
_LIVE_BYTES_RULE = "source_plus_decoded_plus_downmix_plus_canonical_projected_before_decode"
_METADATA_VALIDATION = "validated_before_projection_and_allocation"
_RES_TYPE = "soxr_hq"
_OUTPUT_LENGTH_RULE = "ceil_source_samples_times_target_rate_over_source_rate"

# Exact and ordered, not a set that merely contains something. A membership check would
# accept a list naming one irrelevant operation while omitting every operation that matters.
_FORBIDDEN_OPERATIONS = (
    "integer_quantisation",
    "dither",
    "clipping",
    "peak_normalisation",
    "loudness_normalisation",
    "gain_adjustment",
)

# Two separate reasons produce this set, and conflating them is how PCM_8 would slip in.
# Determinism: integer PCM converts exactly into float64 and its decoded amplitude is bounded
# by construction. Reference fidelity: below 16 bits the quantisation noise becomes part of
# the supposed clean reference. Pairs rather than two independent sets, so an unintended
# combination cannot be assembled from two individually permitted values.
_SOURCE_FORMATS = (("FLAC", "PCM_16"), ("FLAC", "PCM_24"), ("FLAC", "PCM_32"))

_CANONICAL_AMPLITUDE_BOUND = "none"
_RMS_OPERATION = "float64_mean_square_sqrt_dbfs_no_epsilon"

# A decoder rejection is evidence about the locally acquired artifact and nothing more. It
# may be charged to the source only after the environment has decoded a known-good fixture
# and the exact local bytes have been identified, because our own retrieval could equally
# have damaged them.
_ATTRIBUTION_RULE = "verified_local_artifact_rejection_is_a_candidate_outcome"
_PRECONDITION_ORDER = (
    "protocol_semantics",
    "runtime_conformance",
    "golden_fixture",
    "local_artifact_identity",
    "inspect_and_decode",
    "classify_outcome",
)
_LOCAL_IDENTITY_BASIS = "local_sha256_of_the_decoded_buffer_or_of_the_streamed_artifact"
_DIAGNOSTIC_HANDLING = "sanitised_for_publication_raw_retained_in_run_log"

_REDUCTION_CHAIN = (
    "selected_sources",
    "local_artifacts_acquired",
    "inspection_succeeded",
    "decode_succeeded",
    "source_domain_supported",
    "recordings_eligible",
    "complete_frames_produced",
    "frames_eligible",
)

# Held apart from the eligibility vocabulary so a harness failure is not merely discouraged
# from becoming a source verdict but cannot be represented as one. Any overlap between the
# two is rejected at load.
_ABORT_REASONS = (
    "SOURCE_HASH_MISMATCH",
    "LOCAL_ARTIFACT_UNSTABLE",
    "LOCAL_ARTIFACT_NOT_REGULAR_FILE",
    "GOLDEN_FIXTURE_FAILURE",
    "ENVIRONMENT_MISMATCH",
    "ENVIRONMENT_CAPACITY_FAILURE",
    "UNEXPECTED_IO_FAILURE",
    "INTERNAL_CANONICALISATION_ERROR",
)
# Grouped, because "12 sources rejected" invites the reader to conclude the collection is
# defective when most of them may simply sit outside limits this benchmark declared in
# advance. Every reason belongs to exactly one category; the partition is asserted over
# these constants in the test suite, where it can still fail, rather than re-checked
# against a config the loader already requires to equal them.
_REASON_CATEGORIES = (
    (
        "frozen_domain_exclusion",
        (
            "SOURCE_EXCEEDS_MAX_BYTES",
            "SOURCE_EXCEEDS_MAX_DURATION",
            "SOURCE_EXCEEDS_MAX_DECODED_BYTES",
            "SOURCE_BELOW_MIN_SAMPLE_RATE",
        ),
    ),
    (
        "unsupported_representation",
        ("UNSUPPORTED_CONTAINER", "UNSUPPORTED_SUBTYPE", "UNSUPPORTED_CHANNEL_COUNT"),
    ),
    (
        "artifact_or_decoder_outcome",
        ("SOURCE_INSPECTION_REJECTED", "SOURCE_DECODE_REJECTED", "SOURCE_METADATA_INCONSISTENT"),
    ),
    (
        "decoded_waveform_violation",
        ("EMPTY_SOURCE", "NONFINITE_SOURCE", "INVALID_SOURCE_AMPLITUDE"),
    ),
)
_INELIGIBLE_REASONS = tuple(code for _, codes in _REASON_CATEGORIES for code in codes)

# Exactly one primary reason is counted per failed source, taken as the first applicable
# entry here. Without a frozen order two conforming implementations publish different
# category totals for the same artifact. The order follows what can be known at each stage,
# not any judgement about severity; within the metadata stage it is a frozen convention
# listing what the file is before what it contains.
_REASON_PRECEDENCE = (
    "SOURCE_EXCEEDS_MAX_BYTES",
    "SOURCE_INSPECTION_REJECTED",
    "UNSUPPORTED_CONTAINER",
    "UNSUPPORTED_SUBTYPE",
    "UNSUPPORTED_CHANNEL_COUNT",
    "SOURCE_BELOW_MIN_SAMPLE_RATE",
    "SOURCE_EXCEEDS_MAX_DURATION",
    "SOURCE_EXCEEDS_MAX_DECODED_BYTES",
    "SOURCE_DECODE_REJECTED",
    "SOURCE_METADATA_INCONSISTENT",
    "EMPTY_SOURCE",
    "NONFINITE_SOURCE",
    "INVALID_SOURCE_AMPLITUDE",
)
_SUPPLEMENTAL_REASONS = "established_before_termination_recorded_not_counted"

# An oversized source is rejected from fstat, before the artifact is buffered, so the
# single-buffer digest does not exist for it. Hashing the bounded prefix and calling it
# local_sha256 would claim an identity for the whole artifact that was never computed.
_EXCLUDED_SOURCE_IDENTITY = "bounded_streaming_sha256_within_identity_stream_bound"
_ARTIFACT_STABILITY_CHECK = "same_descriptor_dev_ino_size_mtime_ns_ctime_ns_before_and_after"
_IDENTITY_STATUS_VALUES = ("complete_sha256", "unavailable_above_identity_stream_bound")
_REASON_SELECTION = "all_violations_established_in_the_reached_stage_then_within_stage_precedence"

# The stage machine the record model depends on. Every term around it was load-bearing
# while nothing defined it, which is the same defect v4 exists to close one level up: two
# conforming implementations could disagree on whether inspection and metadata validation
# are one stage or two and emit different records for the same artifact.
_EVALUATION_STAGE_ORDER = (
    "local_file_check",
    "inspection",
    "metadata_validation",
    "decode",
    "decoded_metadata_validation",
    "waveform_validation",
)

# What each stage evaluates, so that "every safe check in the reached stage" means the same
# thing in two implementations. Six checks share metadata_validation, which is where
# supplemental reasons actually arise.
_STAGE_CHECKS = (
    ("local_file_check", ("regular_file", "source_bytes")),
    ("inspection", ("decoder_inspection",)),
    (
        "metadata_validation",
        (
            "container",
            "subtype",
            "channel_count",
            "minimum_sample_rate",
            "duration",
            "projected_decoded_bytes",
        ),
    ),
    ("decode", ("decoder_decode",)),
    ("decoded_metadata_validation", ("metadata_matches_decoded_array",)),
    ("waveform_validation", ("non_empty", "finite", "amplitude_range")),
)

# Identity acquisition is evidence collection, not eligibility adjudication, and is
# deliberately absent from the stage order. Placing it there produced a contradiction: an
# oversized artifact terminates at local_file_check, so the derived unevaluated set named
# the identity stage, while the same record carried a digest only that stage could produce.
# Names for how the digest was established, and nothing else. An earlier draft named the
# first for hashing, inspecting, and decoding one buffer, which is the canonicalisation
# byte-path contract rather than an identity method, and it overstated the record: a source
# rejected at inspection or on its subtype is never decoded. What ran is already carried by
# the terminating evaluation stage and the unevaluated suffix.
_IDENTITY_METHOD_VALUES = (
    "bounded_single_buffer_sha256",
    "bounded_streaming_sha256",
    "not_computed_above_bound",
)
_IDENTITY_DERIVATION = "by_source_bytes_against_max_source_bytes_then_max_identity_stream_bytes"
# Every ineligible reason names exactly one stage, so the stage on a record is derived
# rather than chosen. local_identity carries no ineligible reason: its only failures are
# aborts. Six reasons share metadata_validation, which is why a within-stage precedence
# exists and why all six are evaluated before one is selected.
_REASON_STAGE = (
    ("SOURCE_EXCEEDS_MAX_BYTES", "local_file_check"),
    ("SOURCE_INSPECTION_REJECTED", "inspection"),
    ("UNSUPPORTED_CONTAINER", "metadata_validation"),
    ("UNSUPPORTED_SUBTYPE", "metadata_validation"),
    ("UNSUPPORTED_CHANNEL_COUNT", "metadata_validation"),
    ("SOURCE_BELOW_MIN_SAMPLE_RATE", "metadata_validation"),
    ("SOURCE_EXCEEDS_MAX_DURATION", "metadata_validation"),
    ("SOURCE_EXCEEDS_MAX_DECODED_BYTES", "metadata_validation"),
    ("SOURCE_DECODE_REJECTED", "decode"),
    ("SOURCE_METADATA_INCONSISTENT", "decoded_metadata_validation"),
    ("EMPTY_SOURCE", "waveform_validation"),
    ("NONFINITE_SOURCE", "waveform_validation"),
    ("INVALID_SOURCE_AMPLITUDE", "waveform_validation"),
)
_UNEVALUATED_STAGES_DERIVATION = "suffix_of_evaluation_stage_order_after_the_terminating_stage"
_RECORD_SCOPE = "source_ineligible_outcomes_only"

# Constraints on an emitted record, not on this file. Ordering is included because two
# implementations holding the same set and serialising it differently produce different
# manifest bytes, and therefore different digests, which would break the content addressing
# the benchmark rests on.
_RECORD_CONSTRAINTS = (
    "primary_reason_code_in_ineligible_reasons",
    "supplemental_reason_codes_subset_of_ineligible_reasons",
    "primary_reason_code_not_in_supplemental_reason_codes",
    "supplemental_reason_codes_unique",
    "supplemental_reason_codes_share_primary_reason_stage",
    "supplemental_reason_codes_ordered_by_reason_precedence",
    "unevaluated_evaluation_stages_ordered_by_evaluation_stage_order",
    # Status, method, digest presence, and source size are four views of one fact, and any
    # pair can be made to disagree by a plausible typo.
    "identity_status_consistent_with_identity_method",
    "local_sha256_present_iff_identity_complete",
    "identity_method_consistent_with_source_byte_bounds",
    "digest_discarded_when_instability_detected",
)

# The identity stream bound is a resource-security limit, so unlike other magnitudes it is
# also constrained relative to the bound that triggers it. A variant configuration must not
# be able to set it to a petabyte while passing every positivity and ordering check. The
# multiple is a convention with room for a plausible anomaly, not a derived quantity.
_MAX_IDENTITY_STREAM_MULTIPLE = 16

# Required on every failure record, plus the conditional evidence below. One flat tuple
# would force placeholders that pretend evidence exists, and a reader could not then tell
# an absent value from an inapplicable one.
_RECORD_REQUIRED_CORE = (
    "logical_source_id",
    "terminating_evaluation_stage",
    "primary_reason_code",
    "identity_status",
    "identity_method",
    "protocol_sha256",
    "unevaluated_evaluation_stages",
)
_RECORD_CONDITIONAL = (
    ("local_sha256", "when_identity_complete_sha256"),
    ("observed_metadata", "when_inspection_succeeded"),
    ("decoder_diagnostic", "when_decoder_rejected"),
    ("supplemental_reason_codes", "when_additional_reasons_established"),
)
_SIGNED_ZERO = "normalise_to_positive"
_NON_FINITE = "reject"
_HASH_DTYPE = "float64"
_HASH_BYTE_ORDER = "little"
_HASH_CONTAINER = "none"
_REPRODUCIBILITY_MODEL = "reference_artifact_authoritative"
_PARTIAL_TRAILING_FRAME = "dropped_and_counted"
_ELIGIBILITY_COMPUTED_FROM = "canonical_waveform"
_RMS_COMPARATOR = "greater_than"
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
        "canonicalisation",
        "framing",
        "environment",
        "selection",
        "eligibility",
        "source_outcomes",
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
class Decode:
    """How a source file becomes samples, pinned to an exact call.

    The source-format pairs and the resource bounds live here rather than with eligibility
    because they decide whether decoding is attempted at all, and the decoded footprint is
    knowable from inspection before a single sample is allocated.
    """

    library: str
    library_version: str
    libsndfile_version: str
    dtype: str
    always_2d: bool
    metadata_authority: str
    metadata_authority_note: str
    byte_identity: str
    byte_identity_note: str
    source_formats: tuple[tuple[str, str], ...]
    source_formats_note: str
    read_bound_enforcement: str
    read_bound_enforcement_note: str
    require_regular_file: bool
    require_regular_file_note: str
    metadata_validation: str
    metadata_validation_note: str
    max_source_bytes: int
    max_identity_stream_bytes: int
    max_identity_stream_bytes_note: str
    max_decoded_bytes: int
    max_projected_live_bytes: int
    canonicalisation_concurrency: int
    canonicalisation_concurrency_note: str
    live_bytes_rule: str
    live_bytes_rule_note: str
    max_source_duration_s: int
    resource_bounds_note: str
    note: str


@dataclass(frozen=True)
class Downmix:
    """How channels become one, stated per channel count rather than as a general mean."""

    operation: str
    formula: str
    channels_min: int
    channels_max: int
    note: str


@dataclass(frozen=True)
class Resample:
    """Every resampler argument named, none inherited from a library default."""

    library: str
    library_version: str
    soxr_version: str
    target_sample_rate: int
    res_type: str
    fix: bool
    scale: bool
    axis: int
    output_length_rule: str
    output_length_rule_note: str


@dataclass(frozen=True)
class HashBytes:
    """The byte representation a content digest is taken over."""

    dtype: str
    byte_order: str
    container: str
    note: str


@dataclass(frozen=True)
class SourceValidation:
    """What a decoded source must satisfy to enter the transformation."""

    require_finite: bool
    require_non_empty: bool
    require_positive_sample_rate: bool
    amplitude_min: float
    amplitude_max: float
    amplitude_bounds_inclusive: bool
    min_sample_rate: int
    note: str


@dataclass(frozen=True)
class CanonicalValidation:
    """What the canonical waveform must satisfy, deliberately without an amplitude bound.

    The asymmetry with `SourceValidation` is the whole point: band-limited resampling
    overshoots full scale, and the only ways to force the result back inside [-1, 1] are
    operations `forbidden_operations` names.
    """

    dtype: str
    ndim: int
    require_finite: bool
    require_non_empty: bool
    amplitude_bound: str
    note: str


@dataclass(frozen=True)
class Canonicalisation:
    """The whole source-to-canonical transformation.

    v3 named the output shape and left every producing step undefined, so the identity
    chain had no fixed head. Each field here was measured to change the canonical bytes.
    """

    note: str
    order: tuple[str, ...]
    order_note: str
    decode: Decode
    downmix: Downmix
    resample: Resample
    forbidden_operations: tuple[str, ...]
    forbidden_operations_note: str
    signed_zero: str
    signed_zero_note: str
    non_finite: str
    hash_bytes: HashBytes
    source_validation: SourceValidation
    canonical_validation: CanonicalValidation
    reproducibility_model: str
    reproducibility_note: str


@dataclass(frozen=True)
class Framing:
    """Evaluation frame geometry, which v3 left to incidental prose."""

    note: str
    frame_duration_s: int
    frame_duration_note: str
    frame_samples: int
    hop_samples: int
    first_frame_start_sample: int
    partial_trailing_frame: str
    partial_trailing_frame_note: str


@dataclass(frozen=True)
class SourceOutcomes:
    """What record exists when the transformation cannot complete.

    Two vocabularies, never one. `ineligible_reasons` are predeclared outcomes that are
    recorded, published, and allow the run to continue. `abort_reasons` stop the run and
    block publication. Keeping them structurally separate is what makes it impossible to
    represent a harness failure as a fact about a source.
    """

    note: str
    evaluation_stage_order: tuple[str, ...]
    evaluation_stage_order_note: str
    stage_checks: Mapping[str, tuple[str, ...]]
    stage_checks_note: str
    attribution_rule: str
    attribution_note: str
    precondition_order: tuple[str, ...]
    local_identity_basis: str
    upstream_checksum_available: bool
    local_identity_note: str
    identity_for_excluded_sources: str
    identity_for_excluded_sources_note: str
    identity_status_values: tuple[str, ...]
    identity_status_note: str
    identity_method_values: tuple[str, ...]
    identity_derivation: str
    identity_derivation_note: str
    artifact_stability_check: str
    artifact_stability_note: str
    ineligible_reasons: tuple[str, ...]
    ineligible_reasons_note: str
    reason_precedence: tuple[str, ...]
    reason_precedence_note: str
    reason_stage: Mapping[str, str]
    reason_stage_note: str
    reason_selection: str
    reason_selection_note: str
    supplemental_reasons: str
    supplemental_reasons_note: str
    unevaluated_stages_recorded: bool
    unevaluated_stages_derivation: str
    unevaluated_stages_note: str
    reason_categories: Mapping[str, tuple[str, ...]]
    reason_categories_note: str
    abort_reasons: tuple[str, ...]
    abort_reasons_note: str
    diagnostic_handling: str
    diagnostic_handling_note: str
    record_required_core: tuple[str, ...]
    record_conditional: Mapping[str, str]
    record_scope: str
    record_constraints: tuple[str, ...]
    record_constraints_note: str
    record_note: str
    reduction_chain: tuple[str, ...]
    reduction_chain_note: str


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
    rms_operation: str
    rms_definition: str
    computed_from: str
    recording_duration_rule: str
    recording_rms_comparator: str
    frame_rms_comparator: str
    comparator_note: str
    denominator_note: str


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
    canonicalisation: Canonicalisation
    framing: Framing
    environment: Environment
    selection: Selection
    eligibility: Eligibility
    source_outcomes: SourceOutcomes
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


def _parse_canonicalisation(data: Mapping[str, Any]) -> Canonicalisation:
    """Build the canonicalisation section and its six sub-objects."""
    at = "canonicalisation"
    return _build(
        Canonicalisation,
        data,
        at,
        order=tuple(data["order"]),
        forbidden_operations=tuple(data["forbidden_operations"]),
        decode=_build(
            Decode,
            data["decode"],
            f"{at}.decode",
            # Pairs, so that a permitted container and a permitted subtype cannot be
            # recombined into a combination nobody approved.
            source_formats=tuple(tuple(pair) for pair in data["decode"]["source_formats"]),
        ),
        downmix=_build(Downmix, data["downmix"], f"{at}.downmix"),
        resample=_build(Resample, data["resample"], f"{at}.resample"),
        hash_bytes=_build(HashBytes, data["hash_bytes"], f"{at}.hash_bytes"),
        source_validation=_build(
            SourceValidation, data["source_validation"], f"{at}.source_validation"
        ),
        canonical_validation=_build(
            CanonicalValidation, data["canonical_validation"], f"{at}.canonical_validation"
        ),
    )


def _parse_source_outcomes(data: Mapping[str, Any]) -> SourceOutcomes:
    """Build the source-outcome section with immutable vocabularies."""
    return _build(
        SourceOutcomes,
        data,
        "source_outcomes",
        precondition_order=tuple(data["precondition_order"]),
        ineligible_reasons=tuple(data["ineligible_reasons"]),
        reason_precedence=tuple(data["reason_precedence"]),
        abort_reasons=tuple(data["abort_reasons"]),
        reason_categories=MappingProxyType(
            {name: tuple(codes) for name, codes in data["reason_categories"].items()}
        ),
        evaluation_stage_order=tuple(data["evaluation_stage_order"]),
        stage_checks=MappingProxyType(
            {name: tuple(checks) for name, checks in data["stage_checks"].items()}
        ),
        identity_method_values=tuple(data["identity_method_values"]),
        reason_stage=MappingProxyType(dict(data["reason_stage"])),
        record_constraints=tuple(data["record_constraints"]),
        record_required_core=tuple(data["record_required_core"]),
        record_conditional=MappingProxyType(dict(data["record_conditional"])),
        identity_status_values=tuple(data["identity_status_values"]),
        reduction_chain=tuple(data["reduction_chain"]),
    )


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
        canonicalisation=_parse_canonicalisation(data["canonicalisation"]),
        source_outcomes=_parse_source_outcomes(data["source_outcomes"]),
        framing=_build(Framing, data["framing"], "framing"),
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


def _validate_canonicalisation(canon: Canonicalisation) -> None:
    """Every step of the source-to-canonical transformation is pinned to one value.

    Semantics only. Whether *this machine* may execute the transformation is a different
    question, answered by `verify_canonicalisation_environment`, and keeping the two apart
    means a reader of the protocol is not answering a question about a runtime.
    """
    for field, actual, expected in (
        ("order", canon.order, _CANONICALISATION_ORDER),
        ("downmix.operation", canon.downmix.operation, _DOWNMIX_OPERATION),
        ("decode.dtype", canon.decode.dtype, _DECODE_DTYPE),
        ("decode.metadata_authority", canon.decode.metadata_authority, _METADATA_AUTHORITY),
        ("decode.byte_identity", canon.decode.byte_identity, _BYTE_IDENTITY),
        (
            "decode.read_bound_enforcement",
            canon.decode.read_bound_enforcement,
            _READ_BOUND_ENFORCEMENT,
        ),
        ("decode.live_bytes_rule", canon.decode.live_bytes_rule, _LIVE_BYTES_RULE),
        ("decode.metadata_validation", canon.decode.metadata_validation, _METADATA_VALIDATION),
        ("decode.source_formats", canon.decode.source_formats, _SOURCE_FORMATS),
        ("resample.res_type", canon.resample.res_type, _RES_TYPE),
        ("resample.output_length_rule", canon.resample.output_length_rule, _OUTPUT_LENGTH_RULE),
        ("forbidden_operations", canon.forbidden_operations, _FORBIDDEN_OPERATIONS),
        ("signed_zero", canon.signed_zero, _SIGNED_ZERO),
        ("non_finite", canon.non_finite, _NON_FINITE),
        ("hash_bytes.dtype", canon.hash_bytes.dtype, _HASH_DTYPE),
        ("hash_bytes.byte_order", canon.hash_bytes.byte_order, _HASH_BYTE_ORDER),
        ("hash_bytes.container", canon.hash_bytes.container, _HASH_CONTAINER),
        ("canonical_validation.dtype", canon.canonical_validation.dtype, _DECODE_DTYPE),
        (
            "canonical_validation.amplitude_bound",
            canon.canonical_validation.amplitude_bound,
            _CANONICAL_AMPLITUDE_BOUND,
        ),
        ("reproducibility_model", canon.reproducibility_model, _REPRODUCIBILITY_MODEL),
    ):
        if actual != expected:
            raise ProtocolValueError(
                f"canonicalisation.{field} must be {expected!r} in v4, got {actual!r}"
            )

    if canon.decode.always_2d is not True:
        raise ProtocolValueError(
            "canonicalisation.decode.always_2d must be true; without it a mono source "
            "returns a different rank than a multichannel one and downmix would branch "
            "on source shape"
        )
    resample = canon.resample
    if resample.fix is not True or resample.scale is not False or resample.axis != -1:
        raise ProtocolValueError(
            "canonicalisation.resample must pin fix=true, scale=false, axis=-1; leaving "
            "any of them to a library default recreates the gap v4 exists to close"
        )
    if resample.target_sample_rate <= 0:
        raise ProtocolValueError("canonicalisation.resample.target_sample_rate must be positive")

    _validate_downmix_range(canon.downmix)
    _validate_source_domain(canon.source_validation, canon.decode, canon.resample)
    _validate_canonical_domain(canon.canonical_validation)


def _validate_downmix_range(downmix: Downmix) -> None:
    """Channel counts above two would need a layout policy this benchmark has not analysed."""
    if downmix.channels_min != 1 or downmix.channels_max != 2:
        raise ProtocolValueError(
            "canonicalisation.downmix must accept 1 to 2 channels; averaging arbitrary channel "
            "counts is not a defensible layout policy and every training recording is stereo"
        )


def _validate_source_domain(source: SourceValidation, decode: Decode, resample: Resample) -> None:
    """The source bound follows from the permitted subtypes rather than being imposed on them."""
    if not (
        source.require_finite and source.require_non_empty and source.require_positive_sample_rate
    ):
        raise ProtocolValueError(
            "canonicalisation.source_validation must require finite, non-empty samples at a "
            "positive rate; each of those is assumed by everything downstream"
        )
    if source.amplitude_min != -1.0 or source.amplitude_max != 1.0:
        raise ProtocolValueError(
            "canonicalisation.source_validation must bound source amplitude to [-1.0, 1.0], "
            "which is what integer PCM decoding guarantees by construction"
        )
    if source.amplitude_bounds_inclusive is not True:
        raise ProtocolValueError(
            "canonicalisation.source_validation.amplitude_bounds_inclusive must be true; "
            "PCM reaches exactly -1.0 at the negative rail, so an exclusive bound would "
            "reject legitimate full-scale material"
        )
    if source.min_sample_rate != resample.target_sample_rate:
        raise ProtocolValueError(
            f"canonicalisation.source_validation.min_sample_rate {source.min_sample_rate} must "
            f"equal the canonical rate {resample.target_sample_rate}; a lower floor admits "
            "sources that cannot supply the bandwidth being measured, and it also breaks the "
            "live-memory derivation, which relies on the canonical waveform being no larger "
            "than the downmix"
        )
    for name, value in (
        ("max_source_bytes", decode.max_source_bytes),
        ("max_decoded_bytes", decode.max_decoded_bytes),
        ("max_projected_live_bytes", decode.max_projected_live_bytes),
        ("max_identity_stream_bytes", decode.max_identity_stream_bytes),
        ("max_source_duration_s", decode.max_source_duration_s),
    ):
        if value <= 0:
            raise ProtocolValueError(f"canonicalisation.decode.{name} must be positive")
    if decode.require_regular_file is not True:
        raise ProtocolValueError(
            "canonicalisation.decode.require_regular_file must be true; fstat reports nothing "
            "meaningful for a FIFO, a device, or a virtual filesystem object, so the size check "
            "would be performed on a number that means nothing"
        )
    if decode.canonicalisation_concurrency != 1:
        raise ProtocolValueError(
            f"canonicalisation.decode.canonicalisation_concurrency is "
            f"{decode.canonicalisation_concurrency}; the live-memory budget is per process, so "
            "any value above one multiplies the requirement and the declared budget no longer "
            "bounds the machine"
        )
    if decode.max_identity_stream_bytes <= decode.max_source_bytes:
        raise ProtocolValueError(
            "canonicalisation.decode.max_identity_stream_bytes must exceed max_source_bytes; "
            "the streaming digest exists for artifacts already rejected as too large, so a "
            "bound at or below the size that triggers the rejection could never apply"
        )
    expected_stream = _MAX_IDENTITY_STREAM_MULTIPLE * decode.max_source_bytes
    if decode.max_identity_stream_bytes != expected_stream:
        raise ProtocolValueError(
            f"canonicalisation.decode.max_identity_stream_bytes "
            f"{decode.max_identity_stream_bytes} must equal {_MAX_IDENTITY_STREAM_MULTIPLE}x "
            f"max_source_bytes ({expected_stream}); this is a resource-security limit, and an "
            "accepted range would leave a band of configurations that are permitted and have "
            "never been analysed"
        )
    if decode.max_decoded_bytes <= decode.max_source_bytes:
        raise ProtocolValueError(
            "canonicalisation.decode.max_decoded_bytes must exceed max_source_bytes; decoding "
            "expands, measured at 4.85x for FLAC PCM_24 to float64, so a decoded bound at or "
            "below the file bound would bound the smaller allocation and admit the larger"
        )
    # The downmix is never larger than the decoded array, since frames*8 is at most
    # frames*channels*8 for one or more channels, and the canonical waveform is never larger
    # than the downmix, since min_sample_rate equals the target rate. So the worst set alive at
    # once is bounded by max_source_bytes + 3 * max_decoded_bytes, for every admitted channel
    # count. An earlier draft used 2 * max_decoded_bytes, which silently assumed two channels:
    # at one channel the downmix equals the decoded array instead of halving it.
    # Checked as a relationship between frozen bounds rather than per source: a source any
    # conforming implementation could process must not be ruled ineligible because this machine
    # was small, since capacity is ours and not a property of the source.
    worst_live = decode.max_source_bytes + 3 * decode.max_decoded_bytes
    if worst_live > decode.max_projected_live_bytes:
        raise ProtocolValueError(
            f"canonicalisation.decode bounds admit a worst live set of {worst_live} bytes against "
            f"a declared budget of {decode.max_projected_live_bytes}; bounding the pieces "
            "individually does not bound the process"
        )


def _validate_canonical_domain(canonical: CanonicalValidation) -> None:
    """No amplitude bound here, deliberately, and the loader refuses to have one added."""
    if canonical.ndim != 1:
        raise ProtocolValueError("canonicalisation.canonical_validation.ndim must be 1")
    if not (canonical.require_finite and canonical.require_non_empty):
        raise ProtocolValueError(
            "canonicalisation.canonical_validation must require finite, non-empty samples"
        )


def verify_canonicalisation_environment(protocol: Protocol) -> None:
    """Check that this runtime is the one the canonical waveform is defined by.

    Separate from semantic validation because "is this a valid v4 protocol" and "may this
    machine authoritatively execute it" are different questions. `load_protocol` calls this,
    with no way to switch it off, so the normal path stays fail-closed; canonicalisation and
    the CI conformance job call it directly for the same reason under their own names.

    Only the Python packages are enforced. The lockfile makes those deterministic, while
    native libsndfile and libsoxr vary by platform and are recorded as provenance instead.
    Package metadata is read rather than the modules imported, because importing librosa on
    every protocol load would be expensive.

    Args:
        protocol: A parsed protocol whose semantics have already been validated.

    Raises:
        ProtocolEnvironmentError: If a pinned package is missing or is a different version.
    """
    canon = protocol.canonicalisation
    for package, pinned in (
        ("soundfile", canon.decode.library_version),
        ("librosa", canon.resample.library_version),
        ("soxr", canon.resample.soxr_version),
    ):
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as e:
            raise ProtocolEnvironmentError(f"{package} is not installed") from e
        if installed != pinned:
            raise ProtocolEnvironmentError(
                f"protocol pins {package} {pinned} but the runtime has {installed}; the "
                "canonical waveform depends on it, so every content hash would differ"
            )


def _validate_framing(framing: Framing, canon: Canonicalisation, conditioning_rate: int) -> None:
    """Frame geometry, and its agreement with the rest of the protocol.

    The duration is checked against the config's own `frame_duration_s` rather than against a
    constant here. That keeps the magnitude in the protocol, where its digest pins it, while
    still making the claim machine-checkable: a config asserting two seconds cannot carry ten
    seconds of samples. Prose alone cannot enforce that, because code cannot read prose.

    `frame_duration_s` carries no separate positivity check: `frame_samples > 0` and a
    positive rate already force it through the equality below.
    """
    if framing.frame_samples <= 0 or framing.hop_samples <= 0:
        raise ProtocolValueError("framing frame_samples and hop_samples must be positive")
    if canon.resample.target_sample_rate != conditioning_rate:
        raise ProtocolValueError(
            f"canonicalisation.resample.target_sample_rate {canon.resample.target_sample_rate} "
            f"and conditioning.sample_rate {conditioning_rate} name the same fact and disagree; "
            "two authorities for one value is how they drift"
        )
    if framing.frame_samples != framing.frame_duration_s * conditioning_rate:
        raise ProtocolValueError(
            f"framing.frame_samples {framing.frame_samples} is not "
            f"{framing.frame_duration_s} s at {conditioning_rate} Hz"
        )
    if framing.hop_samples != framing.frame_samples:
        raise ProtocolValueError(
            f"framing.hop_samples {framing.hop_samples} must equal frame_samples "
            f"{framing.frame_samples}; evaluation frames do not overlap, and an overlapping "
            "hop would count the same audio into several frames of one denominator"
        )
    if framing.first_frame_start_sample != 0:
        raise ProtocolValueError(
            "framing.first_frame_start_sample must be 0; a nonzero start is an undeclared "
            "alignment choice"
        )
    if framing.partial_trailing_frame != _PARTIAL_TRAILING_FRAME:
        raise ProtocolValueError(
            f"framing.partial_trailing_frame must be {_PARTIAL_TRAILING_FRAME!r}, got "
            f"{framing.partial_trailing_frame!r}; a remainder that is neither scored nor "
            "counted hides how the candidate set was formed"
        )


def _validate_source_outcomes(outcomes: SourceOutcomes) -> None:
    """Failure classification, frozen before the first unreadable file is met.

    Both vocabularies are matched exactly against the code, which is what makes a harness
    failure structurally unable to appear as a source verdict. Their disjointness, the
    ordering that puts the golden fixture before any candidate, and the presence of the
    identity stage are all properties of those code tuples once exact equality holds, so they
    are asserted over the constants in the test suite rather than re-checked here. A check
    that no mutation can reach is not a control; the mutation sweep is what surfaced that
    three such checks had been written.
    """
    for field, actual, expected in (
        ("attribution_rule", outcomes.attribution_rule, _ATTRIBUTION_RULE),
        ("precondition_order", outcomes.precondition_order, _PRECONDITION_ORDER),
        ("local_identity_basis", outcomes.local_identity_basis, _LOCAL_IDENTITY_BASIS),
        ("ineligible_reasons", outcomes.ineligible_reasons, _INELIGIBLE_REASONS),
        ("abort_reasons", outcomes.abort_reasons, _ABORT_REASONS),
        ("reduction_chain", outcomes.reduction_chain, _REDUCTION_CHAIN),
        ("record_required_core", outcomes.record_required_core, _RECORD_REQUIRED_CORE),
        (
            "record_conditional",
            tuple(outcomes.record_conditional.items()),
            _RECORD_CONDITIONAL,
        ),
        ("identity_status_values", outcomes.identity_status_values, _IDENTITY_STATUS_VALUES),
        ("reason_selection", outcomes.reason_selection, _REASON_SELECTION),
        ("evaluation_stage_order", outcomes.evaluation_stage_order, _EVALUATION_STAGE_ORDER),
        (
            "stage_checks",
            tuple((name, checks) for name, checks in outcomes.stage_checks.items()),
            _STAGE_CHECKS,
        ),
        ("identity_method_values", outcomes.identity_method_values, _IDENTITY_METHOD_VALUES),
        ("identity_derivation", outcomes.identity_derivation, _IDENTITY_DERIVATION),
        ("reason_stage", tuple(outcomes.reason_stage.items()), _REASON_STAGE),
        (
            "unevaluated_stages_derivation",
            outcomes.unevaluated_stages_derivation,
            _UNEVALUATED_STAGES_DERIVATION,
        ),
        ("record_scope", outcomes.record_scope, _RECORD_SCOPE),
        ("record_constraints", outcomes.record_constraints, _RECORD_CONSTRAINTS),
        ("reason_precedence", outcomes.reason_precedence, _REASON_PRECEDENCE),
        ("supplemental_reasons", outcomes.supplemental_reasons, _SUPPLEMENTAL_REASONS),
        (
            "identity_for_excluded_sources",
            outcomes.identity_for_excluded_sources,
            _EXCLUDED_SOURCE_IDENTITY,
        ),
        ("artifact_stability_check", outcomes.artifact_stability_check, _ARTIFACT_STABILITY_CHECK),
        ("diagnostic_handling", outcomes.diagnostic_handling, _DIAGNOSTIC_HANDLING),
        (
            "reason_categories",
            tuple((name, codes) for name, codes in outcomes.reason_categories.items()),
            _REASON_CATEGORIES,
        ),
    ):
        if actual != expected:
            raise ProtocolValueError(
                f"source_outcomes.{field} must be {expected!r} in v4, got {actual!r}"
            )
    if outcomes.unevaluated_stages_recorded is not True:
        raise ProtocolValueError(
            "source_outcomes.unevaluated_stages_recorded must be true; without it the absence "
            "of a reason reads as evidence that the check passed, when the stage may never "
            "have been reached"
        )
    if outcomes.upstream_checksum_available is not False:
        raise ProtocolValueError(
            "source_outcomes.upstream_checksum_available must be false; the collection "
            "publishes no checksum and the acquisition script records none, so claiming one "
            "would let publication assert upstream identity the run cannot establish"
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
    if el.rms_operation != _RMS_OPERATION:
        raise ProtocolValueError(
            f"eligibility.rms_operation must be {_RMS_OPERATION!r}; an epsilon or a different "
            "reduction changes which frames sit either side of the floor"
        )
    if el.computed_from != _ELIGIBILITY_COMPUTED_FROM:
        raise ProtocolValueError(
            f"eligibility.computed_from must be {_ELIGIBILITY_COMPUTED_FROM!r}; deciding "
            "eligibility from anything but the canonical waveform means framing, RMS, and "
            "hashes could each be computed over a different representation"
        )
    for name, comparator in (
        ("recording_rms_comparator", el.recording_rms_comparator),
        ("frame_rms_comparator", el.frame_rms_comparator),
    ):
        if comparator != _RMS_COMPARATOR:
            raise ProtocolValueError(
                f"eligibility.{name} must be {_RMS_COMPARATOR!r}; the document says "
                "'above' the floor and code must not infer strictness from English"
            )
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
    _validate_canonicalisation(protocol.canonicalisation)
    verify_canonicalisation_environment(protocol)
    _validate_framing(
        protocol.framing, protocol.canonicalisation, protocol.conditioning.sample_rate
    )
    _validate_source_outcomes(protocol.source_outcomes)
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
