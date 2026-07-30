"""Make a system's output comparable to the clean reference, or say why it is not.

This layer exists for one distinction, and the type signature carries it rather than a
status field. A returned value is an observation about the evaluation item. A raised
exception is the evaluator saying it cannot produce an observation at all.

    condition_pair(...) -> ConditionedPair | InvalidSystemOutput
    ConditioningError                       raised, never returned

Collapsing those is the failure mode. A system that deterministically cannot process an
eligible item is a benchmark result and belongs in that system's coverage; a harness
defect invalidates the run and must not burn the test partition. If both arrived as
return values, a caller writing the obvious loop over results would fold evaluator
failures into the population, and coverage would stop meaning what it claims.

Attribution follows ownership of the artifact. The reference is ours, so a malformed
reference is a harness failure. The estimate is the system's, so a malformed estimate is a
result about that system. This function cannot verify that provenance from its arguments,
so it is the caller's obligation: invoke it only once the system artifact has been
successfully acquired and decoded, because a mistyped variable passed as `estimate` would
otherwise be blamed on the system.

What a `ConditionedPair` promises is structural comparability, and nothing more. It does
not promise the pair is scoreable, which is not even a single property: a constant
reference is scored by reconstruction SNR and rejected by SI-SNR under protocol v3, and a
silent reference is rejected by both. Per-metric mathematical validity belongs to the
metrics, which own those domains, and duplicating them here would put the same rule in two
places. The consequence is an obligation further up: when a metric rejects a pair that
conditioned successfully, the orchestration layer must attribute it, and reference-side
domain failures are dataset or harness failures rather than failures of the system whose
output happened to be paired with them.

Length is the substance. The protocol permits a difference of at most
`max_length_difference_samples`, applied symmetrically, because the longer side is
truncated to the shorter whichever side that is. Beyond that the item is invalid for the
system rather than being truncated silently, which would let a system escape being scored
on a missing tail.

There is no alignment search and no parameter that could request one. Outputs are
compared at the offset they arrive with. Front-alignment plus truncation is the whole
policy, deliberately, because an "alignment fix" is indistinguishable from shifting
outputs until the metric improves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.protocol import Conditioning


class ConditioningError(Exception):
    """Raised when the harness cannot establish a result for an item.

    Never returned, so a harness defect cannot be mistaken for a system observation and
    cannot enter an aggregate. The reference is the harness's own artifact, so a defect
    in it lands here rather than being charged to the system under evaluation.
    """


@dataclass(frozen=True)
class ConditionedPair:
    """A comparable pair, with what conditioning discarded to get there.

    Both discard counts are recorded even though one is always zero, because a single
    combined figure would not say which side was long. Provenance that cannot answer
    "whose tail was dropped" answers nothing useful later.


    The arrays are copies, not views, and are read-only. `frozen=True` only stops a field
    being rebound; without copying, the result would alias the caller's buffers and a later
    write to either would silently rewrite a recorded result. Sample counts rather than
    durations are the canonical stored form of what the protocol calls discarded duration,
    because counts are lossless and the duration follows from the frozen sample rate.
    """

    reference: np.ndarray
    estimate: np.ndarray
    reference_samples_discarded: int
    estimate_samples_discarded: int


@dataclass(frozen=True)
class InvalidSystemOutput:
    """The system's output cannot be scored, which is a result about the system.

    Feeds coverage as a system failure. Never dropped from the population: coverage
    below complete is published as a reliability finding, because a comparison over
    whichever items both systems happened to survive rewards failure.
    """

    reason: str
    reference_minus_estimate_samples: int


ConditioningOutcome = ConditionedPair | InvalidSystemOutput


def _harness_checked_reference(reference: np.ndarray) -> np.ndarray:
    """Validate the reference, whose defects belong to the harness.

    Raises:
        ConditioningError: If the reference is not a one-dimensional, non-empty, finite
            waveform. The benchmark produced it, so no system is answerable for it.
    """
    try:
        clean = np.asarray(reference, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ConditioningError(f"reference is not numeric: {e}") from e
    if clean.ndim != 1:
        raise ConditioningError(f"reference must be 1-D, got shape {clean.shape}")
    if clean.size == 0:
        raise ConditioningError("reference is empty")
    if not np.all(np.isfinite(clean)):
        raise ConditioningError("reference contains non-finite samples")
    return clean


def condition_pair(
    reference: np.ndarray, estimate: np.ndarray, config: Conditioning
) -> ConditioningOutcome:
    """Truncate a pair to a comparable length, or report it invalid for the system.

    Args:
        reference: The clean reference for this item.
        estimate: The system's output for the same item.
        config: The frozen `conditioning` section.

    Returns:
        A `ConditionedPair` when the lengths differ by at most the configured tolerance,
        with the longer side front-aligned and truncated to the shorter. Otherwise an
        `InvalidSystemOutput` describing why.

    Raises:
        ConditioningError: If the reference is malformed, or the configuration requests
            an alignment search this implementation does not perform.
    """
    if config.alignment_search:
        # Unreachable while the loader pins this false, and stated anyway: silently
        # ignoring a request to search would be worse than refusing it.
        raise ConditioningError(
            "config requests an alignment search; the protocol forbids one and this "
            "implementation performs none"
        )

    clean = _harness_checked_reference(reference)

    try:
        restored = np.asarray(estimate, dtype=np.float64)
    except (TypeError, ValueError) as e:
        # The harness holds the artifact and can attribute the defect to it.
        return InvalidSystemOutput(
            reason=f"output is not numeric: {e}", reference_minus_estimate_samples=0
        )
    if restored.ndim != 1:
        return InvalidSystemOutput(
            reason=f"output must be 1-D, got shape {restored.shape}",
            reference_minus_estimate_samples=0,
        )
    if restored.size == 0:
        return InvalidSystemOutput(
            reason="output is empty", reference_minus_estimate_samples=int(clean.size)
        )
    if not np.all(np.isfinite(restored)):
        return InvalidSystemOutput(
            reason="output contains non-finite samples", reference_minus_estimate_samples=0
        )

    difference = int(clean.size) - int(restored.size)
    if abs(difference) > config.max_length_difference_samples:
        return InvalidSystemOutput(
            reason=(
                f"length difference {abs(difference)} exceeds the "
                f"{config.max_length_difference_samples}-sample tolerance; truncating "
                "further would score the system on a tail it never produced"
            ),
            reference_minus_estimate_samples=difference,
        )

    # Front-aligned, truncate the longer to the shorter. Symmetric because the protocol
    # truncates "the longer" without naming which side that is.
    keep = min(clean.size, restored.size)
    conditioned_reference = np.array(clean[:keep], dtype=np.float64, copy=True)
    conditioned_estimate = np.array(restored[:keep], dtype=np.float64, copy=True)
    conditioned_reference.flags.writeable = False
    conditioned_estimate.flags.writeable = False
    return ConditionedPair(
        reference=conditioned_reference,
        estimate=conditioned_estimate,
        reference_samples_discarded=int(clean.size) - keep,
        estimate_samples_discarded=int(restored.size) - keep,
    )
