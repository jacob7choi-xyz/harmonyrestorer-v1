"""The three co-primary signal-fidelity metrics, exactly as protocol v3 defines them.

Every configurable constant arrives as a frozen protocol section. Nothing is defaulted
here, because a default in this module would be a second authority for a value the
protocol already fixes, and the two would eventually disagree.

Two rules govern the numerics and both exist because breaking them produces a plausible
number rather than an error.

An invalid item raises instead of returning a score. Epsilons and spectral offsets exist
for numerical stability on valid inputs; they must never widen the valid domain. Left
alone, `log_spectral_distance` on two silent frames returns 0 dB, which reads as perfect
restoration, and a zero-energy reference divides reconstruction SNR by nothing.

Caps are recorded against the raw value before clamping. Deriving `capped` from the
returned number is impossible once it has been clamped to exactly the cap, so a cap that
binds on every item would report binding on none. Whether the cap binds is diagnostic:
for reconstruction SNR to reach 60 dB the residual must be a millionth of the signal,
which no denoiser approaches on real audio, so a nonzero rate means something degenerate
happened upstream rather than that restoration was excellent.

There are two censoring paths for reconstruction SNR and both set `capped`. A residual
below `residual_floor` is censored by the protocol's own floor rule, with no raw value to
compare. A residual above the floor whose ratio still exceeds `cap_db` is censored by the
cap. They are tested separately because only the second is a clamp of a computed number.

Zero-mean SI-SNR is undefined when the reference is constant, since centring then leaves
no direction to project onto while the raw energy check passes. Protocol v3 makes that a
normative invalidity rule rather than leaving epsilon to decide, because whether an item
receives a score determines the valid population and two conforming implementations must
not disagree on it. The condition is stated as constancy rather than as zero energy after
centring: those look equivalent and are not, because whether a constant array centres to
exactly zero depends on whether its value is binary-representable. The state is expected
to be unreachable for admitted references, and is rejected independently of that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.protocol import LogSpectralDistance, ReconstructionSnr, SiSnr

# The dB factor follows from the named operation rather than being configurable, because
# 20 against 10 distinguishes amplitude spectra from power spectra and is a kind of
# measurement rather than a magnitude anyone should be free to choose.
_DB_MULTIPLIER = {"amplitude_db_additive_offset": 20.0}


class MetricInputError(ValueError):
    """Raised when a pair cannot be scored at all.

    Distinct from a poor score. A caller receiving this has an invalid benchmark item,
    not a system that performed badly, and the two must not be conflated: silently
    scoring an invalid item is how a degenerate frame becomes a flattering number.
    """


@dataclass(frozen=True)
class MetricResult:
    """A metric value together with whether the protocol's cap censored it."""

    value: float
    capped: bool


def _validated_pair(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Check a pair is scoreable and return it as float64.

    Args:
        reference: Clean reference signal.
        estimate: A system's output for the same item.

    Returns:
        Both signals as float64 arrays, unmodified otherwise.

    Raises:
        MetricInputError: If either signal is not one-dimensional, they differ in
            length, either is empty or non-finite, or the reference carries no energy.
    """
    # Convert first, then inspect the converted arrays. Checking one representation and
    # then indexing another is how a list argument turns into an AttributeError instead
    # of a MetricInputError. Converting before arithmetic also matters on its own:
    # subtracting and squaring int16 overflows silently.
    try:
        clean = np.asarray(reference, dtype=np.float64)
        restored = np.asarray(estimate, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise MetricInputError(f"signals must be numeric: {e}") from e

    for name, signal in (("reference", clean), ("estimate", restored)):
        if signal.ndim != 1:
            # Never reshape. A stereo array flattens into something that looks like
            # mono, and interleaved samples would score as a signal.
            raise MetricInputError(f"{name} must be 1-D, got shape {signal.shape}")
        if signal.size == 0:
            raise MetricInputError(f"{name} is empty")

    if clean.shape != restored.shape:
        raise MetricInputError(
            f"length mismatch: reference {clean.shape[0]}, estimate {restored.shape[0]}. "
            "Conditioning owns alignment; metrics do not resize."
        )

    for name, signal in (("reference", clean), ("estimate", restored)):
        if not np.all(np.isfinite(signal)):
            raise MetricInputError(f"{name} contains non-finite samples")

    if not float(np.sum(clean**2)) > 0.0:
        raise MetricInputError(
            "reference carries no energy; a silent reference has no fidelity to measure "
            "and would divide reconstruction SNR by nothing"
        )
    return clean, restored


def _capped(raw: float, cap_db: float) -> MetricResult:
    """Apply a cap, recording whether it censored the raw value."""
    return MetricResult(value=min(raw, cap_db), capped=raw > cap_db)


def reconstruction_snr(
    reference: np.ndarray, estimate: np.ndarray, config: ReconstructionSnr
) -> MetricResult:
    """Sample-wise fidelity in dB. Higher is better.

    Penalises gain error, timing error, and residual noise alike, because no projection
    or filtering is permitted. A one-sample shift degrades it substantially, which is
    why the protocol forbids alignment search.

    Args:
        reference: Clean reference signal.
        estimate: A system's output for the same item.
        config: The frozen `metrics.reconstruction_snr` section.

    Returns:
        The value with its cap state.

    Raises:
        MetricInputError: If the pair is not scoreable.
    """
    clean, restored = _validated_pair(reference, estimate)
    residual = float(np.sum((clean - restored) ** 2))
    if residual < config.residual_floor:
        # Indistinguishable from the reference. Reported at the cap and marked censored,
        # rather than dividing by a residual that is numerically zero.
        return MetricResult(value=config.cap_db, capped=True)
    raw = 10.0 * np.log10(float(np.sum(clean**2)) / residual)
    return _capped(float(raw), config.cap_db)


def si_snr(reference: np.ndarray, estimate: np.ndarray, config: SiSnr) -> MetricResult:
    """Fidelity with global gain projected out. Higher is better.

    Blind to level mismatch by construction, which is the point: it is reported beside
    reconstruction SNR so that a system emitting `0.3 x clean` scores well here and
    badly there, separating a gain error from a reconstruction error. The zero-mean
    formulation is frozen because "standard SI-SNR" is not one thing.

    Args:
        reference: Clean reference signal.
        estimate: A system's output for the same item.
        config: The frozen `metrics.si_snr` section.

    Returns:
        The value with its cap state.

    Raises:
        MetricInputError: If the pair is not scoreable.
    """
    clean, restored = _validated_pair(reference, estimate)

    if config.constant_reference != "invalid_item":
        raise NotImplementedError(
            "this implementation only invalidates the item, config asks for "
            f"{config.constant_reference!r}"
        )
    # Tested before centring, on constancy itself, because that is the mathematical
    # condition. Whether a constant array centres to exactly zero depends on whether its
    # value is binary-representable: 0.5 and 0.25 do, while 0.1 and 0.3 leave residues
    # near 1e-30 that a post-centring zero check accepts and then scores at 0.0 dB.
    # Constancy is exact and needs no tolerance.
    if config.zero_mean and bool(np.all(clean == clean[0])):
        raise MetricInputError(
            "reference is constant, so centring leaves no direction for SI-SNR to "
            "project onto; epsilon may not make this scoreable (protocol v3)"
        )

    if config.zero_mean:
        clean = clean - clean.mean()
        restored = restored - restored.mean()

    # Fail-closed on input the frozen formula cannot be evaluated over, which is not the
    # same claim as SI-SNR being undefined. A non-constant reference always keeps a nonzero
    # centred vector, so a projection direction mathematically exists, but its energy can
    # still underflow: samples near 1e-150 differing by 1e-165 leave residues squaring to
    # about 1e-331. That is a representation limit, not a domain fact, so it is deliberately
    # not a protocol population rule. For an admitted reference it is unreachable, and by
    # argument rather than by margin: a sum of non-negative terms reaches exactly zero only
    # if every term underflows, so every residue would have to fall below 2.2e-162, which
    # makes the frame near-constant and forces the mean to carry the whole RMS. At the -45
    # dBFS floor that mean is near 5.6e-3, where two distinct doubles differ by at least one
    # ULP of 8.7e-19, some 144 orders above the required residue. So every sample would have
    # to be the identical double, which is exactly the constant case rejected above.
    # Continuing here would hand epsilon-dominated arithmetic to the caller as a score.
    if not float(np.dot(clean, clean)) > 0.0:
        raise MetricInputError(
            "reference has no energy after centring, so SI-SNR has no projection direction"
        )

    epsilon = config.epsilon
    # Epsilon in this denominator can no longer prevent a division by zero, because v3
    # rejects the only input that produced one. It is preserved because it is part of the
    # frozen formula and still shifts the result for a very quiet reference, where the
    # centred energy is comparable to epsilon itself.
    projection = float(np.dot(restored, clean)) / (float(np.dot(clean, clean)) + epsilon)
    target = projection * clean
    residual = restored - target

    raw = 10.0 * np.log10(
        (float(np.sum(target**2)) + epsilon) / (float(np.sum(residual**2)) + epsilon)
    )
    return _capped(float(raw), config.cap_db)


def periodic_hann(length: int) -> np.ndarray:
    """The analysis window, from the protocol's equation rather than from a library.

    `w[n] = 0.5 - 0.5*cos(2*pi*n/length)` for `0 <= n < length`. The symmetric
    convention divides by `length - 1`, answers to the same name, and is a different
    window: `numpy.hanning` returns that one. Implementing the equation directly leaves
    library implementations available as independent cross-checks instead of as the
    definition.
    """
    n = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)


def stft_magnitude(signal: np.ndarray, config: LogSpectralDistance) -> np.ndarray:
    """Magnitude STFT under the frozen transform.

    Args:
        signal: One-dimensional float64 signal.
        config: The frozen `metrics.log_spectral_distance` section.

    Returns:
        Magnitudes shaped `(frames, bins)`. For the 2-second evaluation unit at 16 kHz,
        32000 samples padded by 512 each side gives 33024, which with a 1024-sample
        window, a hop of 256, and no partial trailing frame is 126 frames of 513 bins.
    """
    framing = config.framing
    # This module implements one transform. The protocol already pins these three, so the
    # check is defensive, but stating it here makes the dependency explicit instead of
    # letting a config string flow into library calls that accept only certain values.
    supported = ("rfft", "backward", "constant")
    if (config.fft.kind, config.fft.norm, framing.pad_mode) != supported:
        raise NotImplementedError(
            f"this implementation covers {supported} only, config asks for "
            f"{(config.fft.kind, config.fft.norm, framing.pad_mode)}"
        )
    padded = np.pad(
        signal,
        (framing.pad_left_samples, framing.pad_right_samples),
        mode="constant",
        constant_values=framing.pad_value,
    )
    window = periodic_hann(config.win_length)
    # Frame starts at multiples of hop from zero, and only whole windows are analysed:
    # a partial trailing frame would be measured against a differently shaped window.
    frame_count = 1 + (padded.size - config.win_length) // config.hop_length
    frames = np.stack(
        [
            padded[start : start + config.win_length] * window
            for start in range(0, frame_count * config.hop_length, config.hop_length)
        ]
    )
    return np.abs(np.fft.rfft(frames, n=config.n_fft, axis=1, norm="backward"))


def log_spectral_distance(
    reference: np.ndarray, estimate: np.ndarray, config: LogSpectralDistance
) -> MetricResult:
    """Spectral magnitude fidelity in dB. Lower is better, and blind to phase.

    The offset is additive, not a clamp: `max(|X|, offset)` is a different metric. The
    reduction order is load-bearing too, RMS across bins and then a mean across frames,
    which is not the same statistic as the reverse.

    Args:
        reference: Clean reference signal.
        estimate: A system's output for the same item.
        config: The frozen `metrics.log_spectral_distance` section.

    Returns:
        The value, never censored, since no cap is defined for a distance.

    Raises:
        MetricInputError: If the pair is not scoreable.
        KeyError: If the config names a log operation this module does not implement.
    """
    clean, restored = _validated_pair(reference, estimate)
    multiplier = _DB_MULTIPLIER[config.log_magnitude.operation]
    offset = config.log_magnitude.offset

    clean_db = multiplier * np.log10(stft_magnitude(clean, config) + offset)
    restored_db = multiplier * np.log10(stft_magnitude(restored, config) + offset)

    per_frame = np.sqrt(np.mean((clean_db - restored_db) ** 2, axis=1))
    return MetricResult(value=float(np.mean(per_frame)), capped=False)
