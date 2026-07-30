"""Tests for the production protocol loader.

Every test here loads through `benchmark.protocol.load_protocol`. Nothing in this
file reimplements loading, validation, or metric behaviour, because a test that
recreates the thing it checks proves only that the test agrees with itself.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.protocol import (
    _PROTOCOL_PATH,
    EFFECT_FOR_DIRECTION,
    FROZEN_GATES,
    ProtocolEnvironmentError,
    ProtocolSchemaError,
    ProtocolValueError,
    _load_from_path,
    load_protocol,
)

REPO_ROOT = _PROTOCOL_PATH.parents[2]

# Digests of every superseded artifact, as frozen at the commit that archived it. A
# tripwire, not a trust root:
# git history and review establish which protocol was approved. This exists so that
# editing an archived artifact together with the digest recorded in v2 fails here
# rather than passing quietly.
SUPERSEDED_ARTIFACTS = {
    # Content digests of public, checked-in files. The allowlist pragmas are for the
    # secrets scanner, which cannot tell a SHA-256 from a 64-character credential.
    "benchmark/protocols/v1.json": (
        "b25a964a6a3415a998959d63f0222fe45e207629a90048c3226635fe9e5f9efd"  # pragma: allowlist secret
    ),
    "docs/benchmark-protocol-v1.md": (
        "8a24244281fe9a20e58b1558f503f0974b45583df853937ffc1d7253fcca45e2"  # pragma: allowlist secret
    ),
    "benchmark/protocols/v2.json": (
        "f683218f6eb313a042f0d004c117753ce93827321d5a19dedd2f7071bf6981de"  # pragma: allowlist secret
    ),
    "docs/benchmark-protocol-v2.md": (
        "51d25785235b075bc5b4489163a4d766d2f5b2a066615369b33b7247a14d0df6"  # pragma: allowlist secret
    ),
}


def digest_of(relative_path: str) -> str:
    """SHA-256 of a repository file, read as bytes."""
    return hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()


def variant(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Write a copy of the real config with one deliberate change applied."""
    data = json.loads(_PROTOCOL_PATH.read_text())
    mutate(data)
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(data))
    return path


class TestRealProtocol:
    """The checked-in config must load and describe itself accurately."""

    def test_it_loads(self) -> None:
        assert load_protocol().protocol_version == 3

    def test_digest_is_of_the_exact_file_bytes(self) -> None:
        """The provenance claim is 'these bytes governed this run', so the digest
        must come from the file as read, not from a reserialisation of it."""
        expected = hashlib.sha256(_PROTOCOL_PATH.read_bytes()).hexdigest()
        assert load_protocol().source_sha256 == expected

    def test_pinned_numpy_matches_the_runtime(self) -> None:
        """A pin that no longer matches the environment would block every later
        benchmark step, so it is worth failing here with a clear reason."""
        assert load_protocol().environment.numpy_version == np.__version__

    def test_paired_effects_are_the_intended_ones(self) -> None:
        metrics = load_protocol().metrics
        assert metrics.metric_config("reconstruction_snr").paired_effect == "opgan_minus_uvr"
        assert metrics.metric_config("si_snr").paired_effect == "opgan_minus_uvr"
        assert metrics.metric_config("log_spectral_distance").paired_effect == "uvr_minus_opgan"

    def test_threshold_sits_inside_its_own_calibration_evidence(self) -> None:
        dd = load_protocol().duplicate_detection
        ceiling = max(dd.calibration.negatives_max, dd.calibration.same_session_negatives_max)
        assert ceiling < dd.flag_threshold < dd.calibration.positives_min

    def test_unknown_metric_name_is_rejected(self) -> None:
        with pytest.raises(ProtocolValueError, match="not a configured primary metric"):
            load_protocol().metrics.metric_config("pesq")


class TestImmutability:
    """A security-critical config that can be edited mid-run is not a control."""

    def test_top_level_field_cannot_be_reassigned(self) -> None:
        protocol = load_protocol()
        with pytest.raises(dataclasses.FrozenInstanceError):
            protocol.protocol_version = 99  # type: ignore[misc]

    def test_nested_field_cannot_be_reassigned(self) -> None:
        protocol = load_protocol()
        with pytest.raises(dataclasses.FrozenInstanceError):
            protocol.aggregation.bootstrap.iterations = 10  # type: ignore[misc]

    def test_partitions_mapping_rejects_mutation(self) -> None:
        protocol = load_protocol()
        with pytest.raises(TypeError):
            protocol.selection.partitions["test"] = 1  # type: ignore[index]

    def test_sequences_are_tuples_not_lists(self) -> None:
        protocol = load_protocol()
        for value in (
            protocol.metrics.primary,
            protocol.metrics.legacy,
            protocol.publication_gates.required,
            protocol.selection.permanent_failure_codes,
            protocol.aggregation.across_tracks,
        ):
            assert isinstance(value, tuple)


class TestSchemaRejection:
    """A typo must fail loudly rather than fall back to a default."""

    def test_unknown_root_key(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d.update({"extra_setting": 1}))
        with pytest.raises(ProtocolSchemaError, match="unknown key"):
            _load_from_path(path)

    def test_unknown_nested_key(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["metrics"]["si_snr"].update({"cap": 60}))
        with pytest.raises(ProtocolSchemaError, match=r"metrics\.si_snr: unknown key"):
            _load_from_path(path)

    def test_missing_key(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["aggregation"]["bootstrap"].pop("seed"))
        with pytest.raises(ProtocolSchemaError, match="missing key"):
            _load_from_path(path)

    def test_missing_whole_section(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d.pop("reporting"))
        with pytest.raises(ProtocolSchemaError, match="missing key"):
            _load_from_path(path)

    def test_wrong_json_type(self, tmp_path: Path) -> None:
        """A quoted number parses fine and would otherwise reach a comparison
        that raises somewhere far from the cause."""
        path = variant(tmp_path, lambda d: d["selection"].update({"seed": "20260726"}))
        with pytest.raises(ProtocolSchemaError, match="wrong JSON type"):
            _load_from_path(path)

    def test_not_json(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        with pytest.raises(ProtocolSchemaError, match="not valid JSON"):
            _load_from_path(path)

    def test_top_level_not_an_object(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ProtocolSchemaError, match="must be a JSON object"):
            _load_from_path(path)


class TestEnvironmentEnforcement:
    """A version recorded in JSON but never checked is documentation, not a pin."""

    def test_numpy_version_mismatch_refuses_to_load(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["environment"].update({"numpy_version": "1.99.0"}))
        with pytest.raises(ProtocolEnvironmentError, match=r"pins NumPy 1\.99\.0"):
            _load_from_path(path)

    def test_substituting_the_bit_generator_is_rejected(self, tmp_path: Path) -> None:
        """MT19937 exists and would run, which is exactly why the name is pinned:
        the seeded-stream guarantee is specific to the named generator."""
        path = variant(tmp_path, lambda d: d["environment"].update({"bit_generator": "MT19937"}))
        with pytest.raises(ProtocolValueError, match="must be 'PCG64'"):
            _load_from_path(path)


class TestVersionCommitments:
    """Protocol promises are enforced at load, so weakening one requires a version bump."""

    def test_unsupported_version(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d.update({"protocol_version": 4}))
        with pytest.raises(ProtocolValueError, match="unsupported protocol_version 4"):
            _load_from_path(path)

    def test_composite_score_cannot_be_enabled(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["metrics"].update({"composite_score": True}))
        with pytest.raises(ProtocolValueError, match="composite_score must be false"):
            _load_from_path(path)

    def test_legacy_metrics_cannot_enter_the_headline(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["metrics"].update({"legacy_in_headline": True}))
        with pytest.raises(ProtocolValueError, match="legacy_in_headline must be false"):
            _load_from_path(path)

    def test_omnibus_winner_cannot_be_enabled(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["reporting"].update({"omnibus_winner": True}))
        with pytest.raises(ProtocolValueError, match="omnibus_winner must be false"):
            _load_from_path(path)

    def test_significance_language_cannot_be_enabled(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["reporting"].update({"significance_language": True}))
        with pytest.raises(ProtocolValueError, match="significance_language must be false"):
            _load_from_path(path)

    def test_co_primary_reporting_cannot_be_disabled(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path, lambda d: d["reporting"].update({"co_primary_reported_together": False})
        )
        with pytest.raises(ProtocolValueError, match="permits selective emphasis"):
            _load_from_path(path)

    def test_alignment_search_cannot_be_enabled(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["conditioning"].update({"alignment_search": True}))
        with pytest.raises(ProtocolValueError, match="fitting the metric"):
            _load_from_path(path)

    def test_uncertain_duplicate_verdict_cannot_admit(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path, lambda d: d["duplicate_detection"].update({"action_on_uncertain": "admit"})
        )
        with pytest.raises(ProtocolValueError, match="fail-open duplicate policy"):
            _load_from_path(path)

    def test_bootstrap_cannot_resample_frames(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["aggregation"]["bootstrap"].update({"unit": "frame"}))
        with pytest.raises(ProtocolValueError, match="overstates confidence"):
            _load_from_path(path)

    def test_bootstrap_cannot_be_unpaired(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["aggregation"]["bootstrap"].update({"paired": False}))
        with pytest.raises(ProtocolValueError, match="discards the pairing"):
            _load_from_path(path)

    def test_centred_reference_rule_cannot_be_weakened(self, tmp_path: Path) -> None:
        """v3's whole content. Anything other than invalidating the item lets epsilon
        return a finite score for a metric with no projection direction."""
        path = variant(
            tmp_path,
            lambda d: d["metrics"]["si_snr"].update({"constant_reference": "score_with_epsilon"}),
        )
        with pytest.raises(ProtocolValueError, match="no projection direction"):
            _load_from_path(path)

    def test_si_snr_formulation_is_frozen(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["metrics"]["si_snr"].update({"zero_mean": False}))
        with pytest.raises(ProtocolValueError, match="zero_mean must be true"):
            _load_from_path(path)


class TestEffectDirection:
    """A sign error inverts a conclusion while every calculation stays correct."""

    def test_effect_contradicting_direction_is_rejected(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path,
            lambda d: d["metrics"]["log_spectral_distance"].update(
                {"paired_effect": "opgan_minus_uvr"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="lower_better"):
            _load_from_path(path)

    def test_unknown_direction_is_rejected(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path, lambda d: d["metrics"]["si_snr"].update({"direction": "higher_beter"})
        )
        with pytest.raises(ProtocolValueError, match="direction must be one of"):
            _load_from_path(path)

    def test_every_direction_maps_to_exactly_one_effect(self) -> None:
        assert set(EFFECT_FOR_DIRECTION.values()) == {"opgan_minus_uvr", "uvr_minus_opgan"}
        assert len(EFFECT_FOR_DIRECTION) == 2


class TestCalibrationSupportsThreshold:
    """The duplicate threshold must lie inside the evidence offered to justify it."""

    def test_threshold_below_the_negatives_ceiling_is_rejected(self, tmp_path: Path) -> None:
        """0.05 sits under the loudest known non-duplicate pair, so it would flag
        recordings the calibration says are distinct."""
        path = variant(
            tmp_path, lambda d: d["duplicate_detection"].update({"flag_threshold": 0.05})
        )
        with pytest.raises(ProtocolValueError, match="outside its own evidence"):
            _load_from_path(path)

    def test_threshold_above_the_positives_floor_is_rejected(self, tmp_path: Path) -> None:
        """0.90 sits above the quietest known duplicate pair, so a real duplicate
        would pass unflagged."""
        path = variant(
            tmp_path, lambda d: d["duplicate_detection"].update({"flag_threshold": 0.90})
        )
        with pytest.raises(ProtocolValueError, match="outside its own evidence"):
            _load_from_path(path)

    def test_threshold_outside_cosine_range_is_rejected(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["duplicate_detection"].update({"flag_threshold": 1.5}))
        with pytest.raises(ProtocolValueError, match=r"must lie in \[-1, 1\]"):
            _load_from_path(path)


class TestNumericBounds:
    """Values that parse but cannot describe a runnable benchmark."""

    def test_empty_partition_is_rejected(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["selection"]["partitions"].update({"demo": 0}))
        with pytest.raises(ProtocolValueError, match=r"partitions\.demo must be positive"):
            _load_from_path(path)

    def test_confidence_at_one_is_rejected(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path, lambda d: d["aggregation"]["bootstrap"].update({"confidence": 1.0})
        )
        with pytest.raises(ProtocolValueError, match=r"confidence must lie in \(0, 1\)"):
            _load_from_path(path)

    def test_zero_bootstrap_iterations_is_rejected(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["aggregation"]["bootstrap"].update({"iterations": 0}))
        with pytest.raises(ProtocolValueError, match="iterations must be positive"):
            _load_from_path(path)

    def test_window_longer_than_the_transform_is_rejected(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path, lambda d: d["metrics"]["log_spectral_distance"].update({"win_length": 2048})
        )
        with pytest.raises(ProtocolValueError, match="exceeds n_fft"):
            _load_from_path(path)

    def test_hop_longer_than_the_window_is_rejected(self, tmp_path: Path) -> None:
        """A hop past the window leaves samples unanalysed, so the metric would
        silently score less than the signal it was handed."""
        path = variant(
            tmp_path, lambda d: d["metrics"]["log_spectral_distance"].update({"hop_length": 2048})
        )
        with pytest.raises(ProtocolValueError, match="leave samples unanalysed"):
            _load_from_path(path)

    def test_zero_log_magnitude_offset_is_rejected(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path,
            lambda d: d["metrics"]["log_spectral_distance"]["log_magnitude"].update(
                {"offset": 0.0}
            ),
        )
        with pytest.raises(ProtocolValueError, match=r"log_magnitude\.offset must be positive"):
            _load_from_path(path)

    def test_positive_dbfs_floor_is_rejected(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["eligibility"].update({"frame_rms_dbfs_min": 3.0}))
        with pytest.raises(ProtocolValueError, match="must be <= 0"):
            _load_from_path(path)

    def test_duplicate_primary_metric_is_rejected(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["metrics"].update({"primary": ["si_snr", "si_snr"]}))
        with pytest.raises(ProtocolValueError, match="primary contains duplicates"):
            _load_from_path(path)

    def test_metric_that_is_both_primary_and_legacy_is_rejected(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["metrics"].update({"legacy": ["pesq", "si_snr"]}))
        with pytest.raises(ProtocolValueError, match="legacy overlaps"):
            _load_from_path(path)


class TestPublicationGates:
    """The gate set is frozen in both directions, not merely floored."""

    @pytest.mark.parametrize("gate", sorted(FROZEN_GATES))
    def test_dropping_any_gate_is_rejected(self, tmp_path: Path, gate: str) -> None:
        def drop(d: dict[str, Any]) -> None:
            d["publication_gates"]["required"] = [
                g for g in d["publication_gates"]["required"] if g != gate
            ]

        path = variant(tmp_path, drop)
        with pytest.raises(ProtocolValueError, match="exactly the frozen v1 set"):
            _load_from_path(path)

    def test_duplicate_gate_is_rejected(self, tmp_path: Path) -> None:
        def duplicate(d: dict[str, Any]) -> None:
            d["publication_gates"]["required"].append("coverage_uvr_complete")

        path = variant(tmp_path, duplicate)
        with pytest.raises(ProtocolValueError, match="contains duplicates"):
            _load_from_path(path)

    def test_adding_a_gate_also_requires_a_version_bump(self, tmp_path: Path) -> None:
        """A stricter gate is not automatically a safer one. `minimum_effect_size_met`
        reads as rigour, but added once results exist it suppresses the runs that
        disagree with the desired conclusion and passes the ones that agree."""

        def add(d: dict[str, Any]) -> None:
            d["publication_gates"]["required"].append("minimum_effect_size_met")

        path = variant(tmp_path, add)
        with pytest.raises(ProtocolValueError, match="exactly the frozen v1 set"):
            _load_from_path(path)

    def test_renaming_a_gate_is_rejected(self, tmp_path: Path) -> None:
        """Equal count, different meaning: a set check catches what a length check
        would not."""

        def rename(d: dict[str, Any]) -> None:
            gates = d["publication_gates"]["required"]
            gates[gates.index("coverage_uvr_complete")] = "coverage_uvr_mostly_complete"

        path = variant(tmp_path, rename)
        with pytest.raises(ProtocolValueError, match="exactly the frozen v1 set"):
            _load_from_path(path)


class TestCanonicalAccessPath:
    """Only the loader may reach the config file.

    This guard is vacuous today, because protocol.py is the only production module in
    the package. It is written now so that the modules arriving in later steps cannot
    quietly introduce a second reader that skips every validation above.

    It catches the two realistic bypasses: naming the file directly, and importing the
    module's private path constant. It cannot catch a path reassembled from pieces at
    runtime. The point is to make a bypass deliberate and visible in review rather than
    something reached for by accident, not to make it impossible.
    """

    FORBIDDEN = ("v1.json", "v2.json", "_PROTOCOL_PATH", "_load_from_path", "protocols/")

    def test_no_other_production_module_reaches_the_config_directly(self) -> None:
        package = _PROTOCOL_PATH.parents[1]
        offenders = []
        for source in sorted(package.rglob("*.py")):
            if source.name == "protocol.py" or "tests" in source.parts:
                continue
            text = source.read_text()
            hits = [token for token in self.FORBIDDEN if token in text]
            if hits:
                offenders.append(f"{source.relative_to(package)} references {hits}")
        assert not offenders, (
            f"these modules must load through benchmark.protocol.load_protocol: {offenders}"
        )


class TestOnlyOneApprovedConfiguration:
    """Being shaped like v1 is weaker than being v1.

    Every validator in the module can pass on a file that describes a different
    benchmark. So the production entrance takes no path at all, and the supported API
    cannot bless an unregistered alternative as protocol v1.
    """

    def test_production_loader_accepts_no_alternate_config(self) -> None:
        assert inspect.signature(load_protocol).parameters == {}

    def test_a_valid_looking_alternate_cannot_be_loaded_in_production(self, tmp_path: Path) -> None:
        """A different selection seed changes which recordings are admitted, and a
        different partition size changes how many. Both satisfy every validator and
        both still say version 1, which is precisely why validation alone cannot be
        the control here."""

        def alternate(d: dict[str, Any]) -> None:
            d["selection"]["seed"] = 999
            d["selection"]["partitions"]["test"] = 50
            d["aggregation"]["bootstrap"]["iterations"] = 50_000

        parsed = _load_from_path(variant(tmp_path, alternate))
        assert parsed.protocol_version == 3
        assert parsed.selection.seed == 999

        approved = load_protocol()
        assert approved.selection.seed != 999
        assert approved.selection.partitions["test"] != 50
        assert approved.aggregation.bootstrap.iterations != 50_000
        assert approved.source_sha256 != parsed.source_sha256


class TestAmendmentChain:
    """A superseded protocol stays in the tree and is named by digest, not by number."""

    def test_chain_names_the_immediately_superseded_artifacts(self) -> None:
        amends = load_protocol().amends
        assert amends.protocol_version == 2
        assert amends.protocol_path == "benchmark/protocols/v2.json"
        assert amends.document_path == "docs/benchmark-protocol-v2.md"

    def test_recorded_digests_match_the_archived_files(self) -> None:
        amends = load_protocol().amends
        assert digest_of(amends.protocol_path) == amends.protocol_sha256
        assert digest_of(amends.document_path) == amends.document_sha256

    def test_archived_v1_artifacts_are_byte_identical_to_their_frozen_state(self) -> None:
        for relative_path, expected in SUPERSEDED_ARTIFACTS.items():
            assert digest_of(relative_path) == expected, relative_path

    def test_a_modified_predecessor_is_detected(self, tmp_path: Path) -> None:
        """Recording a digest is worth little unless something recomputes it."""
        wrong = "0" * 64
        path = variant(tmp_path, lambda d: d["amends"].update({"protocol_sha256": wrong}))
        with pytest.raises(ProtocolValueError, match="has been modified"):
            _load_from_path(path)

    def test_a_pruned_predecessor_is_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The paths are frozen, so this cannot be provoked from config. It is still
        reachable the way it would actually happen: someone tidies the archived
        artifacts out of the tree and the chain has nothing left to verify against."""
        monkeypatch.setattr("benchmark.protocol._REPO_ROOT", tmp_path)
        with pytest.raises(ProtocolValueError, match="is missing"):
            _load_from_path(_PROTOCOL_PATH)

    def test_a_malformed_digest_is_rejected(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["amends"].update({"protocol_sha256": "abc123"}))
        with pytest.raises(ProtocolValueError, match="64 lowercase hex digits"):
            _load_from_path(path)

    def test_an_amendment_without_a_reason_is_rejected(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["amends"].update({"reason": "   "}))
        with pytest.raises(ProtocolValueError, match="indistinguishable from an edit"):
            _load_from_path(path)

    def test_the_predecessor_must_be_the_immediately_previous_version(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["amends"].update({"protocol_version": 0}))
        with pytest.raises(ProtocolValueError, match=r"amends\.protocol_version must be 2"):
            _load_from_path(path)

    def test_the_superseded_config_is_not_loadable_as_current(self) -> None:
        """Proof the version bump is real rather than cosmetic: the archived v1 file
        still parses, and the current validator refuses it."""
        with pytest.raises(ProtocolValueError, match="unsupported protocol_version 2"):
            _load_from_path(REPO_ROOT / "benchmark" / "protocols" / "v2.json")


class TestFrozenTransform:
    """Two LSD conventions were unspecified in v1 and each moved the number.

    Pre-evaluation characterisation on non-benchmark fixtures: window symmetry shifted
    LSD by about 0.0015 dB and FFT normalisation by about 0.088 dB. Those figures
    describe the fixtures examined, not a bound on benchmark impact. Both conventions
    are now pinned to a single value.
    """

    @staticmethod
    def lsd_field(data: dict[str, Any], section: str) -> dict[str, Any]:
        return data["metrics"]["log_spectral_distance"][section]

    def test_symmetric_window_is_rejected(self, tmp_path: Path) -> None:
        """np.hanning gives the symmetric window and scipy gives the periodic one.
        Both answer to the name 'hann' and they are different windows."""
        path = variant(
            tmp_path, lambda d: self.lsd_field(d, "window").update({"symmetry": "symmetric"})
        )
        with pytest.raises(ProtocolValueError, match=r"window\.symmetry must be 'periodic'"):
            _load_from_path(path)

    @pytest.mark.parametrize("norm", ["ortho", "forward"])
    def test_other_fft_normalisations_are_rejected(self, tmp_path: Path, norm: str) -> None:
        """Normalisation would cancel in a pure log ratio. It does not cancel here,
        because the offset is additive: scaling moves magnitudes relative to it."""
        path = variant(tmp_path, lambda d: self.lsd_field(d, "fft").update({"norm": norm}))
        with pytest.raises(ProtocolValueError, match=r"fft\.norm must be 'backward'"):
            _load_from_path(path)

    def test_unstated_pad_value_is_rejected(self, tmp_path: Path) -> None:
        """'constant' padding without a value inherits whatever the library defaults to."""
        path = variant(tmp_path, lambda d: self.lsd_field(d, "framing").update({"pad_value": 1.0}))
        with pytest.raises(ProtocolValueError, match=r"pad_value must be 0\.0"):
            _load_from_path(path)

    def test_asymmetric_padding_is_rejected(self, tmp_path: Path) -> None:
        """Stating padding per side is why 'width 512' cannot quietly mean 512 total."""
        path = variant(
            tmp_path, lambda d: self.lsd_field(d, "framing").update({"pad_right_samples": 0})
        )
        with pytest.raises(ProtocolValueError, match=r"pad_right_samples must be n_fft // 2"):
            _load_from_path(path)

    def test_partial_trailing_frame_is_rejected(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path,
            lambda d: self.lsd_field(d, "framing").update({"partial_trailing_frame": True}),
        )
        with pytest.raises(ProtocolValueError, match="differently shaped window"):
            _load_from_path(path)

    def test_clamp_semantics_are_rejected(self, tmp_path: Path) -> None:
        """max(|X|, offset) is a different metric from |X| + offset. v1 specified the
        additive form, so v2 carries it unchanged rather than improving it in passing."""
        path = variant(
            tmp_path,
            lambda d: self.lsd_field(d, "log_magnitude").update({"operation": "clamp_then_log10"}),
        )
        with pytest.raises(ProtocolValueError, match=r"log_magnitude\.operation must be"):
            _load_from_path(path)

    def test_reduction_order_is_pinned(self, tmp_path: Path) -> None:
        """RMS across bins then mean across frames is not the same as mean then RMS."""
        path = variant(
            tmp_path, lambda d: self.lsd_field(d, "reduction").update({"bins": "arithmetic_mean"})
        )
        with pytest.raises(ProtocolValueError, match=r"reduction\.bins must be 'rms'"):
            _load_from_path(path)

    def test_the_operation_names_the_amplitude_convention(self) -> None:
        """The dB factor is carried by the operation's name, not by a setting, so a
        config cannot select the power convention by editing a number."""
        log_magnitude = load_protocol().metrics.log_spectral_distance.log_magnitude
        assert log_magnitude.operation == "amplitude_db_additive_offset"
        assert not hasattr(log_magnitude, "multiplier")


class TestAmendmentChainIsNotRedirectable:
    """Digest verification alone would not establish which artifacts are amended.

    A config naming some other file together with that file's own hash satisfies
    "named file matches named digest" perfectly well. Freezing the paths in code also
    means a relative path escaping the repository is never reachable.
    """

    def test_pointing_the_protocol_path_at_another_real_file_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """The substitute exists and its recorded digest is correct, so only the frozen
        path check stands between this config and a redirected chain."""
        substitute = "benchmark/manifests/training_corpus.json"

        def redirect(d: dict[str, Any]) -> None:
            d["amends"]["protocol_path"] = substitute
            d["amends"]["protocol_sha256"] = digest_of(substitute)

        path = variant(tmp_path, redirect)
        with pytest.raises(ProtocolValueError, match="not redirectable by configuration"):
            _load_from_path(path)

    def test_pointing_the_document_path_at_another_real_file_is_rejected(
        self, tmp_path: Path
    ) -> None:
        substitute = "docs/benchmark-protocol.md"

        def redirect(d: dict[str, Any]) -> None:
            d["amends"]["document_path"] = substitute
            d["amends"]["document_sha256"] = digest_of(substitute)

        path = variant(tmp_path, redirect)
        with pytest.raises(ProtocolValueError, match="not redirectable by configuration"):
            _load_from_path(path)

    def test_a_path_escaping_the_repository_is_rejected(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path, lambda d: d["amends"].update({"document_path": "../../../etc/hosts"})
        )
        with pytest.raises(ProtocolValueError, match="not redirectable by configuration"):
            _load_from_path(path)


class TestAmplitudeConventionIsPinned:
    """Amplitude against power dB is a kind of measurement, so code owns it."""

    def test_power_convention_is_rejected(self, tmp_path: Path) -> None:
        """Naming the power convention is the readable way to get 10*log10, and the
        closed vocabulary refuses it. There is no factor to edit instead."""
        path = variant(
            tmp_path,
            lambda d: d["metrics"]["log_spectral_distance"]["log_magnitude"].update(
                {"operation": "power_db_additive_offset"}
            ),
        )
        with pytest.raises(ProtocolValueError, match=r"log_magnitude\.operation must be"):
            _load_from_path(path)

    def test_reintroducing_a_configurable_factor_is_rejected(self, tmp_path: Path) -> None:
        """Unknown-key rejection is what stops the factor coming back as a setting."""
        path = variant(
            tmp_path,
            lambda d: d["metrics"]["log_spectral_distance"]["log_magnitude"].update(
                {"multiplier": 10.0}
            ),
        )
        with pytest.raises(ProtocolSchemaError, match="unknown key"):
            _load_from_path(path)

    def test_magnitudes_are_not_duplicated_into_code(self, tmp_path: Path) -> None:
        """The counterpart to the rule above, asserted so the division stays honest.

        A different transform length is well formed and is not the approved experiment.
        The config digest is what distinguishes them, which is why `load_protocol` takes
        no path. Copying these numbers into the validator would create a second
        authority for a fact the digest already pins.
        """

        def resize(d: dict[str, Any]) -> None:
            lsd = d["metrics"]["log_spectral_distance"]
            lsd["n_fft"] = 2048
            lsd["win_length"] = 2048
            lsd["framing"]["pad_left_samples"] = 1024
            lsd["framing"]["pad_right_samples"] = 1024

        parsed = _load_from_path(variant(tmp_path, resize))
        assert parsed.metrics.log_spectral_distance.n_fft == 2048
        assert load_protocol().metrics.log_spectral_distance.n_fft != 2048
