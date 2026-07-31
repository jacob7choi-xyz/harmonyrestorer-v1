"""Tests for the production protocol loader.

Every test here loads through `benchmark.protocol.load_protocol`. Nothing in this
file reimplements loading, validation, or metric behaviour, because a test that
recreates the thing it checks proves only that the test agrees with itself.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.protocol import (
    _ABORT_REASONS,
    _EVALUATION_STAGE_ORDER,
    _INELIGIBLE_REASONS,
    _PRECONDITION_ORDER,
    _PROTOCOL_PATH,
    _REASON_CATEGORIES,
    _REASON_PRECEDENCE,
    _REASON_STAGE,
    _STAGE_CHECKS,
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
    "benchmark/protocols/v3.json": (
        "c22e36dc55b131a6d1708b90f2fecd12df9fbfeace70faf836c3ddbe92145f77"  # pragma: allowlist secret
    ),
    "docs/benchmark-protocol-v3.md": (
        "49fcdee5636519f0871159da03ab6e595c961ede861e2fe4c388f1796324ce35"  # pragma: allowlist secret
    ),
}


def reason_hint(reasons: tuple[str, ...]) -> str:
    """Failure message for the overclaim check, which is otherwise opaque."""
    return f"a reason code asserts more than a decoder error establishes: {reasons}"


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
        assert load_protocol().protocol_version == 4

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
        path = variant(tmp_path, lambda d: d.update({"protocol_version": 5}))
        with pytest.raises(ProtocolValueError, match="unsupported protocol_version 5"):
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
        assert parsed.protocol_version == 4
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
        assert amends.protocol_version == 3
        assert amends.protocol_path == "benchmark/protocols/v3.json"
        assert amends.document_path == "docs/benchmark-protocol-v3.md"

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
        with pytest.raises(ProtocolValueError, match=r"amends\.protocol_version must be 3"):
            _load_from_path(path)

    def test_the_superseded_config_is_not_loadable_as_current(self) -> None:
        """Proof the version bump is real rather than cosmetic: the archived v1 file
        still parses, and the current validator refuses it."""
        with pytest.raises(ProtocolValueError, match="unsupported protocol_version 3"):
            _load_from_path(REPO_ROOT / "benchmark" / "protocols" / "v3.json")


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


class TestCanonicalWaveformIsPinned:
    """v4's subject: the source-to-canonical transformation and the frames cut from it.

    v3 named the output shape, 16 kHz mono, and left every step producing it
    implementation-defined. Measured on a training FLAC: four `res_type` values give
    four digests, mean-of-channels and left-channel downmix give two, and downmixing
    before resampling differs from doing it after by 5.2e-10. Each free choice below
    changes the canonical bytes, so it changes every content hash derived from them.
    """

    @staticmethod
    def canon(data: dict[str, Any], section: str) -> dict[str, Any]:
        return data["canonicalisation"][section]

    @pytest.mark.parametrize(
        ("section", "key", "value", "expected"),
        [
            ("canonicalisation", "signed_zero", "preserve", "signed_zero must be"),
            ("canonicalisation", "non_finite", "allow", "non_finite must be"),
            (
                "canonicalisation",
                "reproducibility_model",
                "bit_identical_everywhere",
                "reproducibility_model must be",
            ),
        ],
    )
    def test_canonicalisation_kinds_cannot_be_weakened(
        self, tmp_path: Path, section: str, key: str, value: str, expected: str
    ) -> None:
        """Each was implementation-defined in v3 and each changes the canonical bytes."""
        path = variant(tmp_path, lambda d: d[section].update({key: value}))
        with pytest.raises(ProtocolValueError, match=expected):
            _load_from_path(path)

    def test_resample_arguments_cannot_be_left_to_a_default(self, tmp_path: Path) -> None:
        """soxr_hq is already librosa's default, which is exactly why it is named. Relying
        on a default is how the gap v4 closes got into the prose."""
        path = variant(tmp_path, lambda d: d["canonicalisation"]["resample"].update({"fix": False}))
        with pytest.raises(ProtocolValueError, match="fix=true, scale=false, axis=-1"):
            _load_from_path(path)

    def test_downmix_must_precede_resampling(self, tmp_path: Path) -> None:
        """Both orders are natural readings and they produce different bytes, measured at
        5.2e-10 on a training FLAC."""
        swapped = ["decode", "validate_source", "resample", "downmix", "validate_canonical"]
        path = variant(tmp_path, lambda d: d["canonicalisation"].update({"order": swapped}))
        with pytest.raises(ProtocolValueError, match=r"canonicalisation\.order must be"):
            _load_from_path(path)

    def test_the_rms_operation_is_a_closed_token(self, tmp_path: Path) -> None:
        """An epsilon or a different reduction moves which frames sit either side of the
        floor, so it decides population membership rather than only precision."""
        path = variant(
            tmp_path,
            lambda d: d["eligibility"].update(
                {"rms_operation": "float64_mean_square_sqrt_dbfs_epsilon_1e12"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="rms_operation must be"):
            _load_from_path(path)

    def test_eligibility_must_come_from_the_canonical_waveform(self, tmp_path: Path) -> None:
        path = variant(tmp_path, lambda d: d["eligibility"].update({"computed_from": "source"}))
        with pytest.raises(ProtocolValueError, match="computed_from must be"):
            _load_from_path(path)

    def test_rms_comparators_are_explicit_not_inferred(self, tmp_path: Path) -> None:
        """The document says 'above'. Code must not infer strictness from English."""
        path = variant(
            tmp_path, lambda d: d["eligibility"].update({"frame_rms_comparator": "at_least"})
        )
        with pytest.raises(ProtocolValueError, match="frame_rms_comparator must be"):
            _load_from_path(path)

    def test_a_partial_trailing_frame_must_be_counted(self, tmp_path: Path) -> None:
        """A remainder that is neither scored nor counted hides how the candidate set was
        formed, which is the denominator problem one level down."""
        path = variant(
            tmp_path, lambda d: d["framing"].update({"partial_trailing_frame": "dropped"})
        )
        with pytest.raises(ProtocolValueError, match="partial_trailing_frame must be"):
            _load_from_path(path)

    @pytest.mark.parametrize(
        ("section", "key", "value", "expected"),
        [
            ("downmix", "operation", "left_channel", r"downmix\.operation must be"),
            ("resample", "res_type", "kaiser_best", r"resample\.res_type must be"),
            ("decode", "dtype", "float32", r"decode\.dtype must be"),
            ("hash_bytes", "dtype", "float32", r"hash_bytes\.dtype must be"),
            ("hash_bytes", "byte_order", "big", r"hash_bytes\.byte_order must be"),
            ("hash_bytes", "container", "wav", r"hash_bytes\.container must be"),
        ],
    )
    def test_nested_transformation_choices_are_pinned(
        self, tmp_path: Path, section: str, key: str, value: str, expected: str
    ) -> None:
        """Every substitute here is a real option a faithful implementer could pick."""
        path = variant(tmp_path, lambda d: self.canon(d, section).update({key: value}))
        with pytest.raises(ProtocolValueError, match=expected):
            _load_from_path(path)

    def test_decode_must_always_be_two_dimensional(self, tmp_path: Path) -> None:
        """Without it a mono source returns rank 1 and a stereo source rank 2, so the
        downmix step would branch on source shape and the branches could diverge."""
        path = variant(tmp_path, lambda d: self.canon(d, "decode").update({"always_2d": False}))
        with pytest.raises(ProtocolValueError, match="always_2d must be true"):
            _load_from_path(path)

    @pytest.mark.parametrize(
        ("substitute", "why"),
        [
            ([], "empty forbids nothing"),
            (["banana"], "non-empty but names nothing that matters"),
            (
                [
                    "dither",
                    "clipping",
                    "peak_normalisation",
                    "loudness_normalisation",
                    "gain_adjustment",
                ],
                "one operation quietly dropped from an otherwise correct list",
            ),
            (
                [
                    "dither",
                    "integer_quantisation",
                    "clipping",
                    "peak_normalisation",
                    "loudness_normalisation",
                    "gain_adjustment",
                ],
                "same set, different order",
            ),
        ],
    )
    def test_the_forbidden_list_is_exact_not_merely_present(
        self, tmp_path: Path, substitute: list[str], why: str
    ) -> None:
        """A non-empty check is documentation. `["banana"]` satisfies it while permitting
        every operation that would change a fidelity measurement, and the dropped-entry case
        is the one that would actually happen."""
        path = variant(
            tmp_path, lambda d: d["canonicalisation"].update({"forbidden_operations": substitute})
        )
        with pytest.raises(ProtocolValueError, match="forbidden_operations must be"):
            _load_from_path(path)

    def test_the_output_length_rule_is_a_closed_token(self, tmp_path: Path) -> None:
        """It replaced a prose formula. A sentence a reader must reimplement is not
        executable authority, and two readers can implement it differently."""
        path = variant(
            tmp_path,
            lambda d: self.canon(d, "resample").update({"output_length_rule": "round_half_even"}),
        )
        with pytest.raises(ProtocolValueError, match="output_length_rule must be"):
            _load_from_path(path)

    def test_the_forbidden_list_is_load_bearing_not_declarative(self) -> None:
        """Resampling overshoots full scale, measured at 1.194538 on a 0.999 square wave.
        The only ways back inside [-1, 1] are these three, which is why the canonical
        waveform carries no amplitude bound."""
        forbidden = load_protocol().canonicalisation.forbidden_operations
        for operation in ("clipping", "peak_normalisation", "gain_adjustment"):
            assert operation in forbidden

    def test_frames_must_start_at_the_beginning(self, tmp_path: Path) -> None:
        """A nonzero start is an alignment choice, and an undeclared one shifts which
        audio each frame index refers to."""
        path = variant(tmp_path, lambda d: d["framing"].update({"first_frame_start_sample": 1}))
        with pytest.raises(ProtocolValueError, match="first_frame_start_sample must be 0"):
            _load_from_path(path)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("frame_samples", 0),
            ("frame_samples", -32000),
            ("hop_samples", 0),
        ],
    )
    def test_frame_geometry_must_be_positive(self, tmp_path: Path, field: str, value: int) -> None:
        """Nothing else rejects these. Both survive the whole-seconds check, because
        `0 % 16000` and `-32000 % 16000` are both 0, and a zero hop would emit frames
        forever at the same offset."""
        path = variant(tmp_path, lambda d: d["framing"].update({field: value}))
        with pytest.raises(ProtocolValueError, match="must be positive"):
            _load_from_path(path)

    def test_frame_samples_must_match_the_declared_duration(self, tmp_path: Path) -> None:
        """The config declares two seconds and the sample count has to mean that. Prose
        cannot enforce it, because code cannot read prose."""
        path = variant(tmp_path, lambda d: d["framing"].update({"frame_samples": 32001}))
        with pytest.raises(ProtocolValueError, match=r"is not 2 s at 16000 Hz"):
            _load_from_path(path)

    def test_a_ten_second_frame_is_rejected_even_though_it_is_whole_seconds(
        self, tmp_path: Path
    ) -> None:
        """The check this replaced accepted it: 160000 % 16000 == 0. Frame duration
        determines the candidate population, eligibility, hashes, and every denominator."""
        path = variant(
            tmp_path,
            lambda d: d["framing"].update({"frame_samples": 160000, "hop_samples": 160000}),
        )
        with pytest.raises(ProtocolValueError, match=r"is not 2 s at 16000 Hz"):
            _load_from_path(path)

    def test_changing_the_declared_duration_alone_is_rejected(self, tmp_path: Path) -> None:
        """Both fields must move together, so neither can be edited to disagree with the
        other. This is what keeps the magnitude in the protocol without letting it drift."""
        path = variant(tmp_path, lambda d: d["framing"].update({"frame_duration_s": 10}))
        with pytest.raises(ProtocolValueError, match=r"is not 10 s at 16000 Hz"):
            _load_from_path(path)

    @pytest.mark.parametrize("hop", [16000, 8000, 31999])
    def test_evaluation_frames_may_not_overlap(self, tmp_path: Path, hop: int) -> None:
        """An overlapping hop counts the same audio into several frames of one denominator.
        Every value here passed the whole-seconds check that preceded this one."""
        path = variant(tmp_path, lambda d: d["framing"].update({"hop_samples": hop}))
        with pytest.raises(ProtocolValueError, match="must equal frame_samples"):
            _load_from_path(path)

    def test_the_two_sample_rate_fields_may_not_disagree(self, tmp_path: Path) -> None:
        """canonicalisation.resample.target_sample_rate and conditioning.sample_rate name
        one fact. Before this check, 22050 against 16000 loaded without complaint.

        The source-rate floor moves with the target so that this test isolates the
        disagreement it is named for; leaving the floor at 16000 would trip that check first
        and the test would pass for a reason it does not describe.
        """

        def retarget(d: dict[str, Any]) -> None:
            d["canonicalisation"]["resample"]["target_sample_rate"] = 22050
            d["canonicalisation"]["source_validation"]["min_sample_rate"] = 22050

        path = variant(tmp_path, retarget)
        with pytest.raises(ProtocolValueError, match="name the same fact and disagree"):
            _load_from_path(path)


class TestSourceDomainIsFrozenBeforeAcquisition:
    """The source contract is chosen now, before the candidate universe exists.

    Pinning only the subtype the 145 training files happen to carry would mean meeting a
    PCM_16 candidate later and widening the contract after inspecting the population, which
    is the governance shape this protocol exists to prevent. The admitted set instead rests
    on a proof: integer PCM converts exactly into float64 and its decoded amplitude is
    bounded by construction. PCM_8 satisfies that proof and is still excluded, for the
    separate reason that its quantisation noise would become part of the clean reference.
    """

    @pytest.mark.parametrize(
        ("substitute", "why"),
        [
            ([["FLAC", "PCM_24"]], "narrowed to the observed corpus"),
            (
                [["FLAC", "PCM_16"], ["FLAC", "PCM_24"], ["FLAC", "PCM_32"], ["FLAC", "PCM_S8"]],
                "8-bit admitted despite being exactly representable",
            ),
            (
                [["FLAC", "PCM_16"], ["FLAC", "PCM_24"], ["FLAC", "FLOAT"]],
                "float subtype, where neither the bound nor exactness holds",
            ),
            (
                [["WAV", "PCM_24"], ["FLAC", "PCM_16"], ["FLAC", "PCM_24"], ["FLAC", "PCM_32"]],
                "container outside the analysed acquisition path",
            ),
        ],
    )
    def test_the_admitted_source_formats_are_exact(
        self, tmp_path: Path, substitute: list[list[str]], why: str
    ) -> None:
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update({"source_formats": substitute}),
        )
        with pytest.raises(ProtocolValueError, match="source_formats must be"):
            _load_from_path(path)

    def test_formats_are_pairs_so_they_cannot_be_recombined(self) -> None:
        """Two independent sets would bless FLAC+FLOAT from a permitted container and a
        permitted subtype that were never approved together."""
        formats = load_protocol().canonicalisation.decode.source_formats
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in formats)

    def test_metadata_comes_from_the_decoder_not_the_filename(self, tmp_path: Path) -> None:
        """A file named .wav containing FLAC reports format=FLAC subtype=PCM_24, so
        extension-based dispatch decodes it under the wrong contract."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"metadata_authority": "file_extension"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="metadata_authority must be"):
            _load_from_path(path)

    def test_the_hashed_bytes_must_be_the_decoded_bytes(self, tmp_path: Path) -> None:
        """Re-reading the path between hashing and decoding leaves a window in which the
        recorded digest identifies bytes other than the ones decoded."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"byte_identity": "hash_then_reread_path"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="byte_identity must be"):
            _load_from_path(path)

    @pytest.mark.parametrize("channels_max", [1, 6])
    def test_channel_range_is_pinned_to_one_or_two(self, tmp_path: Path, channels_max: int) -> None:
        """Above two, arithmetic averaging is not a defensible layout policy; below two,
        every training recording would be rejected."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["downmix"].update({"channels_max": channels_max}),
        )
        with pytest.raises(ProtocolValueError, match="must accept 1 to 2 channels"):
            _load_from_path(path)


class TestResourceBoundsGuardTheRightAllocation:
    """The decoded array, not the file, is what a source can make the process allocate.

    The largest training source is 166.5 MB and decodes to 807.3 MB of float64, an expansion
    of 4.8x. Bounding only the buffer would admit a file that then allocates several times
    its own size. Both bounds are magnitudes owned by the config; what the loader enforces is
    the relationship between them.
    """

    @pytest.mark.parametrize(
        "field", ["max_source_bytes", "max_decoded_bytes", "max_source_duration_s"]
    )
    def test_bounds_must_be_positive(self, tmp_path: Path, field: str) -> None:
        path = variant(tmp_path, lambda d: d["canonicalisation"]["decode"].update({field: 0}))
        with pytest.raises(ProtocolValueError, match=f"decode.{field} must be positive"):
            _load_from_path(path)

    def test_the_read_bound_must_be_enforced_before_allocating(self, tmp_path: Path) -> None:
        """A bound checked after read_bytes() has returned does not bound the allocation it
        names. The buffer is already resident by then."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"read_bound_enforcement": "after_read"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="read_bound_enforcement must be"):
            _load_from_path(path)

    def test_the_bounds_may_not_admit_more_than_the_declared_budget(self, tmp_path: Path) -> None:
        """Bounding the pieces individually does not bound the process. Source plus decoded
        plus downmix plus canonical are alive together, so the config's own limits have to
        add up to something the declared budget can hold."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update({"max_decoded_bytes": 2147483648}),
        )
        with pytest.raises(ProtocolValueError, match="does not bound the process"):
            _load_from_path(path)

    def test_the_superseded_two_decoded_bound_no_longer_certifies(self, tmp_path: Path) -> None:
        """The exact configuration the previous draft accepted: 512 MiB and 1.5 GiB against a
        4 GiB budget, which `source + 2 * decoded` passes at 3.50 GiB and `source + 3 *
        decoded` refuses at 5.00 GiB."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"max_projected_live_bytes": 4294967296}
            ),
        )
        with pytest.raises(ProtocolValueError, match="does not bound the process"):
            _load_from_path(path)

    @pytest.mark.parametrize("rate", [0, 8000, 15999, 22050])
    def test_the_source_rate_floor_must_equal_the_canonical_rate(
        self, tmp_path: Path, rate: int
    ) -> None:
        """Positivity alone froze nothing: 8000, 15999, and 22050 all loaded while the
        resampler still targeted 16000. Two invariants ride on the equality. A source below
        the canonical rate cannot supply the bandwidth being measured, and a floor below the
        target lets the canonical waveform exceed the downmix, which is the step the
        live-memory bound depends on. A floor above the target is refused too, because it
        would exclude legitimate 16 kHz material for no stated reason."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["source_validation"].update({"min_sample_rate": rate}),
        )
        with pytest.raises(ProtocolValueError, match="must equal the canonical rate"):
            _load_from_path(path)

    def test_the_accepted_floor_is_the_canonical_rate(self) -> None:
        canon = load_protocol().canonicalisation
        assert canon.source_validation.min_sample_rate == canon.resample.target_sample_rate

    def test_an_irregular_file_is_refused_rather_than_reasoned_about(self, tmp_path: Path) -> None:
        """fstat reports nothing meaningful for a FIFO or a device, so the early size check
        would run on a number that means nothing."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update({"require_regular_file": False}),
        )
        with pytest.raises(ProtocolValueError, match="require_regular_file must be true"):
            _load_from_path(path)

    def test_metadata_is_validated_before_it_sizes_an_allocation(self, tmp_path: Path) -> None:
        """Frames and channels come from the container, so they are attacker-adjacent input
        in the general case. The hazard is not arithmetic, since Python integers do not
        overflow; it is believing absurd metadata far enough to allocate on it."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"metadata_validation": "trusted_from_container"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="metadata_validation must be"):
            _load_from_path(path)

    @pytest.mark.parametrize("concurrency", [2, 4])
    def test_concurrency_above_one_invalidates_the_budget(
        self, tmp_path: Path, concurrency: int
    ) -> None:
        """The budget is per process. Two concurrent canonicalisations against 6 GiB need
        12 GiB, so a protocol that declares a budget without declaring the concurrency has
        declared nothing about the machine."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"canonicalisation_concurrency": concurrency}
            ),
        )
        with pytest.raises(ProtocolValueError, match="multiplies the requirement"):
            _load_from_path(path)

    def test_the_live_projection_rule_is_a_closed_token(self, tmp_path: Path) -> None:
        """It names which allocations are counted and when. Left as prose, an implementation
        could count the decoded array alone and still claim conformance."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update({"live_bytes_rule": "decoded_only"}),
        )
        with pytest.raises(ProtocolValueError, match="live_bytes_rule must be"):
            _load_from_path(path)

    def test_the_budget_holds_for_every_admitted_channel_count(self) -> None:
        """The bound has to cover mono, not just the stereo corpus we happen to have. At one
        channel the downmix equals the decoded array rather than halving it, so the earlier
        `source + 2 * decoded` proof understated the worst case by a whole decoded array and
        certified 5.00 GiB of allocation against a 4 GiB budget."""
        decode = load_protocol().canonicalisation.decode
        canon = load_protocol().canonicalisation
        for channels in range(canon.downmix.channels_min, canon.downmix.channels_max + 1):
            frames = decode.max_decoded_bytes // (channels * 8)
            decoded = frames * channels * 8
            downmix = frames * 8
            canonical = downmix  # min_sample_rate equals the target, so never longer
            worst = decode.max_source_bytes + decoded + downmix + canonical
            assert worst <= decode.max_projected_live_bytes, channels

    def test_the_declared_budget_covers_the_conservative_bound(self) -> None:
        decode = load_protocol().canonicalisation.decode
        assert decode.max_source_bytes + 3 * decode.max_decoded_bytes <= (
            decode.max_projected_live_bytes
        )

    def test_the_identity_stream_is_bounded_too(self, tmp_path: Path) -> None:
        """Streaming an excluded artifact through SHA-256 bounds memory and bounds nothing
        else. At 200 MB/s a 100 GB file costs about 8 minutes of I/O to populate one audit
        field for a candidate already excluded. A validly excluded source must not be able
        to buy unbounded work from the harness as its parting act."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update({"max_identity_stream_bytes": 0}),
        )
        with pytest.raises(ProtocolValueError, match="max_identity_stream_bytes must be positive"):
            _load_from_path(path)

    def test_the_identity_stream_bound_must_exceed_the_size_that_triggers_it(
        self, tmp_path: Path
    ) -> None:
        """It only ever applies to artifacts already rejected for exceeding the file bound,
        so at or below that size it could never apply to anything."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"max_identity_stream_bytes": 268435456}
            ),
        )
        with pytest.raises(ProtocolValueError, match="max_identity_stream_bytes must exceed"):
            _load_from_path(path)

    def test_the_decoded_bound_must_exceed_the_file_bound(self, tmp_path: Path) -> None:
        """Equal bounds would be the mistake in its most plausible form: a reviewer setting
        one number for 'the size limit' without noticing decoding expands."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update({"max_decoded_bytes": 536870912}),
        )
        with pytest.raises(ProtocolValueError, match="max_decoded_bytes must exceed"):
            _load_from_path(path)

    def test_the_bounds_accommodate_the_observed_corpus(self) -> None:
        """A bound below the material it must admit would surface as a run failure rather
        than a review comment, so the headroom is asserted rather than described."""
        decode = load_protocol().canonicalisation.decode
        assert decode.max_source_bytes > 166_500_000
        assert decode.max_decoded_bytes > 807_300_000
        assert decode.max_source_duration_s > 17.5 * 60


class TestAmplitudeAsymmetryIsDeliberate:
    """The source is bounded and the canonical waveform is not, and that is a decision.

    Band-limited interpolation overshoots: a 0.999 square wave resampled to 16 kHz reaches
    1.194538 and a full-scale 5 kHz sine reaches 1.020793. Bounding the canonical waveform
    would reject loud legitimate material or force a clip, a peak normalisation, or a gain,
    which is why those three appear in `forbidden_operations`.
    """

    @pytest.mark.parametrize(
        "requirement", ["require_finite", "require_non_empty", "require_positive_sample_rate"]
    )
    def test_basic_source_requirements_cannot_be_switched_off(
        self, tmp_path: Path, requirement: str
    ) -> None:
        """Each is assumed by everything downstream, which is exactly why none of them is
        safe to leave to an implementation's judgement."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["source_validation"].update({requirement: False}),
        )
        with pytest.raises(ProtocolValueError, match="must require finite, non-empty samples at"):
            _load_from_path(path)

    def test_the_canonical_waveform_must_be_one_dimensional(self, tmp_path: Path) -> None:
        """Everything downstream indexes it as a single channel of samples."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["canonical_validation"].update({"ndim": 2}),
        )
        with pytest.raises(ProtocolValueError, match=r"canonical_validation\.ndim must be 1"):
            _load_from_path(path)

    def test_the_source_bound_is_pinned(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["source_validation"].update({"amplitude_max": 2.0}),
        )
        with pytest.raises(ProtocolValueError, match=r"bound source amplitude to \[-1.0, 1.0\]"):
            _load_from_path(path)

    def test_the_source_bound_stays_inclusive(self, tmp_path: Path) -> None:
        """PCM reaches exactly -1.0 at the negative rail, so an exclusive bound would
        reject legitimate full-scale material."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["source_validation"].update(
                {"amplitude_bounds_inclusive": False}
            ),
        )
        with pytest.raises(ProtocolValueError, match="amplitude_bounds_inclusive must be true"):
            _load_from_path(path)

    def test_an_amplitude_bound_cannot_be_added_to_the_canonical_waveform(
        self, tmp_path: Path
    ) -> None:
        """The tempting edit, and the one that would silently start rejecting loud
        recordings for a reason nobody would connect to the resampler."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["canonical_validation"].update(
                {"amplitude_bound": "unit_scale"}
            ),
        )
        with pytest.raises(ProtocolValueError, match=r"canonical_validation\.amplitude_bound"):
            _load_from_path(path)

    def test_the_canonical_waveform_is_still_required_to_be_finite(self, tmp_path: Path) -> None:
        """No amplitude bound is not the same as no validation."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["canonical_validation"].update(
                {"require_finite": False}
            ),
        )
        with pytest.raises(ProtocolValueError, match="must require finite, non-empty"):
            _load_from_path(path)


class TestSourceOutcomeAttribution:
    """A decoder rejection is evidence about our local artifact, not about the collection.

    The collection publishes no checksum and the acquisition script records none, so the
    local digest identifies the exact bytes inspected and decoded and establishes nothing
    about the publisher's original bytes. Every control here exists so that a defect in our
    own retrieval cannot be published as a finding about someone else's file.
    """

    def test_a_harness_reason_cannot_be_spelled_as_a_source_verdict(self, tmp_path: Path) -> None:
        """The load-bearing one. Two vocabularies stop being a control the moment a code
        appears in both."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {
                    "ineligible_reasons": [
                        *d["source_outcomes"]["ineligible_reasons"],
                        "SOURCE_HASH_MISMATCH",
                    ]
                }
            ),
        )
        with pytest.raises(ProtocolValueError, match="ineligible_reasons must be"):
            _load_from_path(path)

    def test_the_two_vocabularies_are_disjoint_as_loaded(self) -> None:
        outcomes = load_protocol().source_outcomes
        assert not set(outcomes.ineligible_reasons) & set(outcomes.abort_reasons)
        assert "SOURCE_HASH_MISMATCH" in outcomes.abort_reasons

    def test_no_reason_code_asserts_more_than_a_decoder_error_proves(self) -> None:
        """An earlier draft used MALFORMED_OR_TRUNCATED_AUDIO. A rejection can also come
        from an unusual but valid encoding feature or a decoder limitation, and it can come
        from our own retrieval having damaged the bytes."""
        reasons = load_protocol().source_outcomes.ineligible_reasons
        for overclaim in ("MALFORMED", "TRUNCATED", "CORRUPT"):
            assert not any(overclaim in reason for reason in reasons), reason_hint(reasons)
        assert "SOURCE_DECODE_REJECTED" in reasons
        assert "SOURCE_INSPECTION_REJECTED" in reasons

    @pytest.mark.parametrize(
        ("field", "substitute"),
        [
            ("attribution_rule", "decoder_rejection_is_always_source_owned"),
            ("abort_reasons", ["GOLDEN_FIXTURE_FAILURE"]),
            ("reduction_chain", ["selected_sources", "frames_eligible"]),
        ],
    )
    def test_the_outcome_vocabularies_are_exact(
        self, tmp_path: Path, field: str, substitute: object
    ) -> None:
        """The attribution substitute is the belief this section exists to refuse. The
        shortened chain is denominator laundering: two endpoints and no visible narrowing."""
        path = variant(tmp_path, lambda d: d["source_outcomes"].update({field: substitute}))
        with pytest.raises(ProtocolValueError, match=rf"source_outcomes\.{field} must be"):
            _load_from_path(path)

    def test_the_attribution_token_claims_no_more_than_the_note_beside_it(self) -> None:
        """An earlier draft read `source_owned`, while its own note said the finding is about
        the locally acquired artifact. A closed field that overstates its note is worse than
        prose, because tools quote the field and not the note."""
        rule = load_protocol().source_outcomes.attribution_rule
        assert "source_owned" not in rule
        assert "local_artifact" in rule

    def test_every_ineligible_reason_has_exactly_one_category(self) -> None:
        """Asserted over the constants, where it can still fail. Against a loaded config the
        loader already requires equality with these tuples, so a partition check there would
        be another line no mutation can reach."""
        categorised = [code for _, codes in _REASON_CATEGORIES for code in codes]
        assert sorted(categorised) == sorted(_INELIGIBLE_REASONS)
        assert len(categorised) == len(set(categorised))

    @pytest.mark.parametrize(
        ("vocabulary", "name"),
        [
            (_INELIGIBLE_REASONS, "ineligible"),
            (_ABORT_REASONS, "abort"),
            (_REASON_CATEGORIES, "categories"),
        ],
    )
    def test_no_vocabulary_contains_a_duplicate(
        self, vocabulary: tuple[object, ...], name: str
    ) -> None:
        """A repeated code would double-count a narrowing in the published reduction chain."""
        assert len(vocabulary) == len(set(vocabulary)), name

    def test_resource_exclusions_are_not_reported_as_defective_audio(self) -> None:
        """A recording refused for exceeding a predeclared size or duration limit has nothing
        wrong with it. Publishing that beside a decoder rejection would invite the reader to
        conclude the collection is defective."""
        categories = dict(_REASON_CATEGORIES)
        assert "SOURCE_EXCEEDS_MAX_BYTES" in categories["frozen_domain_exclusion"]
        assert "SOURCE_DECODE_REJECTED" in categories["artifact_or_decoder_outcome"]
        assert not set(categories["frozen_domain_exclusion"]) & set(
            categories["artifact_or_decoder_outcome"]
        )

    @pytest.mark.parametrize(
        ("substitute", "why"),
        [
            (
                {
                    "all_source_failures": [
                        "SOURCE_EXCEEDS_MAX_BYTES",
                        "SOURCE_EXCEEDS_MAX_DURATION",
                        "SOURCE_BELOW_MIN_SAMPLE_RATE",
                        "UNSUPPORTED_CONTAINER",
                        "UNSUPPORTED_SUBTYPE",
                        "UNSUPPORTED_CHANNEL_COUNT",
                        "SOURCE_INSPECTION_REJECTED",
                        "SOURCE_DECODE_REJECTED",
                        "SOURCE_METADATA_INCONSISTENT",
                        "EMPTY_SOURCE",
                        "NONFINITE_SOURCE",
                        "INVALID_SOURCE_AMPLITUDE",
                    ]
                },
                "collapsed into one bucket, which is the reporting failure itself",
            ),
            (
                {
                    "frozen_domain_exclusion": ["SOURCE_EXCEEDS_MAX_BYTES"],
                    "unsupported_representation": [
                        "UNSUPPORTED_CONTAINER",
                        "UNSUPPORTED_SUBTYPE",
                        "UNSUPPORTED_CHANNEL_COUNT",
                    ],
                    "artifact_or_decoder_outcome": [
                        "SOURCE_INSPECTION_REJECTED",
                        "SOURCE_DECODE_REJECTED",
                        "SOURCE_METADATA_INCONSISTENT",
                        "SOURCE_EXCEEDS_MAX_DURATION",
                    ],
                    "decoded_waveform_violation": [
                        "EMPTY_SOURCE",
                        "NONFINITE_SOURCE",
                        "INVALID_SOURCE_AMPLITUDE",
                        "SOURCE_BELOW_MIN_SAMPLE_RATE",
                    ],
                },
                "a size limit filed as a decoder outcome, which reads as a defective file",
            ),
        ],
    )
    def test_the_reason_taxonomy_is_exact(
        self, tmp_path: Path, substitute: dict[str, list[str]], why: str
    ) -> None:
        """The categories are what stop 'twelve sources rejected' from reading as 'twelve
        bad files'. Both substitutes keep every code present and still misreport."""
        path = variant(
            tmp_path, lambda d: d["source_outcomes"].update({"reason_categories": substitute})
        )
        with pytest.raises(ProtocolValueError, match="reason_categories must be"):
            _load_from_path(path)

    def test_raw_decoder_diagnostics_are_not_published(self, tmp_path: Path) -> None:
        """Native diagnostics carry filesystem paths and local filenames. Nothing about a
        decode failure is worth leaking a local path into a public artifact."""
        path = variant(
            tmp_path, lambda d: d["source_outcomes"].update({"diagnostic_handling": "raw"})
        )
        with pytest.raises(ProtocolValueError, match="diagnostic_handling must be"):
            _load_from_path(path)

    def test_the_required_core_of_a_failure_record_cannot_lose_a_field(
        self, tmp_path: Path
    ) -> None:
        """Dropping the stage would leave a rejection that cannot say how far it got."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {
                    "record_required_core": [
                        f
                        for f in d["source_outcomes"]["record_required_core"]
                        if f != "terminating_evaluation_stage"
                    ]
                }
            ),
        )
        with pytest.raises(ProtocolValueError, match="record_required_core must be"):
            _load_from_path(path)

    def test_the_digest_is_conditional_not_required(self) -> None:
        """An oversized artifact is rejected from fstat and may sit above the identity
        stream bound, in which case no whole-artifact digest exists. Requiring one on every
        record forced either a mislabelled prefix hash or unbounded reading."""
        outcomes = load_protocol().source_outcomes
        assert "local_sha256" not in outcomes.record_required_core
        assert outcomes.record_conditional["local_sha256"] == "when_identity_complete_sha256"
        assert "identity_status" in outcomes.record_required_core

    def test_each_conditional_field_states_when_it_applies(self, tmp_path: Path) -> None:
        """A field list without conditions forces placeholders that pretend evidence exists,
        and a reader cannot tell an absent value from an inapplicable one."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"]["record_conditional"].update(
                {"decoder_diagnostic": "always"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="record_conditional must be"):
            _load_from_path(path)

    def test_an_absent_reason_is_distinguishable_from_a_passed_check(self, tmp_path: Path) -> None:
        """Without the unevaluated stages, missing UNSUPPORTED_SUBTYPE reads as evidence the
        subtype was supported, when inspection may never have happened."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update({"unevaluated_stages_recorded": False}),
        )
        with pytest.raises(ProtocolValueError, match="unevaluated_stages_recorded must be true"):
            _load_from_path(path)

    def test_supplemental_reasons_are_only_those_actually_established(self, tmp_path: Path) -> None:
        """An earlier draft promised every other applicable reason. For a source rejected
        from fstat and never inspected, whether its subtype or duration would also have
        failed is unknowable without doing the work the exclusion exists to avoid."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update({"supplemental_reasons": "all_applicable"}),
        )
        with pytest.raises(ProtocolValueError, match="supplemental_reasons must be"):
            _load_from_path(path)

    def test_every_safe_check_in_the_reached_stage_runs_before_choosing(
        self, tmp_path: Path
    ) -> None:
        """Otherwise an implementation that noticed an unsupported subtype would stop and
        never record an already-knowable duration violation, and two conforming
        implementations would publish different supplemental sets for one artifact."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update({"reason_selection": "first_violation_wins"}),
        )
        with pytest.raises(ProtocolValueError, match="reason_selection must be"):
            _load_from_path(path)

    def test_the_identity_basis_names_both_paths_that_establish_it(self, tmp_path: Path) -> None:
        """An artifact excluded before decoding is never decoded, so a basis naming only the
        decoded buffer described one of the two paths while claiming to state the basis for
        both. Found by cross-checking the section against itself, not by a failing test."""
        basis = load_protocol().source_outcomes.local_identity_basis
        assert "decoded_buffer" in basis and "streamed_artifact" in basis
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {"local_identity_basis": "local_sha256_of_the_decoded_buffer"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="local_identity_basis must be"):
            _load_from_path(path)

    def test_identity_method_is_a_closed_vocabulary(self, tmp_path: Path) -> None:
        """A prefix hash is not one of the three ways identity can be established, and a
        method vocabulary that admits one would let a record name it as evidence."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {
                    "identity_method_values": [
                        "bounded_single_buffer_sha256",
                        "prefix_sha256_of_bounded_read",
                        "not_computed_above_bound",
                    ]
                }
            ),
        )
        with pytest.raises(ProtocolValueError, match="identity_method_values must be"):
            _load_from_path(path)

    def test_identity_status_is_a_closed_vocabulary(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {"identity_status_values": ["complete_sha256", "prefix_sha256"]}
            ),
        )
        with pytest.raises(ProtocolValueError, match="identity_status_values must be"):
            _load_from_path(path)

    def test_upstream_identity_cannot_be_claimed(self, tmp_path: Path) -> None:
        """Flipping this would let publication assert equivalence to the publisher's bytes
        that no artifact in this repository can establish."""
        path = variant(
            tmp_path, lambda d: d["source_outcomes"].update({"upstream_checksum_available": True})
        )
        with pytest.raises(ProtocolValueError, match="upstream_checksum_available must be false"):
            _load_from_path(path)

    def test_the_stage_vocabulary_is_frozen(self, tmp_path: Path) -> None:
        """`stage`, `unevaluated_stages`, and the within-stage precedence were all
        load-bearing while nothing defined a stage. Two conforming implementations could
        disagree on whether inspection and metadata validation are one stage or two and emit
        different records for the same artifact."""
        merged = [
            "local_file_check",
            "inspection",
            "decode",
            "decoded_metadata_validation",
            "waveform_validation",
        ]
        path = variant(
            tmp_path, lambda d: d["source_outcomes"].update({"evaluation_stage_order": merged})
        )
        with pytest.raises(ProtocolValueError, match="evaluation_stage_order must be"):
            _load_from_path(path)

    def test_identity_acquisition_is_not_an_eligibility_stage(self) -> None:
        """The contradiction this separation removes. An oversized artifact terminates at
        local_file_check, so a stage order containing identity would derive an unevaluated
        set naming it, while the same record carried a streaming digest that only that stage
        could have produced. A stage cannot be both skipped and executed."""
        so = load_protocol().source_outcomes
        assert not any("identity" in stage for stage in so.evaluation_stage_order)
        terminating = so.reason_stage["SOURCE_EXCEEDS_MAX_BYTES"]
        index = so.evaluation_stage_order.index(terminating)
        assert not any("identity" in stage for stage in so.evaluation_stage_order[index + 1 :])
        assert so.identity_method_values[1] == "bounded_streaming_sha256"

    def test_no_identity_method_claims_work_it_did_not_do(self) -> None:
        """A source rejected at inspection or on its subtype is never decoded, so a method
        named for hashing, inspecting, and decoding one buffer would have the record assert
        execution that did not happen. Identity names how the digest was established; what
        ran is carried by the terminating stage and the unevaluated suffix."""
        methods = load_protocol().source_outcomes.identity_method_values
        for method in methods:
            assert "decode" not in method
            assert "inspect" not in method

    def test_identity_status_method_and_digest_cannot_disagree(self) -> None:
        """Four views of one fact: status, method, digest presence, and source size. Each
        pairing is individually plausible as a typo and each would publish an identity claim
        that never happened."""
        constraints = load_protocol().source_outcomes.record_constraints
        for constraint in (
            "identity_status_consistent_with_identity_method",
            "local_sha256_present_iff_identity_complete",
            "identity_method_consistent_with_source_byte_bounds",
        ):
            assert constraint in constraints

    def test_the_identity_method_is_recorded_not_inferred(self, tmp_path: Path) -> None:
        """Two of the three methods both report complete_sha256 and reach it by materially
        different evidence paths, so a status alone cannot distinguish them."""
        assert "identity_method" in load_protocol().source_outcomes.record_required_core
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {"identity_derivation": "implementation_defined"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="identity_derivation must be"):
            _load_from_path(path)

    def test_each_stage_declares_what_it_checks(self, tmp_path: Path) -> None:
        """Otherwise 'evaluate every safe check in the stage' refers to a set nothing
        defines, and one implementation stops at an unsupported subtype while another also
        establishes an already-knowable duration violation."""
        checks = dict(_STAGE_CHECKS)
        assert sorted(checks) == sorted(_EVALUATION_STAGE_ORDER)
        assert len(checks["metadata_validation"]) == 6
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"]["stage_checks"].update(
                {"metadata_validation": ["container", "subtype"]}
            ),
        )
        with pytest.raises(ProtocolValueError, match="stage_checks must be"):
            _load_from_path(path)

    def test_supplemental_reasons_must_come_from_the_terminating_stage(self) -> None:
        """Without this, a record could pair UNSUPPORTED_SUBTYPE with NONFINITE_SOURCE, a
        reason from a stage that was never reached. An earlier stage cannot hold an
        unrecorded violation either, because it would already have terminated processing."""
        constraints = load_protocol().source_outcomes.record_constraints
        assert "supplemental_reason_codes_share_primary_reason_stage" in constraints

    def test_an_irregular_local_file_has_an_outcome_to_record(self) -> None:
        """require_regular_file was enforced with no code able to record it, the same shape
        as the missing decoded-size reason two rounds earlier. It aborts rather than
        excluding the candidate: a FIFO where a downloaded file should be says something
        about this environment and nothing about the recording."""
        outcomes = load_protocol().source_outcomes
        assert "LOCAL_ARTIFACT_NOT_REGULAR_FILE" in outcomes.abort_reasons
        assert "LOCAL_ARTIFACT_NOT_REGULAR_FILE" not in outcomes.ineligible_reasons

    def test_every_reason_names_exactly_one_stage(self) -> None:
        """The stage on a record is derived from the reason, not chosen by whoever writes
        it. A reason with no stage would leave `unevaluated_stages` underivable."""
        mapping = dict(_REASON_STAGE)
        assert sorted(mapping) == sorted(_INELIGIBLE_REASONS)
        assert set(mapping.values()) <= set(_EVALUATION_STAGE_ORDER)

    def test_the_precedence_never_contradicts_the_stage_order(self) -> None:
        """A reason cannot be selected before the stage that establishes it has run, so the
        precedence must be non-decreasing in stage position."""
        mapping = dict(_REASON_STAGE)
        positions = [_EVALUATION_STAGE_ORDER.index(mapping[code]) for code in _REASON_PRECEDENCE]
        assert positions == sorted(positions), positions

    def test_every_evaluation_stage_carries_at_least_one_reason(self) -> None:
        """Now that identity acquisition has moved off this axis, a stage with no reason
        mapped to it would be a stage that can never terminate processing, which means it
        would never appear as a terminating stage and its presence in the order would be
        decorative."""
        mapped = set(dict(_REASON_STAGE).values())
        assert mapped == set(_EVALUATION_STAGE_ORDER)

    def test_the_reason_to_stage_mapping_is_pinned(self, tmp_path: Path) -> None:
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"]["reason_stage"].update(
                {"SOURCE_EXCEEDS_MAX_DURATION": "decode"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="reason_stage must be"):
            _load_from_path(path)

    def test_unevaluated_stages_are_derived_not_authored(self, tmp_path: Path) -> None:
        """An ordered suffix of the stage order. Two implementations serialising the same
        stages differently would produce different manifest bytes and different digests."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {"unevaluated_stages_derivation": "implementation_defined"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="unevaluated_stages_derivation must be"):
            _load_from_path(path)

    def test_the_record_schema_covers_ineligible_outcomes_only(self, tmp_path: Path) -> None:
        """An abort can happen before any source is in hand: a golden-fixture failure has no
        logical source id, no stage in this machine, and no reason from this vocabulary.
        Requiring the core on every record would have made those unrepresentable."""
        assert load_protocol().source_outcomes.record_scope == "source_ineligible_outcomes_only"
        path = variant(
            tmp_path, lambda d: d["source_outcomes"].update({"record_scope": "all_failures"})
        )
        with pytest.raises(ProtocolValueError, match="record_scope must be"):
            _load_from_path(path)

    def test_record_constraints_pin_ordering_not_just_membership(self, tmp_path: Path) -> None:
        """Two implementations holding the same supplemental set and serialising it in
        different orders produce different manifest bytes, and therefore different digests,
        which breaks the content addressing the benchmark rests on."""
        constraints = load_protocol().source_outcomes.record_constraints
        assert "supplemental_reason_codes_ordered_by_reason_precedence" in constraints
        assert "unevaluated_evaluation_stages_ordered_by_evaluation_stage_order" in constraints
        assert "digest_discarded_when_instability_detected" in constraints
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {
                    "record_constraints": [
                        c
                        for c in d["source_outcomes"]["record_constraints"]
                        if c != "digest_discarded_when_instability_detected"
                    ]
                }
            ),
        )
        with pytest.raises(ProtocolValueError, match="record_constraints must be"):
            _load_from_path(path)

    def test_conditions_are_closed_tokens_not_sentences(self) -> None:
        """The conditions were English prose, which is the pattern v4 spent every round
        eliminating everywhere else."""
        for condition in load_protocol().source_outcomes.record_conditional.values():
            assert condition.startswith("when_")
            assert " " not in condition

    def test_the_identity_stream_bound_cannot_be_set_arbitrarily_large(
        self, tmp_path: Path
    ) -> None:
        """A resource-security limit, so unlike other magnitudes it is bounded relative to
        the size that triggers it. Positivity and ordering checks alone would pass a
        petabyte."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"max_identity_stream_bytes": 1 << 50}
            ),
        )
        with pytest.raises(ProtocolValueError, match="must equal 16x max_source_bytes"):
            _load_from_path(path)

    def test_the_identity_stream_bound_is_derived_not_merely_capped(self, tmp_path: Path) -> None:
        """A permitted range would leave a band of configurations that are semantically
        accepted and have never been analysed. 12x is inside any plausible cap and is still
        refused."""
        path = variant(
            tmp_path,
            lambda d: d["canonicalisation"]["decode"].update(
                {"max_identity_stream_bytes": 12 * 536870912}
            ),
        )
        with pytest.raises(ProtocolValueError, match="must equal 16x max_source_bytes"):
            _load_from_path(path)

    def test_every_frozen_bound_has_a_reason_to_reject_against(self) -> None:
        """A bound the loader enforces with no code to record it cannot appear in the
        reduction chain. `max_decoded_bytes` had none: a highly compressible 192 kHz source
        can sit under the byte cap and under the duration cap while decoding to 10.3 GiB
        against a 1.50 GiB limit, so the case was reachable and unnameable."""
        reasons = load_protocol().source_outcomes.ineligible_reasons
        for code in (
            "SOURCE_EXCEEDS_MAX_BYTES",
            "SOURCE_EXCEEDS_MAX_DURATION",
            "SOURCE_EXCEEDS_MAX_DECODED_BYTES",
            "SOURCE_BELOW_MIN_SAMPLE_RATE",
        ):
            assert code in reasons

    def test_exactly_one_reason_is_counted_and_the_order_is_frozen(self, tmp_path: Path) -> None:
        """A candidate can be oversized and unsupported and too long at once. Without a
        frozen order, two conforming implementations publish different category totals for
        the same artifact."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {"reason_precedence": list(reversed(d["source_outcomes"]["reason_precedence"]))}
            ),
        )
        with pytest.raises(ProtocolValueError, match="reason_precedence must be"):
            _load_from_path(path)

    def test_the_precedence_covers_the_vocabulary_exactly(self) -> None:
        """A code missing from the precedence has no defined position, and a code in the
        precedence but not the vocabulary could be recorded and never counted."""
        assert sorted(_REASON_PRECEDENCE) == sorted(_INELIGIBLE_REASONS)
        assert len(_REASON_PRECEDENCE) == len(set(_REASON_PRECEDENCE))

    def test_the_precedence_follows_what_is_knowable_at_each_stage(self) -> None:
        """The size is known from fstat before anything is read, and the array checks need a
        decoded array, so those bracket the order. Anything else would place a reason before
        the point at which it could be evaluated."""
        order = list(_REASON_PRECEDENCE)
        assert order[0] == "SOURCE_EXCEEDS_MAX_BYTES"
        assert order.index("SOURCE_INSPECTION_REJECTED") < order.index("UNSUPPORTED_CONTAINER")
        assert order.index("SOURCE_DECODE_REJECTED") < order.index("EMPTY_SOURCE")
        assert order[-1] == "INVALID_SOURCE_AMPLITUDE"

    def test_supplemental_reasons_are_recorded_without_being_counted(self, tmp_path: Path) -> None:
        """Counting them would make the category totals overlap and stop them summing to the
        narrowing they describe."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update({"supplemental_reasons": "recorded_and_counted"}),
        )
        with pytest.raises(ProtocolValueError, match="supplemental_reasons must be"):
            _load_from_path(path)

    def test_an_excluded_source_still_gets_a_whole_artifact_digest(self, tmp_path: Path) -> None:
        """`record_on_failure` requires local_sha256 on every failure, but an oversized source
        is rejected from fstat before it is ever buffered. Hashing the bounded prefix and
        calling it local_sha256 would claim an identity for the whole artifact that was never
        computed, so exclusions stream through SHA-256 without decoding."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update(
                {"identity_for_excluded_sources": "prefix_sha256_of_bounded_read"}
            ),
        )
        with pytest.raises(ProtocolValueError, match="identity_for_excluded_sources must be"):
            _load_from_path(path)

    def test_the_streamed_artifact_must_be_proven_stable(self, tmp_path: Path) -> None:
        """The single-buffer path is stable by construction. The streaming path reads over an
        interval, so a file changing underneath the hash would be published as a stable
        identity that was never true."""
        path = variant(
            tmp_path,
            lambda d: d["source_outcomes"].update({"artifact_stability_check": "none"}),
        )
        with pytest.raises(ProtocolValueError, match="artifact_stability_check must be"):
            _load_from_path(path)

    def test_an_unstable_local_file_aborts_rather_than_blaming_the_source(self) -> None:
        """A file mutating on our own disk is a fact about this environment."""
        outcomes = load_protocol().source_outcomes
        assert "LOCAL_ARTIFACT_UNSTABLE" in outcomes.abort_reasons
        assert "LOCAL_ARTIFACT_UNSTABLE" not in outcomes.ineligible_reasons

    def test_a_capacity_failure_is_an_abort_and_never_a_source_verdict(self) -> None:
        """A source inside every frozen bound that this machine cannot allocate for is a fact
        about where the run happened. Recording it against the recording would publish our own
        capacity as a property of someone else's audio. It is listed apart from
        ENVIRONMENT_MISMATCH because only one of the two is fixed by installing something."""
        outcomes = load_protocol().source_outcomes
        assert "ENVIRONMENT_CAPACITY_FAILURE" in outcomes.abort_reasons
        assert "ENVIRONMENT_CAPACITY_FAILURE" not in outcomes.ineligible_reasons
        assert not any("CAPACITY" in code for code in outcomes.ineligible_reasons)

    def test_the_two_code_vocabularies_are_disjoint(self) -> None:
        """Asserted over the constants rather than over a loaded config, because once the
        config is required to equal them exactly, a config-level overlap check can never
        fire. This is the property that has to hold, checked where it can still fail."""
        assert not set(_INELIGIBLE_REASONS) & set(_ABORT_REASONS)

    def test_the_code_precondition_order_puts_the_fixture_first(self) -> None:
        """Same reasoning. If the environment cannot decode a known-good file, the first
        candidate it meets must not be blamed for it."""
        assert _PRECONDITION_ORDER.index("golden_fixture") < _PRECONDITION_ORDER.index(
            "inspect_and_decode"
        )
        assert _PRECONDITION_ORDER.index("local_artifact_identity") < _PRECONDITION_ORDER.index(
            "inspect_and_decode"
        )
        assert _PRECONDITION_ORDER[-1] == "classify_outcome"

    def test_the_golden_fixture_must_be_decoded_before_any_candidate(self, tmp_path: Path) -> None:
        """Otherwise a decoder that cannot decode anything is charged to the first source
        it meets, and the collection takes the blame for our environment."""
        reordered = [
            "protocol_semantics",
            "runtime_conformance",
            "local_artifact_identity",
            "inspect_and_decode",
            "golden_fixture",
            "classify_outcome",
        ]
        path = variant(
            tmp_path, lambda d: d["source_outcomes"].update({"precondition_order": reordered})
        )
        with pytest.raises(ProtocolValueError, match="precondition_order must be"):
            _load_from_path(path)

    def test_the_reduction_chain_records_every_narrowing(self) -> None:
        """A chain that reports only survivors makes the denominator unreconstructable,
        which is the withdrawn benchmark's defect one level up."""
        chain = load_protocol().source_outcomes.reduction_chain
        assert chain[0] == "selected_sources"
        assert chain[-1] == "frames_eligible"
        for stage in ("decode_succeeded", "source_domain_supported", "complete_frames_produced"):
            assert stage in chain

    def test_a_failed_source_still_carries_its_identity(self) -> None:
        outcomes = load_protocol().source_outcomes
        for field in (
            "logical_source_id",
            "terminating_evaluation_stage",
            "primary_reason_code",
            "identity_status",
        ):
            assert field in outcomes.record_required_core


class TestCanonicalisationLibrariesAreEnforced:
    """The canonical waveform is the output of these libraries, so a version recorded
    but unchecked would let a different library produce artifacts under this protocol's
    name. Only the Python packages are enforced: the lockfile makes those deterministic,
    while native libsndfile and libsoxr vary by platform and are recorded as provenance.
    """

    @pytest.mark.parametrize("package", ["soundfile", "librosa", "soxr"])
    def test_a_substituted_package_version_refuses_to_load(
        self, monkeypatch: pytest.MonkeyPatch, package: str
    ) -> None:
        real = importlib.metadata.version

        def fake(name: str) -> str:
            return "0.0.1" if name == package else real(name)

        monkeypatch.setattr(importlib.metadata, "version", fake)
        with pytest.raises(ProtocolEnvironmentError, match=f"protocol pins {package} "):
            load_protocol()

    @pytest.mark.parametrize("package", ["soundfile", "librosa", "soxr"])
    def test_a_missing_package_refuses_to_load(
        self, monkeypatch: pytest.MonkeyPatch, package: str
    ) -> None:
        real = importlib.metadata.version

        def fake(name: str) -> str:
            if name == package:
                raise importlib.metadata.PackageNotFoundError(name)
            return real(name)

        monkeypatch.setattr(importlib.metadata, "version", fake)
        with pytest.raises(ProtocolEnvironmentError, match=f"{package} is not installed"):
            load_protocol()


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
