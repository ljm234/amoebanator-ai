"""
Phase 1.1 De-identification Module — Comprehensive Test Suite.

Tests cover:
  - Safe Harbor processing (18 identifiers, age cap, ZIP truncation,
    date generalisation, free-text scrubbing, pseudonymization)
  - k-Anonymity enforcement (equivalence classes, generalisation,
    suppression, information loss metric)
  - Differential privacy mechanisms (Laplace, Gaussian, Exponential)
  - Privacy budget tracking and allocation
  - Full pipeline orchestration across all three layers
  - Factory functions and report generation
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from ml.data.deidentification import (
    DeidentificationConfig,
    DeidentificationMethod,
    ExponentialMechanism,
    GaussianMechanism,
    KAnonymityConfig,
    KAnonymityProcessor,
    LaplaceMechanism,
    PrivacyBudget,
    PrivacyLevel,
    SafeHarborConfig,
    SafeHarborProcessor,
    create_deidentification_pipeline,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_clinical_record(**overrides: object) -> dict[str, object]:
    """Build a sample clinical record with optional overrides."""
    base: dict[str, object] = {
        "patient_id": "PAT-001",
        "name": "John Doe",
        "email": "jdoe@example.com",
        "ssn": "123-45-6789",
        "phone": "801-555-0100",
        "date_of_birth": "1985-06-15",
        "zip_code": "84408",
        "age": 38,
        "sex": "M",
        "geographic_region": "Utah, Western USA",
        "csf_glucose": 45.0,
        "csf_protein": 120.0,
        "csf_wbc": 350.0,
        "diagnosis": "suspected_pam",
        "collection_date": "2025-09-10T14:30:00",
    }
    base.update(overrides)
    return base


def _make_record_batch(n: int = 20) -> list[dict[str, object]]:
    """Build a batch of clinical records for k-anonymity testing."""
    records: list[dict[str, object]] = []
    ages = [25, 25, 30, 30, 35, 35, 40, 40, 45, 45, 50, 50, 55, 55, 60, 60, 65, 65, 70, 70]
    sexes = ["M", "F"] * 10
    regions = ["Utah, Western USA"] * 10 + ["Arizona, Western USA"] * 10
    for i in range(min(n, 20)):
        rec = _make_clinical_record(
            patient_id=f"PAT-{i:03d}",
            name=f"Patient {i}",
            email=f"pat{i}@example.com",
            age=ages[i],
            sex=sexes[i],
            geographic_region=regions[i],
        )
        records.append(rec)
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Safe Harbor Processor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSafeHarborProcessor:
    """HIPAA Safe Harbor de-identification tests."""

    def test_removes_direct_identifiers(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record()
        result = proc.process_record(record)
        assert "name" not in result
        assert "email" not in result
        assert "ssn" not in result
        assert "phone" not in result
        assert "date_of_birth" not in result

    def test_preserves_non_identifier_fields(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record()
        result = proc.process_record(record)
        assert "diagnosis" in result
        assert "csf_glucose" in result
        assert "csf_protein" in result

    def test_age_cap_at_89(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(age=95)
        result = proc.process_record(record)
        assert result["age"] == 89

    def test_age_below_cap_unchanged(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(age=45)
        result = proc.process_record(record)
        assert result["age"] == 45

    def test_custom_age_cap(self) -> None:
        config = SafeHarborConfig(age_cap=80)
        proc = SafeHarborProcessor(config)
        record = _make_clinical_record(age=85)
        result = proc.process_record(record)
        assert result["age"] == 80

    def test_zip_truncation_to_3_digits(self) -> None:
        proc = SafeHarborProcessor()
        # Use "postal_code" — "zip_code" is a direct identifier that gets
        # removed before geographic truncation runs.  The truncation step
        # matches any surviving key containing "zip" or "postal".
        record = _make_clinical_record(postal_code="84408")
        result = proc.process_record(record)
        assert result["postal_code"] == "844"

    def test_zip_small_population_prefix(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(postal_code="03601")
        result = proc.process_record(record)
        assert result["postal_code"] == "000"

    def test_zip_short_code(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(postal_code="84")
        result = proc.process_record(record)
        assert result["postal_code"] == "000"

    def test_date_generalization_to_year(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(collection_date="2025-09-10T14:30:00")
        result = proc.process_record(record)
        assert result["collection_date"] == "2025"

    def test_date_generalization_to_month(self) -> None:
        config = SafeHarborConfig(date_precision="month")
        proc = SafeHarborProcessor(config)
        record = _make_clinical_record(collection_date="2025-09-10T14:30:00")
        result = proc.process_record(record)
        assert result["collection_date"] == "2025-09"

    def test_date_generalization_to_day(self) -> None:
        config = SafeHarborConfig(date_precision="day")
        proc = SafeHarborProcessor(config)
        record = _make_clinical_record(collection_date="2025-09-10T14:30:00")
        result = proc.process_record(record)
        assert result["collection_date"] == "2025-09-10"

    def test_date_none_returns_none(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(collection_date=None)
        result = proc.process_record(record)
        assert result["collection_date"] is None

    def test_date_invalid_returns_none(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record(collection_date="not-a-date")
        result = proc.process_record(record)
        assert result["collection_date"] is None

    def test_date_datetime_object(self) -> None:
        proc = SafeHarborProcessor()
        dt = datetime(2025, 9, 10, 14, 30, tzinfo=timezone.utc)
        record = _make_clinical_record(collection_date=dt)
        result = proc.process_record(record)
        assert result["collection_date"] == "2025"

    def test_free_text_scrubs_phone(self) -> None:
        proc = SafeHarborProcessor()
        record = {"notes": "Contact patient at 801-555-1234 for follow-up"}
        result = proc.process_record(record)
        assert "801-555-1234" not in result["notes"]
        assert "[REDACTED]" in result["notes"]

    def test_free_text_scrubs_ssn(self) -> None:
        proc = SafeHarborProcessor()
        record = {"notes": "SSN is 123-45-6789 as confirmed"}
        result = proc.process_record(record)
        assert "123-45-6789" not in result["notes"]

    def test_free_text_scrubs_email_in_text(self) -> None:
        proc = SafeHarborProcessor()
        record = {"notes": "Send results to user@hospital.org tomorrow"}
        result = proc.process_record(record)
        assert "user@hospital.org" not in result["notes"]

    def test_free_text_scrubs_dates(self) -> None:
        proc = SafeHarborProcessor()
        record = {"notes": "Patient born on 06/15/1985 confirmed exposure"}
        result = proc.process_record(record)
        assert "06/15/1985" not in result["notes"]

    def test_actions_log_populated(self) -> None:
        proc = SafeHarborProcessor()
        record = _make_clinical_record()
        proc.process_record(record)
        actions = proc.actions
        assert len(actions) > 0
        removal_actions = [
            a for a in actions if a.method == DeidentificationMethod.REMOVAL
        ]
        assert len(removal_actions) >= 4  # name, email, ssn, phone, etc.

    def test_process_batch(self) -> None:
        proc = SafeHarborProcessor()
        batch = _make_record_batch(5)
        results = proc.process_batch(batch)
        assert len(results) == 5
        for r in results:
            assert "name" not in r

    def test_pseudonymize(self) -> None:
        proc = SafeHarborProcessor()
        pseudo = proc._pseudonymize("John Doe")
        assert pseudo.startswith("PSEUDO_")
        assert len(pseudo) == 19  # PSEUDO_ + 12 hex chars

    def test_pseudonymize_deterministic(self) -> None:
        config = SafeHarborConfig(salt=b"fixed-salt-for-test")
        proc = SafeHarborProcessor(config)
        p1 = proc._pseudonymize("John Doe")
        p2 = proc._pseudonymize("John Doe")
        assert p1 == p2

    def test_date_non_string_non_datetime_returns_none(self) -> None:
        proc = SafeHarborProcessor()
        result = proc._generalise_date(12345)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# k-Anonymity Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKAnonymityProcessor:
    """k-Anonymity enforcement tests."""

    def test_k_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 2"):
            KAnonymityConfig(k=1)

    def test_enforce_returns_k_anonymous_dataset(self) -> None:
        config = KAnonymityConfig(
            k=2,
            quasi_identifiers=("age", "sex", "geographic_region"),
        )
        proc = KAnonymityProcessor(config)
        records = _make_record_batch(20)
        result = proc.enforce(records)
        # Every equivalence class should have >= k members
        classes = proc._compute_equivalence_classes(result)
        for key, count in classes.items():
            assert count >= 2, f"Equivalence class {key} has only {count} members"

    def test_suppressed_count_tracked(self) -> None:
        config = KAnonymityConfig(
            k=10,
            quasi_identifiers=("age", "sex"),
        )
        proc = KAnonymityProcessor(config)
        records = _make_record_batch(20)
        result = proc.enforce(records)
        # Some records may be suppressed
        assert proc.suppressed_count >= 0
        assert len(result) + proc.suppressed_count <= 20

    def test_information_loss_bounded(self) -> None:
        config = KAnonymityConfig(k=2)
        proc = KAnonymityProcessor(config)
        original = _make_record_batch(20)
        anonymised = proc.enforce(original)
        loss = proc.get_information_loss(original, anonymised)
        assert 0.0 <= loss <= 1.0

    def test_information_loss_empty_dataset(self) -> None:
        proc = KAnonymityProcessor()
        assert proc.get_information_loss([], []) == 0.0

    def test_generalisation_applies(self) -> None:
        config = KAnonymityConfig(
            k=5,
            quasi_identifiers=("age",),
            generalisation_hierarchies={
                "age": [
                    lambda v: (v // 10) * 10,
                    lambda _: "*",
                ],
            },
        )
        proc = KAnonymityProcessor(config)
        records = [{"age": 25 + i} for i in range(10)]
        result = proc.enforce(records)
        for r in result:
            if r["age"] != "*":
                assert isinstance(r["age"], int)
                assert r["age"] % 10 == 0

    def test_suppress_violations(self) -> None:
        config = KAnonymityConfig(
            k=5,
            quasi_identifiers=("age",),
            generalisation_hierarchies={"age": []},
        )
        proc = KAnonymityProcessor(config)
        # Only 1 record per age → all must be suppressed
        records = [{"age": i} for i in range(5)]
        result = proc.enforce(records)
        assert len(result) == 0
        assert proc.suppressed_count == 5

    def test_already_k_anonymous(self) -> None:
        config = KAnonymityConfig(k=2, quasi_identifiers=("age",))
        proc = KAnonymityProcessor(config)
        records = [{"age": 30}] * 10  # all same QI
        result = proc.enforce(records)
        assert len(result) == 10

    def test_generalisation_error_handling(self) -> None:
        def _bad_generaliser(_: object) -> str:
            raise TypeError("deliberate generalisation failure")

        config = KAnonymityConfig(
            k=2,
            quasi_identifiers=("age",),
            generalisation_hierarchies={
                "age": [_bad_generaliser],
            },
        )
        proc = KAnonymityProcessor(config)
        # Unique ages → each equivalence class has size 1, which is < k=2.
        # The generaliser raises TypeError (caught by the handler),
        # collapsing every value to "*", which makes a single equivalence
        # class of size 5 ≥ k=2.
        records = [{"age": i} for i in range(5)]
        result = proc.enforce(records)
        assert all(r["age"] == "*" for r in result)


# ═══════════════════════════════════════════════════════════════════════════
# Differential Privacy Mechanism Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLaplaceMechanism:
    """Laplace noise mechanism tests."""

    def test_invalid_sensitivity_raises(self) -> None:
        with pytest.raises(ValueError, match="Sensitivity must be positive"):
            LaplaceMechanism(sensitivity=0, epsilon=1.0)

    def test_invalid_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            LaplaceMechanism(sensitivity=1.0, epsilon=0)

    def test_scale_computation(self) -> None:
        mech = LaplaceMechanism(sensitivity=2.0, epsilon=0.5)
        assert mech.scale == pytest.approx(4.0)

    def test_add_noise_changes_value(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        noised = mech.add_noise(100.0)
        assert noised != 100.0

    def test_add_noise_mean_converges(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        samples = [mech.add_noise(100.0) for _ in range(10000)]
        mean = np.mean(samples)
        assert abs(mean - 100.0) < 0.5  # should be close to true value

    def test_add_noise_batch(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        values = np.array([10.0, 20.0, 30.0])
        noised = mech.add_noise_batch(values)
        assert noised.shape == (3,)
        assert not np.array_equal(values, noised)

    def test_confidence_interval(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        lower, upper = mech.confidence_interval(100.0, confidence=0.95)
        assert lower < 100.0
        assert upper > 100.0
        assert upper - lower > 0

    def test_confidence_interval_wider_at_lower_epsilon(self) -> None:
        mech_tight = LaplaceMechanism(sensitivity=1.0, epsilon=2.0)
        mech_wide = LaplaceMechanism(sensitivity=1.0, epsilon=0.1)
        ci_tight = mech_tight.confidence_interval(0.0, 0.95)
        ci_wide = mech_wide.confidence_interval(0.0, 0.95)
        width_tight = ci_tight[1] - ci_tight[0]
        width_wide = ci_wide[1] - ci_wide[0]
        assert width_wide > width_tight


class TestGaussianMechanism:
    """Gaussian noise mechanism tests."""

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=0, epsilon=1.0)
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=0)
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=0)

    def test_sigma_computation(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert mech.sigma > 0

    def test_add_noise(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        noised = mech.add_noise(50.0)
        assert noised != 50.0

    def test_add_noise_batch(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        values = np.array([10.0, 20.0, 30.0])
        noised = mech.add_noise_batch(values)
        assert noised.shape == (3,)

    def test_noise_mean_converges(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, seed=42)
        samples = [mech.add_noise(0.0) for _ in range(10000)]
        mean = np.mean(samples)
        assert abs(mean) < 0.5


class TestExponentialMechanism:
    """Exponential mechanism for categorical selection tests."""

    def test_select_from_candidates(self) -> None:
        mech = ExponentialMechanism(epsilon=1.0, seed=42)
        candidates = ["cat_A", "cat_B", "cat_C"]
        utilities = [10.0, 1.0, 1.0]
        selected = mech.select(candidates, utilities)
        assert selected in candidates

    def test_high_utility_selected_more_often(self) -> None:
        mech = ExponentialMechanism(epsilon=5.0, seed=42)
        candidates = ["best", "worst"]
        utilities = [100.0, 0.0]
        counts = {"best": 0, "worst": 0}
        for _ in range(1000):
            s = mech.select(candidates, utilities)
            counts[s] += 1
        assert counts["best"] > counts["worst"]

    def test_handles_zero_weights(self) -> None:
        mech = ExponentialMechanism(epsilon=1.0, seed=42)
        candidates = ["a", "b"]
        utilities = [-1e10, -1e10]  # very low → near-zero weights
        selected = mech.select(candidates, utilities)
        assert selected in candidates


# ═══════════════════════════════════════════════════════════════════════════
# Privacy Budget Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyBudget:
    """Privacy budget tracking tests."""

    def test_initial_state(self) -> None:
        pb = PrivacyBudget(total_epsilon=1.0)
        assert pb.remaining_epsilon == pytest.approx(1.0)
        assert pb.spent_epsilon == 0.0

    def test_allocate_succeeds(self) -> None:
        pb = PrivacyBudget(total_epsilon=1.0)
        assert pb.allocate("age", 0.3) is True
        assert pb.remaining_epsilon == pytest.approx(0.7)

    def test_allocate_exceeds_budget(self) -> None:
        pb = PrivacyBudget(total_epsilon=1.0)
        pb.allocate("age", 0.8)
        assert pb.allocate("csf_wbc", 0.5) is False

    def test_reset(self) -> None:
        pb = PrivacyBudget(total_epsilon=1.0)
        pb.allocate("age", 0.5)
        pb.reset()
        assert pb.remaining_epsilon == pytest.approx(1.0)
        assert pb.field_budgets == {}

    def test_remaining_epsilon_never_negative(self) -> None:
        pb = PrivacyBudget(total_epsilon=0.1)
        pb.allocate("age", 0.1)
        assert pb.remaining_epsilon >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# De-identification Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDeidentificationPipeline:
    """Full pipeline orchestration tests."""

    def test_safe_harbor_only(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.SAFE_HARBOR_ONLY,
        )
        records = _make_record_batch(5)
        result = pipeline.process(records)
        assert len(result) == 5
        for r in result:
            assert "name" not in r
        report = pipeline.report
        assert report.privacy_level == "safe_harbor"
        assert report.safe_harbor_actions > 0

    def test_k_anonymous(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.K_ANONYMOUS,
            k=2,
        )
        records = _make_record_batch(20)
        result = pipeline.process(records)
        assert len(result) <= 20
        report = pipeline.report
        assert report.privacy_level == "k_anonymous"

    def test_full_pipeline(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.FULL_PIPELINE,
            k=2,
            total_epsilon=1.0,
            seed=42,
        )
        records = _make_record_batch(20)
        result = pipeline.process(records)
        assert len(result) <= 20
        report = pipeline.report
        assert report.privacy_level == "full_pipeline"
        assert report.epsilon_spent > 0

    def test_report_to_dict(self) -> None:
        pipeline = create_deidentification_pipeline(seed=42)
        records = _make_record_batch(10)
        pipeline.process(records)
        d = pipeline.report.to_dict()
        assert "input_count" in d
        assert d["input_count"] == 10

    def test_pipeline_config_defaults(self) -> None:
        config = DeidentificationConfig()
        assert config.privacy_level == PrivacyLevel.FULL_PIPELINE
        assert config.k_anonymity.k == 5

    def test_report_timestamp(self) -> None:
        pipeline = create_deidentification_pipeline(seed=42)
        pipeline.process(_make_record_batch(5))
        assert pipeline.report.timestamp != ""

    def test_differentially_private_level(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.DIFFERENTIALLY_PRIVATE,
            k=2,
            total_epsilon=2.0,
            seed=42,
        )
        records = _make_record_batch(20)
        result = pipeline.process(records)
        assert pipeline.report.epsilon_spent > 0
        assert len(result) > 0

    def test_pipeline_handles_non_numeric_gracefully(self) -> None:
        pipeline = create_deidentification_pipeline(
            privacy_level=PrivacyLevel.FULL_PIPELINE,
            k=2,
            seed=42,
        )
        records = [{"age": "unknown", "sex": "M", "geographic_region": "Utah"}] * 10
        result = pipeline.process(records)
        assert len(result) > 0
