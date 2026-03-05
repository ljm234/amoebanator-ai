"""
Phase 1.1 Compliance Module — Comprehensive Test Suite.

Tests cover:
  - ORCID validation (format, checksum, ISO 7064 Mod 11-2)
  - Researcher identity construction and validation
  - CITI training completeness and expiration checks
  - IRB application state machine (all valid and invalid transitions)
  - CDC Form 0.1310 field validation
  - Security attestation signing, verification, and expiration
  - Compliance gate evaluation (all 4 stages)
  - Factory functions
  - Serialisation round-trips
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ml.data.compliance import (
    ALL_SAFEGUARDS,
    AttestationStatus,
    CDCDataRequestForm,
    CDCFormStatus,
    CITICompletion,
    ComplianceGate,
    ComplianceGateResult,
    IRBApplication,
    IRBStatus,
    IRBTransition,
    ResearcherIdentity,
    ResearcherValidator,
    SafeguardCategory,
    SecurityAttestation,
    ValidationIssue,
    _validate_orcid_checksum,
    create_compliance_gate,
    create_irb_application,
    create_researcher_identity,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_citi_completions(
    expired: bool = False,
    low_score: bool = False,
    missing: frozenset[str] | None = None,
) -> tuple[CITICompletion, ...]:
    """Build a set of CITI completions for testing."""
    from ml.data.compliance import CITI_REQUIRED_MODULES

    now = datetime.now(tz=timezone.utc)
    modules = CITI_REQUIRED_MODULES - (missing or frozenset())
    comps = []
    for mod_id in sorted(modules):
        exp_date = now - timedelta(days=30) if expired else now + timedelta(days=365)
        score = 65.0 if low_score else 95.0
        comps.append(
            CITICompletion(
                module_id=mod_id,
                module_name=mod_id.replace("_", " ").title(),
                completion_date=now - timedelta(days=60),
                expiration_date=exp_date,
                score=score,
                certificate_id=f"CERT-{mod_id[:4].upper()}",
            )
        )
    return tuple(comps)


def _make_valid_identity() -> ResearcherIdentity:
    """Create a fully valid researcher identity."""
    return ResearcherIdentity(
        orcid="0000-0002-1825-0097",
        full_name="Jordan Montenegro",
        email="jmontenegro@weber.edu",
        institution="Weber State University",
        department="Computer Science",
        role="Independent Researcher",
        citi_completions=_make_citi_completions(),
        verified_at=datetime.now(tz=timezone.utc),
    )


def _make_approved_irb(identity: ResearcherIdentity | None = None) -> IRBApplication:
    """Create an IRB application in APPROVED state with future expiration."""
    identity = identity or _make_valid_identity()
    irb = IRBApplication(
        protocol_title="PAM Diagnostic ML Pipeline",
        protocol_number="IRB-2026-001",
        pi_identity=identity,
    )
    irb.transition_to(IRBStatus.SUBMITTED, actor="PI")
    irb.transition_to(IRBStatus.UNDER_REVIEW, actor="IRB Chair")
    irb.transition_to(IRBStatus.APPROVED, actor="IRB Committee")
    return irb


def _make_submitted_cdc_form() -> CDCDataRequestForm:
    """Create a CDC form that has passed validation and been submitted."""
    form = CDCDataRequestForm()
    form.fields = {
        "investigator_name": "Jordan Montenegro",
        "institution": "Weber State University",
        "project_title": "PAM Diagnostic Pipeline",
        "data_description": "De-identified PAM surveillance records 2015-2025",
        "justification": "Machine learning model for rapid PAM screening",
        "irb_protocol_number": "IRB-2026-001",
        "irb_approval_date": datetime.now(tz=timezone.utc).isoformat(),
        "data_security_plan": (
            "All data will be encrypted using AES-256-GCM at rest and TLS 1.3 "
            "in transit. Access is restricted via role-based access control with "
            "multi-factor authentication. Physical safeguards include locked "
            "server rooms with biometric access. Annual HIPAA training is mandatory "
            "for all personnel with data access. Audit logs are maintained with "
            "cryptographic tamper detection."
        ),
        "destruction_plan": (
            "Data destruction follows NIST SP 800-88 Rev. 1 media sanitization "
            "guidelines. All electronic media containing PHI will undergo "
            "cryptographic erasure followed by physical destruction and shredding."
        ),
        "signature_date": datetime.now(tz=timezone.utc).isoformat(),
    }
    issues = form.validate()
    assert not any(i.severity == "error" for i in issues)
    form.mark_submitted(receipt_id="CDC-REC-2026-0001")
    return form


def _make_signed_attestation(identity: ResearcherIdentity | None = None) -> tuple[SecurityAttestation, bytes]:
    """Create a signed security attestation with all safeguards compliant."""
    att = SecurityAttestation(researcher=identity or _make_valid_identity())
    for sg in ALL_SAFEGUARDS:
        att.respond(sg.check_id, compliant=True)
    key = b"test-signing-key-32-bytes-long!!"
    att.sign(key)
    return att, key


# ═══════════════════════════════════════════════════════════════════════════
# ORCID Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestORCIDValidation:
    """ORCID identifier format and checksum tests."""

    def test_valid_orcid_checksum(self) -> None:
        assert _validate_orcid_checksum("0000000218250097") is True

    def test_invalid_orcid_checksum(self) -> None:
        assert _validate_orcid_checksum("0000000218250098") is False

    def test_orcid_with_x_check_digit(self) -> None:
        assert _validate_orcid_checksum("000000012281955X") is True

    def test_validator_rejects_bad_format(self) -> None:
        validator = ResearcherValidator()
        issues = validator._validate_orcid("not-an-orcid")
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "XXXX-XXXX-XXXX-XXXX" in issues[0].message

    def test_validator_rejects_bad_checksum(self) -> None:
        validator = ResearcherValidator()
        issues = validator._validate_orcid("0000-0002-1825-0098")
        assert len(issues) == 1
        assert "checksum" in issues[0].message.lower()

    def test_validator_accepts_valid_orcid(self) -> None:
        validator = ResearcherValidator()
        issues = validator._validate_orcid("0000-0002-1825-0097")
        assert issues == []


# ═══════════════════════════════════════════════════════════════════════════
# Researcher Identity & Validator Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResearcherIdentity:
    """Researcher identity construction and frozen dataclass behaviour."""

    def test_identity_is_frozen(self) -> None:
        identity = _make_valid_identity()
        with pytest.raises(AttributeError):
            identity.full_name = "Modified"  # type: ignore[misc]

    def test_identity_fields(self) -> None:
        identity = _make_valid_identity()
        assert identity.orcid == "0000-0002-1825-0097"
        assert identity.institution == "Weber State University"
        assert len(identity.citi_completions) > 0

    def test_factory_creates_verified_identity(self) -> None:
        identity = create_researcher_identity(
            orcid="0000-0002-1825-0097",
            full_name="Test User",
            email="test@example.edu",
            institution="Test University",
            department="CS",
        )
        assert identity.verified_at is not None


class TestResearcherValidator:
    """Comprehensive researcher validation tests."""

    def test_valid_researcher_passes(self) -> None:
        identity = _make_valid_identity()
        validator = ResearcherValidator()
        issues = validator.validate(identity)
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []

    def test_invalid_email_format(self) -> None:
        identity = ResearcherIdentity(
            orcid="0000-0002-1825-0097",
            full_name="Test",
            email="not-valid-email",
            institution="Test",
            department="CS",
            citi_completions=_make_citi_completions(),
        )
        validator = ResearcherValidator()
        issues = validator.validate(identity)
        email_issues = [i for i in issues if i.field_name == "email"]
        assert any(i.severity == "error" for i in email_issues)

    def test_non_institutional_email_warning(self) -> None:
        identity = ResearcherIdentity(
            orcid="0000-0002-1825-0097",
            full_name="Test",
            email="user@gmail.com",
            institution="Test",
            department="CS",
            citi_completions=_make_citi_completions(),
        )
        validator = ResearcherValidator()
        issues = validator.validate(identity)
        email_warns = [i for i in issues if i.field_name == "email" and i.severity == "warning"]
        assert len(email_warns) == 1
        assert "non-institutional" in email_warns[0].message.lower()

    def test_expired_citi_module(self) -> None:
        identity = ResearcherIdentity(
            orcid="0000-0002-1825-0097",
            full_name="Test",
            email="test@weber.edu",
            institution="Test",
            department="CS",
            citi_completions=_make_citi_completions(expired=True),
        )
        validator = ResearcherValidator()
        issues = validator.validate(identity)
        expired_issues = [
            i for i in issues
            if "expired" in i.message.lower()
        ]
        assert len(expired_issues) > 0

    def test_low_citi_score_warning(self) -> None:
        identity = ResearcherIdentity(
            orcid="0000-0002-1825-0097",
            full_name="Test",
            email="test@weber.edu",
            institution="Test",
            department="CS",
            citi_completions=_make_citi_completions(low_score=True),
        )
        validator = ResearcherValidator()
        issues = validator.validate(identity)
        score_warns = [
            i for i in issues
            if "80%" in i.message
        ]
        assert len(score_warns) > 0

    def test_missing_citi_modules(self) -> None:
        identity = ResearcherIdentity(
            orcid="0000-0002-1825-0097",
            full_name="Test",
            email="test@weber.edu",
            institution="Test",
            department="CS",
            citi_completions=_make_citi_completions(
                missing=frozenset({"hipaa_privacy", "data_security"})
            ),
        )
        validator = ResearcherValidator()
        issues = validator.validate(identity)
        missing_issues = [
            i for i in issues
            if "not completed" in i.message.lower()
        ]
        assert len(missing_issues) == 2

    def test_custom_required_modules(self) -> None:
        validator = ResearcherValidator(
            required_modules=frozenset({"custom_module"}),
        )
        identity = ResearcherIdentity(
            orcid="0000-0002-1825-0097",
            full_name="Test",
            email="test@weber.edu",
            institution="Test",
            department="CS",
            citi_completions=(),
        )
        issues = validator.validate(identity)
        missing = [i for i in issues if "custom_module" in i.message]
        assert len(missing) == 1

    def test_edu_domain_accepted(self) -> None:
        validator = ResearcherValidator()
        issues = validator._validate_email("user@stanford.edu")
        assert issues == []

    def test_gov_domain_accepted(self) -> None:
        validator = ResearcherValidator()
        issues = validator._validate_email("user@cdc.gov")
        assert issues == []

    def test_ac_domain_accepted(self) -> None:
        validator = ResearcherValidator()
        issues = validator._validate_email("user@oxford.ac.uk")
        assert issues == []


# ═══════════════════════════════════════════════════════════════════════════
# IRB Application Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIRBApplication:
    """IRB state machine, transitions, and lifecycle checks."""

    def test_initial_state_is_draft(self) -> None:
        irb = IRBApplication()
        assert irb.status == IRBStatus.DRAFT

    def test_valid_transition_draft_to_submitted(self) -> None:
        irb = IRBApplication()
        irb.transition_to(IRBStatus.SUBMITTED, actor="PI")
        assert irb.status == IRBStatus.SUBMITTED
        assert irb.submitted_date is not None

    def test_valid_full_approval_path(self) -> None:
        irb = _make_approved_irb()
        assert irb.status == IRBStatus.APPROVED
        assert irb.approval_date is not None
        assert irb.expiration_date is not None

    def test_transition_records_history(self) -> None:
        irb = IRBApplication()
        irb.transition_to(IRBStatus.SUBMITTED, actor="PI", notes="Initial submission")
        assert len(irb.transitions) == 1
        t = irb.transitions[0]
        assert t.from_status == IRBStatus.DRAFT
        assert t.to_status == IRBStatus.SUBMITTED
        assert t.actor == "PI"
        assert t.notes == "Initial submission"

    def test_invalid_transition_raises(self) -> None:
        irb = IRBApplication()
        with pytest.raises(ValueError, match="Invalid transition"):
            irb.transition_to(IRBStatus.APPROVED)

    def test_terminated_has_no_valid_transitions(self) -> None:
        irb = IRBApplication()
        irb.transition_to(IRBStatus.SUBMITTED)
        irb.transition_to(IRBStatus.UNDER_REVIEW)
        irb.transition_to(IRBStatus.TERMINATED)
        with pytest.raises(ValueError):
            irb.transition_to(IRBStatus.APPROVED)

    def test_revisions_requested_loops_to_submitted(self) -> None:
        irb = IRBApplication()
        irb.transition_to(IRBStatus.SUBMITTED)
        irb.transition_to(IRBStatus.REVISIONS_REQUESTED)
        irb.transition_to(IRBStatus.SUBMITTED)
        assert irb.status == IRBStatus.SUBMITTED
        assert len(irb.transitions) == 3

    def test_conditional_approval_to_approved(self) -> None:
        irb = IRBApplication()
        irb.transition_to(IRBStatus.SUBMITTED)
        irb.transition_to(IRBStatus.UNDER_REVIEW)
        irb.transition_to(IRBStatus.CONDITIONALLY_APPROVED)
        irb.transition_to(IRBStatus.APPROVED)
        assert irb.status == IRBStatus.APPROVED

    def test_is_current_when_approved(self) -> None:
        irb = _make_approved_irb()
        assert irb.is_current() is True

    def test_is_current_false_when_not_approved(self) -> None:
        irb = IRBApplication()
        assert irb.is_current() is False

    def test_days_until_expiration(self) -> None:
        irb = _make_approved_irb()
        days = irb.days_until_expiration()
        assert days is not None
        assert days > 1000  # 3 years ≈ 1095 days

    def test_days_until_expiration_none_when_no_expiry(self) -> None:
        irb = IRBApplication()
        assert irb.days_until_expiration() is None

    def test_compute_protocol_hash(self) -> None:
        irb = IRBApplication()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test protocol content for hashing")
            f.flush()
            h = irb.compute_protocol_hash(Path(f.name))
        assert len(h) == 64  # SHA-256 hex digest
        assert irb.protocol_hash == h

    def test_to_dict_serialisation(self) -> None:
        irb = _make_approved_irb()
        d = irb.to_dict()
        assert d["status"] == "approved"
        assert d["approval_date"] is not None
        assert d["expiration_date"] is not None
        assert d["transitions_count"] == 3

    def test_factory_creates_draft(self) -> None:
        identity = _make_valid_identity()
        irb = create_irb_application(
            protocol_title="Test",
            protocol_number="TP-001",
            pi_identity=identity,
        )
        assert irb.status == IRBStatus.DRAFT
        assert irb.pi_identity == identity

    def test_expired_to_draft_resubmission(self) -> None:
        irb = _make_approved_irb()
        irb.transition_to(IRBStatus.EXPIRED)
        assert irb.status == IRBStatus.EXPIRED
        irb.transition_to(IRBStatus.DRAFT)
        assert irb.status == IRBStatus.DRAFT

    def test_suspended_to_approved(self) -> None:
        irb = _make_approved_irb()
        irb.transition_to(IRBStatus.SUSPENDED)
        irb.transition_to(IRBStatus.APPROVED)
        assert irb.status == IRBStatus.APPROVED

    def test_suspended_to_terminated(self) -> None:
        irb = _make_approved_irb()
        irb.transition_to(IRBStatus.SUSPENDED)
        irb.transition_to(IRBStatus.TERMINATED)
        assert irb.status == IRBStatus.TERMINATED

    def test_renewal_pending_to_approved(self) -> None:
        irb = _make_approved_irb()
        irb.transition_to(IRBStatus.RENEWAL_PENDING)
        irb.transition_to(IRBStatus.APPROVED)
        assert irb.status == IRBStatus.APPROVED

    def test_renewal_pending_to_expired(self) -> None:
        irb = _make_approved_irb()
        irb.transition_to(IRBStatus.RENEWAL_PENDING)
        irb.transition_to(IRBStatus.EXPIRED)
        assert irb.status == IRBStatus.EXPIRED


# ═══════════════════════════════════════════════════════════════════════════
# CDC Form 0.1310 Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCDCDataRequestForm:
    """CDC Form 0.1310 field validation and lifecycle tests."""

    def test_initial_status_not_started(self) -> None:
        form = CDCDataRequestForm()
        assert form.status == CDCFormStatus.NOT_STARTED

    def test_set_field_moves_to_in_progress(self) -> None:
        form = CDCDataRequestForm()
        form.set_field("investigator_name", "Test")
        assert form.status == CDCFormStatus.IN_PROGRESS

    def test_empty_form_fails_validation(self) -> None:
        form = CDCDataRequestForm()
        issues = form.validate()
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 10  # at least 10 required fields missing
        assert form.status == CDCFormStatus.VALIDATION_FAILED

    def test_complete_form_passes(self) -> None:
        form = _make_submitted_cdc_form()
        assert form.status == CDCFormStatus.SUBMITTED

    def test_irb_date_too_old_warning(self) -> None:
        form = CDCDataRequestForm()
        form.fields = {
            "investigator_name": "Test",
            "institution": "Test U",
            "project_title": "Test Project",
            "data_description": "Test data",
            "justification": "Test justification",
            "irb_protocol_number": "IRB-001",
            "irb_approval_date": (
                datetime.now(tz=timezone.utc) - timedelta(days=400)
            ).isoformat(),
            "data_security_plan": " ".join(["word"] * 60),
            "destruction_plan": "NIST SP 800-88 sanitization and shredding",
            "signature_date": datetime.now(tz=timezone.utc).isoformat(),
        }
        issues = form.validate()
        date_warns = [
            i for i in issues
            if i.field_name == "irb_approval_date" and i.severity == "warning"
        ]
        assert len(date_warns) == 1

    def test_irb_date_invalid_format(self) -> None:
        form = CDCDataRequestForm()
        form.fields = {"irb_approval_date": "not-a-date"}
        form.validate()
        date_errs = [
            i for i in form.validation_issues
            if i.field_name == "irb_approval_date" and i.severity == "error"
        ]
        assert any("ISO 8601" in e.message for e in date_errs)

    def test_security_plan_too_short(self) -> None:
        form = CDCDataRequestForm()
        form.fields = {"data_security_plan": "Use encryption"}
        form.validate()
        plan_warns = [
            i for i in form.validation_issues
            if i.field_name == "data_security_plan" and i.severity == "warning"
        ]
        assert len(plan_warns) == 1

    def test_destruction_plan_missing_nist(self) -> None:
        form = CDCDataRequestForm()
        form.fields = {"destruction_plan": "We will delete the files"}
        form.validate()
        dest_infos = [
            i for i in form.validation_issues
            if i.field_name == "destruction_plan" and i.severity == "info"
        ]
        assert len(dest_infos) == 1
        assert dest_infos[0].suggestion is not None
        assert "NIST" in dest_infos[0].suggestion

    def test_destruction_plan_with_nist_passes(self) -> None:
        form = CDCDataRequestForm()
        form.fields = {"destruction_plan": "Follow NIST SP 800-88 guidelines"}
        form.validate()
        dest_infos = [
            i for i in form.validation_issues
            if i.field_name == "destruction_plan" and i.severity == "info"
        ]
        assert len(dest_infos) == 0

    def test_mark_submitted(self) -> None:
        form = _make_submitted_cdc_form()
        assert form.submission_receipt == "CDC-REC-2026-0001"
        assert form.submitted_at is not None

    def test_to_dict(self) -> None:
        form = _make_submitted_cdc_form()
        d = form.to_dict()
        assert d["form_version"] == "0.1310-2024"
        assert d["status"] == "submitted"
        assert d["submission_receipt"] is not None

    def test_ready_to_submit_status(self) -> None:
        form = CDCDataRequestForm()
        form.fields = {
            "investigator_name": "Test",
            "institution": "Test U",
            "project_title": "Test",
            "data_description": "Test",
            "justification": "Test",
            "irb_protocol_number": "IRB-001",
            "irb_approval_date": datetime.now(tz=timezone.utc).isoformat(),
            "data_security_plan": " ".join(["security"] * 60),
            "destruction_plan": "NIST 800-88 sanitization and destruction",
            "signature_date": datetime.now(tz=timezone.utc).isoformat(),
        }
        form.validate()
        assert form.status == CDCFormStatus.READY_TO_SUBMIT


# ═══════════════════════════════════════════════════════════════════════════
# Security Attestation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSecurityAttestation:
    """Security attestation signing, verification, and safeguard checks."""

    def test_initial_status_pending(self) -> None:
        att = SecurityAttestation()
        assert att.status == AttestationStatus.PENDING

    def test_respond_records_compliance(self) -> None:
        att = SecurityAttestation()
        att.respond("PHY-001", compliant=True)
        assert att.safeguard_responses["PHY-001"] is True

    def test_respond_invalid_check_raises(self) -> None:
        att = SecurityAttestation()
        with pytest.raises(ValueError, match="Unknown safeguard"):
            att.respond("INVALID-001", compliant=True)

    def test_all_checks_completed_false(self) -> None:
        att = SecurityAttestation()
        att.respond("PHY-001", compliant=True)
        assert att.all_checks_completed() is False

    def test_all_checks_completed_true(self) -> None:
        att = SecurityAttestation()
        for sg in ALL_SAFEGUARDS:
            att.respond(sg.check_id, compliant=True)
        assert att.all_checks_completed() is True

    def test_all_compliant(self) -> None:
        att = SecurityAttestation()
        for sg in ALL_SAFEGUARDS:
            att.respond(sg.check_id, compliant=True)
        assert att.all_compliant() is True

    def test_all_compliant_false_with_non_compliant(self) -> None:
        att = SecurityAttestation()
        for sg in ALL_SAFEGUARDS:
            att.respond(sg.check_id, compliant=True)
        att.respond("PHY-001", compliant=False)
        assert att.all_compliant() is False

    def test_sign_succeeds(self) -> None:
        att, key = _make_signed_attestation()
        assert att.status == AttestationStatus.SIGNED
        assert att.signed_at is not None
        assert att.expires_at is not None
        assert len(att.signature) == 64  # HMAC-SHA256 hex digest

    def test_sign_fails_incomplete(self) -> None:
        att = SecurityAttestation()
        att.respond("PHY-001", compliant=True)
        with pytest.raises(ValueError, match="not all safeguard checks"):
            att.sign(b"key")

    def test_sign_fails_non_compliant(self) -> None:
        att = SecurityAttestation()
        for sg in ALL_SAFEGUARDS:
            att.respond(sg.check_id, compliant=(sg.check_id != "PHY-001"))
        with pytest.raises(ValueError, match="not all safeguards are compliant"):
            att.sign(b"key")

    def test_verify_signature(self) -> None:
        att, key = _make_signed_attestation()
        assert att.verify_signature(key) is True

    def test_verify_signature_wrong_key(self) -> None:
        att, _ = _make_signed_attestation()
        assert att.verify_signature(b"wrong-key") is False

    def test_verify_signature_unsigned(self) -> None:
        att = SecurityAttestation()
        assert att.verify_signature(b"key") is False

    def test_is_current(self) -> None:
        att, _ = _make_signed_attestation()
        assert att.is_current() is True

    def test_is_current_false_when_pending(self) -> None:
        att = SecurityAttestation()
        assert att.is_current() is False

    def test_non_compliant_checks(self) -> None:
        att = SecurityAttestation()
        for sg in ALL_SAFEGUARDS:
            att.respond(sg.check_id, compliant=(sg.check_id != "TEC-003"))
        nc = att.non_compliant_checks()
        assert any(sg.check_id == "TEC-003" for sg in nc)

    def test_non_compliant_checks_includes_unanswered(self) -> None:
        att = SecurityAttestation()
        nc = att.non_compliant_checks()
        assert len(nc) == len(ALL_SAFEGUARDS)

    def test_to_dict(self) -> None:
        att, _ = _make_signed_attestation()
        d = att.to_dict()
        assert d["status"] == "signed"
        assert d["total_checks"] == len(ALL_SAFEGUARDS)
        assert d["compliant_checks"] == len(ALL_SAFEGUARDS)
        assert d["signature_present"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Safeguard Constants Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSafeguardConstants:
    """Validate safeguard definitions and categories."""

    def test_all_safeguards_count(self) -> None:
        assert len(ALL_SAFEGUARDS) == 12  # 3 physical + 5 technical + 4 admin

    def test_physical_safeguards_category(self) -> None:
        from ml.data.compliance import PHYSICAL_SAFEGUARDS
        for sg in PHYSICAL_SAFEGUARDS:
            assert sg.category == SafeguardCategory.PHYSICAL

    def test_technical_safeguards_category(self) -> None:
        from ml.data.compliance import TECHNICAL_SAFEGUARDS
        for sg in TECHNICAL_SAFEGUARDS:
            assert sg.category == SafeguardCategory.TECHNICAL

    def test_administrative_safeguards_category(self) -> None:
        from ml.data.compliance import ADMINISTRATIVE_SAFEGUARDS
        for sg in ADMINISTRATIVE_SAFEGUARDS:
            assert sg.category == SafeguardCategory.ADMINISTRATIVE

    def test_all_have_nist_control(self) -> None:
        for sg in ALL_SAFEGUARDS:
            assert sg.nist_control, f"{sg.check_id} missing NIST control"

    def test_all_have_hipaa_reference(self) -> None:
        for sg in ALL_SAFEGUARDS:
            assert sg.hipaa_reference, f"{sg.check_id} missing HIPAA reference"


# ═══════════════════════════════════════════════════════════════════════════
# Compliance Gate Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComplianceGate:
    """Full compliance gate evaluation tests."""

    def test_empty_gate_fails(self) -> None:
        gate = ComplianceGate()
        result, blockers = gate.evaluate()
        assert result == ComplianceGateResult.FAILED
        assert len(blockers) == 4

    def test_full_gate_passes(self) -> None:
        identity = _make_valid_identity()
        irb = _make_approved_irb(identity)
        form = _make_submitted_cdc_form()
        att, _ = _make_signed_attestation(identity)

        gate = create_compliance_gate(
            researcher=identity,
            irb=irb,
            cdc_form=form,
            attestation=att,
        )
        result, blockers = gate.evaluate()
        assert result == ComplianceGateResult.PASSED
        assert blockers == []

    def test_missing_irb_fails(self) -> None:
        identity = _make_valid_identity()
        form = _make_submitted_cdc_form()
        att, _ = _make_signed_attestation(identity)

        gate = ComplianceGate(
            researcher_identity=identity,
            cdc_form=form,
            security_attestation=att,
        )
        result, blockers = gate.evaluate()
        assert result == ComplianceGateResult.FAILED
        assert any("IRB" in b for b in blockers)

    def test_draft_irb_fails(self) -> None:
        identity = _make_valid_identity()
        irb = IRBApplication(pi_identity=identity)  # DRAFT
        form = _make_submitted_cdc_form()
        att, _ = _make_signed_attestation(identity)

        gate = ComplianceGate(
            researcher_identity=identity,
            irb_application=irb,
            cdc_form=form,
            security_attestation=att,
        )
        result, blockers = gate.evaluate()
        assert result == ComplianceGateResult.FAILED

    def test_cdc_form_not_submitted_fails(self) -> None:
        identity = _make_valid_identity()
        irb = _make_approved_irb(identity)
        form = CDCDataRequestForm()  # NOT_STARTED
        att, _ = _make_signed_attestation(identity)

        gate = ComplianceGate(
            researcher_identity=identity,
            irb_application=irb,
            cdc_form=form,
            security_attestation=att,
        )
        result, blockers = gate.evaluate()
        assert result == ComplianceGateResult.FAILED
        assert any("CDC form" in b.lower() or "cdc" in b.lower() for b in blockers)

    def test_unsigned_attestation_fails(self) -> None:
        identity = _make_valid_identity()
        irb = _make_approved_irb(identity)
        form = _make_submitted_cdc_form()
        att = SecurityAttestation()  # PENDING

        gate = ComplianceGate(
            researcher_identity=identity,
            irb_application=irb,
            cdc_form=form,
            security_attestation=att,
        )
        result, blockers = gate.evaluate()
        assert result == ComplianceGateResult.FAILED

    def test_generate_report(self) -> None:
        identity = _make_valid_identity()
        irb = _make_approved_irb(identity)
        form = _make_submitted_cdc_form()
        att, _ = _make_signed_attestation(identity)

        gate = create_compliance_gate(
            researcher=identity, irb=irb, cdc_form=form, attestation=att,
        )
        report = gate.generate_report()
        assert report["gate_result"] == "passed"
        assert "stages" in report
        assert "researcher_identity" in report["stages"]
        assert "irb_application" in report["stages"]
        assert "cdc_form" in report["stages"]
        assert "security_attestation" in report["stages"]

    def test_generate_report_empty_gate(self) -> None:
        gate = ComplianceGate()
        report = gate.generate_report()
        assert report["gate_result"] == "failed"
        assert report["stages"]["researcher_identity"]["status"] == "not_provided"


# ═══════════════════════════════════════════════════════════════════════════
# Value Object Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValueObjects:
    """NamedTuple and enum tests."""

    def test_irb_status_values(self) -> None:
        assert IRBStatus.DRAFT.value == "draft"
        assert IRBStatus.APPROVED.value == "approved"
        assert len(IRBStatus) == 10

    def test_cdc_form_status_values(self) -> None:
        assert CDCFormStatus.NOT_STARTED.value == "not_started"
        assert len(CDCFormStatus) == 9

    def test_attestation_status_values(self) -> None:
        assert AttestationStatus.PENDING.value == "pending"
        assert len(AttestationStatus) == 5

    def test_compliance_gate_result_values(self) -> None:
        assert ComplianceGateResult.PASSED.value == "passed"
        assert len(ComplianceGateResult) == 4

    def test_citi_completion_namedtuple(self) -> None:
        comp = CITICompletion(
            module_id="test",
            module_name="Test Module",
            completion_date=datetime.now(tz=timezone.utc),
            expiration_date=datetime.now(tz=timezone.utc),
            score=95.0,
            certificate_id="CERT-001",
        )
        assert comp.score == 95.0

    def test_irb_transition_namedtuple(self) -> None:
        t = IRBTransition(
            from_status=IRBStatus.DRAFT,
            to_status=IRBStatus.SUBMITTED,
            timestamp=datetime.now(tz=timezone.utc),
            actor="PI",
            notes="",
        )
        assert t.from_status == IRBStatus.DRAFT

    def test_validation_issue_namedtuple(self) -> None:
        vi = ValidationIssue(
            field_name="test",
            severity="error",
            message="test error",
            suggestion=None,
        )
        assert vi.suggestion is None
