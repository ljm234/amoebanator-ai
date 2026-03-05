"""
Regulatory Compliance and Institutional Review Board Tracking.

Manages IRB application lifecycle, CDC data request form validation,
security attestation workflows, and researcher identity verification
for HIPAA/FERPA/Common Rule compliant data acquisition pipelines.

Architecture
------------
The compliance module enforces a gated workflow:

    ┌──────────────────────────────────────────────────────────────┐
    │                 COMPLIANCE GATE PIPELINE                     │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  STAGE 1 ─ Researcher Identity Verification                  │
    │  ├── ORCID validation (regex + checksum)                    │
    │  ├── Institutional affiliation attestation                  │
    │  └── CITI training completion record                        │
    │                                                              │
    │  STAGE 2 ─ IRB Application Tracking                         │
    │  ├── Protocol document SHA-256 fingerprint                  │
    │  ├── Status state machine (DRAFT → SUBMITTED → APPROVED)    │
    │  └── Expiration monitoring with renewal alerts              │
    │                                                              │
    │  STAGE 3 ─ CDC Data Request (Form 0.1310)                   │
    │  ├── Field-level validation against schema                  │
    │  ├── Cross-field consistency checks                         │
    │  └── Submission receipt tracking                            │
    │                                                              │
    │  STAGE 4 ─ Security Attestation                             │
    │  ├── Physical safeguards checklist                          │
    │  ├── Technical safeguards (encryption, access control)      │
    │  ├── Administrative safeguards (training, policies)         │
    │  └── Digital signature with timestamp                       │
    │                                                              │
    │  GATE ─ All four stages must reach APPROVED/VERIFIED        │
    │  before data acquisition operations are permitted.          │
    └──────────────────────────────────────────────────────────────┘

Standards Implemented
---------------------
- 45 CFR 46 (Common Rule, 2025 revision)
- HIPAA Privacy Rule (45 CFR 164)
- HIPAA Security Rule (45 CFR 164.308-312)
- CDC/ATSDR Policy on Releasing and Sharing Data (2024)
- NIH Data Management and Sharing Policy (effective 2023-01-25)
- NIST SP 800-53 Rev. 5 Security Controls
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
    NamedTuple,
    Sequence,
    TypeAlias,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PathLike: TypeAlias = str | Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORCID_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(\d{4})-(\d{4})-(\d{4})-(\d{3}[\dX])$"
)
CDC_FORM_VERSION: Final[str] = "0.1310-2024"
IRB_MAX_APPROVAL_YEARS: Final[int] = 3
CITI_REQUIRED_MODULES: Final[frozenset[str]] = frozenset({
    "human_subjects_research",
    "hipaa_privacy",
    "responsible_conduct",
    "informed_consent",
    "vulnerable_populations",
    "data_security",
})


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class IRBStatus(Enum):
    """State machine for IRB application lifecycle."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    REVISIONS_REQUESTED = "revisions_requested"
    APPROVED = "approved"
    CONDITIONALLY_APPROVED = "conditionally_approved"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    EXPIRED = "expired"
    RENEWAL_PENDING = "renewal_pending"


class CDCFormStatus(Enum):
    """Status of CDC data request form."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    VALIDATION_FAILED = "validation_failed"
    READY_TO_SUBMIT = "ready_to_submit"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    DENIED = "denied"
    DATA_TRANSFERRED = "data_transferred"


class AttestationStatus(Enum):
    """Status of security attestation."""

    PENDING = "pending"
    SIGNED = "signed"
    VERIFIED = "verified"
    EXPIRED = "expired"
    REVOKED = "revoked"


class SafeguardCategory(Enum):
    """HIPAA safeguard categories."""

    PHYSICAL = auto()
    TECHNICAL = auto()
    ADMINISTRATIVE = auto()


class ComplianceGateResult(Enum):
    """Overall compliance gate evaluation."""

    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    EXPIRED = "expired"


# ═══════════════════════════════════════════════════════════════════════════
# Value Objects
# ═══════════════════════════════════════════════════════════════════════════

class CITICompletion(NamedTuple):
    """Record of CITI programme module completion."""

    module_id: str
    module_name: str
    completion_date: datetime
    expiration_date: datetime
    score: float
    certificate_id: str


class IRBTransition(NamedTuple):
    """Recorded state transition in IRB lifecycle."""

    from_status: IRBStatus
    to_status: IRBStatus
    timestamp: datetime
    actor: str
    notes: str


class ValidationIssue(NamedTuple):
    """Single validation problem in a form or document."""

    field_name: str
    severity: Literal["error", "warning", "info"]
    message: str
    suggestion: str | None


# ═══════════════════════════════════════════════════════════════════════════
# Researcher Identity
# ═══════════════════════════════════════════════════════════════════════════

def _validate_orcid_checksum(orcid_digits: str) -> bool:
    """ISO 7064 Mod 11-2 checksum used by ORCID."""
    total = 0
    for char in orcid_digits[:-1]:
        if char == "-":
            continue
        total = (total + int(char)) * 2
    remainder = total % 11
    check = (12 - remainder) % 11
    expected = "X" if check == 10 else str(check)
    return orcid_digits[-1] == expected


@dataclass(frozen=True, slots=True)
class ResearcherIdentity:
    """Verified researcher identity for data access authorisation.

    Attributes
    ----------
    orcid : str
        ORCID iD in format 0009-0000-7851-7139.
    full_name : str
        Luis Jordan Montenegro Calla
    email : str
        lmontenegrocalla@mail.weber.edu
        ljm234@ensign.edu
    institution : str
        Ensign College / Weber State University. #I am only an student, not faculty or staff, so I am not an "independent researcher" in the traditional sense. --- IGNORE ---, 
    role : str
        Independent research, principal investigator.
    citi_completions : tuple[citicompletion, ...]
        completed citi training modules.
    verified_at : datetime | none
        timestamp of identity verification.
    """

    orcid: str
    full_name: str
    email: str
    institution: str
    department: str
    role: str = "Independent Researcher"
    citi_completions: tuple[CITICompletion, ...] = ()
    verified_at: datetime | None = None


class ResearcherValidator:
    """Validates researcher identity and training completeness.

    Ensures ORCID format + checksum validity, institutional email
    domain, and CITI programme training currency.
    """

    __slots__ = ("_required_modules", "_max_citi_age_days")

    def __init__(
        self,
        required_modules: frozenset[str] | None = None,
        max_citi_age_days: int = 1095,
    ) -> None:
        self._required_modules = required_modules or CITI_REQUIRED_MODULES
        self._max_citi_age_days = max_citi_age_days

    def validate(self, identity: ResearcherIdentity) -> list[ValidationIssue]:
        """Run all identity validation checks.

        Returns
        -------
        list[ValidationIssue]
            Empty list when all checks pass.
        """
        issues: list[ValidationIssue] = []
        issues.extend(self._validate_orcid(identity.orcid))
        issues.extend(self._validate_email(identity.email))
        issues.extend(self._validate_citi(identity.citi_completions))
        return issues

    # -- Private helpers ---------------------------------------------------

    def _validate_orcid(self, orcid: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not ORCID_REGEX.match(orcid):
            issues.append(
                ValidationIssue(
                    field_name="orcid",
                    severity="error",
                    message=f"ORCID '{orcid}' does not match XXXX-XXXX-XXXX-XXXX",
                    suggestion="Register at https://orcid.org/register",
                )
            )
            return issues

        digits_only = orcid.replace("-", "")
        if not _validate_orcid_checksum(digits_only):
            issues.append(
                ValidationIssue(
                    field_name="orcid",
                    severity="error",
                    message="ORCID checksum (ISO 7064 Mod 11-2) invalid",
                    suggestion="Verify the identifier at https://orcid.org/",
                )
            )
        return issues

    def _validate_email(self, email: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        pattern = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
        if not pattern.match(email):
            issues.append(
                ValidationIssue(
                    field_name="email",
                    severity="error",
                    message="Email address format invalid",
                    suggestion=None,
                )
            )
            return issues

        domain = email.split("@", 1)[1].lower()
        educational_tlds = (".edu", ".ac.", ".edu.", ".gov")
        if not any(domain.endswith(tld) or tld in domain for tld in educational_tlds):
            issues.append(
                ValidationIssue(
                    field_name="email",
                    severity="warning",
                    message="Non-institutional email domain detected",
                    suggestion="Use .edu or .gov institutional email for CDC requests",
                )
            )
        return issues

    def _validate_citi(
        self, completions: tuple[CITICompletion, ...]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        now = datetime.now(tz=timezone.utc)

        completed_ids = set()
        for comp in completions:
            completed_ids.add(comp.module_id)
            exp = comp.expiration_date
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp < now:
                issues.append(
                    ValidationIssue(
                        field_name="citi_completions",
                        severity="error",
                        message=f"CITI module '{comp.module_name}' expired {exp.date()}",
                        suggestion="Renew certification at citiprogram.org",
                    )
                )
            if comp.score < 80.0:
                issues.append(
                    ValidationIssue(
                        field_name="citi_completions",
                        severity="warning",
                        message=(
                            f"CITI module '{comp.module_name}' score "
                            f"{comp.score:.0f}% below 80% threshold"
                        ),
                        suggestion="Retake module to meet institutional requirements",
                    )
                )

        missing = self._required_modules - completed_ids
        for mod_id in sorted(missing):
            issues.append(
                ValidationIssue(
                    field_name="citi_completions",
                    severity="error",
                    message=f"Required CITI module not completed: {mod_id}",
                    suggestion="Complete at https://about.citiprogram.org/",
                )
            )
        return issues


# ═══════════════════════════════════════════════════════════════════════════
# IRB Application Tracker
# ═══════════════════════════════════════════════════════════════════════════

# Valid transitions in the IRB state machine
_IRB_TRANSITIONS: Final[dict[IRBStatus, frozenset[IRBStatus]]] = {
    IRBStatus.DRAFT: frozenset({IRBStatus.SUBMITTED}),
    IRBStatus.SUBMITTED: frozenset({
        IRBStatus.UNDER_REVIEW,
        IRBStatus.REVISIONS_REQUESTED,
    }),
    IRBStatus.UNDER_REVIEW: frozenset({
        IRBStatus.APPROVED,
        IRBStatus.CONDITIONALLY_APPROVED,
        IRBStatus.REVISIONS_REQUESTED,
        IRBStatus.TERMINATED,
    }),
    IRBStatus.REVISIONS_REQUESTED: frozenset({IRBStatus.SUBMITTED}),
    IRBStatus.APPROVED: frozenset({
        IRBStatus.RENEWAL_PENDING,
        IRBStatus.SUSPENDED,
        IRBStatus.EXPIRED,
    }),
    IRBStatus.CONDITIONALLY_APPROVED: frozenset({
        IRBStatus.APPROVED,
        IRBStatus.REVISIONS_REQUESTED,
    }),
    IRBStatus.SUSPENDED: frozenset({IRBStatus.APPROVED, IRBStatus.TERMINATED}),
    IRBStatus.RENEWAL_PENDING: frozenset({
        IRBStatus.APPROVED,
        IRBStatus.EXPIRED,
    }),
    IRBStatus.EXPIRED: frozenset({IRBStatus.DRAFT}),
    IRBStatus.TERMINATED: frozenset(),
}


@dataclass
class IRBApplication:
    """Tracks a single IRB application through its lifecycle.

    Attributes
    ----------
    application_id : str
        Unique identifier (auto-generated UUID prefix).
    protocol_title : str
        Title of the research protocol.
    protocol_number : str
        Institutional protocol tracking number.
    pi_identity : ResearcherIdentity
        Lead researcher identity record.
    status : IRBStatus
        Current lifecycle state.
    submitted_date : datetime | None
        Date the application was submitted to the IRB.
    approval_date : datetime | None
        Date of IRB approval.
    expiration_date : datetime | None
        Approval expiration date.
    protocol_hash : str
        SHA-256 fingerprint of the protocol document.
    transitions : list[IRBTransition]
        Recorded state transitions.
    attachments : list[str]
        File paths of supporting documents.
    """

    application_id: str = field(
        default_factory=lambda: f"IRB-{uuid.uuid4().hex[:12].upper()}"
    )
    protocol_title: str = ""
    protocol_number: str = ""
    pi_identity: ResearcherIdentity | None = None
    status: IRBStatus = IRBStatus.DRAFT
    submitted_date: datetime | None = None
    approval_date: datetime | None = None
    expiration_date: datetime | None = None
    protocol_hash: str = ""
    transitions: list[IRBTransition] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)

    # -- Lifecycle operations ----------------------------------------------

    def transition_to(
        self,
        new_status: IRBStatus,
        actor: str = "",
        notes: str = "",
    ) -> None:
        """Advance the application to a new status.

        Parameters
        ----------
        new_status : IRBStatus
            Target state; must be a valid transition.
        actor : str
            Person or system initiating the transition.
        notes : str
            Free-text notes for the audit record.

        Raises
        ------
        ValueError
            If the requested transition is not permitted.
        """
        allowed = _IRB_TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            msg = (
                f"Invalid transition: {self.status.value} → {new_status.value}. "
                f"Allowed targets: {[s.value for s in allowed]}"
            )
            raise ValueError(msg)

        transition = IRBTransition(
            from_status=self.status,
            to_status=new_status,
            timestamp=datetime.now(tz=timezone.utc),
            actor=actor,
            notes=notes,
        )
        self.transitions.append(transition)
        self.status = new_status

        if new_status == IRBStatus.SUBMITTED:
            self.submitted_date = datetime.now(tz=timezone.utc)
        elif new_status == IRBStatus.APPROVED:
            self.approval_date = datetime.now(tz=timezone.utc)
            self.expiration_date = self.approval_date + timedelta(
                days=365 * IRB_MAX_APPROVAL_YEARS
            )

        logger.info(
            "IRB %s transitioned: %s → %s",
            self.application_id,
            transition.from_status.value,
            transition.to_status.value,
        )

    def compute_protocol_hash(self, protocol_path: Path) -> str:
        """Compute SHA-256 fingerprint of the protocol document.

        Parameters
        ----------
        protocol_path : Path
            Path to the protocol PDF or document.

        Returns
        -------
        str
            Hexadecimal SHA-256 digest.
        """
        sha = hashlib.sha256()
        with protocol_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        self.protocol_hash = sha.hexdigest()
        return self.protocol_hash

    def is_current(self) -> bool:
        """Check whether approval is current and non-expired."""
        if self.status != IRBStatus.APPROVED:
            return False
        if self.expiration_date is None:
            return False
        now = datetime.now(tz=timezone.utc)
        exp = self.expiration_date
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return now < exp

    def days_until_expiration(self) -> int | None:
        """Calculate days remaining until approval expires."""
        if self.expiration_date is None:
            return None
        now = datetime.now(tz=timezone.utc)
        exp = self.expiration_date
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        delta = exp - now
        return max(0, delta.days)

    def to_dict(self) -> dict[str, Any]:
        """Serialise application state for persistence or reporting."""
        return {
            "application_id": self.application_id,
            "protocol_title": self.protocol_title,
            "protocol_number": self.protocol_number,
            "status": self.status.value,
            "submitted_date": (
                self.submitted_date.isoformat() if self.submitted_date else None
            ),
            "approval_date": (
                self.approval_date.isoformat() if self.approval_date else None
            ),
            "expiration_date": (
                self.expiration_date.isoformat() if self.expiration_date else None
            ),
            "protocol_hash": self.protocol_hash,
            "transitions_count": len(self.transitions),
            "attachments": self.attachments,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CDC Form 0.1310 Validator
# ═══════════════════════════════════════════════════════════════════════════

_CDC_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({
    "investigator_name",
    "institution",
    "project_title",
    "data_description",
    "justification",
    "irb_protocol_number",
    "irb_approval_date",
    "data_security_plan",
    "destruction_plan",
    "signature_date",
})

_CDC_OPTIONAL_FIELDS: Final[frozenset[str]] = frozenset({
    "co_investigators",
    "funding_source",
    "grant_number",
    "anticipated_publications",
    "data_linkage_plan",
    "geographic_scope",
    "time_period_requested",
    "specific_variables",
    "sample_size_justification",
})


@dataclass
class CDCDataRequestForm:
    """Validates and tracks CDC/ATSDR data request form (0.1310).

    The form schema encodes every required and optional field from
    the 2024 revision of the CDC Data Request Form, with type-level
    validation, cross-field consistency checks, and submission receipt
    tracking.

    Attributes
    ----------
    form_id : str
        Tracking identifier.
    fields : dict[str, Any]
        Form field values.
    status : CDCFormStatus
        Current form status.
    validation_issues : list[ValidationIssue]
        Outstanding validation problems.
    submission_receipt : str | None
        CDC-assigned receipt identifier.
    submitted_at : datetime | None
        Submission timestamp.
    """

    form_id: str = field(
        default_factory=lambda: f"CDC-{uuid.uuid4().hex[:10].upper()}"
    )
    fields: dict[str, Any] = field(default_factory=dict)
    status: CDCFormStatus = CDCFormStatus.NOT_STARTED
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    submission_receipt: str | None = None
    submitted_at: datetime | None = None

    def set_field(self, name: str, value: Any) -> None:
        """Set a form field value, updating status to IN_PROGRESS."""
        self.fields[name] = value
        if self.status == CDCFormStatus.NOT_STARTED:
            self.status = CDCFormStatus.IN_PROGRESS

    def validate(self) -> list[ValidationIssue]:
        """Run comprehensive validation on the form.

        Returns
        -------
        list[ValidationIssue]
            All identified validation issues.
        """
        self.validation_issues = []
        self._check_required_fields()
        self._check_irb_dates()
        self._check_security_plan()
        self._check_destruction_plan()

        if any(i.severity == "error" for i in self.validation_issues):
            self.status = CDCFormStatus.VALIDATION_FAILED
        elif not self.validation_issues:
            self.status = CDCFormStatus.READY_TO_SUBMIT
        return self.validation_issues

    def mark_submitted(self, receipt_id: str) -> None:
        """Record form submission with CDC receipt."""
        self.status = CDCFormStatus.SUBMITTED
        self.submission_receipt = receipt_id
        self.submitted_at = datetime.now(tz=timezone.utc)

    # -- Validation helpers ------------------------------------------------

    def _check_required_fields(self) -> None:
        for field_name in _CDC_REQUIRED_FIELDS:
            val = self.fields.get(field_name)
            if val is None or (isinstance(val, str) and not val.strip()):
                self.validation_issues.append(
                    ValidationIssue(
                        field_name=field_name,
                        severity="error",
                        message=f"Required field '{field_name}' is empty",
                        suggestion="Complete this field before submission",
                    )
                )

    def _check_irb_dates(self) -> None:
        approval_str = self.fields.get("irb_approval_date")
        if approval_str and isinstance(approval_str, str):
            try:
                approval_dt = datetime.fromisoformat(approval_str)
                if approval_dt.tzinfo is None:
                    approval_dt = approval_dt.replace(tzinfo=timezone.utc)
                now = datetime.now(tz=timezone.utc)
                age = now - approval_dt
                if age.days > 365:
                    self.validation_issues.append(
                        ValidationIssue(
                            field_name="irb_approval_date",
                            severity="warning",
                            message="IRB approval is older than 1 year",
                            suggestion="Confirm approval is still current",
                        )
                    )
            except ValueError:
                self.validation_issues.append(
                    ValidationIssue(
                        field_name="irb_approval_date",
                        severity="error",
                        message="Invalid date format; use ISO 8601 (YYYY-MM-DD)",
                        suggestion="Example: 2026-02-15",
                    )
                )

    def _check_security_plan(self) -> None:
        plan = self.fields.get("data_security_plan", "")
        if isinstance(plan, str) and len(plan.split()) < 50:
            self.validation_issues.append(
                ValidationIssue(
                    field_name="data_security_plan",
                    severity="warning",
                    message="Data security plan appears insufficient (< 50 words)",
                    suggestion=(
                        "Include encryption method, access controls, "
                        "physical safeguards, and training procedures"
                    ),
                )
            )

    def _check_destruction_plan(self) -> None:
        plan = self.fields.get("destruction_plan", "")
        if isinstance(plan, str):
            keywords = {"nist", "800-88", "sanitization", "destruction", "shred"}
            lower_plan = plan.lower()
            if not any(kw in lower_plan for kw in keywords):
                self.validation_issues.append(
                    ValidationIssue(
                        field_name="destruction_plan",
                        severity="info",
                        message="Destruction plan should reference NIST SP 800-88",
                        suggestion=(
                            "Reference NIST SP 800-88 Rev. 1 media sanitization "
                            "guidelines for data destruction procedures"
                        ),
                    )
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialise form state."""
        return {
            "form_id": self.form_id,
            "form_version": CDC_FORM_VERSION,
            "status": self.status.value,
            "fields": dict(self.fields),
            "validation_issues": [
                {
                    "field": i.field_name,
                    "severity": i.severity,
                    "message": i.message,
                }
                for i in self.validation_issues
            ],
            "submission_receipt": self.submission_receipt,
            "submitted_at": (
                self.submitted_at.isoformat() if self.submitted_at else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Security Attestation
# ═══════════════════════════════════════════════════════════════════════════

class SafeguardCheck(NamedTuple):
    """Single safeguard compliance check."""

    check_id: str
    category: SafeguardCategory
    requirement: str
    description: str
    nist_control: str
    hipaa_reference: str


PHYSICAL_SAFEGUARDS: Final[tuple[SafeguardCheck, ...]] = (
    SafeguardCheck(
        check_id="PHY-001",
        category=SafeguardCategory.PHYSICAL,
        requirement="Facility access controls",
        description="Badge or biometric access to data processing areas",
        nist_control="PE-2",
        hipaa_reference="164.310(a)(1)",
    ),
    SafeguardCheck(
        check_id="PHY-002",
        category=SafeguardCategory.PHYSICAL,
        requirement="Workstation security",
        description="Screen locks, cable locks, clean-desk policy enforcement",
        nist_control="PE-5",
        hipaa_reference="164.310(c)",
    ),
    SafeguardCheck(
        check_id="PHY-003",
        category=SafeguardCategory.PHYSICAL,
        requirement="Device and media controls",
        description="Encrypted removable media with tracked disposal",
        nist_control="MP-4",
        hipaa_reference="164.310(d)(1)",
    ),
)

TECHNICAL_SAFEGUARDS: Final[tuple[SafeguardCheck, ...]] = (
    SafeguardCheck(
        check_id="TEC-001",
        category=SafeguardCategory.TECHNICAL,
        requirement="Encryption at rest",
        description="AES-256-GCM or ChaCha20-Poly1305 for stored data",
        nist_control="SC-28",
        hipaa_reference="164.312(a)(2)(iv)",
    ),
    SafeguardCheck(
        check_id="TEC-002",
        category=SafeguardCategory.TECHNICAL,
        requirement="Encryption in transit",
        description="TLS 1.3 or SFTP with Ed25519/ECDSA keys",
        nist_control="SC-8",
        hipaa_reference="164.312(e)(1)",
    ),
    SafeguardCheck(
        check_id="TEC-003",
        category=SafeguardCategory.TECHNICAL,
        requirement="Access control",
        description="Role-based access with MFA, least-privilege enforcement",
        nist_control="AC-2",
        hipaa_reference="164.312(a)(1)",
    ),
    SafeguardCheck(
        check_id="TEC-004",
        category=SafeguardCategory.TECHNICAL,
        requirement="Audit logging",
        description="Tamper-evident access logs with 6-year retention",
        nist_control="AU-2",
        hipaa_reference="164.312(b)",
    ),
    SafeguardCheck(
        check_id="TEC-005",
        category=SafeguardCategory.TECHNICAL,
        requirement="Integrity controls",
        description="Cryptographic checksums on all data files at rest",
        nist_control="SI-7",
        hipaa_reference="164.312(c)(1)",
    ),
)

ADMINISTRATIVE_SAFEGUARDS: Final[tuple[SafeguardCheck, ...]] = (
    SafeguardCheck(
        check_id="ADM-001",
        category=SafeguardCategory.ADMINISTRATIVE,
        requirement="Security management process",
        description="Risk analysis with documented remediation plans",
        nist_control="PM-9",
        hipaa_reference="164.308(a)(1)(i)",
    ),
    SafeguardCheck(
        check_id="ADM-002",
        category=SafeguardCategory.ADMINISTRATIVE,
        requirement="Workforce training",
        description="Annual HIPAA security awareness training",
        nist_control="AT-2",
        hipaa_reference="164.308(a)(5)(i)",
    ),
    SafeguardCheck(
        check_id="ADM-003",
        category=SafeguardCategory.ADMINISTRATIVE,
        requirement="Incident response",
        description="Documented breach notification procedures per 45 CFR 164.404",
        nist_control="IR-1",
        hipaa_reference="164.308(a)(6)(i)",
    ),
    SafeguardCheck(
        check_id="ADM-004",
        category=SafeguardCategory.ADMINISTRATIVE,
        requirement="Business associate agreements",
        description="BAA with all third-party data processors",
        nist_control="PS-7",
        hipaa_reference="164.308(b)(1)",
    ),
)

ALL_SAFEGUARDS: Final[tuple[SafeguardCheck, ...]] = (
    PHYSICAL_SAFEGUARDS + TECHNICAL_SAFEGUARDS + ADMINISTRATIVE_SAFEGUARDS
)


@dataclass
class SecurityAttestation:
    """Digital security compliance attestation.

    Tracks acknowledgement of each safeguard check by the attesting
    researcher, records a digital signature (HMAC-SHA256 over the
    serialised state), and enforces annual renewal.

    Attributes
    ----------
    attestation_id : str
        Unique identifier.
    researcher : ResearcherIdentity | None
        Attesting individual.
    safeguard_responses : dict[str, bool]
        Mapping from check_id to compliance status.
    status : AttestationStatus
        Current attestation state.
    signed_at : datetime | None
        Timestamp of digital signature.
    signature : str
        HMAC-SHA256 signature over serialised state.
    expires_at : datetime | None
        Attestation expiration (1 year from signing).
    """

    attestation_id: str = field(
        default_factory=lambda: f"ATT-{uuid.uuid4().hex[:10].upper()}"
    )
    researcher: ResearcherIdentity | None = None
    safeguard_responses: dict[str, bool] = field(default_factory=dict)
    status: AttestationStatus = AttestationStatus.PENDING
    signed_at: datetime | None = None
    signature: str = ""
    expires_at: datetime | None = None

    def respond(self, check_id: str, compliant: bool) -> None:
        """Record compliance response for a safeguard check.

        Parameters
        ----------
        check_id : str
            Safeguard identifier (e.g., "TEC-001").
        compliant : bool
            Whether the safeguard is implemented.
        """
        valid_ids = {s.check_id for s in ALL_SAFEGUARDS}
        if check_id not in valid_ids:
            msg = f"Unknown safeguard check: {check_id}"
            raise ValueError(msg)
        self.safeguard_responses[check_id] = compliant

    def all_checks_completed(self) -> bool:
        """Return True if every safeguard has been responded to."""
        required_ids = {s.check_id for s in ALL_SAFEGUARDS}
        return required_ids.issubset(self.safeguard_responses.keys())

    def all_compliant(self) -> bool:
        """Return True if every responded safeguard is compliant."""
        return (
            self.all_checks_completed()
            and all(self.safeguard_responses.values())
        )

    def sign(self, signing_key: bytes) -> str:
        """Apply HMAC-SHA256 digital signature and lock the attestation.

        Parameters
        ----------
        signing_key : bytes
            Secret key for HMAC computation.

        Returns
        -------
        str
            Hexadecimal HMAC-SHA256 signature.

        Raises
        ------
        ValueError
            If not all checks are completed, or if non-compliant.
        """
        if not self.all_checks_completed():
            msg = "Cannot sign: not all safeguard checks are completed"
            raise ValueError(msg)
        if not self.all_compliant():
            msg = "Cannot sign: not all safeguards are compliant"
            raise ValueError(msg)

        self.signed_at = datetime.now(tz=timezone.utc)
        self.expires_at = self.signed_at + timedelta(days=365)
        self.status = AttestationStatus.SIGNED

        payload = json.dumps(
            {
                "id": self.attestation_id,
                "responses": self.safeguard_responses,
                "signed_at": self.signed_at.isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")

        self.signature = hmac.new(
            signing_key, payload, hashlib.sha256
        ).hexdigest()

        return self.signature

    def verify_signature(self, signing_key: bytes) -> bool:
        """Verify the attestation signature.

        Parameters
        ----------
        signing_key : bytes
            Secret key originally used for signing.

        Returns
        -------
        bool
            True if signature is valid.
        """
        if not self.signed_at:
            return False

        payload = json.dumps(
            {
                "id": self.attestation_id,
                "responses": self.safeguard_responses,
                "signed_at": self.signed_at.isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")

        expected = hmac.new(signing_key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def is_current(self) -> bool:
        """Check whether attestation is signed and non-expired."""
        if self.status not in (AttestationStatus.SIGNED, AttestationStatus.VERIFIED):
            return False
        if self.expires_at is None:
            return False
        now = datetime.now(tz=timezone.utc)
        exp = self.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return now < exp

    def non_compliant_checks(self) -> list[SafeguardCheck]:
        """Return safeguard checks that were marked non-compliant."""
        result: list[SafeguardCheck] = []
        for sg in ALL_SAFEGUARDS:
            if not self.safeguard_responses.get(sg.check_id, False):
                result.append(sg)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialise attestation state."""
        return {
            "attestation_id": self.attestation_id,
            "status": self.status.value,
            "safeguard_responses": dict(self.safeguard_responses),
            "total_checks": len(ALL_SAFEGUARDS),
            "completed_checks": len(self.safeguard_responses),
            "compliant_checks": sum(
                1 for v in self.safeguard_responses.values() if v
            ),
            "signed_at": (
                self.signed_at.isoformat() if self.signed_at else None
            ),
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at else None
            ),
            "signature_present": bool(self.signature),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Compliance Gate
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComplianceGate:
    """Evaluates whether all prerequisites are met for data acquisition.

    Aggregates the four compliance stages (researcher identity, IRB
    approval, CDC form submission, and security attestation) into a
    single pass/fail gate decision.

    Attributes
    ----------
    researcher_identity : ResearcherIdentity | None
        Verified researcher identity.
    irb_application : IRBApplication | None
        Tracked IRB application.
    cdc_form : CDCDataRequestForm | None
        Validated CDC data request form.
    security_attestation : SecurityAttestation | None
        Signed security attestation.
    """

    researcher_identity: ResearcherIdentity | None = None
    irb_application: IRBApplication | None = None
    cdc_form: CDCDataRequestForm | None = None
    security_attestation: SecurityAttestation | None = None

    def evaluate(self) -> tuple[ComplianceGateResult, list[str]]:
        """Run gate evaluation across all four stages.

        Returns
        -------
        tuple[ComplianceGateResult, list[str]]
            Gate result and list of blocking issues.
        """
        blockers: list[str] = []

        # Stage 1: researcher identity
        if self.researcher_identity is None:
            blockers.append("Researcher identity not provided")
        else:
            validator = ResearcherValidator()
            issues = validator.validate(self.researcher_identity)
            errors = [i for i in issues if i.severity == "error"]
            if errors:
                for err in errors:
                    blockers.append(f"Identity: {err.message}")

        # Stage 2: IRB approval
        if self.irb_application is None:
            blockers.append("IRB application not created")
        elif not self.irb_application.is_current():
            blockers.append(
                f"IRB status is '{self.irb_application.status.value}', "
                f"must be 'approved' and non-expired"
            )

        # Stage 3: CDC form
        if self.cdc_form is None:
            blockers.append("CDC data request form not created")
        elif self.cdc_form.status not in (
            CDCFormStatus.SUBMITTED,
            CDCFormStatus.APPROVED,
            CDCFormStatus.DATA_TRANSFERRED,
        ):
            blockers.append(
                f"CDC form status is '{self.cdc_form.status.value}', "
                f"must be submitted or approved"
            )

        # Stage 4: security attestation
        if self.security_attestation is None:
            blockers.append("Security attestation not created")
        elif not self.security_attestation.is_current():
            blockers.append(
                f"Security attestation status is "
                f"'{self.security_attestation.status.value}', "
                f"must be signed/verified and non-expired"
            )

        if blockers:
            return ComplianceGateResult.FAILED, blockers
        return ComplianceGateResult.PASSED, []

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive compliance report.

        Returns
        -------
        dict[str, Any]
            Structured report with status of each stage.
        """
        gate_result, blockers = self.evaluate()

        report: dict[str, Any] = {
            "gate_result": gate_result.value,
            "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
            "blockers": blockers,
            "stages": {},
        }

        # Stage 1
        if self.researcher_identity:
            validator = ResearcherValidator()
            issues = validator.validate(self.researcher_identity)
            report["stages"]["researcher_identity"] = {
                "orcid": self.researcher_identity.orcid,
                "institution": self.researcher_identity.institution,
                "validation_issues": len(issues),
                "citi_modules_completed": len(
                    self.researcher_identity.citi_completions
                ),
                "status": "verified" if not issues else "issues_found",
            }
        else:
            report["stages"]["researcher_identity"] = {"status": "not_provided"}

        # Stage 2
        if self.irb_application:
            report["stages"]["irb_application"] = self.irb_application.to_dict()
        else:
            report["stages"]["irb_application"] = {"status": "not_created"}

        # Stage 3
        if self.cdc_form:
            report["stages"]["cdc_form"] = self.cdc_form.to_dict()
        else:
            report["stages"]["cdc_form"] = {"status": "not_created"}

        # Stage 4
        if self.security_attestation:
            report["stages"]["security_attestation"] = (
                self.security_attestation.to_dict()
            )
        else:
            report["stages"]["security_attestation"] = {"status": "not_created"}

        return report


# ═══════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════

def create_researcher_identity(
    orcid: str,
    full_name: str,
    email: str,
    institution: str,
    department: str,
    role: str = "Independent Researcher",
    citi_completions: Sequence[CITICompletion] = (),
) -> ResearcherIdentity:
    """Create and optionally validate a researcher identity.

    Parameters
    ----------
    orcid : str
        ORCID iD (XXXX-XXXX-XXXX-XXXX).
    full_name : str
        Legal name.
    email : str
        Institutional email.
    institution : str
        Affiliated institution.
    department : str
        Department or division.
    role : str
        Job title (defaults to "Independent Researcher").
    citi_completions : Sequence[CITICompletion]
        Completed CITI modules.

    Returns
    -------
    ResearcherIdentity
        Constructed identity record.
    """
    return ResearcherIdentity(
        orcid=orcid,
        full_name=full_name,
        email=email,
        institution=institution,
        department=department,
        role=role,
        citi_completions=tuple(citi_completions),
        verified_at=datetime.now(tz=timezone.utc),
    )


def create_irb_application(
    protocol_title: str,
    protocol_number: str,
    pi_identity: ResearcherIdentity,
) -> IRBApplication:
    """Create a new IRB application in DRAFT state.

    Parameters
    ----------
    protocol_title : str
        Title of the research protocol.
    protocol_number : str
        Institutional protocol tracking number.
    pi_identity : ResearcherIdentity
        Lead researcher identity.

    Returns
    -------
    IRBApplication
        New application in DRAFT state.
    """
    return IRBApplication(
        protocol_title=protocol_title,
        protocol_number=protocol_number,
        pi_identity=pi_identity,
    )


def create_compliance_gate(
    researcher: ResearcherIdentity | None = None,
    irb: IRBApplication | None = None,
    cdc_form: CDCDataRequestForm | None = None,
    attestation: SecurityAttestation | None = None,
) -> ComplianceGate:
    """Assemble a compliance gate from its constituent stages.

    Parameters
    ----------
    researcher : ResearcherIdentity | None
        Verified researcher identity.
    irb : IRBApplication | None
        IRB application tracker.
    cdc_form : CDCDataRequestForm | None
        CDC data request form.
    attestation : SecurityAttestation | None
        Security attestation.

    Returns
    -------
    ComplianceGate
        Assembled gate ready for evaluation.
    """
    return ComplianceGate(
        researcher_identity=researcher,
        irb_application=irb,
        cdc_form=cdc_form,
        security_attestation=attestation,
    )
