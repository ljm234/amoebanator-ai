"""Phase 1.8: Expert Annotation Protocol Implementation.

This module implements structured annotation protocols for expert pathologists
and microscopists. Includes calibration sessions, annotation guidelines,
disagreement resolution workflows, and gold standard curation.

Follows FDA/CLIA guidance for clinical diagnostic validation with
documented training and proficiency requirements.

Architecture Overview
---------------------

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                   EXPERT ANNOTATION PROTOCOL ARCHITECTURE                   │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                      EXPERT MANAGEMENT LAYER                         │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Expert      │───▶│  Credential  │───▶│  Proficiency │         │   │
    │  │   │  Registry    │    │  Validator   │    │  Tracker     │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Training    │    │  Certification│   │  Performance │         │   │
    │  │   │  Tracker     │    │  Manager     │    │  Evaluator   │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                    │                                        │
    │                                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                      CALIBRATION LAYER                               │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Calibration │───▶│  Test Case   │───▶│  Score       │         │   │
    │  │   │  Manager     │    │  Generator   │    │  Calculator  │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Recertify   │    │  Difficulty  │    │  Trend       │         │   │
    │  │   │  Scheduler   │    │  Analyzer    │    │  Analyzer    │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                    │                                        │
    │                                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                    DISAGREEMENT RESOLUTION LAYER                     │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Disagreement│───▶│  Arbitration │───▶│  Consensus   │         │   │
    │  │   │  Detector    │    │  Router      │    │  Builder     │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Pattern     │    │  Senior      │    │  Molecular   │         │   │
    │  │   │  Analyzer    │    │  Review      │    │  Confirmation│         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                    │                                        │
    │                                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                     GOLD STANDARD LAYER                              │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Gold        │───▶│  Case        │───▶│  Quality     │         │   │
    │  │   │  Curator     │    │  Validator   │    │  Scorer      │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Version     │    │  Provenance  │    │  Citation    │         │   │
    │  │   │  Manager     │    │  Tracker     │    │  Manager     │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                    │                                        │
    │                                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                      GUIDELINES & COMPLIANCE LAYER                   │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Guideline   │───▶│  Version     │───▶│  Compliance  │         │   │
    │  │   │  Repository  │    │  Control     │    │  Checker     │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Training    │    │  Audit       │    │  Report      │         │   │
    │  │   │  Materials   │    │  Trail       │    │  Generator   │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

Key Components
--------------

ExpertRegistry
    Manages expert annotator profiles and credentials.

CalibrationManager
    Handles calibration sessions and proficiency testing.

DisagreementResolver
    Routes and resolves annotation disagreements.

GoldStandardCurator
    Curates and validates gold standard cases.

GuidelineRepository
    Version-controlled annotation guidelines.

ProficiencyTracker
    Tracks expert performance over time.

CredentialValidator
    Validates expert credentials and certifications.

ConsensusBuilder
    Builds consensus from multiple expert opinions.

References
----------
.. [1] CLIA Regulations for Laboratory Testing (42 CFR 493)
.. [2] FDA Guidance for Premarket Clearance of IVD Software
.. [3] CAP Laboratory Accreditation Checklist
.. [4] WHO Guidelines for Laboratory Quality Management Systems
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Final,
    NamedTuple,
    Protocol,
    Sequence,
    TypeAlias,
)

import numpy as np

logger: Final = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

JSONDict: TypeAlias = dict[str, Any]
NDArrayFloat: TypeAlias = np.ndarray


class ExpertLevel(Enum):
    """Expert qualification levels."""

    TRAINEE = auto()
    CERTIFIED = auto()
    SPECIALIST = auto()
    SENIOR_SPECIALIST = auto()
    REFERENCE_EXPERT = auto()


class CalibrationStatus(Enum):
    """Status of expert calibration."""

    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    PASSED = auto()
    FAILED = auto()
    EXPIRED = auto()


class DisagreementType(Enum):
    """Types of annotation disagreements."""

    SPECIES_MISMATCH = auto()
    STAGE_MISMATCH = auto()
    PRESENCE_ABSENCE = auto()
    LOCALIZATION = auto()
    QUALITY_ASSESSMENT = auto()


class ResolutionMethod(Enum):
    """Methods for resolving disagreements."""

    MAJORITY_VOTE = auto()
    SENIOR_ARBITRATION = auto()
    CONSENSUS_DISCUSSION = auto()
    REFERENCE_EXPERT = auto()
    MOLECULAR_CONFIRMATION = auto()


class AnnotationConfidence(Enum):
    """Confidence level in annotation."""

    CERTAIN = auto()
    PROBABLE = auto()
    POSSIBLE = auto()
    UNCERTAIN = auto()


class ExpertProfile(NamedTuple):
    """Expert annotator profile."""

    expert_id: str
    name: str
    credentials: tuple[str, ...]
    institution: str
    level: ExpertLevel
    specializations: tuple[str, ...]
    years_experience: int
    calibration_status: CalibrationStatus
    last_calibration: datetime | None


class CalibrationResult(NamedTuple):
    """Result of calibration session."""

    session_id: str
    expert_id: str
    completed_at: datetime
    accuracy_score: float
    sensitivity: float
    specificity: float
    passed: bool
    notes: str


class AnnotationGuideline(NamedTuple):
    """Single annotation guideline rule."""

    rule_id: str
    category: str
    description: str
    examples: tuple[str, ...]
    common_errors: tuple[str, ...]
    reference_images: tuple[str, ...]


class DisagreementCase(NamedTuple):
    """Case of annotation disagreement."""

    case_id: str
    task_id: str
    disagreement_type: DisagreementType
    annotations: tuple[JSONDict, ...]
    annotator_ids: tuple[str, ...]
    resolution_status: str
    resolved_label: str | None


class GoldStandardCase(NamedTuple):
    """Confirmed gold standard annotation."""

    case_id: str
    image_path: str
    ground_truth_label: str
    confirmed_by: tuple[str, ...]
    confirmation_method: str
    created_at: datetime
    metadata: JSONDict


class ExpertVote(NamedTuple):
    """Expert vote on annotation."""

    expert_id: str
    label: str
    confidence: AnnotationConfidence
    reasoning: str
    timestamp: datetime


class NotificationService(Protocol):
    """Protocol for notification delivery."""

    def notify_expert(self, expert_id: str, message: str) -> None:
        """Send notification to expert."""
        ...


class MolecularConfirmation(Protocol):
    """Protocol for molecular confirmation service."""

    def request_confirmation(self, sample_id: str) -> str:
        """Request PCR/molecular confirmation."""
        ...

    def get_result(self, request_id: str) -> JSONDict | None:
        """Get confirmation result if available."""
        ...


@dataclass(frozen=True, slots=True)
class CalibrationCriteria:
    """Criteria for passing calibration."""

    min_accuracy: float = 0.90
    min_sensitivity: float = 0.85
    min_specificity: float = 0.85
    min_cases_reviewed: int = 50
    max_critical_errors: int = 0
    validity_days: int = 365


@dataclass(frozen=True, slots=True)
class AnnotationSchema:
    """Schema defining valid annotation structure."""

    schema_version: str
    organism_classes: tuple[str, ...]
    morphological_stages: tuple[str, ...]
    quality_indicators: tuple[str, ...]
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]

    def validate(self, annotation: JSONDict) -> tuple[bool, list[str]]:
        """Validate annotation against schema."""
        errors: list[str] = []

        for field_name in self.required_fields:
            if field_name not in annotation:
                errors.append(f"Missing required field: {field_name}")

        if "organism_class" in annotation:
            if annotation["organism_class"] not in self.organism_classes:
                errors.append(
                    f"Invalid organism class: {annotation['organism_class']}"
                )

        if "stage" in annotation:
            if annotation["stage"] not in self.morphological_stages:
                errors.append(f"Invalid stage: {annotation['stage']}")

        return len(errors) == 0, errors


@dataclass(slots=True)
class ExpertRegistry:
    """Registry of qualified expert annotators."""

    storage_dir: Path
    _experts: dict[str, ExpertProfile] = field(
        default_factory=dict, init=False, repr=False
    )
    _calibrations: dict[str, list[CalibrationResult]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize storage."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_experts()

    def _load_experts(self) -> None:
        """Load expert profiles from storage."""
        experts_file = self.storage_dir / "experts.json"
        if not experts_file.exists():
            return

        try:
            with experts_file.open("r") as f:
                data = json.load(f)
            for item in data:
                profile = ExpertProfile(
                    expert_id=item["expert_id"],
                    name=item["name"],
                    credentials=tuple(item["credentials"]),
                    institution=item["institution"],
                    level=ExpertLevel[item["level"]],
                    specializations=tuple(item["specializations"]),
                    years_experience=item["years_experience"],
                    calibration_status=CalibrationStatus[item["calibration_status"]],
                    last_calibration=(
                        datetime.fromisoformat(item["last_calibration"])
                        if item.get("last_calibration")
                        else None
                    ),
                )
                self._experts[profile.expert_id] = profile
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load experts: %s", e)

    def _save_experts(self) -> None:
        """Persist expert profiles."""
        experts_file = self.storage_dir / "experts.json"
        data = [
            {
                "expert_id": e.expert_id,
                "name": e.name,
                "credentials": list(e.credentials),
                "institution": e.institution,
                "level": e.level.name,
                "specializations": list(e.specializations),
                "years_experience": e.years_experience,
                "calibration_status": e.calibration_status.name,
                "last_calibration": (
                    e.last_calibration.isoformat() if e.last_calibration else None
                ),
            }
            for e in self._experts.values()
        ]
        with experts_file.open("w") as f:
            json.dump(data, f, indent=2)

    def register_expert(self, profile: ExpertProfile) -> None:
        """Register new expert."""
        self._experts[profile.expert_id] = profile
        self._save_experts()
        logger.info("Registered expert: %s", profile.name)

    def get_expert(self, expert_id: str) -> ExpertProfile | None:
        """Retrieve expert profile."""
        return self._experts.get(expert_id)

    def get_calibrated_experts(self) -> list[ExpertProfile]:
        """Get list of currently calibrated experts."""
        return [
            e
            for e in self._experts.values()
            if e.calibration_status == CalibrationStatus.PASSED
        ]

    def get_experts_by_level(self, min_level: ExpertLevel) -> list[ExpertProfile]:
        """Get experts at or above specified level."""
        return [
            e for e in self._experts.values() if e.level.value >= min_level.value
        ]

    def update_calibration_status(
        self, expert_id: str, status: CalibrationStatus
    ) -> None:
        """Update expert's calibration status."""
        expert = self._experts.get(expert_id)
        if not expert:
            return

        updated = ExpertProfile(
            expert_id=expert.expert_id,
            name=expert.name,
            credentials=expert.credentials,
            institution=expert.institution,
            level=expert.level,
            specializations=expert.specializations,
            years_experience=expert.years_experience,
            calibration_status=status,
            last_calibration=(
                datetime.now() if status == CalibrationStatus.PASSED
                else expert.last_calibration
            ),
        )
        self._experts[expert_id] = updated
        self._save_experts()


@dataclass(slots=True)
class CalibrationManager:
    """Manages expert calibration sessions."""

    storage_dir: Path
    criteria: CalibrationCriteria = field(default_factory=CalibrationCriteria)
    _sessions: dict[str, list[JSONDict]] = field(
        default_factory=dict, init=False, repr=False
    )
    _gold_standards: list[GoldStandardCase] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize storage."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_gold_standards()

    def _load_gold_standards(self) -> None:
        """Load gold standard cases."""
        gs_file = self.storage_dir / "gold_standards.json"
        if not gs_file.exists():
            return

        try:
            with gs_file.open("r") as f:
                data = json.load(f)
            for item in data:
                case = GoldStandardCase(
                    case_id=item["case_id"],
                    image_path=item["image_path"],
                    ground_truth_label=item["ground_truth_label"],
                    confirmed_by=tuple(item["confirmed_by"]),
                    confirmation_method=item["confirmation_method"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    metadata=item.get("metadata", {}),
                )
                self._gold_standards.append(case)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load gold standards: %s", e)

    def create_calibration_set(
        self, n_cases: int = 50
    ) -> list[GoldStandardCase]:
        """Create balanced calibration case set."""
        if len(self._gold_standards) < n_cases:
            return list(self._gold_standards)

        import random

        return random.sample(self._gold_standards, n_cases)

    def evaluate_calibration(
        self, expert_id: str, responses: Sequence[tuple[str, str]]
    ) -> CalibrationResult:
        """Evaluate expert's calibration responses."""
        import uuid

        session_id = str(uuid.uuid4())[:8]

        # Map responses to gold standards
        correct = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for case_id, response in responses:
            gold = next(
                (g for g in self._gold_standards if g.case_id == case_id), None
            )
            if not gold:
                continue

            is_positive = "histolytica" in gold.ground_truth_label.lower()
            predicted_positive = "histolytica" in response.lower()

            if response == gold.ground_truth_label:
                correct += 1

            if is_positive and predicted_positive:
                true_positive += 1
            elif not is_positive and not predicted_positive:
                true_negative += 1
            elif is_positive and not predicted_positive:
                false_negative += 1
            else:
                false_positive += 1

        total = len(responses)
        accuracy = correct / max(total, 1)
        sensitivity = true_positive / max(true_positive + false_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)

        passed = (
            accuracy >= self.criteria.min_accuracy
            and sensitivity >= self.criteria.min_sensitivity
            and specificity >= self.criteria.min_specificity
            and total >= self.criteria.min_cases_reviewed
        )

        result = CalibrationResult(
            session_id=session_id,
            expert_id=expert_id,
            completed_at=datetime.now(),
            accuracy_score=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            passed=passed,
            notes=f"Reviewed {total} cases",
        )

        return result


@dataclass(slots=True)
class GuidelineRepository:
    """Repository of annotation guidelines."""

    storage_dir: Path
    _guidelines: list[AnnotationGuideline] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize and create default guidelines."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._create_default_guidelines()

    def _create_default_guidelines(self) -> None:
        """Create standard entamoeba annotation guidelines."""
        self._guidelines = [
            AnnotationGuideline(
                rule_id="EH-TROPH-001",
                category="E. histolytica Trophozoite",
                description=(
                    "Identify trophozoites by size (12-60 μm), single nucleus "
                    "with central karyosome, finely granular cytoplasm, and "
                    "presence of ingested RBCs (pathognomonic)."
                ),
                examples=(
                    "Elongated pseudopod formation",
                    "Ingested erythrocytes visible",
                    "Progressive motility pattern",
                ),
                common_errors=(
                    "Confusion with macrophages",
                    "Missing small trophozoites",
                    "Misidentifying E. coli as E. histolytica",
                ),
                reference_images=("ref/eh_troph_001.png", "ref/eh_troph_002.png"),
            ),
            AnnotationGuideline(
                rule_id="EH-CYST-001",
                category="E. histolytica Cyst",
                description=(
                    "Identify cysts by spherical shape (10-16 μm), 1-4 nuclei "
                    "with central karyosome, chromatoid bars with rounded ends."
                ),
                examples=(
                    "Mature cyst with 4 nuclei",
                    "Immature cyst with 1-2 nuclei",
                    "Chromatoid bars visible",
                ),
                common_errors=(
                    "Confusion with E. coli cysts (8 nuclei)",
                    "Missing immature cysts",
                    "Misidentifying artifacts as cysts",
                ),
                reference_images=("ref/eh_cyst_001.png", "ref/eh_cyst_002.png"),
            ),
            AnnotationGuideline(
                rule_id="NEG-001",
                category="Negative",
                description=(
                    "Confirm absence of Entamoeba species. Document any "
                    "look-alike structures and rationale for negative call."
                ),
                examples=(
                    "WBCs misidentified as trophozoites",
                    "Yeast cells mistaken for cysts",
                    "Plant cells from dietary material",
                ),
                common_errors=(
                    "False negative due to low parasite load",
                    "Missing partially obscured organisms",
                    "Over-reliance on automated pre-screening",
                ),
                reference_images=("ref/neg_001.png", "ref/neg_002.png"),
            ),
        ]

    def get_guidelines_by_category(
        self, category: str
    ) -> list[AnnotationGuideline]:
        """Get guidelines for specific category."""
        return [g for g in self._guidelines if g.category == category]

    def get_all_guidelines(self) -> list[AnnotationGuideline]:
        """Get all annotation guidelines."""
        return list(self._guidelines)

    def export_guidelines_pdf(self, output_path: Path) -> None:
        """Export guidelines as PDF document."""
        # Simplified export - would use reportlab in production
        markdown_content = "# Annotation Guidelines\n\n"
        for guideline in self._guidelines:
            markdown_content += f"## {guideline.category}\n\n"
            markdown_content += f"**ID:** {guideline.rule_id}\n\n"
            markdown_content += f"{guideline.description}\n\n"
            markdown_content += "**Examples:**\n"
            for example in guideline.examples:
                markdown_content += f"- {example}\n"
            markdown_content += "\n**Common Errors:**\n"
            for error in guideline.common_errors:
                markdown_content += f"- {error}\n"
            markdown_content += "\n---\n\n"

        output_path.write_text(markdown_content)


@dataclass(slots=True)
class DisagreementResolver:
    """Resolves annotation disagreements."""

    expert_registry: ExpertRegistry
    storage_dir: Path
    _cases: list[DisagreementCase] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize storage."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_cases()

    def _load_cases(self) -> None:
        """Load disagreement cases."""
        cases_file = self.storage_dir / "disagreements.json"
        if not cases_file.exists():
            return

        try:
            with cases_file.open("r") as f:
                data = json.load(f)
            for item in data:
                case = DisagreementCase(
                    case_id=item["case_id"],
                    task_id=item["task_id"],
                    disagreement_type=DisagreementType[item["disagreement_type"]],
                    annotations=tuple(item["annotations"]),
                    annotator_ids=tuple(item["annotator_ids"]),
                    resolution_status=item["resolution_status"],
                    resolved_label=item.get("resolved_label"),
                )
                self._cases.append(case)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load disagreement cases: %s", e)

    def _save_cases(self) -> None:
        """Persist disagreement cases."""
        cases_file = self.storage_dir / "disagreements.json"
        data = [
            {
                "case_id": c.case_id,
                "task_id": c.task_id,
                "disagreement_type": c.disagreement_type.name,
                "annotations": list(c.annotations),
                "annotator_ids": list(c.annotator_ids),
                "resolution_status": c.resolution_status,
                "resolved_label": c.resolved_label,
            }
            for c in self._cases
        ]
        with cases_file.open("w") as f:
            json.dump(data, f, indent=2)

    def detect_disagreement(
        self,
        task_id: str,
        annotations: Sequence[JSONDict],
        annotator_ids: Sequence[str],
    ) -> DisagreementCase | None:
        """Detect if annotations have significant disagreement."""
        if len(annotations) < 2:
            return None

        # Extract primary labels with strict null coalescing for type safety
        labels: list[str] = [
            str(a.get("label") or a.get("organism_class") or "")
            for a in annotations
        ]
        unique_labels: set[str] = set(labels)

        if len(unique_labels) <= 1:
            return None

        # Determine disagreement type with type-safe string operations
        if any("histolytica" in lbl for lbl in labels) and any(
            "dispar" in lbl for lbl in labels
        ):
            dtype = DisagreementType.SPECIES_MISMATCH
        elif any("negative" in lbl.lower() for lbl in labels if lbl):
            dtype = DisagreementType.PRESENCE_ABSENCE
        else:
            dtype = DisagreementType.STAGE_MISMATCH

        import uuid

        case = DisagreementCase(
            case_id=str(uuid.uuid4())[:8],
            task_id=task_id,
            disagreement_type=dtype,
            annotations=tuple(annotations),
            annotator_ids=tuple(annotator_ids),
            resolution_status="pending",
            resolved_label=None,
        )
        self._cases.append(case)
        self._save_cases()
        return case

    def resolve_by_majority(self, case_id: str) -> str | None:
        """Resolve disagreement by majority vote."""
        case = next((c for c in self._cases if c.case_id == case_id), None)
        if not case:
            return None

        labels = [a.get("label", "") for a in case.annotations]
        vote_counts: dict[str, int] = {}
        for label in labels:
            vote_counts[label] = vote_counts.get(label, 0) + 1

        majority = max(vote_counts.keys(), key=lambda k: vote_counts[k])

        # Update case
        idx = next(i for i, c in enumerate(self._cases) if c.case_id == case_id)
        self._cases[idx] = DisagreementCase(
            case_id=case.case_id,
            task_id=case.task_id,
            disagreement_type=case.disagreement_type,
            annotations=case.annotations,
            annotator_ids=case.annotator_ids,
            resolution_status="resolved_majority",
            resolved_label=majority,
        )
        self._save_cases()
        return majority

    def resolve_by_senior(
        self, case_id: str, senior_expert_id: str, final_label: str
    ) -> bool:
        """Resolve disagreement by senior expert arbitration."""
        case = next((c for c in self._cases if c.case_id == case_id), None)
        if not case:
            return False

        expert = self.expert_registry.get_expert(senior_expert_id)
        if not expert or expert.level.value < ExpertLevel.SENIOR_SPECIALIST.value:
            logger.warning("Expert %s not qualified for arbitration", senior_expert_id)
            return False

        idx = next(i for i, c in enumerate(self._cases) if c.case_id == case_id)
        self._cases[idx] = DisagreementCase(
            case_id=case.case_id,
            task_id=case.task_id,
            disagreement_type=case.disagreement_type,
            annotations=case.annotations,
            annotator_ids=case.annotator_ids,
            resolution_status=f"resolved_by_{senior_expert_id}",
            resolved_label=final_label,
        )
        self._save_cases()
        return True

    def get_unresolved_cases(self) -> list[DisagreementCase]:
        """Get all unresolved disagreement cases."""
        return [c for c in self._cases if c.resolution_status == "pending"]


@dataclass(slots=True)
class GoldStandardCurator:
    """Curates gold standard annotation dataset."""

    storage_dir: Path
    min_confirmations: int = 3
    _gold_standards: list[GoldStandardCase] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize storage."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_gold_standard(
        self,
        image_path: str,
        label: str,
        confirmed_by: Sequence[str],
        method: str = "expert_consensus",
        metadata: JSONDict | None = None,
    ) -> GoldStandardCase | None:
        """Create new gold standard case if criteria met."""
        if len(confirmed_by) < self.min_confirmations:
            logger.warning(
                "Insufficient confirmations: %d < %d",
                len(confirmed_by),
                self.min_confirmations,
            )
            return None

        import uuid

        case = GoldStandardCase(
            case_id=str(uuid.uuid4())[:8],
            image_path=image_path,
            ground_truth_label=label,
            confirmed_by=tuple(confirmed_by),
            confirmation_method=method,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        self._gold_standards.append(case)
        self._save_gold_standards()
        return case

    def _save_gold_standards(self) -> None:
        """Persist gold standards."""
        gs_file = self.storage_dir / "gold_standards.json"
        data = [
            {
                "case_id": g.case_id,
                "image_path": g.image_path,
                "ground_truth_label": g.ground_truth_label,
                "confirmed_by": list(g.confirmed_by),
                "confirmation_method": g.confirmation_method,
                "created_at": g.created_at.isoformat(),
                "metadata": g.metadata,
            }
            for g in self._gold_standards
        ]
        with gs_file.open("w") as f:
            json.dump(data, f, indent=2)

    def get_gold_standards_by_label(self, label: str) -> list[GoldStandardCase]:
        """Get gold standards for specific label."""
        return [g for g in self._gold_standards if g.ground_truth_label == label]

    def export_for_validation(self, output_dir: Path) -> dict[str, list[str]]:
        """Export gold standards for model validation."""
        output_dir.mkdir(parents=True, exist_ok=True)

        by_label: dict[str, list[str]] = {}
        for case in self._gold_standards:
            if case.ground_truth_label not in by_label:
                by_label[case.ground_truth_label] = []
            by_label[case.ground_truth_label].append(case.image_path)

        manifest = output_dir / "gold_standard_manifest.json"
        with manifest.open("w") as f:
            json.dump(by_label, f, indent=2)

        return by_label


@dataclass(slots=True)
class ExpertAnnotationProtocol:
    """Main coordinator for expert annotation workflow."""

    storage_dir: Path
    expert_registry: ExpertRegistry = field(init=False)
    calibration_manager: CalibrationManager = field(init=False)
    guideline_repository: GuidelineRepository = field(init=False)
    disagreement_resolver: DisagreementResolver = field(init=False)
    gold_curator: GoldStandardCurator = field(init=False)
    annotation_schema: AnnotationSchema = field(init=False)

    def __post_init__(self) -> None:
        """Initialize all components."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.expert_registry = ExpertRegistry(
            storage_dir=self.storage_dir / "experts"
        )
        self.calibration_manager = CalibrationManager(
            storage_dir=self.storage_dir / "calibration"
        )
        self.guideline_repository = GuidelineRepository(
            storage_dir=self.storage_dir / "guidelines"
        )
        self.disagreement_resolver = DisagreementResolver(
            expert_registry=self.expert_registry,
            storage_dir=self.storage_dir / "disagreements",
        )
        self.gold_curator = GoldStandardCurator(
            storage_dir=self.storage_dir / "gold_standards"
        )
        self.annotation_schema = AnnotationSchema(
            schema_version="1.0.0",
            organism_classes=(
                "E. histolytica trophozoite",
                "E. histolytica cyst",
                "E. dispar trophozoite",
                "E. dispar cyst",
                "Negative",
            ),
            morphological_stages=("trophozoite", "cyst", "precyst"),
            quality_indicators=("excellent", "good", "acceptable", "poor"),
            required_fields=("organism_class", "confidence"),
            optional_fields=("stage", "quality", "notes"),
        )

    def onboard_expert(
        self,
        name: str,
        credentials: Sequence[str],
        institution: str,
        specializations: Sequence[str],
        years_experience: int,
    ) -> ExpertProfile:
        """Onboard new expert annotator."""
        import uuid

        profile = ExpertProfile(
            expert_id=str(uuid.uuid4())[:8],
            name=name,
            credentials=tuple(credentials),
            institution=institution,
            level=ExpertLevel.TRAINEE,
            specializations=tuple(specializations),
            years_experience=years_experience,
            calibration_status=CalibrationStatus.NOT_STARTED,
            last_calibration=None,
        )
        self.expert_registry.register_expert(profile)
        return profile

    def run_calibration(
        self, expert_id: str
    ) -> CalibrationResult:
        """Run calibration session for expert."""
        cases = self.calibration_manager.create_calibration_set()
        # Simulated responses for demonstration
        responses = [(c.case_id, c.ground_truth_label) for c in cases]
        result = self.calibration_manager.evaluate_calibration(expert_id, responses)

        status = (
            CalibrationStatus.PASSED if result.passed else CalibrationStatus.FAILED
        )
        self.expert_registry.update_calibration_status(expert_id, status)

        return result


def create_annotation_protocol(storage_dir: Path) -> ExpertAnnotationProtocol:
    """Factory function for expert annotation protocol."""
    return ExpertAnnotationProtocol(storage_dir=storage_dir)


class TrainingModuleStatus(Enum):
    """Status of training module completion."""

    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    NEEDS_REVIEW = auto()


class CertificationLevel(Enum):
    """Certification levels for experts."""

    BASIC = auto()
    INTERMEDIATE = auto()
    ADVANCED = auto()
    SPECIALIST = auto()
    MASTER = auto()


class AuditAction(Enum):
    """Types of audit actions."""

    LOGIN = auto()
    ANNOTATION_CREATED = auto()
    ANNOTATION_MODIFIED = auto()
    CALIBRATION_ATTEMPTED = auto()
    GUIDELINE_ACCESSED = auto()
    CASE_REVIEWED = auto()
    DISAGREEMENT_RESOLVED = auto()


class ProficiencyTrend(Enum):
    """Trend in proficiency over time."""

    IMPROVING = auto()
    STABLE = auto()
    DECLINING = auto()
    INSUFFICIENT_DATA = auto()


class TrainingModule(NamedTuple):
    """Training module definition."""

    module_id: str
    title: str
    description: str
    estimated_hours: float
    required_for_level: CertificationLevel
    prerequisites: tuple[str, ...]
    created_at: datetime


class TrainingCompletion(NamedTuple):
    """Record of training module completion."""

    expert_id: str
    module_id: str
    started_at: datetime
    completed_at: datetime | None
    score: float
    status: TrainingModuleStatus


class ProficiencyScore(NamedTuple):
    """Proficiency score for an expert."""

    expert_id: str
    timestamp: datetime
    overall_score: float
    accuracy: float
    consistency: float
    speed: float
    edge_case_handling: float


class AuditRecord(NamedTuple):
    """Audit trail record."""

    record_id: str
    expert_id: str
    action: AuditAction
    timestamp: datetime
    details: dict[str, Any]
    ip_address: str


class CertificationRecord(NamedTuple):
    """Record of expert certification."""

    expert_id: str
    level: CertificationLevel
    granted_at: datetime
    expires_at: datetime
    granting_authority: str
    certificate_id: str


class DisagreementPattern(NamedTuple):
    """Pattern in annotation disagreements."""

    pattern_id: str
    pattern_type: str
    frequency: int
    experts_involved: tuple[str, ...]
    common_labels: tuple[str, ...]
    suggested_action: str


class ConsensusVote(NamedTuple):
    """Vote in consensus building."""

    expert_id: str
    label: int
    confidence: float
    timestamp: datetime
    reasoning: str


class GoldStandardVersion(NamedTuple):
    """Version of gold standard case."""

    case_id: str
    version: int
    label: int
    updated_by: str
    updated_at: datetime
    change_reason: str


@dataclass(slots=True)
class TrainingTracker:
    """Tracks expert training progress.

    Manages training modules, completion status,
    and prerequisites for certification.
    """

    storage_dir: Path
    _modules: dict[str, TrainingModule] = field(
        default_factory=dict, init=False, repr=False
    )
    _completions: dict[str, list[TrainingCompletion]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize and load training data."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_modules()

    def _load_modules(self) -> None:
        """Load training modules from storage."""
        modules_file = self.storage_dir / "training_modules.json"
        if not modules_file.exists():
            self._create_default_modules()
            return

        try:
            with modules_file.open("r") as f:
                data = json.load(f)

            for item in data.get("modules", []):
                module = TrainingModule(
                    module_id=item["module_id"],
                    title=item["title"],
                    description=item["description"],
                    estimated_hours=item["estimated_hours"],
                    required_for_level=CertificationLevel[item["required_for_level"]],
                    prerequisites=tuple(item.get("prerequisites", [])),
                    created_at=datetime.fromisoformat(item["created_at"]),
                )
                self._modules[module.module_id] = module
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load training modules: %s", e)

    def _create_default_modules(self) -> None:
        """Create default training modules."""
        defaults = [
            TrainingModule(
                module_id="TM001",
                title="Entamoeba Morphology Basics",
                description="Introduction to Entamoeba species morphology",
                estimated_hours=2.0,
                required_for_level=CertificationLevel.BASIC,
                prerequisites=(),
                created_at=datetime.now(),
            ),
            TrainingModule(
                module_id="TM002",
                title="Trophozoite vs Cyst Differentiation",
                description="Distinguishing life cycle stages",
                estimated_hours=3.0,
                required_for_level=CertificationLevel.BASIC,
                prerequisites=("TM001",),
                created_at=datetime.now(),
            ),
            TrainingModule(
                module_id="TM003",
                title="E. histolytica vs E. dispar",
                description="Differentiating pathogenic from non-pathogenic",
                estimated_hours=4.0,
                required_for_level=CertificationLevel.INTERMEDIATE,
                prerequisites=("TM001", "TM002"),
                created_at=datetime.now(),
            ),
        ]

        for module in defaults:
            self._modules[module.module_id] = module

    def get_module(self, module_id: str) -> TrainingModule | None:
        """Get training module by ID."""
        return self._modules.get(module_id)

    def list_modules(
        self,
        level: CertificationLevel | None = None,
    ) -> list[TrainingModule]:
        """List training modules with optional filtering.

        Parameters
        ----------
        level : CertificationLevel, optional
            Filter by required certification level.

        Returns
        -------
        list[TrainingModule]
            Matching training modules.
        """
        modules = list(self._modules.values())

        if level is not None:
            modules = [m for m in modules if m.required_for_level == level]

        return modules

    def start_module(
        self, expert_id: str, module_id: str
    ) -> TrainingCompletion:
        """Start a training module for an expert.

        Parameters
        ----------
        expert_id : str
            Expert identifier.
        module_id : str
            Training module identifier.

        Returns
        -------
        TrainingCompletion
            Completion record.
        """
        completion = TrainingCompletion(
            expert_id=expert_id,
            module_id=module_id,
            started_at=datetime.now(),
            completed_at=None,
            score=0.0,
            status=TrainingModuleStatus.IN_PROGRESS,
        )

        with self._lock:
            self._completions[expert_id].append(completion)

        return completion

    def complete_module(
        self,
        expert_id: str,
        module_id: str,
        score: float,
    ) -> TrainingCompletion:
        """Mark a module as completed.

        Parameters
        ----------
        expert_id : str
            Expert identifier.
        module_id : str
            Training module identifier.
        score : float
            Completion score (0-100).

        Returns
        -------
        TrainingCompletion
            Updated completion record.
        """
        status = (
            TrainingModuleStatus.COMPLETED
            if score >= 80.0
            else TrainingModuleStatus.NEEDS_REVIEW
        )

        completion = TrainingCompletion(
            expert_id=expert_id,
            module_id=module_id,
            started_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now(),
            score=score,
            status=status,
        )

        with self._lock:
            existing = [
                c for c in self._completions[expert_id]
                if c.module_id != module_id
            ]
            existing.append(completion)
            self._completions[expert_id] = existing

        return completion

    def get_expert_progress(
        self, expert_id: str
    ) -> dict[str, TrainingCompletion]:
        """Get training progress for expert.

        Parameters
        ----------
        expert_id : str
            Expert identifier.

        Returns
        -------
        dict[str, TrainingCompletion]
            Module ID to completion mapping.
        """
        completions = self._completions.get(expert_id, [])
        return {c.module_id: c for c in completions}

    def check_prerequisites(
        self,
        expert_id: str,
        module_id: str,
    ) -> tuple[bool, list[str]]:
        """Check if expert meets module prerequisites.

        Parameters
        ----------
        expert_id : str
            Expert identifier.
        module_id : str
            Target module ID.

        Returns
        -------
        tuple[bool, list[str]]
            (All met, missing prerequisite IDs).
        """
        module = self._modules.get(module_id)
        if not module:
            return False, [module_id]

        progress = self.get_expert_progress(expert_id)
        missing = []

        for prereq_id in module.prerequisites:
            prereq_completion = progress.get(prereq_id)
            if (
                not prereq_completion
                or prereq_completion.status != TrainingModuleStatus.COMPLETED
            ):
                missing.append(prereq_id)

        return len(missing) == 0, missing


@dataclass(slots=True)
class ProficiencyTracker:
    """Tracks expert proficiency over time.

    Monitors performance trends and identifies
    areas needing improvement.
    """

    _scores: dict[str, list[ProficiencyScore]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    min_samples_for_trend: int = 5

    def record_score(
        self,
        expert_id: str,
        accuracy: float,
        consistency: float,
        speed: float,
        edge_case_handling: float,
    ) -> ProficiencyScore:
        """Record proficiency score for expert.

        Parameters
        ----------
        expert_id : str
            Expert identifier.
        accuracy : float
            Accuracy score (0-1).
        consistency : float
            Consistency score (0-1).
        speed : float
            Speed score (0-1).
        edge_case_handling : float
            Edge case handling score (0-1).

        Returns
        -------
        ProficiencyScore
            Recorded score.
        """
        overall = (
            0.4 * accuracy
            + 0.3 * consistency
            + 0.1 * speed
            + 0.2 * edge_case_handling
        )

        score = ProficiencyScore(
            expert_id=expert_id,
            timestamp=datetime.now(),
            overall_score=overall,
            accuracy=accuracy,
            consistency=consistency,
            speed=speed,
            edge_case_handling=edge_case_handling,
        )

        self._scores[expert_id].append(score)

        if len(self._scores[expert_id]) > 100:
            self._scores[expert_id] = self._scores[expert_id][-50:]

        return score

    def get_current_score(
        self, expert_id: str
    ) -> ProficiencyScore | None:
        """Get most recent proficiency score.

        Parameters
        ----------
        expert_id : str
            Expert identifier.

        Returns
        -------
        ProficiencyScore or None
            Most recent score if available.
        """
        scores = self._scores.get(expert_id, [])
        return scores[-1] if scores else None

    def get_trend(self, expert_id: str) -> ProficiencyTrend:
        """Determine proficiency trend for expert.

        Parameters
        ----------
        expert_id : str
            Expert identifier.

        Returns
        -------
        ProficiencyTrend
            Trend direction.
        """
        scores = self._scores.get(expert_id, [])

        if len(scores) < self.min_samples_for_trend:
            return ProficiencyTrend.INSUFFICIENT_DATA

        recent = scores[-self.min_samples_for_trend:]
        overall_scores = [s.overall_score for s in recent]

        first_half = overall_scores[: len(overall_scores) // 2]
        second_half = overall_scores[len(overall_scores) // 2 :]

        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)

        diff = avg_second - avg_first

        if diff > 0.05:
            return ProficiencyTrend.IMPROVING
        if diff < -0.05:
            return ProficiencyTrend.DECLINING

        return ProficiencyTrend.STABLE

    def get_weakest_area(self, expert_id: str) -> str | None:
        """Identify weakest proficiency area for expert.

        Parameters
        ----------
        expert_id : str
            Expert identifier.

        Returns
        -------
        str or None
            Name of weakest area or None.
        """
        score = self.get_current_score(expert_id)
        if not score:
            return None

        areas = {
            "accuracy": score.accuracy,
            "consistency": score.consistency,
            "speed": score.speed,
            "edge_case_handling": score.edge_case_handling,
        }

        return min(areas, key=lambda k: areas[k])

    def get_ranking(self) -> list[tuple[str, float]]:
        """Get expert ranking by proficiency.

        Returns
        -------
        list[tuple[str, float]]
            (expert_id, score) sorted by score descending.
        """
        rankings = []

        for expert_id in self._scores:
            score = self.get_current_score(expert_id)
            if score:
                rankings.append((expert_id, score.overall_score))

        return sorted(rankings, key=lambda x: -x[1])


@dataclass(slots=True)
class CertificationManager:
    """Manages expert certification lifecycle.

    Handles certification requirements, issuance,
    and expiration tracking.
    """

    storage_dir: Path
    certification_validity_days: int = 365
    _certifications: dict[str, list[CertificationRecord]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _level_requirements: dict[CertificationLevel, tuple[str, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize certification manager."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._setup_requirements()

    def _setup_requirements(self) -> None:
        """Setup certification level requirements."""
        self._level_requirements = {
            CertificationLevel.BASIC: ("TM001", "TM002"),
            CertificationLevel.INTERMEDIATE: ("TM001", "TM002", "TM003"),
            CertificationLevel.ADVANCED: (
                "TM001", "TM002", "TM003", "CAL001"
            ),
            CertificationLevel.SPECIALIST: (
                "TM001", "TM002", "TM003", "CAL001", "CAL002"
            ),
            CertificationLevel.MASTER: (
                "TM001", "TM002", "TM003", "CAL001", "CAL002", "RES001"
            ),
        }

    def check_eligibility(
        self,
        expert_id: str,
        level: CertificationLevel,
        completed_modules: Sequence[str],
    ) -> tuple[bool, list[str]]:
        """Check if expert is eligible for certification.

        Parameters
        ----------
        expert_id : str
            Expert identifier.
        level : CertificationLevel
            Target certification level.
        completed_modules : Sequence[str]
            Completed training module IDs.

        Returns
        -------
        tuple[bool, list[str]]
            (Is eligible, missing requirements).
        """
        required = self._level_requirements.get(level, ())
        completed_set = set(completed_modules)
        missing = [r for r in required if r not in completed_set]

        return len(missing) == 0, missing

    def issue_certification(
        self,
        expert_id: str,
        level: CertificationLevel,
        granting_authority: str = "Amoebanator Training Board",
    ) -> CertificationRecord:
        """Issue certification to expert.

        Parameters
        ----------
        expert_id : str
            Expert identifier.
        level : CertificationLevel
            Certification level.
        granting_authority : str
            Authority issuing the certification.

        Returns
        -------
        CertificationRecord
            The certification record.
        """
        import uuid

        now = datetime.now()
        cert = CertificationRecord(
            expert_id=expert_id,
            level=level,
            granted_at=now,
            expires_at=now + timedelta(days=self.certification_validity_days),
            granting_authority=granting_authority,
            certificate_id=str(uuid.uuid4())[:12].upper(),
        )

        self._certifications[expert_id].append(cert)
        logger.info(
            "Issued %s certification to %s", level.name, expert_id
        )

        return cert

    def get_current_certification(
        self, expert_id: str
    ) -> CertificationRecord | None:
        """Get current valid certification for expert.

        Parameters
        ----------
        expert_id : str
            Expert identifier.

        Returns
        -------
        CertificationRecord or None
            Current certification if valid.
        """
        certs = self._certifications.get(expert_id, [])
        now = datetime.now()

        valid_certs = [c for c in certs if c.expires_at > now]

        if not valid_certs:
            return None

        return max(valid_certs, key=lambda c: c.level.value)

    def get_expiring_soon(
        self, days: int = 30
    ) -> list[CertificationRecord]:
        """Get certifications expiring soon.

        Parameters
        ----------
        days : int
            Number of days to check ahead.

        Returns
        -------
        list[CertificationRecord]
            Certifications expiring within days.
        """
        now = datetime.now()
        threshold = now + timedelta(days=days)
        expiring = []

        for certs in self._certifications.values():
            for cert in certs:
                if now < cert.expires_at <= threshold:
                    expiring.append(cert)

        return expiring


@dataclass(slots=True)
class AuditTrail:
    """Maintains audit trail for compliance.

    Records all significant actions for
    regulatory compliance and traceability.
    """

    storage_dir: Path
    _records: list[AuditRecord] = field(
        default_factory=list, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    max_records: int = 10000

    def __post_init__(self) -> None:
        """Initialize audit trail."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        expert_id: str,
        action: AuditAction,
        details: dict[str, Any] | None = None,
        ip_address: str = "127.0.0.1",
    ) -> AuditRecord:
        """Log an audit record.

        Parameters
        ----------
        expert_id : str
            Expert who performed the action.
        action : AuditAction
            Type of action.
        details : dict[str, Any], optional
            Additional details about the action.
        ip_address : str
            IP address of the client.

        Returns
        -------
        AuditRecord
            The created audit record.
        """
        import uuid

        record = AuditRecord(
            record_id=str(uuid.uuid4())[:12],
            expert_id=expert_id,
            action=action,
            timestamp=datetime.now(),
            details=details or {},
            ip_address=ip_address,
        )

        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records // 2 :]

        return record

    def query(
        self,
        expert_id: str | None = None,
        action: AuditAction | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        """Query audit records.

        Parameters
        ----------
        expert_id : str, optional
            Filter by expert.
        action : AuditAction, optional
            Filter by action type.
        start_time : datetime, optional
            Start of time range.
        end_time : datetime, optional
            End of time range.
        limit : int
            Maximum records to return.

        Returns
        -------
        list[AuditRecord]
            Matching audit records.
        """
        results = self._records.copy()

        if expert_id:
            results = [r for r in results if r.expert_id == expert_id]

        if action:
            results = [r for r in results if r.action == action]

        if start_time:
            results = [r for r in results if r.timestamp >= start_time]

        if end_time:
            results = [r for r in results if r.timestamp <= end_time]

        return results[-limit:]

    def export(self, output_file: Path) -> int:
        """Export audit trail to file.

        Parameters
        ----------
        output_file : Path
            Output file path.

        Returns
        -------
        int
            Number of records exported.
        """
        data = [
            {
                "record_id": r.record_id,
                "expert_id": r.expert_id,
                "action": r.action.name,
                "timestamp": r.timestamp.isoformat(),
                "details": r.details,
                "ip_address": r.ip_address,
            }
            for r in self._records
        ]

        with output_file.open("w") as f:
            json.dump(data, f, indent=2)

        return len(data)


@dataclass(slots=True)
class DisagreementPatternAnalyzer:
    """Analyzes patterns in annotation disagreements.

    Identifies systematic disagreement patterns
    to improve guidelines and training.
    """

    min_frequency_threshold: int = 3
    _patterns: list[DisagreementPattern] = field(
        default_factory=list, init=False, repr=False
    )
    _disagreement_log: list[tuple[str, str, str, str]] = field(
        default_factory=list, init=False, repr=False
    )

    def record_disagreement(
        self,
        case_id: str,
        expert_a: str,
        expert_b: str,
        label_a: str,
        label_b: str,
    ) -> None:
        """Record a disagreement instance.

        Parameters
        ----------
        case_id : str
            Case identifier.
        expert_a : str
            First expert ID.
        expert_b : str
            Second expert ID.
        label_a : str
            First expert's label.
        label_b : str
            Second expert's label.
        """
        self._disagreement_log.append(
            (case_id, f"{expert_a},{expert_b}", label_a, label_b)
        )

    def analyze_patterns(self) -> list[DisagreementPattern]:
        """Analyze recorded disagreements for patterns.

        Returns
        -------
        list[DisagreementPattern]
            Identified patterns.
        """
        from collections import Counter

        label_pair_counts: Counter[tuple[str, str]] = Counter()
        pair_experts: dict[tuple[str, str], set[str]] = defaultdict(set)

        for case_id, experts_str, label_a, label_b in self._disagreement_log:
            sorted_pair = sorted([label_a, label_b])
            pair: tuple[str, str] = (sorted_pair[0], sorted_pair[1])
            label_pair_counts[pair] += 1
            for exp in experts_str.split(","):
                pair_experts[pair].add(exp)

        patterns = []
        for pair, count in label_pair_counts.most_common():
            if count >= self.min_frequency_threshold:
                pattern = DisagreementPattern(
                    pattern_id=f"DP_{pair[0][:3]}_{pair[1][:3]}",
                    pattern_type="label_confusion",
                    frequency=count,
                    experts_involved=tuple(pair_experts[pair]),
                    common_labels=pair,
                    suggested_action=self._suggest_action(pair, count),
                )
                patterns.append(pattern)

        self._patterns = patterns
        return patterns

    def _suggest_action(
        self,
        label_pair: tuple[str, str],
        frequency: int,
    ) -> str:
        """Suggest action based on disagreement pattern.

        Parameters
        ----------
        label_pair : tuple[str, str]
            Confused label pair.
        frequency : int
            Disagreement frequency.

        Returns
        -------
        str
            Suggested action.
        """
        if frequency >= 10:
            return "Update guidelines with detailed differentiation criteria"
        if frequency >= 5:
            return "Create training module for this specific confusion"
        return "Monitor and discuss in next calibration session"

    def get_top_patterns(self, n: int = 5) -> list[DisagreementPattern]:
        """Get top N most frequent disagreement patterns.

        Parameters
        ----------
        n : int
            Number of patterns to return.

        Returns
        -------
        list[DisagreementPattern]
            Top patterns by frequency.
        """
        return sorted(
            self._patterns, key=lambda p: -p.frequency
        )[:n]


@dataclass(slots=True)
class ConsensusProtocol:
    """Protocol for building expert consensus.

    Implements structured consensus building
    with evidence documentation.
    """

    required_agreement_threshold: float = 0.8
    min_experts_for_consensus: int = 3
    _votes: dict[str, list[ConsensusVote]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    def submit_vote(
        self,
        case_id: str,
        expert_id: str,
        label: int,
        confidence: float,
        reasoning: str = "",
    ) -> ConsensusVote:
        """Submit vote for consensus case.

        Parameters
        ----------
        case_id : str
            Case identifier.
        expert_id : str
            Expert identifier.
        label : int
            Expert's label.
        confidence : float
            Confidence in vote (0-1).
        reasoning : str
            Reasoning for the vote.

        Returns
        -------
        ConsensusVote
            The recorded vote.
        """
        vote = ConsensusVote(
            expert_id=expert_id,
            label=label,
            confidence=confidence,
            timestamp=datetime.now(),
            reasoning=reasoning,
        )

        self._votes[case_id].append(vote)
        return vote

    def check_consensus(
        self, case_id: str
    ) -> tuple[bool, int | None, float]:
        """Check if consensus has been reached.

        Parameters
        ----------
        case_id : str
            Case identifier.

        Returns
        -------
        tuple[bool, int | None, float]
            (Consensus reached, consensus label, agreement ratio).
        """
        votes = self._votes.get(case_id, [])

        if len(votes) < self.min_experts_for_consensus:
            return False, None, 0.0

        label_counts: dict[int, int] = defaultdict(int)
        for vote in votes:
            label_counts[vote.label] += 1

        total = len(votes)
        most_common = max(label_counts.items(), key=lambda x: x[1])
        agreement = most_common[1] / total

        if agreement >= self.required_agreement_threshold:
            return True, most_common[0], agreement

        return False, None, agreement

    def get_weighted_consensus(
        self, case_id: str
    ) -> tuple[int | None, float]:
        """Get consensus weighted by confidence scores.

        Parameters
        ----------
        case_id : str
            Case identifier.

        Returns
        -------
        tuple[int | None, float]
            (Consensus label, weighted confidence).
        """
        votes = self._votes.get(case_id, [])

        if not votes:
            return None, 0.0

        label_weights: dict[int, float] = defaultdict(float)
        for vote in votes:
            label_weights[vote.label] += vote.confidence

        total_weight = sum(label_weights.values())
        if total_weight == 0:
            return None, 0.0

        best_label = max(label_weights, key=lambda k: label_weights[k])
        weighted_conf = label_weights[best_label] / total_weight

        return best_label, weighted_conf

    def get_dissenting_experts(
        self, case_id: str
    ) -> list[str]:
        """Get experts who dissented from majority.

        Parameters
        ----------
        case_id : str
            Case identifier.

        Returns
        -------
        list[str]
            Expert IDs who dissented.
        """
        votes = self._votes.get(case_id, [])

        if len(votes) < 2:
            return []

        label_counts: dict[int, int] = defaultdict(int)
        for vote in votes:
            label_counts[vote.label] += 1

        majority_label = max(label_counts, key=lambda k: label_counts[k])

        return [v.expert_id for v in votes if v.label != majority_label]


@dataclass(slots=True)
class GoldStandardVersionManager:
    """Manages versions of gold standard cases.

    Tracks changes and maintains history
    for gold standard labels.
    """

    _versions: dict[str, list[GoldStandardVersion]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    def create_version(
        self,
        case_id: str,
        label: int,
        updated_by: str,
        change_reason: str,
    ) -> GoldStandardVersion:
        """Create new version of gold standard.

        Parameters
        ----------
        case_id : str
            Case identifier.
        label : int
            New gold standard label.
        updated_by : str
            Expert who made the update.
        change_reason : str
            Reason for the change.

        Returns
        -------
        GoldStandardVersion
            The new version.
        """
        existing = self._versions.get(case_id, [])
        version_num = len(existing) + 1

        version = GoldStandardVersion(
            case_id=case_id,
            version=version_num,
            label=label,
            updated_by=updated_by,
            updated_at=datetime.now(),
            change_reason=change_reason,
        )

        self._versions[case_id].append(version)
        return version

    def get_current_version(
        self, case_id: str
    ) -> GoldStandardVersion | None:
        """Get current version of gold standard.

        Parameters
        ----------
        case_id : str
            Case identifier.

        Returns
        -------
        GoldStandardVersion or None
            Current version if exists.
        """
        versions = self._versions.get(case_id, [])
        return versions[-1] if versions else None

    def get_version_history(
        self, case_id: str
    ) -> list[GoldStandardVersion]:
        """Get complete version history.

        Parameters
        ----------
        case_id : str
            Case identifier.

        Returns
        -------
        list[GoldStandardVersion]
            All versions chronologically.
        """
        return self._versions.get(case_id, []).copy()

    def revert_to_version(
        self,
        case_id: str,
        version_num: int,
        reverted_by: str,
    ) -> GoldStandardVersion | None:
        """Revert to a previous version.

        Parameters
        ----------
        case_id : str
            Case identifier.
        version_num : int
            Version number to revert to.
        reverted_by : str
            Expert performing reversion.

        Returns
        -------
        GoldStandardVersion or None
            New version (reversion) if successful.
        """
        versions = self._versions.get(case_id, [])

        target = None
        for v in versions:
            if v.version == version_num:
                target = v
                break

        if not target:
            return None

        return self.create_version(
            case_id=case_id,
            label=target.label,
            updated_by=reverted_by,
            change_reason=f"Reverted to version {version_num}",
        )


@dataclass(slots=True)
class ComplianceReporter:
    """Generates compliance reports for audits.

    Creates comprehensive reports documenting
    annotation protocol compliance.
    """

    output_dir: Path

    def __post_init__(self) -> None:
        """Initialize reporter."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_expert_summary(
        self,
        expert_profiles: Sequence[ExpertProfile],
        certifications: dict[str, CertificationRecord | None],
        proficiency_scores: dict[str, ProficiencyScore | None],
    ) -> dict[str, Any]:
        """Generate expert summary report.

        Parameters
        ----------
        expert_profiles : Sequence[ExpertProfile]
            All expert profiles.
        certifications : dict[str, CertificationRecord | None]
            Expert certifications.
        proficiency_scores : dict[str, ProficiencyScore | None]
            Expert proficiency scores.

        Returns
        -------
        dict[str, Any]
            Summary report data.
        """
        now = datetime.now()

        return {
            "report_date": now.isoformat(),
            "total_experts": len(expert_profiles),
            "by_level": self._count_by_level(expert_profiles),
            "by_calibration_status": self._count_by_calibration(
                expert_profiles
            ),
            "certification_summary": {
                "certified": sum(
                    1 for c in certifications.values() if c is not None
                ),
                "uncertified": sum(
                    1 for c in certifications.values() if c is None
                ),
            },
            "proficiency_summary": self._summarize_proficiency(
                proficiency_scores
            ),
        }

    def _count_by_level(
        self, profiles: Sequence[ExpertProfile]
    ) -> dict[str, int]:
        """Count experts by level."""
        counts: dict[str, int] = defaultdict(int)
        for p in profiles:
            counts[p.level.name] += 1
        return dict(counts)

    def _count_by_calibration(
        self, profiles: Sequence[ExpertProfile]
    ) -> dict[str, int]:
        """Count experts by calibration status."""
        counts: dict[str, int] = defaultdict(int)
        for p in profiles:
            counts[p.calibration_status.name] += 1
        return dict(counts)

    def _summarize_proficiency(
        self, scores: dict[str, ProficiencyScore | None]
    ) -> dict[str, float]:
        """Summarize proficiency scores."""
        valid_scores = [s for s in scores.values() if s is not None]

        if not valid_scores:
            return {
                "avg_overall": 0.0,
                "avg_accuracy": 0.0,
                "avg_consistency": 0.0,
            }

        return {
            "avg_overall": statistics.mean(
                s.overall_score for s in valid_scores
            ),
            "avg_accuracy": statistics.mean(
                s.accuracy for s in valid_scores
            ),
            "avg_consistency": statistics.mean(
                s.consistency for s in valid_scores
            ),
        }

    def write_report(
        self,
        report_data: dict[str, Any],
        filename: str = "compliance_report.json",
    ) -> Path:
        """Write report to file.

        Parameters
        ----------
        report_data : dict[str, Any]
            Report data.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to written file.
        """
        output_path = self.output_dir / filename

        with output_path.open("w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info("Wrote compliance report to %s", output_path)
        return output_path


__all__ = [
    "ExpertLevel",
    "CalibrationStatus",
    "DisagreementType",
    "ResolutionMethod",
    "AnnotationConfidence",
    "ExpertProfile",
    "CalibrationResult",
    "AnnotationGuideline",
    "DisagreementCase",
    "GoldStandardCase",
    "ExpertVote",
    "NotificationService",
    "MolecularConfirmation",
    "CalibrationCriteria",
    "AnnotationSchema",
    "ExpertRegistry",
    "CalibrationManager",
    "GuidelineRepository",
    "DisagreementResolver",
    "GoldStandardCurator",
    "ExpertAnnotationProtocol",
    "create_annotation_protocol",
    "TrainingModuleStatus",
    "CertificationLevel",
    "AuditAction",
    "ProficiencyTrend",
    "TrainingModule",
    "TrainingCompletion",
    "ProficiencyScore",
    "AuditRecord",
    "CertificationRecord",
    "DisagreementPattern",
    "ConsensusVote",
    "GoldStandardVersion",
    "TrainingTracker",
    "ProficiencyTracker",
    "CertificationManager",
    "AuditTrail",
    "DisagreementPatternAnalyzer",
    "ConsensusProtocol",
    "GoldStandardVersionManager",
    "ComplianceReporter",
]
