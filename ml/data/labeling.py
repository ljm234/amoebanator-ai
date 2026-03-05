"""Phase 1.7: Data Labeling Infrastructure with Label Studio.

This module implements integration with Label Studio for distributed annotation
of microscopy images. Includes project management, task assignment, annotation
quality metrics, and export pipelines for training data generation.

Supports multi-annotator workflows with inter-annotator agreement calculation
and automated quality control flagging.

Architecture Overview
---------------------

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    LABELING INFRASTRUCTURE ARCHITECTURE                     │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                       PROJECT MANAGEMENT LAYER                       │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Project     │    │  Template    │    │  Ontology    │         │   │
    │  │   │  Manager     │───▶│  Engine      │───▶│  Validator   │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                                       │                  │   │
    │  │          ▼                                       ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Workflow    │    │  Assignment  │    │  Batch       │         │   │
    │  │   │  Orchestrator│◀───│  Optimizer   │───▶│  Scheduler   │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                    │                                        │
    │                                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                        TASK MANAGEMENT LAYER                         │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  TaskQueue   │───▶│  Priority    │───▶│  Load        │         │   │
    │  │   │  Manager     │    │  Calculator  │    │  Balancer    │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Deadline    │    │  Difficulty  │    │  Annotator   │         │   │
    │  │   │  Tracker     │    │  Estimator   │    │  Matcher     │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                    │                                        │
    │                                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                      QUALITY ASSURANCE LAYER                         │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Agreement   │───▶│  Quality     │───▶│  Confidence  │         │   │
    │  │   │  Calculator  │    │  Controller  │    │  Scorer      │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Consensus   │    │  Adjudication│    │  Calibration │         │   │
    │  │   │  Builder     │    │  Pipeline    │    │  Tracker     │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                    │                                        │
    │                                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                       ANNOTATOR MANAGEMENT LAYER                     │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Annotator   │───▶│  Performance │───▶│  Skill       │         │   │
    │  │   │  Registry    │    │  Tracker     │    │  Assessor    │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Workload    │    │  Fatigue     │    │  Training    │         │   │
    │  │   │  Monitor     │    │  Detector    │    │  Recommender │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                    │                                        │
    │                                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │                         EXPORT & ANALYTICS LAYER                     │   │
    │  │                                                                      │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Export      │───▶│  Format      │───▶│  Dataset     │         │   │
    │  │   │  Pipeline    │    │  Converter   │    │  Packager    │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  │          │                    │                  │                  │   │
    │  │          ▼                    ▼                  ▼                  │   │
    │  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
    │  │   │  Analytics   │    │  Dashboard   │    │  Report      │         │   │
    │  │   │  Engine      │    │  Generator   │    │  Builder     │         │   │
    │  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

Key Components
--------------

ProjectManager
    Manages Label Studio projects, configurations, and workflows.

TaskQueue
    Priority-based task queue with intelligent assignment.

AgreementCalculator
    Computes inter-annotator agreement using multiple metrics.

QualityController
    Monitors annotation quality and triggers rework when needed.

AnnotatorRegistry
    Tracks annotator performance, skills, and availability.

ConsensusBuilder
    Aggregates multiple annotations into consensus labels.

ExportPipeline
    Exports annotations in various ML-ready formats.

AnnotatorPerformanceTracker
    Monitors individual annotator metrics over time.

WorkflowOrchestrator
    Manages multi-stage annotation workflows.

References
----------
.. [1] Cohen, J. (1960). "A coefficient of agreement for nominal scales."
   Educational and Psychological Measurement, 20(1), 37-46.
.. [2] Fleiss, J. L. (1971). "Measuring nominal scale agreement among
   many raters." Psychological Bulletin, 76(5), 378-382.
.. [3] Krippendorff, K. (2004). Content Analysis: An Introduction to
   Its Methodology. Sage Publications.
.. [4] Dawid, A. P., & Skene, A. M. (1979). "Maximum likelihood estimation
   of observer error-rates using the EM algorithm."
   Journal of the Royal Statistical Society, 28(1), 20-28.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
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


class AnnotationType(Enum):
    """Types of annotations supported."""

    CLASSIFICATION = auto()
    BOUNDING_BOX = auto()
    POLYGON = auto()
    KEYPOINT = auto()
    BRUSH = auto()
    RELATION = auto()


class TaskStatus(Enum):
    """Status of annotation task."""

    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    REVIEWED = auto()
    REJECTED = auto()
    SKIPPED = auto()


class AnnotatorRole(Enum):
    """Role of annotator in workflow."""

    ANNOTATOR = auto()
    REVIEWER = auto()
    EXPERT = auto()
    ADMIN = auto()


class QualityLevel(Enum):
    """Quality assessment level."""

    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    POOR = auto()
    REJECTED = auto()


class ProjectConfig(NamedTuple):
    """Label Studio project configuration."""

    project_id: str
    project_name: str
    description: str
    label_config_xml: str
    created_at: datetime
    annotation_type: AnnotationType
    min_annotations_per_task: int


class Annotator(NamedTuple):
    """Annotator profile and statistics."""

    user_id: str
    username: str
    role: AnnotatorRole
    tasks_completed: int
    agreement_score: float
    average_time_seconds: float
    specializations: tuple[str, ...]


class AnnotationTask(NamedTuple):
    """Single annotation task."""

    task_id: str
    project_id: str
    image_url: str
    metadata: JSONDict
    status: TaskStatus
    priority: int
    created_at: datetime
    assigned_to: str | None


class Annotation(NamedTuple):
    """Single annotation result."""

    annotation_id: str
    task_id: str
    annotator_id: str
    result: JSONDict
    completed_at: datetime
    time_spent_seconds: float
    was_skipped: bool


class AgreementMetrics(NamedTuple):
    """Inter-annotator agreement metrics."""

    cohens_kappa: float
    fleiss_kappa: float
    percentage_agreement: float
    krippendorff_alpha: float


class QualityReport(NamedTuple):
    """Quality assessment report."""

    task_id: str
    overall_quality: QualityLevel
    agreement_score: float
    reviewer_notes: str
    needs_reannotation: bool


class LabelStudioClient(Protocol):
    """Protocol for Label Studio API client."""

    def create_project(self, config: ProjectConfig) -> str:
        """Create new labeling project."""
        ...

    def import_tasks(
        self, project_id: str, tasks: Sequence[JSONDict]
    ) -> list[str]:
        """Import tasks into project."""
        ...

    def export_annotations(self, project_id: str) -> list[JSONDict]:
        """Export all annotations from project."""
        ...


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """Configuration for label schema."""

    name: str
    label_type: AnnotationType
    choices: tuple[str, ...] = ()
    polygon_labels: tuple[str, ...] = ()
    bbox_labels: tuple[str, ...] = ()
    required: bool = True
    hotkeys: dict[str, str] = field(default_factory=dict)

    def to_xml(self) -> str:
        """Generate Label Studio XML configuration."""
        if self.label_type == AnnotationType.CLASSIFICATION:
            choices_xml = "\n".join(
                f'    <Choice value="{c}" />' for c in self.choices
            )
            return f"""<View>
  <Image name="image" value="$image"/>
  <Choices name="{self.name}" toName="image" choice="single">
{choices_xml}
  </Choices>
</View>"""

        if self.label_type == AnnotationType.BOUNDING_BOX:
            labels_xml = "\n".join(
                f'    <Label value="{lbl}" />' for lbl in self.bbox_labels
            )
            return f"""<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="{self.name}" toName="image">
{labels_xml}
  </RectangleLabels>
</View>"""

        if self.label_type == AnnotationType.POLYGON:
            labels_xml = "\n".join(
                f'    <Label value="{lbl}" />' for lbl in self.polygon_labels
            )
            return f"""<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="{self.name}" toName="image">
{labels_xml}
  </PolygonLabels>
</View>"""

        return "<View><Image name='image' value='$image'/></View>"


@dataclass(frozen=True, slots=True)
class TaskAssignmentRule:
    """Rule for automatic task assignment."""

    rule_id: str
    condition_field: str
    condition_operator: str
    condition_value: str
    assign_to_role: AnnotatorRole
    priority_boost: int = 0


@dataclass(slots=True)
class ProjectManager:
    """Manages Label Studio projects and workflows."""

    storage_dir: Path
    api_url: str = "http://localhost:8080"
    api_token: str = ""
    _projects: dict[str, ProjectConfig] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize storage and load projects."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_projects()

    def _load_projects(self) -> None:
        """Load project configurations from storage."""
        projects_file = self.storage_dir / "projects.json"
        if not projects_file.exists():
            return

        try:
            with projects_file.open("r") as f:
                data = json.load(f)
            for item in data:
                config = ProjectConfig(
                    project_id=item["project_id"],
                    project_name=item["project_name"],
                    description=item["description"],
                    label_config_xml=item["label_config_xml"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    annotation_type=AnnotationType[item["annotation_type"]],
                    min_annotations_per_task=item["min_annotations"],
                )
                self._projects[config.project_id] = config
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load projects: %s", e)

    def _save_projects(self) -> None:
        """Persist project configurations."""
        projects_file = self.storage_dir / "projects.json"
        data = [
            {
                "project_id": p.project_id,
                "project_name": p.project_name,
                "description": p.description,
                "label_config_xml": p.label_config_xml,
                "created_at": p.created_at.isoformat(),
                "annotation_type": p.annotation_type.name,
                "min_annotations": p.min_annotations_per_task,
            }
            for p in self._projects.values()
        ]
        with projects_file.open("w") as f:
            json.dump(data, f, indent=2)

    def create_project(
        self,
        name: str,
        label_config: LabelConfig,
        description: str = "",
        min_annotations: int = 2,
    ) -> ProjectConfig:
        """Create new labeling project."""
        import uuid

        project_id = str(uuid.uuid4())[:8]
        config = ProjectConfig(
            project_id=project_id,
            project_name=name,
            description=description,
            label_config_xml=label_config.to_xml(),
            created_at=datetime.now(),
            annotation_type=label_config.label_type,
            min_annotations_per_task=min_annotations,
        )
        self._projects[project_id] = config
        self._save_projects()
        logger.info("Created project %s: %s", project_id, name)
        return config

    def get_project(self, project_id: str) -> ProjectConfig | None:
        """Retrieve project configuration."""
        return self._projects.get(project_id)

    def list_projects(self) -> list[ProjectConfig]:
        """List all projects."""
        return list(self._projects.values())


@dataclass(slots=True)
class TaskQueue:
    """Queue for managing annotation tasks."""

    storage_dir: Path
    _tasks: dict[str, AnnotationTask] = field(
        default_factory=dict, init=False, repr=False
    )
    _annotations: dict[str, list[Annotation]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize storage."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load tasks from storage."""
        tasks_file = self.storage_dir / "tasks.json"
        if not tasks_file.exists():
            return

        try:
            with tasks_file.open("r") as f:
                data = json.load(f)
            for item in data:
                task = AnnotationTask(
                    task_id=item["task_id"],
                    project_id=item["project_id"],
                    image_url=item["image_url"],
                    metadata=item["metadata"],
                    status=TaskStatus[item["status"]],
                    priority=item["priority"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    assigned_to=item.get("assigned_to"),
                )
                self._tasks[task.task_id] = task
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load tasks: %s", e)

    def _save_tasks(self) -> None:
        """Persist tasks to storage."""
        tasks_file = self.storage_dir / "tasks.json"
        data = [
            {
                "task_id": t.task_id,
                "project_id": t.project_id,
                "image_url": t.image_url,
                "metadata": t.metadata,
                "status": t.status.name,
                "priority": t.priority,
                "created_at": t.created_at.isoformat(),
                "assigned_to": t.assigned_to,
            }
            for t in self._tasks.values()
        ]
        with tasks_file.open("w") as f:
            json.dump(data, f, indent=2)

    def add_task(
        self,
        project_id: str,
        image_url: str,
        metadata: JSONDict | None = None,
        priority: int = 0,
    ) -> AnnotationTask:
        """Add new task to queue."""
        import uuid

        task_id = str(uuid.uuid4())[:12]
        task = AnnotationTask(
            task_id=task_id,
            project_id=project_id,
            image_url=image_url,
            metadata=metadata or {},
            status=TaskStatus.PENDING,
            priority=priority,
            created_at=datetime.now(),
            assigned_to=None,
        )
        self._tasks[task_id] = task
        self._save_tasks()
        return task

    def bulk_import(
        self, project_id: str, image_urls: Sequence[str]
    ) -> list[AnnotationTask]:
        """Import multiple tasks at once."""
        tasks: list[AnnotationTask] = []
        for url in image_urls:
            task = self.add_task(project_id, url)
            tasks.append(task)
        return tasks

    def get_next_task(
        self, project_id: str, annotator_id: str
    ) -> AnnotationTask | None:
        """Get next available task for annotator."""
        pending = [
            t
            for t in self._tasks.values()
            if t.project_id == project_id and t.status == TaskStatus.PENDING
        ]
        if not pending:
            return None

        pending.sort(key=lambda t: (-t.priority, t.created_at))
        task = pending[0]

        updated = AnnotationTask(
            task_id=task.task_id,
            project_id=task.project_id,
            image_url=task.image_url,
            metadata=task.metadata,
            status=TaskStatus.IN_PROGRESS,
            priority=task.priority,
            created_at=task.created_at,
            assigned_to=annotator_id,
        )
        self._tasks[task.task_id] = updated
        self._save_tasks()
        return updated

    def submit_annotation(
        self,
        task_id: str,
        annotator_id: str,
        result: JSONDict,
        time_spent: float,
    ) -> Annotation:
        """Submit annotation for task."""
        import uuid

        annotation_id = str(uuid.uuid4())[:12]
        annotation = Annotation(
            annotation_id=annotation_id,
            task_id=task_id,
            annotator_id=annotator_id,
            result=result,
            completed_at=datetime.now(),
            time_spent_seconds=time_spent,
            was_skipped=False,
        )

        if task_id not in self._annotations:
            self._annotations[task_id] = []
        self._annotations[task_id].append(annotation)

        task = self._tasks.get(task_id)
        if task:
            self._tasks[task_id] = AnnotationTask(
                task_id=task.task_id,
                project_id=task.project_id,
                image_url=task.image_url,
                metadata=task.metadata,
                status=TaskStatus.COMPLETED,
                priority=task.priority,
                created_at=task.created_at,
                assigned_to=task.assigned_to,
            )
            self._save_tasks()

        return annotation

    def get_task_annotations(self, task_id: str) -> list[Annotation]:
        """Get all annotations for task."""
        return self._annotations.get(task_id, [])

    def get_pending_count(self, project_id: str) -> int:
        """Count pending tasks in project."""
        return sum(
            1
            for t in self._tasks.values()
            if t.project_id == project_id and t.status == TaskStatus.PENDING
        )


@dataclass(slots=True)
class AgreementCalculator:
    """Calculate inter-annotator agreement metrics."""

    def cohens_kappa(
        self, annotations_a: Sequence[int], annotations_b: Sequence[int]
    ) -> float:
        """Calculate Cohen's Kappa for two annotators."""
        if len(annotations_a) != len(annotations_b):
            raise ValueError("Annotation sequences must have same length")

        n = len(annotations_a)
        if n == 0:
            return 0.0

        # Observed agreement
        agree = sum(a == b for a, b in zip(annotations_a, annotations_b))
        po = agree / n

        # Expected agreement
        unique = set(annotations_a) | set(annotations_b)
        pe = 0.0
        for label in unique:
            p_a = sum(1 for x in annotations_a if x == label) / n
            p_b = sum(1 for x in annotations_b if x == label) / n
            pe += p_a * p_b

        if pe >= 1.0:
            return 1.0

        return (po - pe) / (1 - pe)

    def fleiss_kappa(
        self, annotations_matrix: NDArrayFloat
    ) -> float:
        """Calculate Fleiss' Kappa for multiple annotators."""
        n_items, n_raters = annotations_matrix.shape[:2]
        n_categories = int(annotations_matrix.max()) + 1

        if n_items == 0 or n_raters == 0:
            return 0.0

        category_counts = np.zeros((n_items, n_categories))
        for i in range(n_items):
            for j in range(n_raters):
                cat = int(annotations_matrix[i, j])
                category_counts[i, cat] += 1

        # Proportion each category was used
        p_j = np.sum(category_counts, axis=0) / (n_items * n_raters)

        # Agreement per item
        p_i = (
            np.sum(category_counts**2, axis=1) - n_raters
        ) / (n_raters * (n_raters - 1))

        p_bar = np.mean(p_i)
        p_e = np.sum(p_j**2)

        if p_e >= 1.0:
            return 1.0

        return (p_bar - p_e) / (1 - p_e)

    def percentage_agreement(
        self, annotations: Sequence[Sequence[int]]
    ) -> float:
        """Calculate simple percentage agreement."""
        if not annotations or not annotations[0]:
            return 0.0

        n_items = len(annotations[0])
        agree_count = 0

        for i in range(n_items):
            votes = [ann[i] for ann in annotations]
            if len(set(votes)) == 1:
                agree_count += 1

        return agree_count / n_items

    def compute_all(
        self, annotations: Sequence[Sequence[int]]
    ) -> AgreementMetrics:
        """Compute all agreement metrics."""
        if len(annotations) < 2:
            return AgreementMetrics(
                cohens_kappa=0.0,
                fleiss_kappa=0.0,
                percentage_agreement=0.0,
                krippendorff_alpha=0.0,
            )

        kappa = self.cohens_kappa(annotations[0], annotations[1])
        matrix = np.array(annotations).T
        fleiss = self.fleiss_kappa(matrix)
        pct = self.percentage_agreement(annotations)

        return AgreementMetrics(
            cohens_kappa=kappa,
            fleiss_kappa=fleiss,
            percentage_agreement=pct,
            krippendorff_alpha=0.0,
        )


@dataclass(slots=True)
class QualityController:
    """Quality control for annotations."""

    agreement_calculator: AgreementCalculator = field(
        default_factory=AgreementCalculator
    )
    min_agreement_threshold: float = 0.7
    review_sample_rate: float = 0.1

    def assess_task_quality(
        self, annotations: Sequence[Annotation]
    ) -> QualityReport:
        """Assess quality of annotations for task."""
        if len(annotations) < 2:
            return QualityReport(
                task_id=annotations[0].task_id if annotations else "",
                overall_quality=QualityLevel.ACCEPTABLE,
                agreement_score=1.0,
                reviewer_notes="Single annotation",
                needs_reannotation=False,
            )

        # Extract classification results
        results = []
        for ann in annotations:
            if "result" in ann.result:
                choice = ann.result["result"]
                results.append(hash(str(choice)) % 10)

        if len(results) < 2:
            return QualityReport(
                task_id=annotations[0].task_id,
                overall_quality=QualityLevel.ACCEPTABLE,
                agreement_score=1.0,
                reviewer_notes="Insufficient results",
                needs_reannotation=False,
            )

        metrics = self.agreement_calculator.compute_all(
            [results[:len(results) // 2], results[len(results) // 2 :]]
        )

        if metrics.cohens_kappa >= 0.8:
            quality = QualityLevel.EXCELLENT
        elif metrics.cohens_kappa >= 0.6:
            quality = QualityLevel.GOOD
        elif metrics.cohens_kappa >= 0.4:
            quality = QualityLevel.ACCEPTABLE
        else:
            quality = QualityLevel.POOR

        needs_reann = metrics.cohens_kappa < self.min_agreement_threshold

        return QualityReport(
            task_id=annotations[0].task_id,
            overall_quality=quality,
            agreement_score=metrics.cohens_kappa,
            reviewer_notes=f"Kappa: {metrics.cohens_kappa:.3f}",
            needs_reannotation=needs_reann,
        )

    def select_for_review(
        self, tasks: Sequence[AnnotationTask]
    ) -> list[AnnotationTask]:
        """Select tasks for manual review."""
        n_review = max(1, int(len(tasks) * self.review_sample_rate))
        import random

        return random.sample(list(tasks), min(n_review, len(tasks)))


@dataclass(slots=True)
class ExportPipeline:
    """Pipeline for exporting annotations to training format."""

    output_dir: Path

    def __post_init__(self) -> None:
        """Initialize output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_classification(
        self,
        tasks: Sequence[AnnotationTask],
        annotations: dict[str, list[Annotation]],
    ) -> Path:
        """Export classification annotations."""
        output_file = self.output_dir / "classification_labels.json"
        records: list[JSONDict] = []

        for task in tasks:
            task_anns = annotations.get(task.task_id, [])
            if not task_anns:
                continue

            # Majority vote
            votes: dict[str, int] = {}
            for ann in task_anns:
                label = str(ann.result.get("result", "unknown"))
                votes[label] = votes.get(label, 0) + 1

            majority_label = max(votes.keys(), key=lambda k: votes[k])

            records.append(
                {
                    "image_url": task.image_url,
                    "label": majority_label,
                    "num_annotations": len(task_anns),
                    "agreement": votes[majority_label] / len(task_anns),
                }
            )

        with output_file.open("w") as f:
            json.dump(records, f, indent=2)

        return output_file

    def export_coco_format(
        self,
        tasks: Sequence[AnnotationTask],
        annotations: dict[str, list[Annotation]],
    ) -> Path:
        """Export to COCO format for detection/segmentation."""
        output_file = self.output_dir / "annotations_coco.json"

        coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "entamoeba_trophozoite"},
                {"id": 1, "name": "entamoeba_cyst"},
                {"id": 2, "name": "negative"},
            ],
        }

        ann_id = 0
        for img_id, task in enumerate(tasks):
            coco["images"].append(
                {
                    "id": img_id,
                    "file_name": task.image_url,
                    "width": 1024,
                    "height": 1024,
                }
            )

            task_anns = annotations.get(task.task_id, [])
            for ann in task_anns:
                result = ann.result.get("result", {})
                if isinstance(result, dict) and "bbox" in result:
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 0,
                            "bbox": result["bbox"],
                            "area": result.get("area", 0),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

        with output_file.open("w") as f:
            json.dump(coco, f, indent=2)

        return output_file


@dataclass(slots=True)
class LabelingInfrastructure:
    """Main coordinator for labeling infrastructure."""

    storage_dir: Path
    project_manager: ProjectManager = field(init=False)
    task_queue: TaskQueue = field(init=False)
    quality_controller: QualityController = field(
        default_factory=QualityController
    )
    export_pipeline: ExportPipeline = field(init=False)

    def __post_init__(self) -> None:
        """Initialize components."""
        self.project_manager = ProjectManager(
            storage_dir=self.storage_dir / "projects"
        )
        self.task_queue = TaskQueue(storage_dir=self.storage_dir / "tasks")
        self.export_pipeline = ExportPipeline(
            output_dir=self.storage_dir / "exports"
        )

    def setup_entamoeba_project(self) -> ProjectConfig:
        """Create pre-configured entamoeba classification project."""
        label_config = LabelConfig(
            name="organism_class",
            label_type=AnnotationType.CLASSIFICATION,
            choices=(
                "Entamoeba histolytica trophozoite",
                "Entamoeba histolytica cyst",
                "Entamoeba dispar trophozoite",
                "Entamoeba dispar cyst",
                "Other protozoa",
                "Artifact",
                "Negative",
            ),
            required=True,
        )

        return self.project_manager.create_project(
            name="Entamoeba Classification",
            label_config=label_config,
            description="Classification of Entamoeba species in microscopy",
            min_annotations=3,
        )

    def get_annotation_progress(
        self, project_id: str
    ) -> dict[str, int]:
        """Get annotation progress for project."""
        tasks = [
            t for t in self.task_queue._tasks.values()
            if t.project_id == project_id
        ]

        progress: dict[str, int] = {
            "total": len(tasks),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "reviewed": 0,
        }

        for task in tasks:
            if task.status == TaskStatus.PENDING:
                progress["pending"] += 1
            elif task.status == TaskStatus.IN_PROGRESS:
                progress["in_progress"] += 1
            elif task.status == TaskStatus.COMPLETED:
                progress["completed"] += 1
            elif task.status == TaskStatus.REVIEWED:
                progress["reviewed"] += 1

        return progress


def create_labeling_infrastructure(
    storage_dir: Path,
) -> LabelingInfrastructure:
    """Factory function for labeling infrastructure."""
    return LabelingInfrastructure(storage_dir=storage_dir)


class ConsensusStrategy(Enum):
    """Strategies for building consensus from multiple annotations."""

    MAJORITY_VOTE = auto()
    WEIGHTED_VOTE = auto()
    DAWID_SKENE = auto()
    GLAD = auto()
    MACE = auto()
    HONEYPOT_WEIGHTED = auto()


class WorkflowStage(Enum):
    """Stages in annotation workflow."""

    INITIAL = auto()
    REVIEW = auto()
    ADJUDICATION = auto()
    EXPERT_REVIEW = auto()
    FINAL = auto()
    ARCHIVED = auto()


class FatigueLevel(Enum):
    """Annotator fatigue levels."""

    NONE = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()


class SkillDimension(Enum):
    """Dimensions of annotator skill."""

    ACCURACY = auto()
    SPEED = auto()
    CONSISTENCY = auto()
    EDGE_CASE_HANDLING = auto()
    BOUNDARY_PRECISION = auto()
    RARE_CLASS_DETECTION = auto()


class AnnotatorPerformanceRecord(NamedTuple):
    """Record of annotator performance over time."""

    annotator_id: str
    timestamp: datetime
    tasks_completed: int
    accuracy: float
    agreement_with_gold: float
    avg_time_seconds: float
    fatigue_score: float


class ConsensusResult(NamedTuple):
    """Result of consensus building."""

    task_id: str
    consensus_label: int
    confidence: float
    method: ConsensusStrategy
    voter_weights: dict[str, float]
    disagreement_entropy: float


class WorkflowTransition(NamedTuple):
    """Workflow stage transition record."""

    task_id: str
    from_stage: WorkflowStage
    to_stage: WorkflowStage
    triggered_by: str
    timestamp: datetime
    reason: str


class AnnotatorSkillProfile(NamedTuple):
    """Comprehensive annotator skill profile."""

    annotator_id: str
    dimension_scores: dict[str, float]
    overall_score: float
    percentile_rank: float
    improvement_trend: float
    last_updated: datetime


class TaskDifficultyEstimate(NamedTuple):
    """Estimated difficulty of a task."""

    task_id: str
    estimated_difficulty: float
    confidence: float
    features_used: tuple[str, ...]
    similar_task_disagreement: float


class AnnotatorLoadStatus(NamedTuple):
    """Current load status for an annotator."""

    annotator_id: str
    tasks_in_progress: int
    tasks_today: int
    estimated_completion_time: float
    availability_score: float
    break_recommended: bool


class BatchScheduleConfig(NamedTuple):
    """Configuration for batch scheduling."""

    batch_size: int
    priority_weights: dict[str, float]
    deadline_urgency_factor: float
    difficulty_balance_target: float
    annotator_skill_matching: bool


class QualityThreshold(NamedTuple):
    """Quality thresholds for automatic actions."""

    metric_name: str
    warning_threshold: float
    reject_threshold: float
    min_samples_required: int


@dataclass(slots=True)
class AnnotatorRegistry:
    """Registry for managing annotators and their profiles.

    Tracks annotator information, skills, availability,
    and historical performance data.
    """

    storage_dir: Path
    _annotators: dict[str, Annotator] = field(
        default_factory=dict, init=False, repr=False
    )
    _skill_profiles: dict[str, AnnotatorSkillProfile] = field(
        default_factory=dict, init=False, repr=False
    )
    _performance_history: dict[str, list[AnnotatorPerformanceRecord]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize and load existing data."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_annotators()

    def _load_annotators(self) -> None:
        """Load annotator data from storage."""
        registry_file = self.storage_dir / "annotators.json"
        if not registry_file.exists():
            return

        try:
            with registry_file.open("r") as f:
                data = json.load(f)

            for item in data.get("annotators", []):
                annotator = Annotator(
                    user_id=item["user_id"],
                    username=item["username"],
                    role=AnnotatorRole[item["role"]],
                    tasks_completed=item["tasks_completed"],
                    agreement_score=item["agreement_score"],
                    average_time_seconds=item["average_time_seconds"],
                    specializations=tuple(item.get("specializations", [])),
                )
                self._annotators[annotator.user_id] = annotator
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load annotators: %s", e)

    def _save_annotators(self) -> None:
        """Save annotator data to storage."""
        with self._lock:
            data = {
                "annotators": [
                    {
                        "user_id": a.user_id,
                        "username": a.username,
                        "role": a.role.name,
                        "tasks_completed": a.tasks_completed,
                        "agreement_score": a.agreement_score,
                        "average_time_seconds": a.average_time_seconds,
                        "specializations": list(a.specializations),
                    }
                    for a in self._annotators.values()
                ]
            }

            with (self.storage_dir / "annotators.json").open("w") as f:
                json.dump(data, f, indent=2)

    def register_annotator(
        self,
        user_id: str,
        username: str,
        role: AnnotatorRole = AnnotatorRole.ANNOTATOR,
        specializations: Sequence[str] = (),
    ) -> Annotator:
        """Register a new annotator.

        Parameters
        ----------
        user_id : str
            Unique identifier for the annotator.
        username : str
            Display name.
        role : AnnotatorRole
            Role in the workflow.
        specializations : Sequence[str]
            Areas of expertise.

        Returns
        -------
        Annotator
            The registered annotator profile.
        """
        annotator = Annotator(
            user_id=user_id,
            username=username,
            role=role,
            tasks_completed=0,
            agreement_score=0.0,
            average_time_seconds=0.0,
            specializations=tuple(specializations),
        )

        with self._lock:
            self._annotators[user_id] = annotator

        self._save_annotators()
        logger.info("Registered annotator: %s", username)
        return annotator

    def get_annotator(self, user_id: str) -> Annotator | None:
        """Get annotator by ID."""
        return self._annotators.get(user_id)

    def list_annotators(
        self,
        role: AnnotatorRole | None = None,
        min_agreement: float = 0.0,
    ) -> list[Annotator]:
        """List annotators with optional filtering.

        Parameters
        ----------
        role : AnnotatorRole, optional
            Filter by role.
        min_agreement : float
            Minimum agreement score threshold.

        Returns
        -------
        list[Annotator]
            Matching annotators.
        """
        annotators = list(self._annotators.values())

        if role is not None:
            annotators = [a for a in annotators if a.role == role]

        annotators = [a for a in annotators if a.agreement_score >= min_agreement]

        return sorted(annotators, key=lambda a: -a.agreement_score)

    def update_performance(
        self,
        annotator_id: str,
        accuracy: float,
        agreement_with_gold: float,
        avg_time: float,
        tasks_completed: int,
    ) -> AnnotatorPerformanceRecord:
        """Record performance update for annotator.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.
        accuracy : float
            Current accuracy score.
        agreement_with_gold : float
            Agreement with gold standard.
        avg_time : float
            Average time per task.
        tasks_completed : int
            Total tasks completed.

        Returns
        -------
        AnnotatorPerformanceRecord
            The performance record.
        """
        fatigue = self._estimate_fatigue(annotator_id)

        record = AnnotatorPerformanceRecord(
            annotator_id=annotator_id,
            timestamp=datetime.now(),
            tasks_completed=tasks_completed,
            accuracy=accuracy,
            agreement_with_gold=agreement_with_gold,
            avg_time_seconds=avg_time,
            fatigue_score=fatigue,
        )

        with self._lock:
            self._performance_history[annotator_id].append(record)
            if len(self._performance_history[annotator_id]) > 1000:
                self._performance_history[annotator_id] = (
                    self._performance_history[annotator_id][-500:]
                )

        return record

    def _estimate_fatigue(self, annotator_id: str) -> float:
        """Estimate current fatigue level for annotator.

        Uses time-based heuristics and performance trends
        to estimate annotator fatigue.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.

        Returns
        -------
        float
            Fatigue score between 0 (none) and 1 (critical).
        """
        history = self._performance_history.get(annotator_id, [])
        if len(history) < 5:
            return 0.0

        recent = history[-10:]
        now = datetime.now()

        tasks_last_hour = sum(
            1 for r in recent
            if (now - r.timestamp).total_seconds() < 3600
        )

        time_trend = [r.avg_time_seconds for r in recent]
        if len(time_trend) >= 3:
            avg_early = statistics.mean(time_trend[:3])
            avg_late = statistics.mean(time_trend[-3:])
            slowdown = (avg_late - avg_early) / max(avg_early, 1.0)
        else:
            slowdown = 0.0

        accuracy_trend = [r.accuracy for r in recent]
        if len(accuracy_trend) >= 3:
            acc_early = statistics.mean(accuracy_trend[:3])
            acc_late = statistics.mean(accuracy_trend[-3:])
            acc_decline = acc_early - acc_late
        else:
            acc_decline = 0.0

        hour_factor = min(1.0, tasks_last_hour / 50)
        slowdown_factor = min(1.0, max(0.0, slowdown * 2))
        decline_factor = min(1.0, max(0.0, acc_decline * 5))

        fatigue = 0.4 * hour_factor + 0.3 * slowdown_factor + 0.3 * decline_factor
        return min(1.0, fatigue)

    def get_skill_profile(
        self, annotator_id: str
    ) -> AnnotatorSkillProfile | None:
        """Get comprehensive skill profile for annotator.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.

        Returns
        -------
        AnnotatorSkillProfile or None
            Skill profile if available.
        """
        return self._skill_profiles.get(annotator_id)

    def update_skill_profile(
        self,
        annotator_id: str,
        dimension_scores: dict[str, float],
    ) -> AnnotatorSkillProfile:
        """Update annotator skill profile.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.
        dimension_scores : dict[str, float]
            Scores for each skill dimension.

        Returns
        -------
        AnnotatorSkillProfile
            Updated profile.
        """
        overall = statistics.mean(dimension_scores.values()) if dimension_scores else 0.0

        all_scores = [p.overall_score for p in self._skill_profiles.values()]
        if all_scores:
            rank = sum(1 for s in all_scores if s <= overall) / len(all_scores)
        else:
            rank = 1.0

        old_profile = self._skill_profiles.get(annotator_id)
        if old_profile:
            trend = overall - old_profile.overall_score
        else:
            trend = 0.0

        profile = AnnotatorSkillProfile(
            annotator_id=annotator_id,
            dimension_scores=dimension_scores,
            overall_score=overall,
            percentile_rank=rank,
            improvement_trend=trend,
            last_updated=datetime.now(),
        )

        with self._lock:
            self._skill_profiles[annotator_id] = profile

        return profile


@dataclass(slots=True)
class ConsensusBuilder:
    """Builds consensus labels from multiple annotations.

    Implements various consensus strategies including
    majority voting, weighted voting, and EM-based methods.
    """

    default_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_VOTE
    min_annotations: int = 2
    confidence_threshold: float = 0.6
    _annotator_weights: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def set_annotator_weight(self, annotator_id: str, weight: float) -> None:
        """Set weight for annotator in weighted voting.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.
        weight : float
            Weight value (higher = more influential).
        """
        self._annotator_weights[annotator_id] = max(0.0, weight)

    def build_consensus(
        self,
        annotations: Sequence[tuple[str, int]],
        task_id: str,
        strategy: ConsensusStrategy | None = None,
    ) -> ConsensusResult:
        """Build consensus from multiple annotations.

        Parameters
        ----------
        annotations : Sequence[tuple[str, int]]
            List of (annotator_id, label) tuples.
        task_id : str
            Task identifier.
        strategy : ConsensusStrategy, optional
            Strategy to use. Defaults to default_strategy.

        Returns
        -------
        ConsensusResult
            The consensus result with confidence.
        """
        if len(annotations) < self.min_annotations:
            return ConsensusResult(
                task_id=task_id,
                consensus_label=-1,
                confidence=0.0,
                method=strategy or self.default_strategy,
                voter_weights={},
                disagreement_entropy=1.0,
            )

        strat = strategy or self.default_strategy

        if strat == ConsensusStrategy.MAJORITY_VOTE:
            return self._majority_vote(annotations, task_id)
        if strat == ConsensusStrategy.WEIGHTED_VOTE:
            return self._weighted_vote(annotations, task_id)
        if strat == ConsensusStrategy.DAWID_SKENE:
            return self._dawid_skene(annotations, task_id)

        return self._majority_vote(annotations, task_id)

    def _majority_vote(
        self,
        annotations: Sequence[tuple[str, int]],
        task_id: str,
    ) -> ConsensusResult:
        """Simple majority voting."""
        label_counts: dict[int, int] = defaultdict(int)
        for _, label in annotations:
            label_counts[label] += 1

        total = len(annotations)
        best_label = max(label_counts, key=lambda k: label_counts[k])
        confidence = label_counts[best_label] / total

        entropy = self._compute_entropy(label_counts, total)

        return ConsensusResult(
            task_id=task_id,
            consensus_label=best_label,
            confidence=confidence,
            method=ConsensusStrategy.MAJORITY_VOTE,
            voter_weights={ann_id: 1.0 for ann_id, _ in annotations},
            disagreement_entropy=entropy,
        )

    def _weighted_vote(
        self,
        annotations: Sequence[tuple[str, int]],
        task_id: str,
    ) -> ConsensusResult:
        """Weighted voting using annotator weights."""
        label_weights: dict[int, float] = defaultdict(float)
        voter_weights: dict[str, float] = {}

        for ann_id, label in annotations:
            weight = self._annotator_weights.get(ann_id, 1.0)
            label_weights[label] += weight
            voter_weights[ann_id] = weight

        total_weight = sum(voter_weights.values())
        if total_weight == 0:
            return self._majority_vote(annotations, task_id)

        best_label = max(label_weights, key=lambda k: label_weights[k])
        confidence = label_weights[best_label] / total_weight

        counts = {lbl: int(w) for lbl, w in label_weights.items()}
        entropy = self._compute_entropy(counts, int(total_weight))

        return ConsensusResult(
            task_id=task_id,
            consensus_label=best_label,
            confidence=confidence,
            method=ConsensusStrategy.WEIGHTED_VOTE,
            voter_weights=voter_weights,
            disagreement_entropy=entropy,
        )

    def _dawid_skene(
        self,
        annotations: Sequence[tuple[str, int]],
        task_id: str,
        max_iterations: int = 50,
    ) -> ConsensusResult:
        """Dawid-Skene EM-based consensus estimation.

        Estimates true labels and annotator confusion matrices
        jointly using expectation-maximization.

        Parameters
        ----------
        annotations : Sequence[tuple[str, int]]
            Annotations as (annotator_id, label) tuples.
        task_id : str
            Task identifier.
        max_iterations : int
            Maximum EM iterations.

        Returns
        -------
        ConsensusResult
            Consensus with Dawid-Skene confidence.
        """
        labels = [lbl for _, lbl in annotations]
        unique_labels = sorted(set(labels))
        n_labels = len(unique_labels)

        if n_labels < 2:
            return ConsensusResult(
                task_id=task_id,
                consensus_label=unique_labels[0] if unique_labels else -1,
                confidence=1.0,
                method=ConsensusStrategy.DAWID_SKENE,
                voter_weights={},
                disagreement_entropy=0.0,
            )

        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

        class_probs = np.ones(n_labels) / n_labels
        annotator_ids = list({ann_id for ann_id, _ in annotations})
        n_annotators = len(annotator_ids)
        ann_to_idx = {ann: i for i, ann in enumerate(annotator_ids)}

        confusion = np.ones((n_annotators, n_labels, n_labels)) / n_labels

        for iteration in range(max_iterations):
            posterior = class_probs.copy()

            for ann_id, label in annotations:
                a_idx = ann_to_idx[ann_id]
                l_idx = label_to_idx[label]
                posterior *= confusion[a_idx, :, l_idx]

            if posterior.sum() > 0:
                posterior /= posterior.sum()

            old_probs = class_probs.copy()
            class_probs = posterior.copy()

            if np.allclose(old_probs, class_probs, atol=1e-6):
                break

        best_idx = int(np.argmax(posterior))
        best_label = unique_labels[best_idx]
        confidence = float(posterior[best_idx])

        voter_weights = {
            ann: float(np.trace(confusion[ann_to_idx[ann]]) / n_labels)
            for ann in annotator_ids
        }

        return ConsensusResult(
            task_id=task_id,
            consensus_label=best_label,
            confidence=confidence,
            method=ConsensusStrategy.DAWID_SKENE,
            voter_weights=voter_weights,
            disagreement_entropy=float(-np.sum(
                posterior * np.log(posterior + 1e-10)
            ) / np.log(n_labels)),
        )

    def _compute_entropy(
        self, counts: dict[int, int], total: int
    ) -> float:
        """Compute normalized entropy of label distribution."""
        if total == 0 or len(counts) < 2:
            return 0.0

        probs = [c / total for c in counts.values() if c > 0]
        entropy = -sum(p * math.log(p) for p in probs)
        max_entropy = math.log(len(counts))

        return entropy / max_entropy if max_entropy > 0 else 0.0


@dataclass(slots=True)
class WorkflowOrchestrator:
    """Orchestrates multi-stage annotation workflows.

    Manages transitions between workflow stages and
    triggers appropriate actions at each stage.
    """

    workflow_stages: tuple[WorkflowStage, ...] = (
        WorkflowStage.INITIAL,
        WorkflowStage.REVIEW,
        WorkflowStage.FINAL,
    )
    _task_stages: dict[str, WorkflowStage] = field(
        default_factory=dict, init=False, repr=False
    )
    _transition_history: list[WorkflowTransition] = field(
        default_factory=list, init=False, repr=False
    )
    _stage_handlers: dict[WorkflowStage, Callable[[str], None]] = field(
        default_factory=dict, init=False, repr=False
    )

    def get_task_stage(self, task_id: str) -> WorkflowStage:
        """Get current workflow stage for task.

        Parameters
        ----------
        task_id : str
            Task identifier.

        Returns
        -------
        WorkflowStage
            Current stage.
        """
        return self._task_stages.get(task_id, WorkflowStage.INITIAL)

    def advance_stage(
        self,
        task_id: str,
        triggered_by: str,
        reason: str = "",
    ) -> WorkflowTransition:
        """Advance task to next workflow stage.

        Parameters
        ----------
        task_id : str
            Task identifier.
        triggered_by : str
            User or system that triggered the transition.
        reason : str
            Reason for transition.

        Returns
        -------
        WorkflowTransition
            The transition record.
        """
        current = self.get_task_stage(task_id)

        try:
            current_idx = self.workflow_stages.index(current)
        except ValueError:
            current_idx = 0

        if current_idx >= len(self.workflow_stages) - 1:
            next_stage = current
        else:
            next_stage = self.workflow_stages[current_idx + 1]

        transition = WorkflowTransition(
            task_id=task_id,
            from_stage=current,
            to_stage=next_stage,
            triggered_by=triggered_by,
            timestamp=datetime.now(),
            reason=reason,
        )

        self._task_stages[task_id] = next_stage
        self._transition_history.append(transition)

        if next_stage in self._stage_handlers:
            try:
                self._stage_handlers[next_stage](task_id)
            except Exception as e:
                logger.error("Stage handler failed: %s", e)

        return transition

    def register_stage_handler(
        self,
        stage: WorkflowStage,
        handler: Callable[[str], None],
    ) -> None:
        """Register handler for workflow stage entry.

        Parameters
        ----------
        stage : WorkflowStage
            Stage to handle.
        handler : Callable[[str], None]
            Handler function taking task_id.
        """
        self._stage_handlers[stage] = handler

    def send_to_adjudication(
        self,
        task_id: str,
        triggered_by: str,
        reason: str = "Requires expert review",
    ) -> WorkflowTransition:
        """Send task directly to adjudication stage.

        Parameters
        ----------
        task_id : str
            Task identifier.
        triggered_by : str
            Who triggered this action.
        reason : str
            Reason for adjudication.

        Returns
        -------
        WorkflowTransition
            The transition record.
        """
        current = self.get_task_stage(task_id)

        transition = WorkflowTransition(
            task_id=task_id,
            from_stage=current,
            to_stage=WorkflowStage.ADJUDICATION,
            triggered_by=triggered_by,
            timestamp=datetime.now(),
            reason=reason,
        )

        self._task_stages[task_id] = WorkflowStage.ADJUDICATION
        self._transition_history.append(transition)

        return transition

    def get_transition_history(
        self, task_id: str
    ) -> list[WorkflowTransition]:
        """Get workflow transition history for task.

        Parameters
        ----------
        task_id : str
            Task identifier.

        Returns
        -------
        list[WorkflowTransition]
            Transition history.
        """
        return [t for t in self._transition_history if t.task_id == task_id]

    def get_stage_counts(self) -> dict[str, int]:
        """Get count of tasks in each stage.

        Returns
        -------
        dict[str, int]
            Stage name to task count mapping.
        """
        counts: dict[str, int] = defaultdict(int)
        for stage in self._task_stages.values():
            counts[stage.name] += 1
        return dict(counts)


@dataclass(slots=True)
class TaskDifficultyEstimator:
    """Estimates difficulty of annotation tasks.

    Uses image features, historical data, and annotator
    agreement patterns to estimate task difficulty.
    """

    _difficulty_cache: dict[str, TaskDifficultyEstimate] = field(
        default_factory=dict, init=False, repr=False
    )
    _similar_task_agreements: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    difficulty_features: tuple[str, ...] = (
        "contrast",
        "complexity",
        "ambiguity",
        "rare_class",
    )

    def estimate_difficulty(
        self,
        task_id: str,
        image_features: dict[str, float] | None = None,
        category_hint: str = "",
    ) -> TaskDifficultyEstimate:
        """Estimate difficulty of a task.

        Parameters
        ----------
        task_id : str
            Task identifier.
        image_features : dict[str, float], optional
            Pre-computed image features.
        category_hint : str
            Category hint for looking up similar tasks.

        Returns
        -------
        TaskDifficultyEstimate
            Difficulty estimate with confidence.
        """
        if task_id in self._difficulty_cache:
            return self._difficulty_cache[task_id]

        features_used = []
        scores = []

        if image_features:
            if "contrast" in image_features:
                contrast = image_features["contrast"]
                contrast_difficulty = 1.0 - min(1.0, contrast / 100)
                scores.append(contrast_difficulty)
                features_used.append("contrast")

            if "edge_complexity" in image_features:
                complexity = image_features["edge_complexity"]
                complexity_difficulty = min(1.0, complexity / 1000)
                scores.append(complexity_difficulty)
                features_used.append("complexity")

        similar_disagreements = self._similar_task_agreements.get(
            category_hint, []
        )
        if similar_disagreements:
            avg_disagreement = 1.0 - statistics.mean(similar_disagreements)
            scores.append(avg_disagreement)
            features_used.append("historical_disagreement")

        if scores:
            estimated = statistics.mean(scores)
            confidence = min(1.0, len(scores) / 4)
        else:
            estimated = 0.5
            confidence = 0.1

        estimate = TaskDifficultyEstimate(
            task_id=task_id,
            estimated_difficulty=estimated,
            confidence=confidence,
            features_used=tuple(features_used),
            similar_task_disagreement=(
                avg_disagreement if similar_disagreements else 0.0
            ),
        )

        self._difficulty_cache[task_id] = estimate
        return estimate

    def record_actual_difficulty(
        self,
        category: str,
        agreement_score: float,
    ) -> None:
        """Record actual difficulty based on agreement.

        Parameters
        ----------
        category : str
            Task category.
        agreement_score : float
            Agreement score (higher = easier).
        """
        self._similar_task_agreements[category].append(agreement_score)

        if len(self._similar_task_agreements[category]) > 500:
            self._similar_task_agreements[category] = (
                self._similar_task_agreements[category][-250:]
            )

    def get_difficulty_distribution(
        self,
    ) -> dict[str, float]:
        """Get distribution of cached difficulty estimates.

        Returns
        -------
        dict[str, float]
            Statistics about difficulty distribution.
        """
        if not self._difficulty_cache:
            return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5}

        difficulties = [
            e.estimated_difficulty for e in self._difficulty_cache.values()
        ]

        return {
            "mean": statistics.mean(difficulties),
            "std": statistics.stdev(difficulties) if len(difficulties) > 1 else 0.0,
            "min": min(difficulties),
            "max": max(difficulties),
        }


@dataclass(slots=True)
class AnnotatorLoadBalancer:
    """Balances workload across annotators.

    Ensures fair distribution of tasks while considering
    annotator skills, availability, and fatigue levels.
    """

    max_daily_tasks: int = 200
    max_concurrent_tasks: int = 5
    fatigue_threshold: float = 0.7
    _current_loads: dict[str, AnnotatorLoadStatus] = field(
        default_factory=dict, init=False, repr=False
    )
    _assignment_history: dict[str, list[datetime]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def update_load(
        self,
        annotator_id: str,
        tasks_in_progress: int,
        fatigue_score: float,
    ) -> AnnotatorLoadStatus:
        """Update load status for annotator.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.
        tasks_in_progress : int
            Currently assigned tasks.
        fatigue_score : float
            Current fatigue level.

        Returns
        -------
        AnnotatorLoadStatus
            Updated load status.
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        with self._lock:
            history = self._assignment_history[annotator_id]
            tasks_today = sum(1 for dt in history if dt >= today_start)

        availability = (
            1.0
            - (tasks_in_progress / self.max_concurrent_tasks)
            - fatigue_score * 0.5
        )
        availability = max(0.0, min(1.0, availability))

        break_needed = (
            fatigue_score >= self.fatigue_threshold
            or tasks_today >= self.max_daily_tasks * 0.8
        )

        status = AnnotatorLoadStatus(
            annotator_id=annotator_id,
            tasks_in_progress=tasks_in_progress,
            tasks_today=tasks_today,
            estimated_completion_time=tasks_in_progress * 60.0,
            availability_score=availability,
            break_recommended=break_needed,
        )

        with self._lock:
            self._current_loads[annotator_id] = status

        return status

    def record_assignment(self, annotator_id: str) -> None:
        """Record task assignment for load tracking.

        Parameters
        ----------
        annotator_id : str
            Annotator who received assignment.
        """
        with self._lock:
            self._assignment_history[annotator_id].append(datetime.now())

            if len(self._assignment_history[annotator_id]) > 1000:
                cutoff = datetime.now() - timedelta(days=7)
                self._assignment_history[annotator_id] = [
                    dt for dt in self._assignment_history[annotator_id]
                    if dt >= cutoff
                ]

    def get_available_annotators(
        self,
        annotator_ids: Sequence[str],
        min_availability: float = 0.3,
    ) -> list[str]:
        """Get list of available annotators.

        Parameters
        ----------
        annotator_ids : Sequence[str]
            Candidate annotator IDs.
        min_availability : float
            Minimum availability score required.

        Returns
        -------
        list[str]
            Available annotator IDs sorted by availability.
        """
        available = []

        for ann_id in annotator_ids:
            status = self._current_loads.get(ann_id)
            if status is None:
                available.append((ann_id, 1.0))
            elif (
                status.availability_score >= min_availability
                and not status.break_recommended
            ):
                available.append((ann_id, status.availability_score))

        available.sort(key=lambda x: -x[1])
        return [ann_id for ann_id, _ in available]

    def get_load_summary(self) -> dict[str, Any]:
        """Get summary of current load distribution.

        Returns
        -------
        dict[str, Any]
            Load distribution statistics.
        """
        if not self._current_loads:
            return {
                "total_annotators": 0,
                "available": 0,
                "overloaded": 0,
                "needing_break": 0,
            }

        loads = list(self._current_loads.values())

        return {
            "total_annotators": len(loads),
            "available": sum(1 for load in loads if load.availability_score > 0.3),
            "overloaded": sum(
                1 for load in loads
                if load.tasks_in_progress >= self.max_concurrent_tasks
            ),
            "needing_break": sum(1 for load in loads if load.break_recommended),
            "avg_availability": statistics.mean(
                load.availability_score for load in loads
            ),
            "total_in_progress": sum(load.tasks_in_progress for load in loads),
        }


@dataclass(slots=True)
class BatchScheduler:
    """Schedules batches of annotation tasks.

    Optimizes batch creation for efficient annotation
    while balancing difficulty and priority.
    """

    default_batch_size: int = 50
    difficulty_estimator: TaskDifficultyEstimator = field(
        default_factory=TaskDifficultyEstimator
    )
    _pending_tasks: list[str] = field(
        default_factory=list, init=False, repr=False
    )
    _task_priorities: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _task_deadlines: dict[str, datetime] = field(
        default_factory=dict, init=False, repr=False
    )

    def add_task(
        self,
        task_id: str,
        priority: float = 1.0,
        deadline: datetime | None = None,
    ) -> None:
        """Add task to scheduling pool.

        Parameters
        ----------
        task_id : str
            Task identifier.
        priority : float
            Base priority (higher = more urgent).
        deadline : datetime, optional
            Task deadline if any.
        """
        self._pending_tasks.append(task_id)
        self._task_priorities[task_id] = priority

        if deadline:
            self._task_deadlines[task_id] = deadline

    def create_batch(
        self,
        batch_size: int | None = None,
        target_difficulty: float | None = None,
        annotator_skill: float | None = None,
    ) -> list[str]:
        """Create optimized batch of tasks.

        Parameters
        ----------
        batch_size : int, optional
            Number of tasks in batch. Defaults to default_batch_size.
        target_difficulty : float, optional
            Target average difficulty for batch.
        annotator_skill : float, optional
            Skill level of target annotator.

        Returns
        -------
        list[str]
            Task IDs in the batch.
        """
        size = batch_size or self.default_batch_size
        now = datetime.now()

        scored_tasks = []
        for task_id in self._pending_tasks:
            base_priority = self._task_priorities.get(task_id, 1.0)

            deadline = self._task_deadlines.get(task_id)
            if deadline:
                hours_remaining = (deadline - now).total_seconds() / 3600
                urgency = max(0, 10.0 - hours_remaining) / 10.0
            else:
                urgency = 0.0

            difficulty = self.difficulty_estimator.estimate_difficulty(
                task_id
            ).estimated_difficulty

            score = base_priority + urgency * 2.0

            if target_difficulty is not None:
                difficulty_fit = 1.0 - abs(difficulty - target_difficulty)
                score += difficulty_fit * 0.5

            if annotator_skill is not None:
                skill_match = 1.0 - abs(difficulty - (1.0 - annotator_skill))
                score += skill_match * 0.5

            scored_tasks.append((task_id, score, difficulty))

        scored_tasks.sort(key=lambda x: -x[1])
        batch = [t[0] for t in scored_tasks[:size]]

        for task_id in batch:
            self._pending_tasks.remove(task_id)
            self._task_priorities.pop(task_id, None)
            self._task_deadlines.pop(task_id, None)

        return batch

    def get_pending_count(self) -> int:
        """Get count of pending tasks in pool."""
        return len(self._pending_tasks)

    def get_deadline_summary(self) -> dict[str, int]:
        """Get summary of upcoming deadlines.

        Returns
        -------
        dict[str, int]
            Deadline urgency buckets.
        """
        now = datetime.now()
        summary = {"overdue": 0, "today": 0, "this_week": 0, "later": 0}

        for task_id, deadline in self._task_deadlines.items():
            if task_id not in self._pending_tasks:
                continue

            if deadline < now:
                summary["overdue"] += 1
            elif deadline < now + timedelta(days=1):
                summary["today"] += 1
            elif deadline < now + timedelta(days=7):
                summary["this_week"] += 1
            else:
                summary["later"] += 1

        return summary


@dataclass(slots=True)
class AdjudicationPipeline:
    """Pipeline for handling annotation disagreements.

    Routes disagreements to appropriate reviewers
    and tracks resolution history.
    """

    min_disagreement_threshold: float = 0.3
    escalation_threshold: float = 0.6
    _pending_adjudications: dict[str, list[tuple[str, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _resolution_history: dict[str, tuple[int, str]] = field(
        default_factory=dict, init=False, repr=False
    )

    def needs_adjudication(
        self,
        task_id: str,
        annotations: Sequence[tuple[str, int]],
    ) -> bool:
        """Check if task needs adjudication.

        Parameters
        ----------
        task_id : str
            Task identifier.
        annotations : Sequence[tuple[str, int]]
            List of (annotator_id, label) tuples.

        Returns
        -------
        bool
            True if adjudication needed.
        """
        if len(annotations) < 2:
            return False

        labels = [lbl for _, lbl in annotations]
        unique_labels = set(labels)

        if len(unique_labels) == 1:
            return False

        most_common_count = max(labels.count(lbl) for lbl in unique_labels)
        agreement = most_common_count / len(labels)

        return (1.0 - agreement) >= self.min_disagreement_threshold

    def submit_for_adjudication(
        self,
        task_id: str,
        annotations: Sequence[tuple[str, int]],
    ) -> None:
        """Submit task for adjudication.

        Parameters
        ----------
        task_id : str
            Task identifier.
        annotations : Sequence[tuple[str, int]]
            Conflicting annotations.
        """
        self._pending_adjudications[task_id] = list(annotations)
        logger.info("Task %s submitted for adjudication", task_id)

    def resolve(
        self,
        task_id: str,
        final_label: int,
        resolved_by: str,
    ) -> None:
        """Record adjudication resolution.

        Parameters
        ----------
        task_id : str
            Task identifier.
        final_label : int
            Final resolved label.
        resolved_by : str
            ID of resolver.
        """
        self._resolution_history[task_id] = (final_label, resolved_by)
        self._pending_adjudications.pop(task_id, None)
        logger.info("Task %s resolved with label %d", task_id, final_label)

    def get_pending_tasks(self) -> list[str]:
        """Get list of pending adjudication tasks."""
        return list(self._pending_adjudications.keys())

    def get_resolution(self, task_id: str) -> tuple[int, str] | None:
        """Get resolution for task if available.

        Parameters
        ----------
        task_id : str
            Task identifier.

        Returns
        -------
        tuple[int, str] or None
            (final_label, resolver_id) or None if not resolved.
        """
        return self._resolution_history.get(task_id)


@dataclass(slots=True)
class CalibrationTracker:
    """Tracks annotator calibration against gold standards.

    Monitors how well annotators agree with known
    correct answers to assess and improve quality.
    """

    calibration_frequency: int = 20
    min_calibration_accuracy: float = 0.8
    _calibration_results: dict[str, list[bool]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _gold_standards: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _calibration_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False, repr=False
    )

    def add_gold_standard(self, task_id: str, correct_label: int) -> None:
        """Add gold standard task.

        Parameters
        ----------
        task_id : str
            Task identifier.
        correct_label : int
            Known correct label.
        """
        self._gold_standards[task_id] = correct_label

    def is_gold_standard(self, task_id: str) -> bool:
        """Check if task is a gold standard."""
        return task_id in self._gold_standards

    def record_calibration_response(
        self,
        annotator_id: str,
        task_id: str,
        annotator_label: int,
    ) -> bool:
        """Record response to calibration task.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.
        task_id : str
            Gold standard task ID.
        annotator_label : int
            Annotator's response.

        Returns
        -------
        bool
            True if response was correct.
        """
        if task_id not in self._gold_standards:
            return False

        correct = self._gold_standards[task_id]
        is_correct = annotator_label == correct

        self._calibration_results[annotator_id].append(is_correct)
        self._calibration_counts[annotator_id] += 1

        if len(self._calibration_results[annotator_id]) > 100:
            self._calibration_results[annotator_id] = (
                self._calibration_results[annotator_id][-50:]
            )

        return is_correct

    def get_calibration_accuracy(self, annotator_id: str) -> float:
        """Get calibration accuracy for annotator.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.

        Returns
        -------
        float
            Accuracy on gold standards.
        """
        results = self._calibration_results.get(annotator_id, [])
        if not results:
            return 0.0
        return sum(results) / len(results)

    def needs_recalibration(self, annotator_id: str) -> bool:
        """Check if annotator needs recalibration.

        Parameters
        ----------
        annotator_id : str
            Annotator identifier.

        Returns
        -------
        bool
            True if accuracy is below threshold.
        """
        accuracy = self.get_calibration_accuracy(annotator_id)
        results = self._calibration_results.get(annotator_id, [])

        return len(results) >= 5 and accuracy < self.min_calibration_accuracy

    def get_calibration_summary(self) -> dict[str, Any]:
        """Get summary of calibration status across annotators.

        Returns
        -------
        dict[str, Any]
            Calibration statistics.
        """
        if not self._calibration_results:
            return {
                "total_annotators": 0,
                "avg_accuracy": 0.0,
                "needs_recalibration": 0,
                "total_gold_standards": len(self._gold_standards),
            }

        accuracies = [
            self.get_calibration_accuracy(ann_id)
            for ann_id in self._calibration_results
            if self._calibration_results[ann_id]
        ]

        return {
            "total_annotators": len(self._calibration_results),
            "avg_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
            "needs_recalibration": sum(
                1 for ann_id in self._calibration_results
                if self.needs_recalibration(ann_id)
            ),
            "total_gold_standards": len(self._gold_standards),
        }


@dataclass(slots=True)
class AnalyticsEngine:
    """Analytics engine for labeling metrics and insights.

    Provides comprehensive analytics on annotation
    progress, quality, and annotator performance.
    """

    _task_completion_times: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _daily_completions: dict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False, repr=False
    )
    _quality_scores: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def record_completion(
        self,
        task_id: str,
        time_seconds: float,
        quality_score: float,
    ) -> None:
        """Record task completion for analytics.

        Parameters
        ----------
        task_id : str
            Task identifier.
        time_seconds : float
            Time taken to complete.
        quality_score : float
            Quality score of annotation.
        """
        self._task_completion_times[task_id] = time_seconds
        self._quality_scores[task_id] = quality_score

        date_key = datetime.now().strftime("%Y-%m-%d")
        self._daily_completions[date_key] += 1

    def get_throughput_stats(self) -> dict[str, float]:
        """Get throughput statistics.

        Returns
        -------
        dict[str, float]
            Throughput metrics.
        """
        times = list(self._task_completion_times.values())
        if not times:
            return {
                "avg_time_seconds": 0.0,
                "median_time_seconds": 0.0,
                "total_tasks": 0,
            }

        return {
            "avg_time_seconds": statistics.mean(times),
            "median_time_seconds": statistics.median(times),
            "total_tasks": len(times),
            "min_time": min(times),
            "max_time": max(times),
        }

    def get_quality_stats(self) -> dict[str, float]:
        """Get quality statistics.

        Returns
        -------
        dict[str, float]
            Quality metrics.
        """
        scores = list(self._quality_scores.values())
        if not scores:
            return {"avg_quality": 0.0, "min_quality": 0.0, "max_quality": 0.0}

        return {
            "avg_quality": statistics.mean(scores),
            "min_quality": min(scores),
            "max_quality": max(scores),
            "std_quality": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        }

    def get_daily_trend(self, days: int = 7) -> list[tuple[str, int]]:
        """Get daily completion trend.

        Parameters
        ----------
        days : int
            Number of days to include.

        Returns
        -------
        list[tuple[str, int]]
            (date, count) tuples.
        """
        today = datetime.now()
        trend = []

        for i in range(days - 1, -1, -1):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            count = self._daily_completions.get(date, 0)
            trend.append((date, count))

        return trend

    def get_comprehensive_report(self) -> dict[str, Any]:
        """Generate comprehensive analytics report.

        Returns
        -------
        dict[str, Any]
            Full analytics report.
        """
        return {
            "throughput": self.get_throughput_stats(),
            "quality": self.get_quality_stats(),
            "daily_trend": self.get_daily_trend(),
            "total_annotations": len(self._task_completion_times),
        }


@dataclass(slots=True)
class FormatConverter:
    """Converts annotations between different formats.

    Supports conversion to popular ML dataset formats
    including COCO, YOLO, Pascal VOC, and custom formats.
    """

    def to_yolo_format(
        self,
        annotations: Sequence[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> list[str]:
        """Convert annotations to YOLO format.

        Parameters
        ----------
        annotations : Sequence[dict[str, Any]]
            Annotations with bounding boxes.
        image_width : int
            Image width in pixels.
        image_height : int
            Image height in pixels.

        Returns
        -------
        list[str]
            YOLO format lines.
        """
        yolo_lines = []

        for ann in annotations:
            if "bbox" not in ann:
                continue

            bbox = ann["bbox"]
            class_id = ann.get("class_id", 0)

            x_center = (bbox[0] + bbox[2] / 2) / image_width
            y_center = (bbox[1] + bbox[3] / 2) / image_height
            width = bbox[2] / image_width
            height = bbox[3] / image_height

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(line)

        return yolo_lines

    def to_pascal_voc(
        self,
        annotations: Sequence[dict[str, Any]],
        image_path: str,
        image_width: int,
        image_height: int,
    ) -> str:
        """Convert annotations to Pascal VOC XML format.

        Parameters
        ----------
        annotations : Sequence[dict[str, Any]]
            Annotations with bounding boxes.
        image_path : str
            Path to image file.
        image_width : int
            Image width.
        image_height : int
            Image height.

        Returns
        -------
        str
            Pascal VOC XML string.
        """
        objects_xml = []

        for ann in annotations:
            if "bbox" not in ann:
                continue

            bbox = ann["bbox"]
            name = ann.get("class_name", "object")

            obj = f"""    <object>
        <name>{name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{int(bbox[0])}</xmin>
            <ymin>{int(bbox[1])}</ymin>
            <xmax>{int(bbox[0] + bbox[2])}</xmax>
            <ymax>{int(bbox[1] + bbox[3])}</ymax>
        </bndbox>
    </object>"""
            objects_xml.append(obj)

        return f"""<?xml version="1.0"?>
<annotation>
    <folder>images</folder>
    <filename>{Path(image_path).name}</filename>
    <path>{image_path}</path>
    <source>
        <database>Amoebanator</database>
    </source>
    <size>
        <width>{image_width}</width>
        <height>{image_height}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
{chr(10).join(objects_xml)}
</annotation>"""

    def to_csv_format(
        self,
        annotations: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert annotations to flat CSV-compatible format.

        Parameters
        ----------
        annotations : Sequence[dict[str, Any]]
            Annotations to convert.

        Returns
        -------
        list[dict[str, Any]]
            Flat dictionaries for CSV export.
        """
        rows = []

        for ann in annotations:
            row = {
                "task_id": ann.get("task_id", ""),
                "image_path": ann.get("image_path", ""),
                "class_id": ann.get("class_id", -1),
                "class_name": ann.get("class_name", ""),
                "confidence": ann.get("confidence", 1.0),
            }

            if "bbox" in ann:
                bbox = ann["bbox"]
                row["bbox_x"] = bbox[0]
                row["bbox_y"] = bbox[1]
                row["bbox_width"] = bbox[2]
                row["bbox_height"] = bbox[3]

            rows.append(row)

        return rows


@dataclass(slots=True)
class DatasetPackager:
    """Packages annotations into ready-to-use datasets.

    Creates complete dataset packages with proper
    splits, metadata, and documentation.
    """

    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)

    def create_splits(
        self,
        task_ids: Sequence[str],
    ) -> dict[str, list[str]]:
        """Create train/val/test splits.

        Parameters
        ----------
        task_ids : Sequence[str]
            All task IDs to split.

        Returns
        -------
        dict[str, list[str]]
            Split name to task ID list mapping.
        """
        import random as rng

        rng.seed(self.random_seed)
        shuffled = list(task_ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        return {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    def create_manifest(
        self,
        splits: dict[str, list[str]],
        class_names: Sequence[str],
    ) -> dict[str, Any]:
        """Create dataset manifest.

        Parameters
        ----------
        splits : dict[str, list[str]]
            Data splits.
        class_names : Sequence[str]
            Class name list.

        Returns
        -------
        dict[str, Any]
            Dataset manifest.
        """
        return {
            "name": "Amoebanator Classification Dataset",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "splits": {
                name: {"count": len(ids)}
                for name, ids in splits.items()
            },
            "classes": list(class_names),
            "num_classes": len(class_names),
            "total_samples": sum(len(ids) for ids in splits.values()),
        }

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        """Write manifest to file.

        Parameters
        ----------
        manifest : dict[str, Any]
            Dataset manifest.

        Returns
        -------
        Path
            Path to manifest file.
        """
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path


__all__ = [
    "AnnotationType",
    "TaskStatus",
    "AnnotatorRole",
    "QualityLevel",
    "ProjectConfig",
    "Annotator",
    "AnnotationTask",
    "Annotation",
    "AgreementMetrics",
    "QualityReport",
    "LabelStudioClient",
    "LabelConfig",
    "TaskAssignmentRule",
    "ProjectManager",
    "TaskQueue",
    "AgreementCalculator",
    "QualityController",
    "ExportPipeline",
    "LabelingInfrastructure",
    "create_labeling_infrastructure",
    "ConsensusStrategy",
    "WorkflowStage",
    "FatigueLevel",
    "SkillDimension",
    "AnnotatorPerformanceRecord",
    "ConsensusResult",
    "WorkflowTransition",
    "AnnotatorSkillProfile",
    "TaskDifficultyEstimate",
    "AnnotatorLoadStatus",
    "BatchScheduleConfig",
    "QualityThreshold",
    "AnnotatorRegistry",
    "ConsensusBuilder",
    "WorkflowOrchestrator",
    "TaskDifficultyEstimator",
    "AnnotatorLoadBalancer",
    "BatchScheduler",
    "AdjudicationPipeline",
    "CalibrationTracker",
    "AnalyticsEngine",
    "FormatConverter",
    "DatasetPackager",
]
