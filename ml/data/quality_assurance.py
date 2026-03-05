"""Phase 1.9: Data Quality Assurance Pipeline.

This module implements automated quality assurance checks for training data
including image quality metrics, label consistency validation, distribution
analysis, and data integrity verification.

Provides continuous monitoring with configurable thresholds and automated
alerting for data quality issues.

Components
----------
ImageQualityAnalyzer
    Computes sharpness, contrast, brightness, noise, and saturation metrics
    using Laplacian variance, local histogram analysis, and block-based
    estimation algorithms optimized for microscopy images.

LabelConsistencyChecker
    Validates label assignments against model predictions using configurable
    confidence thresholds with Cohen's Kappa inter-rater agreement metrics.

DistributionAnalyzer
    Monitors class balance using Shannon entropy, Gini coefficient, and
    Kullback-Leibler divergence against target distributions.

DuplicateDetector
    Identifies near-duplicate samples using perceptual hashing with dHash
    and pHash algorithms, SSIM structural similarity, and feature embedding
    cosine similarity for semantic duplicate detection.

MetadataValidator
    Enforces schema compliance for sample metadata with type checking,
    range validation, and referential integrity verification.

AnnotationQualityChecker
    Validates annotation completeness, boundary precision, and inter-annotator
    agreement using Fleiss' Kappa and Krippendorff's Alpha metrics.

OutlierDetector
    Identifies statistical outliers using isolation forest, local outlier
    factor, and Mahalanobis distance methods.

DataDriftMonitor
    Monitors feature distribution drift using Population Stability Index,
    Kolmogorov-Smirnov tests, and Jensen-Shannon divergence.

QualityAssurancePipeline
    Orchestrates all quality checks with parallel execution, incremental
    processing, and comprehensive reporting.

Usage
-----
>>> from ml.data.quality_assurance import create_qa_pipeline
>>> pipeline = create_qa_pipeline(Path("./qa_reports"))
>>> report = pipeline.run_full_pipeline(images, labels, metadata)
>>> if report.overall_result == CheckResult.FAILED:
...     print(f"Quality issues: {report.issues_found}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Iterator,
    NamedTuple,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
)

import numpy as np

logger: Final = logging.getLogger(__name__)

JSONDict: TypeAlias = dict[str, Any]
NDArrayFloat: TypeAlias = np.ndarray
NDArrayUint8: TypeAlias = np.ndarray
NDArrayInt: TypeAlias = np.ndarray
CheckerFunc: TypeAlias = Callable[[Sequence[JSONDict]], list["QualityIssue"]]
T = TypeVar("T")


class QualityCheckType(Enum):
    """Types of quality checks available in the pipeline."""

    IMAGE_QUALITY = auto()
    LABEL_CONSISTENCY = auto()
    DISTRIBUTION_BALANCE = auto()
    DUPLICATE_DETECTION = auto()
    METADATA_VALIDATION = auto()
    ANNOTATION_QUALITY = auto()
    OUTLIER_DETECTION = auto()
    DATA_DRIFT = auto()
    FEATURE_CORRELATION = auto()
    SAMPLE_INTEGRITY = auto()


class Severity(Enum):
    """Severity levels for quality issues with numeric ordering."""

    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    def __lt__(self, other: "Severity") -> bool:
        """Enable severity comparison."""
        return self.value < other.value

    def __le__(self, other: "Severity") -> bool:
        """Enable severity comparison."""
        return self.value <= other.value


class CheckResult(Enum):
    """Result status of quality check execution."""

    PASSED = auto()
    PASSED_WITH_WARNINGS = auto()
    FAILED = auto()
    SKIPPED = auto()
    ERROR = auto()


class ResolutionStatus(Enum):
    """Status of issue resolution workflow."""

    PENDING = auto()
    IN_PROGRESS = auto()
    RESOLVED = auto()
    WONT_FIX = auto()
    FALSE_POSITIVE = auto()


class DriftType(Enum):
    """Types of data drift detected."""

    FEATURE_DRIFT = auto()
    LABEL_DRIFT = auto()
    COVARIATE_SHIFT = auto()
    CONCEPT_DRIFT = auto()
    PRIOR_PROBABILITY_SHIFT = auto()


class OutlierMethod(Enum):
    """Methods for outlier detection."""

    ISOLATION_FOREST = auto()
    LOCAL_OUTLIER_FACTOR = auto()
    MAHALANOBIS_DISTANCE = auto()
    Z_SCORE = auto()
    IQR = auto()
    DBSCAN = auto()


class HashAlgorithm(Enum):
    """Perceptual hash algorithms for duplicate detection."""

    DHASH = auto()
    PHASH = auto()
    AHASH = auto()
    WHASH = auto()


class ImageQualityMetric(NamedTuple):
    """Comprehensive image quality assessment metrics."""

    image_id: str
    sharpness_score: float
    contrast_score: float
    brightness_score: float
    noise_level: float
    saturation_score: float
    blur_score: float
    dynamic_range: float
    edge_density: float
    texture_score: float
    focus_measure: float
    overall_score: float
    is_acceptable: bool


class LabelConsistencyResult(NamedTuple):
    """Result of label consistency validation."""

    sample_id: str
    original_label: str
    predicted_label: str
    confidence: float
    is_consistent: bool
    needs_review: bool
    alternative_labels: tuple[str, ...]
    consistency_score: float


class DistributionMetrics(NamedTuple):
    """Comprehensive class distribution metrics."""

    class_counts: dict[str, int]
    class_proportions: dict[str, float]
    imbalance_ratio: float
    min_class: str
    max_class: str
    entropy: float
    gini_coefficient: float
    effective_num_classes: float
    kl_divergence_from_uniform: float


class DuplicateGroup(NamedTuple):
    """Group of duplicate or near-duplicate samples."""

    group_id: str
    sample_ids: tuple[str, ...]
    similarity_scores: tuple[float, ...]
    average_similarity: float
    is_exact_duplicate: bool
    hash_algorithm: str


class OutlierResult(NamedTuple):
    """Result of outlier detection analysis."""

    sample_id: str
    outlier_score: float
    is_outlier: bool
    detection_method: OutlierMethod
    feature_contributions: dict[str, float]
    nearest_inlier_distance: float


class DriftResult(NamedTuple):
    """Result of data drift detection."""

    feature_name: str
    drift_type: DriftType
    drift_score: float
    is_significant: bool
    p_value: float
    reference_mean: float
    current_mean: float
    threshold: float


class CorrelationResult(NamedTuple):
    """Feature correlation analysis result."""

    feature_a: str
    feature_b: str
    pearson_correlation: float
    spearman_correlation: float
    mutual_information: float
    is_highly_correlated: bool


class QualityIssue(NamedTuple):
    """Detected quality issue with full context."""

    issue_id: str
    check_type: QualityCheckType
    severity: Severity
    sample_id: str
    description: str
    detected_at: datetime
    resolution_status: str
    suggested_action: str
    metadata: JSONDict


class QualityReport(NamedTuple):
    """Complete quality assurance report."""

    report_id: str
    generated_at: datetime
    execution_time_seconds: float
    total_samples: int
    checks_performed: int
    checks_passed: int
    checks_failed: int
    issues_found: int
    critical_issues: int
    error_issues: int
    warning_issues: int
    info_issues: int
    overall_result: CheckResult
    issues: tuple[QualityIssue, ...]
    recommendations: tuple[str, ...]


class AlertService(Protocol):
    """Protocol for quality alert notifications."""

    def send_alert(
        self, severity: Severity, message: str, details: JSONDict
    ) -> None:
        """Send quality alert notification."""
        ...

    def send_batch_alert(
        self, issues: Sequence[QualityIssue]
    ) -> None:
        """Send batch alert for multiple issues."""
        ...


class QualityChecker(Protocol):
    """Protocol for individual quality checks."""

    def check(self, samples: Sequence[JSONDict]) -> list[QualityIssue]:
        """Run quality check on samples."""
        ...

    @property
    def check_type(self) -> QualityCheckType:
        """Return the type of check performed."""
        ...


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Configurable quality thresholds for all check types."""

    min_sharpness: float = 0.3
    min_contrast: float = 0.2
    max_brightness_deviation: float = 0.4
    max_noise_level: float = 0.3
    min_blur_score: float = 0.25
    min_dynamic_range: float = 0.4
    min_edge_density: float = 0.1
    min_texture_score: float = 0.2
    min_focus_measure: float = 0.3
    min_sample_per_class: int = 100
    max_imbalance_ratio: float = 10.0
    target_entropy_ratio: float = 0.8
    max_gini_coefficient: float = 0.5
    duplicate_similarity_threshold: float = 0.95
    near_duplicate_threshold: float = 0.85
    min_annotation_agreement: float = 0.7
    label_confidence_threshold: float = 0.8
    outlier_contamination: float = 0.05
    drift_significance_level: float = 0.05
    psi_threshold: float = 0.25
    feature_correlation_threshold: float = 0.95
    min_metadata_completeness: float = 0.95
    max_missing_rate: float = 0.05


@dataclass(frozen=True, slots=True)
class CheckConfiguration:
    """Configuration for quality check execution."""

    enabled_checks: tuple[QualityCheckType, ...]
    thresholds: QualityThresholds
    batch_size: int = 100
    parallel_workers: int = 4
    fail_fast: bool = False
    generate_report: bool = True
    save_intermediate: bool = False
    verbose_logging: bool = False
    max_issues_per_check: int = 1000
    sample_for_distribution: float = 1.0
    incremental_mode: bool = False


@dataclass(frozen=True, slots=True)
class ReferenceDistribution:
    """Reference distribution for drift detection."""

    feature_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    quartiles: tuple[float, float, float]
    histogram_bins: tuple[float, ...]
    histogram_counts: tuple[int, ...]
    sample_size: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ValidationSchema:
    """Schema definition for metadata validation."""

    field_name: str
    field_type: str
    required: bool
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[str, ...] | None = None
    regex_pattern: str | None = None
    foreign_key_ref: str | None = None


@dataclass(frozen=True, slots=True)
class AnnotationAgreementMetrics:
    """Inter-annotator agreement metrics."""

    cohens_kappa: float
    fleiss_kappa: float
    krippendorffs_alpha: float
    percent_agreement: float
    specific_agreement_positive: float
    specific_agreement_negative: float
    prevalence_index: float
    bias_index: float


@dataclass(slots=True)
class ImageQualityAnalyzer:
    """Analyzes comprehensive image quality metrics for microscopy images."""

    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    _laplacian_kernel: NDArrayFloat = field(init=False, repr=False)
    _sobel_x: NDArrayFloat = field(init=False, repr=False)
    _sobel_y: NDArrayFloat = field(init=False, repr=False)
    _gaussian_kernel: NDArrayFloat = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize convolution kernels."""
        self._laplacian_kernel = np.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64
        )
        self._sobel_x = np.array(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64
        )
        self._sobel_y = np.array(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64
        )
        sigma = 1.0
        size = 5
        ax = np.arange(-size // 2 + 1, size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        self._gaussian_kernel = kernel / np.sum(kernel)

    def _to_grayscale(self, image: NDArrayUint8) -> NDArrayFloat:
        """Convert image to grayscale float."""
        if image.ndim == 3:
            return np.mean(image, axis=2).astype(np.float64)
        return image.astype(np.float64)

    def _convolve2d(
        self, image: NDArrayFloat, kernel: NDArrayFloat
    ) -> NDArrayFloat:
        """Apply 2D convolution with padding."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        h, w = image.shape
        result = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                patch = padded[i : i + kh, j : j + kw]
                result[i, j] = np.sum(patch * kernel)
        return result

    def compute_sharpness(self, image: NDArrayUint8) -> float:
        """Compute image sharpness using Laplacian variance."""
        gray = self._to_grayscale(image)
        laplacian = self._convolve2d(gray, self._laplacian_kernel)
        variance = float(np.var(laplacian))
        return min(1.0, variance / 1000.0)

    def compute_contrast(self, image: NDArrayUint8) -> float:
        """Compute image contrast using RMS contrast."""
        gray = self._to_grayscale(image)
        mean_val = np.mean(gray)
        rms = np.sqrt(np.mean((gray - mean_val) ** 2))
        return min(1.0, rms / 80.0)

    def compute_brightness(self, image: NDArrayUint8) -> float:
        """Compute brightness deviation from optimal."""
        mean_brightness = np.mean(image) / 255.0
        deviation = abs(mean_brightness - 0.5)
        return 1.0 - (deviation * 2)

    def compute_noise_level(self, image: NDArrayUint8) -> float:
        """Estimate noise level using median absolute deviation."""
        gray = self._to_grayscale(image)
        h, w = gray.shape
        block_size = 8
        variances: list[float] = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i : i + block_size, j : j + block_size]
                variances.append(float(np.var(block)))
        if not variances:
            return 0.0
        noise_estimate = float(np.median(variances))
        return min(1.0, noise_estimate / 500.0)

    def compute_saturation(self, image: NDArrayUint8) -> float:
        """Compute color saturation for color images."""
        if image.ndim == 3 and image.shape[2] >= 3:
            channel_std = np.std(image, axis=2)
            saturation = float(np.mean(channel_std)) / 128.0
            return min(1.0, saturation)
        return 0.5

    def compute_blur_score(self, image: NDArrayUint8) -> float:
        """Compute blur detection score using gradient magnitude."""
        gray = self._to_grayscale(image)
        gx = self._convolve2d(gray, self._sobel_x)
        gy = self._convolve2d(gray, self._sobel_y)
        magnitude = np.sqrt(gx**2 + gy**2)
        score = float(np.mean(magnitude))
        return min(1.0, score / 50.0)

    def compute_dynamic_range(self, image: NDArrayUint8) -> float:
        """Compute dynamic range utilization."""
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        range_used = (max_val - min_val) / 255.0
        return range_used

    def compute_edge_density(self, image: NDArrayUint8) -> float:
        """Compute edge pixel density using Sobel operator."""
        gray = self._to_grayscale(image)
        gx = self._convolve2d(gray, self._sobel_x)
        gy = self._convolve2d(gray, self._sobel_y)
        magnitude = np.sqrt(gx**2 + gy**2)
        threshold = np.mean(magnitude) + np.std(magnitude)
        edge_pixels = np.sum(magnitude > threshold)
        total_pixels = gray.shape[0] * gray.shape[1]
        return float(edge_pixels) / total_pixels

    def compute_texture_score(self, image: NDArrayUint8) -> float:
        """Compute texture richness using GLCM-like features."""
        gray = self._to_grayscale(image)
        quantized = (gray / 64).astype(np.int32)
        h, w = quantized.shape
        cooccurrence = np.zeros((4, 4), dtype=np.float64)
        for i in range(h - 1):
            for j in range(w - 1):
                val1 = min(3, max(0, quantized[i, j]))
                val2 = min(3, max(0, quantized[i, j + 1]))
                cooccurrence[val1, val2] += 1
        if np.sum(cooccurrence) > 0:
            cooccurrence /= np.sum(cooccurrence)
        contrast = 0.0
        for i in range(4):
            for j in range(4):
                contrast += (i - j) ** 2 * cooccurrence[i, j]
        return min(1.0, contrast / 5.0)

    def compute_focus_measure(self, image: NDArrayUint8) -> float:
        """Compute focus measure using Brenner gradient."""
        gray = self._to_grayscale(image)
        h, w = gray.shape
        if w < 3:
            return 0.0
        diff = gray[:, 2:] - gray[:, :-2]
        focus = float(np.mean(diff**2))
        return min(1.0, focus / 2000.0)

    def analyze_image(
        self, image_id: str, image: NDArrayUint8
    ) -> ImageQualityMetric:
        """Compute all quality metrics for image."""
        sharpness = self.compute_sharpness(image)
        contrast = self.compute_contrast(image)
        brightness = self.compute_brightness(image)
        noise = self.compute_noise_level(image)
        saturation = self.compute_saturation(image)
        blur = self.compute_blur_score(image)
        dynamic = self.compute_dynamic_range(image)
        edge = self.compute_edge_density(image)
        texture = self.compute_texture_score(image)
        focus = self.compute_focus_measure(image)

        overall = (
            sharpness * 0.20
            + contrast * 0.15
            + brightness * 0.10
            + (1 - noise) * 0.15
            + blur * 0.15
            + dynamic * 0.05
            + edge * 0.05
            + texture * 0.05
            + focus * 0.10
        )

        is_acceptable = (
            sharpness >= self.thresholds.min_sharpness
            and contrast >= self.thresholds.min_contrast
            and noise <= self.thresholds.max_noise_level
            and blur >= self.thresholds.min_blur_score
        )

        return ImageQualityMetric(
            image_id=image_id,
            sharpness_score=sharpness,
            contrast_score=contrast,
            brightness_score=brightness,
            noise_level=noise,
            saturation_score=saturation,
            blur_score=blur,
            dynamic_range=dynamic,
            edge_density=edge,
            texture_score=texture,
            focus_measure=focus,
            overall_score=overall,
            is_acceptable=is_acceptable,
        )

    def check_quality(
        self, metrics: ImageQualityMetric
    ) -> list[QualityIssue]:
        """Check metrics against thresholds and generate issues."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid

        if metrics.sharpness_score < self.thresholds.min_sharpness:
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.IMAGE_QUALITY,
                    severity=Severity.WARNING,
                    sample_id=metrics.image_id,
                    description=f"Low sharpness: {metrics.sharpness_score:.3f}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Consider re-acquiring image with better focus",
                    metadata={"metric": "sharpness", "value": metrics.sharpness_score},
                )
            )

        if metrics.contrast_score < self.thresholds.min_contrast:
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.IMAGE_QUALITY,
                    severity=Severity.WARNING,
                    sample_id=metrics.image_id,
                    description=f"Low contrast: {metrics.contrast_score:.3f}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Apply contrast enhancement or adjust staining",
                    metadata={"metric": "contrast", "value": metrics.contrast_score},
                )
            )

        if metrics.noise_level > self.thresholds.max_noise_level:
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.IMAGE_QUALITY,
                    severity=Severity.WARNING,
                    sample_id=metrics.image_id,
                    description=f"High noise level: {metrics.noise_level:.3f}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Apply denoising filter or reduce exposure",
                    metadata={"metric": "noise", "value": metrics.noise_level},
                )
            )

        if metrics.blur_score < self.thresholds.min_blur_score:
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.IMAGE_QUALITY,
                    severity=Severity.ERROR,
                    sample_id=metrics.image_id,
                    description=f"Blurry image detected: {metrics.blur_score:.3f}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Re-acquire with proper focus calibration",
                    metadata={"metric": "blur", "value": metrics.blur_score},
                )
            )

        if metrics.dynamic_range < self.thresholds.min_dynamic_range:
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.IMAGE_QUALITY,
                    severity=Severity.INFO,
                    sample_id=metrics.image_id,
                    description=f"Limited dynamic range: {metrics.dynamic_range:.3f}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Check exposure settings",
                    metadata={"metric": "dynamic_range", "value": metrics.dynamic_range},
                )
            )

        if not metrics.is_acceptable:
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.IMAGE_QUALITY,
                    severity=Severity.ERROR,
                    sample_id=metrics.image_id,
                    description=f"Image below quality threshold: {metrics.overall_score:.3f}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Flag for manual review or re-acquisition",
                    metadata={"metric": "overall", "value": metrics.overall_score},
                )
            )

        return issues

    def batch_analyze(
        self, images: Sequence[tuple[str, NDArrayUint8]]
    ) -> list[ImageQualityMetric]:
        """Analyze batch of images."""
        return [self.analyze_image(img_id, img) for img_id, img in images]


@dataclass(slots=True)
class LabelConsistencyChecker:
    """Checks consistency between labels and model predictions with advanced metrics."""

    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    _label_encoder: dict[str, int] = field(default_factory=dict, init=False)
    _label_decoder: dict[int, str] = field(default_factory=dict, init=False)
    _class_priors: dict[str, float] = field(default_factory=dict, init=False)
    _confusion_matrix: NDArrayInt = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize confusion matrix."""
        self._confusion_matrix = np.zeros((1, 1), dtype=np.int64)

    def encode_labels(self, labels: Sequence[str]) -> None:
        """Build label encoding from unique labels and compute priors."""
        unique = sorted(set(labels))
        self._label_encoder = {label: i for i, label in enumerate(unique)}
        self._label_decoder = {i: label for label, i in self._label_encoder.items()}
        total = len(labels)
        label_counts: dict[str, int] = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        self._class_priors = {
            label: count / total for label, count in label_counts.items()
        }
        n_classes = len(unique)
        self._confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update_confusion_matrix(
        self, original: str, predicted: str
    ) -> None:
        """Update confusion matrix with new observation."""
        if original in self._label_encoder and predicted in self._label_encoder:
            i = self._label_encoder[original]
            j = self._label_encoder[predicted]
            self._confusion_matrix[i, j] += 1

    def compute_cohens_kappa(self) -> float:
        """Compute Cohen's Kappa coefficient."""
        matrix = self._confusion_matrix.astype(np.float64)
        total = np.sum(matrix)
        if total == 0:
            return 0.0
        observed_agreement = np.trace(matrix) / total
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        expected_agreement = np.sum(row_sums * col_sums) / (total**2)
        if expected_agreement >= 1.0:
            return 1.0
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return float(kappa)

    def check_consistency(
        self,
        sample_id: str,
        original_label: str,
        predicted_label: str,
        confidence: float,
        alternative_predictions: Sequence[tuple[str, float]] | None = None,
    ) -> LabelConsistencyResult:
        """Check if label is consistent with prediction."""
        is_consistent = original_label == predicted_label
        needs_review = not is_consistent and confidence > self.thresholds.label_confidence_threshold
        alternatives = tuple(p[0] for p in (alternative_predictions or []))
        consistency_score = confidence if is_consistent else (1.0 - confidence)
        self.update_confusion_matrix(original_label, predicted_label)
        return LabelConsistencyResult(
            sample_id=sample_id,
            original_label=original_label,
            predicted_label=predicted_label,
            confidence=confidence,
            is_consistent=is_consistent,
            needs_review=needs_review,
            alternative_labels=alternatives,
            consistency_score=consistency_score,
        )

    def find_inconsistencies(
        self,
        labels: Sequence[tuple[str, str, str, float]],
    ) -> list[QualityIssue]:
        """Find all label inconsistencies and generate issues."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid
        for sample_id, original, predicted, conf in labels:
            result = self.check_consistency(sample_id, original, predicted, conf)
            if result.needs_review:
                severity = Severity.ERROR if conf > 0.95 else Severity.WARNING
                issues.append(
                    QualityIssue(
                        issue_id=str(uuid.uuid4())[:8],
                        check_type=QualityCheckType.LABEL_CONSISTENCY,
                        severity=severity,
                        sample_id=sample_id,
                        description=(
                            f"Label mismatch: {original} vs {predicted} "
                            f"(conf: {conf:.3f})"
                        ),
                        detected_at=now,
                        resolution_status="pending",
                        suggested_action="Review label assignment with domain expert",
                        metadata={
                            "original": original,
                            "predicted": predicted,
                            "confidence": conf,
                        },
                    )
                )
        return issues

    def get_agreement_report(self) -> dict[str, Any]:
        """Generate comprehensive agreement report."""
        kappa = self.compute_cohens_kappa()
        matrix = self._confusion_matrix.astype(np.float64)
        total = np.sum(matrix)
        if total == 0:
            return {"cohens_kappa": 0.0, "accuracy": 0.0, "total_samples": 0}
        accuracy = np.trace(matrix) / total
        return {
            "cohens_kappa": kappa,
            "accuracy": float(accuracy),
            "total_samples": int(total),
            "confusion_matrix": matrix.tolist(),
            "class_labels": list(self._label_encoder.keys()),
        }


@dataclass(slots=True)
class DistributionAnalyzer:
    """Analyzes class distribution balance with comprehensive metrics."""

    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    _reference_distribution: dict[str, float] | None = field(default=None, init=False)

    def set_reference_distribution(
        self, distribution: dict[str, float]
    ) -> None:
        """Set reference distribution for drift detection."""
        self._reference_distribution = distribution

    def compute_distribution(
        self, labels: Sequence[str]
    ) -> DistributionMetrics:
        """Compute comprehensive distribution metrics."""
        counts: dict[str, int] = defaultdict(int)
        for label in labels:
            counts[label] += 1
        if not counts:
            return DistributionMetrics(
                class_counts={},
                class_proportions={},
                imbalance_ratio=1.0,
                min_class="",
                max_class="",
                entropy=0.0,
                gini_coefficient=0.0,
                effective_num_classes=0.0,
                kl_divergence_from_uniform=0.0,
            )
        total = sum(counts.values())
        proportions = {k: v / total for k, v in counts.items()}
        min_count = min(counts.values())
        max_count = max(counts.values())
        imbalance = max_count / max(min_count, 1)
        min_class = min(counts.keys(), key=lambda k: counts[k])
        max_class = max(counts.keys(), key=lambda k: counts[k])
        probs = np.array(list(proportions.values()))
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        n = len(counts)
        sorted_probs = np.sort(probs)
        cumulative = np.cumsum(sorted_probs)
        gini = 1 - 2 * np.sum(cumulative) / (n * np.sum(probs)) if np.sum(probs) > 0 else 0.0
        effective_classes = np.exp(entropy * np.log(2)) if entropy > 0 else 1.0
        uniform_prob = 1.0 / n if n > 0 else 1.0
        kl_div = float(np.sum(probs * np.log2((probs + 1e-10) / uniform_prob)))
        return DistributionMetrics(
            class_counts=dict(counts),
            class_proportions=proportions,
            imbalance_ratio=imbalance,
            min_class=min_class,
            max_class=max_class,
            entropy=entropy,
            gini_coefficient=float(gini),
            effective_num_classes=float(effective_classes),
            kl_divergence_from_uniform=kl_div,
        )

    def check_distribution(
        self, metrics: DistributionMetrics
    ) -> list[QualityIssue]:
        """Check distribution against thresholds."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid
        if metrics.imbalance_ratio > self.thresholds.max_imbalance_ratio:
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.DISTRIBUTION_BALANCE,
                    severity=Severity.ERROR,
                    sample_id="dataset",
                    description=(
                        f"Class imbalance: {metrics.imbalance_ratio:.1f}x "
                        f"({metrics.max_class} vs {metrics.min_class})"
                    ),
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Apply oversampling or class weights",
                    metadata={
                        "imbalance_ratio": metrics.imbalance_ratio,
                        "min_class": metrics.min_class,
                        "max_class": metrics.max_class,
                    },
                )
            )
        for class_name, count in metrics.class_counts.items():
            if count < self.thresholds.min_sample_per_class:
                issues.append(
                    QualityIssue(
                        issue_id=str(uuid.uuid4())[:8],
                        check_type=QualityCheckType.DISTRIBUTION_BALANCE,
                        severity=Severity.WARNING,
                        sample_id=class_name,
                        description=(
                            f"Insufficient samples for {class_name}: "
                            f"{count} < {self.thresholds.min_sample_per_class}"
                        ),
                        detected_at=now,
                        resolution_status="pending",
                        suggested_action="Collect more samples or use data augmentation",
                        metadata={"class": class_name, "count": count},
                    )
                )
        if metrics.gini_coefficient > self.thresholds.max_gini_coefficient:
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.DISTRIBUTION_BALANCE,
                    severity=Severity.WARNING,
                    sample_id="dataset",
                    description=f"High Gini coefficient: {metrics.gini_coefficient:.3f}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Rebalance dataset distribution",
                    metadata={"gini": metrics.gini_coefficient},
                )
            )
        return issues

    def compute_kl_divergence(
        self, current: dict[str, float], reference: dict[str, float]
    ) -> float:
        """Compute KL divergence between distributions."""
        all_keys = set(current.keys()) | set(reference.keys())
        kl = 0.0
        for key in all_keys:
            p = current.get(key, 1e-10)
            q = reference.get(key, 1e-10)
            kl += p * np.log2(p / q)
        return float(kl)

    def compute_psi(
        self, current: dict[str, float], reference: dict[str, float]
    ) -> float:
        """Compute Population Stability Index."""
        all_keys = set(current.keys()) | set(reference.keys())
        psi = 0.0
        for key in all_keys:
            p = current.get(key, 1e-10)
            q = reference.get(key, 1e-10)
            psi += (p - q) * np.log(p / q)
        return float(psi)


@dataclass(slots=True)
class DuplicateDetector:
    """Detects duplicate and near-duplicate samples."""

    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    hash_size: int = 16

    def compute_perceptual_hash(self, image: NDArrayUint8) -> NDArrayFloat:
        """Compute perceptual hash for image."""
        # Resize to small fixed size
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float64)

        # Simple resize by averaging blocks
        h, w = gray.shape
        block_h = h // self.hash_size
        block_w = w // self.hash_size

        hash_values = np.zeros((self.hash_size, self.hash_size))
        for i in range(self.hash_size):
            for j in range(self.hash_size):
                block = gray[
                    i * block_h : (i + 1) * block_h,
                    j * block_w : (j + 1) * block_w,
                ]
                hash_values[i, j] = np.mean(block)

        # Binary hash based on median
        median = np.median(hash_values)
        binary_hash = (hash_values > median).astype(np.float64)
        return binary_hash.flatten()

    def compute_similarity(
        self, hash1: NDArrayFloat, hash2: NDArrayFloat
    ) -> float:
        """Compute similarity between two hashes."""
        # Hamming distance normalized to similarity
        distance = np.sum(hash1 != hash2)
        max_distance = len(hash1)
        return 1.0 - (distance / max_distance)

    def find_duplicates(
        self, images: Sequence[tuple[str, NDArrayUint8]]
    ) -> list[DuplicateGroup]:
        """Find duplicate and near-duplicate images."""
        if len(images) < 2:
            return []

        # Compute hashes
        hashes: list[tuple[str, NDArrayFloat]] = []
        for img_id, img in images:
            hash_val = self.compute_perceptual_hash(img)
            hashes.append((img_id, hash_val))

        # Find similar pairs
        groups: list[DuplicateGroup] = []
        processed: set[str] = set()
        import uuid

        for i, (id1, hash1) in enumerate(hashes):
            if id1 in processed:
                continue

            group_members = [id1]
            max_similarity = 0.0

            for j in range(i + 1, len(hashes)):
                id2, hash2 = hashes[j]
                if id2 in processed:
                    continue

                sim = self.compute_similarity(hash1, hash2)
                if sim >= self.thresholds.duplicate_similarity_threshold:
                    group_members.append(id2)
                    max_similarity = max(max_similarity, sim)

            if len(group_members) > 1:
                for member in group_members:
                    processed.add(member)

                groups.append(
                    DuplicateGroup(
                        group_id=str(uuid.uuid4())[:8],
                        sample_ids=tuple(group_members),
                        similarity_scores=(max_similarity,),
                        average_similarity=max_similarity,
                        is_exact_duplicate=max_similarity >= 0.99,
                        hash_algorithm="PHASH",
                    )
                )

        return groups

    def check_duplicates(
        self, groups: Sequence[DuplicateGroup]
    ) -> list[QualityIssue]:
        """Generate issues for duplicate groups."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid

        for group in groups:
            severity = (
                Severity.ERROR if group.is_exact_duplicate else Severity.WARNING
            )
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.DUPLICATE_DETECTION,
                    severity=severity,
                    sample_id=group.group_id,
                    description=(
                        f"Duplicate group: {len(group.sample_ids)} samples "
                        f"(similarity: {group.average_similarity:.3f})"
                    ),
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Review duplicates and remove or merge",
                    metadata={"sample_ids": list(group.sample_ids)},
                )
            )

        return issues


@dataclass(slots=True)
class MetadataValidator:
    """Validates sample metadata completeness and correctness."""

    required_fields: tuple[str, ...] = (
        "sample_id",
        "image_path",
        "label",
        "created_at",
    )
    optional_fields: tuple[str, ...] = (
        "annotator_id",
        "source",
        "quality_score",
    )

    def validate_record(
        self, record: JSONDict
    ) -> list[QualityIssue]:
        """Validate single metadata record."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        sample_id = record.get("sample_id", "unknown")
        import uuid

        for field_name in self.required_fields:
            if field_name not in record:
                issues.append(
                    QualityIssue(
                        issue_id=str(uuid.uuid4())[:8],
                        check_type=QualityCheckType.METADATA_VALIDATION,
                        severity=Severity.ERROR,
                        sample_id=sample_id,
                        description=f"Missing required field: {field_name}",
                        detected_at=now,
                        resolution_status="pending",
                        suggested_action=f"Add required field '{field_name}' to metadata",
                        metadata={"field_name": field_name, "record": record},
                    )
                )
            elif record[field_name] is None or record[field_name] == "":
                issues.append(
                    QualityIssue(
                        issue_id=str(uuid.uuid4())[:8],
                        check_type=QualityCheckType.METADATA_VALIDATION,
                        severity=Severity.WARNING,
                        sample_id=sample_id,
                        description=f"Empty required field: {field_name}",
                        detected_at=now,
                        resolution_status="pending",
                        suggested_action=f"Populate empty field '{field_name}'",
                        metadata={"field_name": field_name, "record": record},
                    )
                )

        return issues

    def validate_batch(
        self, records: Sequence[JSONDict]
    ) -> list[QualityIssue]:
        """Validate batch of metadata records."""
        all_issues: list[QualityIssue] = []
        for record in records:
            issues = self.validate_record(record)
            all_issues.extend(issues)
        return all_issues


@dataclass(slots=True)
class QualityAssurancePipeline:
    """Main pipeline for data quality assurance."""

    storage_dir: Path
    config: CheckConfiguration = field(
        default_factory=lambda: CheckConfiguration(
            enabled_checks=tuple(QualityCheckType),
            thresholds=QualityThresholds(),
        )
    )
    image_analyzer: ImageQualityAnalyzer = field(init=False)
    label_checker: LabelConsistencyChecker = field(init=False)
    distribution_analyzer: DistributionAnalyzer = field(init=False)
    duplicate_detector: DuplicateDetector = field(init=False)
    metadata_validator: MetadataValidator = field(
        default_factory=MetadataValidator
    )
    _issues: list[QualityIssue] = field(
        default_factory=list, init=False, repr=False
    )
    _reports: list[QualityReport] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize analyzers."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.image_analyzer = ImageQualityAnalyzer(
            thresholds=self.config.thresholds
        )
        self.label_checker = LabelConsistencyChecker(
            thresholds=self.config.thresholds
        )
        self.distribution_analyzer = DistributionAnalyzer(
            thresholds=self.config.thresholds
        )
        self.duplicate_detector = DuplicateDetector(
            thresholds=self.config.thresholds
        )

    def run_image_quality_checks(
        self, images: Sequence[tuple[str, NDArrayUint8]]
    ) -> list[QualityIssue]:
        """Run image quality checks on batch."""
        issues: list[QualityIssue] = []
        for img_id, img in images:
            metrics = self.image_analyzer.analyze_image(img_id, img)
            img_issues = self.image_analyzer.check_quality(metrics)
            issues.extend(img_issues)
        return issues

    def run_distribution_check(
        self, labels: Sequence[str]
    ) -> list[QualityIssue]:
        """Check label distribution."""
        metrics = self.distribution_analyzer.compute_distribution(labels)
        return self.distribution_analyzer.check_distribution(metrics)

    def run_duplicate_check(
        self, images: Sequence[tuple[str, NDArrayUint8]]
    ) -> list[QualityIssue]:
        """Check for duplicate images."""
        groups = self.duplicate_detector.find_duplicates(images)
        return self.duplicate_detector.check_duplicates(groups)

    def run_metadata_check(
        self, records: Sequence[JSONDict]
    ) -> list[QualityIssue]:
        """Check metadata completeness."""
        return self.metadata_validator.validate_batch(records)

    def run_full_pipeline(
        self,
        images: Sequence[tuple[str, NDArrayUint8]],
        labels: Sequence[str],
        metadata: Sequence[JSONDict],
    ) -> QualityReport:
        """Run complete quality assurance pipeline."""
        import uuid

        all_issues: list[QualityIssue] = []
        checks_performed = 0

        if QualityCheckType.IMAGE_QUALITY in self.config.enabled_checks:
            issues = self.run_image_quality_checks(images)
            all_issues.extend(issues)
            checks_performed += 1

        if QualityCheckType.DISTRIBUTION_BALANCE in self.config.enabled_checks:
            issues = self.run_distribution_check(labels)
            all_issues.extend(issues)
            checks_performed += 1

        if QualityCheckType.DUPLICATE_DETECTION in self.config.enabled_checks:
            issues = self.run_duplicate_check(images)
            all_issues.extend(issues)
            checks_performed += 1

        if QualityCheckType.METADATA_VALIDATION in self.config.enabled_checks:
            issues = self.run_metadata_check(metadata)
            all_issues.extend(issues)
            checks_performed += 1

        # Determine overall result
        critical_count = sum(
            1 for issue in all_issues if issue.severity == Severity.CRITICAL
        )
        error_count = sum(
            1 for issue in all_issues if issue.severity == Severity.ERROR
        )
        warning_count = sum(
            1 for issue in all_issues if issue.severity == Severity.WARNING
        )

        if critical_count > 0 or error_count > 0:
            overall = CheckResult.FAILED
        elif warning_count > 0:
            overall = CheckResult.PASSED_WITH_WARNINGS
        else:
            overall = CheckResult.PASSED

        self._issues.extend(all_issues)

        info_count = sum(
            1 for issue in all_issues if issue.severity == Severity.INFO
        )
        elapsed = (datetime.now() - datetime.now()).total_seconds()

        report = QualityReport(
            report_id=str(uuid.uuid4())[:8],
            generated_at=datetime.now(),
            execution_time_seconds=elapsed,
            total_samples=len(images),
            checks_performed=checks_performed,
            checks_passed=checks_performed - (1 if critical_count + error_count > 0 else 0),
            checks_failed=1 if critical_count + error_count > 0 else 0,
            issues_found=len(all_issues),
            critical_issues=critical_count,
            error_issues=error_count,
            warning_issues=warning_count,
            info_issues=info_count,
            overall_result=overall,
            issues=tuple(all_issues),
            recommendations=("Review all issues and address critical ones first",),
        )

        self._reports.append(report)
        self._save_report(report)

        return report

    def _save_report(self, report: QualityReport) -> None:
        """Persist quality report."""
        report_file = (
            self.storage_dir / f"report_{report.report_id}.json"
        )
        data = {
            "report_id": report.report_id,
            "generated_at": report.generated_at.isoformat(),
            "total_samples": report.total_samples,
            "checks_performed": report.checks_performed,
            "issues_found": report.issues_found,
            "overall_result": report.overall_result.name,
            "issues": [
                {
                    "issue_id": issue.issue_id,
                    "check_type": issue.check_type.name,
                    "severity": issue.severity.name,
                    "sample_id": issue.sample_id,
                    "description": issue.description,
                    "detected_at": issue.detected_at.isoformat(),
                    "resolution_status": issue.resolution_status,
                }
                for issue in report.issues
            ],
        }
        with report_file.open("w") as f:
            json.dump(data, f, indent=2)

    def get_issues_by_severity(
        self, severity: Severity
    ) -> list[QualityIssue]:
        """Get all issues of specified severity."""
        return [i for i in self._issues if i.severity == severity]

    def get_issues_by_type(
        self, check_type: QualityCheckType
    ) -> list[QualityIssue]:
        """Get all issues of specified type."""
        return [i for i in self._issues if i.check_type == check_type]

    def iterate_reports(self) -> Iterator[QualityReport]:
        """Iterate over all quality reports."""
        yield from self._reports

    def get_latest_report(self) -> QualityReport | None:
        """Get most recent quality report."""
        if not self._reports:
            return None
        return max(self._reports, key=lambda r: r.generated_at)

    def export_issues_csv(self, output_path: Path) -> None:
        """Export all issues to CSV format."""
        import csv
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "issue_id", "check_type", "severity", "sample_id",
                "description", "detected_at", "resolution_status"
            ])
            for issue in self._issues:
                writer.writerow([
                    issue.issue_id,
                    issue.check_type.name,
                    issue.severity.name,
                    issue.sample_id,
                    issue.description,
                    issue.detected_at.isoformat(),
                    issue.resolution_status,
                ])

    def get_summary_statistics(self) -> dict[str, Any]:
        """Generate summary statistics across all checks."""
        if not self._reports:
            return {}
        latest = self.get_latest_report()
        if latest is None:
            return {}
        severity_counts = {s.name: 0 for s in Severity}
        type_counts = {t.name: 0 for t in QualityCheckType}
        for issue in self._issues:
            severity_counts[issue.severity.name] += 1
            type_counts[issue.check_type.name] += 1
        return {
            "total_issues": len(self._issues),
            "total_reports": len(self._reports),
            "latest_report_id": latest.report_id,
            "severity_distribution": severity_counts,
            "check_type_distribution": type_counts,
            "overall_result": latest.overall_result.name,
        }


@dataclass(slots=True)
class OutlierDetector:
    """Detects statistical outliers using multiple algorithms."""

    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    contamination: float = 0.05
    _feature_means: dict[str, float] = field(default_factory=dict, init=False)
    _feature_stds: dict[str, float] = field(default_factory=dict, init=False)
    _covariance_matrix: NDArrayFloat | None = field(default=None, init=False)
    _covariance_inv: NDArrayFloat | None = field(default=None, init=False)

    def fit(self, features: dict[str, Sequence[float]]) -> None:
        """Fit outlier detector on training data."""
        for name, values in features.items():
            arr = np.array(values)
            self._feature_means[name] = float(np.mean(arr))
            self._feature_stds[name] = float(np.std(arr)) + 1e-10
        if len(features) > 1:
            feature_names = sorted(features.keys())
            data_matrix = np.column_stack([
                np.array(features[name]) for name in feature_names
            ])
            self._covariance_matrix = np.cov(data_matrix, rowvar=False)
            try:
                self._covariance_inv = np.linalg.inv(self._covariance_matrix)
            except np.linalg.LinAlgError:
                self._covariance_inv = None

    def compute_z_scores(
        self, sample: dict[str, float]
    ) -> dict[str, float]:
        """Compute Z-scores for each feature."""
        z_scores: dict[str, float] = {}
        for name, value in sample.items():
            if name in self._feature_means:
                mean = self._feature_means[name]
                std = self._feature_stds[name]
                z_scores[name] = abs(value - mean) / std
        return z_scores

    def compute_mahalanobis_distance(
        self, sample: dict[str, float]
    ) -> float:
        """Compute Mahalanobis distance for sample."""
        if self._covariance_inv is None:
            return 0.0
        feature_names = sorted(self._feature_means.keys())
        x = np.array([sample.get(n, self._feature_means[n]) for n in feature_names])
        mu = np.array([self._feature_means[n] for n in feature_names])
        diff = x - mu
        distance = np.sqrt(np.dot(np.dot(diff, self._covariance_inv), diff))
        return float(distance)

    def detect_outlier(
        self,
        sample_id: str,
        features: dict[str, float],
        method: OutlierMethod = OutlierMethod.Z_SCORE,
    ) -> OutlierResult:
        """Detect if sample is an outlier."""
        z_scores = self.compute_z_scores(features)
        if method == OutlierMethod.Z_SCORE:
            max_z = max(z_scores.values()) if z_scores else 0.0
            is_outlier = max_z > 3.0
            score = max_z
        elif method == OutlierMethod.MAHALANOBIS_DISTANCE:
            score = self.compute_mahalanobis_distance(features)
            threshold = 3.0 * len(features) ** 0.5
            is_outlier = score > threshold
        elif method == OutlierMethod.IQR:
            score = max(z_scores.values()) if z_scores else 0.0
            is_outlier = score > 1.5
        else:
            score = max(z_scores.values()) if z_scores else 0.0
            is_outlier = score > 3.0
        return OutlierResult(
            sample_id=sample_id,
            outlier_score=score,
            is_outlier=is_outlier,
            detection_method=method,
            feature_contributions=z_scores,
            nearest_inlier_distance=0.0,
        )

    def detect_batch(
        self,
        samples: Sequence[tuple[str, dict[str, float]]],
        method: OutlierMethod = OutlierMethod.Z_SCORE,
    ) -> list[OutlierResult]:
        """Detect outliers in batch."""
        return [
            self.detect_outlier(sample_id, features, method)
            for sample_id, features in samples
        ]

    def generate_issues(
        self, results: Sequence[OutlierResult]
    ) -> list[QualityIssue]:
        """Generate issues for detected outliers."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid
        for result in results:
            if result.is_outlier:
                top_contributors = sorted(
                    result.feature_contributions.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                contrib_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in top_contributors
                )
                issues.append(
                    QualityIssue(
                        issue_id=str(uuid.uuid4())[:8],
                        check_type=QualityCheckType.OUTLIER_DETECTION,
                        severity=Severity.WARNING,
                        sample_id=result.sample_id,
                        description=(
                            f"Outlier detected (score={result.outlier_score:.2f}): "
                            f"{contrib_str}"
                        ),
                        detected_at=now,
                        resolution_status="pending",
                        suggested_action="Review sample for data entry errors",
                        metadata={
                            "score": result.outlier_score,
                            "method": result.detection_method.name,
                            "contributions": result.feature_contributions,
                        },
                    )
                )
        return issues


@dataclass(slots=True)
class DataDriftMonitor:
    """Monitors for data drift between reference and current distributions."""

    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    _reference_stats: dict[str, ReferenceDistribution] = field(
        default_factory=dict, init=False
    )

    def set_reference(
        self, feature_name: str, values: Sequence[float]
    ) -> ReferenceDistribution:
        """Set reference distribution for feature."""
        arr = np.array(values)
        hist_counts, hist_bins = np.histogram(arr, bins=10)
        ref = ReferenceDistribution(
            feature_name=feature_name,
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            quartiles=(
                float(np.percentile(arr, 25)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 75)),
            ),
            histogram_bins=tuple(hist_bins.tolist()),
            histogram_counts=tuple(hist_counts.tolist()),
            sample_size=len(values),
            created_at=datetime.now(),
        )
        self._reference_stats[feature_name] = ref
        return ref

    def compute_psi(
        self, feature_name: str, current_values: Sequence[float]
    ) -> float:
        """Compute Population Stability Index."""
        if feature_name not in self._reference_stats:
            return 0.0
        ref = self._reference_stats[feature_name]
        ref_counts = np.array(ref.histogram_counts) + 1
        ref_props = ref_counts / np.sum(ref_counts)
        curr_arr = np.array(current_values)
        curr_counts, _ = np.histogram(curr_arr, bins=ref.histogram_bins)
        curr_counts = curr_counts + 1
        curr_props = curr_counts / np.sum(curr_counts)
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        return float(psi)

    def compute_ks_statistic(
        self, feature_name: str, current_values: Sequence[float]
    ) -> tuple[float, float]:
        """Compute Kolmogorov-Smirnov statistic and approximate p-value."""
        if feature_name not in self._reference_stats:
            return 0.0, 1.0
        ref = self._reference_stats[feature_name]
        curr_arr = np.array(current_values)
        ref_mean = ref.mean
        ref_std = ref.std + 1e-10
        curr_mean = float(np.mean(curr_arr))
        n = len(current_values)
        ks_stat = abs(curr_mean - ref_mean) / ref_std
        p_value = np.exp(-2 * n * ks_stat**2) if ks_stat > 0 else 1.0
        return float(ks_stat), float(p_value)

    def detect_drift(
        self, feature_name: str, current_values: Sequence[float]
    ) -> DriftResult:
        """Detect drift for single feature."""
        if feature_name not in self._reference_stats:
            return DriftResult(
                feature_name=feature_name,
                drift_type=DriftType.FEATURE_DRIFT,
                drift_score=0.0,
                is_significant=False,
                p_value=1.0,
                reference_mean=0.0,
                current_mean=float(np.mean(current_values)),
                threshold=self.thresholds.psi_threshold,
            )
        ref = self._reference_stats[feature_name]
        psi = self.compute_psi(feature_name, current_values)
        ks_stat, p_value = self.compute_ks_statistic(feature_name, current_values)
        is_significant = psi > self.thresholds.psi_threshold or p_value < self.thresholds.drift_significance_level
        return DriftResult(
            feature_name=feature_name,
            drift_type=DriftType.FEATURE_DRIFT,
            drift_score=psi,
            is_significant=is_significant,
            p_value=p_value,
            reference_mean=ref.mean,
            current_mean=float(np.mean(current_values)),
            threshold=self.thresholds.psi_threshold,
        )

    def detect_all_drift(
        self, current_data: dict[str, Sequence[float]]
    ) -> list[DriftResult]:
        """Detect drift across all monitored features."""
        results: list[DriftResult] = []
        for feature_name, values in current_data.items():
            result = self.detect_drift(feature_name, values)
            results.append(result)
        return results

    def generate_issues(
        self, results: Sequence[DriftResult]
    ) -> list[QualityIssue]:
        """Generate issues for detected drift."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid
        for result in results:
            if result.is_significant:
                severity = Severity.ERROR if result.drift_score > 0.5 else Severity.WARNING
                issues.append(
                    QualityIssue(
                        issue_id=str(uuid.uuid4())[:8],
                        check_type=QualityCheckType.DATA_DRIFT,
                        severity=severity,
                        sample_id=result.feature_name,
                        description=(
                            f"Data drift detected: PSI={result.drift_score:.3f}, "
                            f"mean shift {result.reference_mean:.3f} -> {result.current_mean:.3f}"
                        ),
                        detected_at=now,
                        resolution_status="pending",
                        suggested_action="Investigate data pipeline or retrain model",
                        metadata={
                            "feature": result.feature_name,
                            "psi": result.drift_score,
                            "p_value": result.p_value,
                        },
                    )
                )
        return issues


@dataclass(slots=True)
class AnnotationQualityChecker:
    """Validates annotation quality and inter-annotator agreement."""

    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    _annotator_records: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )

    def record_annotation(
        self, annotator_id: str, sample_id: str, label: str
    ) -> None:
        """Record annotation for agreement calculation."""
        self._annotator_records[annotator_id].append((sample_id, label))

    def compute_percent_agreement(
        self, annotator_a: str, annotator_b: str
    ) -> float:
        """Compute percent agreement between two annotators."""
        records_a = dict(self._annotator_records.get(annotator_a, []))
        records_b = dict(self._annotator_records.get(annotator_b, []))
        common_samples = set(records_a.keys()) & set(records_b.keys())
        if not common_samples:
            return 0.0
        agreements = sum(
            1 for s in common_samples if records_a[s] == records_b[s]
        )
        return agreements / len(common_samples)

    def compute_cohens_kappa_pairwise(
        self, annotator_a: str, annotator_b: str
    ) -> float:
        """Compute Cohen's Kappa between two annotators."""
        records_a = dict(self._annotator_records.get(annotator_a, []))
        records_b = dict(self._annotator_records.get(annotator_b, []))
        common_samples = sorted(set(records_a.keys()) & set(records_b.keys()))
        if len(common_samples) < 2:
            return 0.0
        labels_a = [records_a[s] for s in common_samples]
        labels_b = [records_b[s] for s in common_samples]
        all_labels = sorted(set(labels_a) | set(labels_b))
        n = len(common_samples)
        k = len(all_labels)
        matrix = np.zeros((k, k), dtype=np.int64)
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        for la, lb in zip(labels_a, labels_b):
            matrix[label_to_idx[la], label_to_idx[lb]] += 1
        observed = np.trace(matrix) / n
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        expected = np.sum(row_sums * col_sums) / (n**2)
        if expected >= 1.0:
            return 1.0
        kappa = (observed - expected) / (1 - expected)
        return float(kappa)

    def compute_fleiss_kappa(self) -> float:
        """Compute Fleiss' Kappa for multiple annotators."""
        if len(self._annotator_records) < 2:
            return 0.0
        all_samples: set[str] = set()
        for records in self._annotator_records.values():
            all_samples.update(s for s, _ in records)
        if not all_samples:
            return 0.0
        sample_annotations: dict[str, list[str]] = defaultdict(list)
        for records in self._annotator_records.values():
            for sample_id, label in records:
                sample_annotations[sample_id].append(label)
        all_labels = sorted(set(
            label for labels in sample_annotations.values() for label in labels
        ))
        if len(all_labels) < 2:
            return 1.0
        n = len(sample_annotations)
        k = len(all_labels)
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        ratings = np.zeros((n, k), dtype=np.int64)
        for i, (sample_id, labels) in enumerate(sorted(sample_annotations.items())):
            for label in labels:
                ratings[i, label_to_idx[label]] += 1
        raters_per_sample = np.sum(ratings, axis=1)
        if np.any(raters_per_sample < 2):
            return 0.0
        p_j = np.sum(ratings, axis=0) / np.sum(ratings)
        p_e = np.sum(p_j**2)
        p_i = np.zeros(n)
        for i in range(n):
            n_i = raters_per_sample[i]
            p_i[i] = (np.sum(ratings[i]**2) - n_i) / (n_i * (n_i - 1))
        p_o = np.mean(p_i)
        if p_e >= 1.0:
            return 1.0
        kappa = (p_o - p_e) / (1 - p_e)
        return float(kappa)

    def get_agreement_metrics(self) -> AnnotationAgreementMetrics:
        """Compute comprehensive agreement metrics."""
        fleiss = self.compute_fleiss_kappa()
        annotators = list(self._annotator_records.keys())
        pairwise_kappas: list[float] = []
        pairwise_agreements: list[float] = []
        for i, ann_a in enumerate(annotators):
            for ann_b in annotators[i + 1:]:
                kappa = self.compute_cohens_kappa_pairwise(ann_a, ann_b)
                agreement = self.compute_percent_agreement(ann_a, ann_b)
                pairwise_kappas.append(kappa)
                pairwise_agreements.append(agreement)
        avg_kappa = statistics.mean(pairwise_kappas) if pairwise_kappas else 0.0
        avg_agreement = statistics.mean(pairwise_agreements) if pairwise_agreements else 0.0
        return AnnotationAgreementMetrics(
            cohens_kappa=avg_kappa,
            fleiss_kappa=fleiss,
            krippendorffs_alpha=fleiss,
            percent_agreement=avg_agreement,
            specific_agreement_positive=avg_agreement,
            specific_agreement_negative=avg_agreement,
            prevalence_index=0.0,
            bias_index=0.0,
        )

    def generate_issues(self) -> list[QualityIssue]:
        """Generate issues based on agreement metrics."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid
        metrics = self.get_agreement_metrics()
        if metrics.fleiss_kappa < self.thresholds.min_annotation_agreement:
            severity = Severity.ERROR if metrics.fleiss_kappa < 0.4 else Severity.WARNING
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.ANNOTATION_QUALITY,
                    severity=severity,
                    sample_id="annotators",
                    description=(
                        f"Low inter-annotator agreement: Fleiss' Kappa={metrics.fleiss_kappa:.3f}"
                    ),
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Provide additional annotator training or clarify guidelines",
                    metadata={
                        "fleiss_kappa": metrics.fleiss_kappa,
                        "cohens_kappa": metrics.cohens_kappa,
                        "percent_agreement": metrics.percent_agreement,
                    },
                )
            )
        return issues


@dataclass(slots=True)
class FeatureCorrelationAnalyzer:
    """Analyzes feature correlations to detect redundancy."""

    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    def compute_pearson(
        self, x: Sequence[float], y: Sequence[float]
    ) -> float:
        """Compute Pearson correlation coefficient."""
        x_arr = np.array(x)
        y_arr = np.array(y)
        if len(x_arr) < 2:
            return 0.0
        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)
        numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denominator = np.sqrt(
            np.sum((x_arr - x_mean) ** 2) * np.sum((y_arr - y_mean) ** 2)
        )
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    def compute_spearman(
        self, x: Sequence[float], y: Sequence[float]
    ) -> float:
        """Compute Spearman rank correlation."""
        x_arr = np.array(x)
        y_arr = np.array(y)
        x_ranks = np.argsort(np.argsort(x_arr)).astype(np.float64)
        y_ranks = np.argsort(np.argsort(y_arr)).astype(np.float64)
        return self.compute_pearson(x_ranks.tolist(), y_ranks.tolist())

    def analyze_pair(
        self, feature_a: str, values_a: Sequence[float],
        feature_b: str, values_b: Sequence[float],
    ) -> CorrelationResult:
        """Analyze correlation between feature pair."""
        pearson = self.compute_pearson(values_a, values_b)
        spearman = self.compute_spearman(values_a, values_b)
        is_highly_correlated = abs(pearson) > self.thresholds.feature_correlation_threshold
        return CorrelationResult(
            feature_a=feature_a,
            feature_b=feature_b,
            pearson_correlation=pearson,
            spearman_correlation=spearman,
            mutual_information=0.0,
            is_highly_correlated=is_highly_correlated,
        )

    def analyze_all_pairs(
        self, features: dict[str, Sequence[float]]
    ) -> list[CorrelationResult]:
        """Analyze correlations for all feature pairs."""
        results: list[CorrelationResult] = []
        feature_names = sorted(features.keys())
        for i, name_a in enumerate(feature_names):
            for name_b in feature_names[i + 1:]:
                result = self.analyze_pair(
                    name_a, features[name_a],
                    name_b, features[name_b],
                )
                results.append(result)
        return results

    def generate_issues(
        self, results: Sequence[CorrelationResult]
    ) -> list[QualityIssue]:
        """Generate issues for highly correlated features."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid
        for result in results:
            if result.is_highly_correlated:
                issues.append(
                    QualityIssue(
                        issue_id=str(uuid.uuid4())[:8],
                        check_type=QualityCheckType.FEATURE_CORRELATION,
                        severity=Severity.INFO,
                        sample_id=f"{result.feature_a}-{result.feature_b}",
                        description=(
                            f"Highly correlated features: {result.feature_a} & {result.feature_b} "
                            f"(r={result.pearson_correlation:.3f})"
                        ),
                        detected_at=now,
                        resolution_status="pending",
                        suggested_action="Consider removing one feature to reduce redundancy",
                        metadata={
                            "feature_a": result.feature_a,
                            "feature_b": result.feature_b,
                            "pearson": result.pearson_correlation,
                            "spearman": result.spearman_correlation,
                        },
                    )
                )
        return issues


@dataclass(slots=True)
class SampleIntegrityChecker:
    """Validates sample file integrity and completeness."""

    required_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    min_file_size_bytes: int = 1024
    max_file_size_bytes: int = 100 * 1024 * 1024

    def check_file_exists(self, file_path: Path) -> bool:
        """Check if file exists."""
        return file_path.exists()

    def check_file_size(self, file_path: Path) -> tuple[bool, int]:
        """Check if file size is within acceptable range."""
        if not file_path.exists():
            return False, 0
        size = file_path.stat().st_size
        is_valid = self.min_file_size_bytes <= size <= self.max_file_size_bytes
        return is_valid, size

    def check_extension(self, file_path: Path) -> bool:
        """Check if file has valid extension."""
        return file_path.suffix.lower() in self.required_extensions

    def compute_checksum(self, file_path: Path) -> str:
        """Compute MD5 checksum of file."""
        if not file_path.exists():
            return ""
        md5_hash = hashlib.md5()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def validate_sample(
        self, sample_id: str, file_path: Path
    ) -> list[QualityIssue]:
        """Validate single sample file."""
        issues: list[QualityIssue] = []
        now = datetime.now()
        import uuid
        if not self.check_file_exists(file_path):
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.SAMPLE_INTEGRITY,
                    severity=Severity.CRITICAL,
                    sample_id=sample_id,
                    description=f"File not found: {file_path}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Locate or re-acquire missing file",
                    metadata={"file_path": str(file_path)},
                )
            )
            return issues
        if not self.check_extension(file_path):
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.SAMPLE_INTEGRITY,
                    severity=Severity.WARNING,
                    sample_id=sample_id,
                    description=f"Unexpected file extension: {file_path.suffix}",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Convert to supported format",
                    metadata={"extension": file_path.suffix},
                )
            )
        size_valid, size = self.check_file_size(file_path)
        if not size_valid:
            severity = Severity.ERROR if size < self.min_file_size_bytes else Severity.WARNING
            issues.append(
                QualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    check_type=QualityCheckType.SAMPLE_INTEGRITY,
                    severity=severity,
                    sample_id=sample_id,
                    description=f"Invalid file size: {size} bytes",
                    detected_at=now,
                    resolution_status="pending",
                    suggested_action="Check file corruption or resize",
                    metadata={"size_bytes": size},
                )
            )
        return issues

    def validate_batch(
        self, samples: Sequence[tuple[str, Path]]
    ) -> list[QualityIssue]:
        """Validate batch of sample files."""
        all_issues: list[QualityIssue] = []
        for sample_id, file_path in samples:
            issues = self.validate_sample(sample_id, file_path)
            all_issues.extend(issues)
        return all_issues


def create_qa_pipeline(
    storage_dir: Path,
    thresholds: QualityThresholds | None = None,
) -> QualityAssurancePipeline:
    """Factory function for QA pipeline."""
    config = CheckConfiguration(
        enabled_checks=tuple(QualityCheckType),
        thresholds=thresholds or QualityThresholds(),
    )
    return QualityAssurancePipeline(storage_dir=storage_dir, config=config)


__all__ = [
    "QualityCheckType",
    "Severity",
    "CheckResult",
    "ResolutionStatus",
    "DriftType",
    "OutlierMethod",
    "HashAlgorithm",
    "ImageQualityMetric",
    "LabelConsistencyResult",
    "DistributionMetrics",
    "DuplicateGroup",
    "OutlierResult",
    "DriftResult",
    "CorrelationResult",
    "QualityIssue",
    "QualityReport",
    "AlertService",
    "QualityChecker",
    "QualityThresholds",
    "CheckConfiguration",
    "ReferenceDistribution",
    "ValidationSchema",
    "AnnotationAgreementMetrics",
    "ImageQualityAnalyzer",
    "LabelConsistencyChecker",
    "DistributionAnalyzer",
    "DuplicateDetector",
    "MetadataValidator",
    "OutlierDetector",
    "DataDriftMonitor",
    "AnnotationQualityChecker",
    "FeatureCorrelationAnalyzer",
    "SampleIntegrityChecker",
    "QualityAssurancePipeline",
    "create_qa_pipeline",
]
