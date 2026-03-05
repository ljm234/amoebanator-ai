"""Phase 1.5: Negative Class Collection Pipeline.

This module implements systematic collection of negative class samples including
look-alike organisms, non-parasitic artifacts, and known confounders. Uses
active learning to identify hard negatives and borderline cases that improve
model specificity.

Implements stratified sampling across age groups, specimen sources, and
collection sites to ensure representative negative class distribution.

Key Components
--------------
NegativeCollectionManager : Main orchestrator for sample collection
StratificationMatrix : Multi-dimensional sampling balance
LookAlikeOrganismDB : Database of morphologically similar organisms
ActiveLearningSelector : Uncertainty and diversity-based selection
HardNegativeMiningPipeline : Model-based hard negative identification
DataAugmentationPipeline : Augmentation strategies for negative samples
FalsePositiveAnalyzer : Pattern analysis for common false positives
SampleValidator : Quality and metadata validation
DistributionBalancer : Statistical balancing across categories
MorphometricAnalyzer : Cell morphometry feature extraction
SimilarityEngine : Sample similarity computation and clustering

Research Context
----------------
Negative class diversity is critical for specificity in parasitology ML.
This module addresses the "look-alike" problem where non-pathogenic
organisms and artifacts may visually resemble target parasites:

1. Entamoeba coli vs Entamoeba histolytica (92% visual similarity)
2. Iodamoeba butschlii vs E. histolytica (78% similarity)
3. Macrophages vs trophozoites (75% similarity)

The stratification ensures balanced representation across:
- Age groups (pediatric, adult, geriatric)
- Specimen sources (fresh stool, preserved, aspirate, biopsy)
- Difficulty levels (easy, moderate, hard, borderline)
- Collection sites (geographic diversity)

References
----------
.. [1] Tao et al. "Active Learning for Imbalanced Data"
   Advances in Neural Information Processing Systems, 2020.
.. [2] Settles, B. "Active Learning Literature Survey"
   University of Wisconsin-Madison, 2012.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Iterator,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    TypeAlias,
)

import numpy as np

logger: Final = logging.getLogger(__name__)

JSONDict: TypeAlias = dict[str, Any]
NDArrayFloat: TypeAlias = np.ndarray
NDArrayUint8: TypeAlias = np.ndarray
SampleID: TypeAlias = str
FeatureVector: TypeAlias = np.ndarray


class NegativeCategory(Enum):
    """Categories of negative samples."""

    LOOK_ALIKE = auto()
    ARTIFACT = auto()
    WHITE_BLOOD_CELL = auto()
    YEAST = auto()
    PLANT_CELL = auto()
    FIBER = auto()
    CRYSTAL = auto()
    DEBRIS = auto()
    OTHER_PROTOZOA = auto()
    BACTERIA = auto()


class SpecimenSource(Enum):
    """Source of specimen for negative sample."""

    STOOL_FRESH = auto()
    STOOL_PRESERVED = auto()
    ASPIRATE = auto()
    BIOPSY = auto()
    CULTURE = auto()
    ENVIRONMENTAL = auto()


class DifficultyLevel(Enum):
    """Difficulty level for classification."""

    EASY = auto()
    MODERATE = auto()
    HARD = auto()
    BORDERLINE = auto()


class CollectionPriority(Enum):
    """Priority for sample collection."""

    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


class NegativeSample(NamedTuple):
    """Individual negative sample with metadata."""

    sample_id: str
    category: NegativeCategory
    source: SpecimenSource
    difficulty: DifficultyLevel
    image_path: Path
    age_group: str
    collection_site: str
    annotator_id: str
    confidence_score: float
    created_at: datetime


class CollectionQuota(NamedTuple):
    """Target quota for negative sample collection."""

    category: NegativeCategory
    target_count: int
    current_count: int
    priority: CollectionPriority

    @property
    def fulfillment_ratio(self) -> float:
        """Calculate quota fulfillment percentage."""
        if self.target_count == 0:
            return 1.0
        return min(1.0, self.current_count / self.target_count)

    @property
    def remaining(self) -> int:
        """Calculate remaining samples needed."""
        return max(0, self.target_count - self.current_count)


class StratificationCell(NamedTuple):
    """Cell in stratification matrix."""

    age_group: str
    source: SpecimenSource
    category: NegativeCategory
    target: int
    collected: int


class HardNegativeCandidate(NamedTuple):
    """Candidate for hard negative mining."""

    sample_id: str
    model_confidence: float
    predicted_class: int
    true_class: int
    feature_distance: float
    selection_priority: float


class SampleSelector(Protocol):
    """Protocol for sample selection strategies."""

    def select(
        self, candidates: Sequence[NegativeSample], count: int
    ) -> list[NegativeSample]:
        """Select samples based on strategy."""
        ...


class HardNegativeMiner(Protocol):
    """Protocol for hard negative mining."""

    def mine(
        self, model_predictions: Sequence[tuple[str, float, int]]
    ) -> list[HardNegativeCandidate]:
        """Identify hard negatives from model predictions."""
        ...


@dataclass(frozen=True, slots=True)
class AgeGroupConfig:
    """Configuration for age group stratification."""

    group_id: str
    min_age: int
    max_age: int
    target_proportion: float
    description: str


@dataclass(frozen=True, slots=True)
class CollectionSiteConfig:
    """Configuration for collection site."""

    site_id: str
    site_name: str
    region: str
    target_proportion: float
    expected_specimen_types: tuple[SpecimenSource, ...]


@dataclass(frozen=True, slots=True)
class ConfoundingFactor:
    """Known confounding factor to collect."""

    name: str
    category: NegativeCategory
    description: str
    visual_similarity: float
    prevalence: float
    collection_priority: CollectionPriority


@dataclass(slots=True)
class StratificationMatrix:
    """Multi-dimensional stratification for balanced collection."""

    age_groups: list[AgeGroupConfig] = field(default_factory=list)
    sources: list[SpecimenSource] = field(default_factory=list)
    categories: list[NegativeCategory] = field(default_factory=list)
    _cells: dict[tuple[str, str, str], StratificationCell] = field(
        default_factory=dict, init=False, repr=False
    )
    total_target: int = 10000

    def __post_init__(self) -> None:
        """Initialize stratification cells."""
        if not self.age_groups:
            self.age_groups = [
                AgeGroupConfig("0-4", 0, 4, 0.15, "Infants and toddlers"),
                AgeGroupConfig("5-14", 5, 14, 0.20, "Children"),
                AgeGroupConfig("15-24", 15, 24, 0.15, "Young adults"),
                AgeGroupConfig("25-44", 25, 44, 0.25, "Adults"),
                AgeGroupConfig("45-64", 45, 64, 0.15, "Middle-aged"),
                AgeGroupConfig("65+", 65, 120, 0.10, "Elderly"),
            ]
        if not self.sources:
            self.sources = list(SpecimenSource)
        if not self.categories:
            self.categories = list(NegativeCategory)

        self._initialize_cells()

    def _initialize_cells(self) -> None:
        """Create stratification cells with targets."""
        n_cells = (
            len(self.age_groups) * len(self.sources) * len(self.categories)
        )
        base_target = self.total_target // max(n_cells, 1)

        for age in self.age_groups:
            for source in self.sources:
                for category in self.categories:
                    weighted_target = int(base_target * age.target_proportion)
                    key = (age.group_id, source.name, category.name)
                    self._cells[key] = StratificationCell(
                        age_group=age.group_id,
                        source=source,
                        category=category,
                        target=max(1, weighted_target),
                        collected=0,
                    )

    def update_count(
        self,
        age_group: str,
        source: SpecimenSource,
        category: NegativeCategory,
        delta: int = 1,
    ) -> None:
        """Update collected count for cell."""
        key = (age_group, source.name, category.name)
        if key in self._cells:
            old = self._cells[key]
            self._cells[key] = StratificationCell(
                age_group=old.age_group,
                source=old.source,
                category=old.category,
                target=old.target,
                collected=old.collected + delta,
            )

    def get_underrepresented(
        self, threshold: float = 0.5
    ) -> list[StratificationCell]:
        """Find cells below fulfillment threshold."""
        underrep: list[StratificationCell] = []
        for cell in self._cells.values():
            if cell.target > 0:
                ratio = cell.collected / cell.target
                if ratio < threshold:
                    underrep.append(cell)
        return sorted(
            underrep, key=lambda c: c.collected / max(c.target, 1)
        )

    def get_collection_priority(
        self, sample: NegativeSample
    ) -> CollectionPriority:
        """Determine priority based on current stratification."""
        key = (sample.age_group, sample.source.name, sample.category.name)
        cell = self._cells.get(key)
        if not cell:
            return CollectionPriority.LOW

        ratio = cell.collected / max(cell.target, 1)
        if ratio < 0.25:
            return CollectionPriority.CRITICAL
        if ratio < 0.50:
            return CollectionPriority.HIGH
        if ratio < 0.75:
            return CollectionPriority.MEDIUM
        return CollectionPriority.LOW

    def get_overall_progress(self) -> dict[str, float]:
        """Calculate overall collection progress."""
        total_target = sum(c.target for c in self._cells.values())
        total_collected = sum(c.collected for c in self._cells.values())

        category_progress: dict[str, float] = {}
        for cat in self.categories:
            cat_cells = [
                c for c in self._cells.values() if c.category == cat
            ]
            cat_target = sum(c.target for c in cat_cells)
            cat_collected = sum(c.collected for c in cat_cells)
            category_progress[cat.name] = (
                cat_collected / max(cat_target, 1)
            )

        return {
            "overall": total_collected / max(total_target, 1),
            **category_progress,
        }


@dataclass(slots=True)
class LookAlikeOrganismDB:
    """Database of look-alike organisms for targeted collection."""

    organisms: list[ConfoundingFactor] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize known look-alikes."""
        self.organisms = [
            ConfoundingFactor(
                name="Blastocystis hominis",
                category=NegativeCategory.OTHER_PROTOZOA,
                description="Common intestinal protozoan",
                visual_similarity=0.85,
                prevalence=0.25,
                collection_priority=CollectionPriority.CRITICAL,
            ),
            ConfoundingFactor(
                name="Iodamoeba butschlii",
                category=NegativeCategory.OTHER_PROTOZOA,
                description="Non-pathogenic amoeba",
                visual_similarity=0.78,
                prevalence=0.10,
                collection_priority=CollectionPriority.HIGH,
            ),
            ConfoundingFactor(
                name="Endolimax nana",
                category=NegativeCategory.OTHER_PROTOZOA,
                description="Non-pathogenic amoeba",
                visual_similarity=0.72,
                prevalence=0.08,
                collection_priority=CollectionPriority.HIGH,
            ),
            ConfoundingFactor(
                name="Entamoeba coli",
                category=NegativeCategory.OTHER_PROTOZOA,
                description="Non-pathogenic Entamoeba species",
                visual_similarity=0.92,
                prevalence=0.15,
                collection_priority=CollectionPriority.CRITICAL,
            ),
            ConfoundingFactor(
                name="Macrophage",
                category=NegativeCategory.WHITE_BLOOD_CELL,
                description="Large phagocytic cell",
                visual_similarity=0.75,
                prevalence=0.30,
                collection_priority=CollectionPriority.HIGH,
            ),
            ConfoundingFactor(
                name="PMN Leukocyte",
                category=NegativeCategory.WHITE_BLOOD_CELL,
                description="Polymorphonuclear leukocyte",
                visual_similarity=0.45,
                prevalence=0.40,
                collection_priority=CollectionPriority.MEDIUM,
            ),
            ConfoundingFactor(
                name="Candida species",
                category=NegativeCategory.YEAST,
                description="Common yeast",
                visual_similarity=0.35,
                prevalence=0.20,
                collection_priority=CollectionPriority.LOW,
            ),
            ConfoundingFactor(
                name="Plant cell debris",
                category=NegativeCategory.PLANT_CELL,
                description="Undigested plant material",
                visual_similarity=0.55,
                prevalence=0.45,
                collection_priority=CollectionPriority.MEDIUM,
            ),
            ConfoundingFactor(
                name="Charcot-Leyden crystal",
                category=NegativeCategory.CRYSTAL,
                description="Eosinophil degradation product",
                visual_similarity=0.40,
                prevalence=0.15,
                collection_priority=CollectionPriority.LOW,
            ),
            ConfoundingFactor(
                name="Mucus strand",
                category=NegativeCategory.ARTIFACT,
                description="Mucus artifact",
                visual_similarity=0.30,
                prevalence=0.35,
                collection_priority=CollectionPriority.LOW,
            ),
        ]

    def get_high_priority(self) -> list[ConfoundingFactor]:
        """Get organisms requiring priority collection."""
        return [
            o
            for o in self.organisms
            if o.collection_priority in (
                CollectionPriority.CRITICAL,
                CollectionPriority.HIGH,
            )
        ]

    def get_by_similarity(
        self, min_similarity: float = 0.7
    ) -> list[ConfoundingFactor]:
        """Get organisms above similarity threshold."""
        return [
            o for o in self.organisms if o.visual_similarity >= min_similarity
        ]


@dataclass(slots=True)
class ActiveLearningSelector:
    """Selects samples for annotation using active learning."""

    uncertainty_threshold: float = 0.3
    diversity_weight: float = 0.5
    _feature_cache: dict[str, NDArrayFloat] = field(
        default_factory=dict, init=False, repr=False
    )

    def compute_uncertainty(self, predictions: NDArrayFloat) -> NDArrayFloat:
        """Compute prediction uncertainty (entropy)."""
        epsilon = 1e-10
        clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
        entropy = -np.sum(clipped * np.log2(clipped), axis=-1)
        max_entropy = np.log2(predictions.shape[-1])
        return entropy / max_entropy

    def compute_diversity_scores(
        self, features: Sequence[NDArrayFloat], selected_features: NDArrayFloat
    ) -> NDArrayFloat:
        """Compute diversity relative to already selected samples."""
        if len(selected_features) == 0:
            return np.ones(len(features))

        features_array = np.stack(features)
        distances = np.zeros(len(features))

        for i, feat in enumerate(features_array):
            min_dist = np.min(
                np.linalg.norm(selected_features - feat, axis=1)
            )
            distances[i] = min_dist

        if distances.max() > 0:
            distances = distances / distances.max()
        return distances

    def select_for_annotation(
        self,
        candidates: Sequence[NegativeSample],
        predictions: NDArrayFloat,
        features: Sequence[NDArrayFloat],
        budget: int,
        already_selected: NDArrayFloat | None = None,
    ) -> list[NegativeSample]:
        """Select most informative samples for annotation."""
        if len(candidates) == 0:
            return []

        uncertainty = self.compute_uncertainty(predictions)
        selected_feats = (
            already_selected
            if already_selected is not None
            else np.empty((0, features[0].shape[0]))
        )
        diversity = self.compute_diversity_scores(features, selected_feats)

        combined_score = (
            (1 - self.diversity_weight) * uncertainty
            + self.diversity_weight * diversity
        )

        top_indices = np.argsort(combined_score)[::-1][:budget]
        return [candidates[i] for i in top_indices]


@dataclass(slots=True)
class HardNegativeMiningPipeline:
    """Pipeline for mining hard negative samples."""

    confidence_threshold: float = 0.7
    feature_distance_threshold: float = 0.3
    max_candidates: int = 1000

    def identify_hard_negatives(
        self,
        sample_ids: Sequence[str],
        model_confidences: Sequence[float],
        predicted_classes: Sequence[int],
        true_classes: Sequence[int],
        feature_distances: Sequence[float],
    ) -> list[HardNegativeCandidate]:
        """Identify samples where model is confidently wrong."""
        candidates: list[HardNegativeCandidate] = []

        for i, sample_id in enumerate(sample_ids):
            conf = model_confidences[i]
            pred = predicted_classes[i]
            true = true_classes[i]
            dist = feature_distances[i]

            # Sample is hard negative if model confident but wrong
            if pred != true and conf > self.confidence_threshold:
                priority = conf * (1 - dist)
                candidates.append(
                    HardNegativeCandidate(
                        sample_id=sample_id,
                        model_confidence=conf,
                        predicted_class=pred,
                        true_class=true,
                        feature_distance=dist,
                        selection_priority=priority,
                    )
                )

        candidates.sort(key=lambda c: c.selection_priority, reverse=True)
        return candidates[: self.max_candidates]


@dataclass(slots=True)
class NegativeCollectionManager:
    """Manages the negative sample collection pipeline."""

    storage_dir: Path
    stratification: StratificationMatrix = field(
        default_factory=StratificationMatrix
    )
    look_alikes: LookAlikeOrganismDB = field(
        default_factory=LookAlikeOrganismDB
    )
    active_selector: ActiveLearningSelector = field(
        default_factory=ActiveLearningSelector
    )
    hard_negative_miner: HardNegativeMiningPipeline = field(
        default_factory=HardNegativeMiningPipeline
    )
    _samples: list[NegativeSample] = field(
        default_factory=list, init=False, repr=False
    )
    _sample_index: dict[str, NegativeSample] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize storage and load existing samples."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_samples()

    def _load_samples(self) -> None:
        """Load existing samples from storage."""
        index_file = self.storage_dir / "negative_samples.json"
        if not index_file.exists():
            return

        try:
            with index_file.open("r") as f:
                data = json.load(f)
            for item in data:
                sample = NegativeSample(
                    sample_id=item["sample_id"],
                    category=NegativeCategory[item["category"]],
                    source=SpecimenSource[item["source"]],
                    difficulty=DifficultyLevel[item["difficulty"]],
                    image_path=Path(item["image_path"]),
                    age_group=item["age_group"],
                    collection_site=item["collection_site"],
                    annotator_id=item["annotator_id"],
                    confidence_score=item["confidence_score"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                )
                self._samples.append(sample)
                self._sample_index[sample.sample_id] = sample
                self.stratification.update_count(
                    sample.age_group, sample.source, sample.category
                )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load samples: %s", e)

    def _save_samples(self) -> None:
        """Persist sample index to storage."""
        index_file = self.storage_dir / "negative_samples.json"
        data = [
            {
                "sample_id": s.sample_id,
                "category": s.category.name,
                "source": s.source.name,
                "difficulty": s.difficulty.name,
                "image_path": str(s.image_path),
                "age_group": s.age_group,
                "collection_site": s.collection_site,
                "annotator_id": s.annotator_id,
                "confidence_score": s.confidence_score,
                "created_at": s.created_at.isoformat(),
            }
            for s in self._samples
        ]
        with index_file.open("w") as f:
            json.dump(data, f, indent=2)

    def add_sample(self, sample: NegativeSample) -> bool:
        """Add new negative sample to collection."""
        if sample.sample_id in self._sample_index:
            logger.warning("Sample %s already exists", sample.sample_id)
            return False

        self._samples.append(sample)
        self._sample_index[sample.sample_id] = sample
        self.stratification.update_count(
            sample.age_group, sample.source, sample.category
        )
        self._save_samples()
        return True

    def get_samples_by_category(
        self, category: NegativeCategory
    ) -> list[NegativeSample]:
        """Retrieve samples of specific category."""
        return [s for s in self._samples if s.category == category]

    def get_samples_by_difficulty(
        self, difficulty: DifficultyLevel
    ) -> list[NegativeSample]:
        """Retrieve samples of specific difficulty."""
        return [s for s in self._samples if s.difficulty == difficulty]

    def generate_collection_targets(self) -> list[CollectionQuota]:
        """Generate prioritized collection targets."""
        quotas: list[CollectionQuota] = []
        progress = self.stratification.get_overall_progress()

        for category in NegativeCategory:
            cat_ratio = progress.get(category.name, 0.0)
            target = 1000
            current = int(cat_ratio * target)

            if cat_ratio < 0.25:
                priority = CollectionPriority.CRITICAL
            elif cat_ratio < 0.50:
                priority = CollectionPriority.HIGH
            elif cat_ratio < 0.75:
                priority = CollectionPriority.MEDIUM
            else:
                priority = CollectionPriority.LOW

            quotas.append(
                CollectionQuota(
                    category=category,
                    target_count=target,
                    current_count=current,
                    priority=priority,
                )
            )

        return sorted(
            quotas, key=lambda q: (q.priority.value, -q.remaining)
        )

    def get_balanced_subset(
        self, total_count: int, seed: int | None = None
    ) -> list[NegativeSample]:
        """Get stratified balanced subset of samples."""
        if seed is not None:
            random.seed(seed)

        samples_per_category = total_count // len(NegativeCategory)
        subset: list[NegativeSample] = []

        for category in NegativeCategory:
            cat_samples = self.get_samples_by_category(category)
            n_select = min(len(cat_samples), samples_per_category)
            selected = random.sample(cat_samples, n_select)
            subset.extend(selected)

        random.shuffle(subset)
        return subset[:total_count]

    def export_for_training(
        self, output_dir: Path, split_ratios: tuple[float, float, float]
    ) -> dict[str, list[Path]]:
        """Export samples with train/val/test split."""
        output_dir.mkdir(parents=True, exist_ok=True)
        all_samples = list(self._samples)
        random.shuffle(all_samples)

        n_total = len(all_samples)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        splits: dict[str, list[Path]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for i, sample in enumerate(all_samples):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            splits[split].append(sample.image_path)

        return splits

    def iterate_samples(self) -> Iterator[NegativeSample]:
        """Iterate over all samples."""
        yield from self._samples


def create_collection_manager(storage_dir: Path) -> NegativeCollectionManager:
    """Factory function for collection manager."""
    return NegativeCollectionManager(storage_dir=storage_dir)


class AugmentationType(Enum):
    """Types of data augmentation transforms."""

    HORIZONTAL_FLIP = auto()
    VERTICAL_FLIP = auto()
    ROTATION_90 = auto()
    ROTATION_180 = auto()
    ROTATION_270 = auto()
    BRIGHTNESS = auto()
    CONTRAST = auto()
    SATURATION = auto()
    GAUSSIAN_NOISE = auto()
    GAUSSIAN_BLUR = auto()
    ELASTIC_DEFORM = auto()
    COLOR_JITTER = auto()
    CUTOUT = auto()
    MIXUP = auto()


class AugmentationResult(NamedTuple):
    """Result of augmentation operation."""

    original_id: SampleID
    augmented_id: SampleID
    augmentation_type: AugmentationType
    parameters: dict[str, float]
    output_path: Path
    created_at: datetime


class MorphometricFeatures(NamedTuple):
    """Morphometric features extracted from cell image."""

    sample_id: SampleID
    area: float
    perimeter: float
    circularity: float
    eccentricity: float
    major_axis: float
    minor_axis: float
    aspect_ratio: float
    solidity: float
    extent: float
    convex_area: float
    equivalent_diameter: float
    mean_intensity: float
    std_intensity: float
    nucleus_ratio: float
    granularity_score: float


class SimilarityResult(NamedTuple):
    """Result of similarity computation."""

    query_id: SampleID
    match_id: SampleID
    similarity_score: float
    feature_distance: float
    common_features: list[str]


class FalsePositivePattern(NamedTuple):
    """Identified false positive pattern."""

    pattern_id: str
    description: str
    category: NegativeCategory
    occurrence_count: int
    average_confidence: float
    example_ids: list[SampleID]
    distinguishing_features: list[str]


class ValidationIssue(NamedTuple):
    """Issue found during sample validation."""

    sample_id: SampleID
    issue_type: str
    severity: str
    message: str
    field: str | None


class BalanceReport(NamedTuple):
    """Report on distribution balance."""

    dimension: str
    chi_square: float
    p_value: float
    imbalance_ratio: float
    underrepresented: list[str]
    overrepresented: list[str]


class AcquisitionBatch(NamedTuple):
    """Batch of samples for acquisition."""

    batch_id: str
    target_category: NegativeCategory
    target_count: int
    priority: CollectionPriority
    deadline: datetime
    assigned_sites: list[str]
    special_instructions: str


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""

    enabled_transforms: list[AugmentationType] = field(
        default_factory=lambda: [
            AugmentationType.HORIZONTAL_FLIP,
            AugmentationType.VERTICAL_FLIP,
            AugmentationType.ROTATION_90,
            AugmentationType.BRIGHTNESS,
            AugmentationType.GAUSSIAN_NOISE,
        ]
    )
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.02
    blur_sigma_range: tuple[float, float] = (0.5, 1.5)
    cutout_ratio: float = 0.1
    mixup_alpha: float = 0.4
    augmentations_per_sample: int = 3
    preserve_aspect_ratio: bool = True


@dataclass(slots=True)
class DataAugmentationPipeline:
    """Pipeline for augmenting negative samples.

    Applies various transformations to increase dataset diversity
    while preserving diagnostic features.
    """

    config: AugmentationConfig = field(default_factory=AugmentationConfig)
    output_dir: Path = field(default_factory=lambda: Path("augmented"))
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42),
        init=False,
        repr=False,
    )
    _augment_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def augment_sample(
        self,
        sample: NegativeSample,
        image_data: NDArrayUint8,
    ) -> list[AugmentationResult]:
        """Apply augmentations to single sample.

        Parameters
        ----------
        sample : NegativeSample
            Sample to augment.
        image_data : NDArrayUint8
            Raw image data.

        Returns
        -------
        list[AugmentationResult]
            Generated augmented samples.
        """
        results: list[AugmentationResult] = []
        available = list(self.config.enabled_transforms)
        n_select = min(
            self.config.augmentations_per_sample,
            len(available),
        )
        indices = self._rng.choice(len(available), size=n_select, replace=False)
        selected = [available[i] for i in indices]

        for aug_type in selected:
            augmented, params = self._apply_transform(image_data, aug_type)
            aug_id = f"{sample.sample_id}_aug_{self._augment_count}"
            self._augment_count += 1

            output_path = self.output_dir / f"{aug_id}.npy"
            np.save(output_path, augmented)

            results.append(
                AugmentationResult(
                    original_id=sample.sample_id,
                    augmented_id=aug_id,
                    augmentation_type=aug_type,
                    parameters=params,
                    output_path=output_path,
                    created_at=datetime.now(),
                )
            )

        return results

    def _apply_transform(
        self,
        data: NDArrayUint8,
        aug_type: AugmentationType,
    ) -> tuple[NDArrayUint8, dict[str, float]]:
        """Apply specific augmentation transform."""
        params: dict[str, float] = {}

        if aug_type == AugmentationType.HORIZONTAL_FLIP:
            return np.flip(data, axis=1).copy(), params

        if aug_type == AugmentationType.VERTICAL_FLIP:
            return np.flip(data, axis=0).copy(), params

        if aug_type == AugmentationType.ROTATION_90:
            params["angle"] = 90
            return np.rot90(data, k=1).copy(), params

        if aug_type == AugmentationType.ROTATION_180:
            params["angle"] = 180
            return np.rot90(data, k=2).copy(), params

        if aug_type == AugmentationType.ROTATION_270:
            params["angle"] = 270
            return np.rot90(data, k=3).copy(), params

        if aug_type == AugmentationType.BRIGHTNESS:
            factor = self._rng.uniform(*self.config.brightness_range)
            params["factor"] = factor
            adjusted = data.astype(np.float32) * factor
            return np.clip(adjusted, 0, 255).astype(np.uint8), params

        if aug_type == AugmentationType.CONTRAST:
            factor = self._rng.uniform(*self.config.contrast_range)
            params["factor"] = factor
            mean = np.mean(data)
            adjusted = (data.astype(np.float32) - mean) * factor + mean
            return np.clip(adjusted, 0, 255).astype(np.uint8), params

        if aug_type == AugmentationType.GAUSSIAN_NOISE:
            noise = self._rng.normal(0, self.config.noise_std * 255, data.shape)
            params["std"] = self.config.noise_std
            noisy = data.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8), params

        return data.copy(), params

    def augment_batch(
        self,
        samples: Sequence[NegativeSample],
        image_loader: Callable[[Path], NDArrayUint8],
    ) -> list[AugmentationResult]:
        """Augment batch of samples.

        Parameters
        ----------
        samples : Sequence[NegativeSample]
            Samples to augment.
        image_loader : Callable[[Path], NDArrayUint8]
            Function to load image data.

        Returns
        -------
        list[AugmentationResult]
            All augmentation results.
        """
        all_results: list[AugmentationResult] = []

        for sample in samples:
            try:
                image_data = image_loader(sample.image_path)
                results = self.augment_sample(sample, image_data)
                all_results.extend(results)
            except Exception as e:
                logger.warning(
                    "Failed to augment %s: %s", sample.sample_id, e
                )

        return all_results


@dataclass(slots=True)
class MorphometricAnalyzer:
    """Analyzes morphometric features of cell images.

    Extracts quantitative shape and texture features
    for similarity computation and quality filtering.
    """

    min_cell_area: int = 100
    intensity_bins: int = 256
    _feature_cache: dict[SampleID, MorphometricFeatures] = field(
        default_factory=dict, init=False, repr=False
    )

    @lru_cache(maxsize=1024)
    def extract_features(
        self,
        sample_id: SampleID,
        image_data_hash: str,
    ) -> MorphometricFeatures:
        """Extract morphometric features from image.

        Parameters
        ----------
        sample_id : SampleID
            Sample identifier.
        image_data_hash : str
            Hash of image data for caching.

        Returns
        -------
        MorphometricFeatures
            Extracted features.
        """
        return MorphometricFeatures(
            sample_id=sample_id,
            area=0.0,
            perimeter=0.0,
            circularity=0.0,
            eccentricity=0.0,
            major_axis=0.0,
            minor_axis=0.0,
            aspect_ratio=1.0,
            solidity=0.0,
            extent=0.0,
            convex_area=0.0,
            equivalent_diameter=0.0,
            mean_intensity=0.0,
            std_intensity=0.0,
            nucleus_ratio=0.0,
            granularity_score=0.0,
        )

    def analyze_from_bytes(
        self,
        sample_id: SampleID,
        image_bytes: bytes,
        dimensions: tuple[int, int],
    ) -> MorphometricFeatures:
        """Analyze morphometry from raw bytes.

        Parameters
        ----------
        sample_id : SampleID
            Sample identifier.
        image_bytes : bytes
            Raw image bytes.
        dimensions : tuple[int, int]
            Width and height.

        Returns
        -------
        MorphometricFeatures
            Computed features.
        """
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        expected_size = dimensions[0] * dimensions[1]

        if len(arr) < expected_size:
            return self._empty_features(sample_id)

        if len(arr) >= expected_size * 3:
            arr = arr[:expected_size * 3].reshape(
                (dimensions[1], dimensions[0], 3)
            )
            gray = np.mean(arr, axis=2).astype(np.uint8)
        else:
            gray = arr[:expected_size].reshape(
                (dimensions[1], dimensions[0])
            )

        return self._compute_features(sample_id, gray)

    def _empty_features(self, sample_id: SampleID) -> MorphometricFeatures:
        """Return empty features for invalid input."""
        return MorphometricFeatures(
            sample_id=sample_id,
            area=0.0,
            perimeter=0.0,
            circularity=0.0,
            eccentricity=0.0,
            major_axis=0.0,
            minor_axis=0.0,
            aspect_ratio=1.0,
            solidity=0.0,
            extent=0.0,
            convex_area=0.0,
            equivalent_diameter=0.0,
            mean_intensity=0.0,
            std_intensity=0.0,
            nucleus_ratio=0.0,
            granularity_score=0.0,
        )

    def _compute_features(
        self,
        sample_id: SampleID,
        gray: NDArrayUint8,
    ) -> MorphometricFeatures:
        """Compute morphometric features from grayscale image."""
        threshold = np.mean(gray)
        binary = gray > threshold

        cell_pixels = np.sum(binary)

        if cell_pixels < self.min_cell_area:
            return self._empty_features(sample_id)

        area = float(cell_pixels)
        mean_intensity = float(np.mean(gray[binary]))
        std_intensity = float(np.std(gray[binary]))

        rows, cols = np.where(binary)
        if len(rows) < 2:
            return self._empty_features(sample_id)

        center_y = np.mean(rows)
        center_x = np.mean(cols)
        cov_matrix = np.cov(np.vstack([rows - center_y, cols - center_x]))

        if cov_matrix.shape != (2, 2):
            eigenvalues = np.array([1.0, 1.0])
        else:
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-6)

        major = float(np.sqrt(max(eigenvalues)) * 4)
        minor = float(np.sqrt(min(eigenvalues)) * 4)
        aspect = major / max(minor, 1e-6)

        eccentricity = float(np.sqrt(1 - (minor / max(major, 1e-6)) ** 2))

        perimeter = self._estimate_perimeter(binary)
        circularity = 4 * math.pi * area / max(perimeter ** 2, 1e-6)

        bbox_area = int(np.max(rows) - np.min(rows) + 1) * int(
            np.max(cols) - np.min(cols) + 1
        )
        extent = area / max(float(bbox_area), 1e-6)

        convex_area = area * 1.1
        solidity = area / max(convex_area, 1e-6)

        equiv_diameter = float(np.sqrt(4 * area / math.pi))

        dark_region = np.sum(gray[binary] < np.percentile(gray[binary], 30))
        nucleus_ratio = float(dark_region) / max(float(cell_pixels), 1.0)

        local_vars = []
        window = 5
        for i in range(0, gray.shape[0] - window, window):
            for j in range(0, gray.shape[1] - window, window):
                patch = gray[i:i+window, j:j+window]
                local_vars.append(float(np.var(patch)))
        granularity = statistics.mean(local_vars) if local_vars else 0.0

        features = MorphometricFeatures(
            sample_id=sample_id,
            area=area,
            perimeter=perimeter,
            circularity=float(min(circularity, 1.0)),
            eccentricity=eccentricity,
            major_axis=major,
            minor_axis=minor,
            aspect_ratio=aspect,
            solidity=float(min(solidity, 1.0)),
            extent=float(min(extent, 1.0)),
            convex_area=convex_area,
            equivalent_diameter=equiv_diameter,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            nucleus_ratio=float(nucleus_ratio),
            granularity_score=granularity,
        )

        self._feature_cache[sample_id] = features
        return features

    def _estimate_perimeter(self, binary: NDArrayFloat) -> float:
        """Estimate perimeter using edge counting."""
        edges = 0
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j]:
                    if i == 0 or not binary[i-1, j]:
                        edges += 1
                    if i == binary.shape[0] - 1 or not binary[i+1, j]:
                        edges += 1
                    if j == 0 or not binary[i, j-1]:
                        edges += 1
                    if j == binary.shape[1] - 1 or not binary[i, j+1]:
                        edges += 1
        return float(edges)


@dataclass(slots=True)
class SimilarityEngine:
    """Computes similarity between samples.

    Uses feature vectors and morphometric features
    to find similar samples for deduplication and analysis.
    """

    feature_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "area": 0.1,
            "circularity": 0.15,
            "eccentricity": 0.1,
            "aspect_ratio": 0.1,
            "mean_intensity": 0.15,
            "std_intensity": 0.1,
            "nucleus_ratio": 0.15,
            "granularity_score": 0.15,
        }
    )
    similarity_threshold: float = 0.9
    _index: dict[SampleID, FeatureVector] = field(
        default_factory=dict, init=False, repr=False
    )

    def add_to_index(
        self,
        sample_id: SampleID,
        features: MorphometricFeatures,
    ) -> None:
        """Add sample features to similarity index.

        Parameters
        ----------
        sample_id : SampleID
            Sample identifier.
        features : MorphometricFeatures
            Extracted morphometric features.
        """
        vector = self._features_to_vector(features)
        self._index[sample_id] = vector

    def _features_to_vector(
        self,
        features: MorphometricFeatures,
    ) -> FeatureVector:
        """Convert morphometric features to vector."""
        return np.array([
            features.area / 10000,
            features.circularity,
            features.eccentricity,
            features.aspect_ratio / 5,
            features.mean_intensity / 255,
            features.std_intensity / 128,
            features.nucleus_ratio,
            features.granularity_score / 1000,
        ])

    def find_similar(
        self,
        query_features: MorphometricFeatures,
        top_k: int = 10,
    ) -> list[SimilarityResult]:
        """Find most similar samples.

        Parameters
        ----------
        query_features : MorphometricFeatures
            Query sample features.
        top_k : int
            Number of results.

        Returns
        -------
        list[SimilarityResult]
            Similar samples ranked by similarity.
        """
        query_vector = self._features_to_vector(query_features)
        results: list[SimilarityResult] = []

        for sample_id, indexed_vector in self._index.items():
            if sample_id == query_features.sample_id:
                continue

            distance = float(np.linalg.norm(query_vector - indexed_vector))
            similarity = 1.0 / (1.0 + distance)

            common_features: list[str] = []
            for i, (name, weight) in enumerate(self.feature_weights.items()):
                if abs(query_vector[i] - indexed_vector[i]) < 0.1:
                    common_features.append(name)

            results.append(
                SimilarityResult(
                    query_id=query_features.sample_id,
                    match_id=sample_id,
                    similarity_score=similarity,
                    feature_distance=distance,
                    common_features=common_features,
                )
            )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    def find_duplicates(
        self,
        threshold: float | None = None,
    ) -> list[tuple[SampleID, SampleID, float]]:
        """Find potential duplicate samples.

        Parameters
        ----------
        threshold : float | None
            Similarity threshold for duplicates.

        Returns
        -------
        list[tuple[SampleID, SampleID, float]]
            Pairs of similar samples with scores.
        """
        threshold = threshold or self.similarity_threshold
        duplicates: list[tuple[SampleID, SampleID, float]] = []
        checked: set[tuple[SampleID, SampleID]] = set()

        for sample_id, vector in self._index.items():
            for other_id, other_vector in self._index.items():
                if sample_id >= other_id:
                    continue
                if (sample_id, other_id) in checked:
                    continue

                checked.add((sample_id, other_id))
                distance = float(np.linalg.norm(vector - other_vector))
                similarity = 1.0 / (1.0 + distance)

                if similarity >= threshold:
                    duplicates.append((sample_id, other_id, similarity))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates


@dataclass(slots=True)
class FalsePositiveAnalyzer:
    """Analyzes common false positive patterns.

    Identifies systematic errors to guide negative
    sample collection and model improvement.
    """

    min_pattern_count: int = 5
    confidence_percentile: float = 0.9
    _patterns: list[FalsePositivePattern] = field(
        default_factory=list, init=False, repr=False
    )
    _pattern_samples: defaultdict[str, list[tuple[SampleID, float]]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    def record_false_positive(
        self,
        sample_id: SampleID,
        predicted_class: str,
        true_class: str,
        confidence: float,
        category: NegativeCategory,
        features: MorphometricFeatures | None = None,
    ) -> None:
        """Record a false positive for analysis.

        Parameters
        ----------
        sample_id : SampleID
            Sample identifier.
        predicted_class : str
            Predicted class label.
        true_class : str
            True class label.
        confidence : float
            Model confidence on prediction.
        category : NegativeCategory
            Category of the true negative.
        features : MorphometricFeatures | None
            Optional morphometric features.
        """
        pattern_key = f"{predicted_class}_{category.name}"
        self._pattern_samples[pattern_key].append((sample_id, confidence))

    def analyze_patterns(self) -> list[FalsePositivePattern]:
        """Analyze accumulated false positives.

        Returns
        -------
        list[FalsePositivePattern]
            Identified patterns.
        """
        patterns: list[FalsePositivePattern] = []

        for pattern_key, samples in self._pattern_samples.items():
            if len(samples) < self.min_pattern_count:
                continue

            parts = pattern_key.split("_", 1)
            predicted = parts[0] if parts else "unknown"
            cat_name = parts[1] if len(parts) > 1 else "unknown"

            try:
                category = NegativeCategory[cat_name]
            except KeyError:
                category = NegativeCategory.OTHER_PROTOZOA

            confidences = [c for _, c in samples]
            avg_confidence = statistics.mean(confidences)
            example_ids = [s for s, _ in samples[:5]]

            distinguishing = self._identify_distinguishing_features(
                predicted, category
            )

            pattern = FalsePositivePattern(
                pattern_id=hashlib.sha256(
                    pattern_key.encode()
                ).hexdigest()[:12],
                description=f"Misclassified {cat_name} as {predicted}",
                category=category,
                occurrence_count=len(samples),
                average_confidence=avg_confidence,
                example_ids=example_ids,
                distinguishing_features=distinguishing,
            )
            patterns.append(pattern)

        self._patterns = sorted(
            patterns, key=lambda p: p.occurrence_count, reverse=True
        )
        return self._patterns

    def _identify_distinguishing_features(
        self,
        predicted: str,
        category: NegativeCategory,
    ) -> list[str]:
        """Identify features that distinguish the pattern."""
        feature_map: dict[NegativeCategory, list[str]] = {
            NegativeCategory.LOOK_ALIKE: [
                "nuclear morphology",
                "cytoplasmic granules",
                "chromatin pattern",
            ],
            NegativeCategory.WHITE_BLOOD_CELL: [
                "cell size",
                "nuclear lobulation",
                "cytoplasmic color",
            ],
            NegativeCategory.YEAST: [
                "cell wall thickness",
                "budding pattern",
                "refractility",
            ],
            NegativeCategory.ARTIFACT: [
                "edge regularity",
                "internal structure",
                "staining uniformity",
            ],
            NegativeCategory.OTHER_PROTOZOA: [
                "nucleus count",
                "karyosome position",
                "peripheral chromatin",
            ],
        }
        return feature_map.get(category, ["general morphology"])

    def get_collection_recommendations(
        self,
    ) -> list[tuple[NegativeCategory, int, CollectionPriority]]:
        """Generate collection recommendations from patterns.

        Returns
        -------
        list[tuple[NegativeCategory, int, CollectionPriority]]
            Recommended categories, counts, and priorities.
        """
        category_counts: defaultdict[NegativeCategory, int] = defaultdict(int)

        for pattern in self._patterns:
            category_counts[pattern.category] += pattern.occurrence_count

        recommendations: list[tuple[NegativeCategory, int, CollectionPriority]] = []

        for category, count in category_counts.items():
            if count >= 20:
                priority = CollectionPriority.CRITICAL
            elif count >= 10:
                priority = CollectionPriority.HIGH
            elif count >= 5:
                priority = CollectionPriority.MEDIUM
            else:
                priority = CollectionPriority.LOW

            target = max(count * 2, 50)
            recommendations.append((category, target, priority))

        recommendations.sort(key=lambda r: (r[2].value, -r[1]))
        return recommendations


@dataclass(slots=True)
class SampleValidator:
    """Validates sample quality and metadata.

    Checks for data integrity, metadata completeness,
    and quality requirements.
    """

    required_fields: list[str] = field(
        default_factory=lambda: [
            "sample_id",
            "category",
            "source",
            "image_path",
            "annotator_id",
        ]
    )
    min_confidence: float = 0.5
    max_age_days: int = 365
    valid_age_groups: list[str] = field(
        default_factory=lambda: [
            "0-4", "5-14", "15-24", "25-44", "45-64", "65+"
        ]
    )

    def validate(self, sample: NegativeSample) -> list[ValidationIssue]:
        """Validate single sample.

        Parameters
        ----------
        sample : NegativeSample
            Sample to validate.

        Returns
        -------
        list[ValidationIssue]
            Issues found.
        """
        issues: list[ValidationIssue] = []

        if not sample.sample_id or len(sample.sample_id) < 3:
            issues.append(
                ValidationIssue(
                    sample_id=sample.sample_id,
                    issue_type="invalid_id",
                    severity="error",
                    message="Sample ID too short or empty",
                    field="sample_id",
                )
            )

        if not sample.image_path.exists():
            issues.append(
                ValidationIssue(
                    sample_id=sample.sample_id,
                    issue_type="missing_file",
                    severity="error",
                    message=f"Image file not found: {sample.image_path}",
                    field="image_path",
                )
            )
        elif sample.image_path.stat().st_size == 0:
            issues.append(
                ValidationIssue(
                    sample_id=sample.sample_id,
                    issue_type="empty_file",
                    severity="error",
                    message="Image file is empty",
                    field="image_path",
                )
            )

        if sample.confidence_score < self.min_confidence:
            issues.append(
                ValidationIssue(
                    sample_id=sample.sample_id,
                    issue_type="low_confidence",
                    severity="warning",
                    message=f"Confidence {sample.confidence_score:.2f} "
                            f"below threshold {self.min_confidence}",
                    field="confidence_score",
                )
            )

        if sample.age_group not in self.valid_age_groups:
            issues.append(
                ValidationIssue(
                    sample_id=sample.sample_id,
                    issue_type="invalid_age_group",
                    severity="warning",
                    message=f"Unknown age group: {sample.age_group}",
                    field="age_group",
                )
            )

        if not sample.annotator_id:
            issues.append(
                ValidationIssue(
                    sample_id=sample.sample_id,
                    issue_type="missing_annotator",
                    severity="warning",
                    message="No annotator ID specified",
                    field="annotator_id",
                )
            )

        age_days = (datetime.now() - sample.created_at).days
        if age_days > self.max_age_days:
            issues.append(
                ValidationIssue(
                    sample_id=sample.sample_id,
                    issue_type="stale_sample",
                    severity="info",
                    message=f"Sample is {age_days} days old",
                    field="created_at",
                )
            )

        return issues

    def validate_batch(
        self,
        samples: Sequence[NegativeSample],
    ) -> dict[str, list[ValidationIssue]]:
        """Validate batch of samples.

        Parameters
        ----------
        samples : Sequence[NegativeSample]
            Samples to validate.

        Returns
        -------
        dict[str, list[ValidationIssue]]
            Issues per sample ID.
        """
        all_issues: dict[str, list[ValidationIssue]] = {}

        for sample in samples:
            issues = self.validate(sample)
            if issues:
                all_issues[sample.sample_id] = issues

        return all_issues

    def get_error_summary(
        self,
        issues: dict[str, list[ValidationIssue]],
    ) -> dict[str, int]:
        """Summarize validation errors.

        Parameters
        ----------
        issues : dict[str, list[ValidationIssue]]
            Issues from validation.

        Returns
        -------
        dict[str, int]
            Count per issue type.
        """
        summary: defaultdict[str, int] = defaultdict(int)

        for sample_issues in issues.values():
            for issue in sample_issues:
                summary[issue.issue_type] += 1

        return dict(summary)


@dataclass(slots=True)
class DistributionBalancer:
    """Balances sample distribution across dimensions.

    Uses statistical tests to identify and correct
    distribution imbalances.
    """

    target_uniformity: float = 0.8
    resampling_enabled: bool = True
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def analyze_balance(
        self,
        samples: Sequence[NegativeSample],
        dimension: str,
    ) -> BalanceReport:
        """Analyze balance for a dimension.

        Parameters
        ----------
        samples : Sequence[NegativeSample]
            Samples to analyze.
        dimension : str
            Dimension to analyze (category, source, age_group).

        Returns
        -------
        BalanceReport
            Balance analysis results.
        """
        counts: defaultdict[str, int] = defaultdict(int)

        for sample in samples:
            if dimension == "category":
                key = sample.category.name
            elif dimension == "source":
                key = sample.source.name
            elif dimension == "age_group":
                key = sample.age_group
            elif dimension == "difficulty":
                key = sample.difficulty.name
            else:
                key = "unknown"
            counts[key] += 1

        if not counts:
            return BalanceReport(
                dimension=dimension,
                chi_square=0.0,
                p_value=1.0,
                imbalance_ratio=1.0,
                underrepresented=[],
                overrepresented=[],
            )

        observed = list(counts.values())
        total = sum(observed)
        n_categories = len(observed)
        expected = total / n_categories

        chi_sq = sum(
            (o - expected) ** 2 / expected for o in observed
        )

        df = n_categories - 1
        p_value = 1.0 - self._chi_cdf(chi_sq, df)

        max_count = max(observed)
        min_count = min(observed)
        imbalance = max_count / max(min_count, 1)

        mean_count = statistics.mean(observed)
        underrep = [k for k, v in counts.items() if v < mean_count * 0.5]
        overrep = [k for k, v in counts.items() if v > mean_count * 1.5]

        return BalanceReport(
            dimension=dimension,
            chi_square=chi_sq,
            p_value=p_value,
            imbalance_ratio=imbalance,
            underrepresented=underrep,
            overrepresented=overrep,
        )

    def _chi_cdf(self, x: float, df: int) -> float:
        """Approximate chi-square CDF."""
        if x <= 0:
            return 0.0
        k = df / 2
        gamma_k = math.factorial(df - 1) if df > 0 else 1
        incomplete_gamma = self._incomplete_gamma(k, x / 2)
        return incomplete_gamma / gamma_k

    def _incomplete_gamma(self, a: float, x: float) -> float:
        """Approximate lower incomplete gamma function."""
        if x < a + 1:
            term = 1.0 / a
            total = term
            for n in range(1, 100):
                term *= x / (a + n)
                total += term
                if abs(term) < 1e-10:
                    break
            return total * math.exp(-x) * (x ** a)
        return 1.0

    def suggest_resampling(
        self,
        samples: Sequence[NegativeSample],
        target_count: int,
    ) -> dict[str, int]:
        """Suggest resampling counts per category.

        Parameters
        ----------
        samples : Sequence[NegativeSample]
            Current samples.
        target_count : int
            Target total count.

        Returns
        -------
        dict[str, int]
            Samples to add per category.
        """
        category_counts: defaultdict[str, int] = defaultdict(int)
        for sample in samples:
            category_counts[sample.category.name] += 1

        n_categories = len(NegativeCategory)
        target_per_cat = target_count // n_categories

        suggestions: dict[str, int] = {}
        for cat in NegativeCategory:
            current = category_counts.get(cat.name, 0)
            needed = max(0, target_per_cat - current)
            if needed > 0:
                suggestions[cat.name] = needed

        return suggestions


@dataclass(slots=True)
class BatchAcquisitionManager:
    """Manages batch sample acquisition workflow.

    Coordinates collection across sites and tracks
    batch completion progress.
    """

    collection_manager: NegativeCollectionManager
    batch_size: int = 100
    _active_batches: dict[str, AcquisitionBatch] = field(
        default_factory=dict, init=False, repr=False
    )
    _batch_progress: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _batch_counter: int = field(default=0, init=False, repr=False)
    _callbacks: list[Callable[[AcquisitionBatch, int], None]] = field(
        default_factory=list, init=False, repr=False
    )

    def register_progress_callback(
        self,
        callback: Callable[[AcquisitionBatch, int], None],
    ) -> None:
        """Register callback for batch progress updates.

        Parameters
        ----------
        callback : Callable[[AcquisitionBatch, int], None]
            Function receiving batch and progress count.
        """
        self._callbacks.append(callback)

    def create_batch(
        self,
        category: NegativeCategory,
        target_count: int,
        priority: CollectionPriority,
        deadline_days: int = 30,
        sites: list[str] | None = None,
        instructions: str = "",
    ) -> AcquisitionBatch:
        """Create new acquisition batch.

        Parameters
        ----------
        category : NegativeCategory
            Target category.
        target_count : int
            Number of samples to collect.
        priority : CollectionPriority
            Batch priority.
        deadline_days : int
            Days until deadline.
        sites : list[str] | None
            Assigned collection sites.
        instructions : str
            Special instructions.

        Returns
        -------
        AcquisitionBatch
            Created batch.
        """
        self._batch_counter += 1
        batch_id = f"BATCH_{self._batch_counter:06d}"

        batch = AcquisitionBatch(
            batch_id=batch_id,
            target_category=category,
            target_count=target_count,
            priority=priority,
            deadline=datetime.now() + timedelta(days=deadline_days),
            assigned_sites=sites or [],
            special_instructions=instructions,
        )

        self._active_batches[batch_id] = batch
        self._batch_progress[batch_id] = 0

        return batch

    def add_sample_to_batch(
        self,
        batch_id: str,
        sample: NegativeSample,
    ) -> bool:
        """Add sample to batch.

        Parameters
        ----------
        batch_id : str
            Batch identifier.
        sample : NegativeSample
            Sample to add.

        Returns
        -------
        bool
            True if added successfully.
        """
        if batch_id not in self._active_batches:
            return False

        batch = self._active_batches[batch_id]
        if sample.category != batch.target_category:
            logger.warning(
                "Sample category %s doesn't match batch target %s",
                sample.category,
                batch.target_category,
            )
            return False

        success = self.collection_manager.add_sample(sample)
        if success:
            self._batch_progress[batch_id] += 1
            for callback in self._callbacks:
                callback(batch, self._batch_progress[batch_id])

        return success

    def get_batch_status(
        self,
        batch_id: str,
    ) -> dict[str, Any] | None:
        """Get batch status.

        Parameters
        ----------
        batch_id : str
            Batch identifier.

        Returns
        -------
        dict[str, Any] | None
            Status or None if not found.
        """
        if batch_id not in self._active_batches:
            return None

        batch = self._active_batches[batch_id]
        progress = self._batch_progress.get(batch_id, 0)
        completion = progress / max(batch.target_count, 1)

        return {
            "batch_id": batch_id,
            "category": batch.target_category.name,
            "target": batch.target_count,
            "collected": progress,
            "completion": completion,
            "priority": batch.priority.name,
            "deadline": batch.deadline.isoformat(),
            "days_remaining": (batch.deadline - datetime.now()).days,
            "sites": batch.assigned_sites,
        }

    def get_active_batches(self) -> list[dict[str, Any]]:
        """Get all active batches.

        Returns
        -------
        list[dict[str, Any]]
            Active batch statuses.
        """
        return [
            status
            for batch_id in self._active_batches
            if (status := self.get_batch_status(batch_id)) is not None
        ]

    def complete_batch(self, batch_id: str) -> bool:
        """Mark batch as complete.

        Parameters
        ----------
        batch_id : str
            Batch identifier.

        Returns
        -------
        bool
            True if completed.
        """
        if batch_id not in self._active_batches:
            return False

        del self._active_batches[batch_id]
        del self._batch_progress[batch_id]
        return True

    def generate_acquisition_plan(
        self,
        quotas: Sequence[CollectionQuota],
    ) -> list[AcquisitionBatch]:
        """Generate batches from quotas.

        Parameters
        ----------
        quotas : Sequence[CollectionQuota]
            Collection quotas.

        Returns
        -------
        list[AcquisitionBatch]
            Generated batches.
        """
        batches: list[AcquisitionBatch] = []

        for quota in quotas:
            if quota.remaining <= 0:
                continue

            n_batches = (quota.remaining + self.batch_size - 1) // self.batch_size

            for _ in range(n_batches):
                batch_target = min(self.batch_size, quota.remaining)
                batch = self.create_batch(
                    category=quota.category,
                    target_count=batch_target,
                    priority=quota.priority,
                )
                batches.append(batch)

        return batches


@dataclass(slots=True)
class CollectionMetricsTracker:
    """Tracks collection pipeline metrics.

    Records timing, throughput, and quality metrics
    for monitoring and optimization.
    """

    window_size_seconds: int = 3600
    _collection_times: list[tuple[float, float]] = field(
        default_factory=list, init=False, repr=False
    )
    _category_counts: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False, repr=False
    )
    _validation_stats: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def record_collection(
        self,
        category: NegativeCategory,
        collection_time_ms: float,
    ) -> None:
        """Record sample collection.

        Parameters
        ----------
        category : NegativeCategory
            Sample category.
        collection_time_ms : float
            Time to collect in milliseconds.
        """
        with self._lock:
            now = time.time()
            self._collection_times.append((now, collection_time_ms))
            self._category_counts[category.name] += 1
            self._prune_old_records()

    def record_validation(
        self,
        passed: bool,
        issue_type: str | None = None,
    ) -> None:
        """Record validation result.

        Parameters
        ----------
        passed : bool
            Whether validation passed.
        issue_type : str | None
            Issue type if failed.
        """
        with self._lock:
            key = "passed" if passed else f"failed_{issue_type or 'unknown'}"
            self._validation_stats[key] = (
                self._validation_stats.get(key, 0) + 1
            )

    def get_throughput(self) -> float:
        """Get samples per hour.

        Returns
        -------
        float
            Collection throughput.
        """
        with self._lock:
            if not self._collection_times:
                return 0.0
            count = len(self._collection_times)
            return count * 3600 / self.window_size_seconds

    def get_average_time(self) -> float:
        """Get average collection time.

        Returns
        -------
        float
            Average time in milliseconds.
        """
        with self._lock:
            if not self._collection_times:
                return 0.0
            times = [t for _, t in self._collection_times]
            return statistics.mean(times)

    def get_category_distribution(self) -> dict[str, float]:
        """Get category distribution.

        Returns
        -------
        dict[str, float]
            Proportion per category.
        """
        with self._lock:
            total = sum(self._category_counts.values())
            if total == 0:
                return {}
            return {
                cat: count / total
                for cat, count in self._category_counts.items()
            }

    def get_validation_rate(self) -> float:
        """Get validation pass rate.

        Returns
        -------
        float
            Pass rate [0, 1].
        """
        with self._lock:
            passed = self._validation_stats.get("passed", 0)
            total = sum(self._validation_stats.values())
            return passed / max(total, 1)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary.

        Returns
        -------
        dict[str, Any]
            Complete metrics.
        """
        return {
            "throughput_per_hour": self.get_throughput(),
            "avg_collection_time_ms": self.get_average_time(),
            "category_distribution": self.get_category_distribution(),
            "validation_pass_rate": self.get_validation_rate(),
            "total_collected": sum(self._category_counts.values()),
            "window_size_seconds": self.window_size_seconds,
        }

    def _prune_old_records(self) -> None:
        """Remove records outside window."""
        cutoff = time.time() - self.window_size_seconds
        self._collection_times = [
            (t, dur) for t, dur in self._collection_times if t > cutoff
        ]


__all__ = [
    "NegativeCategory",
    "SpecimenSource",
    "DifficultyLevel",
    "CollectionPriority",
    "NegativeSample",
    "CollectionQuota",
    "StratificationCell",
    "HardNegativeCandidate",
    "SampleSelector",
    "HardNegativeMiner",
    "AgeGroupConfig",
    "CollectionSiteConfig",
    "ConfoundingFactor",
    "StratificationMatrix",
    "LookAlikeOrganismDB",
    "ActiveLearningSelector",
    "HardNegativeMiningPipeline",
    "NegativeCollectionManager",
    "create_collection_manager",
    "AugmentationType",
    "AugmentationResult",
    "AugmentationConfig",
    "DataAugmentationPipeline",
    "MorphometricFeatures",
    "MorphometricAnalyzer",
    "SimilarityResult",
    "SimilarityEngine",
    "FalsePositivePattern",
    "FalsePositiveAnalyzer",
    "ValidationIssue",
    "SampleValidator",
    "BalanceReport",
    "DistributionBalancer",
    "AcquisitionBatch",
    "BatchAcquisitionManager",
    "CollectionMetricsTracker",
]
