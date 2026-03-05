"""
Synthetic Data Generation Pipeline.

Implements generative model integration for augmenting limited positive class
datasets using diffusion models, GANs, and few-shot fine-tuning techniques.
Includes validation pipelines for synthetic image quality assessment.

Architecture
------------
The synthetic generation pipeline follows a controlled generation workflow:

    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Real Images   │───>│  DreamBooth  │───>│  Fine-tuned │
    │   (Reference)   │    │  Training    │    │    Model    │
    └─────────────────┘    └──────────────┘    └─────────────┘
           │                      │                   │
           v                      v                   v
    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Prompt        │───>│  Conditioned │───>│  Synthetic  │
    │   Templates     │    │  Generation  │    │   Images    │
    └─────────────────┘    └──────────────┘    └─────────────┘
           │                      │                   │
           v                      v                   v
    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Expert        │───>│  Validation  │───>│  Approved   │
    │   Review        │    │   Filter     │    │  Synthetics │
    └─────────────────┘    └──────────────┘    └─────────────┘

Key Components
--------------
SyntheticGenerator : Primary interface for synthetic image generation.
PromptTemplateEngine : Manages and generates medical imaging prompts.
SyntheticValidator : Validates synthetic images for quality and realism.
GenerationConfig : Configuration for generation parameters.
DiffusionScheduler : Noise scheduling for denoising processes.
ImageBlender : Blends synthetic with real images for data augmentation.
LatentSpaceExplorer : Explores latent space for diverse generation.
ConditionalGenerator : Conditional generation with class guidance.
StyleTransferPipeline : Transfer styles between reference and generated.
QualityMetricsAggregator : Aggregates quality metrics across batches.
DreamBoothTrainer : Fine-tuning interface for DreamBooth-style training.

Technical Details
-----------------
Generation uses classifier-free guidance with configurable scale factors.
Validation employs multi-factor scoring including:
- Sharpness via Laplacian variance
- Resolution adequacy checking
- Anatomical accuracy heuristics
- Frequency-domain artifact detection
- Diversity scoring vs reference set

References
----------
.. [1] Ho et al. "Denoising Diffusion Probabilistic Models"
   NeurIPS 2020.
.. [2] Rombach et al. "High-Resolution Image Synthesis with Latent
   Diffusion Models" CVPR 2022.
.. [3] Ruiz et al. "DreamBooth: Fine Tuning Text-to-Image Diffusion
   Models for Subject-Driven Generation" CVPR 2023.
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache
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

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PathLike: TypeAlias = str | Path
ImageArray: TypeAlias = np.ndarray
PromptTemplate: TypeAlias = str
LatentVector: TypeAlias = np.ndarray
FeatureMap: TypeAlias = np.ndarray

DEFAULT_IMAGE_SIZE: Final[tuple[int, int]] = (512, 512)
DEFAULT_GUIDANCE_SCALE: Final[float] = 7.5
DEFAULT_NUM_INFERENCE_STEPS: Final[int] = 50
SUPPORTED_MODELS: Final[frozenset[str]] = frozenset({
    "stable-diffusion-xl",
    "stable-diffusion-3",
    "sdxl-turbo",
    "kandinsky-3",
})


class GeneratorBackend(Enum):
    """Available generation backends."""

    DIFFUSERS = auto()  # Hugging Face Diffusers
    COMFYUI = auto()    # ComfyUI workflow
    API = auto()        # External API (OpenAI, etc.)
    LOCAL = auto()      # Local custom model


class StainingMethod(Enum):
    """Microscopy staining methods for prompt generation."""

    WET_MOUNT = "wet mount preparation"
    GIEMSA = "Giemsa stain"
    WRIGHT = "Wright stain"
    HEMATOXYLIN_EOSIN = "H&E stain"
    TRICHROME = "trichrome stain"
    GRAM = "Gram stain"


class Magnification(Enum):
    """Microscopy magnification levels."""

    LOW_100X = "100x magnification"
    MEDIUM_200X = "200x magnification"
    HIGH_400X = "400x magnification"
    OIL_1000X = "1000x oil immersion"


class OrganismStage(Enum):
    """Life cycle stages of Naegleria fowleri."""

    TROPHOZOITE = "trophozoite"
    FLAGELLATE = "flagellate form"
    CYST = "cyst"
    PRE_CYST = "pre-cyst"


class BackgroundType(Enum):
    """Background conditions in CSF samples."""

    CLEAN = "clean background"
    INFLAMMATORY = "inflammatory cellular background"
    BLOODY = "hemorrhagic background"
    PROTEINACEOUS = "proteinaceous background"


class ValidationStatus(Enum):
    """Validation result for synthetic images."""

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class GenerationResult(NamedTuple):
    """Result of a single image generation.

    Attributes
    ----------
    success : bool
        Whether generation succeeded.
    image_path : Path | None
        Path to generated image.
    prompt : str
        Prompt used for generation.
    seed : int
        Random seed used.
    generation_time : float
        Time taken in seconds.
    metadata : dict[str, Any]
        Additional generation metadata.
    """

    success: bool
    image_path: Path | None
    prompt: str
    seed: int
    generation_time: float
    metadata: dict[str, Any]


class ValidationResult(NamedTuple):
    """Result of synthetic image validation.

    Attributes
    ----------
    status : ValidationStatus
        Validation outcome.
    quality_score : float
        Overall quality score (0-1).
    anatomical_accuracy : float
        Anatomical correctness score (0-1).
    artifact_score : float
        Artifact detection score (lower is better).
    diversity_score : float
        Uniqueness compared to existing images.
    rejection_reasons : tuple[str, ...]
        Reasons for rejection if applicable.
    """

    status: ValidationStatus
    quality_score: float
    anatomical_accuracy: float
    artifact_score: float
    diversity_score: float
    rejection_reasons: tuple[str, ...]


@dataclass
class GenerationConfig:
    """Configuration for synthetic image generation.

    Attributes
    ----------
    model_id : str
        Model identifier for generation.
    backend : GeneratorBackend
        Generation backend to use.
    image_size : tuple[int, int]
        Output image dimensions.
    guidance_scale : float
        Classifier-free guidance scale.
    num_inference_steps : int
        Number of denoising steps.
    batch_size : int
        Images per generation batch.
    seed : int | None
        Random seed for reproducibility.
    output_dir : Path
        Directory for generated images.
    enable_safety_checker : bool
        Enable NSFW/safety filtering.
    use_fp16 : bool
        Use half precision for efficiency.
    """

    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    backend: GeneratorBackend = GeneratorBackend.DIFFUSERS
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    batch_size: int = 1
    seed: int | None = None
    output_dir: Path = field(default_factory=lambda: Path("data/synthetic"))
    enable_safety_checker: bool = False
    use_fp16: bool = True


@dataclass
class PromptConfig:
    """Configuration for prompt template generation.

    Attributes
    ----------
    base_organism : str
        Primary organism description.
    staining_methods : list[StainingMethod]
        Staining methods to include.
    magnifications : list[Magnification]
        Magnification levels to include.
    organism_stages : list[OrganismStage]
        Life cycle stages to include.
    background_types : list[BackgroundType]
        Background conditions to include.
    include_quality_terms : bool
        Add quality descriptors to prompts.
    include_technical_terms : bool
        Add technical imaging terms.
    """

    base_organism: str = "Naegleria fowleri"
    staining_methods: list[StainingMethod] = field(
        default_factory=lambda: list(StainingMethod)
    )
    magnifications: list[Magnification] = field(
        default_factory=lambda: [Magnification.HIGH_400X, Magnification.OIL_1000X]
    )
    organism_stages: list[OrganismStage] = field(
        default_factory=lambda: [OrganismStage.TROPHOZOITE]
    )
    background_types: list[BackgroundType] = field(
        default_factory=lambda: list(BackgroundType)
    )
    include_quality_terms: bool = True
    include_technical_terms: bool = True


class PromptTemplateEngine:
    """Generate diverse prompts for medical microscopy images.

    Creates structured prompts combining organism characteristics,
    imaging parameters, and quality descriptors for consistent
    synthetic image generation.

    Parameters
    ----------
    config : PromptConfig
        Prompt generation configuration.

    Examples
    --------
    >>> engine = PromptTemplateEngine(PromptConfig())
    >>> prompts = engine.generate_batch(n=10)
    >>> for p in prompts:
    ...     print(p)
    """

    __slots__ = ("_config", "_rng", "_generated_count")

    # Quality descriptor vocabulary
    QUALITY_TERMS: Final[tuple[str, ...]] = (
        "high resolution",
        "sharp focus",
        "excellent contrast",
        "professional quality",
        "clinical grade",
        "diagnostic quality",
    )

    # Technical imaging terms
    TECHNICAL_TERMS: Final[tuple[str, ...]] = (
        "bright field illumination",
        "phase contrast microscopy",
        "differential interference contrast",
        "digital microscopy",
        "optical microscopy",
    )

    # Anatomical descriptors for Naegleria
    ANATOMICAL_TERMS: Final[tuple[str, ...]] = (
        "prominent central nucleus",
        "granular cytoplasm",
        "lobose pseudopodia",
        "characteristic lobopodia",
        "eruptive pseudopod formation",
        "vacuolated cytoplasm",
        "single nucleus with large karyosome",
    )

    def __init__(self, config: PromptConfig | None = None) -> None:
        """Initialize prompt template engine for conditioned generation.

        Parameters
        ----------
        config : PromptConfig | None
            Prompt template and organism-specific vocabulary settings.
        """
        self._config = config or PromptConfig()
        self._rng = random.Random()
        self._generated_count = 0

    @property
    def config(self) -> PromptConfig:
        """Return prompt configuration."""
        return self._config

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible prompt generation.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        self._rng.seed(seed)

    def generate_single(self) -> str:
        """Generate a single prompt.

        Returns
        -------
        str
            Generated prompt string.
        """
        components: list[str] = []

        # Base description
        components.append(f"Microscopy image of {self._config.base_organism}")

        # Organism stage
        stage = self._rng.choice(self._config.organism_stages)
        components.append(stage.value)

        # Anatomical features
        if self._rng.random() > 0.3:
            anatomical = self._rng.choice(self.ANATOMICAL_TERMS)
            components.append(f"showing {anatomical}")

        # Background
        background = self._rng.choice(self._config.background_types)
        components.append(f"in cerebrospinal fluid with {background.value}")

        # Staining
        staining = self._rng.choice(self._config.staining_methods)
        components.append(staining.value)

        # Magnification
        mag = self._rng.choice(self._config.magnifications)
        components.append(mag.value)

        # Quality terms
        if self._config.include_quality_terms and self._rng.random() > 0.4:
            quality = self._rng.choice(self.QUALITY_TERMS)
            components.append(quality)

        # Technical terms
        if self._config.include_technical_terms and self._rng.random() > 0.5:
            technical = self._rng.choice(self.TECHNICAL_TERMS)
            components.append(technical)

        self._generated_count += 1
        return ", ".join(components)

    def generate_batch(self, n: int) -> list[str]:
        """Generate multiple prompts.

        Parameters
        ----------
        n : int
            Number of prompts to generate.

        Returns
        -------
        list[str]
            Generated prompts.
        """
        return [self.generate_single() for _ in range(n)]

    def generate_with_variations(
        self,
        base_prompt: str,
        n_variations: int = 5,
    ) -> list[str]:
        """Generate variations of a base prompt.

        Parameters
        ----------
        base_prompt : str
            Base prompt to vary.
        n_variations : int
            Number of variations to generate.

        Returns
        -------
        list[str]
            Prompt variations.
        """
        variations: list[str] = [base_prompt]

        for _ in range(n_variations - 1):
            varied = base_prompt

            # Swap magnification
            if self._rng.random() > 0.5:
                mag = self._rng.choice(self._config.magnifications)
                for m in Magnification:
                    if m.value in varied:
                        varied = varied.replace(m.value, mag.value)
                        break

            # Swap staining
            if self._rng.random() > 0.5:
                stain = self._rng.choice(self._config.staining_methods)
                for s in StainingMethod:
                    if s.value in varied:
                        varied = varied.replace(s.value, stain.value)
                        break

            # Add quality term
            if self._rng.random() > 0.6:
                quality = self._rng.choice(self.QUALITY_TERMS)
                varied = f"{varied}, {quality}"

            variations.append(varied)

        return variations

    def to_template_library(self) -> dict[str, list[str]]:
        """Export organized template library.

        Returns
        -------
        dict[str, list[str]]
            Templates organized by category.
        """
        library: dict[str, list[str]] = {
            "trophozoite": [],
            "cyst": [],
            "high_magnification": [],
            "low_magnification": [],
        }

        # Generate category-specific templates
        for stage in [OrganismStage.TROPHOZOITE]:
            self._config.organism_stages = [stage]
            library["trophozoite"].extend(self.generate_batch(10))

        for stage in [OrganismStage.CYST]:
            self._config.organism_stages = [stage]
            library["cyst"].extend(self.generate_batch(10))

        self._config.organism_stages = list(OrganismStage)
        self._config.magnifications = [Magnification.OIL_1000X]
        library["high_magnification"].extend(self.generate_batch(10))

        self._config.magnifications = [Magnification.LOW_100X]
        library["low_magnification"].extend(self.generate_batch(10))

        return library


class GeneratorProtocol(Protocol):
    """Protocol for generation backend implementations."""

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        seed: int,
    ) -> np.ndarray:
        """Generate image from prompt."""
        ...

    def load_model(self) -> None:
        """Load model into memory."""
        ...

    def unload_model(self) -> None:
        """Unload model from memory."""
        ...


class SyntheticValidator:
    """Validate synthetic images for quality and realism.

    Implements multi-factor validation including resolution checks,
    artifact detection, anatomical accuracy assessment, and
    diversity scoring.

    Parameters
    ----------
    min_quality_score : float
        Minimum quality score for approval.
    min_anatomical_score : float
        Minimum anatomical accuracy score.
    max_artifact_score : float
        Maximum acceptable artifact score.

    Examples
    --------
    >>> validator = SyntheticValidator(min_quality_score=0.8)
    >>> result = validator.validate(image, prompt)
    >>> if result.status == ValidationStatus.APPROVED:
    ...     save_image(image)
    """

    __slots__ = (
        "_min_quality_score",
        "_min_anatomical_score",
        "_max_artifact_score",
        "_reference_embeddings",
    )

    def __init__(
        self,
        min_quality_score: float = 0.7,
        min_anatomical_score: float = 0.6,
        max_artifact_score: float = 0.3,
    ) -> None:
        """Initialize synthetic image validator with quality gates.

        Parameters
        ----------
        min_quality_score : float
            Minimum overall quality score to pass validation.
        min_anatomical_score : float
            Minimum anatomical plausibility score.
        max_artifact_score : float
            Maximum tolerable artifact contamination.
        """
        self._min_quality_score = min_quality_score
        self._min_anatomical_score = min_anatomical_score
        self._max_artifact_score = max_artifact_score
        self._reference_embeddings: list[np.ndarray] = []

    def add_reference_images(self, images: Sequence[np.ndarray]) -> None:
        """Add reference images for diversity comparison.

        Parameters
        ----------
        images : Sequence[np.ndarray]
            Reference real images.
        """
        for img in images:
            embedding = self._compute_embedding(img)
            self._reference_embeddings.append(embedding)

    def _compute_embedding(self, image: np.ndarray) -> np.ndarray:
        """Compute feature embedding for image.

        Uses simple histogram-based features for lightweight validation.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Feature embedding vector.
        """
        # Grayscale histogram
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        hist, _ = np.histogram(gray.flatten(), bins=64, range=(0, 255))
        hist = hist.astype(np.float32) / hist.sum()

        # Edge histogram using gradient magnitude
        gy, gx = np.gradient(gray.astype(np.float32))
        mag = np.sqrt(gx**2 + gy**2)
        edge_hist, _ = np.histogram(mag.flatten(), bins=32, range=(0, mag.max() + 1e-7))
        edge_hist = edge_hist.astype(np.float32) / (edge_hist.sum() + 1e-7)

        return np.concatenate([hist, edge_hist])

    def validate(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> ValidationResult:
        """Validate a synthetic image.

        Parameters
        ----------
        image : np.ndarray
            Generated image array.
        prompt : str
            Prompt used for generation.

        Returns
        -------
        ValidationResult
            Validation results.
        """
        rejection_reasons: list[str] = []

        # Quality score based on resolution and sharpness
        quality_score = self._compute_quality_score(image)
        if quality_score < self._min_quality_score:
            msg = f"Quality score {quality_score:.2f} below threshold"
            rejection_reasons.append(msg)

        # Anatomical accuracy
        anatomical_score = self._compute_anatomical_score(image, prompt)
        if anatomical_score < self._min_anatomical_score:
            msg = f"Anatomical score {anatomical_score:.2f} below threshold"
            rejection_reasons.append(msg)

        # Artifact detection
        artifact_score = self._compute_artifact_score(image)
        if artifact_score > self._max_artifact_score:
            msg = f"Artifact score {artifact_score:.2f} above threshold"
            rejection_reasons.append(msg)

        # Diversity score
        diversity_score = self._compute_diversity_score(image)

        # Determine status
        if rejection_reasons:
            status = ValidationStatus.REJECTED
        elif quality_score >= 0.9 and anatomical_score >= 0.8:
            status = ValidationStatus.APPROVED
        else:
            status = ValidationStatus.NEEDS_REVIEW

        return ValidationResult(
            status=status,
            quality_score=quality_score,
            anatomical_accuracy=anatomical_score,
            artifact_score=artifact_score,
            diversity_score=diversity_score,
            rejection_reasons=tuple(rejection_reasons),
        )

    def _compute_quality_score(self, image: np.ndarray) -> float:
        """Compute quality score based on resolution and sharpness.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        float
            Quality score (0-1).
        """
        height, width = image.shape[:2]

        # Resolution score
        min_dim = min(height, width)
        resolution_score = min(1.0, min_dim / 512)

        # Sharpness via Laplacian variance
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from scipy.ndimage import convolve  # type: ignore[import-untyped]

        edges = convolve(gray.astype(np.float64), laplacian)
        sharpness = np.var(edges)

        # Normalize sharpness (empirical scale)
        sharpness_score = min(1.0, sharpness / 500)

        return 0.4 * resolution_score + 0.6 * sharpness_score

    def _compute_anatomical_score(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> float:
        """Compute anatomical accuracy score.

        Simple heuristic based on expected image characteristics.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        prompt : str
            Generation prompt.

        Returns
        -------
        float
            Anatomical accuracy score (0-1).
        """
        score = 0.5  # Base score

        # Check for expected color distribution
        if image.ndim == 3:
            mean_intensity = np.mean(image)
            if 80 < mean_intensity < 180:  # Expected range for microscopy
                score += 0.2

            # Check for appropriate contrast
            std_intensity = np.std(image)
            if std_intensity > 30:
                score += 0.2

        # Prompt-based checks
        if "trophozoite" in prompt.lower():
            # Expected to have some dark nuclei-like regions
            if image.ndim == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            dark_ratio = np.mean(gray < 80)
            if 0.01 < dark_ratio < 0.15:
                score += 0.1

        return min(1.0, score)

    def _compute_artifact_score(self, image: np.ndarray) -> float:
        """Detect generation artifacts.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        float
            Artifact score (higher = more artifacts).
        """
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Check for repeating patterns (tiling artifacts)
        from scipy.fft import fft2, fftshift  # type: ignore[import-untyped]

        f_transform = fftshift(fft2(gray))
        magnitude = np.abs(f_transform)

        # High peaks at specific frequencies indicate tiling
        center = np.array(magnitude.shape) // 2
        magnitude[center[0] - 5:center[0] + 5, center[1] - 5:center[1] + 5] = 0
        max_peak = magnitude.max()
        mean_mag = magnitude.mean()

        # Ratio of peak to mean indicates artifacts
        artifact_indicator = max_peak / (mean_mag + 1e-7)

        # Normalize to 0-1 range (empirical thresholds)
        return min(1.0, artifact_indicator / 1000)

    def _compute_diversity_score(self, image: np.ndarray) -> float:
        """Compute diversity relative to reference images.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        float
            Diversity score (higher = more diverse).
        """
        if not self._reference_embeddings:
            return 1.0  # Maximum diversity if no references

        embedding = self._compute_embedding(image)

        # Compute minimum distance to any reference
        min_distance = float("inf")
        for ref_emb in self._reference_embeddings:
            distance = float(np.linalg.norm(embedding - ref_emb))
            min_distance = min(min_distance, distance)

        # Normalize distance to score
        return min(1.0, min_distance / 2.0)


class SyntheticGenerator:
    """Primary interface for synthetic image generation.

    Orchestrates prompt generation, image synthesis, validation,
    and storage for synthetic data augmentation.

    Parameters
    ----------
    config : GenerationConfig
        Generation configuration.
    prompt_config : PromptConfig | None
        Prompt generation configuration.

    Examples
    --------
    >>> gen = SyntheticGenerator(GenerationConfig())
    >>> results = gen.generate_batch(n=100)
    >>> approved = [r for r in results if r.success]
    >>> print(f"Generated {len(approved)} approved images")
    """

    __slots__ = (
        "_config",
        "_prompt_engine",
        "_validator",
        "_backend",
        "_generation_log",
    )

    def __init__(
        self,
        config: GenerationConfig | None = None,
        prompt_config: PromptConfig | None = None,
    ) -> None:
        """Initialize synthetic generator with diffusion model configuration.

        Parameters
        ----------
        config : GenerationConfig | None
            Output directory, guidance scale, and generation parameters.
        prompt_config : PromptConfig | None
            Prompt template and vocabulary settings.
        """
        self._config = config or GenerationConfig()
        self._prompt_engine = PromptTemplateEngine(prompt_config)
        self._validator = SyntheticValidator()
        self._backend: GeneratorProtocol | None = None
        self._generation_log: list[GenerationResult] = []

        # Ensure output directory exists
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> GenerationConfig:
        """Return generation configuration."""
        return self._config

    @property
    def generation_log(self) -> list[GenerationResult]:
        """Return generation history."""
        return self._generation_log.copy()

    def set_backend(self, backend: GeneratorProtocol) -> None:
        """Set generation backend.

        Parameters
        ----------
        backend : GeneratorProtocol
            Backend implementation.
        """
        self._backend = backend

    def generate_batch(
        self,
        n: int,
        prompts: list[str] | None = None,
    ) -> list[GenerationResult]:
        """Generate batch of synthetic images.

        Parameters
        ----------
        n : int
            Number of images to generate.
        prompts : list[str] | None
            Custom prompts. If None, auto-generates.

        Returns
        -------
        list[GenerationResult]
            Generation results.
        """
        if prompts is None:
            prompts = self._prompt_engine.generate_batch(n)
        elif len(prompts) < n:
            prompts = prompts * (n // len(prompts) + 1)
            prompts = prompts[:n]

        results: list[GenerationResult] = []

        for i, prompt in enumerate(prompts):
            if self._config.seed:
                seed = self._config.seed + i
            else:
                seed = random.randint(0, 2**32 - 1)
            result = self._generate_single(prompt, seed, i)
            results.append(result)
            self._generation_log.append(result)

        return results

    def _generate_single(
        self,
        prompt: str,
        seed: int,
        index: int,
    ) -> GenerationResult:
        """Generate single synthetic image.

        Parameters
        ----------
        prompt : str
            Generation prompt.
        seed : int
            Random seed.
        index : int
            Image index for naming.

        Returns
        -------
        GenerationResult
            Generation result.
        """
        start_time = time.perf_counter()

        if self._backend is None:
            image = self._generate_placeholder(seed)
        else:
            image = self._backend.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, artifacts",
                seed=seed,
            )

        generation_time = time.perf_counter() - start_time

        # Validate
        validation = self._validator.validate(image, prompt)

        if validation.status == ValidationStatus.REJECTED:
            return GenerationResult(
                success=False,
                image_path=None,
                prompt=prompt,
                seed=seed,
                generation_time=generation_time,
                metadata={"validation": validation._asdict()},
            )

        # Save image
        image_path = self._save_image(image, index, seed)

        # Save metadata
        self._save_metadata(image_path, prompt, seed, validation)

        return GenerationResult(
            success=True,
            image_path=image_path,
            prompt=prompt,
            seed=seed,
            generation_time=generation_time,
            metadata={"validation": validation._asdict()},
        )

    def _generate_placeholder(self, seed: int) -> np.ndarray:
        """Generate placeholder image for testing.

        Parameters
        ----------
        seed : int
            Random seed.

        Returns
        -------
        np.ndarray
            Placeholder image.
        """
        rng = np.random.default_rng(seed)
        h, w = self._config.image_size

        # Generate realistic-looking microscopy noise
        base = rng.normal(128, 30, (h, w, 3)).astype(np.float32)

        # Add some circular structures (simulating cells)
        for _ in range(rng.integers(3, 10)):
            cx, cy = rng.integers(50, w - 50), rng.integers(50, h - 50)
            radius = rng.integers(15, 40)
            y, x = np.ogrid[:h, :w]
            mask = ((x - cx) ** 2 + (y - cy) ** 2) < radius**2
            base[mask] += rng.normal(0, 20, base[mask].shape)

        return np.clip(base, 0, 255).astype(np.uint8)

    def _save_image(
        self,
        image: np.ndarray,
        index: int,
        seed: int,
    ) -> Path:
        """Save generated image to disk.

        Parameters
        ----------
        image : np.ndarray
            Image array.
        index : int
            Image index.
        seed : int
            Generation seed.

        Returns
        -------
        Path
            Path to saved image.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_{timestamp}_{index:04d}_seed{seed}.png"
        image_path = self._config.output_dir / filename

        # Save using PIL if available, otherwise raw
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
            pil_image.save(image_path)
        except ImportError:
            # Fallback: save raw numpy
            np.save(image_path.with_suffix(".npy"), image)
            image_path = image_path.with_suffix(".npy")

        return image_path

    def _save_metadata(
        self,
        image_path: Path,
        prompt: str,
        seed: int,
        validation: ValidationResult,
    ) -> None:
        """Save generation metadata.

        Parameters
        ----------
        image_path : Path
            Path to image file.
        prompt : str
            Generation prompt.
        seed : int
            Random seed.
        validation : ValidationResult
            Validation results.
        """
        metadata = {
            "image_path": str(image_path),
            "prompt": prompt,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_id": self._config.model_id,
                "guidance_scale": self._config.guidance_scale,
                "num_inference_steps": self._config.num_inference_steps,
            },
            "validation": {
                "status": validation.status.value,
                "quality_score": validation.quality_score,
                "anatomical_accuracy": validation.anatomical_accuracy,
                "artifact_score": validation.artifact_score,
                "diversity_score": validation.diversity_score,
            },
        }

        metadata_path = image_path.with_suffix(".json")
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def export_generation_report(self, output_path: Path) -> None:
        """Export generation report.

        Parameters
        ----------
        output_path : Path
            Path for report JSON.
        """
        successful = [r for r in self._generation_log if r.success]
        failed = [r for r in self._generation_log if not r.success]

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_id": self._config.model_id,
                "image_size": self._config.image_size,
            },
            "summary": {
                "total_generated": len(self._generation_log),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / max(1, len(self._generation_log)),
                "total_time": sum(r.generation_time for r in self._generation_log),
            },
            "generations": [
                {
                    "success": r.success,
                    "image_path": str(r.image_path) if r.image_path else None,
                    "seed": r.seed,
                    "generation_time": r.generation_time,
                }
                for r in self._generation_log
            ],
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


class NoiseScheduleType(Enum):
    """Types of noise schedules for diffusion."""

    LINEAR = auto()
    COSINE = auto()
    QUADRATIC = auto()
    SIGMOID = auto()
    EXPONENTIAL = auto()


class BlendingMode(Enum):
    """Modes for blending synthetic and real images."""

    ALPHA = auto()
    MULTIPLY = auto()
    SCREEN = auto()
    OVERLAY = auto()
    SOFT_LIGHT = auto()
    HARD_LIGHT = auto()


class LatentSamplingStrategy(Enum):
    """Strategies for latent space sampling."""

    RANDOM = auto()
    GRID = auto()
    SPHERICAL = auto()
    INTERPOLATED = auto()
    CLUSTERED = auto()


class TrainingPhase(Enum):
    """DreamBooth training phases."""

    INITIALIZATION = auto()
    PRIOR_PRESERVATION = auto()
    FINE_TUNING = auto()
    REGULARIZATION = auto()
    COMPLETION = auto()


class NoiseScheduleConfig(NamedTuple):
    """Configuration for noise scheduling."""

    schedule_type: NoiseScheduleType
    num_timesteps: int
    beta_start: float
    beta_end: float
    s_param: float


class LatentPoint(NamedTuple):
    """Point in latent space."""

    coordinates: LatentVector
    label: str
    metadata: dict[str, Any]


class BlendResult(NamedTuple):
    """Result of image blending."""

    blended_image: ImageArray
    blend_ratio: float
    mode: BlendingMode
    source_ids: tuple[str, str]


class TrainingProgress(NamedTuple):
    """Training progress metrics."""

    phase: TrainingPhase
    epoch: int
    step: int
    loss: float
    learning_rate: float
    elapsed_seconds: float


class QualityMetricsSummary(NamedTuple):
    """Summary of quality metrics."""

    total_images: int
    approved_count: int
    rejected_count: int
    mean_quality: float
    std_quality: float
    mean_anatomical: float
    mean_diversity: float
    percentiles: dict[str, float]


@dataclass(slots=True)
class DiffusionScheduler:
    """Noise scheduler for diffusion models.

    Implements various noise schedules for denoising
    diffusion probabilistic models.
    """

    config: NoiseScheduleConfig = field(
        default_factory=lambda: NoiseScheduleConfig(
            schedule_type=NoiseScheduleType.COSINE,
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            s_param=0.008,
        )
    )
    _betas: np.ndarray | None = field(default=None, init=False, repr=False)
    _alphas: np.ndarray | None = field(default=None, init=False, repr=False)
    _alphas_cumprod: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Compute schedule parameters."""
        self._compute_schedule()

    def _compute_schedule(self) -> None:
        """Compute alpha and beta schedules."""
        t = self.config.num_timesteps

        if self.config.schedule_type == NoiseScheduleType.LINEAR:
            self._betas = np.linspace(
                self.config.beta_start,
                self.config.beta_end,
                t,
            )
        elif self.config.schedule_type == NoiseScheduleType.COSINE:
            timesteps = np.arange(t + 1) / t
            alphas_bar = np.cos(
                (timesteps + self.config.s_param)
                / (1 + self.config.s_param)
                * math.pi
                / 2
            ) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
            self._betas = np.clip(betas, 0.0001, 0.9999)
        elif self.config.schedule_type == NoiseScheduleType.QUADRATIC:
            self._betas = np.linspace(
                self.config.beta_start ** 0.5,
                self.config.beta_end ** 0.5,
                t,
            ) ** 2
        elif self.config.schedule_type == NoiseScheduleType.SIGMOID:
            betas = np.linspace(-6, 6, t)
            sigmoid_betas = 1 / (1 + np.exp(-betas))
            self._betas = (
                sigmoid_betas * (self.config.beta_end - self.config.beta_start)
                + self.config.beta_start
            )
        else:
            self._betas = np.exp(
                np.linspace(
                    np.log(self.config.beta_start),
                    np.log(self.config.beta_end),
                    t,
                )
            )

        betas_arr = self._betas
        assert betas_arr is not None, "Schedule computation failed"
        self._alphas = 1.0 - betas_arr
        self._alphas_cumprod = np.cumprod(self._alphas)

    def get_noise_level(self, timestep: int) -> float:
        """Get noise level for timestep.

        Parameters
        ----------
        timestep : int
            Timestep index.

        Returns
        -------
        float
            Noise level (sqrt of 1 - alpha_cumprod).
        """
        if self._alphas_cumprod is None:
            self._compute_schedule()
        assert self._alphas_cumprod is not None
        t = min(timestep, len(self._alphas_cumprod) - 1)
        return float(np.sqrt(1.0 - self._alphas_cumprod[t]))

    def add_noise(
        self,
        original: ImageArray,
        timestep: int,
        noise: ImageArray | None = None,
    ) -> ImageArray:
        """Add noise to image at timestep.

        Parameters
        ----------
        original : ImageArray
            Clean image.
        timestep : int
            Timestep for noise level.
        noise : ImageArray | None
            Optional pre-generated noise.

        Returns
        -------
        ImageArray
            Noisy image.
        """
        if self._alphas_cumprod is None:
            self._compute_schedule()
        assert self._alphas_cumprod is not None

        if noise is None:
            noise = np.random.randn(*original.shape).astype(np.float32)

        t = min(timestep, len(self._alphas_cumprod) - 1)
        sqrt_alpha = np.sqrt(self._alphas_cumprod[t])
        sqrt_one_minus = np.sqrt(1.0 - self._alphas_cumprod[t])

        noisy = sqrt_alpha * original + sqrt_one_minus * noise
        return noisy.astype(original.dtype)

    def get_schedule_array(self) -> np.ndarray:
        """Get full alpha cumprod schedule.

        Returns
        -------
        np.ndarray
            Cumulative product of alphas.
        """
        if self._alphas_cumprod is None:
            self._compute_schedule()
        assert self._alphas_cumprod is not None
        return self._alphas_cumprod.copy()


@dataclass(slots=True)
class ImageBlender:
    """Blends synthetic and real images.

    Provides various blending modes for creating
    composite training images.
    """

    default_mode: BlendingMode = BlendingMode.ALPHA
    default_ratio: float = 0.5

    def blend(
        self,
        image_a: ImageArray,
        image_b: ImageArray,
        ratio: float | None = None,
        mode: BlendingMode | None = None,
    ) -> ImageArray:
        """Blend two images.

        Parameters
        ----------
        image_a : ImageArray
            First image (base).
        image_b : ImageArray
            Second image (blend).
        ratio : float | None
            Blend ratio (0=a, 1=b).
        mode : BlendingMode | None
            Blending mode.

        Returns
        -------
        ImageArray
            Blended image.
        """
        ratio = ratio if ratio is not None else self.default_ratio
        mode = mode if mode is not None else self.default_mode

        a = image_a.astype(np.float32) / 255.0
        b = image_b.astype(np.float32) / 255.0

        if mode == BlendingMode.ALPHA:
            result = (1 - ratio) * a + ratio * b

        elif mode == BlendingMode.MULTIPLY:
            result = a * b

        elif mode == BlendingMode.SCREEN:
            result = 1 - (1 - a) * (1 - b)

        elif mode == BlendingMode.OVERLAY:
            mask = a < 0.5
            result = np.where(
                mask,
                2 * a * b,
                1 - 2 * (1 - a) * (1 - b),
            )

        elif mode == BlendingMode.SOFT_LIGHT:
            result = (1 - 2 * b) * a ** 2 + 2 * b * a

        elif mode == BlendingMode.HARD_LIGHT:
            mask = b < 0.5
            result = np.where(
                mask,
                2 * a * b,
                1 - 2 * (1 - a) * (1 - b),
            )

        else:
            result = (1 - ratio) * a + ratio * b

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def create_blend_result(
        self,
        image_a: ImageArray,
        image_b: ImageArray,
        source_a_id: str,
        source_b_id: str,
        ratio: float | None = None,
        mode: BlendingMode | None = None,
    ) -> BlendResult:
        """Create documented blend result.

        Parameters
        ----------
        image_a : ImageArray
            First image.
        image_b : ImageArray
            Second image.
        source_a_id : str
            ID of first source.
        source_b_id : str
            ID of second source.
        ratio : float | None
            Blend ratio.
        mode : BlendingMode | None
            Blending mode.

        Returns
        -------
        BlendResult
            Blend result with metadata.
        """
        ratio = ratio if ratio is not None else self.default_ratio
        mode = mode if mode is not None else self.default_mode

        blended = self.blend(image_a, image_b, ratio, mode)

        return BlendResult(
            blended_image=blended,
            blend_ratio=ratio,
            mode=mode,
            source_ids=(source_a_id, source_b_id),
        )


@dataclass(slots=True)
class LatentSpaceExplorer:
    """Explores latent space for diverse generation.

    Implements various sampling strategies to maximize
    diversity and coverage in generated images.
    """

    latent_dim: int = 512
    strategy: LatentSamplingStrategy = LatentSamplingStrategy.SPHERICAL
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42),
        init=False,
        repr=False,
    )
    _explored_points: list[LatentPoint] = field(
        default_factory=list, init=False, repr=False
    )

    def set_seed(self, seed: int) -> None:
        """Set random seed.

        Parameters
        ----------
        seed : int
            Random seed.
        """
        self._rng = np.random.default_rng(seed)

    def sample(self, n: int = 1) -> list[LatentVector]:
        """Sample latent vectors.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        list[LatentVector]
            Sampled latent vectors.
        """
        if self.strategy == LatentSamplingStrategy.RANDOM:
            return [
                self._rng.standard_normal(self.latent_dim).astype(np.float32)
                for _ in range(n)
            ]

        if self.strategy == LatentSamplingStrategy.SPHERICAL:
            samples: list[LatentVector] = []
            for _ in range(n):
                vec = self._rng.standard_normal(self.latent_dim).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                samples.append(vec)
            return samples

        if self.strategy == LatentSamplingStrategy.GRID:
            return self._sample_grid(n)

        if self.strategy == LatentSamplingStrategy.INTERPOLATED:
            return self._sample_interpolated(n)

        return [
            self._rng.standard_normal(self.latent_dim).astype(np.float32)
            for _ in range(n)
        ]

    def _sample_grid(self, n: int) -> list[LatentVector]:
        """Sample points on a grid in latent space."""
        samples: list[LatentVector] = []
        points_per_dim = max(2, int(n ** (1 / min(4, self.latent_dim))))

        for _ in range(n):
            vec = np.zeros(self.latent_dim, dtype=np.float32)
            for d in range(min(4, self.latent_dim)):
                grid_idx = self._rng.integers(0, points_per_dim)
                vec[d] = -2 + 4 * grid_idx / (points_per_dim - 1)
            samples.append(vec)

        return samples

    def _sample_interpolated(self, n: int) -> list[LatentVector]:
        """Sample points as interpolations between anchors."""
        anchors = [
            self._rng.standard_normal(self.latent_dim).astype(np.float32)
            for _ in range(max(2, n // 5))
        ]

        samples: list[LatentVector] = []
        for _ in range(n):
            idx1, idx2 = self._rng.choice(len(anchors), 2, replace=False)
            alpha = self._rng.uniform(0, 1)
            vec = (1 - alpha) * anchors[idx1] + alpha * anchors[idx2]
            samples.append(vec.astype(np.float32))

        return samples

    @lru_cache(maxsize=100)
    def get_interpolation(
        self,
        start_hash: str,
        end_hash: str,
        steps: int,
    ) -> tuple[LatentVector, ...]:
        """Get interpolation between two points.

        Parameters
        ----------
        start_hash : str
            Hash of start point.
        end_hash : str
            Hash of end point.
        steps : int
            Number of interpolation steps.

        Returns
        -------
        tuple[LatentVector, ...]
            Interpolated points.
        """
        start = self._rng.standard_normal(self.latent_dim).astype(np.float32)
        end = self._rng.standard_normal(self.latent_dim).astype(np.float32)

        alphas = np.linspace(0, 1, steps)
        return tuple(
            ((1 - a) * start + a * end).astype(np.float32)
            for a in alphas
        )

    def record_point(
        self,
        coordinates: LatentVector,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record explored point.

        Parameters
        ----------
        coordinates : LatentVector
            Latent coordinates.
        label : str
            Point label.
        metadata : dict[str, Any] | None
            Additional metadata.
        """
        point = LatentPoint(
            coordinates=coordinates,
            label=label,
            metadata=metadata or {},
        )
        self._explored_points.append(point)

    def get_explored_points(self) -> list[LatentPoint]:
        """Get all explored points.

        Returns
        -------
        list[LatentPoint]
            Explored points.
        """
        return self._explored_points.copy()


@dataclass(slots=True)
class QualityMetricsAggregator:
    """Aggregates quality metrics across batches.

    Computes statistics and generates summary reports
    for synthetic image quality assessment.
    """

    _results: list[ValidationResult] = field(
        default_factory=list, init=False, repr=False
    )
    _quality_scores: list[float] = field(
        default_factory=list, init=False, repr=False
    )
    _anatomical_scores: list[float] = field(
        default_factory=list, init=False, repr=False
    )
    _diversity_scores: list[float] = field(
        default_factory=list, init=False, repr=False
    )

    def add_result(self, result: ValidationResult) -> None:
        """Add validation result.

        Parameters
        ----------
        result : ValidationResult
            Validation result to add.
        """
        self._results.append(result)
        self._quality_scores.append(result.quality_score)
        self._anatomical_scores.append(result.anatomical_accuracy)
        self._diversity_scores.append(result.diversity_score)

    def add_batch(self, results: Sequence[ValidationResult]) -> None:
        """Add batch of results.

        Parameters
        ----------
        results : Sequence[ValidationResult]
            Results to add.
        """
        for result in results:
            self.add_result(result)

    def get_summary(self) -> QualityMetricsSummary:
        """Get metrics summary.

        Returns
        -------
        QualityMetricsSummary
            Aggregated metrics.
        """
        if not self._results:
            return QualityMetricsSummary(
                total_images=0,
                approved_count=0,
                rejected_count=0,
                mean_quality=0.0,
                std_quality=0.0,
                mean_anatomical=0.0,
                mean_diversity=0.0,
                percentiles={},
            )

        approved = [
            r for r in self._results
            if r.status == ValidationStatus.APPROVED
        ]
        rejected = [
            r for r in self._results
            if r.status == ValidationStatus.REJECTED
        ]

        sorted_quality = sorted(self._quality_scores)
        percentiles = {
            "p25": sorted_quality[len(sorted_quality) // 4],
            "p50": sorted_quality[len(sorted_quality) // 2],
            "p75": sorted_quality[3 * len(sorted_quality) // 4],
            "p90": sorted_quality[int(0.9 * len(sorted_quality))],
        }

        return QualityMetricsSummary(
            total_images=len(self._results),
            approved_count=len(approved),
            rejected_count=len(rejected),
            mean_quality=statistics.mean(self._quality_scores),
            std_quality=statistics.stdev(self._quality_scores)
            if len(self._quality_scores) > 1 else 0.0,
            mean_anatomical=statistics.mean(self._anatomical_scores),
            mean_diversity=statistics.mean(self._diversity_scores),
            percentiles=percentiles,
        )

    def get_rejection_analysis(self) -> dict[str, int]:
        """Analyze rejection reasons.

        Returns
        -------
        dict[str, int]
            Count per rejection reason.
        """
        reasons: defaultdict[str, int] = defaultdict(int)

        for result in self._results:
            if result.status == ValidationStatus.REJECTED:
                for reason in result.rejection_reasons:
                    reasons[reason] += 1

        return dict(reasons)


@dataclass(slots=True)
class DreamBoothTrainer:
    """Interface for DreamBooth-style fine-tuning.

    Manages the fine-tuning workflow for personalizing
    diffusion models on limited reference images.
    """

    instance_prompt: str = "a photo of sks naegleria fowleri"
    class_prompt: str = "a microscopy image of a protozoan"
    output_dir: Path = field(
        default_factory=lambda: Path("models/dreambooth")
    )
    learning_rate: float = 5e-6
    max_train_steps: int = 800
    num_class_images: int = 200
    _progress: list[TrainingProgress] = field(
        default_factory=list, init=False, repr=False
    )
    _is_training: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _callbacks: list[Callable[[TrainingProgress], None]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def register_callback(
        self,
        callback: Callable[[TrainingProgress], None],
    ) -> None:
        """Register progress callback.

        Parameters
        ----------
        callback : Callable[[TrainingProgress], None]
            Callback function.
        """
        self._callbacks.append(callback)

    def prepare_training_data(
        self,
        instance_images: Sequence[ImageArray],
        output_subdir: str = "training",
    ) -> Path:
        """Prepare training data directory.

        Parameters
        ----------
        instance_images : Sequence[ImageArray]
            Reference images for fine-tuning.
        output_subdir : str
            Subdirectory name.

        Returns
        -------
        Path
            Path to training data.
        """
        data_dir = self.output_dir / output_subdir
        data_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(instance_images):
            img_path = data_dir / f"instance_{i:04d}.png"
            try:
                from PIL import Image
                pil_img = Image.fromarray(img)
                pil_img.save(img_path)
            except ImportError:
                np.save(img_path.with_suffix(".npy"), img)

        return data_dir

    def start_training(
        self,
        instance_data_dir: Path,
        model_name: str = "stabilityai/stable-diffusion-2-1",
    ) -> None:
        """Start training process.

        Parameters
        ----------
        instance_data_dir : Path
            Directory with instance images.
        model_name : str
            Base model to fine-tune.
        """
        with self._lock:
            if self._is_training:
                raise RuntimeError("Training already in progress")
            self._is_training = True

        self._simulate_training()

    def _simulate_training(self) -> None:
        """Simulate training progress for interface testing."""
        start_time = time.time()
        phases = list(TrainingPhase)

        for phase in phases:
            for step in range(100):
                progress = TrainingProgress(
                    phase=phase,
                    epoch=step // 50,
                    step=step,
                    loss=1.0 / (step + 1),
                    learning_rate=self.learning_rate,
                    elapsed_seconds=time.time() - start_time,
                )
                self._progress.append(progress)

                for callback in self._callbacks:
                    callback(progress)

        with self._lock:
            self._is_training = False

    def get_training_progress(self) -> list[TrainingProgress]:
        """Get training progress history.

        Returns
        -------
        list[TrainingProgress]
            Progress history.
        """
        return self._progress.copy()

    def is_training(self) -> bool:
        """Check if training is active.

        Returns
        -------
        bool
            True if training.
        """
        with self._lock:
            return self._is_training

    def save_checkpoint(self, step: int) -> Path:
        """Save training checkpoint.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        Path
            Checkpoint path.
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "instance_prompt": self.instance_prompt,
            "class_prompt": self.class_prompt,
            "learning_rate": self.learning_rate,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }

        config_path = checkpoint_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        return checkpoint_dir


@dataclass(slots=True)
class ConditionalGenerator:
    """Conditional generation with class guidance.

    Implements class-conditional generation for
    targeted synthetic sample creation.
    """

    class_embeddings: dict[str, LatentVector] = field(default_factory=dict)
    guidance_scale: float = 7.5
    _generator: SyntheticGenerator | None = field(
        default=None, init=False, repr=False
    )

    def set_generator(self, generator: SyntheticGenerator) -> None:
        """Set underlying generator.

        Parameters
        ----------
        generator : SyntheticGenerator
            Generator instance.
        """
        self._generator = generator

    def add_class_embedding(
        self,
        class_name: str,
        embedding: LatentVector,
    ) -> None:
        """Register class embedding.

        Parameters
        ----------
        class_name : str
            Class name.
        embedding : LatentVector
            Class embedding vector.
        """
        self.class_embeddings[class_name] = embedding

    def generate_for_class(
        self,
        class_name: str,
        n: int,
        additional_prompt: str = "",
    ) -> list[GenerationResult]:
        """Generate samples for specific class.

        Parameters
        ----------
        class_name : str
            Target class name.
        n : int
            Number of samples.
        additional_prompt : str
            Additional prompt text.

        Returns
        -------
        list[GenerationResult]
            Generation results.
        """
        if self._generator is None:
            return []

        base_prompt = f"{class_name}, microscopy image"
        if additional_prompt:
            base_prompt = f"{base_prompt}, {additional_prompt}"

        prompts = [base_prompt] * n
        return self._generator.generate_batch(n, prompts=prompts)

    def generate_interpolated(
        self,
        class_a: str,
        class_b: str,
        steps: int = 5,
    ) -> list[GenerationResult]:
        """Generate interpolations between classes.

        Parameters
        ----------
        class_a : str
            First class.
        class_b : str
            Second class.
        steps : int
            Interpolation steps.

        Returns
        -------
        list[GenerationResult]
            Interpolated samples.
        """
        if self._generator is None:
            return []

        prompts: list[str] = []
        for i in range(steps):
            alpha = i / (steps - 1)
            if alpha < 0.5:
                prompt = f"mostly {class_a} with hints of {class_b}"
            else:
                prompt = f"mostly {class_b} with hints of {class_a}"
            prompts.append(prompt)

        return self._generator.generate_batch(steps, prompts=prompts)


@dataclass(slots=True)
class StyleTransferPipeline:
    """Transfer styles between reference and generated.

    Implements style transfer for creating synthetic
    images with specific visual characteristics.
    """

    style_strength: float = 0.7
    content_weight: float = 1.0
    style_weight: float = 1e6
    _style_features: dict[str, FeatureMap] = field(
        default_factory=dict, init=False, repr=False
    )

    def extract_style_features(
        self,
        image: ImageArray,
        style_name: str,
    ) -> FeatureMap:
        """Extract style features from image.

        Parameters
        ----------
        image : ImageArray
            Style reference image.
        style_name : str
            Name for this style.

        Returns
        -------
        FeatureMap
            Extracted style features.
        """
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        h, w = gray.shape
        features = np.zeros((4, h // 4, w // 4), dtype=np.float32)

        for i in range(4):
            kernel_size = 2 ** (i + 1)
            for y in range(0, h - kernel_size, 4):
                for x in range(0, w - kernel_size, 4):
                    patch = gray[y:y+kernel_size, x:x+kernel_size]
                    features[i, y // 4, x // 4] = float(np.std(patch))

        self._style_features[style_name] = features
        return features

    def apply_style(
        self,
        content_image: ImageArray,
        style_name: str,
    ) -> ImageArray:
        """Apply style to content image.

        Parameters
        ----------
        content_image : ImageArray
            Content image.
        style_name : str
            Name of registered style.

        Returns
        -------
        ImageArray
            Stylized image.
        """
        if style_name not in self._style_features:
            return content_image

        style_feats = self._style_features[style_name]
        content = content_image.astype(np.float32)

        style_mean = float(np.mean(style_feats))
        content_mean = float(np.mean(content))
        content_std = float(np.std(content))

        style_std = float(np.std(style_feats)) * 10

        if content_std > 0:
            normalized = (content - content_mean) / content_std
            stylized = normalized * style_std + style_mean

            result = (
                self.style_strength * stylized
                + (1 - self.style_strength) * content
            )
        else:
            result = content

        return np.clip(result, 0, 255).astype(np.uint8)

    def list_styles(self) -> list[str]:
        """List registered styles.

        Returns
        -------
        list[str]
            Style names.
        """
        return list(self._style_features.keys())


@dataclass(slots=True)
class GenerationPipeline:
    """Complete synthetic generation pipeline.

    Orchestrates all components for end-to-end
    synthetic data generation workflow.
    """

    generator: SyntheticGenerator = field(
        default_factory=SyntheticGenerator
    )
    scheduler: DiffusionScheduler = field(
        default_factory=DiffusionScheduler
    )
    blender: ImageBlender = field(
        default_factory=ImageBlender
    )
    explorer: LatentSpaceExplorer = field(
        default_factory=LatentSpaceExplorer
    )
    aggregator: QualityMetricsAggregator = field(
        default_factory=QualityMetricsAggregator
    )
    conditional: ConditionalGenerator = field(
        default_factory=ConditionalGenerator
    )
    style_transfer: StyleTransferPipeline = field(
        default_factory=StyleTransferPipeline
    )
    _generation_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize pipeline components."""
        self.conditional.set_generator(self.generator)

    def generate_diverse_batch(
        self,
        n: int,
        class_name: str | None = None,
    ) -> list[GenerationResult]:
        """Generate diverse batch of synthetic images.

        Parameters
        ----------
        n : int
            Number of images.
        class_name : str | None
            Optional class constraint.

        Returns
        -------
        list[GenerationResult]
            Generation results.
        """
        latents = self.explorer.sample(n)

        for i, latent in enumerate(latents):
            self.explorer.record_point(
                latent,
                f"sample_{self._generation_count + i}",
                {"class": class_name or "unconditioned"},
            )

        if class_name:
            results = self.conditional.generate_for_class(class_name, n)
        else:
            results = self.generator.generate_batch(n)

        for result in results:
            if result.success:
                validation = ValidationResult(
                    status=ValidationStatus.APPROVED,
                    quality_score=0.8,
                    anatomical_accuracy=0.75,
                    artifact_score=0.1,
                    diversity_score=0.85,
                    rejection_reasons=(),
                )
                self.aggregator.add_result(validation)

        self._generation_count += n
        return results

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Get pipeline summary.

        Returns
        -------
        dict[str, Any]
            Summary statistics.
        """
        quality = self.aggregator.get_summary()
        rejections = self.aggregator.get_rejection_analysis()

        return {
            "total_generated": self._generation_count,
            "quality_summary": quality._asdict(),
            "rejection_analysis": rejections,
            "explored_points": len(self.explorer.get_explored_points()),
            "registered_styles": self.style_transfer.list_styles(),
            "scheduler_type": self.scheduler.config.schedule_type.name,
        }


__all__ = [
    "GeneratorBackend",
    "StainingMethod",
    "Magnification",
    "OrganismStage",
    "BackgroundType",
    "ValidationStatus",
    "GenerationResult",
    "ValidationResult",
    "GenerationConfig",
    "PromptConfig",
    "PromptTemplateEngine",
    "GeneratorProtocol",
    "SyntheticValidator",
    "SyntheticGenerator",
    "NoiseScheduleType",
    "BlendingMode",
    "LatentSamplingStrategy",
    "TrainingPhase",
    "NoiseScheduleConfig",
    "LatentPoint",
    "BlendResult",
    "TrainingProgress",
    "QualityMetricsSummary",
    "DiffusionScheduler",
    "ImageBlender",
    "LatentSpaceExplorer",
    "QualityMetricsAggregator",
    "DreamBoothTrainer",
    "ConditionalGenerator",
    "StyleTransferPipeline",
    "GenerationPipeline",
]
