"""
Microscopy Image Loader and Preprocessing Pipeline.

Provides specialized loaders for medical microscopy images including
CSF wet mount preparations, Giemsa-stained slides, and Wright-stained
slides. Implements format detection, quality validation, and standardized
preprocessing for downstream neural network consumption.

Architecture
------------
The microscopy pipeline follows a multi-stage processing architecture:

    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Source    │───>│   Quality    │───>│   Format    │
    │   Images    │    │   Filter     │    │   Detect    │
    └─────────────┘    └──────────────┘    └─────────────┘
           │                  │                   │
           v                  v                   v
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Resize    │───>│   Normalize  │───>│   Tensor    │
    │   & Pad     │    │   & Augment  │    │   Output    │
    └─────────────┘    └──────────────┘    └─────────────┘

Classes
-------
MicroscopyLoader
    Primary interface for loading and preprocessing microscopy images.
QualityFilter
    Filters images based on resolution, blur, and artifact criteria.
ImageNormalizer
    Standardizes image intensities and color balance.
AugmentationPipeline
    Applies training-time augmentations for regularization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import (
    Final,
    Literal,
    NamedTuple,
    Sequence,
    TypeAlias,
)

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Type aliases
PathLike: TypeAlias = str | Path
ImageArray: TypeAlias = np.ndarray  # Shape: (H, W, C) or (H, W)
NormalizationMethod: TypeAlias = Literal["minmax", "zscore", "percentile"]

# Constants
DEFAULT_IMAGE_SIZE: Final[tuple[int, int]] = (512, 512)
MIN_RESOLUTION: Final[int] = 256
MAX_RESOLUTION: Final[int] = 4096
SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset(
    {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"}
)


class StainType(Enum):
    """Classification of microscopy staining methods."""

    WET_MOUNT = auto()
    GIEMSA = auto()
    WRIGHT = auto()
    HEMATOXYLIN_EOSIN = auto()
    PHASE_CONTRAST = auto()
    BRIGHTFIELD = auto()
    UNKNOWN = auto()


class QualityLevel(Enum):
    """Quality classification for microscopy images."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


class ImageMetadata(NamedTuple):
    """Metadata extracted from microscopy images.

    Attributes
    ----------
    file_path : Path
        Original file path.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    channels : int
        Number of color channels.
    bit_depth : int
        Bits per channel.
    stain_type : StainType
        Detected or labeled staining method.
    magnification : str
        Objective magnification if known.
    quality_level : QualityLevel
        Assessed quality classification.
    blur_score : float
        Laplacian variance (higher = sharper).
    """

    file_path: Path
    width: int
    height: int
    channels: int
    bit_depth: int
    stain_type: StainType
    magnification: str
    quality_level: QualityLevel
    blur_score: float


@dataclass
class QualityThresholds:
    """Configurable thresholds for image quality assessment.

    Attributes
    ----------
    min_width : int
        Minimum acceptable width in pixels.
    min_height : int
        Minimum acceptable height in pixels.
    min_blur_score : float
        Minimum Laplacian variance for sharpness.
    max_saturation_ratio : float
        Maximum ratio of saturated pixels allowed.
    min_contrast : float
        Minimum standard deviation of intensities.
    """

    min_width: int = 256
    min_height: int = 256
    min_blur_score: float = 100.0
    max_saturation_ratio: float = 0.05
    min_contrast: float = 10.0


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing pipeline.

    Attributes
    ----------
    target_size : tuple[int, int]
        Output image dimensions (height, width).
    normalization : NormalizationMethod
        Intensity normalization method.
    preserve_aspect : bool
        Whether to preserve aspect ratio during resize.
    pad_value : int
        Padding value when preserving aspect ratio.
    to_grayscale : bool
        Convert color images to grayscale.
    percentile_low : float
        Lower percentile for percentile normalization.
    percentile_high : float
        Upper percentile for percentile normalization.
    """

    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    normalization: NormalizationMethod = "minmax"
    preserve_aspect: bool = True
    pad_value: int = 0
    to_grayscale: bool = False
    percentile_low: float = 1.0
    percentile_high: float = 99.0


@dataclass
class AugmentationConfig:
    """Configuration for training-time augmentations.

    Attributes
    ----------
    enabled : bool
        Whether augmentation is active.
    horizontal_flip : bool
        Apply random horizontal flips.
    vertical_flip : bool
        Apply random vertical flips.
    rotation_degrees : float
        Maximum rotation angle in degrees.
    brightness_range : tuple[float, float]
        Brightness adjustment range (min, max).
    contrast_range : tuple[float, float]
        Contrast adjustment range (min, max).
    gaussian_noise_std : float
        Standard deviation of Gaussian noise.
    random_crop_scale : tuple[float, float]
        Scale range for random cropping.
    """

    enabled: bool = False
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_degrees: float = 15.0
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    gaussian_noise_std: float = 0.01
    random_crop_scale: tuple[float, float] = (0.85, 1.0)


class QualityFilter:
    """Filter microscopy images based on quality criteria.

    Applies resolution, sharpness, saturation, and contrast checks
    to determine image suitability for model training.

    Parameters
    ----------
    thresholds : QualityThresholds
        Configurable quality thresholds.

    Examples
    --------
    >>> qf = QualityFilter(QualityThresholds(min_blur_score=150.0))
    >>> result = qf.assess(image_array)
    >>> if result.quality_level != QualityLevel.REJECTED:
    ...     process_image(image_array)
    """

    __slots__ = ("_thresholds",)

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        """Initialize quality filter with acceptance thresholds.

        Parameters
        ----------
        thresholds : QualityThresholds | None
            Minimum quality metrics; uses defaults when None.
        """
        self._thresholds = thresholds or QualityThresholds()

    @property
    def thresholds(self) -> QualityThresholds:
        """Return current quality thresholds."""
        return self._thresholds

    def compute_blur_score(self, image: ImageArray) -> float:
        """Compute Laplacian variance as blur metric.

        Higher values indicate sharper images.

        Parameters
        ----------
        image : ImageArray
            Input image array.

        Returns
        -------
        float
            Laplacian variance score.
        """
        if image.ndim == 3:
            gray = self._to_grayscale(image)
        else:
            gray = image

        # Laplacian kernel for edge detection
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ], dtype=np.float64)

        # Convolve with Laplacian kernel
        from scipy.ndimage import convolve  # type: ignore[import-untyped]

        edges = convolve(gray.astype(np.float64), laplacian)
        return float(np.var(edges))

    def compute_saturation_ratio(self, image: ImageArray) -> float:
        """Compute ratio of saturated pixels.

        Parameters
        ----------
        image : ImageArray
            Input image array (8-bit assumed).

        Returns
        -------
        float
            Fraction of pixels at min or max intensity.
        """
        total_pixels = image.size
        saturated = np.sum((image == 0) | (image == 255))
        return float(saturated / total_pixels)

    def compute_contrast(self, image: ImageArray) -> float:
        """Compute contrast as intensity standard deviation.

        Parameters
        ----------
        image : ImageArray
            Input image array.

        Returns
        -------
        float
            Standard deviation of pixel intensities.
        """
        return float(np.std(image.astype(np.float64)))

    def _to_grayscale(self, image: ImageArray) -> ImageArray:
        """Convert RGB image to grayscale.

        Parameters
        ----------
        image : ImageArray
            RGB image with shape (H, W, 3).

        Returns
        -------
        ImageArray
            Grayscale image with shape (H, W).
        """
        if image.ndim == 2:
            return image
        # ITU-R BT.601 luma coefficients
        weights = np.array([0.299, 0.587, 0.114])
        return np.dot(image[..., :3], weights).astype(image.dtype)

    def assess(self, image: ImageArray, file_path: Path | None = None) -> ImageMetadata:
        """Assess image quality and extract metadata.

        Parameters
        ----------
        image : ImageArray
            Input image array.
        file_path : Path | None
            Original file path for metadata.

        Returns
        -------
        ImageMetadata
            Complete metadata including quality assessment.
        """
        height, width = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        bit_depth = 8 if image.dtype == np.uint8 else 16

        blur_score = self.compute_blur_score(image)
        saturation_ratio = self.compute_saturation_ratio(image)
        contrast = self.compute_contrast(image)

        # Determine quality level
        quality_level = self._determine_quality_level(
            width, height, blur_score, saturation_ratio, contrast
        )

        return ImageMetadata(
            file_path=file_path or Path("unknown"),
            width=width,
            height=height,
            channels=channels,
            bit_depth=bit_depth,
            stain_type=StainType.UNKNOWN,
            magnification="unknown",
            quality_level=quality_level,
            blur_score=blur_score,
        )

    def _determine_quality_level(
        self,
        width: int,
        height: int,
        blur_score: float,
        saturation_ratio: float,
        contrast: float,
    ) -> QualityLevel:
        """Classify image quality based on multiple criteria.

        Parameters
        ----------
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
        blur_score : float
            Computed blur score.
        saturation_ratio : float
            Computed saturation ratio.
        contrast : float
            Computed contrast value.

        Returns
        -------
        QualityLevel
            Quality classification.
        """
        t = self._thresholds

        # Check rejection criteria
        if width < t.min_width or height < t.min_height:
            return QualityLevel.REJECTED
        if blur_score < t.min_blur_score * 0.5:
            return QualityLevel.REJECTED
        if saturation_ratio > t.max_saturation_ratio * 2:
            return QualityLevel.REJECTED

        # Check poor quality
        if blur_score < t.min_blur_score:
            return QualityLevel.POOR
        if saturation_ratio > t.max_saturation_ratio:
            return QualityLevel.POOR
        if contrast < t.min_contrast:
            return QualityLevel.POOR

        # Check acceptable
        if blur_score < t.min_blur_score * 1.5:
            return QualityLevel.ACCEPTABLE

        # Check good vs excellent
        high_blur = blur_score > t.min_blur_score * 3
        low_saturation = saturation_ratio < t.max_saturation_ratio * 0.5
        if high_blur and low_saturation:
            return QualityLevel.EXCELLENT

        return QualityLevel.GOOD

    def filter_batch(
        self,
        images: Sequence[tuple[ImageArray, Path]],
        min_quality: QualityLevel = QualityLevel.ACCEPTABLE,
    ) -> list[tuple[ImageArray, ImageMetadata]]:
        """Filter a batch of images by quality threshold.

        Parameters
        ----------
        images : Sequence[tuple[ImageArray, Path]]
            List of (image, path) tuples.
        min_quality : QualityLevel
            Minimum acceptable quality level.

        Returns
        -------
        list[tuple[ImageArray, ImageMetadata]]
            Filtered list of images with metadata.
        """
        quality_order = [
            QualityLevel.REJECTED,
            QualityLevel.POOR,
            QualityLevel.ACCEPTABLE,
            QualityLevel.GOOD,
            QualityLevel.EXCELLENT,
        ]
        min_index = quality_order.index(min_quality)

        results: list[tuple[ImageArray, ImageMetadata]] = []
        for image, path in images:
            metadata = self.assess(image, path)
            if quality_order.index(metadata.quality_level) >= min_index:
                results.append((image, metadata))

        return results


class ImageNormalizer:
    """Normalize image intensities for consistent model input.

    Supports min-max scaling, z-score normalization, and percentile-based
    normalization for robust handling of outlier intensities.

    Parameters
    ----------
    config : PreprocessingConfig
        Preprocessing configuration.

    Examples
    --------
    >>> normalizer = ImageNormalizer(PreprocessingConfig(normalization="zscore"))
    >>> normalized = normalizer.normalize(image)
    """

    __slots__ = ("_config", "_running_mean", "_running_std")

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Initialize image normalizer with preprocessing configuration.

        Parameters
        ----------
        config : PreprocessingConfig | None
            Normalization method and target image size.
        """
        self._config = config or PreprocessingConfig()
        self._running_mean: float | None = None
        self._running_std: float | None = None

    @property
    def config(self) -> PreprocessingConfig:
        """Return preprocessing configuration."""
        return self._config

    def normalize(self, image: ImageArray) -> np.ndarray:
        """Normalize image intensities.

        Parameters
        ----------
        image : ImageArray
            Input image array.

        Returns
        -------
        np.ndarray
            Normalized image with float32 dtype.
        """
        img = image.astype(np.float32)

        if self._config.normalization == "minmax":
            return self._minmax_normalize(img)
        if self._config.normalization == "zscore":
            return self._zscore_normalize(img)
        if self._config.normalization == "percentile":
            return self._percentile_normalize(img)
        return img

    def _minmax_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to [0, 1] range.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Normalized image in [0, 1].
        """
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min < 1e-7:
            return np.zeros_like(image)
        return (image - img_min) / (img_max - img_min)

    def _zscore_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Standardized image with mean=0, std=1.
        """
        mean = image.mean()
        std = image.std()
        if std < 1e-7:
            return image - mean
        return (image - mean) / std

    def _percentile_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply percentile-based normalization.

        Robust to outlier intensities by clipping to percentile range.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Normalized image in [0, 1].
        """
        low = np.percentile(image, self._config.percentile_low)
        high = np.percentile(image, self._config.percentile_high)
        if high - low < 1e-7:
            return np.zeros_like(image)
        clipped = np.clip(image, low, high)
        return (clipped - low) / (high - low)

    def fit(self, images: Sequence[ImageArray]) -> None:
        """Compute running statistics from a dataset.

        For z-score normalization with consistent scaling.

        Parameters
        ----------
        images : Sequence[ImageArray]
            Collection of images to compute statistics from.
        """
        all_pixels = np.concatenate([img.flatten() for img in images])
        self._running_mean = float(np.mean(all_pixels))
        self._running_std = float(np.std(all_pixels))

    def normalize_with_fitted_stats(self, image: ImageArray) -> np.ndarray:
        """Normalize using pre-computed statistics.

        Parameters
        ----------
        image : ImageArray
            Input image.

        Returns
        -------
        np.ndarray
            Normalized image using fitted mean and std.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self._running_mean is None or self._running_std is None:
            msg = "Statistics not fitted. Call fit() first."
            raise RuntimeError(msg)

        img = image.astype(np.float32)
        if self._running_std < 1e-7:
            return img - self._running_mean
        return (img - self._running_mean) / self._running_std


# =============================================================================
# Advanced Microscopy Processing Components
# =============================================================================


class ColorSpace(Enum):
    """Color space representations for microscopy images."""

    RGB = "rgb"
    BGR = "bgr"
    HSV = "hsv"
    LAB = "lab"
    GRAYSCALE = "grayscale"
    HED = "hed"  # Hematoxylin-Eosin-DAB stain separation


class FocusMetric(Enum):
    """Focus quality metrics for microscopy."""

    LAPLACIAN = "laplacian"
    GRADIENT = "gradient"
    TENENGRAD = "tenengrad"
    BRENNER = "brenner"
    VOLLATH = "vollath"


class ArtifactType(Enum):
    """Types of microscopy image artifacts."""

    DUST = "dust"
    BUBBLE = "bubble"
    SCRATCH = "scratch"
    VIGNETTING = "vignetting"
    UNEVEN_ILLUMINATION = "uneven_illumination"
    OUT_OF_FOCUS = "out_of_focus"
    MOTION_BLUR = "motion_blur"
    OVEREXPOSURE = "overexposure"
    UNDEREXPOSURE = "underexposure"


class SegmentationType(Enum):
    """Segmentation methods for cell detection."""

    THRESHOLD = "threshold"
    OTSU = "otsu"
    ADAPTIVE = "adaptive"
    WATERSHED = "watershed"
    CONNECTED_COMPONENTS = "connected_components"


class MorphologyOperation(Enum):
    """Morphological operations for image processing."""

    EROSION = "erosion"
    DILATION = "dilation"
    OPENING = "opening"
    CLOSING = "closing"
    GRADIENT = "gradient"
    TOP_HAT = "top_hat"
    BLACK_HAT = "black_hat"


@dataclass(slots=True)
class FocusResult:
    """Result of focus quality assessment.

    Attributes
    ----------
    metric : FocusMetric
        Metric used for assessment.
    score : float
        Focus quality score.
    is_in_focus : bool
        Whether image passes focus threshold.
    threshold : float
        Threshold used for classification.
    regions : list[tuple[int, int, int, int]]
        Regions with poor focus (x, y, w, h).
    """

    metric: FocusMetric
    score: float
    is_in_focus: bool
    threshold: float
    regions: list[tuple[int, int, int, int]]


@dataclass(slots=True)
class ArtifactDetectionResult:
    """Result of artifact detection analysis.

    Attributes
    ----------
    artifacts_found : list[ArtifactType]
        Types of artifacts detected.
    artifact_masks : dict[ArtifactType, np.ndarray]
        Binary masks for each artifact type.
    severity_scores : dict[ArtifactType, float]
        Severity score for each artifact (0-1).
    is_usable : bool
        Whether image is still usable.
    """

    artifacts_found: list[ArtifactType]
    artifact_masks: dict[ArtifactType, np.ndarray]
    severity_scores: dict[ArtifactType, float]
    is_usable: bool


@dataclass(slots=True)
class CellSegmentationResult:
    """Result of cell segmentation.

    Attributes
    ----------
    mask : np.ndarray
        Binary segmentation mask.
    cell_count : int
        Number of cells detected.
    cell_centroids : list[tuple[float, float]]
        Centroids of detected cells.
    cell_areas : list[float]
        Areas of detected cells in pixels.
    cell_contours : list[np.ndarray]
        Contours of detected cells.
    confidence : float
        Overall segmentation confidence.
    """

    mask: np.ndarray
    cell_count: int
    cell_centroids: list[tuple[float, float]]
    cell_areas: list[float]
    cell_contours: list[np.ndarray]
    confidence: float


@dataclass(slots=True)
class StainNormalizationResult:
    """Result of stain normalization.

    Attributes
    ----------
    normalized_image : np.ndarray
        Stain-normalized image.
    stain_matrix : np.ndarray
        Estimated stain matrix.
    concentrations : np.ndarray
        Stain concentrations.
    source_stain_type : StainType
        Detected source stain type.
    """

    normalized_image: np.ndarray
    stain_matrix: np.ndarray
    concentrations: np.ndarray
    source_stain_type: StainType


@dataclass(slots=True)
class TileInfo:
    """Information about an image tile.

    Attributes
    ----------
    x : int
        X coordinate of tile origin.
    y : int
        Y coordinate of tile origin.
    width : int
        Tile width.
    height : int
        Tile height.
    data : np.ndarray
        Tile image data.
    overlap : int
        Overlap with adjacent tiles.
    """

    x: int
    y: int
    width: int
    height: int
    data: np.ndarray
    overlap: int


class FocusAnalyzer:
    """Analyze focus quality in microscopy images.

    Provides multiple focus metrics and regional focus mapping
    for identifying out-of-focus areas.

    Parameters
    ----------
    metric : FocusMetric
        Primary focus metric to use.
    threshold : float
        Threshold for in-focus classification.

    Examples
    --------
    >>> analyzer = FocusAnalyzer(FocusMetric.LAPLACIAN, threshold=100.0)
    >>> result = analyzer.analyze(image)
    >>> if not result.is_in_focus:
    ...     print(f"Image is blurry: score={result.score:.2f}")
    """

    __slots__ = ("_metric", "_threshold", "_window_size")

    def __init__(
        self,
        metric: FocusMetric = FocusMetric.LAPLACIAN,
        threshold: float = 100.0,
        window_size: int = 64,
    ) -> None:
        """Initialize focus analyzer with detection parameters.

        Parameters
        ----------
        metric : FocusMetric
            Focus measurement algorithm.
        threshold : float
            Minimum score for in-focus classification.
        window_size : int
            Local window size for region-based analysis.
        """
        self._metric = metric
        self._threshold = threshold
        self._window_size = window_size

    @property
    def metric(self) -> FocusMetric:
        """Return focus metric in use."""
        return self._metric

    def analyze(self, image: ImageArray) -> FocusResult:
        """Analyze focus quality of an image.

        Parameters
        ----------
        image : ImageArray
            Input image array.

        Returns
        -------
        FocusResult
            Comprehensive focus analysis result.
        """
        gray = self._ensure_grayscale(image)

        score = self._compute_focus_score(gray)
        is_in_focus = score >= self._threshold

        # Find unfocused regions
        regions = self._find_unfocused_regions(gray)

        return FocusResult(
            metric=self._metric,
            score=score,
            is_in_focus=is_in_focus,
            threshold=self._threshold,
            regions=regions,
        )

    def _compute_focus_score(self, gray: np.ndarray) -> float:
        """Compute focus score using selected metric."""
        if self._metric == FocusMetric.LAPLACIAN:
            return self._laplacian_variance(gray)
        if self._metric == FocusMetric.GRADIENT:
            return self._gradient_magnitude(gray)
        if self._metric == FocusMetric.TENENGRAD:
            return self._tenengrad(gray)
        if self._metric == FocusMetric.BRENNER:
            return self._brenner(gray)
        if self._metric == FocusMetric.VOLLATH:
            return self._vollath(gray)
        return 0.0

    def _laplacian_variance(self, gray: np.ndarray) -> float:
        """Compute Laplacian variance."""
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ], dtype=np.float64)
        from scipy.ndimage import convolve
        edges = convolve(gray.astype(np.float64), laplacian)
        return float(np.var(edges))

    def _gradient_magnitude(self, gray: np.ndarray) -> float:
        """Compute gradient magnitude."""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        from scipy.ndimage import convolve
        gx = convolve(gray.astype(np.float64), sobel_x)
        gy = convolve(gray.astype(np.float64), sobel_y)
        magnitude = np.sqrt(gx**2 + gy**2)
        return float(np.mean(magnitude))

    def _tenengrad(self, gray: np.ndarray) -> float:
        """Compute Tenengrad focus measure."""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        from scipy.ndimage import convolve
        gx = convolve(gray.astype(np.float64), sobel_x)
        gy = convolve(gray.astype(np.float64), sobel_y)
        return float(np.mean(gx**2 + gy**2))

    def _brenner(self, gray: np.ndarray) -> float:
        """Compute Brenner focus measure."""
        h, w = gray.shape
        gray_f = gray.astype(np.float64)
        diff = (gray_f[:, 2:] - gray_f[:, :-2])**2
        return float(np.sum(diff))

    def _vollath(self, gray: np.ndarray) -> float:
        """Compute Vollath's correlation focus measure."""
        gray_f = gray.astype(np.float64)
        h, w = gray_f.shape
        vol = np.sum(gray_f[:, :-1] * gray_f[:, 1:]) - np.sum(
            gray_f[:, :-2] * gray_f[:, 2:]
        )
        return float(vol)

    def _find_unfocused_regions(
        self,
        gray: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Find regions with poor focus."""
        regions: list[tuple[int, int, int, int]] = []
        h, w = gray.shape
        ws = self._window_size

        for y in range(0, h - ws + 1, ws):
            for x in range(0, w - ws + 1, ws):
                window = gray[y:y+ws, x:x+ws]
                score = self._laplacian_variance(window)
                if score < self._threshold:
                    regions.append((x, y, ws, ws))

        return regions

    def _ensure_grayscale(self, image: ImageArray) -> np.ndarray:
        """Ensure image is grayscale."""
        if image.ndim == 2:
            return image
        weights = np.array([0.299, 0.587, 0.114])
        return np.dot(image[..., :3], weights).astype(np.float64)


class ArtifactDetector:
    """Detect artifacts in microscopy images.

    Identifies common artifacts like dust, bubbles, scratches,
    and illumination issues that affect image quality.

    Parameters
    ----------
    sensitivity : float
        Detection sensitivity (0.0 to 1.0).

    Examples
    --------
    >>> detector = ArtifactDetector(sensitivity=0.7)
    >>> result = detector.detect(image)
    >>> for artifact in result.artifacts_found:
    ...     print(f"Found {artifact.value} with severity {result.severity_scores[artifact]:.2f}")
    """

    __slots__ = ("_sensitivity", "_min_artifact_size")

    def __init__(
        self,
        sensitivity: float = 0.5,
        min_artifact_size: int = 50,
    ) -> None:
        """Initialize artifact detector with sensitivity controls.

        Parameters
        ----------
        sensitivity : float
            Detection sensitivity from 0.0 (lenient) to 1.0 (strict).
        min_artifact_size : int
            Minimum contiguous pixels to report as artifact.
        """
        self._sensitivity = np.clip(sensitivity, 0.0, 1.0)
        self._min_artifact_size = min_artifact_size

    @property
    def sensitivity(self) -> float:
        """Return detection sensitivity."""
        return self._sensitivity

    def detect(self, image: ImageArray) -> ArtifactDetectionResult:
        """Detect artifacts in an image.

        Parameters
        ----------
        image : ImageArray
            Input image array.

        Returns
        -------
        ArtifactDetectionResult
            Comprehensive artifact detection result.
        """
        artifacts_found: list[ArtifactType] = []
        artifact_masks: dict[ArtifactType, np.ndarray] = {}
        severity_scores: dict[ArtifactType, float] = {}

        # Check for dust/particles
        dust_mask, dust_severity = self._detect_dust(image)
        if dust_severity > self._sensitivity * 0.5:
            artifacts_found.append(ArtifactType.DUST)
            artifact_masks[ArtifactType.DUST] = dust_mask
            severity_scores[ArtifactType.DUST] = dust_severity

        # Check for bubbles
        bubble_mask, bubble_severity = self._detect_bubbles(image)
        if bubble_severity > self._sensitivity * 0.5:
            artifacts_found.append(ArtifactType.BUBBLE)
            artifact_masks[ArtifactType.BUBBLE] = bubble_mask
            severity_scores[ArtifactType.BUBBLE] = bubble_severity

        # Check for uneven illumination
        vignette_mask, vignette_severity = self._detect_vignetting(image)
        if vignette_severity > self._sensitivity * 0.3:
            artifacts_found.append(ArtifactType.VIGNETTING)
            artifact_masks[ArtifactType.VIGNETTING] = vignette_mask
            severity_scores[ArtifactType.VIGNETTING] = vignette_severity

        # Check for exposure issues
        overexposure_severity = self._detect_overexposure(image)
        if overexposure_severity > self._sensitivity * 0.5:
            artifacts_found.append(ArtifactType.OVEREXPOSURE)
            severity_scores[ArtifactType.OVEREXPOSURE] = overexposure_severity

        underexposure_severity = self._detect_underexposure(image)
        if underexposure_severity > self._sensitivity * 0.5:
            artifacts_found.append(ArtifactType.UNDEREXPOSURE)
            severity_scores[ArtifactType.UNDEREXPOSURE] = underexposure_severity

        # Determine overall usability
        total_severity = sum(severity_scores.values())
        is_usable = total_severity < 1.5

        return ArtifactDetectionResult(
            artifacts_found=artifacts_found,
            artifact_masks=artifact_masks,
            severity_scores=severity_scores,
            is_usable=is_usable,
        )

    def _detect_dust(self, image: ImageArray) -> tuple[np.ndarray, float]:
        """Detect dust particles."""
        gray = self._to_grayscale(image)

        # Look for small dark spots
        mean_val = np.mean(gray)
        threshold = mean_val * 0.5
        dust_mask = gray < threshold

        # Morphological opening to remove noise
        from scipy.ndimage import binary_opening
        dust_mask = binary_opening(dust_mask, iterations=1)

        severity = np.sum(dust_mask) / dust_mask.size
        return dust_mask.astype(np.uint8) * 255, min(severity * 10, 1.0)

    def _detect_bubbles(self, image: ImageArray) -> tuple[np.ndarray, float]:
        """Detect bubbles (bright circular regions)."""
        gray = self._to_grayscale(image)

        # Look for bright circular regions
        high_threshold = np.percentile(gray, 95)
        bright_mask = gray > high_threshold

        # Morphological closing to merge nearby bright regions
        from scipy.ndimage import binary_closing
        bubble_mask = binary_closing(bright_mask, iterations=2)

        severity = np.sum(bubble_mask) / bubble_mask.size
        return bubble_mask.astype(np.uint8) * 255, min(severity * 5, 1.0)

    def _detect_vignetting(self, image: ImageArray) -> tuple[np.ndarray, float]:
        """Detect vignetting (corner darkening)."""
        gray = self._to_grayscale(image)
        h, w = gray.shape

        # Compare corner intensities to center
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_mean = np.mean(center_region)

        corner_size = min(h, w) // 8
        corners = [
            gray[:corner_size, :corner_size],  # top-left
            gray[:corner_size, -corner_size:],  # top-right
            gray[-corner_size:, :corner_size],  # bottom-left
            gray[-corner_size:, -corner_size:],  # bottom-right
        ]

        corner_means = [np.mean(c) for c in corners]
        avg_corner = np.mean(corner_means)

        if center_mean > 0:
            vignette_ratio = 1 - (avg_corner / center_mean)
        else:
            vignette_ratio = 0.0

        # Create vignette mask
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        vignette_mask = (distance / max_dist > 0.7).astype(np.uint8) * 255

        return vignette_mask, max(0.0, min(vignette_ratio, 1.0))

    def _detect_overexposure(self, image: ImageArray) -> float:
        """Detect overexposure."""
        if image.dtype == np.uint8:
            saturated = np.sum(image >= 254) / image.size
        else:
            saturated = np.sum(image >= np.iinfo(image.dtype).max - 1) / image.size
        return min(saturated * 10, 1.0)

    def _detect_underexposure(self, image: ImageArray) -> float:
        """Detect underexposure."""
        dark = np.sum(image <= 5) / image.size
        return min(dark * 10, 1.0)

    def _to_grayscale(self, image: ImageArray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if image.ndim == 2:
            return image.astype(np.float64)
        weights = np.array([0.299, 0.587, 0.114])
        return np.dot(image[..., :3], weights).astype(np.float64)


class CellSegmenter:
    """Segment cells in microscopy images.

    Provides multiple segmentation algorithms for detecting
    and delineating individual cells.

    Parameters
    ----------
    method : SegmentationType
        Segmentation method to use.
    min_cell_size : int
        Minimum cell size in pixels.
    max_cell_size : int
        Maximum cell size in pixels.

    Examples
    --------
    >>> segmenter = CellSegmenter(SegmentationType.OTSU)
    >>> result = segmenter.segment(image)
    >>> print(f"Detected {result.cell_count} cells")
    """

    __slots__ = ("_method", "_min_cell_size", "_max_cell_size")

    def __init__(
        self,
        method: SegmentationType = SegmentationType.OTSU,
        min_cell_size: int = 100,
        max_cell_size: int = 10000,
    ) -> None:
        """Initialize cell segmenter with morphometric bounds.

        Parameters
        ----------
        method : SegmentationType
            Thresholding algorithm for segmentation.
        min_cell_size : int
            Minimum object area in pixels to classify as cell.
        max_cell_size : int
            Maximum object area in pixels to classify as cell.
        """
        self._method = method
        self._min_cell_size = min_cell_size
        self._max_cell_size = max_cell_size

    @property
    def method(self) -> SegmentationType:
        """Return segmentation method."""
        return self._method

    def segment(self, image: ImageArray) -> CellSegmentationResult:
        """Segment cells in an image.

        Parameters
        ----------
        image : ImageArray
            Input image array.

        Returns
        -------
        CellSegmentationResult
            Comprehensive segmentation result.
        """
        gray = self._to_grayscale(image)

        if self._method == SegmentationType.THRESHOLD:
            mask = self._threshold_segment(gray)
        elif self._method == SegmentationType.OTSU:
            mask = self._otsu_segment(gray)
        elif self._method == SegmentationType.ADAPTIVE:
            mask = self._adaptive_segment(gray)
        elif self._method == SegmentationType.WATERSHED:
            mask = self._watershed_segment(gray)
        else:
            mask = self._connected_components_segment(gray)

        # Extract cell properties
        centroids, areas, contours = self._extract_cell_properties(mask)

        # Filter by size
        valid_indices = [
            i for i, area in enumerate(areas)
            if self._min_cell_size <= area <= self._max_cell_size
        ]

        centroids = [centroids[i] for i in valid_indices]
        areas = [areas[i] for i in valid_indices]
        contours = [contours[i] for i in valid_indices]

        # Compute confidence based on segmentation quality
        confidence = self._compute_confidence(mask, len(centroids))

        return CellSegmentationResult(
            mask=mask,
            cell_count=len(centroids),
            cell_centroids=centroids,
            cell_areas=areas,
            cell_contours=contours,
            confidence=confidence,
        )

    def _threshold_segment(self, gray: np.ndarray) -> np.ndarray:
        """Simple threshold segmentation."""
        threshold = np.mean(gray) - 0.5 * np.std(gray)
        return (gray < threshold).astype(np.uint8) * 255

    def _otsu_segment(self, gray: np.ndarray) -> np.ndarray:
        """Otsu's automatic thresholding."""
        # Compute histogram
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64) / hist.sum()

        # Compute cumulative sums and means
        cum_sum = np.cumsum(hist)
        cum_mean = np.cumsum(hist * np.arange(256))

        global_mean = cum_mean[-1]

        # Compute between-class variance for each threshold
        variance = np.zeros(256)
        for t in range(1, 256):
            w0 = cum_sum[t]
            w1 = 1 - w0
            if w0 == 0 or w1 == 0:
                continue
            m0 = cum_mean[t] / w0
            m1 = (global_mean - cum_mean[t]) / w1
            variance[t] = w0 * w1 * (m0 - m1) ** 2

        threshold = np.argmax(variance)
        return (gray < threshold).astype(np.uint8) * 255

    def _adaptive_segment(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding."""
        # Use local mean as threshold
        from scipy.ndimage import uniform_filter
        block_size = 51
        local_mean = uniform_filter(gray.astype(np.float64), size=block_size)
        offset = 10
        return (gray < (local_mean - offset)).astype(np.uint8) * 255

    def _watershed_segment(self, gray: np.ndarray) -> np.ndarray:
        """Watershed segmentation."""
        # Basic watershed implementation
        from scipy.ndimage import distance_transform_edt, label

        # Threshold first
        binary = self._otsu_segment(gray) > 0

        # Distance transform
        distance = distance_transform_edt(binary)

        # Find local maxima as markers
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(distance, size=20)
        markers = (distance == local_max) & (distance > 0)

        # Label markers
        labeled_markers, _ = label(markers)

        # For simplicity, return binary mask
        return binary.astype(np.uint8) * 255

    def _connected_components_segment(self, gray: np.ndarray) -> np.ndarray:
        """Connected components segmentation."""
        binary = self._otsu_segment(gray)
        from scipy.ndimage import label
        labeled, num_features = label(binary)
        return (labeled > 0).astype(np.uint8) * 255

    def _extract_cell_properties(
        self,
        mask: np.ndarray,
    ) -> tuple[list[tuple[float, float]], list[float], list[np.ndarray]]:
        """Extract properties of segmented cells."""
        from scipy.ndimage import label, center_of_mass

        labeled, num_features = label(mask)

        centroids: list[tuple[float, float]] = []
        areas: list[float] = []
        contours: list[np.ndarray] = []

        for i in range(1, num_features + 1):
            component = labeled == i
            area = float(np.sum(component))
            areas.append(area)

            # Get centroid
            com = center_of_mass(component)
            centroids.append((float(com[1]), float(com[0])))  # (x, y)

            # Get contour points (simplified)
            coords = np.argwhere(component)
            contours.append(coords)

        return centroids, areas, contours

    def _compute_confidence(self, mask: np.ndarray, cell_count: int) -> float:
        """Compute segmentation confidence."""
        # Based on foreground ratio and cell count
        fg_ratio = np.sum(mask > 0) / mask.size

        # Expect 10-50% foreground for typical microscopy
        ratio_score = float(1.0 - abs(fg_ratio - 0.3) / 0.3)
        ratio_score = float(max(0.0, min(1.0, ratio_score)))

        # Expect reasonable cell count
        count_score = 1.0 if 5 <= cell_count <= 100 else 0.5

        return float((ratio_score + count_score) / 2)

    def _to_grayscale(self, image: ImageArray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if image.ndim == 2:
            return image.astype(np.float64)
        weights = np.array([0.299, 0.587, 0.114])
        return np.dot(image[..., :3], weights).astype(np.float64)


class StainNormalizer:
    """Normalize staining in histopathology images.

    Applies color normalization to reduce staining variability
    between slides and institutions.

    Parameters
    ----------
    target_stain_matrix : np.ndarray | None
        Reference stain matrix for normalization.

    Examples
    --------
    >>> normalizer = StainNormalizer()
    >>> result = normalizer.normalize(image)
    >>> normalized_image = result.normalized_image
    """

    __slots__ = ("_target_stain_matrix",)

    # Default stain matrices for common stains
    HE_STAIN_MATRIX: Final = np.array([
        [0.65, 0.70, 0.29],  # Hematoxylin
        [0.07, 0.99, 0.11],  # Eosin
    ])

    def __init__(self, target_stain_matrix: np.ndarray | None = None) -> None:
        """Initialize stain normalizer with target color matrix.

        Parameters
        ----------
        target_stain_matrix : np.ndarray | None
            Reference stain vectors; defaults to H&E matrix.
        """
        self._target_stain_matrix = (
            target_stain_matrix if target_stain_matrix is not None
            else self.HE_STAIN_MATRIX
        )

    def normalize(self, image: ImageArray) -> StainNormalizationResult:
        """Normalize staining in an image.

        Parameters
        ----------
        image : ImageArray
            RGB image to normalize.

        Returns
        -------
        StainNormalizationResult
            Normalized image with stain information.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            msg = "Input must be RGB image"
            raise ValueError(msg)

        # Convert to optical density
        od = self._rgb_to_od(image)

        # Estimate stain matrix
        stain_matrix = self._estimate_stain_matrix(od)

        # Compute concentrations
        concentrations = self._compute_concentrations(od, stain_matrix)

        # Reconstruct with target stain matrix
        normalized = self._reconstruct_image(concentrations, self._target_stain_matrix)

        # Detect stain type
        stain_type = self._detect_stain_type(stain_matrix)

        return StainNormalizationResult(
            normalized_image=normalized,
            stain_matrix=stain_matrix,
            concentrations=concentrations,
            source_stain_type=stain_type,
        )

    def _rgb_to_od(self, image: ImageArray) -> np.ndarray:
        """Convert RGB to optical density."""
        img = image.astype(np.float64) / 255.0
        img = np.clip(img, 1e-6, 1.0)
        return -np.log(img)

    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density to RGB."""
        rgb = np.exp(-od)
        rgb = np.clip(rgb * 255, 0, 255)
        return rgb.astype(np.uint8)

    def _estimate_stain_matrix(self, od: np.ndarray) -> np.ndarray:
        """Estimate stain matrix using SVD."""
        # Flatten and remove background
        h, w, c = od.shape
        od_flat = od.reshape(-1, c)

        # Keep only pixels with significant optical density
        od_norm = np.linalg.norm(od_flat, axis=1)
        significant = od_norm > 0.15
        od_significant = od_flat[significant]

        if len(od_significant) < 100:
            return self._target_stain_matrix

        # PCA to find principal stain directions
        od_centered = od_significant - od_significant.mean(axis=0)
        _, _, Vt = np.linalg.svd(od_centered, full_matrices=False)

        # Take first two components as stain vectors
        stain_matrix = Vt[:2]

        # Ensure positive optical density
        stain_matrix = np.abs(stain_matrix)

        # Normalize each stain vector
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)

        return stain_matrix

    def _compute_concentrations(
        self,
        od: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute stain concentrations."""
        h, w, c = od.shape
        od_flat = od.reshape(-1, c)

        # Solve for concentrations: OD = C * S
        # C = OD * S^T * (S * S^T)^-1
        pseudo_inverse = np.linalg.pinv(stain_matrix.T)
        concentrations = od_flat @ pseudo_inverse.T

        return concentrations.reshape(h, w, -1)

    def _reconstruct_image(
        self,
        concentrations: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct image from concentrations and stain matrix."""
        h, w, n_stains = concentrations.shape
        conc_flat = concentrations.reshape(-1, n_stains)

        # OD = C * S
        od_reconstructed = conc_flat @ stain_matrix

        # Clip and reshape
        od_reconstructed = np.clip(od_reconstructed, 0, 3.0)
        od_reconstructed = od_reconstructed.reshape(h, w, 3)

        return self._od_to_rgb(od_reconstructed)

    def _detect_stain_type(self, stain_matrix: np.ndarray) -> StainType:
        """Detect stain type from stain matrix."""
        # Compare to reference stain matrices
        he_similarity = np.sum(stain_matrix * self.HE_STAIN_MATRIX) / (
            np.linalg.norm(stain_matrix) * np.linalg.norm(self.HE_STAIN_MATRIX)
        )

        if he_similarity > 0.8:
            return StainType.HEMATOXYLIN_EOSIN
        if he_similarity > 0.6:
            return StainType.GIEMSA
        return StainType.UNKNOWN


class TileManager:
    """Manage tiling of large microscopy images.

    Splits large images into overlapping tiles for efficient
    processing and seamless reconstruction.

    Parameters
    ----------
    tile_size : int
        Size of each tile (square).
    overlap : int
        Overlap between adjacent tiles.

    Examples
    --------
    >>> manager = TileManager(tile_size=256, overlap=32)
    >>> tiles = manager.create_tiles(large_image)
    >>> for tile in tiles:
    ...     processed = process_tile(tile.data)
    >>> reconstructed = manager.merge_tiles(processed_tiles, original_shape)
    """

    __slots__ = ("_tile_size", "_overlap")

    def __init__(self, tile_size: int = 256, overlap: int = 32) -> None:
        """Initialize tile manager with partitioning parameters.

        Parameters
        ----------
        tile_size : int
            Side length of each square tile in pixels.
        overlap : int
            Pixel overlap between adjacent tiles.
        """
        self._tile_size = tile_size
        self._overlap = overlap

    @property
    def tile_size(self) -> int:
        """Return tile size."""
        return self._tile_size

    @property
    def overlap(self) -> int:
        """Return tile overlap."""
        return self._overlap

    def create_tiles(self, image: ImageArray) -> list[TileInfo]:
        """Split image into overlapping tiles.

        Parameters
        ----------
        image : ImageArray
            Input image to tile.

        Returns
        -------
        list[TileInfo]
            List of tile information objects.
        """
        h, w = image.shape[:2]
        stride = self._tile_size - self._overlap
        tiles: list[TileInfo] = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Calculate tile bounds
                x_end = min(x + self._tile_size, w)
                y_end = min(y + self._tile_size, h)

                # Extract tile
                tile_data = image[y:y_end, x:x_end]

                # Pad if necessary
                if tile_data.shape[0] < self._tile_size or tile_data.shape[1] < self._tile_size:
                    if image.ndim == 3:
                        padded = np.zeros(
                            (self._tile_size, self._tile_size, image.shape[2]),
                            dtype=image.dtype,
                        )
                    else:
                        padded = np.zeros(
                            (self._tile_size, self._tile_size),
                            dtype=image.dtype,
                        )
                    padded[:tile_data.shape[0], :tile_data.shape[1]] = tile_data
                    tile_data = padded

                tiles.append(TileInfo(
                    x=x,
                    y=y,
                    width=self._tile_size,
                    height=self._tile_size,
                    data=tile_data,
                    overlap=self._overlap,
                ))

        return tiles

    def merge_tiles(
        self,
        tiles: list[TileInfo],
        output_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Merge tiles back into a single image.

        Parameters
        ----------
        tiles : list[TileInfo]
            List of processed tiles.
        output_shape : tuple[int, ...]
            Original image shape.

        Returns
        -------
        np.ndarray
            Reconstructed image.
        """
        output = np.zeros(output_shape, dtype=tiles[0].data.dtype)
        weight = np.zeros(output_shape[:2], dtype=np.float32)

        for tile in tiles:
            # Calculate valid region (excluding overlaps at edges)
            x_start = tile.x
            y_start = tile.y
            x_end = min(tile.x + tile.width, output_shape[1])
            y_end = min(tile.y + tile.height, output_shape[0])

            tile_h = y_end - y_start
            tile_w = x_end - x_start

            # Add tile to output with blending
            tile_data = tile.data[:tile_h, :tile_w]
            if output.ndim == 3:
                output[y_start:y_end, x_start:x_end] += tile_data
            else:
                output[y_start:y_end, x_start:x_end] += tile_data

            weight[y_start:y_end, x_start:x_end] += 1.0

        # Normalize by weight
        weight = np.maximum(weight, 1.0)
        if output.ndim == 3:
            output = output / weight[:, :, np.newaxis]
        else:
            output = output / weight

        return output.astype(tiles[0].data.dtype)

    def compute_tile_grid(
        self,
        image_shape: tuple[int, int],
    ) -> tuple[int, int]:
        """Compute number of tiles in each dimension.

        Parameters
        ----------
        image_shape : tuple[int, int]
            Image height and width.

        Returns
        -------
        tuple[int, int]
            Number of tiles (rows, cols).
        """
        h, w = image_shape
        stride = self._tile_size - self._overlap
        n_rows = (h + stride - 1) // stride
        n_cols = (w + stride - 1) // stride
        return n_rows, n_cols


class MorphologyProcessor:
    """Apply morphological operations to images.

    Provides erosion, dilation, opening, closing, and other
    morphological operations for image processing.

    Parameters
    ----------
    kernel_size : int
        Size of structuring element.

    Examples
    --------
    >>> processor = MorphologyProcessor(kernel_size=5)
    >>> cleaned = processor.apply(binary_mask, MorphologyOperation.OPENING)
    """

    __slots__ = ("_kernel_size", "_kernel")

    def __init__(self, kernel_size: int = 3) -> None:
        """Initialize morphology processor with structuring element.

        Parameters
        ----------
        kernel_size : int
            Diameter of the circular structuring element.
        """
        self._kernel_size = kernel_size
        self._kernel = self._create_kernel()

    def _create_kernel(self) -> np.ndarray:
        """Create circular structuring element."""
        size = self._kernel_size
        y, x = np.ogrid[:size, :size]
        center = size // 2
        r = size // 2
        mask = ((x - center) ** 2 + (y - center) ** 2) <= r ** 2
        return mask.astype(np.uint8)

    def apply(
        self,
        image: np.ndarray,
        operation: MorphologyOperation,
        iterations: int = 1,
    ) -> np.ndarray:
        """Apply morphological operation.

        Parameters
        ----------
        image : np.ndarray
            Binary or grayscale image.
        operation : MorphologyOperation
            Operation to apply.
        iterations : int
            Number of times to apply operation.

        Returns
        -------
        np.ndarray
            Processed image.
        """
        from scipy.ndimage import (
            binary_erosion,
            binary_dilation,
            binary_opening,
            binary_closing,
            grey_erosion,
            grey_dilation,
        )

        is_binary = np.unique(image).size <= 2

        if is_binary:
            if operation == MorphologyOperation.EROSION:
                return binary_erosion(
                    image, structure=self._kernel, iterations=iterations
                ).astype(image.dtype)
            if operation == MorphologyOperation.DILATION:
                return binary_dilation(
                    image, structure=self._kernel, iterations=iterations
                ).astype(image.dtype)
            if operation == MorphologyOperation.OPENING:
                return binary_opening(
                    image, structure=self._kernel, iterations=iterations
                ).astype(image.dtype)
            if operation == MorphologyOperation.CLOSING:
                return binary_closing(
                    image, structure=self._kernel, iterations=iterations
                ).astype(image.dtype)
        else:
            if operation == MorphologyOperation.EROSION:
                result = image
                for _ in range(iterations):
                    result = grey_erosion(result, footprint=self._kernel)
                return result
            if operation == MorphologyOperation.DILATION:
                result = image
                for _ in range(iterations):
                    result = grey_dilation(result, footprint=self._kernel)
                return result

        # For gradient, top-hat, black-hat
        if operation == MorphologyOperation.GRADIENT:
            dilated = binary_dilation(image, structure=self._kernel)
            eroded = binary_erosion(image, structure=self._kernel)
            return (dilated.astype(np.int16) - eroded.astype(np.int16)).astype(image.dtype)

        if operation == MorphologyOperation.TOP_HAT:
            opened = binary_opening(image, structure=self._kernel)
            return (image.astype(np.int16) - opened.astype(np.int16)).astype(image.dtype)

        if operation == MorphologyOperation.BLACK_HAT:
            closed = binary_closing(image, structure=self._kernel)
            return (closed.astype(np.int16) - image.astype(np.int16)).astype(image.dtype)

        return image


# =============================================================================
# Factory Functions
# =============================================================================


def create_quality_filter(
    min_blur_score: float = 100.0,
    min_width: int = 256,
    min_height: int = 256,
) -> QualityFilter:
    """Create a QualityFilter with specified thresholds.

    Parameters
    ----------
    min_blur_score : float
        Minimum blur score for acceptable quality.
    min_width : int
        Minimum image width.
    min_height : int
        Minimum image height.

    Returns
    -------
    QualityFilter
        Configured quality filter.

    Examples
    --------
    >>> filter = create_quality_filter(min_blur_score=150.0)
    >>> result = filter.assess(image)
    """
    thresholds = QualityThresholds(
        min_blur_score=min_blur_score,
        min_width=min_width,
        min_height=min_height,
    )
    return QualityFilter(thresholds)


def create_image_normalizer(
    normalization: NormalizationMethod = "minmax",
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> ImageNormalizer:
    """Create an ImageNormalizer with specified configuration.

    Parameters
    ----------
    normalization : NormalizationMethod
        Normalization method to use.
    target_size : tuple[int, int]
        Target image size.

    Returns
    -------
    ImageNormalizer
        Configured normalizer.
    """
    config = PreprocessingConfig(
        normalization=normalization,
        target_size=target_size,
    )
    return ImageNormalizer(config)


def create_focus_analyzer(
    metric: FocusMetric = FocusMetric.LAPLACIAN,
    threshold: float = 100.0,
) -> FocusAnalyzer:
    """Create a FocusAnalyzer with specified parameters.

    Parameters
    ----------
    metric : FocusMetric
        Focus metric to use.
    threshold : float
        Threshold for in-focus classification.

    Returns
    -------
    FocusAnalyzer
        Configured focus analyzer.
    """
    return FocusAnalyzer(metric=metric, threshold=threshold)


def create_artifact_detector(
    sensitivity: float = 0.5,
) -> ArtifactDetector:
    """Create an ArtifactDetector with specified sensitivity.

    Parameters
    ----------
    sensitivity : float
        Detection sensitivity (0.0 to 1.0).

    Returns
    -------
    ArtifactDetector
        Configured artifact detector.
    """
    return ArtifactDetector(sensitivity=sensitivity)


def create_cell_segmenter(
    method: SegmentationType = SegmentationType.OTSU,
    min_cell_size: int = 100,
    max_cell_size: int = 10000,
) -> CellSegmenter:
    """Create a CellSegmenter with specified parameters.

    Parameters
    ----------
    method : SegmentationType
        Segmentation method to use.
    min_cell_size : int
        Minimum cell size in pixels.
    max_cell_size : int
        Maximum cell size in pixels.

    Returns
    -------
    CellSegmenter
        Configured cell segmenter.
    """
    return CellSegmenter(
        method=method,
        min_cell_size=min_cell_size,
        max_cell_size=max_cell_size,
    )


def create_stain_normalizer(
    reference_image: ImageArray | None = None,
) -> StainNormalizer:
    """Create a StainNormalizer with optional reference.

    Parameters
    ----------
    reference_image : ImageArray | None
        Reference image to extract target stain matrix.

    Returns
    -------
    StainNormalizer
        Configured stain normalizer.
    """
    target_matrix = None
    if reference_image is not None:
        # Extract stain matrix from reference
        normalizer = StainNormalizer()
        result = normalizer.normalize(reference_image)
        target_matrix = result.stain_matrix
    return StainNormalizer(target_stain_matrix=target_matrix)


def create_tile_manager(
    tile_size: int = 256,
    overlap: int = 32,
) -> TileManager:
    """Create a TileManager with specified parameters.

    Parameters
    ----------
    tile_size : int
        Size of each tile.
    overlap : int
        Overlap between tiles.

    Returns
    -------
    TileManager
        Configured tile manager.
    """
    return TileManager(tile_size=tile_size, overlap=overlap)


def create_morphology_processor(
    kernel_size: int = 3,
) -> MorphologyProcessor:
    """Create a MorphologyProcessor with specified kernel.

    Parameters
    ----------
    kernel_size : int
        Size of structuring element.

    Returns
    -------
    MorphologyProcessor
        Configured morphology processor.
    """
    return MorphologyProcessor(kernel_size=kernel_size)


def load_image_file(
    file_path: PathLike,
    color_space: ColorSpace = ColorSpace.RGB,
) -> ImageArray:
    """Load an image file from disk.

    Parameters
    ----------
    file_path : PathLike
        Path to image file.
    color_space : ColorSpace
        Target color space.

    Returns
    -------
    ImageArray
        Loaded image array.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If file format is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"Image file not found: {path}"
        raise FileNotFoundError(msg)

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        msg = f"Unsupported format: {path.suffix}"
        raise ValueError(msg)

    # Load using PIL if available, otherwise use numpy
    try:
        from PIL import Image
        img = Image.open(path)
        image = np.array(img)
    except ImportError:
        # Fallback: assume numpy .npy format
        image = np.load(path)

    # Convert color space if needed
    if color_space == ColorSpace.GRAYSCALE and image.ndim == 3:
        weights = np.array([0.299, 0.587, 0.114])
        image = np.dot(image[..., :3], weights).astype(image.dtype)

    return image


def save_image_file(
    image: ImageArray,
    file_path: PathLike,
    quality: int = 95,
) -> Path:
    """Save an image to disk.

    Parameters
    ----------
    image : ImageArray
        Image array to save.
    file_path : PathLike
        Output file path.
    quality : int
        JPEG quality (1-100).

    Returns
    -------
    Path
        Path to saved file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        if image.ndim == 2:
            img = Image.fromarray(image.astype(np.uint8), mode="L")
        else:
            img = Image.fromarray(image.astype(np.uint8), mode="RGB")

        if path.suffix.lower() in {".jpg", ".jpeg"}:
            img.save(path, quality=quality)
        else:
            img.save(path)
    except ImportError:
        np.save(path.with_suffix(".npy"), image)
        path = path.with_suffix(".npy")

    logger.info("Saved image to %s", path)
    return path

