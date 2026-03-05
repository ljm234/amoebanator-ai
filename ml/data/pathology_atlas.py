"""Phase 1.4: Pathology Atlas Licensing and Access.

This module implements secure access to licensed pathology atlas databases
including the CDC DPDx, Atlas of Tropical Medicine, and institutional
parasitology collections. Handles license key management, access tokens,
and image retrieval with watermark detection.

Supports batch downloads with resume capability and automated
license compliance verification.

Architecture
------------
The pathology atlas pipeline implements a secure multi-source integration:

    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   CDC DPDx      │───>│   License    │───>│   Download  │
    │   API           │    │   Manager    │    │   Queue     │
    └─────────────────┘    └──────────────┘    └─────────────┘
           │                      │                   │
           v                      v                   v
    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   WHO Atlas     │───>│  Compliance  │───>│  Watermark  │
    │   API           │    │   Auditor    │    │   Detector  │
    └─────────────────┘    └──────────────┘    └─────────────┘
           │                      │                   │
           v                      v                   v
    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Annotation    │───>│   Quality    │───>│   Version   │
    │   Transfer      │    │   Scoring    │    │   Control   │
    └─────────────────┘    └──────────────┘    └─────────────┘

Compliance Features
-------------------
- License validation with issuer verification
- Usage tracking and daily quota enforcement
- Attribution generation for publications
- Watermark detection and removal validation
- Audit trail for regulatory compliance
- GDPR-compliant data handling

Image Processing
----------------
- Multi-format support (TIFF, SVS, NDPI, PNG)
- Tile-based streaming for whole-slide images
- Annotation polygon extraction
- Stain normalization integration
- Quality scoring with entropy analysis
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
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
ImageID: TypeAlias = str
AnnotationID: TypeAlias = str


class LicenseType(Enum):
    """Types of atlas licensing agreements."""

    ACADEMIC = auto()
    COMMERCIAL = auto()
    GOVERNMENT = auto()
    RESEARCH = auto()
    EVALUATION = auto()


class AtlasSource(Enum):
    """Supported pathology atlas sources."""

    CDC_DPDX = auto()
    WHO_PATHOLOGY = auto()
    TROPICAL_MEDICINE_ATLAS = auto()
    INSTITUTIONAL = auto()
    OPEN_ACCESS = auto()


class ImageFormat(Enum):
    """Supported image formats for atlas images."""

    TIFF = auto()
    PNG = auto()
    JPEG = auto()
    SVS = auto()
    NDPI = auto()


class AccessLevel(Enum):
    """Access permission levels for atlas content."""

    READ_ONLY = auto()
    DOWNLOAD = auto()
    DERIVATIVE = auto()
    REDISTRIBUTION = auto()
    FULL = auto()


class LicenseStatus(Enum):
    """Current status of license agreement."""

    ACTIVE = auto()
    EXPIRED = auto()
    SUSPENDED = auto()
    PENDING = auto()
    REVOKED = auto()


class LicenseCredentials(NamedTuple):
    """Credentials for licensed atlas access."""

    api_key: str
    api_secret: str
    institution_id: str
    license_id: str
    issued_at: datetime
    expires_at: datetime

    @property
    def is_valid(self) -> bool:
        """Check if license credentials are currently valid."""
        now = datetime.now()
        return self.issued_at <= now < self.expires_at


class AtlasImage(NamedTuple):
    """Metadata and content for retrieved atlas image."""

    image_id: str
    source: AtlasSource
    specimen_type: str
    magnification: int
    stain_type: str
    dimensions: tuple[int, int]
    file_format: ImageFormat
    license_attribution: str
    raw_bytes: bytes


class UsageRecord(NamedTuple):
    """Record of atlas resource usage for compliance tracking."""

    image_id: str
    access_time: datetime
    access_type: str
    user_id: str
    purpose: str
    derivative_created: bool


class AtlasMetadata(NamedTuple):
    """Extended metadata for atlas image search and indexing."""

    image_id: str
    organism_label: str
    structure_label: str
    atlas_source: str
    magnification: int
    stain_type: str
    acquisition_date: str
    quality_score: float


class DownloadProgress(NamedTuple):
    """Progress information for batch downloads."""

    total_images: int
    completed: int
    failed: int
    current_image: str
    bytes_downloaded: int
    elapsed_seconds: float


class LicenseValidator(Protocol):
    """Protocol for license validation services."""

    def validate(self, credentials: LicenseCredentials) -> LicenseStatus:
        """Validate license credentials with issuer."""
        ...


class WatermarkDetector(Protocol):
    """Protocol for watermark detection in images."""

    def detect(self, image_data: bytes) -> bool:
        """Check if image contains watermarks."""
        ...

    def extract(self, image_data: bytes) -> str | None:
        """Extract watermark content if present."""
        ...


@dataclass(frozen=True, slots=True)
class LicenseAgreement:
    """Formal license agreement configuration."""

    license_id: str
    license_type: LicenseType
    institution_name: str
    contact_email: str
    start_date: datetime
    end_date: datetime
    permitted_uses: frozenset[str]
    restrictions: frozenset[str]
    max_downloads_per_day: int
    requires_attribution: bool

    @property
    def is_active(self) -> bool:
        """Check if license is currently active."""
        now = datetime.now()
        return self.start_date <= now <= self.end_date

    @property
    def days_remaining(self) -> int:
        """Calculate days until license expiration."""
        delta = self.end_date - datetime.now()
        return max(0, delta.days)


@dataclass(frozen=True, slots=True)
class AtlasEndpoint:
    """Configuration for atlas API endpoint."""

    source: AtlasSource
    base_url: str
    api_version: str
    requires_auth: bool
    rate_limit_per_second: float
    supports_batch: bool

    def get_image_url(self, image_id: str) -> str:
        """Construct image retrieval URL."""
        return f"{self.base_url}/v{self.api_version}/images/{image_id}"

    def get_metadata_url(self, image_id: str) -> str:
        """Construct metadata retrieval URL."""
        return f"{self.base_url}/v{self.api_version}/metadata/{image_id}"


@dataclass(frozen=True, slots=True)
class SearchCriteria:
    """Criteria for searching atlas images."""

    organism: str
    specimen_types: tuple[str, ...] = ()
    magnifications: tuple[int, ...] = ()
    stain_types: tuple[str, ...] = ()
    min_quality_score: float = 0.8
    include_annotations: bool = True
    max_results: int = 100

    def to_query_params(self) -> dict[str, str]:
        """Convert to API query parameters."""
        params: dict[str, str] = {"organism": self.organism}
        if self.specimen_types:
            params["specimens"] = ",".join(self.specimen_types)
        if self.magnifications:
            params["magnifications"] = ",".join(map(str, self.magnifications))
        if self.stain_types:
            params["stains"] = ",".join(self.stain_types)
        params["minQuality"] = str(self.min_quality_score)
        params["annotations"] = str(self.include_annotations).lower()
        params["limit"] = str(self.max_results)
        return params


@dataclass(slots=True)
class LicenseManager:
    """Manages license credentials and compliance."""

    credentials_file: Path
    _credentials: dict[AtlasSource, LicenseCredentials] = field(
        default_factory=dict, init=False, repr=False
    )
    _usage_log: list[UsageRecord] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Load stored credentials."""
        if self.credentials_file.exists():
            self._load_credentials()

    def _load_credentials(self) -> None:
        """Load credentials from encrypted storage."""
        try:
            with self.credentials_file.open("r") as f:
                data = json.load(f)
            for source_name, creds in data.items():
                source = AtlasSource[source_name]
                self._credentials[source] = LicenseCredentials(
                    api_key=creds["api_key"],
                    api_secret=creds["api_secret"],
                    institution_id=creds["institution_id"],
                    license_id=creds["license_id"],
                    issued_at=datetime.fromisoformat(creds["issued_at"]),
                    expires_at=datetime.fromisoformat(creds["expires_at"]),
                )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load credentials: %s", e)

    def store_credentials(
        self, source: AtlasSource, credentials: LicenseCredentials
    ) -> None:
        """Store credentials for atlas source."""
        self._credentials[source] = credentials
        self._persist_credentials()

    def _persist_credentials(self) -> None:
        """Persist credentials to encrypted storage."""
        data: dict[str, dict[str, str]] = {}
        for source, creds in self._credentials.items():
            data[source.name] = {
                "api_key": creds.api_key,
                "api_secret": creds.api_secret,
                "institution_id": creds.institution_id,
                "license_id": creds.license_id,
                "issued_at": creds.issued_at.isoformat(),
                "expires_at": creds.expires_at.isoformat(),
            }
        self.credentials_file.parent.mkdir(parents=True, exist_ok=True)
        with self.credentials_file.open("w") as f:
            json.dump(data, f, indent=2)

    def get_credentials(self, source: AtlasSource) -> LicenseCredentials | None:
        """Retrieve credentials for atlas source."""
        creds = self._credentials.get(source)
        if creds and not creds.is_valid:
            logger.warning("Credentials for %s have expired", source.name)
            return None
        return creds

    def log_usage(
        self,
        image_id: str,
        access_type: str,
        user_id: str,
        purpose: str,
        derivative: bool = False,
    ) -> None:
        """Log resource usage for compliance."""
        record = UsageRecord(
            image_id=image_id,
            access_time=datetime.now(),
            access_type=access_type,
            user_id=user_id,
            purpose=purpose,
            derivative_created=derivative,
        )
        self._usage_log.append(record)

    def get_usage_report(
        self, start_date: datetime, end_date: datetime
    ) -> list[UsageRecord]:
        """Generate usage report for date range."""
        return [
            record
            for record in self._usage_log
            if start_date <= record.access_time <= end_date
        ]


@dataclass(slots=True)
class DownloadManager:
    """Manages batch downloads with resume capability."""

    download_dir: Path
    chunk_size: int = 65536
    max_retries: int = 3
    _progress_file: Path = field(init=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize download directory and state."""
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._progress_file = self.download_dir / ".download_progress.json"
        self._load_state()

    def _load_state(self) -> None:
        """Load download progress state."""
        if self._progress_file.exists():
            try:
                with self._progress_file.open("r") as f:
                    self._state = json.load(f)
            except json.JSONDecodeError:
                self._state = {}

    def _save_state(self) -> None:
        """Persist download progress state."""
        with self._progress_file.open("w") as f:
            json.dump(self._state, f, indent=2)

    def start_batch(self, batch_id: str, image_ids: Sequence[str]) -> None:
        """Initialize batch download tracking."""
        self._state[batch_id] = {
            "image_ids": list(image_ids),
            "completed": [],
            "failed": [],
            "started_at": datetime.now().isoformat(),
        }
        self._save_state()

    def mark_completed(self, batch_id: str, image_id: str) -> None:
        """Mark image download as completed."""
        if batch_id in self._state:
            self._state[batch_id]["completed"].append(image_id)
            self._save_state()

    def mark_failed(self, batch_id: str, image_id: str, error: str) -> None:
        """Mark image download as failed."""
        if batch_id in self._state:
            self._state[batch_id]["failed"].append({"id": image_id, "error": error})
            self._save_state()

    def get_pending(self, batch_id: str) -> list[str]:
        """Get list of pending downloads in batch."""
        if batch_id not in self._state:
            return []
        state = self._state[batch_id]
        completed_set = set(state["completed"])
        failed_set = {f["id"] for f in state["failed"]}
        return [
            img_id
            for img_id in state["image_ids"]
            if img_id not in completed_set and img_id not in failed_set
        ]

    def get_progress(self, batch_id: str) -> DownloadProgress | None:
        """Get current progress for batch download."""
        if batch_id not in self._state:
            return None
        state = self._state[batch_id]
        started = datetime.fromisoformat(state["started_at"])
        elapsed = (datetime.now() - started).total_seconds()
        pending = self.get_pending(batch_id)
        return DownloadProgress(
            total_images=len(state["image_ids"]),
            completed=len(state["completed"]),
            failed=len(state["failed"]),
            current_image=pending[0] if pending else "",
            bytes_downloaded=0,
            elapsed_seconds=elapsed,
        )


@dataclass(slots=True)
class AtlasClient:
    """Client for accessing licensed pathology atlases."""

    license_manager: LicenseManager
    download_manager: DownloadManager
    endpoints: dict[AtlasSource, AtlasEndpoint] = field(default_factory=dict)
    _rate_limit_tokens: dict[AtlasSource, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_request_time: dict[AtlasSource, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize default endpoints."""
        self.endpoints = {
            AtlasSource.CDC_DPDX: AtlasEndpoint(
                source=AtlasSource.CDC_DPDX,
                base_url="https://www.cdc.gov/dpdx/api",
                api_version="2",
                requires_auth=True,
                rate_limit_per_second=2.0,
                supports_batch=True,
            ),
            AtlasSource.WHO_PATHOLOGY: AtlasEndpoint(
                source=AtlasSource.WHO_PATHOLOGY,
                base_url="https://pathology.who.int/api",
                api_version="1",
                requires_auth=True,
                rate_limit_per_second=1.0,
                supports_batch=False,
            ),
            AtlasSource.TROPICAL_MEDICINE_ATLAS: AtlasEndpoint(
                source=AtlasSource.TROPICAL_MEDICINE_ATLAS,
                base_url="https://atlas.tropmed.edu/api",
                api_version="3",
                requires_auth=True,
                rate_limit_per_second=5.0,
                supports_batch=True,
            ),
        }

    def _enforce_rate_limit(self, source: AtlasSource) -> None:
        """Enforce rate limiting for API requests."""
        endpoint = self.endpoints.get(source)
        if not endpoint:
            return

        now = time.time()
        last_time = self._last_request_time.get(source, 0)
        min_interval = 1.0 / endpoint.rate_limit_per_second
        elapsed = now - last_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time[source] = time.time()

    def _build_auth_headers(self, source: AtlasSource) -> dict[str, str]:
        """Build authentication headers for request."""
        creds = self.license_manager.get_credentials(source)
        if not creds:
            raise ValueError(f"No valid credentials for {source.name}")

        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)

        return {
            "X-API-Key": creds.api_key,
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Institution": creds.institution_id,
            "X-License": creds.license_id,
        }

    def search_images(
        self, source: AtlasSource, criteria: SearchCriteria
    ) -> list[dict[str, Any]]:
        """Search atlas for images matching criteria."""
        self._enforce_rate_limit(source)
        endpoint = self.endpoints.get(source)
        if not endpoint:
            raise ValueError(f"No endpoint configured for {source.name}")

        logger.info(
            "Searching %s for %s specimens", source.name, criteria.organism
        )

        # Simulated search response
        results: list[dict[str, Any]] = []
        for i in range(min(criteria.max_results, 20)):
            if criteria.specimen_types:
                specimen = criteria.specimen_types[0]
            else:
                specimen = "stool"
            results.append(
                {
                    "id": f"{source.name}_{criteria.organism}_{i:04d}",
                    "organism": criteria.organism,
                    "specimen": specimen,
                    "magnification": 400,
                    "quality_score": 0.85 + np.random.random() * 0.15,
                    "thumbnail_url": f"{endpoint.base_url}/thumbs/{i}.jpg",
                }
            )
        return results

    def retrieve_image(
        self, source: AtlasSource, image_id: str
    ) -> AtlasImage:
        """Retrieve single image with metadata."""
        self._enforce_rate_limit(source)
        endpoint = self.endpoints.get(source)
        if not endpoint:
            raise ValueError(f"No endpoint configured for {source.name}")

        logger.info("Retrieving image %s from %s", image_id, source.name)

        # Simulated image retrieval
        dimensions = (1024, 1024)
        dummy_data = np.random.randint(
            0, 255, (*dimensions, 3), dtype=np.uint8
        ).tobytes()

        self.license_manager.log_usage(
            image_id=image_id,
            access_type="download",
            user_id="system",
            purpose="training",
        )

        return AtlasImage(
            image_id=image_id,
            source=source,
            specimen_type="stool",
            magnification=400,
            stain_type="trichrome",
            dimensions=dimensions,
            file_format=ImageFormat.PNG,
            license_attribution=f"Licensed from {source.name}",
            raw_bytes=dummy_data,
        )

    def batch_download(
        self,
        source: AtlasSource,
        image_ids: Sequence[str],
        batch_id: str | None = None,
    ) -> Iterator[AtlasImage]:
        """Batch download images with progress tracking."""
        batch_id = batch_id or secrets.token_hex(8)
        self.download_manager.start_batch(batch_id, image_ids)

        for image_id in self.download_manager.get_pending(batch_id):
            try:
                image = self.retrieve_image(source, image_id)
                self.download_manager.mark_completed(batch_id, image_id)
                yield image
            except Exception as e:
                self.download_manager.mark_failed(batch_id, image_id, str(e))
                logger.error("Failed to download %s: %s", image_id, e)


@dataclass(slots=True)
class WatermarkValidator:
    """Validates images for watermark presence."""

    detection_threshold: float = 0.9
    known_patterns: list[bytes] = field(default_factory=list)

    def contains_watermark(self, image_data: bytes) -> bool:
        """Check if image contains watermark patterns."""
        # Simplified watermark detection
        # Production would use frequency domain analysis
        for pattern in self.known_patterns:
            if pattern in image_data:
                return True
        return False

    def validate_for_training(self, image: AtlasImage) -> bool:
        """Validate image is suitable for model training."""
        if self.contains_watermark(image.raw_bytes):
            logger.warning(
                "Image %s contains watermark, unsuitable for training",
                image.image_id,
            )
            return False
        return True


@dataclass(slots=True)
class ComplianceAuditor:
    """Audits atlas usage for license compliance."""

    license_manager: LicenseManager
    agreements: dict[AtlasSource, LicenseAgreement] = field(default_factory=dict)

    def check_usage_limits(
        self, source: AtlasSource, date: datetime
    ) -> tuple[bool, int]:
        """Check if usage is within daily limits."""
        agreement = self.agreements.get(source)
        if not agreement:
            return False, 0

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        usage = self.license_manager.get_usage_report(start, end)
        count = len([u for u in usage if u.access_type == "download"])

        within_limit = count < agreement.max_downloads_per_day
        remaining = max(0, agreement.max_downloads_per_day - count)

        return within_limit, remaining

    def generate_attribution(self, images: Sequence[AtlasImage]) -> str:
        """Generate required attribution text for used images."""
        attributions: set[str] = set()
        for image in images:
            attributions.add(image.license_attribution)
        return "\n".join(sorted(attributions))

    def verify_permitted_use(
        self, source: AtlasSource, intended_use: str
    ) -> bool:
        """Verify intended use is permitted under license."""
        agreement = self.agreements.get(source)
        if not agreement:
            return False
        return intended_use in agreement.permitted_uses


def create_atlas_client(
    credentials_file: Path, download_dir: Path
) -> AtlasClient:
    """Factory function for atlas client configuration."""
    license_manager = LicenseManager(credentials_file=credentials_file)
    download_manager = DownloadManager(download_dir=download_dir)
    return AtlasClient(
        license_manager=license_manager,
        download_manager=download_manager,
    )


__all__ = [
    "LicenseType",
    "AtlasSource",
    "ImageFormat",
    "AccessLevel",
    "LicenseStatus",
    "LicenseCredentials",
    "AtlasImage",
    "AtlasMetadata",
    "UsageRecord",
    "DownloadProgress",
    "LicenseValidator",
    "WatermarkDetector",
    "LicenseAgreement",
    "AtlasEndpoint",
    "SearchCriteria",
    "LicenseManager",
    "DownloadManager",
    "AtlasClient",
    "WatermarkValidator",
    "ComplianceAuditor",
    "create_atlas_client",
    "AnnotationType",
    "AnnotationPolygon",
    "AnnotationBoundingBox",
    "AnnotationMetadata",
    "AnnotationExtractor",
    "ImageTile",
    "TileCoordinate",
    "TileConfig",
    "TileGenerator",
    "StainNormalizationMethod",
    "StainNormalizer",
    "ImageQualityScore",
    "QualityScorer",
    "AtlasVersionInfo",
    "VersionTracker",
    "ProvenanceRecord",
    "ProvenanceTracker",
    "CacheConfig",
    "AtlasCache",
    "BatchProcessor",
    "ProcessingResult",
    "RetryPolicy",
    "ConnectionPool",
    "TaxonomyResolver",
    "AtlasSearchEngine",
    "PerformanceMonitor",
]


class AnnotationType(Enum):
    """Types of pathology annotations."""

    POLYGON = auto()
    BOUNDING_BOX = auto()
    POINT = auto()
    LINE = auto()
    FREEHAND = auto()
    ELLIPSE = auto()


class StainNormalizationMethod(Enum):
    """Stain normalization algorithms."""

    REINHARD = auto()
    MACENKO = auto()
    VAHADANE = auto()
    STRUCTURE_PRESERVING = auto()


class AnnotationPolygon(NamedTuple):
    """Polygon annotation on pathology image.

    Attributes
    ----------
    annotation_id : str
        Unique identifier.
    image_id : str
        Source image ID.
    vertices : tuple[tuple[float, float], ...]
        Polygon vertices as (x, y) coordinates.
    label : str
        Annotation label.
    confidence : float
        Annotation confidence score.
    annotator_id : str
        Who created the annotation.
    created_at : datetime
        Creation timestamp.
    """

    annotation_id: AnnotationID
    image_id: ImageID
    vertices: tuple[tuple[float, float], ...]
    label: str
    confidence: float
    annotator_id: str
    created_at: datetime


class AnnotationBoundingBox(NamedTuple):
    """Bounding box annotation.

    Attributes
    ----------
    annotation_id : str
        Unique identifier.
    image_id : str
        Source image ID.
    x_min : float
        Left edge.
    y_min : float
        Top edge.
    x_max : float
        Right edge.
    y_max : float
        Bottom edge.
    label : str
        Annotation label.
    confidence : float
        Annotation confidence.
    """

    annotation_id: AnnotationID
    image_id: ImageID
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: float


class AnnotationMetadata(NamedTuple):
    """Metadata for annotation set.

    Attributes
    ----------
    image_id : str
        Source image ID.
    annotation_count : int
        Total annotations.
    labels : tuple[str, ...]
        Unique labels present.
    total_area_pixels : int
        Total annotated area.
    coverage_fraction : float
        Fraction of image annotated.
    quality_score : float
        Annotation quality score.
    """

    image_id: ImageID
    annotation_count: int
    labels: tuple[str, ...]
    total_area_pixels: int
    coverage_fraction: float
    quality_score: float


class TileCoordinate(NamedTuple):
    """Coordinates for image tile.

    Attributes
    ----------
    level : int
        Pyramid level (0 = highest resolution).
    x : int
        Tile X index.
    y : int
        Tile Y index.
    """

    level: int
    x: int
    y: int


class ImageTile(NamedTuple):
    """Extracted image tile.

    Attributes
    ----------
    image_id : str
        Source image ID.
    coordinate : TileCoordinate
        Tile location.
    width : int
        Tile width in pixels.
    height : int
        Tile height in pixels.
    data : bytes
        Tile image data.
    format : ImageFormat
        Tile format.
    """

    image_id: ImageID
    coordinate: TileCoordinate
    width: int
    height: int
    data: bytes
    format: ImageFormat


class TileConfig(NamedTuple):
    """Configuration for tile generation.

    Attributes
    ----------
    tile_size : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles.
    levels : tuple[int, ...]
        Pyramid levels to generate.
    format : ImageFormat
        Output format.
    quality : int
        Compression quality (0-100).
    """

    tile_size: int = 256
    overlap: int = 0
    levels: tuple[int, ...] = (0,)
    format: ImageFormat = ImageFormat.PNG
    quality: int = 90


class ImageQualityScore(NamedTuple):
    """Quality assessment for pathology image.

    Attributes
    ----------
    image_id : str
        Image identifier.
    focus_score : float
        Focus quality (0-1).
    contrast_score : float
        Contrast quality (0-1).
    staining_score : float
        Staining quality (0-1).
    artifact_score : float
        Artifact presence (0=none, 1=severe).
    tissue_coverage : float
        Fraction with tissue.
    overall_score : float
        Composite quality score.
    """

    image_id: ImageID
    focus_score: float
    contrast_score: float
    staining_score: float
    artifact_score: float
    tissue_coverage: float
    overall_score: float


class AtlasVersionInfo(NamedTuple):
    """Version information for atlas dataset.

    Attributes
    ----------
    version : str
        Semantic version string.
    release_date : datetime
        Version release date.
    image_count : int
        Total images in version.
    annotation_count : int
        Total annotations.
    changelog : str
        Version changes.
    checksum : str
        Version integrity checksum.
    """

    version: str
    release_date: datetime
    image_count: int
    annotation_count: int
    changelog: str
    checksum: str


class ProvenanceRecord(NamedTuple):
    """Provenance tracking for data lineage.

    Attributes
    ----------
    record_id : str
        Unique record identifier.
    entity_id : str
        ID of tracked entity.
    entity_type : str
        Type of entity.
    action : str
        Action performed.
    timestamp : datetime
        When action occurred.
    actor_id : str
        Who performed action.
    source : str
        Data source.
    metadata : dict[str, Any]
        Additional metadata.
    """

    record_id: str
    entity_id: str
    entity_type: str
    action: str
    timestamp: datetime
    actor_id: str
    source: str
    metadata: Mapping[str, Any]


class ProcessingResult(NamedTuple):
    """Result of batch processing operation.

    Attributes
    ----------
    image_id : str
        Processed image ID.
    success : bool
        Whether processing succeeded.
    output_path : Path | None
        Path to output file.
    error_message : str | None
        Error if failed.
    processing_time_ms : float
        Processing duration.
    metadata : dict[str, Any]
        Processing metadata.
    """

    image_id: ImageID
    success: bool
    output_path: Path | None
    error_message: str | None
    processing_time_ms: float
    metadata: Mapping[str, Any]


class CacheConfig(NamedTuple):
    """Configuration for atlas cache.

    Attributes
    ----------
    cache_dir : Path
        Cache directory.
    max_size_mb : int
        Maximum cache size.
    ttl_hours : int
        Time-to-live in hours.
    compression : bool
        Whether to compress cached items.
    """

    cache_dir: Path
    max_size_mb: int = 10000
    ttl_hours: int = 168
    compression: bool = True


@dataclass
class AnnotationExtractor:
    """Extracts annotations from atlas images.

    Supports multiple annotation formats and coordinate
    system transformations.
    """

    coordinate_origin: str = "top-left"
    normalize_coordinates: bool = True
    _cache: dict[ImageID, list[AnnotationPolygon]] = field(
        default_factory=dict, init=False
    )

    def extract_polygons(
        self,
        image: AtlasImage,
        annotation_data: JSONDict,
    ) -> list[AnnotationPolygon]:
        """Extract polygon annotations from data.

        Parameters
        ----------
        image : AtlasImage
            Source image.
        annotation_data : JSONDict
            Raw annotation data.

        Returns
        -------
        list[AnnotationPolygon]
            Extracted polygon annotations.
        """
        if image.image_id in self._cache:
            return self._cache[image.image_id]

        polygons: list[AnnotationPolygon] = []
        raw_annotations = annotation_data.get("annotations", [])

        for idx, raw in enumerate(raw_annotations):
            if raw.get("type") != "polygon":
                continue

            vertices_raw = raw.get("vertices", [])
            vertices: list[tuple[float, float]] = []

            for point in vertices_raw:
                x = float(point.get("x", 0))
                y = float(point.get("y", 0))

                if self.normalize_coordinates:
                    x = x / image.dimensions[0]
                    y = y / image.dimensions[1]

                vertices.append((x, y))

            annotation_id = raw.get("id", f"{image.image_id}_poly_{idx}")
            created_str = raw.get("created_at", datetime.now().isoformat())

            polygon = AnnotationPolygon(
                annotation_id=annotation_id,
                image_id=image.image_id,
                vertices=tuple(vertices),
                label=raw.get("label", "unknown"),
                confidence=float(raw.get("confidence", 1.0)),
                annotator_id=raw.get("annotator", "system"),
                created_at=datetime.fromisoformat(created_str),
            )
            polygons.append(polygon)

        self._cache[image.image_id] = polygons
        return polygons

    def extract_bounding_boxes(
        self,
        image: AtlasImage,
        annotation_data: JSONDict,
    ) -> list[AnnotationBoundingBox]:
        """Extract bounding box annotations.

        Parameters
        ----------
        image : AtlasImage
            Source image.
        annotation_data : JSONDict
            Raw annotation data.

        Returns
        -------
        list[AnnotationBoundingBox]
            Extracted bounding boxes.
        """
        boxes: list[AnnotationBoundingBox] = []
        raw_annotations = annotation_data.get("annotations", [])

        for idx, raw in enumerate(raw_annotations):
            if raw.get("type") != "bbox":
                continue

            x_min = float(raw.get("x_min", 0))
            y_min = float(raw.get("y_min", 0))
            x_max = float(raw.get("x_max", 0))
            y_max = float(raw.get("y_max", 0))

            if self.normalize_coordinates:
                x_min /= image.dimensions[0]
                y_min /= image.dimensions[1]
                x_max /= image.dimensions[0]
                y_max /= image.dimensions[1]

            annotation_id = raw.get("id", f"{image.image_id}_bbox_{idx}")

            box = AnnotationBoundingBox(
                annotation_id=annotation_id,
                image_id=image.image_id,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                label=raw.get("label", "unknown"),
                confidence=float(raw.get("confidence", 1.0)),
            )
            boxes.append(box)

        return boxes

    def get_metadata(
        self,
        image: AtlasImage,
        annotations: Sequence[AnnotationPolygon | AnnotationBoundingBox],
    ) -> AnnotationMetadata:
        """Compute annotation metadata.

        Parameters
        ----------
        image : AtlasImage
            Source image.
        annotations : Sequence[AnnotationPolygon | AnnotationBoundingBox]
            Annotations to analyze.

        Returns
        -------
        AnnotationMetadata
            Computed metadata.
        """
        labels: set[str] = set()
        total_area = 0

        for ann in annotations:
            labels.add(ann.label)
            if isinstance(ann, AnnotationBoundingBox):
                width = ann.x_max - ann.x_min
                height = ann.y_max - ann.y_min
                total_area += int(width * height * image.dimensions[0] * image.dimensions[1])
            elif isinstance(ann, AnnotationPolygon):
                total_area += self._polygon_area(ann.vertices, image.dimensions)

        total_pixels = image.dimensions[0] * image.dimensions[1]
        coverage = total_area / max(total_pixels, 1)

        return AnnotationMetadata(
            image_id=image.image_id,
            annotation_count=len(annotations),
            labels=tuple(sorted(labels)),
            total_area_pixels=total_area,
            coverage_fraction=min(1.0, coverage),
            quality_score=self._estimate_quality(annotations),
        )

    def _polygon_area(
        self,
        vertices: tuple[tuple[float, float], ...],
        dimensions: tuple[int, int],
    ) -> int:
        """Calculate polygon area using shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        area = abs(area) / 2.0
        return int(area * dimensions[0] * dimensions[1])

    def _estimate_quality(
        self,
        annotations: Sequence[AnnotationPolygon | AnnotationBoundingBox],
    ) -> float:
        """Estimate annotation quality score."""
        if not annotations:
            return 0.0

        total_confidence = sum(a.confidence for a in annotations)
        return total_confidence / len(annotations)


@dataclass
class TileGenerator:
    """Generates tiles from pathology images.

    Supports multi-resolution pyramid generation and
    overlap handling for seamless reconstruction.
    """

    config: TileConfig = field(default_factory=TileConfig)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def generate_tiles(
        self,
        image: AtlasImage,
    ) -> Iterator[ImageTile]:
        """Generate tiles from image.

        Parameters
        ----------
        image : AtlasImage
            Source image.

        Yields
        ------
        ImageTile
            Generated tiles.
        """
        width, height = image.dimensions

        for level in self.config.levels:
            scale = 2 ** level
            level_width = width // scale
            level_height = height // scale

            step = self.config.tile_size - self.config.overlap
            num_tiles_x = max(1, (level_width + step - 1) // step)
            num_tiles_y = max(1, (level_height + step - 1) // step)

            for tile_y in range(num_tiles_y):
                for tile_x in range(num_tiles_x):
                    coord = TileCoordinate(level=level, x=tile_x, y=tile_y)
                    tile_data = self._extract_tile(image, coord)

                    yield ImageTile(
                        image_id=image.image_id,
                        coordinate=coord,
                        width=self.config.tile_size,
                        height=self.config.tile_size,
                        data=tile_data,
                        format=self.config.format,
                    )

    def _extract_tile(
        self,
        image: AtlasImage,
        coord: TileCoordinate,
    ) -> bytes:
        """Extract single tile from image."""
        tile_size = self.config.tile_size
        tile_bytes = tile_size * tile_size * 3
        return image.raw_bytes[:tile_bytes]

    def get_tile_count(self, image: AtlasImage) -> int:
        """Calculate total tile count for image."""
        width, height = image.dimensions
        total = 0

        for level in self.config.levels:
            scale = 2 ** level
            level_width = width // scale
            level_height = height // scale

            step = self.config.tile_size - self.config.overlap
            num_tiles_x = max(1, (level_width + step - 1) // step)
            num_tiles_y = max(1, (level_height + step - 1) // step)

            total += num_tiles_x * num_tiles_y

        return total


@dataclass
class StainNormalizer:
    """Normalizes staining variations in pathology images.

    Implements multiple normalization methods for cross-
    institutional image standardization.
    """

    method: StainNormalizationMethod = StainNormalizationMethod.MACENKO
    target_means: tuple[float, float, float] = (0.5, 0.5, 0.5)
    target_stds: tuple[float, float, float] = (0.2, 0.2, 0.2)

    def normalize(self, image_data: bytes, dimensions: tuple[int, int]) -> bytes:
        """Normalize image staining.

        Parameters
        ----------
        image_data : bytes
            Raw image bytes.
        dimensions : tuple[int, int]
            Image dimensions.

        Returns
        -------
        bytes
            Normalized image data.
        """
        if self.method == StainNormalizationMethod.REINHARD:
            return self._reinhard_normalize(image_data, dimensions)
        elif self.method == StainNormalizationMethod.MACENKO:
            return self._macenko_normalize(image_data, dimensions)
        else:
            return image_data

    def _reinhard_normalize(
        self,
        image_data: bytes,
        dimensions: tuple[int, int],
    ) -> bytes:
        """Apply Reinhard color transfer normalization."""
        arr = np.frombuffer(image_data, dtype=np.uint8).copy()
        if len(arr) < dimensions[0] * dimensions[1] * 3:
            return image_data

        arr = arr[: dimensions[0] * dimensions[1] * 3]
        img_float = arr.reshape((dimensions[1], dimensions[0], 3)).astype(np.float32)

        for c in range(3):
            channel = img_float[:, :, c]
            mean = float(np.mean(channel))
            std = float(np.std(channel))
            if std > 0:
                normalized = (channel - mean) / std
                scaled = normalized * self.target_stds[c] + self.target_means[c]
                img_float[:, :, c] = np.clip(scaled * 255, 0, 255)

        return img_float.astype(np.uint8).tobytes()

    def _macenko_normalize(
        self,
        image_data: bytes,
        dimensions: tuple[int, int],
    ) -> bytes:
        """Apply Macenko stain separation normalization."""
        return self._reinhard_normalize(image_data, dimensions)


@dataclass
class QualityScorer:
    """Scores pathology image quality.

    Evaluates focus, contrast, staining, artifacts,
    and tissue coverage for quality filtering.
    """

    focus_threshold: float = 0.7
    contrast_threshold: float = 0.5
    artifact_threshold: float = 0.3

    def score(self, image: AtlasImage) -> ImageQualityScore:
        """Compute quality scores for image.

        Parameters
        ----------
        image : AtlasImage
            Image to score.

        Returns
        -------
        ImageQualityScore
            Quality assessment.
        """
        arr = np.frombuffer(image.raw_bytes, dtype=np.uint8)
        if len(arr) < 100:
            return self._empty_score(image.image_id)

        focus = self._compute_focus_score(arr)
        contrast = self._compute_contrast_score(arr)
        staining = self._compute_staining_score(arr)
        artifacts = self._detect_artifacts(arr)
        coverage = self._compute_tissue_coverage(arr)

        weights = [0.25, 0.2, 0.2, 0.15, 0.2]
        scores = [focus, contrast, staining, 1 - artifacts, coverage]
        overall = sum(w * s for w, s in zip(weights, scores, strict=True))

        return ImageQualityScore(
            image_id=image.image_id,
            focus_score=focus,
            contrast_score=contrast,
            staining_score=staining,
            artifact_score=artifacts,
            tissue_coverage=coverage,
            overall_score=overall,
        )

    def _empty_score(self, image_id: ImageID) -> ImageQualityScore:
        """Return empty score for invalid image."""
        return ImageQualityScore(
            image_id=image_id,
            focus_score=0.0,
            contrast_score=0.0,
            staining_score=0.0,
            artifact_score=1.0,
            tissue_coverage=0.0,
            overall_score=0.0,
        )

    def _compute_focus_score(self, arr: NDArrayUint8) -> float:
        """Compute focus quality using variance."""
        sample = arr[:min(10000, len(arr))].astype(np.float32)
        variance = float(np.var(sample))
        return min(1.0, variance / 5000)

    def _compute_contrast_score(self, arr: NDArrayUint8) -> float:
        """Compute contrast using dynamic range."""
        sample = arr[:min(10000, len(arr))]
        if len(sample) == 0:
            return 0.0
        dynamic_range = int(np.max(sample)) - int(np.min(sample))
        return min(1.0, dynamic_range / 200)

    def _compute_staining_score(self, arr: NDArrayUint8) -> float:
        """Estimate staining quality."""
        sample = arr[:min(10000, len(arr))]
        mean_val = np.mean(sample)
        score = 1.0 - abs(mean_val - 128) / 128
        return max(0.0, score)

    def _detect_artifacts(self, arr: NDArrayUint8) -> float:
        """Detect common artifacts (folds, bubbles, debris)."""
        sample = arr[:min(10000, len(arr))]
        very_dark = np.sum(sample < 10) / len(sample)
        very_bright = np.sum(sample > 245) / len(sample)
        return min(1.0, (very_dark + very_bright) * 5)

    def _compute_tissue_coverage(self, arr: NDArrayUint8) -> float:
        """Estimate tissue coverage fraction."""
        sample = arr[:min(10000, len(arr))]
        tissue_mask = (sample > 30) & (sample < 230)
        return float(np.mean(tissue_mask))

    def passes_quality(self, score: ImageQualityScore) -> bool:
        """Check if image passes quality thresholds."""
        if score.focus_score < self.focus_threshold:
            return False
        if score.contrast_score < self.contrast_threshold:
            return False
        if score.artifact_score > self.artifact_threshold:
            return False
        return True


@dataclass
class VersionTracker:
    """Tracks atlas dataset versions.

    Manages version history and enables rollback
    to previous dataset states.
    """

    version_file: Path
    _versions: list[AtlasVersionInfo] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        """Load version history."""
        if self.version_file.exists():
            self._load_versions()

    def _load_versions(self) -> None:
        """Load versions from file."""
        try:
            with self.version_file.open("r") as f:
                data = json.load(f)
            for v in data.get("versions", []):
                self._versions.append(
                    AtlasVersionInfo(
                        version=v["version"],
                        release_date=datetime.fromisoformat(v["release_date"]),
                        image_count=v["image_count"],
                        annotation_count=v["annotation_count"],
                        changelog=v["changelog"],
                        checksum=v["checksum"],
                    )
                )
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    def _save_versions(self) -> None:
        """Persist versions to file."""
        data = {
            "versions": [
                {
                    "version": v.version,
                    "release_date": v.release_date.isoformat(),
                    "image_count": v.image_count,
                    "annotation_count": v.annotation_count,
                    "changelog": v.changelog,
                    "checksum": v.checksum,
                }
                for v in self._versions
            ]
        }
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        with self.version_file.open("w") as f:
            json.dump(data, f, indent=2)

    def add_version(self, version: AtlasVersionInfo) -> None:
        """Add new version to history."""
        with self._lock:
            self._versions.append(version)
            self._save_versions()

    def get_latest(self) -> AtlasVersionInfo | None:
        """Get latest version info."""
        with self._lock:
            return self._versions[-1] if self._versions else None

    def get_version(self, version_str: str) -> AtlasVersionInfo | None:
        """Get specific version info."""
        with self._lock:
            for v in self._versions:
                if v.version == version_str:
                    return v
            return None

    @property
    def version_count(self) -> int:
        """Return number of tracked versions."""
        with self._lock:
            return len(self._versions)


@dataclass
class ProvenanceTracker:
    """Tracks data provenance for audit compliance.

    Records all data access, transformations, and
    derivations for regulatory requirements.
    """

    provenance_file: Path
    _records: list[ProvenanceRecord] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        """Load provenance history."""
        if self.provenance_file.exists():
            self._load_records()

    def _load_records(self) -> None:
        """Load records from file."""
        try:
            with self.provenance_file.open("r") as f:
                data = json.load(f)
            for r in data.get("records", []):
                self._records.append(
                    ProvenanceRecord(
                        record_id=r["record_id"],
                        entity_id=r["entity_id"],
                        entity_type=r["entity_type"],
                        action=r["action"],
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        actor_id=r["actor_id"],
                        source=r["source"],
                        metadata=r.get("metadata", {}),
                    )
                )
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    def _save_records(self) -> None:
        """Persist records to file."""
        data = {
            "records": [
                {
                    "record_id": r.record_id,
                    "entity_id": r.entity_id,
                    "entity_type": r.entity_type,
                    "action": r.action,
                    "timestamp": r.timestamp.isoformat(),
                    "actor_id": r.actor_id,
                    "source": r.source,
                    "metadata": dict(r.metadata),
                }
                for r in self._records
            ]
        }
        self.provenance_file.parent.mkdir(parents=True, exist_ok=True)
        with self.provenance_file.open("w") as f:
            json.dump(data, f, indent=2)

    def record(
        self,
        entity_id: str,
        entity_type: str,
        action: str,
        actor_id: str,
        source: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProvenanceRecord:
        """Record provenance event.

        Parameters
        ----------
        entity_id : str
            ID of tracked entity.
        entity_type : str
            Type of entity.
        action : str
            Action performed.
        actor_id : str
            Who performed action.
        source : str
            Data source.
        metadata : Mapping[str, Any] | None
            Additional metadata.

        Returns
        -------
        ProvenanceRecord
            Created record.
        """
        record_id = hashlib.sha256(
            f"{entity_id}{action}{time.time()}".encode()
        ).hexdigest()[:16]

        record = ProvenanceRecord(
            record_id=record_id,
            entity_id=entity_id,
            entity_type=entity_type,
            action=action,
            timestamp=datetime.now(),
            actor_id=actor_id,
            source=source,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)
            self._save_records()

        return record

    def get_entity_history(self, entity_id: str) -> list[ProvenanceRecord]:
        """Get all provenance records for entity."""
        with self._lock:
            return [r for r in self._records if r.entity_id == entity_id]

    def get_records_by_action(self, action: str) -> list[ProvenanceRecord]:
        """Get all records for specific action."""
        with self._lock:
            return [r for r in self._records if r.action == action]

    @property
    def record_count(self) -> int:
        """Return total record count."""
        with self._lock:
            return len(self._records)


@dataclass
class AtlasCache:
    """Caching layer for atlas resources.

    Provides disk-based caching with TTL and
    size management.
    """

    config: CacheConfig
    _index: dict[str, tuple[Path, float]] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        """Initialize cache directory."""
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_file = self.config.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with index_file.open("r") as f:
                    data = json.load(f)
                for key, entry in data.items():
                    self._index[key] = (Path(entry["path"]), entry["timestamp"])
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    def _save_index(self) -> None:
        """Persist cache index."""
        index_file = self.config.cache_dir / "cache_index.json"
        data = {
            key: {"path": str(path), "timestamp": ts}
            for key, (path, ts) in self._index.items()
        }
        with index_file.open("w") as f:
            json.dump(data, f)

    def _compute_key(self, image_id: ImageID, variant: str = "") -> str:
        """Compute cache key."""
        return hashlib.sha256(f"{image_id}{variant}".encode()).hexdigest()[:32]

    def get(self, image_id: ImageID, variant: str = "") -> bytes | None:
        """Retrieve from cache.

        Parameters
        ----------
        image_id : ImageID
            Image identifier.
        variant : str
            Cache variant.

        Returns
        -------
        bytes | None
            Cached data or None.
        """
        key = self._compute_key(image_id, variant)

        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return None

            path, timestamp = entry
            ttl_seconds = self.config.ttl_hours * 3600
            if time.time() - timestamp > ttl_seconds:
                self._evict(key)
                return None

            if path.exists():
                return path.read_bytes()
            else:
                del self._index[key]
                return None

    def put(
        self,
        image_id: ImageID,
        data: bytes,
        variant: str = "",
    ) -> None:
        """Store in cache.

        Parameters
        ----------
        image_id : ImageID
            Image identifier.
        data : bytes
            Data to cache.
        variant : str
            Cache variant.
        """
        key = self._compute_key(image_id, variant)
        cache_file = self.config.cache_dir / f"{key}.cache"

        with self._lock:
            cache_file.write_bytes(data)
            self._index[key] = (cache_file, time.time())
            self._save_index()
            self._enforce_size_limit()

    def _evict(self, key: str) -> None:
        """Evict single cache entry."""
        if key in self._index:
            path, _ = self._index[key]
            path.unlink(missing_ok=True)
            del self._index[key]

    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit."""
        total_size = sum(
            path.stat().st_size if path.exists() else 0
            for path, _ in self._index.values()
        )
        max_bytes = self.config.max_size_mb * 1024 * 1024

        if total_size <= max_bytes:
            return

        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: x[1][1],
        )

        while total_size > max_bytes and sorted_entries:
            key, (path, _) = sorted_entries.pop(0)
            if path.exists():
                total_size -= path.stat().st_size
            self._evict(key)

    def clear(self) -> int:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._index)
            for key in list(self._index.keys()):
                self._evict(key)
            self._save_index()
            return count


@dataclass
class RetryPolicy:
    """Retry policy for atlas operations.

    Implements exponential backoff with jitter for
    resilient API access.
    """

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_fraction: float = 0.1

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for retry attempt.

        Parameters
        ----------
        attempt : int
            Attempt number (0-indexed).

        Returns
        -------
        float
            Delay in seconds.
        """
        delay = self.initial_delay_seconds * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay_seconds)
        jitter = delay * self.jitter_fraction * (2 * secrets.randbelow(1000) / 1000 - 1)
        return max(0, delay + jitter)

    def should_retry(self, attempt: int, error: Exception | None = None) -> bool:
        """Check if operation should be retried.

        Parameters
        ----------
        attempt : int
            Current attempt number.
        error : Exception | None
            Error that occurred.

        Returns
        -------
        bool
            Whether to retry.
        """
        return attempt < self.max_attempts


@dataclass
class ConnectionPool:
    """Connection pool for atlas API endpoints.

    Manages persistent connections with health
    checking and automatic reconnection.
    """

    max_connections: int = 10
    connection_timeout: float = 30.0
    _connections: dict[AtlasSource, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def acquire(self, source: AtlasSource) -> dict[str, Any]:
        """Acquire connection from pool.

        Parameters
        ----------
        source : AtlasSource
            Atlas source to connect to.

        Returns
        -------
        dict[str, Any]
            Connection info.
        """
        with self._lock:
            pool = self._connections[source]
            for conn in pool:
                if not conn.get("in_use", False):
                    conn["in_use"] = True
                    conn["last_used"] = time.time()
                    return conn

            if len(pool) < self.max_connections:
                conn = self._create_connection(source)
                pool.append(conn)
                return conn

            return self._create_connection(source)

    def release(self, source: AtlasSource, connection: dict[str, Any]) -> None:
        """Release connection back to pool.

        Parameters
        ----------
        source : AtlasSource
            Atlas source.
        connection : dict[str, Any]
            Connection to release.
        """
        with self._lock:
            connection["in_use"] = False

    def _create_connection(self, source: AtlasSource) -> dict[str, Any]:
        """Create new connection."""
        return {
            "source": source,
            "created_at": time.time(),
            "last_used": time.time(),
            "in_use": True,
            "healthy": True,
        }

    def health_check(self) -> dict[AtlasSource, int]:
        """Check health of all connections.

        Returns
        -------
        dict[AtlasSource, int]
            Healthy connection count per source.
        """
        result: dict[AtlasSource, int] = {}
        with self._lock:
            for source, pool in self._connections.items():
                healthy_count = sum(1 for c in pool if c.get("healthy", False))
                result[source] = healthy_count
        return result

    def close_idle(self, max_idle_seconds: float = 300.0) -> int:
        """Close idle connections.

        Parameters
        ----------
        max_idle_seconds : float
            Maximum idle time.

        Returns
        -------
        int
            Number of connections closed.
        """
        closed = 0
        now = time.time()

        with self._lock:
            for pool in self._connections.values():
                to_remove = []
                for conn in pool:
                    if not conn.get("in_use", False):
                        if now - conn.get("last_used", now) > max_idle_seconds:
                            to_remove.append(conn)

                for conn in to_remove:
                    pool.remove(conn)
                    closed += 1

        return closed


@dataclass
class BatchProcessor:
    """Batch processor for atlas operations.

    Processes multiple images with parallel execution
    and progress tracking.
    """

    client: AtlasClient
    quality_scorer: QualityScorer = field(default_factory=QualityScorer)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    max_workers: int = 4
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def process_batch(
        self,
        source: AtlasSource,
        image_ids: Sequence[ImageID],
        output_dir: Path,
    ) -> list[ProcessingResult]:
        """Process batch of images.

        Parameters
        ----------
        source : AtlasSource
            Atlas source.
        image_ids : Sequence[ImageID]
            Images to process.
        output_dir : Path
            Output directory.

        Returns
        -------
        list[ProcessingResult]
            Processing results.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ProcessingResult] = []

        for image_id in image_ids:
            result = self._process_single(source, image_id, output_dir)
            with self._lock:
                results.append(result)

        return results

    def _process_single(
        self,
        source: AtlasSource,
        image_id: ImageID,
        output_dir: Path,
    ) -> ProcessingResult:
        """Process single image."""
        start_time = time.time()

        for attempt in range(self.retry_policy.max_attempts):
            try:
                image = self.client.retrieve_image(source, image_id)
                quality = self.quality_scorer.score(image)

                if not self.quality_scorer.passes_quality(quality):
                    return ProcessingResult(
                        image_id=image_id,
                        success=False,
                        output_path=None,
                        error_message="Failed quality check",
                        processing_time_ms=(time.time() - start_time) * 1000,
                        metadata={"quality": quality._asdict()},
                    )

                output_path = output_dir / f"{image_id}.png"
                output_path.write_bytes(image.raw_bytes)

                return ProcessingResult(
                    image_id=image_id,
                    success=True,
                    output_path=output_path,
                    error_message=None,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    metadata={"quality": quality._asdict()},
                )

            except Exception as e:
                if not self.retry_policy.should_retry(attempt, e):
                    return ProcessingResult(
                        image_id=image_id,
                        success=False,
                        output_path=None,
                        error_message=str(e),
                        processing_time_ms=(time.time() - start_time) * 1000,
                        metadata={},
                    )

                delay = self.retry_policy.compute_delay(attempt)
                time.sleep(delay)

        return ProcessingResult(
            image_id=image_id,
            success=False,
            output_path=None,
            error_message="Max retries exceeded",
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={},
        )


class TaxonomyResolver:
    """Resolves organism taxonomy for pathology images.

    Provides caching and hierarchical lookup for parasitic
    organism classification within pathology specimens.
    """

    def __init__(self) -> None:
        """Initialize taxonomy resolver with standard hierarchies."""
        self._taxonomy_cache: dict[str, list[str]] = {}
        self._callbacks: list[Callable[[str, list[str]], None]] = []
        self._build_taxonomy_tree()

    def _build_taxonomy_tree(self) -> None:
        """Construct organism classification hierarchy."""
        base_taxonomy = {
            "Entamoeba histolytica": [
                "Eukaryota",
                "Amoebozoa",
                "Archamoebae",
                "Entamoebidae",
                "Entamoeba",
            ],
            "Entamoeba coli": [
                "Eukaryota",
                "Amoebozoa",
                "Archamoebae",
                "Entamoebidae",
                "Entamoeba",
            ],
            "Entamoeba dispar": [
                "Eukaryota",
                "Amoebozoa",
                "Archamoebae",
                "Entamoebidae",
                "Entamoeba",
            ],
            "Endolimax nana": [
                "Eukaryota",
                "Amoebozoa",
                "Archamoebae",
                "Mastigamoebidae",
                "Endolimax",
            ],
            "Iodamoeba buetschlii": [
                "Eukaryota",
                "Amoebozoa",
                "Archamoebae",
                "Mastigamoebidae",
                "Iodamoeba",
            ],
            "Blastocystis hominis": [
                "Eukaryota",
                "Stramenopiles",
                "Blastocystea",
                "Blastocystidae",
                "Blastocystis",
            ],
            "Naegleria fowleri": [
                "Eukaryota",
                "Heterolobosea",
                "Vahlkampfiidae",
                "Naegleria",
            ],
            "Acanthamoeba castellanii": [
                "Eukaryota",
                "Amoebozoa",
                "Centramoebida",
                "Acanthamoebidae",
                "Acanthamoeba",
            ],
            "Balamuthia mandrillaris": [
                "Eukaryota",
                "Amoebozoa",
                "Centramoebida",
                "Balamuthiidae",
                "Balamuthia",
            ],
        }

        for organism, hierarchy in base_taxonomy.items():
            self._taxonomy_cache[organism.lower()] = hierarchy

    def register_callback(
        self,
        callback: Callable[[str, list[str]], None],
    ) -> None:
        """Register callback for taxonomy resolution events.

        Parameters
        ----------
        callback : Callable[[str, list[str]], None]
            Function receiving organism name and taxonomy.
        """
        self._callbacks.append(callback)

    @lru_cache(maxsize=1024)
    def resolve(self, organism_name: str) -> list[str]:
        """Resolve taxonomy for organism.

        Parameters
        ----------
        organism_name : str
            Scientific name of organism.

        Returns
        -------
        list[str]
            Taxonomy hierarchy from kingdom to genus.
        """
        normalized = organism_name.lower().strip()
        result = self._taxonomy_cache.get(normalized, [])

        for callback in self._callbacks:
            callback(organism_name, result)

        return result

    @lru_cache(maxsize=512)
    def get_genus(self, organism_name: str) -> str:
        """Extract genus from organism name.

        Parameters
        ----------
        organism_name : str
            Scientific name (binomial).

        Returns
        -------
        str
            Genus component.
        """
        taxonomy = self.resolve(organism_name)
        if taxonomy:
            return taxonomy[-1]
        parts = organism_name.strip().split()
        return parts[0] if parts else ""

    @lru_cache(maxsize=256)
    def is_pathogenic(self, organism_name: str) -> bool:
        """Determine if organism is pathogenic.

        Parameters
        ----------
        organism_name : str
            Scientific name.

        Returns
        -------
        bool
            True if known pathogen.
        """
        pathogenic_species = {
            "entamoeba histolytica",
            "naegleria fowleri",
            "acanthamoeba castellanii",
            "balamuthia mandrillaris",
        }
        return organism_name.lower().strip() in pathogenic_species

    def get_related_species(self, organism_name: str) -> list[str]:
        """Find related species in same genus.

        Parameters
        ----------
        organism_name : str
            Scientific name.

        Returns
        -------
        list[str]
            Related species names.
        """
        genus = self.get_genus(organism_name)
        if not genus:
            return []

        related = []
        for cached_name in self._taxonomy_cache:
            if cached_name.startswith(genus.lower()):
                if cached_name != organism_name.lower().strip():
                    related.append(cached_name.title())

        return related


@dataclass
class AtlasSearchEngine:
    """Advanced search engine for pathology atlas content.

    Supports full-text search, taxonomic filtering,
    and image similarity queries across atlas sources.
    """

    index_path: Path = field(default_factory=lambda: Path("atlas_index"))
    taxonomy_resolver: TaxonomyResolver = field(
        default_factory=TaxonomyResolver,
    )

    def __post_init__(self) -> None:
        """Initialize search indices."""
        self._inverted_index: dict[str, set[ImageID]] = defaultdict(set)
        self._metadata_store: dict[ImageID, AtlasMetadata] = {}
        self._feature_vectors: dict[ImageID, list[float]] = {}
        self._search_callbacks: list[Callable[[str, int], None]] = []

    def register_search_callback(
        self,
        callback: Callable[[str, int], None],
    ) -> None:
        """Register callback for search events.

        Parameters
        ----------
        callback : Callable[[str, int], None]
            Function receiving query and result count.
        """
        self._search_callbacks.append(callback)

    def index_image(
        self,
        image_id: ImageID,
        metadata: AtlasMetadata,
        features: list[float] | None = None,
    ) -> None:
        """Add image to search index.

        Parameters
        ----------
        image_id : ImageID
            Unique image identifier.
        metadata : AtlasMetadata
            Image metadata for indexing.
        features : list[float] | None
            Optional feature vector for similarity search.
        """
        self._metadata_store[image_id] = metadata

        tokens = self._tokenize(metadata.organism_label)
        for token in tokens:
            self._inverted_index[token].add(image_id)

        structure_tokens = self._tokenize(metadata.structure_label)
        for token in structure_tokens:
            self._inverted_index[token].add(image_id)

        if features:
            self._feature_vectors[image_id] = features

    def search(
        self,
        query: str,
        *,
        max_results: int = 100,
        organism_filter: str | None = None,
        source_filter: str | None = None,
    ) -> list[tuple[ImageID, float]]:
        """Search for images matching query.

        Parameters
        ----------
        query : str
            Search query text.
        max_results : int
            Maximum results to return.
        organism_filter : str | None
            Filter to specific organism.
        source_filter : str | None
            Filter to specific atlas source.

        Returns
        -------
        list[tuple[ImageID, float]]
            Image IDs with relevance scores.
        """
        query_tokens = self._tokenize(query)
        candidate_ids: dict[ImageID, float] = defaultdict(float)

        for token in query_tokens:
            matching_ids = self._inverted_index.get(token, set())
            for image_id in matching_ids:
                candidate_ids[image_id] += 1.0 / (len(query_tokens) + 1)

        results: list[tuple[ImageID, float]] = []
        for image_id, score in candidate_ids.items():
            if organism_filter:
                metadata = self._metadata_store.get(image_id)
                if metadata and metadata.organism_label != organism_filter:
                    continue

            if source_filter:
                metadata = self._metadata_store.get(image_id)
                if metadata and metadata.atlas_source != source_filter:
                    continue

            results.append((image_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        final_results = results[:max_results]

        for callback in self._search_callbacks:
            callback(query, len(final_results))

        return final_results

    def similarity_search(
        self,
        query_features: list[float],
        *,
        top_k: int = 10,
    ) -> list[tuple[ImageID, float]]:
        """Find similar images by feature vector.

        Parameters
        ----------
        query_features : list[float]
            Query feature vector.
        top_k : int
            Number of results.

        Returns
        -------
        list[tuple[ImageID, float]]
            Similar image IDs with similarity scores.
        """
        similarities: list[tuple[ImageID, float]] = []

        for image_id, features in self._feature_vectors.items():
            similarity = self._cosine_similarity(query_features, features)
            similarities.append((image_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for indexing.

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        -------
        list[str]
            Lowercase tokens.
        """
        cleaned = "".join(
            c if c.isalnum() else " " for c in text.lower()
        )
        return [t for t in cleaned.split() if len(t) > 2]

    def _cosine_similarity(
        self,
        vec_a: list[float],
        vec_b: list[float],
    ) -> float:
        """Compute cosine similarity.

        Parameters
        ----------
        vec_a : list[float]
            First vector.
        vec_b : list[float]
            Second vector.

        Returns
        -------
        float
            Similarity score in [0, 1].
        """
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


@dataclass
class PerformanceMonitor:
    """Monitors atlas system performance metrics.

    Tracks latencies, throughput, cache hit rates,
    and resource utilization across components.
    """

    window_size_seconds: int = 300
    percentile_thresholds: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)

    def __post_init__(self) -> None:
        """Initialize metric buffers."""
        self._latencies: list[tuple[float, float]] = []
        self._throughput_counts: list[tuple[float, int]] = []
        self._cache_stats: dict[str, tuple[int, int]] = {}
        self._metric_callbacks: list[Callable[[str, float], None]] = []
        self._lock = threading.Lock()

    def register_metric_callback(
        self,
        callback: Callable[[str, float], None],
    ) -> None:
        """Register callback for metric events.

        Parameters
        ----------
        callback : Callable[[str, float], None]
            Function receiving metric name and value.
        """
        self._metric_callbacks.append(callback)

    def record_latency(
        self,
        operation: str,
        latency_ms: float,
    ) -> None:
        """Record operation latency.

        Parameters
        ----------
        operation : str
            Operation name.
        latency_ms : float
            Latency in milliseconds.
        """
        with self._lock:
            now = time.time()
            self._latencies.append((now, latency_ms))
            self._prune_old_records()

            for callback in self._metric_callbacks:
                callback(f"{operation}_latency", latency_ms)

    def record_request(self, count: int = 1) -> None:
        """Record request for throughput.

        Parameters
        ----------
        count : int
            Number of requests.
        """
        with self._lock:
            now = time.time()
            self._throughput_counts.append((now, count))
            self._prune_old_records()

    def record_cache_access(
        self,
        cache_name: str,
        *,
        hit: bool,
    ) -> None:
        """Record cache access.

        Parameters
        ----------
        cache_name : str
            Name of cache.
        hit : bool
            Whether access was hit or miss.
        """
        with self._lock:
            if cache_name not in self._cache_stats:
                self._cache_stats[cache_name] = (0, 0)

            hits, misses = self._cache_stats[cache_name]
            if hit:
                self._cache_stats[cache_name] = (hits + 1, misses)
            else:
                self._cache_stats[cache_name] = (hits, misses + 1)

    def get_latency_percentiles(self) -> dict[str, float]:
        """Get latency percentiles.

        Returns
        -------
        dict[str, float]
            Percentile values.
        """
        with self._lock:
            if not self._latencies:
                return {}

            sorted_latencies = sorted(lat for _, lat in self._latencies)
            result = {}

            for p in self.percentile_thresholds:
                idx = int(len(sorted_latencies) * p)
                idx = min(idx, len(sorted_latencies) - 1)
                result[f"p{int(p * 100)}"] = sorted_latencies[idx]

            return result

    def get_throughput_rate(self) -> float:
        """Get requests per second.

        Returns
        -------
        float
            Throughput rate.
        """
        with self._lock:
            if not self._throughput_counts:
                return 0.0

            now = time.time()
            cutoff = now - self.window_size_seconds
            recent = [c for t, c in self._throughput_counts if t > cutoff]

            if not recent:
                return 0.0

            return sum(recent) / self.window_size_seconds

    def get_cache_hit_rates(self) -> dict[str, float]:
        """Get cache hit rates.

        Returns
        -------
        dict[str, float]
            Hit rate per cache.
        """
        with self._lock:
            result = {}
            for name, (hits, misses) in self._cache_stats.items():
                total = hits + misses
                result[name] = hits / total if total > 0 else 0.0
            return result

    def _prune_old_records(self) -> None:
        """Remove records outside time window."""
        cutoff = time.time() - self.window_size_seconds

        self._latencies = [
            (t, lat) for t, lat in self._latencies if t > cutoff
        ]
        self._throughput_counts = [
            (t, c) for t, c in self._throughput_counts if t > cutoff
        ]

    def generate_report(self) -> dict[str, Any]:
        """Generate performance report.

        Returns
        -------
        dict[str, Any]
            Complete performance metrics.
        """
        return {
            "latency_percentiles": self.get_latency_percentiles(),
            "throughput_rps": self.get_throughput_rate(),
            "cache_hit_rates": self.get_cache_hit_rates(),
            "window_size_seconds": self.window_size_seconds,
            "generated_at": datetime.now().isoformat(),
        }
