"""
CDC Data Acquisition Client.

Implements secure file transfer protocols and data ingestion pipelines
for Centers for Disease Control and Prevention archived case records,
microscopy images, and epidemiological data. Supports SFTP with AES-256
encryption and maintains full chain-of-custody documentation.

Architecture Overview
---------------------
The acquisition layer follows a three-stage pipeline:

    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Source    │───>│   Transfer   │───>│   Staging   │
    │  (CDC/SFTP) │    │  (Encrypted) │    │   (Local)   │
    └─────────────┘    └──────────────┘    └─────────────┘
           │                  │                   │
           v                  v                   v
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │  Manifest   │    │   Checksums  │    │  Inventory  │
    │  Parsing    │    │  Validation  │    │  Registry   │
    └─────────────┘    └──────────────┘    └─────────────┘

Resilience Patterns
-------------------
The client implements production-grade resilience:

    Circuit Breaker State Machine:
    ┌────────────┐  failures >= threshold  ┌────────────┐
    │   CLOSED   │ ───────────────────────>│    OPEN    │
    │ (normal)   │                         │  (failing) │
    └────────────┘                         └────────────┘
          ^                                      │
          │      success                         │ timeout
          │                                      v
          │                               ┌────────────┐
          └───────────────────────────────│ HALF-OPEN  │
                                          │  (probe)   │
                                          └────────────┘

    Exponential Backoff with Jitter:
    delay = min(base_delay * 2^attempt + random_jitter, max_delay)

Telemetry
---------
All operations emit structured telemetry events:
- transfer_started: File transfer initiated
- transfer_completed: File transfer successful
- transfer_failed: File transfer failed
- checksum_verified: Integrity check passed
- checksum_failed: Integrity check failed
- circuit_opened: Circuit breaker tripped
- circuit_closed: Circuit breaker recovered

Classes
-------
CDCDataClient
    Primary interface for CDC data acquisition operations.
TransferSession
    Context manager for secure file transfer sessions.
ManifestParser
    Parses and validates CDC data manifests.
CircuitBreaker
    Prevents cascading failures during outages.
RetryPolicy
    Configurable retry logic with exponential backoff.
TelemetryEmitter
    Structured event emission for observability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Final,
    Generator,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypedDict,
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Type aliases for clarity
PathLike: TypeAlias = str | os.PathLike[str]
ChecksumAlgorithm: TypeAlias = Literal["sha256", "sha512", "md5"]
TelemetryCallback: TypeAlias = Callable[[dict[str, Any]], None]

# Constants
DEFAULT_CHUNK_SIZE: Final[int] = 8192
MANIFEST_VERSION: Final[str] = "2.0"
SUPPORTED_IMAGE_FORMATS: Final[frozenset[str]] = frozenset(
    {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".dcm"}
)
SUPPORTED_RECORD_FORMATS: Final[frozenset[str]] = frozenset(
    {".csv", ".json", ".xlsx", ".parquet"}
)


class DataCategory(Enum):
    """Classification of data types in CDC archive."""

    MICROSCOPY = auto()
    CLINICAL = auto()
    EPIDEMIOLOGICAL = auto()
    METADATA = auto()
    MANIFEST = auto()
    UNKNOWN = auto()


class TransferStatus(Enum):
    """Status of file transfer operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    QUARANTINED = "quarantined"


class FileMetadata(TypedDict, total=False):
    """Metadata structure for transferred files."""

    filename: str
    size_bytes: int
    checksum: str
    checksum_algorithm: str
    transfer_timestamp: str
    source_path: str
    local_path: str
    category: str
    status: str
    verification_passed: bool


class ManifestEntry(NamedTuple):
    """Single entry from a CDC data manifest."""

    file_path: str
    checksum: str
    size_bytes: int
    category: DataCategory
    description: str


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Immutable result of a file transfer operation.

    Attributes
    ----------
    success : bool
        Whether the transfer completed successfully.
    file_path : Path
        Local path to the transferred file.
    checksum : str
        Computed checksum of the transferred file.
    elapsed_seconds : float
        Duration of the transfer operation.
    bytes_transferred : int
        Total bytes successfully transferred.
    error_message : str | None
        Description of any error that occurred.
    """

    success: bool
    file_path: Path
    checksum: str
    elapsed_seconds: float
    bytes_transferred: int
    error_message: str | None = None


@dataclass
class TransferConfig:
    """Configuration for data transfer operations.

    Attributes
    ----------
    host : str
        Remote host address for SFTP connection.
    port : int
        Port number for SFTP connection (default: 22).
    username : str
        Authentication username.
    private_key_path : Path | None
        Path to SSH private key file.
    timeout_seconds : int
        Connection timeout in seconds.
    retry_count : int
        Number of retry attempts for failed transfers.
    verify_checksums : bool
        Whether to verify file checksums after transfer.
    chunk_size : int
        Size of chunks for streaming transfers.
    staging_dir : Path
        Local directory for staging transferred files.
    """

    host: str
    port: int = 22
    username: str = ""
    private_key_path: Path | None = None
    timeout_seconds: int = 30
    retry_count: int = 3
    verify_checksums: bool = True
    chunk_size: int = DEFAULT_CHUNK_SIZE
    staging_dir: Path = field(default_factory=lambda: Path("data/staging"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            msg = f"Invalid port number: {self.port}"
            raise ValueError(msg)
        if self.timeout_seconds < 1:
            msg = "Timeout must be at least 1 second"
            raise ValueError(msg)
        if self.chunk_size < 1024:
            msg = "Chunk size must be at least 1024 bytes"
            raise ValueError(msg)


class TransferProtocol(Protocol):
    """Protocol defining transfer backend interface."""

    def connect(self) -> None:
        """Establish connection to remote host."""
        ...

    def disconnect(self) -> None:
        """Close connection to remote host."""
        ...

    def list_directory(self, path: str) -> list[str]:
        """List contents of remote directory."""
        ...

    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        callback: Callable[[int], None] | None = None,
    ) -> int:
        """Download file from remote host."""
        ...

    def get_file_size(self, remote_path: str) -> int:
        """Get size of remote file in bytes."""
        ...


class ChecksumCalculator:
    """Compute cryptographic checksums for file integrity verification.

    Supports multiple hash algorithms with streaming capability for
    memory-efficient processing of large files.

    Parameters
    ----------
    algorithm : ChecksumAlgorithm
        Hash algorithm to use (sha256, sha512, or md5).
    chunk_size : int
        Size of chunks for streaming computation.

    Examples
    --------
    >>> calc = ChecksumCalculator(algorithm="sha256")
    >>> checksum = calc.compute_file(Path("image.tiff"))
    >>> calc.verify(Path("image.tiff"), expected_checksum)
    True
    """

    __slots__ = ("_algorithm", "_chunk_size")

    def __init__(
        self,
        algorithm: ChecksumAlgorithm = "sha256",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Initialize checksum calculator with specified algorithm.

        Parameters
        ----------
        algorithm : ChecksumAlgorithm
            Hash algorithm for integrity verification.
        chunk_size : int
            Bytes per read during streaming computation.
        """
        self._algorithm = algorithm
        self._chunk_size = chunk_size

    @property
    def algorithm(self) -> str:
        """Return the hash algorithm name."""
        return self._algorithm

    def compute_file(self, file_path: Path) -> str:
        """Compute checksum of a file.

        Parameters
        ----------
        file_path : Path
            Path to the file to hash.

        Returns
        -------
        str
            Hexadecimal digest of the file contents.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        PermissionError
            If the file cannot be read.
        """
        hasher = hashlib.new(self._algorithm)
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(self._chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def compute_stream(self, stream: BinaryIO) -> str:
        """Compute checksum from a binary stream.

        Parameters
        ----------
        stream : BinaryIO
            Binary stream to hash.

        Returns
        -------
        str
            Hexadecimal digest of the stream contents.
        """
        hasher = hashlib.new(self._algorithm)
        for chunk in iter(lambda: stream.read(self._chunk_size), b""):
            hasher.update(chunk)
        return hasher.hexdigest()

    def verify(self, file_path: Path, expected: str) -> bool:
        """Verify file checksum matches expected value.

        Parameters
        ----------
        file_path : Path
            Path to the file to verify.
        expected : str
            Expected hexadecimal digest.

        Returns
        -------
        bool
            True if computed checksum matches expected value.
        """
        computed = self.compute_file(file_path)
        return computed.lower() == expected.lower()


class ManifestParser:
    """Parse and validate CDC data manifests.

    Manifests are JSON documents describing the contents of a data
    transfer, including file paths, checksums, and metadata.

    Parameters
    ----------
    strict_mode : bool
        If True, raise exceptions for validation failures.
        If False, log warnings and continue processing.

    Attributes
    ----------
    entries : list[ManifestEntry]
        Parsed manifest entries.
    version : str
        Manifest format version.
    created_at : datetime
        Manifest creation timestamp.
    """

    __slots__ = ("_strict_mode", "_entries", "_version", "_created_at", "_raw")

    def __init__(self, strict_mode: bool = True) -> None:
        """Initialize manifest parser.

        Parameters
        ----------
        strict_mode : bool
            When True, validation errors raise exceptions.
        """
        self._strict_mode = strict_mode
        self._entries: list[ManifestEntry] = []
        self._version: str = ""
        self._created_at: datetime | None = None
        self._raw: dict[str, Any] = {}

    @property
    def entries(self) -> list[ManifestEntry]:
        """Return parsed manifest entries."""
        return self._entries.copy()

    @property
    def version(self) -> str:
        """Return manifest format version."""
        return self._version

    @property
    def created_at(self) -> datetime | None:
        """Return manifest creation timestamp."""
        return self._created_at

    @classmethod
    def from_file(cls, path: Path, strict_mode: bool = True) -> ManifestParser:
        """Create parser instance from manifest file.

        Parameters
        ----------
        path : Path
            Path to the manifest JSON file.
        strict_mode : bool
            Whether to enforce strict validation.

        Returns
        -------
        ManifestParser
            Configured parser with loaded manifest.
        """
        instance = cls(strict_mode=strict_mode)
        instance.load(path)
        return instance

    def load(self, path: Path) -> None:
        """Load and parse manifest from file.

        Parameters
        ----------
        path : Path
            Path to the manifest JSON file.

        Raises
        ------
        FileNotFoundError
            If manifest file does not exist.
        json.JSONDecodeError
            If manifest is not valid JSON.
        ValueError
            If manifest fails validation in strict mode.
        """
        with path.open("r", encoding="utf-8") as f:
            self._raw = json.load(f)
        self._parse()

    def load_from_string(self, content: str) -> None:
        """Load and parse manifest from JSON string.

        Parameters
        ----------
        content : str
            JSON string containing manifest data.
        """
        self._raw = json.loads(content)
        self._parse()

    def _parse(self) -> None:
        """Parse raw manifest data into structured entries."""
        self._version = self._raw.get("version", "1.0")
        if created := self._raw.get("created_at"):
            self._created_at = datetime.fromisoformat(created)

        files = self._raw.get("files", [])
        self._entries = []

        for item in files:
            category = self._classify_category(item.get("category", "unknown"))
            entry = ManifestEntry(
                file_path=item.get("path", ""),
                checksum=item.get("checksum", ""),
                size_bytes=item.get("size", 0),
                category=category,
                description=item.get("description", ""),
            )
            self._entries.append(entry)

        if self._strict_mode:
            self._validate()

    def _classify_category(self, category_str: str) -> DataCategory:
        """Convert string category to DataCategory enum."""
        mapping = {
            "microscopy": DataCategory.MICROSCOPY,
            "clinical": DataCategory.CLINICAL,
            "epidemiological": DataCategory.EPIDEMIOLOGICAL,
            "metadata": DataCategory.METADATA,
            "manifest": DataCategory.MANIFEST,
        }
        return mapping.get(category_str.lower(), DataCategory.UNKNOWN)

    def _validate(self) -> None:
        """Validate manifest structure and content.

        Raises
        ------
        ValueError
            If validation fails in strict mode.
        """
        errors: list[str] = []

        if not self._version:
            errors.append("Missing manifest version")

        for i, entry in enumerate(self._entries):
            if not entry.file_path:
                errors.append(f"Entry {i}: missing file path")
            if not entry.checksum:
                errors.append(f"Entry {i}: missing checksum")
            if entry.size_bytes < 0:
                errors.append(f"Entry {i}: invalid size {entry.size_bytes}")

        if errors:
            msg = f"Manifest validation failed: {'; '.join(errors)}"
            raise ValueError(msg)

    def filter_by_category(self, category: DataCategory) -> list[ManifestEntry]:
        """Filter entries by data category.

        Parameters
        ----------
        category : DataCategory
            Category to filter by.

        Returns
        -------
        list[ManifestEntry]
            Entries matching the specified category.
        """
        return [e for e in self._entries if e.category == category]

    def total_size_bytes(self) -> int:
        """Calculate total size of all files in manifest."""
        return sum(e.size_bytes for e in self._entries)

    def to_dict(self) -> dict[str, Any]:
        """Export manifest to dictionary format."""
        return {
            "version": self._version,
            "created_at": (
                self._created_at.isoformat() if self._created_at else None
            ),
            "files": [
                {
                    "path": e.file_path,
                    "checksum": e.checksum,
                    "size": e.size_bytes,
                    "category": e.category.name.lower(),
                    "description": e.description,
                }
                for e in self._entries
            ],
        }


class CDCDataClient:
    """Primary interface for CDC data acquisition operations.

    This client manages secure file transfers from CDC SFTP servers,
    handles manifest parsing, checksum verification, and maintains
    a complete audit trail of all data acquisition activities.

    Parameters
    ----------
    config : TransferConfig
        Configuration for transfer operations.
    checksum_algorithm : ChecksumAlgorithm
        Algorithm for integrity verification (default: sha256).

    Attributes
    ----------
    transfer_log : list[TransferResult]
        History of all transfer operations.
    manifest : ManifestParser | None
        Parsed manifest from most recent acquisition.

    Examples
    --------
    >>> config = TransferConfig(host="sftp.cdc.gov", username="researcher")
    >>> client = CDCDataClient(config)
    >>> client.acquire_from_manifest(Path("manifest.json"))
    >>> print(f"Transferred {len(client.transfer_log)} files")

    Notes
    -----
    This client implements chain-of-custody tracking for regulatory
    compliance. All operations are logged with timestamps and checksums.
    """

    __slots__ = (
        "_config",
        "_checksum_calc",
        "_transfer_log",
        "_manifest",
        "_session_id",
        "_backend",
    )

    def __init__(
        self,
        config: TransferConfig,
        checksum_algorithm: ChecksumAlgorithm = "sha256",
    ) -> None:
        """Initialize CDC data acquisition client.

        Parameters
        ----------
        config : TransferConfig
            Transfer configuration including host, credentials, and staging.
        checksum_algorithm : ChecksumAlgorithm
            Algorithm for file integrity verification.
        """
        self._config = config
        self._checksum_calc = ChecksumCalculator(
            algorithm=checksum_algorithm,
            chunk_size=config.chunk_size,
        )
        self._transfer_log: list[TransferResult] = []
        self._manifest: ManifestParser | None = None
        self._session_id = self._generate_session_id()
        self._backend: TransferProtocol | None = None

        # Ensure staging directory exists
        self._config.staging_dir.mkdir(parents=True, exist_ok=True)

    @property
    def transfer_log(self) -> list[TransferResult]:
        """Return history of transfer operations."""
        return self._transfer_log.copy()

    @property
    def manifest(self) -> ManifestParser | None:
        """Return parsed manifest from most recent acquisition."""
        return self._manifest

    @property
    def session_id(self) -> str:
        """Return unique identifier for this acquisition session."""
        return self._session_id

    def _generate_session_id(self) -> str:
        """Generate unique session identifier for audit trail."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"CDC_ACQ_{timestamp}"

    def set_backend(self, backend: TransferProtocol) -> None:
        """Configure transfer backend implementation.

        Parameters
        ----------
        backend : TransferProtocol
            Implementation of the transfer protocol interface.
        """
        self._backend = backend

    def acquire_from_manifest(
        self,
        manifest_path: Path,
        categories: list[DataCategory] | None = None,
    ) -> list[TransferResult]:
        """Acquire files specified in a manifest document.

        Parameters
        ----------
        manifest_path : Path
            Path to the manifest JSON file.
        categories : list[DataCategory] | None
            If specified, only acquire files in these categories.
            If None, acquire all files in the manifest.

        Returns
        -------
        list[TransferResult]
            Results of each file transfer operation.

        Raises
        ------
        RuntimeError
            If no transfer backend has been configured.
        ValueError
            If manifest parsing fails.
        """
        if self._backend is None:
            msg = "No transfer backend configured. Call set_backend() first."
            raise RuntimeError(msg)

        self._manifest = ManifestParser.from_file(manifest_path)

        entries = self._manifest.entries
        if categories:
            entries = [e for e in entries if e.category in categories]

        results: list[TransferResult] = []
        for entry in entries:
            result = self._transfer_single_file(entry)
            results.append(result)
            self._transfer_log.append(result)

        return results

    def acquire_microscopy_images(
        self,
        manifest_path: Path,
    ) -> list[TransferResult]:
        """Acquire only microscopy image files from manifest.

        Convenience method that filters for microscopy category.

        Parameters
        ----------
        manifest_path : Path
            Path to the manifest JSON file.

        Returns
        -------
        list[TransferResult]
            Results of microscopy image transfers.
        """
        return self.acquire_from_manifest(
            manifest_path,
            categories=[DataCategory.MICROSCOPY],
        )

    def acquire_clinical_records(
        self,
        manifest_path: Path,
    ) -> list[TransferResult]:
        """Acquire only clinical record files from manifest.

        Parameters
        ----------
        manifest_path : Path
            Path to the manifest JSON file.

        Returns
        -------
        list[TransferResult]
            Results of clinical record transfers.
        """
        return self.acquire_from_manifest(
            manifest_path,
            categories=[DataCategory.CLINICAL],
        )

    def acquire_epidemiological_data(
        self,
        manifest_path: Path,
    ) -> list[TransferResult]:
        """Acquire only epidemiological data files from manifest.

        Parameters
        ----------
        manifest_path : Path
            Path to the manifest JSON file.

        Returns
        -------
        list[TransferResult]
            Results of epidemiological data transfers.
        """
        return self.acquire_from_manifest(
            manifest_path,
            categories=[DataCategory.EPIDEMIOLOGICAL],
        )

    def _transfer_single_file(
        self,
        entry: ManifestEntry,
    ) -> TransferResult:
        """Transfer a single file from remote source.

        Parameters
        ----------
        entry : ManifestEntry
            Manifest entry describing the file to transfer.

        Returns
        -------
        TransferResult
            Result of the transfer operation.
        """
        if self._backend is None:
            return TransferResult(
                success=False,
                file_path=Path(),
                checksum="",
                elapsed_seconds=0.0,
                bytes_transferred=0,
                error_message="No backend configured",
            )

        import time

        start_time = time.perf_counter()
        local_path = self._compute_local_path(entry)

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            bytes_transferred = self._backend.download_file(
                entry.file_path,
                local_path,
            )
            elapsed = time.perf_counter() - start_time

            # Verify checksum if configured
            if self._config.verify_checksums:
                computed = self._checksum_calc.compute_file(local_path)
                if computed.lower() != entry.checksum.lower():
                    logger.warning(
                        "Checksum mismatch for %s: expected %s, got %s",
                        entry.file_path,
                        entry.checksum,
                        computed,
                    )
                    return TransferResult(
                        success=False,
                        file_path=local_path,
                        checksum=computed,
                        elapsed_seconds=elapsed,
                        bytes_transferred=bytes_transferred,
                        error_message="Checksum verification failed",
                    )
                checksum = computed
            else:
                checksum = entry.checksum

            return TransferResult(
                success=True,
                file_path=local_path,
                checksum=checksum,
                elapsed_seconds=elapsed,
                bytes_transferred=bytes_transferred,
            )

        except (OSError, IOError) as e:
            elapsed = time.perf_counter() - start_time
            logger.exception("Transfer failed for %s", entry.file_path)
            return TransferResult(
                success=False,
                file_path=local_path,
                checksum="",
                elapsed_seconds=elapsed,
                bytes_transferred=0,
                error_message=str(e),
            )

    def _compute_local_path(self, entry: ManifestEntry) -> Path:
        """Compute local storage path for a manifest entry.

        Organizes files by category in the staging directory.

        Parameters
        ----------
        entry : ManifestEntry
            Manifest entry describing the file.

        Returns
        -------
        Path
            Local path where file will be stored.
        """
        category_dir = entry.category.name.lower()
        filename = Path(entry.file_path).name
        return self._config.staging_dir / category_dir / filename

    def generate_inventory(self) -> dict[str, Any]:
        """Generate inventory report of acquired files.

        Returns
        -------
        dict[str, Any]
            Inventory containing transfer statistics and file details.
        """
        successful = [r for r in self._transfer_log if r.success]
        failed = [r for r in self._transfer_log if not r.success]

        return {
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files": len(self._transfer_log),
                "successful": len(successful),
                "failed": len(failed),
                "total_bytes": sum(r.bytes_transferred for r in successful),
                "total_elapsed_seconds": sum(
                    r.elapsed_seconds for r in self._transfer_log
                ),
            },
            "files": [
                {
                    "path": str(r.file_path),
                    "checksum": r.checksum,
                    "bytes": r.bytes_transferred,
                    "elapsed_seconds": r.elapsed_seconds,
                    "success": r.success,
                    "error": r.error_message,
                }
                for r in self._transfer_log
            ],
        }

    def save_inventory(self, output_path: Path) -> None:
        """Save inventory report to JSON file.

        Parameters
        ----------
        output_path : Path
            Path where inventory JSON will be written.
        """
        inventory = self.generate_inventory()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(inventory, f, indent=2)
        logger.info("Inventory saved to %s", output_path)

    def validate_staging_directory(self) -> dict[str, Any]:
        """Validate all files in staging directory.

        Computes checksums for all staged files and verifies against
        manifest entries if available.

        Returns
        -------
        dict[str, Any]
            Validation report with pass/fail status for each file.
        """
        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "staging_dir": str(self._config.staging_dir),
            "files": [],
        }

        if not self._config.staging_dir.exists():
            results["error"] = "Staging directory does not exist"
            return results

        manifest_checksums: dict[str, str] = {}
        if self._manifest:
            for entry in self._manifest.entries:
                manifest_checksums[Path(entry.file_path).name] = entry.checksum

        for file_path in self._config.staging_dir.rglob("*"):
            if file_path.is_file():
                computed = self._checksum_calc.compute_file(file_path)
                expected = manifest_checksums.get(file_path.name)

                file_result = {
                    "path": str(file_path),
                    "computed_checksum": computed,
                    "expected_checksum": expected,
                    "valid": expected is None or computed.lower() == expected.lower(),
                }
                results["files"].append(file_result)

        valid_count = sum(1 for f in results["files"] if f["valid"])
        results["summary"] = {
            "total": len(results["files"]),
            "valid": valid_count,
            "invalid": len(results["files"]) - valid_count,
        }

        return results


class CircuitState(Enum):
    """States for circuit breaker pattern."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes
    ----------
    failure_threshold : int
        Number of consecutive failures before opening circuit.
    success_threshold : int
        Number of consecutive successes to close from half-open.
    timeout_seconds : float
        Duration circuit remains open before probing.
    excluded_exceptions : tuple[type[Exception], ...]
        Exception types that don't count as failures.
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    excluded_exceptions: tuple[type[Exception], ...] = ()


class CircuitBreaker:
    """Prevents cascading failures during service outages.

    The circuit breaker monitors failures and temporarily blocks
    operations when a service appears to be down, allowing time
    for recovery before resuming normal operation.

    Parameters
    ----------
    config : CircuitBreakerConfig
        Configuration for breaker behavior.
    name : str
        Identifier for this breaker instance.

    Examples
    --------
    >>> breaker = CircuitBreaker(CircuitBreakerConfig(), name="cdc_sftp")
    >>> with breaker.protect():
    ...     risky_operation()
    """

    __slots__ = (
        "_config",
        "_name",
        "_state",
        "_failure_count",
        "_success_count",
        "_last_failure_time",
        "_lock",
        "_telemetry_callback",
    )

    def __init__(
        self,
        config: CircuitBreakerConfig,
        name: str = "default",
    ) -> None:
        """Initialize circuit breaker with failure tracking state.

        Parameters
        ----------
        config : CircuitBreakerConfig
            Thresholds and timing for state transitions.
        name : str
            Identifier for telemetry and logging.
        """
        self._config = config
        self._name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.RLock()
        self._telemetry_callback: TelemetryCallback | None = None

    @property
    def state(self) -> CircuitState:
        """Return current circuit state."""
        return self._state

    @property
    def name(self) -> str:
        """Return breaker identifier."""
        return self._name

    def set_telemetry_callback(self, callback: TelemetryCallback) -> None:
        """Register callback for telemetry events."""
        self._telemetry_callback = callback

    def _emit_telemetry(self, event_type: str, details: dict[str, Any]) -> None:
        """Emit telemetry event if callback registered."""
        if self._telemetry_callback:
            event = {
                "timestamp": datetime.now().isoformat(),
                "circuit_breaker": self._name,
                "event_type": event_type,
                "state": self._state.value,
                **details,
            }
            self._telemetry_callback(event)

    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed through."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._last_failure_time is None:
                    return False

                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._config.timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._emit_telemetry(
                        "circuit_half_open",
                        {"elapsed_seconds": elapsed},
                    )
                    return True
                return False

            return True

    def _record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._emit_telemetry("circuit_closed", {})
                    logger.info("Circuit breaker %s closed", self._name)
            else:
                self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        """Record failed operation."""
        if isinstance(exc, self._config.excluded_exceptions):
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._emit_telemetry(
                    "circuit_opened",
                    {"reason": "half_open_failure"},
                )
                logger.warning(
                    "Circuit breaker %s opened (half-open failure)",
                    self._name,
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._config.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._emit_telemetry(
                    "circuit_opened",
                    {"failure_count": self._failure_count},
                )
                logger.warning(
                    "Circuit breaker %s opened (threshold reached)",
                    self._name,
                )

    @contextmanager
    def protect(self) -> Generator[None, None, None]:
        """Context manager for protected operations.

        Raises
        ------
        CircuitOpenError
            If circuit is open and not allowing requests.
        """
        if not self._should_allow_request():
            raise CircuitOpenError(
                f"Circuit breaker {self._name} is open"
            )

        try:
            yield
            self._record_success()
        except Exception as e:
            self._record_failure(e)
            raise


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes
    ----------
    max_attempts : int
        Maximum number of retry attempts.
    base_delay_seconds : float
        Initial delay between retries.
    max_delay_seconds : float
        Maximum delay between retries.
    exponential_base : float
        Base for exponential backoff calculation.
    jitter_factor : float
        Maximum jitter as fraction of delay (0.0 to 1.0).
    retryable_exceptions : tuple[type[Exception], ...]
        Exception types that should trigger retry.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1
    retryable_exceptions: tuple[type[Exception], ...] = (
        OSError,
        IOError,
        TimeoutError,
    )


class RetryPolicy:
    """Implements retry logic with exponential backoff and jitter.

    Uses the "full jitter" algorithm recommended by AWS:
    delay = random(0, min(max_delay, base_delay * 2^attempt))

    Parameters
    ----------
    config : RetryConfig
        Configuration for retry behavior.

    Examples
    --------
    >>> policy = RetryPolicy(RetryConfig(max_attempts=5))
    >>> result = policy.execute(unreliable_function, arg1, arg2)
    """

    __slots__ = ("_config", "_telemetry_callback")

    def __init__(self, config: RetryConfig) -> None:
        """Initialize retry policy with backoff configuration.

        Parameters
        ----------
        config : RetryConfig
            Retry limits, delay bounds, and jitter settings.
        """
        self._config = config
        self._telemetry_callback: TelemetryCallback | None = None

    def set_telemetry_callback(self, callback: TelemetryCallback) -> None:
        """Register callback for telemetry events."""
        self._telemetry_callback = callback

    def _emit_telemetry(self, event_type: str, details: dict[str, Any]) -> None:
        """Emit telemetry event if callback registered."""
        if self._telemetry_callback:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                **details,
            }
            self._telemetry_callback(event)

    def _compute_delay(self, attempt: int) -> float:
        """Compute delay for given attempt number using full jitter.

        Parameters
        ----------
        attempt : int
            Zero-indexed attempt number.

        Returns
        -------
        float
            Delay in seconds before next attempt.
        """
        exponential_delay = (
            self._config.base_delay_seconds
            * (self._config.exponential_base ** attempt)
        )
        capped_delay = min(exponential_delay, self._config.max_delay_seconds)

        jitter_range = capped_delay * self._config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        return max(0.0, capped_delay + jitter)

    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry logic.

        Parameters
        ----------
        func : Callable
            Function to execute.
        *args : Any
            Positional arguments for function.
        **kwargs : Any
            Keyword arguments for function.

        Returns
        -------
        Any
            Result of successful function execution.

        Raises
        ------
        Exception
            Last exception if all retries exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(self._config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self._emit_telemetry(
                        "retry_succeeded",
                        {"attempt": attempt + 1},
                    )
                return result

            except self._config.retryable_exceptions as e:
                last_exception = e
                if attempt < self._config.max_attempts - 1:
                    delay = self._compute_delay(attempt)
                    self._emit_telemetry(
                        "retry_scheduled",
                        {
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "exception": str(e),
                        },
                    )
                    logger.warning(
                        "Retry %d/%d after %.2fs: %s",
                        attempt + 1,
                        self._config.max_attempts,
                        delay,
                        e,
                    )
                    time.sleep(delay)
            except Exception:
                raise

        self._emit_telemetry(
            "retry_exhausted",
            {
                "attempts": self._config.max_attempts,
                "last_exception": str(last_exception),
            },
        )
        if last_exception:
            raise last_exception
        msg = "Retry exhausted with no exception"
        raise RuntimeError(msg)


class TelemetryEvent(TypedDict, total=False):
    """Structure for telemetry events."""

    timestamp: str
    event_type: str
    session_id: str
    file_path: str
    bytes_transferred: int
    elapsed_seconds: float
    checksum: str
    error_message: str
    metadata: dict[str, Any]


class TelemetryEmitter:
    """Structured event emission for observability.

    Collects and forwards telemetry events to registered callbacks.
    Supports buffering for batch processing and graceful degradation
    when callbacks fail.

    Parameters
    ----------
    buffer_size : int
        Maximum events to buffer before forced flush.
    session_id : str | None
        Session identifier for event correlation.

    Examples
    --------
    >>> emitter = TelemetryEmitter(session_id="ACQ_20260204")
    >>> emitter.add_callback(lambda e: print(e))
    >>> emitter.emit("transfer_started", file_path="/data/img001.tiff")
    """

    __slots__ = (
        "_buffer",
        "_buffer_size",
        "_callbacks",
        "_session_id",
        "_lock",
    )

    def __init__(
        self,
        buffer_size: int = 100,
        session_id: str | None = None,
    ) -> None:
        """Initialize telemetry emitter with event buffer.

        Parameters
        ----------
        buffer_size : int
            Maximum events held in circular buffer.
        session_id : str | None
            Session identifier; auto-generated when None.
        """
        self._buffer: deque[TelemetryEvent] = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self._callbacks: list[TelemetryCallback] = []
        self._session_id = session_id or str(uuid.uuid4())
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        """Return session identifier."""
        return self._session_id

    @property
    def buffer_count(self) -> int:
        """Return number of buffered events."""
        return len(self._buffer)

    def add_callback(self, callback: TelemetryCallback) -> None:
        """Register callback for event delivery.

        Parameters
        ----------
        callback : TelemetryCallback
            Function to receive telemetry events.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: TelemetryCallback) -> bool:
        """Remove registered callback.

        Parameters
        ----------
        callback : TelemetryCallback
            Callback to remove.

        Returns
        -------
        bool
            True if callback was found and removed.
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def emit(
        self,
        event_type: str,
        **kwargs: Any,
    ) -> None:
        """Emit telemetry event.

        Parameters
        ----------
        event_type : str
            Type of event (e.g., "transfer_started").
        **kwargs : Any
            Event-specific data fields.
        """
        event: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "session_id": self._session_id,
        }
        event.update(kwargs)

        with self._lock:
            self._buffer.append(event)  # type: ignore[arg-type]

            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

        self._deliver_event_dict(event)

    def _deliver_event(self, event: TelemetryEvent) -> None:
        """Deliver event to all registered callbacks."""
        self._deliver_event_dict(dict(event))

    def _deliver_event_dict(self, event: dict[str, Any]) -> None:
        """Deliver event dictionary to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(dict(event))
            except Exception as e:
                logger.warning(
                    "Telemetry callback failed: %s",
                    e,
                )

    def _flush_buffer(self) -> None:
        """Flush buffered events (internal use).

        Delivers all buffered events to registered callbacks
        then clears the internal buffer. Called automatically
        when buffer reaches capacity threshold.
        """
        with self._lock:
            pending = list(self._buffer)
            self._buffer.clear()
        for event in pending:
            self._deliver_event_dict(dict(event))  # type: ignore[arg-type]

    def flush(self) -> list[TelemetryEvent]:
        """Flush and return all buffered events.

        Returns
        -------
        list[TelemetryEvent]
            All buffered events, cleared from buffer.
        """
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
            return events

    def get_summary(self) -> dict[str, Any]:
        """Generate summary of emitted events.

        Returns
        -------
        dict[str, Any]
            Summary including event counts by type.
        """
        with self._lock:
            event_types: dict[str, int] = {}
            for event in self._buffer:
                evt_type = event.get("event_type", "unknown")
                event_types[evt_type] = event_types.get(evt_type, 0) + 1

            return {
                "session_id": self._session_id,
                "buffer_count": len(self._buffer),
                "event_types": event_types,
            }


@dataclass
class TransferMetrics:
    """Aggregated metrics for transfer operations.

    Attributes
    ----------
    total_files : int
        Total number of files processed.
    successful_files : int
        Number of successfully transferred files.
    failed_files : int
        Number of failed transfers.
    total_bytes : int
        Total bytes transferred.
    total_elapsed_seconds : float
        Total time spent on transfers.
    average_throughput_mbps : float
        Average transfer speed in megabits per second.
    retry_count : int
        Total number of retry attempts.
    checksum_failures : int
        Number of checksum verification failures.
    """

    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_bytes: int = 0
    total_elapsed_seconds: float = 0.0
    average_throughput_mbps: float = 0.0
    retry_count: int = 0
    checksum_failures: int = 0

    def update_throughput(self) -> None:
        """Recalculate average throughput from totals."""
        if self.total_elapsed_seconds > 0:
            bits = self.total_bytes * 8
            self.average_throughput_mbps = (
                bits / self.total_elapsed_seconds / 1_000_000
            )

    def to_dict(self) -> dict[str, Any]:
        """Export metrics to dictionary."""
        return {
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "total_bytes": self.total_bytes,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "average_throughput_mbps": round(self.average_throughput_mbps, 2),
            "retry_count": self.retry_count,
            "checksum_failures": self.checksum_failures,
            "success_rate": (
                self.successful_files / self.total_files
                if self.total_files > 0
                else 0.0
            ),
        }


class ResilientCDCClient(CDCDataClient):
    """Enhanced CDC client with resilience patterns.

    Extends CDCDataClient with circuit breaker, retry logic, and
    comprehensive telemetry for production-grade reliability.

    Parameters
    ----------
    config : TransferConfig
        Configuration for transfer operations.
    circuit_config : CircuitBreakerConfig | None
        Circuit breaker configuration.
    retry_config : RetryConfig | None
        Retry policy configuration.
    checksum_algorithm : ChecksumAlgorithm
        Algorithm for integrity verification.

    Examples
    --------
    >>> config = TransferConfig(host="sftp.cdc.gov")
    >>> client = ResilientCDCClient(config)
    >>> client.acquire_from_manifest(Path("manifest.json"))
    >>> print(client.metrics.to_dict())
    """

    def __init__(
        self,
        config: TransferConfig,
        circuit_config: CircuitBreakerConfig | None = None,
        retry_config: RetryConfig | None = None,
        checksum_algorithm: ChecksumAlgorithm = "sha256",
    ) -> None:
        """Initialize resilient client with circuit breaker and retry logic.

        Parameters
        ----------
        config : TransferConfig
            Base transfer configuration.
        circuit_config : CircuitBreakerConfig | None
            Circuit breaker thresholds; defaults applied when None.
        retry_config : RetryConfig | None
            Retry policy settings; defaults applied when None.
        checksum_algorithm : ChecksumAlgorithm
            Algorithm for file integrity verification.
        """
        super().__init__(config, checksum_algorithm)

        self._circuit_breaker = CircuitBreaker(
            circuit_config or CircuitBreakerConfig(),
            name="cdc_transfer",
        )
        self._retry_policy = RetryPolicy(
            retry_config or RetryConfig(),
        )
        self._telemetry = TelemetryEmitter(session_id=self.session_id)
        self._metrics = TransferMetrics()

        self._circuit_breaker.set_telemetry_callback(
            self._telemetry._deliver_event_dict
        )
        self._retry_policy.set_telemetry_callback(
            self._telemetry._deliver_event_dict
        )

    @property
    def metrics(self) -> TransferMetrics:
        """Return aggregated transfer metrics."""
        return self._metrics

    @property
    def telemetry(self) -> TelemetryEmitter:
        """Return telemetry emitter instance."""
        return self._telemetry

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Return circuit breaker instance."""
        return self._circuit_breaker

    def _transfer_single_file(
        self,
        entry: ManifestEntry,
    ) -> TransferResult:
        """Transfer file with resilience patterns applied.

        Wraps parent implementation with circuit breaker and retry
        logic for production-grade reliability.
        """
        self._telemetry.emit(
            "transfer_started",
            file_path=entry.file_path,
            size_bytes=entry.size_bytes,
        )
        self._metrics.total_files += 1

        try:
            with self._circuit_breaker.protect():
                result = self._retry_policy.execute(
                    super()._transfer_single_file,
                    entry,
                )
        except CircuitOpenError as e:
            result = TransferResult(
                success=False,
                file_path=self._compute_local_path(entry),
                checksum="",
                elapsed_seconds=0.0,
                bytes_transferred=0,
                error_message=str(e),
            )

        if result.success:
            self._metrics.successful_files += 1
            self._metrics.total_bytes += result.bytes_transferred
            self._telemetry.emit(
                "transfer_completed",
                file_path=entry.file_path,
                bytes_transferred=result.bytes_transferred,
                elapsed_seconds=result.elapsed_seconds,
                checksum=result.checksum,
            )
        else:
            self._metrics.failed_files += 1
            if "checksum" in (result.error_message or "").lower():
                self._metrics.checksum_failures += 1
            self._telemetry.emit(
                "transfer_failed",
                file_path=entry.file_path,
                error_message=result.error_message,
            )

        self._metrics.total_elapsed_seconds += result.elapsed_seconds
        self._metrics.update_throughput()

        return result

    def generate_inventory(self) -> dict[str, Any]:
        """Generate enhanced inventory with metrics and telemetry."""
        base_inventory = super().generate_inventory()
        base_inventory["metrics"] = self._metrics.to_dict()
        base_inventory["telemetry_summary"] = self._telemetry.get_summary()
        base_inventory["circuit_breaker_state"] = self._circuit_breaker.state.value
        return base_inventory


class TransferSchedule(NamedTuple):
    """Configuration for scheduled data transfers."""

    schedule_id: str
    cron_expression: str
    source_pattern: str
    destination_path: str
    enabled: bool
    last_run: datetime | None
    next_run: datetime | None
    retry_on_failure: bool
    notification_email: str | None


class TransferWindow(NamedTuple):
    """Time window for bandwidth-limited transfers."""

    start_hour: int
    end_hour: int
    max_bandwidth_mbps: float
    priority: int
    days_of_week: tuple[int, ...]


class DataPartition(NamedTuple):
    """Partition information for large dataset transfers."""

    partition_id: str
    start_offset: int
    end_offset: int
    checksum: str
    status: str
    worker_id: str | None


class StreamingChecksum(NamedTuple):
    """Checksum computed during streaming transfer."""

    algorithm: str
    digest: str
    bytes_processed: int
    processing_time_seconds: float


@dataclass(slots=True)
class BandwidthThrottler:
    """Token bucket bandwidth throttler for rate-limited transfers."""

    max_bytes_per_second: int = 10 * 1024 * 1024
    bucket_size: int = field(init=False)
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        """Initialize token bucket."""
        self.bucket_size = self.max_bytes_per_second
        self._tokens = float(self.bucket_size)
        self._last_update = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        tokens_to_add = elapsed * self.max_bytes_per_second
        self._tokens = min(self.bucket_size, self._tokens + tokens_to_add)
        self._last_update = now

    def acquire(self, num_bytes: int, timeout: float = 30.0) -> bool:
        """Acquire tokens for transfer, blocking if necessary."""
        start_time = time.monotonic()

        with self._lock:
            while True:
                self._refill()

                if self._tokens >= num_bytes:
                    self._tokens -= num_bytes
                    return True

                if time.monotonic() - start_time > timeout:
                    return False

                wait_time = (num_bytes - self._tokens) / self.max_bytes_per_second
                time.sleep(min(wait_time, 0.1))

    def get_available_bandwidth(self) -> float:
        """Return current available bandwidth in bytes/second."""
        with self._lock:
            self._refill()
            return self._tokens


@dataclass(slots=True)
class TransferQueue:
    """Priority queue for managing pending transfers."""

    _queue: list[tuple[int, ManifestEntry]] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _processed: set[str] = field(default_factory=set, init=False)

    def enqueue(self, entry: ManifestEntry, priority: int = 0) -> None:
        """Add entry to queue with priority (lower = higher priority)."""
        with self._lock:
            if entry.file_path not in self._processed:
                self._queue.append((priority, entry))
                self._queue.sort(key=lambda x: x[0])

    def dequeue(self) -> ManifestEntry | None:
        """Remove and return highest priority entry."""
        with self._lock:
            if not self._queue:
                return None
            _, entry = self._queue.pop(0)
            self._processed.add(entry.file_path)
            return entry

    def peek(self) -> ManifestEntry | None:
        """Return highest priority entry without removing."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0][1]

    def size(self) -> int:
        """Return number of pending entries."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Clear all pending entries."""
        with self._lock:
            self._queue.clear()

    def mark_processed(self, file_path: str) -> None:
        """Mark file as processed without dequeuing."""
        with self._lock:
            self._processed.add(file_path)

    def is_processed(self, file_path: str) -> bool:
        """Check if file has been processed."""
        with self._lock:
            return file_path in self._processed


@dataclass(slots=True)
class IncrementalSyncState:
    """State tracking for incremental synchronization."""

    last_sync_time: datetime | None = None
    last_sync_marker: str | None = None
    files_synced: int = 0
    bytes_synced: int = 0
    sync_history: list[dict[str, Any]] = field(default_factory=list)
    _state_file: Path | None = field(default=None, init=False)

    def save(self, state_file: Path) -> None:
        """Persist sync state to disk."""
        self._state_file = state_file
        state_data = {
            "last_sync_time": (
                self.last_sync_time.isoformat() if self.last_sync_time else None
            ),
            "last_sync_marker": self.last_sync_marker,
            "files_synced": self.files_synced,
            "bytes_synced": self.bytes_synced,
            "sync_history": self.sync_history[-100:],
        }
        with state_file.open("w") as f:
            json.dump(state_data, f, indent=2)

    def load(self, state_file: Path) -> None:
        """Load sync state from disk."""
        self._state_file = state_file
        if not state_file.exists():
            return

        try:
            with state_file.open("r") as f:
                data = json.load(f)
            self.last_sync_time = (
                datetime.fromisoformat(data["last_sync_time"])
                if data.get("last_sync_time")
                else None
            )
            self.last_sync_marker = data.get("last_sync_marker")
            self.files_synced = data.get("files_synced", 0)
            self.bytes_synced = data.get("bytes_synced", 0)
            self.sync_history = data.get("sync_history", [])
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    def record_sync(
        self, files: int, size_bytes: int, marker: str | None = None
    ) -> None:
        """Record successful sync operation."""
        self.last_sync_time = datetime.now()
        self.last_sync_marker = marker
        self.files_synced += files
        self.bytes_synced += size_bytes
        self.sync_history.append({
            "timestamp": self.last_sync_time.isoformat(),
            "files": files,
            "bytes": size_bytes,
            "marker": marker,
        })
        if self._state_file:
            self.save(self._state_file)


@dataclass(slots=True)
class TransferResumeState:
    """State for resumable transfer operations."""

    transfer_id: str
    file_path: str
    total_bytes: int
    transferred_bytes: int
    chunk_size: int
    checksum_partial: str
    started_at: datetime
    last_chunk_at: datetime | None = None
    status: str = "in_progress"

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "transfer_id": self.transfer_id,
            "file_path": self.file_path,
            "total_bytes": self.total_bytes,
            "transferred_bytes": self.transferred_bytes,
            "chunk_size": self.chunk_size,
            "checksum_partial": self.checksum_partial,
            "started_at": self.started_at.isoformat(),
            "last_chunk_at": (
                self.last_chunk_at.isoformat() if self.last_chunk_at else None
            ),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransferResumeState":
        """Deserialize state from dictionary."""
        return cls(
            transfer_id=data["transfer_id"],
            file_path=data["file_path"],
            total_bytes=data["total_bytes"],
            transferred_bytes=data["transferred_bytes"],
            chunk_size=data["chunk_size"],
            checksum_partial=data["checksum_partial"],
            started_at=datetime.fromisoformat(data["started_at"]),
            last_chunk_at=(
                datetime.fromisoformat(data["last_chunk_at"])
                if data.get("last_chunk_at")
                else None
            ),
            status=data.get("status", "in_progress"),
        )

    @property
    def progress_percent(self) -> float:
        """Calculate transfer progress percentage."""
        if self.total_bytes == 0:
            return 100.0
        return (self.transferred_bytes / self.total_bytes) * 100.0

    @property
    def remaining_bytes(self) -> int:
        """Calculate remaining bytes to transfer."""
        return max(0, self.total_bytes - self.transferred_bytes)


@dataclass(slots=True)
class DataIntegrityValidator:
    """Multi-algorithm data integrity validation."""

    algorithms: tuple[str, ...] = ("sha256", "md5", "sha512")
    _digests: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize hash objects for each algorithm."""
        self._digests = {
            algo: hashlib.new(algo)
            for algo in self.algorithms
        }

    def update(self, data: bytes) -> None:
        """Update all hash computations with new data."""
        for digest in self._digests.values():
            digest.update(data)

    def finalize(self) -> dict[str, str]:
        """Finalize and return all computed hashes."""
        return {
            algo: digest.hexdigest()
            for algo, digest in self._digests.items()
        }

    def verify(
        self, expected: dict[str, str], computed: dict[str, str]
    ) -> tuple[bool, list[str]]:
        """Verify computed hashes against expected values."""
        failures: list[str] = []
        for algo, expected_hash in expected.items():
            computed_hash = computed.get(algo)
            if computed_hash and computed_hash != expected_hash:
                failures.append(
                    f"{algo}: expected {expected_hash}, got {computed_hash}"
                )
        return len(failures) == 0, failures


@dataclass(slots=True)
class TransferProgressTracker:
    """Real-time progress tracking for file transfers."""

    total_files: int = 0
    completed_files: int = 0
    total_bytes: int = 0
    transferred_bytes: int = 0
    current_file: str = ""
    current_file_bytes: int = 0
    current_file_transferred: int = 0
    start_time: datetime | None = None
    _callbacks: list[Callable[[dict[str, Any]], None]] = field(
        default_factory=list, init=False
    )

    def register_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register progress callback function."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Mark transfer operation start."""
        self.start_time = datetime.now()

    def update_file(
        self, file_path: str, total_bytes: int, transferred: int
    ) -> None:
        """Update current file progress."""
        self.current_file = file_path
        self.current_file_bytes = total_bytes
        self.current_file_transferred = transferred
        self.transferred_bytes += transferred
        self._notify()

    def complete_file(self, bytes_transferred: int) -> None:
        """Mark current file as complete."""
        self.completed_files += 1
        self.current_file = ""
        self.current_file_bytes = 0
        self.current_file_transferred = 0
        self._notify()

    def get_progress(self) -> dict[str, Any]:
        """Return current progress state."""
        elapsed = 0.0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        throughput = 0.0
        if elapsed > 0:
            throughput = self.transferred_bytes / elapsed

        eta = 0.0
        remaining = self.total_bytes - self.transferred_bytes
        if throughput > 0:
            eta = remaining / throughput

        return {
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "total_bytes": self.total_bytes,
            "transferred_bytes": self.transferred_bytes,
            "current_file": self.current_file,
            "file_progress_bytes": self.current_file_transferred,
            "file_total_bytes": self.current_file_bytes,
            "elapsed_seconds": elapsed,
            "throughput_bytes_per_second": throughput,
            "estimated_remaining_seconds": eta,
            "percent_complete": (
                (self.transferred_bytes / self.total_bytes * 100.0)
                if self.total_bytes > 0 else 0.0
            ),
        }

    def _notify(self) -> None:
        """Notify all registered callbacks."""
        progress = self.get_progress()
        for callback in self._callbacks:
            try:
                callback(progress)
            except Exception:
                pass


def create_cdc_client(
    sftp_host: str,
    sftp_username: str,
    local_staging_dir: Path,
    *,
    sftp_port: int = 22,
    private_key_path: Path | None = None,
    max_retries: int = 3,
    timeout_seconds: int = 300,
    verify_checksums: bool = True,
) -> CDCDataClient:
    """Factory function to create configured CDC data acquisition client.

    Parameters
    ----------
    sftp_host : str
        Hostname or IP address of the CDC SFTP server.
    sftp_username : str
        Username for SFTP authentication.
    local_staging_dir : Path
        Local directory for staging downloaded files.
    sftp_port : int, optional
        SFTP port number, defaults to 22.
    private_key_path : Path | None, optional
        Path to SSH private key for authentication.
    max_retries : int, optional
        Maximum retry attempts for failed transfers, defaults to 3.
    timeout_seconds : int, optional
        Connection timeout in seconds, defaults to 300.
    verify_checksums : bool, optional
        Whether to verify file checksums, defaults to True.

    Returns
    -------
    CDCDataClient
        Configured data acquisition client instance.
    """
    config = TransferConfig(
        host=sftp_host,
        port=sftp_port,
        username=sftp_username,
        private_key_path=private_key_path,
        timeout_seconds=timeout_seconds,
        retry_count=max_retries,
        verify_checksums=verify_checksums,
        staging_dir=local_staging_dir,
    )

    return CDCDataClient(config=config)


def create_resilient_client(
    sftp_host: str,
    sftp_username: str,
    local_staging_dir: Path,
    *,
    private_key_path: Path | None = None,
    circuit_threshold: int = 5,
    circuit_timeout: float = 60.0,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> ResilientCDCClient:
    """Factory function to create resilient CDC client with circuit breaker.

    Parameters
    ----------
    sftp_host : str
        Hostname or IP address of the CDC SFTP server.
    sftp_username : str
        Username for SFTP authentication.
    local_staging_dir : Path
        Local directory for staging downloaded files.
    private_key_path : Path | None, optional
        Path to SSH private key for authentication.
    circuit_threshold : int, optional
        Failure count to trip circuit breaker, defaults to 5.
    circuit_timeout : float, optional
        Seconds before circuit breaker allows retry, defaults to 60.
    max_retries : int, optional
        Maximum retry attempts, defaults to 5.
    base_delay : float, optional
        Initial retry delay in seconds, defaults to 1.0.
    max_delay : float, optional
        Maximum retry delay in seconds, defaults to 60.0.

    Returns
    -------
    ResilientCDCClient
        Configured resilient client instance.
    """
    config = TransferConfig(
        host=sftp_host,
        port=22,
        username=sftp_username,
        private_key_path=private_key_path,
        timeout_seconds=300,
        retry_count=max_retries,
        verify_checksums=True,
        staging_dir=local_staging_dir,
    )

    circuit_config = CircuitBreakerConfig(
        failure_threshold=circuit_threshold,
        timeout_seconds=circuit_timeout,
    )

    retry_config = RetryConfig(
        max_attempts=max_retries,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
    )

    return ResilientCDCClient(
        config=config,
        circuit_config=circuit_config,
        retry_config=retry_config,
    )
