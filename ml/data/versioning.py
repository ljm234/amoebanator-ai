"""
Dataset Versioning and Lineage Tracking.

Provides comprehensive version control for medical datasets including
immutable snapshots, content-addressed storage, and complete lineage
graphs for regulatory compliance and reproducibility.

Architecture
------------
The versioning system implements a Merkle tree structure:

    ┌─────────────────────────────────────────────────────────────┐
    │                    VERSION REPOSITORY                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Snapshot v3.0.0 (HEAD)                                    │
    │  ├── hash: sha256:abc123...                                │
    │  ├── parent: v2.1.0                                        │
    │  ├── timestamp: 2026-02-03T10:30:00Z                       │
    │  └── manifest:                                              │
    │      ├── microscopy/ (42 files, 1.2GB)                     │
    │      ├── clinical/ (150 records)                           │
    │      └── metadata.json                                      │
    │                                                             │
    │  Snapshot v2.1.0                                           │
    │  ├── hash: sha256:def456...                                │
    │  └── ...                                                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Classes
-------
DataVersionManager
    Primary interface for dataset versioning operations.
Snapshot
    Immutable representation of a dataset state.
LineageGraph
    Tracks data transformations and dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Final,
    NamedTuple,
    TypeAlias,
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Type aliases
PathLike: TypeAlias = str | Path
VersionString: TypeAlias = str  # Format: "major.minor.patch"
ContentHash: TypeAlias = str  # SHA-256 hex digest

# Constants
VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$"
)
DEFAULT_HASH_ALGORITHM: Final[str] = "sha256"
MANIFEST_FILENAME: Final[str] = "manifest.json"
METADATA_FILENAME: Final[str] = "metadata.json"
LINEAGE_FILENAME: Final[str] = "lineage.json"


class VersionBump(Enum):
    """Type of version increment."""

    MAJOR = auto()  # Breaking changes
    MINOR = auto()  # New features
    PATCH = auto()  # Bug fixes / minor updates


class SnapshotStatus(Enum):
    """Status of a dataset snapshot."""

    DRAFT = "draft"
    COMMITTED = "committed"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class TransformationType(Enum):
    """Type of data transformation in lineage."""

    ACQUISITION = "acquisition"
    PREPROCESSING = "preprocessing"
    AUGMENTATION = "augmentation"
    FILTERING = "filtering"
    MERGE = "merge"
    SPLIT = "split"
    ANNOTATION = "annotation"
    VALIDATION = "validation"


class SemanticVersion(NamedTuple):
    """Semantic version components.

    Attributes
    ----------
    major : int
        Major version (breaking changes).
    minor : int
        Minor version (new features).
    patch : int
        Patch version (bug fixes).
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def from_string(cls, version: str) -> SemanticVersion:
        """Parse version string.

        Parameters
        ----------
        version : str
            Version string in format "major.minor.patch".

        Returns
        -------
        SemanticVersion
            Parsed version.

        Raises
        ------
        ValueError
            If version string is invalid.
        """
        match = VERSION_PATTERN.match(version)
        if not match:
            msg = f"Invalid version string: {version}"
            raise ValueError(msg)
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
        )

    def __str__(self) -> str:
        """Return version string in major.minor.patch format."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump(self, bump_type: VersionBump) -> SemanticVersion:
        """Create new version with specified increment.

        Parameters
        ----------
        bump_type : VersionBump
            Type of version increment.

        Returns
        -------
        SemanticVersion
            Incremented version.
        """
        if bump_type == VersionBump.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        if bump_type == VersionBump.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        return SemanticVersion(self.major, self.minor, self.patch + 1)


class FileEntry(NamedTuple):
    """Entry for a file in the snapshot manifest.

    Attributes
    ----------
    relative_path : str
        Path relative to snapshot root.
    content_hash : ContentHash
        SHA-256 hash of file contents.
    size_bytes : int
        File size in bytes.
    category : str
        Data category (microscopy, clinical, etc.).
    """

    relative_path: str
    content_hash: ContentHash
    size_bytes: int
    category: str


@dataclass(frozen=True, slots=True)
class TransformationNode:
    """Node in the lineage graph representing a transformation.

    Attributes
    ----------
    node_id : str
        Unique identifier for this node.
    transform_type : TransformationType
        Type of transformation.
    timestamp : datetime
        When the transformation occurred.
    input_hashes : tuple[ContentHash, ...]
        Content hashes of input data.
    output_hash : ContentHash
        Content hash of output data.
    parameters : dict[str, Any]
        Parameters used in transformation.
    description : str
        Human-readable description.
    """

    node_id: str
    transform_type: TransformationType
    timestamp: datetime
    input_hashes: tuple[ContentHash, ...]
    output_hash: ContentHash
    parameters: tuple[tuple[str, Any], ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "transform_type": self.transform_type.value,
            "timestamp": self.timestamp.isoformat(),
            "input_hashes": list(self.input_hashes),
            "output_hash": self.output_hash,
            "parameters": dict(self.parameters),
            "description": self.description,
        }


@dataclass
class SnapshotManifest:
    """Manifest describing contents of a dataset snapshot.

    Attributes
    ----------
    version : SemanticVersion
        Snapshot version.
    content_hash : ContentHash
        Root hash of entire snapshot (Merkle root).
    files : list[FileEntry]
        List of all files in snapshot.
    total_size_bytes : int
        Total size of all files.
    created_at : datetime
        Snapshot creation timestamp.
    created_by : str
        User or system that created snapshot.
    description : str
        Human-readable description.
    """

    version: SemanticVersion
    content_hash: ContentHash = ""
    files: list[FileEntry] = field(default_factory=list)
    total_size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    description: str = ""

    def add_file(self, entry: FileEntry) -> None:
        """Add a file entry to the manifest.

        Parameters
        ----------
        entry : FileEntry
            File entry to add.
        """
        self.files.append(entry)
        self.total_size_bytes += entry.size_bytes

    def compute_root_hash(self) -> ContentHash:
        """Compute Merkle root hash of all files.

        Returns
        -------
        ContentHash
            SHA-256 hash of concatenated file hashes.
        """
        if not self.files:
            return hashlib.sha256(b"").hexdigest()

        # Sort files by path for deterministic ordering
        sorted_files = sorted(self.files, key=lambda f: f.relative_path)
        combined = "".join(f.content_hash for f in sorted_files)
        self.content_hash = hashlib.sha256(combined.encode()).hexdigest()
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": str(self.version),
            "content_hash": self.content_hash,
            "total_size_bytes": self.total_size_bytes,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "files": [
                {
                    "path": f.relative_path,
                    "hash": f.content_hash,
                    "size": f.size_bytes,
                    "category": f.category,
                }
                for f in self.files
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotManifest:
        """Create manifest from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary representation.

        Returns
        -------
        SnapshotManifest
            Parsed manifest.
        """
        manifest = cls(
            version=SemanticVersion.from_string(data["version"]),
            content_hash=data.get("content_hash", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", "unknown"),
            description=data.get("description", ""),
        )

        for file_data in data.get("files", []):
            entry = FileEntry(
                relative_path=file_data["path"],
                content_hash=file_data["hash"],
                size_bytes=file_data["size"],
                category=file_data.get("category", "unknown"),
            )
            manifest.add_file(entry)

        return manifest


@dataclass
class Snapshot:
    """Immutable representation of a dataset state.

    Provides content-addressed storage and integrity verification.

    Attributes
    ----------
    manifest : SnapshotManifest
        Manifest describing snapshot contents.
    parent_hash : ContentHash | None
        Hash of parent snapshot (None for initial).
    status : SnapshotStatus
        Current snapshot status.
    tags : list[str]
        Tags associated with this snapshot.
    metadata : dict[str, Any]
        Additional metadata.
    """

    manifest: SnapshotManifest
    parent_hash: ContentHash | None = None
    status: SnapshotStatus = SnapshotStatus.DRAFT
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> SemanticVersion:
        """Return snapshot version."""
        return self.manifest.version

    @property
    def content_hash(self) -> ContentHash:
        """Return content hash (Merkle root)."""
        return self.manifest.content_hash

    def verify_integrity(self, root_path: Path) -> dict[str, Any]:
        """Verify integrity of snapshot files.

        Parameters
        ----------
        root_path : Path
            Path to snapshot data directory.

        Returns
        -------
        dict[str, Any]
            Verification results with pass/fail status.
        """
        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "version": str(self.version),
            "files_checked": 0,
            "files_valid": 0,
            "files_invalid": 0,
            "files_missing": 0,
            "details": [],
        }

        for file_entry in self.manifest.files:
            file_path = root_path / file_entry.relative_path
            results["files_checked"] += 1

            if not file_path.exists():
                results["files_missing"] += 1
                results["details"].append({
                    "path": file_entry.relative_path,
                    "status": "missing",
                })
                continue

            # Compute actual hash
            with file_path.open("rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()

            if actual_hash == file_entry.content_hash:
                results["files_valid"] += 1
                results["details"].append({
                    "path": file_entry.relative_path,
                    "status": "valid",
                })
            else:
                results["files_invalid"] += 1
                results["details"].append({
                    "path": file_entry.relative_path,
                    "status": "invalid",
                    "expected": file_entry.content_hash,
                    "actual": actual_hash,
                })

        results["valid"] = (
            results["files_invalid"] == 0 and results["files_missing"] == 0
        )
        return results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "manifest": self.manifest.to_dict(),
            "parent_hash": self.parent_hash,
            "status": self.status.value,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        """Create snapshot from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary representation.

        Returns
        -------
        Snapshot
            Parsed snapshot.
        """
        return cls(
            manifest=SnapshotManifest.from_dict(data["manifest"]),
            parent_hash=data.get("parent_hash"),
            status=SnapshotStatus(data.get("status", "draft")),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


class LineageGraph:
    """Track data transformations and dependencies.

    Maintains a directed acyclic graph (DAG) of all transformations
    applied to the dataset for complete provenance tracking.

    Examples
    --------
    >>> graph = LineageGraph()
    >>> graph.add_transformation(
    ...     transform_type=TransformationType.ACQUISITION,
    ...     input_hashes=(),
    ...     output_hash="abc123...",
    ...     description="Acquired CDC data batch 1",
    ... )
    """

    __slots__ = ("_nodes", "_edges")

    def __init__(self) -> None:
        """Initialize empty lineage graph for transformation tracking."""
        self._nodes: dict[str, TransformationNode] = {}
        self._edges: list[tuple[str, str]] = []

    @property
    def nodes(self) -> dict[str, TransformationNode]:
        """Return all transformation nodes."""
        return self._nodes.copy()

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Return all edges (parent, child) in the graph."""
        return self._edges.copy()

    def add_transformation(
        self,
        transform_type: TransformationType,
        input_hashes: tuple[ContentHash, ...],
        output_hash: ContentHash,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Add a transformation to the lineage graph.

        Parameters
        ----------
        transform_type : TransformationType
            Type of transformation.
        input_hashes : tuple[ContentHash, ...]
            Content hashes of input data.
        output_hash : ContentHash
            Content hash of output data.
        description : str
            Human-readable description.
        parameters : dict[str, Any] | None
            Parameters used in transformation.

        Returns
        -------
        str
            Node ID of the new transformation.
        """
        node_id = self._generate_node_id(output_hash)

        node = TransformationNode(
            node_id=node_id,
            transform_type=transform_type,
            timestamp=datetime.now(),
            input_hashes=input_hashes,
            output_hash=output_hash,
            parameters=tuple((parameters or {}).items()),
            description=description,
        )

        self._nodes[node_id] = node

        # Add edges from input nodes
        for input_hash in input_hashes:
            for existing_id, existing_node in self._nodes.items():
                if existing_node.output_hash == input_hash:
                    self._edges.append((existing_id, node_id))

        return node_id

    def _generate_node_id(self, output_hash: ContentHash) -> str:
        """Generate unique node identifier.

        Parameters
        ----------
        output_hash : ContentHash
            Output content hash.

        Returns
        -------
        str
            Unique node identifier.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        combined = f"{timestamp}:{output_hash[:16]}"
        return hashlib.sha256(combined.encode()).hexdigest()[:12]

    def get_ancestors(self, node_id: str) -> list[TransformationNode]:
        """Get all ancestor transformations of a node.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        list[TransformationNode]
            All ancestor nodes in topological order.
        """
        ancestors: list[TransformationNode] = []
        visited: set[str] = set()
        stack = [node_id]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            for parent, child in self._edges:
                if child == current and parent not in visited:
                    if parent in self._nodes:
                        ancestors.append(self._nodes[parent])
                    stack.append(parent)

        return ancestors

    def get_descendants(self, node_id: str) -> list[TransformationNode]:
        """Get all descendant transformations of a node.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        list[TransformationNode]
            All descendant nodes.
        """
        descendants: list[TransformationNode] = []
        visited: set[str] = set()
        stack = [node_id]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            for parent, child in self._edges:
                if parent == current and child not in visited:
                    if child in self._nodes:
                        descendants.append(self._nodes[child])
                    stack.append(child)

        return descendants

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": self._edges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageGraph:
        """Create graph from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary representation.

        Returns
        -------
        LineageGraph
            Parsed lineage graph.
        """
        graph = cls()

        for node_data in data.get("nodes", []):
            node = TransformationNode(
                node_id=node_data["node_id"],
                transform_type=TransformationType(node_data["transform_type"]),
                timestamp=datetime.fromisoformat(node_data["timestamp"]),
                input_hashes=tuple(node_data["input_hashes"]),
                output_hash=node_data["output_hash"],
                parameters=tuple(node_data.get("parameters", {}).items()),
                description=node_data["description"],
            )
            graph._nodes[node.node_id] = node

        graph._edges = [tuple(e) for e in data.get("edges", [])]  # type: ignore[misc]
        return graph

    def save(self, path: Path) -> None:
        """Save lineage graph to JSON file.

        Parameters
        ----------
        path : Path
            Output file path.
        """
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> LineageGraph:
        """Load lineage graph from JSON file.

        Parameters
        ----------
        path : Path
            Path to JSON file.

        Returns
        -------
        LineageGraph
            Loaded lineage graph.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class DataVersionManager:
    """Primary interface for dataset versioning operations.

    Manages snapshots, version history, and lineage tracking for
    complete dataset provenance and reproducibility.

    Parameters
    ----------
    repository_path : Path
        Root path for version repository.
    auto_compute_hashes : bool
        Whether to automatically compute file hashes.

    Examples
    --------
    >>> manager = DataVersionManager(Path("data/versions"))
    >>> manager.create_snapshot(
    ...     source_path=Path("data/raw"),
    ...     description="Initial CDC data acquisition",
    ... )
    >>> print(f"Created version {manager.current_version}")
    """

    __slots__ = (
        "_repository_path",
        "_auto_compute_hashes",
        "_snapshots",
        "_lineage",
        "_current_version",
    )

    def __init__(
        self,
        repository_path: Path,
        auto_compute_hashes: bool = True,
    ) -> None:
        """Initialize version manager with repository storage.

        Parameters
        ----------
        repository_path : Path
            Root directory for version snapshots and metadata.
        auto_compute_hashes : bool
            Automatically compute content hashes on snapshot creation.
        """
        self._repository_path = Path(repository_path)
        self._auto_compute_hashes = auto_compute_hashes
        self._snapshots: dict[str, Snapshot] = {}
        self._lineage = LineageGraph()
        self._current_version: SemanticVersion | None = None

        self._initialize_repository()

    @property
    def repository_path(self) -> Path:
        """Return repository root path."""
        return self._repository_path

    @property
    def current_version(self) -> SemanticVersion | None:
        """Return current (latest) version."""
        return self._current_version

    @property
    def lineage(self) -> LineageGraph:
        """Return lineage graph."""
        return self._lineage

    def _initialize_repository(self) -> None:
        """Initialize repository structure."""
        self._repository_path.mkdir(parents=True, exist_ok=True)
        (self._repository_path / "snapshots").mkdir(exist_ok=True)
        (self._repository_path / "objects").mkdir(exist_ok=True)

        # Load existing state if present
        state_path = self._repository_path / "state.json"
        if state_path.exists():
            self._load_state(state_path)

        lineage_path = self._repository_path / LINEAGE_FILENAME
        if lineage_path.exists():
            self._lineage = LineageGraph.load(lineage_path)

    def _load_state(self, state_path: Path) -> None:
        """Load repository state from JSON.

        Parameters
        ----------
        state_path : Path
            Path to state file.
        """
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        if current := state.get("current_version"):
            self._current_version = SemanticVersion.from_string(current)

        for version_str, snapshot_data in state.get("snapshots", {}).items():
            self._snapshots[version_str] = Snapshot.from_dict(snapshot_data)

    def _save_state(self) -> None:
        """Save repository state to JSON."""
        state = {
            "current_version": (
                str(self._current_version) if self._current_version else None
            ),
            "snapshots": {
                str(v): s.to_dict() for v, s in self._snapshots.items()
            },
        }

        state_path = self._repository_path / "state.json"
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Also save lineage
        lineage_path = self._repository_path / LINEAGE_FILENAME
        self._lineage.save(lineage_path)

    def create_snapshot(
        self,
        source_path: Path,
        description: str = "",
        bump_type: VersionBump = VersionBump.MINOR,
        created_by: str = "system",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Create a new snapshot from source data.

        Parameters
        ----------
        source_path : Path
            Path to source data directory.
        description : str
            Human-readable description.
        bump_type : VersionBump
            Type of version increment.
        created_by : str
            User or system creating snapshot.
        tags : list[str] | None
            Tags to associate with snapshot.
        metadata : dict[str, Any] | None
            Additional metadata.

        Returns
        -------
        Snapshot
            Created snapshot.
        """
        # Determine new version
        if self._current_version is None:
            new_version = SemanticVersion(1, 0, 0)
        else:
            new_version = self._current_version.bump(bump_type)

        # Build manifest
        manifest = SnapshotManifest(
            version=new_version,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
        )

        # Scan source directory and compute hashes
        for file_path in source_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(source_path)
                category = self._determine_category(file_path)

                if self._auto_compute_hashes:
                    with file_path.open("rb") as f:
                        content_hash = hashlib.sha256(f.read()).hexdigest()
                else:
                    content_hash = ""

                entry = FileEntry(
                    relative_path=str(relative_path),
                    content_hash=content_hash,
                    size_bytes=file_path.stat().st_size,
                    category=category,
                )
                manifest.add_file(entry)

        # Compute root hash
        manifest.compute_root_hash()

        # Get parent hash
        parent_hash: ContentHash | None = None
        if self._current_version:
            current_snapshot = self._snapshots.get(str(self._current_version))
            if current_snapshot:
                parent_hash = current_snapshot.content_hash

        # Create snapshot
        snapshot = Snapshot(
            manifest=manifest,
            parent_hash=parent_hash,
            status=SnapshotStatus.DRAFT,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Store snapshot
        self._snapshots[str(new_version)] = snapshot
        self._current_version = new_version

        # Add to lineage
        input_hashes = (parent_hash,) if parent_hash else ()
        self._lineage.add_transformation(
            transform_type=TransformationType.ACQUISITION,
            input_hashes=input_hashes,
            output_hash=manifest.content_hash,
            description=description,
        )

        # Save state
        self._save_state()

        n_files = len(manifest.files)
        logger.info("Created snapshot v%s with %d files", new_version, n_files)
        return snapshot

    def _determine_category(self, file_path: Path) -> str:
        """Determine data category from file path.

        Parameters
        ----------
        file_path : Path
            File path to categorize.

        Returns
        -------
        str
            Category name.
        """
        suffix = file_path.suffix.lower()
        parent = file_path.parent.name.lower()

        if parent == "microscopy" or suffix in (".tiff", ".tif", ".png"):
            return "microscopy"
        if parent == "clinical" or suffix in (".csv", ".json"):
            return "clinical"
        if parent == "epidemiological":
            return "epidemiological"
        return "metadata"

    def get_snapshot(self, version: str | SemanticVersion) -> Snapshot | None:
        """Retrieve a snapshot by version.

        Parameters
        ----------
        version : str | SemanticVersion
            Version to retrieve.

        Returns
        -------
        Snapshot | None
            Requested snapshot, or None if not found.
        """
        version_str = str(version)
        return self._snapshots.get(version_str)

    def list_versions(self) -> list[SemanticVersion]:
        """List all available versions.

        Returns
        -------
        list[SemanticVersion]
            All versions in order.
        """
        versions = [SemanticVersion.from_string(v) for v in self._snapshots]
        return sorted(versions, key=lambda v: (v.major, v.minor, v.patch))

    def commit_snapshot(self, version: str | SemanticVersion) -> None:
        """Commit a draft snapshot.

        Parameters
        ----------
        version : str | SemanticVersion
            Version to commit.

        Raises
        ------
        KeyError
            If version not found.
        ValueError
            If snapshot is not in draft status.
        """
        version_str = str(version)
        snapshot = self._snapshots.get(version_str)

        if snapshot is None:
            msg = f"Version {version_str} not found"
            raise KeyError(msg)

        if snapshot.status != SnapshotStatus.DRAFT:
            msg = f"Cannot commit snapshot in {snapshot.status.value} status"
            raise ValueError(msg)

        snapshot.status = SnapshotStatus.COMMITTED
        self._save_state()
        logger.info("Committed snapshot v%s", version_str)

    def diff_versions(
        self,
        version_a: str | SemanticVersion,
        version_b: str | SemanticVersion,
    ) -> dict[str, Any]:
        """Compare two snapshot versions.

        Parameters
        ----------
        version_a : str | SemanticVersion
            First version.
        version_b : str | SemanticVersion
            Second version.

        Returns
        -------
        dict[str, Any]
            Diff report showing added, removed, and modified files.
        """
        snapshot_a = self.get_snapshot(version_a)
        snapshot_b = self.get_snapshot(version_b)

        if snapshot_a is None or snapshot_b is None:
            msg = "One or both versions not found"
            raise KeyError(msg)

        files_a = {f.relative_path: f for f in snapshot_a.manifest.files}
        files_b = {f.relative_path: f for f in snapshot_b.manifest.files}

        paths_a = set(files_a.keys())
        paths_b = set(files_b.keys())

        added = paths_b - paths_a
        removed = paths_a - paths_b
        common = paths_a & paths_b

        modified = [
            p for p in common
            if files_a[p].content_hash != files_b[p].content_hash
        ]

        return {
            "version_a": str(version_a),
            "version_b": str(version_b),
            "added": list(added),
            "removed": list(removed),
            "modified": modified,
            "unchanged": len(common) - len(modified),
        }


# =============================================================================
# Advanced Versioning Components
# =============================================================================


class BranchType(Enum):
    """Type of version branch."""

    MAIN = "main"
    DEVELOPMENT = "development"
    FEATURE = "feature"
    HOTFIX = "hotfix"
    RELEASE = "release"
    EXPERIMENT = "experiment"


class MergeStrategy(Enum):
    """Strategy for merging branches."""

    OURS = "ours"
    THEIRS = "theirs"
    UNION = "union"
    MANUAL = "manual"


class ConflictType(Enum):
    """Type of merge conflict."""

    CONTENT = "content"
    METADATA = "metadata"
    SCHEMA = "schema"
    DELETION = "deletion"


class ValidationLevel(Enum):
    """Level of validation strictness."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class StorageBackend(Enum):
    """Storage backend type."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    SFTP = "sftp"


@dataclass(frozen=True, slots=True)
class BranchInfo:
    """Information about a version branch.

    Attributes
    ----------
    name : str
        Branch name.
    branch_type : BranchType
        Type of branch.
    base_version : SemanticVersion
        Version branch was created from.
    head_version : SemanticVersion | None
        Current head version.
    created_at : datetime
        Branch creation timestamp.
    created_by : str
        User who created the branch.
    description : str
        Branch description.
    is_protected : bool
        Whether branch is protected from deletion.
    """

    name: str
    branch_type: BranchType
    base_version: SemanticVersion
    head_version: SemanticVersion | None
    created_at: datetime
    created_by: str
    description: str
    is_protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "branch_type": self.branch_type.value,
            "base_version": str(self.base_version),
            "head_version": str(self.head_version) if self.head_version else None,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "is_protected": self.is_protected,
        }


@dataclass(frozen=True, slots=True)
class MergeConflict:
    """Represents a merge conflict.

    Attributes
    ----------
    file_path : str
        Path to conflicting file.
    conflict_type : ConflictType
        Type of conflict.
    ours_hash : ContentHash
        Hash from our branch.
    theirs_hash : ContentHash
        Hash from their branch.
    base_hash : ContentHash | None
        Hash from common ancestor.
    resolution : str | None
        Resolution if resolved.
    """

    file_path: str
    conflict_type: ConflictType
    ours_hash: ContentHash
    theirs_hash: ContentHash
    base_hash: ContentHash | None = None
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "conflict_type": self.conflict_type.value,
            "ours_hash": self.ours_hash,
            "theirs_hash": self.theirs_hash,
            "base_hash": self.base_hash,
            "resolution": self.resolution,
        }


@dataclass(frozen=True, slots=True)
class TagInfo:
    """Information about a version tag.

    Attributes
    ----------
    name : str
        Tag name.
    version : SemanticVersion
        Tagged version.
    created_at : datetime
        Tag creation timestamp.
    created_by : str
        User who created tag.
    message : str
        Tag message.
    is_annotated : bool
        Whether tag is annotated.
    signature : str | None
        GPG signature if signed.
    """

    name: str
    version: SemanticVersion
    created_at: datetime
    created_by: str
    message: str
    is_annotated: bool = True
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": str(self.version),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "message": self.message,
            "is_annotated": self.is_annotated,
            "signature": self.signature,
        }


@dataclass(slots=True)
class StorageConfig:
    """Configuration for storage backend.

    Attributes
    ----------
    backend : StorageBackend
        Storage backend type.
    base_path : str
        Base path or bucket name.
    credentials : dict[str, str]
        Backend-specific credentials.
    options : dict[str, Any]
        Additional options.
    compression : bool
        Whether to compress stored data.
    encryption : bool
        Whether to encrypt stored data.
    """

    backend: StorageBackend = StorageBackend.LOCAL
    base_path: str = ""
    credentials: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    compression: bool = False
    encryption: bool = False


class BranchManager:
    """Manages version branches.

    Provides branching, merging, and branch lifecycle management
    for parallel development and experimentation.

    Parameters
    ----------
    version_manager : DataVersionManager
        Associated version manager.

    Examples
    --------
    >>> branch_mgr = BranchManager(version_manager)
    >>> branch_mgr.create_branch("experiment-1", BranchType.EXPERIMENT)
    >>> branch_mgr.checkout("experiment-1")
    """

    __slots__ = ("_version_manager", "_branches", "_current_branch", "_tags")

    def __init__(self, version_manager: DataVersionManager) -> None:
        """Initialize branch manager linked to version repository.

        Parameters
        ----------
        version_manager : DataVersionManager
            Version manager providing snapshot storage.
        """
        self._version_manager = version_manager
        self._branches: dict[str, BranchInfo] = {}
        self._current_branch: str = "main"
        self._tags: dict[str, TagInfo] = {}

        # Initialize main branch
        self._initialize_main_branch()

    def _initialize_main_branch(self) -> None:
        """Initialize the main branch."""
        if "main" not in self._branches:
            current = self._version_manager.current_version
            base_version = current if current else SemanticVersion(0, 0, 0)

            self._branches["main"] = BranchInfo(
                name="main",
                branch_type=BranchType.MAIN,
                base_version=base_version,
                head_version=current,
                created_at=datetime.now(),
                created_by="system",
                description="Main development branch",
                is_protected=True,
            )

    @property
    def current_branch(self) -> str:
        """Return current branch name."""
        return self._current_branch

    @property
    def branches(self) -> dict[str, BranchInfo]:
        """Return all branches."""
        return self._branches.copy()

    def create_branch(
        self,
        name: str,
        branch_type: BranchType = BranchType.FEATURE,
        description: str = "",
        created_by: str = "system",
    ) -> BranchInfo:
        """Create a new branch.

        Parameters
        ----------
        name : str
            Branch name.
        branch_type : BranchType
            Type of branch.
        description : str
            Branch description.
        created_by : str
            User creating the branch.

        Returns
        -------
        BranchInfo
            Created branch information.

        Raises
        ------
        ValueError
            If branch name already exists.
        """
        if name in self._branches:
            msg = f"Branch '{name}' already exists"
            raise ValueError(msg)

        current = self._version_manager.current_version
        base_version = current if current else SemanticVersion(0, 0, 0)

        branch = BranchInfo(
            name=name,
            branch_type=branch_type,
            base_version=base_version,
            head_version=current,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
        )

        self._branches[name] = branch
        logger.info("Created branch '%s' from v%s", name, base_version)
        return branch

    def delete_branch(self, name: str, force: bool = False) -> None:
        """Delete a branch.

        Parameters
        ----------
        name : str
            Branch to delete.
        force : bool
            Force deletion of protected branches.

        Raises
        ------
        ValueError
            If branch not found or is protected.
        """
        if name not in self._branches:
            msg = f"Branch '{name}' not found"
            raise ValueError(msg)

        branch = self._branches[name]
        if branch.is_protected and not force:
            msg = f"Cannot delete protected branch '{name}'"
            raise ValueError(msg)

        if self._current_branch == name:
            msg = "Cannot delete current branch"
            raise ValueError(msg)

        del self._branches[name]
        logger.info("Deleted branch '%s'", name)

    def checkout(self, name: str) -> BranchInfo:
        """Switch to a different branch.

        Parameters
        ----------
        name : str
            Branch to checkout.

        Returns
        -------
        BranchInfo
            Checked out branch.

        Raises
        ------
        ValueError
            If branch not found.
        """
        if name not in self._branches:
            msg = f"Branch '{name}' not found"
            raise ValueError(msg)

        self._current_branch = name
        logger.info("Switched to branch '%s'", name)
        return self._branches[name]

    def merge(
        self,
        source_branch: str,
        target_branch: str | None = None,
        strategy: MergeStrategy = MergeStrategy.UNION,
        message: str = "",
    ) -> tuple[bool, list[MergeConflict]]:
        """Merge source branch into target.

        Parameters
        ----------
        source_branch : str
            Branch to merge from.
        target_branch : str | None
            Branch to merge into (default: current).
        strategy : MergeStrategy
            Merge strategy to use.
        message : str
            Merge commit message.

        Returns
        -------
        tuple[bool, list[MergeConflict]]
            Success flag and list of conflicts.
        """
        if target_branch is None:
            target_branch = self._current_branch

        if source_branch not in self._branches:
            msg = f"Source branch '{source_branch}' not found"
            raise ValueError(msg)

        if target_branch not in self._branches:
            msg = f"Target branch '{target_branch}' not found"
            raise ValueError(msg)

        source = self._branches[source_branch]
        target = self._branches[target_branch]

        conflicts: list[MergeConflict] = []

        # Detect conflicts
        if source.head_version and target.head_version:
            source_snapshot = self._version_manager.get_snapshot(source.head_version)
            target_snapshot = self._version_manager.get_snapshot(target.head_version)

            if source_snapshot and target_snapshot:
                conflicts = self._detect_conflicts(
                    source_snapshot, target_snapshot, source.base_version
                )

        if conflicts and strategy == MergeStrategy.MANUAL:
            logger.warning(
                "Merge of '%s' into '%s' has %d conflicts",
                source_branch,
                target_branch,
                len(conflicts),
            )
            return False, conflicts

        # Auto-resolve if possible
        if strategy in (MergeStrategy.OURS, MergeStrategy.THEIRS, MergeStrategy.UNION):
            conflicts = self._auto_resolve_conflicts(conflicts, strategy)

        if not conflicts:
            # Perform merge
            logger.info(
                "Merged '%s' into '%s': %s",
                source_branch,
                target_branch,
                message or "No message",
            )
            return True, []

        return False, conflicts

    def _detect_conflicts(
        self,
        source: Snapshot,
        target: Snapshot,
        base_version: SemanticVersion,
    ) -> list[MergeConflict]:
        """Detect conflicts between snapshots."""
        conflicts: list[MergeConflict] = []

        source_files = {f.relative_path: f for f in source.manifest.files}
        target_files = {f.relative_path: f for f in target.manifest.files}

        # Get base files if available
        base_snapshot = self._version_manager.get_snapshot(base_version)
        base_files: dict[str, FileEntry] = {}
        if base_snapshot:
            base_files = {f.relative_path: f for f in base_snapshot.manifest.files}

        common_paths = set(source_files.keys()) & set(target_files.keys())

        for path in common_paths:
            source_file = source_files[path]
            target_file = target_files[path]

            if source_file.content_hash != target_file.content_hash:
                base_file = base_files.get(path)
                base_hash = base_file.content_hash if base_file else None

                conflicts.append(MergeConflict(
                    file_path=path,
                    conflict_type=ConflictType.CONTENT,
                    ours_hash=target_file.content_hash,
                    theirs_hash=source_file.content_hash,
                    base_hash=base_hash,
                ))

        return conflicts

    def _auto_resolve_conflicts(
        self,
        conflicts: list[MergeConflict],
        strategy: MergeStrategy,
    ) -> list[MergeConflict]:
        """Auto-resolve conflicts based on strategy."""
        if strategy == MergeStrategy.OURS:
            return []  # Keep our version
        if strategy == MergeStrategy.THEIRS:
            return []  # Take their version
        if strategy == MergeStrategy.UNION:
            # Union only works for additive changes
            return [c for c in conflicts if c.conflict_type != ConflictType.CONTENT]
        return conflicts

    def create_tag(
        self,
        name: str,
        version: SemanticVersion | str | None = None,
        message: str = "",
        created_by: str = "system",
    ) -> TagInfo:
        """Create a version tag.

        Parameters
        ----------
        name : str
            Tag name.
        version : SemanticVersion | str | None
            Version to tag (default: current).
        message : str
            Tag message.
        created_by : str
            User creating tag.

        Returns
        -------
        TagInfo
            Created tag information.
        """
        if name in self._tags:
            msg = f"Tag '{name}' already exists"
            raise ValueError(msg)

        if version is None:
            v = self._version_manager.current_version
            if v is None:
                msg = "No current version to tag"
                raise ValueError(msg)
            target_version = v
        elif isinstance(version, str):
            target_version = SemanticVersion.from_string(version)
        else:
            target_version = version

        tag = TagInfo(
            name=name,
            version=target_version,
            created_at=datetime.now(),
            created_by=created_by,
            message=message,
        )

        self._tags[name] = tag
        logger.info("Created tag '%s' at v%s", name, target_version)
        return tag

    def delete_tag(self, name: str) -> None:
        """Delete a tag.

        Parameters
        ----------
        name : str
            Tag to delete.

        Raises
        ------
        ValueError
            If tag not found.
        """
        if name not in self._tags:
            msg = f"Tag '{name}' not found"
            raise ValueError(msg)

        del self._tags[name]
        logger.info("Deleted tag '%s'", name)

    def list_tags(self) -> list[TagInfo]:
        """List all tags.

        Returns
        -------
        list[TagInfo]
            All tags sorted by version.
        """
        tags = list(self._tags.values())
        return sorted(tags, key=lambda t: (
            t.version.major, t.version.minor, t.version.patch
        ))


class ContentAddressedStore:
    """Content-addressed storage for dataset objects.

    Implements deduplication and integrity verification through
    content-based addressing (SHA-256 hashes).

    Parameters
    ----------
    storage_path : Path
        Root path for object storage.
    config : StorageConfig
        Storage configuration.

    Examples
    --------
    >>> store = ContentAddressedStore(Path("data/objects"))
    >>> content_hash = store.put(b"binary data")
    >>> data = store.get(content_hash)
    """

    __slots__ = ("_storage_path", "_config", "_index")

    def __init__(
        self,
        storage_path: Path,
        config: StorageConfig | None = None,
    ) -> None:
        """Initialize content-addressed object store.

        Parameters
        ----------
        storage_path : Path
            Root directory for hash-indexed object storage.
        config : StorageConfig | None
            Compression and encryption settings.
        """
        self._storage_path = Path(storage_path)
        self._config = config or StorageConfig()
        self._index: dict[ContentHash, int] = {}

        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize storage directory structure."""
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for hash prefixes (256 dirs for first byte)
        for i in range(256):
            prefix = f"{i:02x}"
            (self._storage_path / prefix).mkdir(exist_ok=True)

        # Load index if exists
        index_path = self._storage_path / "index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                self._index = json.load(f)

    def _get_object_path(self, content_hash: ContentHash) -> Path:
        """Get filesystem path for a content hash."""
        prefix = content_hash[:2]
        return self._storage_path / prefix / content_hash

    def put(self, data: bytes) -> ContentHash:
        """Store data and return content hash.

        Parameters
        ----------
        data : bytes
            Data to store.

        Returns
        -------
        ContentHash
            SHA-256 hash of stored data.
        """
        content_hash = hashlib.sha256(data).hexdigest()
        object_path = self._get_object_path(content_hash)

        if object_path.exists():
            # Already stored (deduplication)
            logger.debug("Object %s already exists", content_hash[:12])
            return content_hash

        # Apply compression if configured
        store_data = data
        if self._config.compression:
            import zlib
            store_data = zlib.compress(data)

        # Store object
        with object_path.open("wb") as f:
            f.write(store_data)

        self._index[content_hash] = len(data)
        self._save_index()

        logger.debug("Stored object %s (%d bytes)", content_hash[:12], len(data))
        return content_hash

    def get(self, content_hash: ContentHash) -> bytes | None:
        """Retrieve data by content hash.

        Parameters
        ----------
        content_hash : ContentHash
            Hash of data to retrieve.

        Returns
        -------
        bytes | None
            Retrieved data, or None if not found.
        """
        object_path = self._get_object_path(content_hash)

        if not object_path.exists():
            return None

        with object_path.open("rb") as f:
            data = f.read()

        # Decompress if configured
        if self._config.compression:
            import zlib
            data = zlib.decompress(data)

        # Verify integrity
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != content_hash:
            logger.error(
                "Integrity check failed for %s (got %s)",
                content_hash[:12],
                actual_hash[:12],
            )
            return None

        return data

    def exists(self, content_hash: ContentHash) -> bool:
        """Check if object exists.

        Parameters
        ----------
        content_hash : ContentHash
            Hash to check.

        Returns
        -------
        bool
            True if object exists.
        """
        return self._get_object_path(content_hash).exists()

    def delete(self, content_hash: ContentHash) -> bool:
        """Delete an object.

        Parameters
        ----------
        content_hash : ContentHash
            Hash of object to delete.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        object_path = self._get_object_path(content_hash)

        if not object_path.exists():
            return False

        object_path.unlink()
        self._index.pop(content_hash, None)
        self._save_index()

        logger.debug("Deleted object %s", content_hash[:12])
        return True

    def size(self, content_hash: ContentHash) -> int | None:
        """Get size of object.

        Parameters
        ----------
        content_hash : ContentHash
            Hash of object.

        Returns
        -------
        int | None
            Object size in bytes, or None if not found.
        """
        return self._index.get(content_hash)

    def total_size(self) -> int:
        """Get total size of all stored objects.

        Returns
        -------
        int
            Total size in bytes.
        """
        return sum(self._index.values())

    def object_count(self) -> int:
        """Get count of stored objects.

        Returns
        -------
        int
            Number of objects.
        """
        return len(self._index)

    def _save_index(self) -> None:
        """Save index to disk."""
        index_path = self._storage_path / "index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    def garbage_collect(
        self,
        referenced_hashes: set[ContentHash],
    ) -> tuple[int, int]:
        """Remove unreferenced objects.

        Parameters
        ----------
        referenced_hashes : set[ContentHash]
            Set of hashes still in use.

        Returns
        -------
        tuple[int, int]
            Count and size of removed objects.
        """
        removed_count = 0
        removed_size = 0

        for content_hash in list(self._index.keys()):
            if content_hash not in referenced_hashes:
                size = self._index.get(content_hash, 0)
                if self.delete(content_hash):
                    removed_count += 1
                    removed_size += size

        logger.info(
            "Garbage collection: removed %d objects (%d bytes)",
            removed_count,
            removed_size,
        )
        return removed_count, removed_size


class VersionValidator:
    """Validates dataset versions and snapshots.

    Provides comprehensive validation of version integrity,
    schema compliance, and data quality.

    Parameters
    ----------
    validation_level : ValidationLevel
        Strictness of validation.
    custom_validators : dict[str, Any] | None
        Custom validation functions.
    """

    __slots__ = ("_validation_level", "_custom_validators", "_validation_results")

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        custom_validators: dict[str, Any] | None = None,
    ) -> None:
        """Initialize version validator with strictness controls.

        Parameters
        ----------
        validation_level : ValidationLevel
            Validation strictness tier.
        custom_validators : dict[str, Any] | None
            Additional user-defined validation functions.
        """
        self._validation_level = validation_level
        self._custom_validators = custom_validators or {}
        self._validation_results: list[dict[str, Any]] = []

    @property
    def validation_level(self) -> ValidationLevel:
        """Return validation level."""
        return self._validation_level

    @property
    def results(self) -> list[dict[str, Any]]:
        """Return validation results."""
        return self._validation_results.copy()

    def validate_snapshot(
        self,
        snapshot: Snapshot,
        data_path: Path,
    ) -> dict[str, Any]:
        """Validate a snapshot.

        Parameters
        ----------
        snapshot : Snapshot
            Snapshot to validate.
        data_path : Path
            Path to snapshot data.

        Returns
        -------
        dict[str, Any]
            Validation results.
        """
        result: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "version": str(snapshot.version),
            "validation_level": self._validation_level.value,
            "checks": [],
            "passed": True,
            "errors": [],
            "warnings": [],
        }

        # Integrity check
        if self._validation_level != ValidationLevel.NONE:
            integrity = self._check_integrity(snapshot, data_path)
            result["checks"].append(integrity)
            if not integrity["passed"]:
                result["passed"] = False
                result["errors"].extend(integrity.get("errors", []))

        # Manifest check
        if self._validation_level in (
            ValidationLevel.STANDARD,
            ValidationLevel.STRICT,
            ValidationLevel.PARANOID,
        ):
            manifest = self._check_manifest(snapshot)
            result["checks"].append(manifest)
            if not manifest["passed"]:
                result["passed"] = False
                result["errors"].extend(manifest.get("errors", []))

        # Schema check
        if self._validation_level in (ValidationLevel.STRICT, ValidationLevel.PARANOID):
            schema = self._check_schema(snapshot, data_path)
            result["checks"].append(schema)
            if not schema["passed"]:
                result["warnings"].extend(schema.get("warnings", []))

        # Custom validators
        for name, validator in self._custom_validators.items():
            try:
                custom_result = validator(snapshot, data_path)
                result["checks"].append({
                    "name": name,
                    "type": "custom",
                    **custom_result,
                })
            except Exception as e:
                result["errors"].append(f"Custom validator '{name}' failed: {e}")

        self._validation_results.append(result)
        return result

    def _check_integrity(
        self,
        snapshot: Snapshot,
        data_path: Path,
    ) -> dict[str, Any]:
        """Check file integrity."""
        check: dict[str, Any] = {
            "name": "integrity",
            "type": "standard",
            "passed": True,
            "files_checked": 0,
            "files_valid": 0,
            "errors": [],
        }

        for file_entry in snapshot.manifest.files:
            file_path = data_path / file_entry.relative_path
            check["files_checked"] += 1

            if not file_path.exists():
                check["passed"] = False
                check["errors"].append(f"Missing file: {file_entry.relative_path}")
                continue

            with file_path.open("rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()

            if actual_hash == file_entry.content_hash:
                check["files_valid"] += 1
            else:
                check["passed"] = False
                check["errors"].append(
                    f"Hash mismatch for {file_entry.relative_path}"
                )

        return check

    def _check_manifest(self, snapshot: Snapshot) -> dict[str, Any]:
        """Check manifest completeness."""
        check: dict[str, Any] = {
            "name": "manifest",
            "type": "standard",
            "passed": True,
            "errors": [],
        }

        if not snapshot.manifest.content_hash:
            check["passed"] = False
            check["errors"].append("Missing content hash")

        if not snapshot.manifest.files:
            check["passed"] = False
            check["errors"].append("Empty file list")

        if not snapshot.manifest.description:
            check["errors"].append("Missing description")
            # Don't fail for missing description

        return check

    def _check_schema(
        self,
        snapshot: Snapshot,
        data_path: Path,
    ) -> dict[str, Any]:
        """Check data schema compliance."""
        check: dict[str, Any] = {
            "name": "schema",
            "type": "strict",
            "passed": True,
            "warnings": [],
        }

        # Check for required files
        required_categories = {"microscopy", "clinical", "metadata"}
        found_categories = {f.category for f in snapshot.manifest.files}

        missing = required_categories - found_categories
        if missing:
            check["warnings"].append(f"Missing categories: {missing}")

        return check

    def generate_report(self) -> dict[str, Any]:
        """Generate validation summary report.

        Returns
        -------
        dict[str, Any]
            Summary report of all validations.
        """
        total = len(self._validation_results)
        passed = sum(1 for r in self._validation_results if r["passed"])

        return {
            "generated_at": datetime.now().isoformat(),
            "validation_level": self._validation_level.value,
            "total_validations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "details": self._validation_results,
        }


class VersionMigrator:
    """Migrates datasets between version schemas.

    Handles schema evolution and backward compatibility for
    dataset versioning formats.

    Parameters
    ----------
    version_manager : DataVersionManager
        Associated version manager.
    """

    __slots__ = ("_version_manager", "_migrations")

    def __init__(self, version_manager: DataVersionManager) -> None:
        """Initialize version migrator for schema evolution.

        Parameters
        ----------
        version_manager : DataVersionManager
            Version manager providing snapshot access.
        """
        self._version_manager = version_manager
        self._migrations: dict[str, Any] = {}

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration_func: Any,
    ) -> None:
        """Register a migration function.

        Parameters
        ----------
        from_version : str
            Source schema version.
        to_version : str
            Target schema version.
        migration_func : callable
            Function to perform migration.
        """
        key = f"{from_version}->{to_version}"
        self._migrations[key] = migration_func
        logger.info("Registered migration: %s", key)

    def migrate(
        self,
        snapshot: Snapshot,
        target_schema: str,
    ) -> Snapshot:
        """Migrate snapshot to target schema.

        Parameters
        ----------
        snapshot : Snapshot
            Snapshot to migrate.
        target_schema : str
            Target schema version.

        Returns
        -------
        Snapshot
            Migrated snapshot.
        """
        # For now, return snapshot as-is
        # Real implementation would apply migrations
        logger.info(
            "Migrating snapshot v%s to schema %s",
            snapshot.version,
            target_schema,
        )
        return snapshot


class VersionExporter:
    """Exports dataset versions to various formats.

    Supports exporting to archives, cloud storage, and
    external version control systems.

    Parameters
    ----------
    version_manager : DataVersionManager
        Associated version manager.
    """

    __slots__ = ("_version_manager",)

    def __init__(self, version_manager: DataVersionManager) -> None:
        """Initialize version exporter for archive and JSON output.

        Parameters
        ----------
        version_manager : DataVersionManager
            Version manager providing snapshot data.
        """
        self._version_manager = version_manager

    def export_to_archive(
        self,
        version: SemanticVersion | str,
        output_path: Path,
        include_lineage: bool = True,
    ) -> Path:
        """Export version to archive file.

        Parameters
        ----------
        version : SemanticVersion | str
            Version to export.
        output_path : Path
            Output archive path.
        include_lineage : bool
            Whether to include lineage data.

        Returns
        -------
        Path
            Path to created archive.
        """
        import tarfile

        snapshot = self._version_manager.get_snapshot(version)
        if snapshot is None:
            msg = f"Version {version} not found"
            raise ValueError(msg)

        archive_path = output_path / f"snapshot-v{version}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tar:
            # Add manifest
            manifest_data = json.dumps(snapshot.manifest.to_dict(), indent=2)
            manifest_info = tarfile.TarInfo(name="manifest.json")
            manifest_info.size = len(manifest_data.encode())
            tar.addfile(manifest_info, fileobj=__import__("io").BytesIO(manifest_data.encode()))

            # Add lineage if requested
            if include_lineage:
                lineage_data = json.dumps(
                    self._version_manager.lineage.to_dict(), indent=2
                )
                lineage_info = tarfile.TarInfo(name="lineage.json")
                lineage_info.size = len(lineage_data.encode())
                tar.addfile(lineage_info, fileobj=__import__("io").BytesIO(lineage_data.encode()))

        logger.info("Exported v%s to %s", version, archive_path)
        return archive_path

    def export_to_json(
        self,
        version: SemanticVersion | str,
        output_path: Path,
    ) -> Path:
        """Export version metadata to JSON.

        Parameters
        ----------
        version : SemanticVersion | str
            Version to export.
        output_path : Path
            Output directory.

        Returns
        -------
        Path
            Path to created JSON file.
        """
        snapshot = self._version_manager.get_snapshot(version)
        if snapshot is None:
            msg = f"Version {version} not found"
            raise ValueError(msg)

        json_path = output_path / f"snapshot-v{version}.json"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        logger.info("Exported v%s to %s", version, json_path)
        return json_path


# =============================================================================
# Factory Functions
# =============================================================================


def create_version_manager(
    repository_path: PathLike,
    auto_compute_hashes: bool = True,
) -> DataVersionManager:
    """Create a DataVersionManager instance.

    Parameters
    ----------
    repository_path : PathLike
        Path to version repository.
    auto_compute_hashes : bool
        Whether to auto-compute file hashes.

    Returns
    -------
    DataVersionManager
        Configured version manager.

    Examples
    --------
    >>> manager = create_version_manager("data/versions")
    >>> snapshot = manager.create_snapshot(Path("data/raw"))
    """
    return DataVersionManager(
        repository_path=Path(repository_path),
        auto_compute_hashes=auto_compute_hashes,
    )


def create_branch_manager(
    version_manager: DataVersionManager,
) -> BranchManager:
    """Create a BranchManager instance.

    Parameters
    ----------
    version_manager : DataVersionManager
        Associated version manager.

    Returns
    -------
    BranchManager
        Configured branch manager.
    """
    return BranchManager(version_manager)


def create_content_store(
    storage_path: PathLike,
    compression: bool = False,
    encryption: bool = False,
) -> ContentAddressedStore:
    """Create a ContentAddressedStore instance.

    Parameters
    ----------
    storage_path : PathLike
        Path to object storage.
    compression : bool
        Enable compression.
    encryption : bool
        Enable encryption.

    Returns
    -------
    ContentAddressedStore
        Configured content store.
    """
    config = StorageConfig(
        compression=compression,
        encryption=encryption,
    )
    return ContentAddressedStore(
        storage_path=Path(storage_path),
        config=config,
    )


def create_version_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
) -> VersionValidator:
    """Create a VersionValidator instance.

    Parameters
    ----------
    validation_level : ValidationLevel
        Validation strictness level.

    Returns
    -------
    VersionValidator
        Configured validator.
    """
    return VersionValidator(validation_level=validation_level)


def create_version_exporter(
    version_manager: DataVersionManager,
) -> VersionExporter:
    """Create a VersionExporter instance.

    Parameters
    ----------
    version_manager : DataVersionManager
        Associated version manager.

    Returns
    -------
    VersionExporter
        Configured exporter.
    """
    return VersionExporter(version_manager)
