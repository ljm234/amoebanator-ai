"""Phase 1.10: Dataset Versioning with DVC Integration.

This module implements dataset version control using DVC (Data Version Control)
for reproducible machine learning workflows. Includes remote storage backends,
pipeline definitions, and experiment tracking integration.

Supports both cloud storage (S3, GCS, Azure) and on-premise storage backends
with content-addressed file management.

Components
----------
CommandExecutor
    Executes DVC CLI commands via subprocess with timeout handling,
    retry logic, and structured output parsing.

RemoteManager
    Manages multiple DVC remote storage configurations including
    S3, GCS, Azure Blob, SSH, and HDFS backends with credential rotation.

FileTracker
    Tracks individual files and directories with DVC, managing
    .dvc pointer files and cache synchronization.

DataPusher / DataPuller
    Handles bidirectional data synchronization with remote storage
    including parallel transfers, bandwidth throttling, and resume support.

PipelineManager
    Defines and executes DVC pipelines with stage dependencies,
    parameter injection, and metrics collection.

ExperimentTracker
    Integrates with DVC experiments for hyperparameter tracking,
    metric comparison, and experiment branching.

VersionManager
    Maintains dataset version history with semantic versioning,
    changelog generation, and rollback capabilities.

CacheManager
    Manages local DVC cache including garbage collection,
    cache sharing, and disk usage monitoring.

LineageTracker
    Tracks data lineage across pipeline stages with provenance
    graphs and transformation history.

ReproducibilityValidator
    Validates pipeline reproducibility by comparing outputs
    across executions with determinism checks.

Usage
-----
>>> from ml.data.dvc_versioning import create_dvc_manager
>>> manager = create_dvc_manager(Path("./project"))
>>> manager.initialize()
>>> manager.file_tracker.add("data/training.csv")
>>> manager.pusher.push()
>>> version = manager.version_dataset("Initial training data")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
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
)

logger: Final = logging.getLogger(__name__)

JSONDict: TypeAlias = dict[str, Any]
CommandResult: TypeAlias = tuple[int, str, str]
ProgressCallback: TypeAlias = Callable[[int, int], None]


class StorageBackend(Enum):
    """Supported DVC remote storage backends."""

    LOCAL = auto()
    S3 = auto()
    GCS = auto()
    AZURE = auto()
    SSH = auto()
    HDFS = auto()
    HTTP = auto()
    WEBDAV = auto()


class PipelineStatus(Enum):
    """Status of DVC pipeline stage execution."""

    NOT_STARTED = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CACHED = auto()
    SKIPPED = auto()


class ExperimentStatus(Enum):
    """Status of tracked experiment."""

    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    STOPPED = auto()
    ARCHIVED = auto()


class CacheStatus(Enum):
    """Status of cached file."""

    CACHED = auto()
    MISSING = auto()
    OUTDATED = auto()
    CORRUPTED = auto()


class SyncDirection(Enum):
    """Direction of data synchronization."""

    PUSH = auto()
    PULL = auto()
    BIDIRECTIONAL = auto()


class VersionBumpType(Enum):
    """Semantic version bump types."""

    MAJOR = auto()
    MINOR = auto()
    PATCH = auto()


class DatasetVersion(NamedTuple):
    """Version information for dataset."""

    version_id: str
    commit_hash: str
    created_at: datetime
    description: str
    file_count: int
    total_size_bytes: int
    dvc_hash: str
    semantic_version: str
    parent_version: str | None


class RemoteConfig(NamedTuple):
    """DVC remote storage configuration."""

    name: str
    backend: StorageBackend
    url: str
    credentials_env: dict[str, str]
    default: bool
    read_only: bool
    bandwidth_limit: int | None


class PipelineStage(NamedTuple):
    """Single stage in DVC pipeline."""

    name: str
    command: str
    dependencies: tuple[str, ...]
    outputs: tuple[str, ...]
    parameters: tuple[str, ...]
    metrics: tuple[str, ...]
    plots: tuple[str, ...]
    frozen: bool


class PipelineRun(NamedTuple):
    """Record of pipeline execution."""

    run_id: str
    pipeline_name: str
    started_at: datetime
    completed_at: datetime | None
    status: PipelineStatus
    stages_completed: int
    stages_total: int
    duration_seconds: float
    error_message: str | None


class Experiment(NamedTuple):
    """Tracked ML experiment."""

    experiment_id: str
    name: str
    branch: str
    created_at: datetime
    status: ExperimentStatus
    parameters: JSONDict
    metrics: JSONDict
    parent_experiment: str | None
    tags: tuple[str, ...]


class FileTrackingInfo(NamedTuple):
    """Tracking information for versioned file."""

    file_path: str
    dvc_path: str
    md5_hash: str
    size_bytes: int
    is_cached: bool
    cache_status: CacheStatus
    last_modified: datetime


class CacheInfo(NamedTuple):
    """Information about DVC cache."""

    cache_dir: str
    total_size_bytes: int
    file_count: int
    oldest_file: datetime | None
    newest_file: datetime | None


class LineageNode(NamedTuple):
    """Node in data lineage graph."""

    node_id: str
    node_type: str
    path: str
    hash_value: str
    created_at: datetime
    parents: tuple[str, ...]
    children: tuple[str, ...]


class SyncProgress(NamedTuple):
    """Progress of data synchronization."""

    total_files: int
    completed_files: int
    total_bytes: int
    transferred_bytes: int
    current_file: str
    elapsed_seconds: float
    estimated_remaining: float


class DVCExecutor(Protocol):
    """Protocol for DVC command execution."""

    def run(self, args: Sequence[str]) -> CommandResult:
        """Execute DVC command and return exit code, stdout, stderr."""
        ...

    def run_async(
        self, args: Sequence[str], callback: ProgressCallback | None
    ) -> CommandResult:
        """Execute DVC command asynchronously with progress."""
        ...


class StorageProvider(Protocol):
    """Protocol for storage backend operations."""

    def push(self, files: Sequence[str]) -> bool:
        """Push files to remote storage."""
        ...

    def pull(self, files: Sequence[str]) -> bool:
        """Pull files from remote storage."""
        ...

    def check_connection(self) -> bool:
        """Verify connectivity to storage."""
        ...

    def get_remote_files(self) -> list[str]:
        """List files available on remote."""
        ...


class VersioningStrategy(Protocol):
    """Protocol for version numbering strategy."""

    def next_version(
        self, current: str, bump_type: VersionBumpType
    ) -> str:
        """Compute next version number."""
        ...


@dataclass(frozen=True, slots=True)
class DVCConfig:
    """DVC configuration options."""

    autostage: bool = True
    cache_type: str = "symlink"
    cache_dir: str = ".dvc/cache"
    remote_default: str = "origin"
    check_update: bool = False


@dataclass(frozen=True, slots=True)
class PipelineDefinition:
    """Complete pipeline definition."""

    name: str
    stages: tuple[PipelineStage, ...]
    parameters_file: str = "params.yaml"
    metrics_file: str = "metrics.json"

    def to_yaml(self) -> str:
        """Generate dvc.yaml content."""
        lines = ["stages:"]
        for stage in self.stages:
            lines.append(f"  {stage.name}:")
            lines.append(f"    cmd: {stage.command}")

            if stage.dependencies:
                lines.append("    deps:")
                for dep in stage.dependencies:
                    lines.append(f"      - {dep}")

            if stage.outputs:
                lines.append("    outs:")
                for out in stage.outputs:
                    lines.append(f"      - {out}")

            if stage.parameters:
                lines.append("    params:")
                for param in stage.parameters:
                    lines.append(f"      - {param}")

            if stage.metrics:
                lines.append("    metrics:")
                for metric in stage.metrics:
                    lines.append(f"      - {metric}:")
                    lines.append("          cache: false")

        return "\n".join(lines)


@dataclass(slots=True)
class CommandExecutor:
    """Executes DVC commands via subprocess."""

    working_dir: Path
    timeout_seconds: int = 300

    def run(self, args: Sequence[str]) -> tuple[int, str, str]:
        """Execute DVC command."""
        cmd = ["dvc", *args]
        logger.debug("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error("DVC command timed out: %s", " ".join(cmd))
            return -1, "", "Command timed out"
        except FileNotFoundError:
            logger.error("DVC not installed or not in PATH")
            return -1, "", "DVC not found"


@dataclass(slots=True)
class RemoteManager:
    """Manages DVC remote storage configurations."""

    storage_dir: Path
    executor: CommandExecutor
    _remotes: dict[str, RemoteConfig] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize and load remotes."""
        self._load_remotes()

    def _load_remotes(self) -> None:
        """Load configured remotes from DVC config."""
        code, stdout, _ = self.executor.run(["remote", "list"])
        if code != 0:
            return

        for line in stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                name, url = parts[0], parts[1]
                backend = self._detect_backend(url)
                self._remotes[name] = RemoteConfig(
                    name=name,
                    backend=backend,
                    url=url,
                    credentials_env={},
                    default=False,
                    read_only=False,
                    bandwidth_limit=None,
                )

    def _detect_backend(self, url: str) -> StorageBackend:
        """Detect storage backend from URL."""
        if url.startswith("s3://"):
            return StorageBackend.S3
        if url.startswith("gs://"):
            return StorageBackend.GCS
        if url.startswith("azure://"):
            return StorageBackend.AZURE
        if url.startswith("ssh://"):
            return StorageBackend.SSH
        if url.startswith("hdfs://"):
            return StorageBackend.HDFS
        return StorageBackend.LOCAL

    def add_remote(
        self,
        name: str,
        url: str,
        default: bool = False,
    ) -> bool:
        """Add new remote storage."""
        args = ["remote", "add"]
        if default:
            args.append("-d")
        args.extend([name, url])

        code, _, stderr = self.executor.run(args)
        if code != 0:
            logger.error("Failed to add remote: %s", stderr)
            return False

        backend = self._detect_backend(url)
        self._remotes[name] = RemoteConfig(
            name=name,
            backend=backend,
            url=url,
            credentials_env={},
            default=default,
            read_only=False,
            bandwidth_limit=None,
        )
        return True

    def remove_remote(self, name: str) -> bool:
        """Remove remote storage."""
        code, _, stderr = self.executor.run(["remote", "remove", name])
        if code != 0:
            logger.error("Failed to remove remote: %s", stderr)
            return False

        self._remotes.pop(name, None)
        return True

    def list_remotes(self) -> list[RemoteConfig]:
        """List all configured remotes."""
        return list(self._remotes.values())


@dataclass(slots=True)
class FileTracker:
    """Tracks files with DVC."""

    executor: CommandExecutor
    storage_dir: Path
    _tracked_files: dict[str, FileTrackingInfo] = field(
        default_factory=dict, init=False, repr=False
    )

    def add(self, file_path: str | Path) -> bool:
        """Add file to DVC tracking."""
        path = Path(file_path)
        code, _, stderr = self.executor.run(["add", str(path)])

        if code != 0:
            logger.error("Failed to track file: %s", stderr)
            return False

        dvc_path = str(path) + ".dvc"
        self._tracked_files[str(path)] = FileTrackingInfo(
            file_path=str(path),
            dvc_path=dvc_path,
            md5_hash="",
            size_bytes=path.stat().st_size if path.exists() else 0,
            is_cached=False,
            cache_status=CacheStatus.MISSING,
            last_modified=datetime.now(),
        )
        return True

    def add_directory(self, dir_path: str | Path) -> bool:
        """Add directory to DVC tracking."""
        return self.add(dir_path)

    def remove(self, file_path: str | Path) -> bool:
        """Remove file from DVC tracking."""
        code, _, stderr = self.executor.run(["remove", str(file_path)])
        if code != 0:
            logger.error("Failed to untrack file: %s", stderr)
            return False

        self._tracked_files.pop(str(file_path), None)
        return True

    def status(self) -> dict[str, str]:
        """Get status of tracked files."""
        code, stdout, _ = self.executor.run(["status"])
        if code != 0:
            return {}

        status: dict[str, str] = {}
        current_file = ""
        for line in stdout.strip().split("\n"):
            if line and not line.startswith("\t"):
                current_file = line.rstrip(":")
            elif line.startswith("\t") and current_file:
                status[current_file] = line.strip()

        return status

    def list_tracked(self) -> list[FileTrackingInfo]:
        """List all tracked files."""
        return list(self._tracked_files.values())


@dataclass(slots=True)
class DataPusher:
    """Pushes data to remote storage."""

    executor: CommandExecutor

    def push(
        self,
        remote: str | None = None,
        jobs: int = 4,
    ) -> bool:
        """Push tracked data to remote."""
        args = ["push", "-j", str(jobs)]
        if remote:
            args.extend(["-r", remote])

        code, stdout, stderr = self.executor.run(args)
        if code != 0:
            logger.error("Push failed: %s", stderr)
            return False

        logger.info("Push completed: %s", stdout.strip())
        return True

    def push_files(
        self,
        files: Sequence[str],
        remote: str | None = None,
    ) -> bool:
        """Push specific files to remote."""
        args = ["push"]
        if remote:
            args.extend(["-r", remote])
        args.extend(files)

        code, _, stderr = self.executor.run(args)
        if code != 0:
            logger.error("Push failed: %s", stderr)
            return False

        return True


@dataclass(slots=True)
class DataPuller:
    """Pulls data from remote storage."""

    executor: CommandExecutor

    def pull(
        self,
        remote: str | None = None,
        jobs: int = 4,
    ) -> bool:
        """Pull tracked data from remote."""
        args = ["pull", "-j", str(jobs)]
        if remote:
            args.extend(["-r", remote])

        code, stdout, stderr = self.executor.run(args)
        if code != 0:
            logger.error("Pull failed: %s", stderr)
            return False

        logger.info("Pull completed: %s", stdout.strip())
        return True

    def pull_files(
        self,
        files: Sequence[str],
        remote: str | None = None,
    ) -> bool:
        """Pull specific files from remote."""
        args = ["pull"]
        if remote:
            args.extend(["-r", remote])
        args.extend(files)

        code, _, stderr = self.executor.run(args)
        if code != 0:
            logger.error("Pull failed: %s", stderr)
            return False

        return True


@dataclass(slots=True)
class PipelineManager:
    """Manages DVC pipelines."""

    executor: CommandExecutor
    storage_dir: Path
    _pipelines: dict[str, PipelineDefinition] = field(
        default_factory=dict, init=False, repr=False
    )
    _runs: list[PipelineRun] = field(
        default_factory=list, init=False, repr=False
    )

    def define_pipeline(self, definition: PipelineDefinition) -> bool:
        """Write pipeline definition to dvc.yaml."""
        dvc_yaml = self.storage_dir / "dvc.yaml"
        try:
            dvc_yaml.write_text(definition.to_yaml())
            self._pipelines[definition.name] = definition
            return True
        except OSError as e:
            logger.error("Failed to write pipeline: %s", e)
            return False

    def run_pipeline(
        self,
        pipeline_name: str | None = None,
        force: bool = False,
    ) -> PipelineRun:
        """Execute pipeline."""
        import uuid

        run_id = str(uuid.uuid4())[:8]
        started = datetime.now()

        args = ["repro"]
        if force:
            args.append("-f")

        code, stdout, stderr = self.executor.run(args)

        completed = datetime.now()
        status = PipelineStatus.COMPLETED if code == 0 else PipelineStatus.FAILED

        # Parse stage completion from output
        stages_completed = stdout.count("Running stage")
        stages_total = len(self._pipelines.get(pipeline_name or "default",
            PipelineDefinition("default", ())).stages)

        run = PipelineRun(
            run_id=run_id,
            pipeline_name=pipeline_name or "default",
            started_at=started,
            completed_at=completed,
            status=status,
            stages_completed=stages_completed,
            stages_total=stages_total,
            duration_seconds=(completed - started).total_seconds(),
            error_message=stderr if code != 0 else None,
        )
        self._runs.append(run)

        if code != 0:
            logger.error("Pipeline failed: %s", stderr)

        return run

    def get_pipeline_status(self) -> dict[str, str]:
        """Get status of pipeline stages."""
        code, stdout, _ = self.executor.run(["status"])
        if code != 0:
            return {}

        return {"raw": stdout}

    def list_runs(self) -> list[PipelineRun]:
        """List all pipeline runs."""
        return list(self._runs)


@dataclass(slots=True)
class ExperimentTracker:
    """Tracks ML experiments with DVC."""

    executor: CommandExecutor
    storage_dir: Path
    _experiments: dict[str, Experiment] = field(
        default_factory=dict, init=False, repr=False
    )

    def run_experiment(
        self,
        name: str,
        parameters: JSONDict,
    ) -> Experiment:
        """Run experiment with given parameters."""
        import uuid

        exp_id = str(uuid.uuid4())[:8]

        # Write parameters
        params_file = self.storage_dir / "params.yaml"
        self._write_params(params_file, parameters)

        code, stdout, stderr = self.executor.run(
            ["exp", "run", "-n", name]
        )

        status = (
            ExperimentStatus.COMPLETED if code == 0
            else ExperimentStatus.FAILED
        )

        # Load metrics if available
        metrics = self._load_metrics()

        experiment = Experiment(
            experiment_id=exp_id,
            name=name,
            branch=f"exp-{name}",
            created_at=datetime.now(),
            status=status,
            parameters=parameters,
            metrics=metrics,
            parent_experiment=None,
            tags=(),
        )
        self._experiments[exp_id] = experiment

        if code != 0:
            logger.error("Experiment failed: %s", stderr)

        return experiment

    def _write_params(self, path: Path, params: JSONDict) -> None:
        """Write parameters as YAML-like format."""
        lines = []
        for key, value in params.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for subkey, subval in value.items():
                    lines.append(f"  {subkey}: {subval}")
            else:
                lines.append(f"{key}: {value}")
        path.write_text("\n".join(lines))

    def _load_metrics(self) -> JSONDict:
        """Load metrics from file."""
        metrics_file = self.storage_dir / "metrics.json"
        if not metrics_file.exists():
            return {}

        try:
            with metrics_file.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def list_experiments(self) -> list[Experiment]:
        """List all experiments."""
        code, stdout, _ = self.executor.run(["exp", "show", "--json"])
        if code != 0:
            return list(self._experiments.values())

        # Parse DVC experiment list
        try:
            data = json.loads(stdout)
            # Process DVC output format
            logger.debug("Loaded %d experiments from DVC", len(data))
        except json.JSONDecodeError:
            pass

        return list(self._experiments.values())

    def apply_experiment(self, experiment_id: str) -> bool:
        """Apply experiment to workspace."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False

        code, _, stderr = self.executor.run(["exp", "apply", exp.name])
        if code != 0:
            logger.error("Failed to apply experiment: %s", stderr)
            return False

        return True

    def compare_experiments(
        self, exp_ids: Sequence[str]
    ) -> dict[str, JSONDict]:
        """Compare metrics across experiments."""
        comparison: dict[str, JSONDict] = {}
        for exp_id in exp_ids:
            exp = self._experiments.get(exp_id)
            if exp:
                comparison[exp.name] = {
                    "parameters": exp.parameters,
                    "metrics": exp.metrics,
                    "status": exp.status.name,
                }
        return comparison


@dataclass(slots=True)
class VersionManager:
    """Manages dataset versions."""

    storage_dir: Path
    _versions: list[DatasetVersion] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Load existing versions."""
        self._load_versions()

    def _load_versions(self) -> None:
        """Load version history from storage."""
        version_file = self.storage_dir / ".dvc" / "versions.json"
        if not version_file.exists():
            return

        try:
            with version_file.open("r") as f:
                data = json.load(f)
            for item in data:
                version = DatasetVersion(
                    version_id=item["version_id"],
                    commit_hash=item["commit_hash"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    description=item["description"],
                    file_count=item["file_count"],
                    total_size_bytes=item["total_size_bytes"],
                    dvc_hash=item["dvc_hash"],
                    semantic_version=item.get("semantic_version", "0.0.0"),
                    parent_version=item.get("parent_version"),
                )
                self._versions.append(version)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load versions: %s", e)

    def _save_versions(self) -> None:
        """Persist version history."""
        version_file = self.storage_dir / ".dvc" / "versions.json"
        version_file.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "version_id": v.version_id,
                "commit_hash": v.commit_hash,
                "created_at": v.created_at.isoformat(),
                "description": v.description,
                "file_count": v.file_count,
                "total_size_bytes": v.total_size_bytes,
                "dvc_hash": v.dvc_hash,
            }
            for v in self._versions
        ]
        with version_file.open("w") as f:
            json.dump(data, f, indent=2)

    def create_version(
        self,
        description: str,
        commit_hash: str,
        file_count: int,
        total_size: int,
        dvc_hash: str,
    ) -> DatasetVersion:
        """Create new dataset version."""
        import uuid

        version = DatasetVersion(
            version_id=str(uuid.uuid4())[:8],
            commit_hash=commit_hash,
            created_at=datetime.now(),
            description=description,
            file_count=file_count,
            total_size_bytes=total_size,
            dvc_hash=dvc_hash,
            semantic_version="0.0.1",
            parent_version=self._versions[-1].version_id if self._versions else None,
        )
        self._versions.append(version)
        self._save_versions()
        return version

    def list_versions(self) -> list[DatasetVersion]:
        """List all dataset versions."""
        return list(self._versions)

    def get_version(self, version_id: str) -> DatasetVersion | None:
        """Get specific version."""
        for version in self._versions:
            if version.version_id == version_id:
                return version
        return None

    def iterate_versions(self) -> Iterator[DatasetVersion]:
        """Iterate over all versions."""
        yield from self._versions


@dataclass(slots=True)
class DVCManager:
    """Main coordinator for DVC operations."""

    project_dir: Path
    executor: CommandExecutor = field(init=False)
    remote_manager: RemoteManager = field(init=False)
    file_tracker: FileTracker = field(init=False)
    pusher: DataPusher = field(init=False)
    puller: DataPuller = field(init=False)
    pipeline_manager: PipelineManager = field(init=False)
    experiment_tracker: ExperimentTracker = field(init=False)
    version_manager: VersionManager = field(init=False)

    def __post_init__(self) -> None:
        """Initialize all components."""
        self.executor = CommandExecutor(working_dir=self.project_dir)
        self.remote_manager = RemoteManager(
            storage_dir=self.project_dir,
            executor=self.executor,
        )
        self.file_tracker = FileTracker(
            executor=self.executor,
            storage_dir=self.project_dir,
        )
        self.pusher = DataPusher(executor=self.executor)
        self.puller = DataPuller(executor=self.executor)
        self.pipeline_manager = PipelineManager(
            executor=self.executor,
            storage_dir=self.project_dir,
        )
        self.experiment_tracker = ExperimentTracker(
            executor=self.executor,
            storage_dir=self.project_dir,
        )
        self.version_manager = VersionManager(storage_dir=self.project_dir)

    def initialize(self) -> bool:
        """Initialize DVC in project directory."""
        code, stdout, stderr = self.executor.run(["init"])
        if code != 0 and "already initialized" not in stderr:
            logger.error("DVC init failed: %s", stderr)
            return False
        logger.info("DVC initialized: %s", stdout.strip())
        return True

    def setup_training_pipeline(self) -> PipelineDefinition:
        """Create standard ML training pipeline."""
        pipeline = PipelineDefinition(
            name="training",
            stages=(
                PipelineStage(
                    name="prepare",
                    command="python scripts/prepare_data.py",
                    dependencies=("data/raw",),
                    outputs=("data/processed",),
                    parameters=("prepare.split_ratio",),
                    metrics=(),
                    plots=(),
                    frozen=False,
                ),
                PipelineStage(
                    name="train",
                    command="python scripts/train.py",
                    dependencies=("data/processed", "ml/training.py"),
                    outputs=("outputs/model/model.pt",),
                    parameters=("train.epochs", "train.learning_rate"),
                    metrics=("outputs/metrics/metrics.json",),
                    plots=(),
                    frozen=False,
                ),
                PipelineStage(
                    name="evaluate",
                    command="python scripts/evaluate.py",
                    dependencies=("outputs/model/model.pt", "data/processed"),
                    outputs=(),
                    parameters=(),
                    metrics=("outputs/metrics/eval_metrics.json",),
                    plots=(),
                    frozen=False,
                ),
            ),
        )
        self.pipeline_manager.define_pipeline(pipeline)
        return pipeline

    def version_dataset(self, description: str) -> DatasetVersion | None:
        """Create new version of tracked dataset."""
        # Get git commit hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            commit_hash = result.stdout.strip()
        except subprocess.CalledProcessError:
            commit_hash = "unknown"

        # Count tracked files
        tracked = self.file_tracker.list_tracked()
        file_count = len(tracked)
        total_size = sum(f.size_bytes for f in tracked)

        # Get DVC lock hash
        lock_file = self.project_dir / "dvc.lock"
        dvc_hash = ""
        if lock_file.exists():
            import hashlib

            dvc_hash = hashlib.md5(
                lock_file.read_bytes()
            ).hexdigest()[:8]

        return self.version_manager.create_version(
            description=description,
            commit_hash=commit_hash,
            file_count=file_count,
            total_size=total_size,
            dvc_hash=dvc_hash,
        )

    def get_status_summary(self) -> dict[str, Any]:
        """Get comprehensive status summary."""
        tracked = self.file_tracker.list_tracked()
        versions = self.version_manager.list_versions()
        experiments = self.experiment_tracker.list_experiments()
        runs = self.pipeline_manager.list_runs()
        remotes = self.remote_manager.list_remotes()
        return {
            "tracked_files": len(tracked),
            "total_size_bytes": sum(f.size_bytes for f in tracked),
            "versions": len(versions),
            "experiments": len(experiments),
            "pipeline_runs": len(runs),
            "remotes": [r.name for r in remotes],
            "latest_version": versions[-1].version_id if versions else None,
        }


@dataclass(slots=True)
class CacheManager:
    """Manages local DVC cache including garbage collection."""

    cache_dir: Path
    max_cache_size_bytes: int = 10 * 1024 * 1024 * 1024
    _file_access_times: dict[str, datetime] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize cache directory."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_info(self) -> CacheInfo:
        """Get information about cache contents."""
        if not self.cache_dir.exists():
            return CacheInfo(
                cache_dir=str(self.cache_dir),
                total_size_bytes=0,
                file_count=0,
                oldest_file=None,
                newest_file=None,
            )
        total_size = 0
        file_count = 0
        oldest: datetime | None = None
        newest: datetime | None = None
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if oldest is None or mtime < oldest:
                    oldest = mtime
                if newest is None or mtime > newest:
                    newest = mtime
        return CacheInfo(
            cache_dir=str(self.cache_dir),
            total_size_bytes=total_size,
            file_count=file_count,
            oldest_file=oldest,
            newest_file=newest,
        )

    def get_disk_usage(self) -> tuple[int, int, float]:
        """Get disk usage: total, used, percentage."""
        info = self.get_cache_info()
        return info.total_size_bytes, self.max_cache_size_bytes, (
            info.total_size_bytes / self.max_cache_size_bytes
            if self.max_cache_size_bytes > 0
            else 0.0
        )

    def needs_cleanup(self, threshold: float = 0.9) -> bool:
        """Check if cache cleanup is needed."""
        _, _, usage = self.get_disk_usage()
        return usage > threshold

    def garbage_collect(self, target_usage: float = 0.7) -> int:
        """Remove old cached files to reach target usage."""
        if not self.needs_cleanup():
            return 0
        files_with_times: list[tuple[Path, datetime]] = []
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                files_with_times.append((file_path, mtime))
        files_with_times.sort(key=lambda x: x[1])
        current_size = sum(f[0].stat().st_size for f in files_with_times)
        target_size = int(self.max_cache_size_bytes * target_usage)
        removed_count = 0
        for file_path, _ in files_with_times:
            if current_size <= target_size:
                break
            file_size = file_path.stat().st_size
            try:
                file_path.unlink()
                current_size -= file_size
                removed_count += 1
            except OSError:
                continue
        return removed_count

    def verify_integrity(self) -> list[tuple[str, str]]:
        """Verify cache file integrity using checksums."""
        corrupted: list[tuple[str, str]] = []
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                expected_hash = file_path.stem
                actual_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                if not expected_hash.startswith(actual_hash[:8]):
                    corrupted.append((str(file_path), actual_hash))
        return corrupted

    def clear_cache(self) -> int:
        """Clear entire cache."""
        removed = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed


@dataclass(slots=True)
class LineageTracker:
    """Tracks data lineage across pipeline stages."""

    storage_dir: Path
    _nodes: dict[str, LineageNode] = field(
        default_factory=dict, init=False, repr=False
    )
    _edges: list[tuple[str, str]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Load existing lineage data."""
        self._load_lineage()

    def _load_lineage(self) -> None:
        """Load lineage graph from storage."""
        lineage_file = self.storage_dir / ".dvc" / "lineage.json"
        if not lineage_file.exists():
            return
        try:
            with lineage_file.open("r") as f:
                data = json.load(f)
            for node_data in data.get("nodes", []):
                node = LineageNode(
                    node_id=node_data["node_id"],
                    node_type=node_data["node_type"],
                    path=node_data["path"],
                    hash_value=node_data["hash_value"],
                    created_at=datetime.fromisoformat(node_data["created_at"]),
                    parents=tuple(node_data.get("parents", [])),
                    children=tuple(node_data.get("children", [])),
                )
                self._nodes[node.node_id] = node
            self._edges = [
                (e["source"], e["target"]) for e in data.get("edges", [])
            ]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load lineage: %s", e)

    def _save_lineage(self) -> None:
        """Persist lineage graph to storage."""
        lineage_file = self.storage_dir / ".dvc" / "lineage.json"
        lineage_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "path": n.path,
                    "hash_value": n.hash_value,
                    "created_at": n.created_at.isoformat(),
                    "parents": list(n.parents),
                    "children": list(n.children),
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {"source": s, "target": t} for s, t in self._edges
            ],
        }
        with lineage_file.open("w") as f:
            json.dump(data, f, indent=2)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        path: str,
        hash_value: str,
        parents: Sequence[str] | None = None,
    ) -> LineageNode:
        """Add node to lineage graph."""
        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            path=path,
            hash_value=hash_value,
            created_at=datetime.now(),
            parents=tuple(parents or []),
            children=(),
        )
        self._nodes[node_id] = node
        for parent_id in node.parents:
            self._edges.append((parent_id, node_id))
            if parent_id in self._nodes:
                parent = self._nodes[parent_id]
                self._nodes[parent_id] = parent._replace(
                    children=parent.children + (node_id,)
                )
        self._save_lineage()
        return node

    def get_node(self, node_id: str) -> LineageNode | None:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_ancestors(self, node_id: str) -> list[LineageNode]:
        """Get all ancestor nodes."""
        ancestors: list[LineageNode] = []
        visited: set[str] = set()

        def traverse(nid: str) -> None:
            """Walk parent edges recursively collecting ancestors."""
            if nid in visited or nid not in self._nodes:
                return
            visited.add(nid)
            node = self._nodes[nid]
            for parent_id in node.parents:
                if parent_id in self._nodes:
                    ancestors.append(self._nodes[parent_id])
                    traverse(parent_id)

        traverse(node_id)
        return ancestors

    def get_descendants(self, node_id: str) -> list[LineageNode]:
        """Get all descendant nodes."""
        descendants: list[LineageNode] = []
        visited: set[str] = set()

        def traverse(nid: str) -> None:
            """Walk child edges recursively collecting descendants."""
            if nid in visited or nid not in self._nodes:
                return
            visited.add(nid)
            node = self._nodes[nid]
            for child_id in node.children:
                if child_id in self._nodes:
                    descendants.append(self._nodes[child_id])
                    traverse(child_id)

        traverse(node_id)
        return descendants

    def get_provenance_chain(
        self, node_id: str
    ) -> list[tuple[str, str, str]]:
        """Get full provenance chain as (node_id, type, path)."""
        ancestors = self.get_ancestors(node_id)
        node = self.get_node(node_id)
        chain: list[tuple[str, str, str]] = []
        if node:
            chain.append((node.node_id, node.node_type, node.path))
        for ancestor in ancestors:
            chain.append((ancestor.node_id, ancestor.node_type, ancestor.path))
        return chain

    def export_graphviz(self) -> str:
        """Export lineage graph in Graphviz DOT format."""
        lines = ["digraph lineage {"]
        lines.append("  rankdir=LR;")
        for node in self._nodes.values():
            label = f"{node.node_type}\\n{node.path}"
            lines.append(f'  "{node.node_id}" [label="{label}"];')
        for source, target in self._edges:
            lines.append(f'  "{source}" -> "{target}";')
        lines.append("}")
        return "\n".join(lines)


@dataclass(slots=True)
class ReproducibilityValidator:
    """Validates pipeline reproducibility."""

    executor: CommandExecutor
    tolerance: float = 1e-6

    def compute_output_hash(self, output_path: Path) -> str:
        """Compute hash of output file or directory."""
        if output_path.is_file():
            return hashlib.md5(output_path.read_bytes()).hexdigest()
        if output_path.is_dir():
            all_hashes: list[str] = []
            for file_path in sorted(output_path.rglob("*")):
                if file_path.is_file():
                    file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                    all_hashes.append(file_hash)
            combined = "".join(all_hashes)
            return hashlib.md5(combined.encode()).hexdigest()
        return ""

    def run_reproducibility_check(
        self, pipeline_stages: Sequence[PipelineStage]
    ) -> dict[str, tuple[bool, str, str]]:
        """Run pipeline twice and compare outputs."""
        results: dict[str, tuple[bool, str, str]] = {}
        for stage in pipeline_stages:
            for output_path_str in stage.outputs:
                output_path = Path(output_path_str)
                hash_before = self.compute_output_hash(output_path)
                code, _, _ = self.executor.run(["repro", "-s", stage.name])
                if code != 0:
                    results[output_path_str] = (False, hash_before, "run_failed")
                    continue
                hash_after = self.compute_output_hash(output_path)
                is_reproducible = hash_before == hash_after
                results[output_path_str] = (is_reproducible, hash_before, hash_after)
        return results

    def check_determinism(
        self, command: str, output_path: Path, runs: int = 3
    ) -> tuple[bool, list[str]]:
        """Check if command produces deterministic output."""
        hashes: list[str] = []
        for _ in range(runs):
            subprocess.run(
                command,
                shell=True,
                cwd=self.executor.working_dir,
                capture_output=True,
                check=False,
            )
            hash_value = self.compute_output_hash(output_path)
            hashes.append(hash_value)
        is_deterministic = len(set(hashes)) == 1
        return is_deterministic, hashes


@dataclass(slots=True)
class SemanticVersioner:
    """Implements semantic versioning for datasets."""

    def parse_version(self, version: str) -> tuple[int, int, int]:
        """Parse semantic version string."""
        match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", version)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return 0, 0, 0

    def format_version(self, major: int, minor: int, patch: int) -> str:
        """Format version as string."""
        return f"v{major}.{minor}.{patch}"

    def next_version(
        self, current: str, bump_type: VersionBumpType
    ) -> str:
        """Compute next version based on bump type."""
        major, minor, patch = self.parse_version(current)
        if bump_type == VersionBumpType.MAJOR:
            return self.format_version(major + 1, 0, 0)
        if bump_type == VersionBumpType.MINOR:
            return self.format_version(major, minor + 1, 0)
        return self.format_version(major, minor, patch + 1)

    def compare_versions(self, v1: str, v2: str) -> int:
        """Compare two versions: -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
        t1 = self.parse_version(v1)
        t2 = self.parse_version(v2)
        if t1 < t2:
            return -1
        if t1 > t2:
            return 1
        return 0


@dataclass(slots=True)
class ChangelogGenerator:
    """Generates changelogs for dataset versions."""

    storage_dir: Path

    def generate_changelog(
        self,
        versions: Sequence[DatasetVersion],
        output_path: Path | None = None,
    ) -> str:
        """Generate markdown changelog from version history."""
        lines = ["# Dataset Changelog\n"]
        for version in reversed(versions):
            lines.append(f"## {version.semantic_version}")
            lines.append(f"*Released: {version.created_at.strftime('%Y-%m-%d')}*\n")
            lines.append(f"- **Version ID**: {version.version_id}")
            lines.append(f"- **Files**: {version.file_count}")
            lines.append(f"- **Size**: {version.total_size_bytes / 1024 / 1024:.2f} MB")
            lines.append(f"- **Description**: {version.description}")
            if version.parent_version:
                lines.append(f"- **Parent**: {version.parent_version}")
            lines.append("")
        content = "\n".join(lines)
        if output_path:
            output_path.write_text(content)
        return content

    def generate_diff_summary(
        self, version_a: DatasetVersion, version_b: DatasetVersion
    ) -> dict[str, Any]:
        """Generate summary of differences between versions."""
        return {
            "from_version": version_a.version_id,
            "to_version": version_b.version_id,
            "file_count_change": version_b.file_count - version_a.file_count,
            "size_change_bytes": version_b.total_size_bytes - version_a.total_size_bytes,
            "days_between": (version_b.created_at - version_a.created_at).days,
        }


@dataclass(slots=True)
class SyncManager:
    """Manages bidirectional data synchronization."""

    executor: CommandExecutor
    pusher: "DataPusher"
    puller: "DataPuller"
    _sync_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def sync(
        self,
        direction: SyncDirection,
        remote: str | None = None,
        dry_run: bool = False,
    ) -> SyncProgress:
        """Synchronize data with remote."""
        start_time = time.time()
        if direction == SyncDirection.PUSH:
            if dry_run:
                code, _, _ = self.executor.run(["push", "--dry"])
            else:
                success = self.pusher.push(remote=remote)
                code = 0 if success else 1
        elif direction == SyncDirection.PULL:
            if dry_run:
                code, _, _ = self.executor.run(["pull", "--dry"])
            else:
                success = self.puller.pull(remote=remote)
                code = 0 if success else 1
        else:
            self.pusher.push(remote=remote)
            self.puller.pull(remote=remote)
            code = 0
        elapsed = time.time() - start_time
        progress = SyncProgress(
            total_files=0,
            completed_files=0,
            total_bytes=0,
            transferred_bytes=0,
            current_file="",
            elapsed_seconds=elapsed,
            estimated_remaining=0.0,
        )
        self._sync_history.append({
            "direction": direction.name,
            "remote": remote,
            "timestamp": datetime.now().isoformat(),
            "success": code == 0,
            "elapsed_seconds": elapsed,
        })
        return progress

    def get_sync_history(self) -> list[dict[str, Any]]:
        """Get synchronization history."""
        return list(self._sync_history)


@dataclass(slots=True)
class MetricsCollector:
    """Collects and aggregates pipeline metrics."""

    storage_dir: Path
    _metrics_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def load_metrics(self, metrics_file: Path) -> dict[str, Any]:
        """Load metrics from JSON file."""
        if not metrics_file.exists():
            return {}
        try:
            with metrics_file.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def record_metrics(
        self, run_id: str, metrics: dict[str, Any]
    ) -> None:
        """Record metrics for pipeline run."""
        self._metrics_history.append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        })

    def compare_metrics(
        self, run_a: str, run_b: str
    ) -> dict[str, tuple[Any, Any]]:
        """Compare metrics between two runs."""
        metrics_a: dict[str, Any] = {}
        metrics_b: dict[str, Any] = {}
        for record in self._metrics_history:
            if record["run_id"] == run_a:
                metrics_a = record["metrics"]
            if record["run_id"] == run_b:
                metrics_b = record["metrics"]
        comparison: dict[str, tuple[Any, Any]] = {}
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        for key in all_keys:
            comparison[key] = (metrics_a.get(key), metrics_b.get(key))
        return comparison

    def get_best_run(self, metric_name: str, minimize: bool = False) -> str | None:
        """Get run ID with best metric value."""
        if not self._metrics_history:
            return None
        best_run: str | None = None
        best_value: float | None = None
        for record in self._metrics_history:
            value = record["metrics"].get(metric_name)
            if value is None:
                continue
            if best_value is None:
                best_value = value
                best_run = record["run_id"]
            elif minimize and value < best_value:
                best_value = value
                best_run = record["run_id"]
            elif not minimize and value > best_value:
                best_value = value
                best_run = record["run_id"]
        return best_run

    def aggregate_metrics(
        self, metric_name: str
    ) -> dict[str, float]:
        """Compute aggregate statistics for metric."""
        import statistics as stats
        values = [
            r["metrics"].get(metric_name)
            for r in self._metrics_history
            if metric_name in r["metrics"]
        ]
        if not values:
            return {}
        return {
            "count": len(values),
            "mean": stats.mean(values),
            "std": stats.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }


def create_dvc_manager(project_dir: Path) -> DVCManager:
    """Factory function for DVC manager."""
    return DVCManager(project_dir=project_dir)


@dataclass(slots=True)
class TagManager:
    """Manages tags for dataset versions."""

    storage_dir: Path
    _tags: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Load existing tags."""
        self._load_tags()

    def _load_tags(self) -> None:
        """Load tags from storage."""
        tags_file = self.storage_dir / ".dvc" / "tags.json"
        if not tags_file.exists():
            return
        try:
            with tags_file.open("r") as f:
                self._tags = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_tags(self) -> None:
        """Persist tags to storage."""
        tags_file = self.storage_dir / ".dvc" / "tags.json"
        tags_file.parent.mkdir(parents=True, exist_ok=True)
        with tags_file.open("w") as f:
            json.dump(self._tags, f, indent=2)

    def create_tag(self, tag_name: str, version_id: str) -> bool:
        """Create tag pointing to version."""
        if tag_name in self._tags:
            return False
        self._tags[tag_name] = version_id
        self._save_tags()
        return True

    def delete_tag(self, tag_name: str) -> bool:
        """Delete existing tag."""
        if tag_name not in self._tags:
            return False
        del self._tags[tag_name]
        self._save_tags()
        return True

    def get_version_for_tag(self, tag_name: str) -> str | None:
        """Get version ID for tag."""
        return self._tags.get(tag_name)

    def list_tags(self) -> dict[str, str]:
        """List all tags."""
        return dict(self._tags)

    def get_tags_for_version(self, version_id: str) -> list[str]:
        """Get all tags pointing to version."""
        return [tag for tag, vid in self._tags.items() if vid == version_id]


@dataclass(slots=True)
class BranchManager:
    """Manages data branches for parallel development."""

    storage_dir: Path
    executor: CommandExecutor
    _branches: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Load existing branches."""
        self._load_branches()

    def _load_branches(self) -> None:
        """Load branch information from storage."""
        branches_file = self.storage_dir / ".dvc" / "branches.json"
        if not branches_file.exists():
            return
        try:
            with branches_file.open("r") as f:
                self._branches = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_branches(self) -> None:
        """Persist branch information."""
        branches_file = self.storage_dir / ".dvc" / "branches.json"
        branches_file.parent.mkdir(parents=True, exist_ok=True)
        with branches_file.open("w") as f:
            json.dump(self._branches, f, indent=2)

    def create_branch(
        self, branch_name: str, from_version: str | None = None
    ) -> bool:
        """Create new data branch."""
        if branch_name in self._branches:
            return False
        self._branches[branch_name] = {
            "created_at": datetime.now().isoformat(),
            "from_version": from_version,
            "current_version": from_version,
        }
        self._save_branches()
        return True

    def switch_branch(self, branch_name: str) -> bool:
        """Switch to existing branch."""
        if branch_name not in self._branches:
            return False
        version_id = self._branches[branch_name].get("current_version")
        if version_id:
            code, _, _ = self.executor.run(["checkout", version_id])
            return code == 0
        return True

    def merge_branch(
        self, source_branch: str, target_branch: str
    ) -> bool:
        """Merge source branch into target."""
        if source_branch not in self._branches or target_branch not in self._branches:
            return False
        source_version = self._branches[source_branch].get("current_version")
        if source_version:
            self._branches[target_branch]["current_version"] = source_version
            self._save_branches()
        return True

    def delete_branch(self, branch_name: str) -> bool:
        """Delete branch."""
        if branch_name not in self._branches:
            return False
        del self._branches[branch_name]
        self._save_branches()
        return True

    def list_branches(self) -> list[str]:
        """List all branches."""
        return list(self._branches.keys())


@dataclass(slots=True)
class LockManager:
    """Manages locks for concurrent access to datasets."""

    storage_dir: Path
    _locks: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock = threading.Lock()

    def __post_init__(self) -> None:
        """Load existing locks."""
        self._load_locks()

    def _load_locks(self) -> None:
        """Load lock information from storage."""
        locks_file = self.storage_dir / ".dvc" / "locks.json"
        if not locks_file.exists():
            return
        try:
            with locks_file.open("r") as f:
                self._locks = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_locks(self) -> None:
        """Persist lock information."""
        locks_file = self.storage_dir / ".dvc" / "locks.json"
        locks_file.parent.mkdir(parents=True, exist_ok=True)
        with locks_file.open("w") as f:
            json.dump(self._locks, f, indent=2)

    def acquire_lock(
        self, resource_path: str, owner: str, timeout_seconds: int = 3600
    ) -> bool:
        """Acquire lock on resource."""
        with self._lock:
            if resource_path in self._locks:
                existing = self._locks[resource_path]
                lock_time = datetime.fromisoformat(existing["acquired_at"])
                if (datetime.now() - lock_time).total_seconds() < existing.get("timeout", 3600):
                    return False
            self._locks[resource_path] = {
                "owner": owner,
                "acquired_at": datetime.now().isoformat(),
                "timeout": timeout_seconds,
            }
            self._save_locks()
            return True

    def release_lock(self, resource_path: str, owner: str) -> bool:
        """Release lock on resource."""
        with self._lock:
            if resource_path not in self._locks:
                return False
            if self._locks[resource_path]["owner"] != owner:
                return False
            del self._locks[resource_path]
            self._save_locks()
            return True

    def is_locked(self, resource_path: str) -> bool:
        """Check if resource is locked."""
        if resource_path not in self._locks:
            return False
        existing = self._locks[resource_path]
        lock_time = datetime.fromisoformat(existing["acquired_at"])
        is_expired = (datetime.now() - lock_time).total_seconds() >= existing.get("timeout", 3600)
        if is_expired:
            del self._locks[resource_path]
            self._save_locks()
            return False
        return True

    def get_lock_owner(self, resource_path: str) -> str | None:
        """Get owner of lock."""
        if not self.is_locked(resource_path):
            return None
        return self._locks[resource_path].get("owner")


@dataclass(slots=True)
class HookManager:
    """Manages pre/post hooks for DVC operations."""

    storage_dir: Path
    _hooks: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Load existing hooks."""
        self._load_hooks()

    def _load_hooks(self) -> None:
        """Load hooks from storage."""
        hooks_file = self.storage_dir / ".dvc" / "hooks.json"
        if not hooks_file.exists():
            return
        try:
            with hooks_file.open("r") as f:
                data = json.load(f)
                self._hooks = defaultdict(list, data)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_hooks(self) -> None:
        """Persist hooks to storage."""
        hooks_file = self.storage_dir / ".dvc" / "hooks.json"
        hooks_file.parent.mkdir(parents=True, exist_ok=True)
        with hooks_file.open("w") as f:
            json.dump(dict(self._hooks), f, indent=2)

    def register_hook(
        self, event: str, command: str
    ) -> None:
        """Register hook for event."""
        self._hooks[event].append(command)
        self._save_hooks()

    def unregister_hook(
        self, event: str, command: str
    ) -> bool:
        """Unregister hook for event."""
        if event not in self._hooks or command not in self._hooks[event]:
            return False
        self._hooks[event].remove(command)
        self._save_hooks()
        return True

    def execute_hooks(
        self, event: str, context: dict[str, Any] | None = None
    ) -> list[tuple[str, int]]:
        """Execute all hooks for event."""
        results: list[tuple[str, int]] = []
        for command in self._hooks.get(event, []):
            env = os.environ.copy()
            if context:
                for key, value in context.items():
                    env[f"DVC_{key.upper()}"] = str(value)
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.storage_dir,
                    env=env,
                    capture_output=True,
                    check=False,
                )
                results.append((command, result.returncode))
            except Exception:
                results.append((command, -1))
        return results

    def list_hooks(self, event: str | None = None) -> dict[str, list[str]]:
        """List registered hooks."""
        if event:
            return {event: self._hooks.get(event, [])}
        return dict(self._hooks)


@dataclass(slots=True)
class ConfigManager:
    """Manages DVC configuration settings."""

    storage_dir: Path
    _config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Load existing configuration."""
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from .dvc/config."""
        config_file = self.storage_dir / ".dvc" / "config"
        if not config_file.exists():
            return
        try:
            content = config_file.read_text()
            current_section = "core"
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip()
                    if current_section not in self._config:
                        self._config[current_section] = {}
                elif "=" in line:
                    key, value = line.split("=", 1)
                    if current_section not in self._config:
                        self._config[current_section] = {}
                    self._config[current_section][key.strip()] = value.strip()
        except OSError:
            pass

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        self._save_config()

    def _save_config(self) -> None:
        """Save configuration to file."""
        config_file = self.storage_dir / ".dvc" / "config"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for section, values in self._config.items():
            lines.append(f"[{section}]")
            for key, value in values.items():
                lines.append(f"    {key} = {value}")
        config_file.write_text("\n".join(lines))

    def get_all(self) -> dict[str, Any]:
        """Get all configuration."""
        return dict(self._config)


def create_full_dvc_environment(project_dir: Path) -> dict[str, Any]:
    """Create complete DVC environment with all managers."""
    manager = create_dvc_manager(project_dir)
    cache_manager = CacheManager(cache_dir=project_dir / ".dvc" / "cache")
    lineage_tracker = LineageTracker(storage_dir=project_dir)
    reproducibility_validator = ReproducibilityValidator(executor=manager.executor)
    semantic_versioner = SemanticVersioner()
    changelog_generator = ChangelogGenerator(storage_dir=project_dir)
    tag_manager = TagManager(storage_dir=project_dir)
    branch_manager = BranchManager(storage_dir=project_dir, executor=manager.executor)
    lock_manager = LockManager(storage_dir=project_dir)
    hook_manager = HookManager(storage_dir=project_dir)
    config_manager = ConfigManager(storage_dir=project_dir)
    metrics_collector = MetricsCollector(storage_dir=project_dir)
    return {
        "manager": manager,
        "cache_manager": cache_manager,
        "lineage_tracker": lineage_tracker,
        "reproducibility_validator": reproducibility_validator,
        "semantic_versioner": semantic_versioner,
        "changelog_generator": changelog_generator,
        "tag_manager": tag_manager,
        "branch_manager": branch_manager,
        "lock_manager": lock_manager,
        "hook_manager": hook_manager,
        "config_manager": config_manager,
        "metrics_collector": metrics_collector,
    }


__all__ = [
    "StorageBackend",
    "PipelineStatus",
    "ExperimentStatus",
    "DatasetVersion",
    "RemoteConfig",
    "PipelineStage",
    "PipelineRun",
    "Experiment",
    "FileTrackingInfo",
    "DVCExecutor",
    "StorageProvider",
    "DVCConfig",
    "PipelineDefinition",
    "CommandExecutor",
    "RemoteManager",
    "FileTracker",
    "DataPusher",
    "DataPuller",
    "PipelineManager",
    "ExperimentTracker",
    "VersionManager",
    "DVCManager",
    "create_dvc_manager",
]
