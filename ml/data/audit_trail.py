"""
Tamper-Evident Chain-of-Custody Audit Trail.

Provides an immutable, cryptographically verifiable audit log for all
data acquisition operations. Each entry is linked to its predecessor
via a SHA-256 hash chain, and periodic Merkle tree checkpoints enable
efficient integrity verification of any log segment.

Architecture
------------
The audit system implements a two-layer integrity structure:

    ┌──────────────────────────────────────────────────────────────┐
    │          TAMPER-EVIDENT AUDIT ARCHITECTURE                   │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  LAYER 1 ─ Hash-Chained Log Entries                         │
    │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐            │
    │  │Entry 0│──>│Entry 1│──>│Entry 2│──>│Entry 3│──> ...      │
    │  │H(data)│   │H(E0+d)│   │H(E1+d)│   │H(E2+d)│            │
    │  └───────┘   └───────┘   └───────┘   └───────┘            │
    │                                                              │
    │  LAYER 2 ─ Merkle Tree Checkpoints                          │
    │              ┌──────────┐                                    │
    │              │   Root   │                                    │
    │              │ H(L + R) │                                    │
    │              └────┬─────┘                                    │
    │            ┌──────┴──────┐                                   │
    │       ┌────┴────┐  ┌────┴────┐                              │
    │       │ H(0+1) │  │ H(2+3) │                               │
    │       └────┬────┘  └────┬────┘                              │
    │        ┌───┴───┐    ┌───┴───┐                               │
    │       H(E0) H(E1) H(E2) H(E3)                              │
    │                                                              │
    │  Verification: O(log n) proof for any single entry          │
    │  Tampering detection: Any modification breaks the chain     │
    └──────────────────────────────────────────────────────────────┘

Compliance Coverage
-------------------
- HIPAA §164.312(b): Audit controls
- HIPAA §164.312(c)(1): Integrity mechanism
- 21 CFR Part 11: Electronic records, electronic signatures
- NIST SP 800-92: Guide to Computer Security Log Management
- NIST SP 800-53 AU-2, AU-3, AU-9: Audit events and protection
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Final,
    NamedTuple,
    Sequence,
    TypeAlias,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
HashDigest: TypeAlias = str
PathLike: TypeAlias = str | Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HASH_ALGORITHM: Final[str] = "sha256"
GENESIS_HASH: Final[str] = "0" * 64
CHECKPOINT_INTERVAL: Final[int] = 100


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class AuditEventType(Enum):
    """Classification of audit events."""

    # Data lifecycle events
    DATA_RECEIVED = "data_received"
    DATA_TRANSFERRED = "data_transferred"
    DATA_VERIFIED = "data_verified"
    DATA_QUARANTINED = "data_quarantined"
    DATA_RELEASED = "data_released"
    DATA_DELETED = "data_deleted"

    # Access events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    ACCESS_REVOKED = "access_revoked"

    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    IRB_STATUS_CHANGE = "irb_status_change"
    ATTESTATION_SIGNED = "attestation_signed"

    # Security events
    ENCRYPTION_APPLIED = "encryption_applied"
    DECRYPTION_APPLIED = "decryption_applied"
    CHECKSUM_VERIFIED = "checksum_verified"
    CHECKSUM_FAILED = "checksum_failed"
    INTEGRITY_VIOLATION = "integrity_violation"

    # System events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONFIGURATION_CHANGE = "configuration_change"


class IntegrityStatus(Enum):
    """Result of integrity verification."""

    VALID = "valid"
    TAMPERED = "tampered"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Audit Log Entry
# ═══════════════════════════════════════════════════════════════════════════

class AuditEntry(NamedTuple):
    """Single immutable audit log entry.

    Attributes
    ----------
    entry_id : str
        Unique entry identifier.
    sequence_number : int
        Monotonically increasing sequence number.
    timestamp : str
        ISO 8601 timestamp (UTC).
    event_type : str
        Classified event type.
    actor : str
        Person or system performing the action.
    resource : str
        Data resource affected.
    action_detail : str
        Human-readable description.
    metadata : dict[str, Any]
        Structured event-specific data.
    previous_hash : str
        SHA-256 of the preceding entry.
    entry_hash : str
        SHA-256 of this entry (computed over all above fields).
    """

    entry_id: str
    sequence_number: int
    timestamp: str
    event_type: str
    actor: str
    resource: str
    action_detail: str
    metadata: dict[str, Any]
    previous_hash: str
    entry_hash: str


def _compute_entry_hash(
    entry_id: str,
    sequence_number: int,
    timestamp: str,
    event_type: str,
    actor: str,
    resource: str,
    action_detail: str,
    metadata: dict[str, Any],
    previous_hash: str,
) -> str:
    """Compute SHA-256 hash over all entry fields except entry_hash."""
    payload = json.dumps(
        {
            "entry_id": entry_id,
            "sequence_number": sequence_number,
            "timestamp": timestamp,
            "event_type": event_type,
            "actor": actor,
            "resource": resource,
            "action_detail": action_detail,
            "metadata": metadata,
            "previous_hash": previous_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Merkle Tree
# ═══════════════════════════════════════════════════════════════════════════

def _hash_pair(left: str, right: str) -> str:
    """Compute parent hash from two child hashes."""
    combined = (left + right).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


class MerkleTree:
    """Binary Merkle tree for efficient batch integrity verification.

    Constructs a balanced binary hash tree over a sequence of
    leaf hashes, enabling O(log n) membership proofs.

    Parameters
    ----------
    leaf_hashes : Sequence[str]
        SHA-256 hashes of individual entries.
    """

    __slots__ = ("_leaves", "_tree", "_root")

    def __init__(self, leaf_hashes: Sequence[str]) -> None:
        self._leaves = list(leaf_hashes)
        self._tree: list[list[str]] = []
        self._root: str = ""
        self._build()

    @property
    def root(self) -> str:
        """Return the Merkle root hash."""
        return self._root

    @property
    def leaf_count(self) -> int:
        """Return the number of leaves."""
        return len(self._leaves)

    @property
    def depth(self) -> int:
        """Return tree depth."""
        return len(self._tree)

    def _build(self) -> None:
        """Construct the Merkle tree bottom-up."""
        if not self._leaves:
            self._root = GENESIS_HASH
            self._tree = []
            return

        current_level = list(self._leaves)
        self._tree = [current_level]

        while len(current_level) > 1:
            next_level: list[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(_hash_pair(left, right))
            self._tree.append(next_level)
            current_level = next_level

        self._root = current_level[0] if current_level else GENESIS_HASH

    def get_proof(self, leaf_index: int) -> list[tuple[str, str]]:
        """Generate inclusion proof for a leaf.

        Parameters
        ----------
        leaf_index : int
            Index of the leaf to prove.

        Returns
        -------
        list[tuple[str, str]]
            List of (sibling_hash, side) pairs, where side is
            "left" or "right".
        """
        if leaf_index < 0 or leaf_index >= len(self._leaves):
            msg = f"Leaf index {leaf_index} out of range [0, {len(self._leaves)})"
            raise IndexError(msg)

        proof: list[tuple[str, str]] = []
        idx = leaf_index

        for level in self._tree[:-1]:
            if idx % 2 == 0:
                sibling_idx = idx + 1
                side = "right"
            else:
                sibling_idx = idx - 1
                side = "left"

            if sibling_idx < len(level):
                proof.append((level[sibling_idx], side))
            else:
                proof.append((level[idx], "right"))

            idx //= 2

        return proof

    @staticmethod
    def verify_proof(
        leaf_hash: str,
        proof: list[tuple[str, str]],
        expected_root: str,
    ) -> bool:
        """Verify a Merkle inclusion proof.

        Parameters
        ----------
        leaf_hash : str
            Hash of the leaf being verified.
        proof : list[tuple[str, str]]
            Proof path from get_proof().
        expected_root : str
            Expected Merkle root.

        Returns
        -------
        bool
            True if the proof is valid.
        """
        current = leaf_hash
        for sibling_hash, side in proof:
            if side == "left":
                current = _hash_pair(sibling_hash, current)
            else:
                current = _hash_pair(current, sibling_hash)
        return current == expected_root


class MerkleCheckpoint(NamedTuple):
    """Periodic checkpoint of Merkle tree state."""

    checkpoint_id: str
    sequence_range: tuple[int, int]
    merkle_root: str
    leaf_count: int
    tree_depth: int
    created_at: str


# ═══════════════════════════════════════════════════════════════════════════
# Audit Log
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AuditLog:
    """Hash-chained, Merkle-checkpointed audit log.

    Provides an immutable record of all data acquisition operations
    with cryptographic tamper-detection guarantees.

    Parameters
    ----------
    log_id : str
        Unique identifier for this log instance.
    checkpoint_interval : int
        Number of entries between Merkle checkpoints.

    Attributes
    ----------
    entries : list[AuditEntry]
        All recorded audit entries.
    checkpoints : list[MerkleCheckpoint]
        Periodic Merkle tree checkpoints.
    """

    log_id: str = field(
        default_factory=lambda: f"AUDIT-{uuid.uuid4().hex[:12].upper()}"
    )
    checkpoint_interval: int = CHECKPOINT_INTERVAL
    entries: list[AuditEntry] = field(default_factory=list)
    checkpoints: list[MerkleCheckpoint] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _last_checkpoint_seq: int = field(default=0, init=False)

    def record(
        self,
        event_type: AuditEventType,
        actor: str,
        resource: str,
        action_detail: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record a new audit event.

        Parameters
        ----------
        event_type : AuditEventType
            Classification of the event.
        actor : str
            Person or system performing the action.
        resource : str
            Data resource affected.
        action_detail : str
            Human-readable description.
        metadata : dict[str, Any] | None
            Additional structured data.

        Returns
        -------
        AuditEntry
            The recorded entry.
        """
        with self._lock:
            seq = len(self.entries)
            previous_hash = (
                self.entries[-1].entry_hash if self.entries else GENESIS_HASH
            )

            entry_id = f"{self.log_id}-{seq:06d}"
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            meta = metadata or {}

            entry_hash = _compute_entry_hash(
                entry_id=entry_id,
                sequence_number=seq,
                timestamp=timestamp,
                event_type=event_type.value,
                actor=actor,
                resource=resource,
                action_detail=action_detail,
                metadata=meta,
                previous_hash=previous_hash,
            )

            entry = AuditEntry(
                entry_id=entry_id,
                sequence_number=seq,
                timestamp=timestamp,
                event_type=event_type.value,
                actor=actor,
                resource=resource,
                action_detail=action_detail,
                metadata=meta,
                previous_hash=previous_hash,
                entry_hash=entry_hash,
            )
            self.entries.append(entry)

            # Periodic Merkle checkpoint
            entries_since = seq - self._last_checkpoint_seq + 1
            if entries_since >= self.checkpoint_interval:
                self._create_checkpoint()

            return entry

    def verify_chain(self) -> tuple[IntegrityStatus, list[int]]:
        """Verify the entire hash chain.

        Returns
        -------
        tuple[IntegrityStatus, list[int]]
            Status and list of tampered entry sequence numbers.
        """
        if not self.entries:
            return IntegrityStatus.VALID, []

        tampered: list[int] = []

        for i, entry in enumerate(self.entries):
            expected_prev = (
                self.entries[i - 1].entry_hash if i > 0 else GENESIS_HASH
            )
            if entry.previous_hash != expected_prev:
                tampered.append(entry.sequence_number)
                continue

            recomputed = _compute_entry_hash(
                entry_id=entry.entry_id,
                sequence_number=entry.sequence_number,
                timestamp=entry.timestamp,
                event_type=entry.event_type,
                actor=entry.actor,
                resource=entry.resource,
                action_detail=entry.action_detail,
                metadata=entry.metadata,
                previous_hash=entry.previous_hash,
            )
            if recomputed != entry.entry_hash:
                tampered.append(entry.sequence_number)

        if tampered:
            return IntegrityStatus.TAMPERED, tampered
        return IntegrityStatus.VALID, []

    def verify_entry(self, sequence_number: int) -> bool:
        """Verify a single entry's hash.

        Parameters
        ----------
        sequence_number : int
            Sequence number of the entry to verify.

        Returns
        -------
        bool
            True if the entry hash is valid.
        """
        if sequence_number < 0 or sequence_number >= len(self.entries):
            return False

        entry = self.entries[sequence_number]
        recomputed = _compute_entry_hash(
            entry_id=entry.entry_id,
            sequence_number=entry.sequence_number,
            timestamp=entry.timestamp,
            event_type=entry.event_type,
            actor=entry.actor,
            resource=entry.resource,
            action_detail=entry.action_detail,
            metadata=entry.metadata,
            previous_hash=entry.previous_hash,
        )
        return recomputed == entry.entry_hash

    def verify_checkpoint(self, checkpoint_index: int) -> bool:
        """Verify a Merkle checkpoint.

        Parameters
        ----------
        checkpoint_index : int
            Index into the checkpoints list.

        Returns
        -------
        bool
            True if the Merkle root matches recomputed tree.
        """
        if checkpoint_index < 0 or checkpoint_index >= len(self.checkpoints):
            return False

        cp = self.checkpoints[checkpoint_index]
        start, end = cp.sequence_range

        if end > len(self.entries):
            return False

        leaf_hashes = [
            self.entries[i].entry_hash for i in range(start, end)
        ]
        tree = MerkleTree(leaf_hashes)
        return tree.root == cp.merkle_root

    def get_merkle_proof(
        self, sequence_number: int
    ) -> tuple[MerkleCheckpoint, list[tuple[str, str]]] | None:
        """Generate Merkle inclusion proof for an entry.

        Parameters
        ----------
        sequence_number : int
            Entry sequence number.

        Returns
        -------
        tuple or None
            (checkpoint, proof_path) or None if not in any checkpoint.
        """
        for cp in self.checkpoints:
            start, end = cp.sequence_range
            if start <= sequence_number < end:
                leaf_hashes = [
                    self.entries[i].entry_hash for i in range(start, end)
                ]
                tree = MerkleTree(leaf_hashes)
                local_idx = sequence_number - start
                proof = tree.get_proof(local_idx)
                return cp, proof
        return None

    def _create_checkpoint(self) -> None:
        """Create a Merkle tree checkpoint over recent entries."""
        start = self._last_checkpoint_seq
        end = len(self.entries)

        if start >= end:
            return

        leaf_hashes = [
            self.entries[i].entry_hash for i in range(start, end)
        ]
        tree = MerkleTree(leaf_hashes)

        checkpoint = MerkleCheckpoint(
            checkpoint_id=f"CP-{uuid.uuid4().hex[:8].upper()}",
            sequence_range=(start, end),
            merkle_root=tree.root,
            leaf_count=tree.leaf_count,
            tree_depth=tree.depth,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self.checkpoints.append(checkpoint)
        self._last_checkpoint_seq = end

        logger.info(
            "Merkle checkpoint created: entries [%d, %d), root=%s",
            start,
            end,
            tree.root[:16],
        )

    def filter_by_type(
        self, event_type: AuditEventType
    ) -> list[AuditEntry]:
        """Filter entries by event type."""
        return [
            e for e in self.entries if e.event_type == event_type.value
        ]

    def filter_by_resource(self, resource: str) -> list[AuditEntry]:
        """Filter entries by resource identifier."""
        return [e for e in self.entries if e.resource == resource]

    def filter_by_actor(self, actor: str) -> list[AuditEntry]:
        """Filter entries by actor."""
        return [e for e in self.entries if e.actor == actor]

    def filter_by_time_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[AuditEntry]:
        """Filter entries by time range."""
        start_iso = start.isoformat()
        end_iso = end.isoformat()
        return [
            e for e in self.entries
            if start_iso <= e.timestamp <= end_iso
        ]

    def export_json(self, output_path: Path) -> None:
        """Export the full audit log to JSON.

        Parameters
        ----------
        output_path : Path
            Destination file path.
        """
        data = {
            "log_id": self.log_id,
            "entry_count": len(self.entries),
            "checkpoint_count": len(self.checkpoints),
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "sequence_number": e.sequence_number,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "actor": e.actor,
                    "resource": e.resource,
                    "action_detail": e.action_detail,
                    "metadata": e.metadata,
                    "previous_hash": e.previous_hash,
                    "entry_hash": e.entry_hash,
                }
                for e in self.entries
            ],
            "checkpoints": [
                {
                    "checkpoint_id": cp.checkpoint_id,
                    "sequence_range": list(cp.sequence_range),
                    "merkle_root": cp.merkle_root,
                    "leaf_count": cp.leaf_count,
                    "tree_depth": cp.tree_depth,
                    "created_at": cp.created_at,
                }
                for cp in self.checkpoints
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_json(self, input_path: Path) -> None:
        """Import an audit log from JSON and verify integrity.

        Parameters
        ----------
        input_path : Path
            Source JSON file.

        Raises
        ------
        ValueError
            If the imported log fails integrity verification.
        """
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.log_id = data.get("log_id", self.log_id)
        self.entries = [
            AuditEntry(
                entry_id=e["entry_id"],
                sequence_number=e["sequence_number"],
                timestamp=e["timestamp"],
                event_type=e["event_type"],
                actor=e["actor"],
                resource=e["resource"],
                action_detail=e["action_detail"],
                metadata=e.get("metadata", {}),
                previous_hash=e["previous_hash"],
                entry_hash=e["entry_hash"],
            )
            for e in data.get("entries", [])
        ]
        self.checkpoints = [
            MerkleCheckpoint(
                checkpoint_id=cp["checkpoint_id"],
                sequence_range=tuple(cp["sequence_range"]),
                merkle_root=cp["merkle_root"],
                leaf_count=cp["leaf_count"],
                tree_depth=cp["tree_depth"],
                created_at=cp["created_at"],
            )
            for cp in data.get("checkpoints", [])
        ]

        # Verify imported data integrity
        status, tampered = self.verify_chain()
        if status == IntegrityStatus.TAMPERED:
            msg = (
                f"Imported audit log failed integrity verification: "
                f"{len(tampered)} tampered entries"
            )
            raise ValueError(msg)

    def generate_summary(self) -> dict[str, Any]:
        """Generate audit log summary statistics.

        Returns
        -------
        dict[str, Any]
            Summary including event type counts and integrity status.
        """
        status, tampered = self.verify_chain()

        event_counts: dict[str, int] = {}
        actor_counts: dict[str, int] = {}
        resource_counts: dict[str, int] = {}

        for entry in self.entries:
            event_counts[entry.event_type] = (
                event_counts.get(entry.event_type, 0) + 1
            )
            actor_counts[entry.actor] = actor_counts.get(entry.actor, 0) + 1
            resource_counts[entry.resource] = (
                resource_counts.get(entry.resource, 0) + 1
            )

        return {
            "log_id": self.log_id,
            "total_entries": len(self.entries),
            "total_checkpoints": len(self.checkpoints),
            "integrity_status": status.value,
            "tampered_entries": len(tampered),
            "event_type_distribution": event_counts,
            "actor_distribution": actor_counts,
            "resource_distribution": resource_counts,
            "first_entry_time": (
                self.entries[0].timestamp if self.entries else None
            ),
            "last_entry_time": (
                self.entries[-1].timestamp if self.entries else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Provenance Tracker
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class DataProvenance:
    """Tracks the full provenance chain of a data asset.

    Records source, transformations, and custody changes throughout
    the data lifecycle.

    Attributes
    ----------
    asset_id : str
        Unique data asset identifier.
    source : str
        Origin of the data (e.g., "CDC SFTP").
    created_at : str
        Asset creation timestamp.
    custody_chain : list[dict[str, Any]]
        Ordered list of custody events.
    transformations : list[dict[str, Any]]
        Ordered list of data transformations.
    current_custodian : str
        Person or system currently holding the data.
    integrity_hash : str
        Current integrity hash of the data.
    """

    asset_id: str = field(
        default_factory=lambda: f"ASSET-{uuid.uuid4().hex[:10].upper()}"
    )
    source: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    custody_chain: list[dict[str, Any]] = field(default_factory=list)
    transformations: list[dict[str, Any]] = field(default_factory=list)
    current_custodian: str = ""
    integrity_hash: str = ""

    def record_custody_transfer(
        self,
        from_custodian: str,
        to_custodian: str,
        reason: str,
        integrity_hash: str | None = None,
    ) -> None:
        """Record a custody transfer event.

        Parameters
        ----------
        from_custodian : str
            Releasing party.
        to_custodian : str
            Receiving party.
        reason : str
            Reason for transfer.
        integrity_hash : str | None
            Hash of data at transfer time.
        """
        event = {
            "event_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "from": from_custodian,
            "to": to_custodian,
            "reason": reason,
            "integrity_hash": integrity_hash or self.integrity_hash,
        }
        self.custody_chain.append(event)
        self.current_custodian = to_custodian
        if integrity_hash:
            self.integrity_hash = integrity_hash

    def record_transformation(
        self,
        transformation_type: str,
        description: str,
        input_hash: str,
        output_hash: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Record a data transformation event.

        Parameters
        ----------
        transformation_type : str
            Type of transformation (e.g., "de-identification").
        description : str
            Human-readable description.
        input_hash : str
            Hash of data before transformation.
        output_hash : str
            Hash of data after transformation.
        parameters : dict[str, Any] | None
            Transformation parameters.
        """
        event = {
            "event_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "type": transformation_type,
            "description": description,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "parameters": parameters or {},
        }
        self.transformations.append(event)
        self.integrity_hash = output_hash

    def to_dict(self) -> dict[str, Any]:
        """Serialise provenance record."""
        return {
            "asset_id": self.asset_id,
            "source": self.source,
            "created_at": self.created_at,
            "current_custodian": self.current_custodian,
            "integrity_hash": self.integrity_hash,
            "custody_events": len(self.custody_chain),
            "transformation_events": len(self.transformations),
            "custody_chain": self.custody_chain,
            "transformations": self.transformations,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════

def create_audit_log(
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> AuditLog:
    """Create a new audit log instance.

    Parameters
    ----------
    checkpoint_interval : int
        Entries between Merkle checkpoints.

    Returns
    -------
    AuditLog
        Configured audit log.
    """
    return AuditLog(checkpoint_interval=checkpoint_interval)


def create_provenance_tracker(
    source: str,
    initial_custodian: str,
    initial_hash: str = "",
) -> DataProvenance:
    """Create a new data provenance tracker.

    Parameters
    ----------
    source : str
        Origin of the data.
    initial_custodian : str
        Initial data custodian.
    initial_hash : str
        Initial integrity hash.

    Returns
    -------
    DataProvenance
        Configured provenance tracker.
    """
    return DataProvenance(
        source=source,
        current_custodian=initial_custodian,
        integrity_hash=initial_hash,
    )
