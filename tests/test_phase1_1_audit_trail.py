"""
Phase 1.1 Audit Trail Module — Comprehensive Test Suite.

Tests cover:
  - Hash-chained audit entry recording
  - Chain integrity verification and tamper detection
  - Merkle tree construction, proofs, and verification
  - Merkle checkpoint creation and validation
  - AuditLog filtering (by type, resource, actor, time range)
  - JSON export/import with integrity verification
  - DataProvenance custody chain and transformation tracking
  - Factory functions
  - Summary generation
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ml.data.audit_trail import (
    GENESIS_HASH,
    AuditEntry,
    AuditEventType,
    AuditLog,
    DataProvenance,
    IntegrityStatus,
    MerkleCheckpoint,
    MerkleTree,
    _compute_entry_hash,
    _hash_pair,
    create_audit_log,
    create_provenance_tracker,
)


# ═══════════════════════════════════════════════════════════════════════════
# Hash Function Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHashFunctions:
    """Low-level hash computation tests."""

    def test_compute_entry_hash_deterministic(self) -> None:
        h1 = _compute_entry_hash(
            entry_id="E-001",
            sequence_number=0,
            timestamp="2025-01-01T00:00:00+00:00",
            event_type="data_received",
            actor="system",
            resource="test.csv",
            action_detail="File received",
            metadata={},
            previous_hash=GENESIS_HASH,
        )
        h2 = _compute_entry_hash(
            entry_id="E-001",
            sequence_number=0,
            timestamp="2025-01-01T00:00:00+00:00",
            event_type="data_received",
            actor="system",
            resource="test.csv",
            action_detail="File received",
            metadata={},
            previous_hash=GENESIS_HASH,
        )
        assert h1 == h2
        assert len(h1) == 64

    def test_different_input_different_hash(self) -> None:
        h1 = _compute_entry_hash("E-001", 0, "t1", "e", "a", "r", "d", {}, GENESIS_HASH)
        h2 = _compute_entry_hash("E-002", 0, "t1", "e", "a", "r", "d", {}, GENESIS_HASH)
        assert h1 != h2

    def test_hash_pair(self) -> None:
        result = _hash_pair("abc", "def")
        assert len(result) == 64
        assert _hash_pair("abc", "def") == _hash_pair("abc", "def")

    def test_hash_pair_order_matters(self) -> None:
        assert _hash_pair("abc", "def") != _hash_pair("def", "abc")


# ═══════════════════════════════════════════════════════════════════════════
# AuditEntry Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditEntry:
    """AuditEntry NamedTuple tests."""

    def test_entry_is_namedtuple(self) -> None:
        entry = AuditEntry(
            entry_id="E-001",
            sequence_number=0,
            timestamp="2025-01-01T00:00:00+00:00",
            event_type="data_received",
            actor="system",
            resource="test.csv",
            action_detail="File received",
            metadata={},
            previous_hash=GENESIS_HASH,
            entry_hash="a" * 64,
        )
        assert entry.entry_id == "E-001"
        assert entry.sequence_number == 0

    def test_entry_immutable(self) -> None:
        entry = AuditEntry("E", 0, "t", "e", "a", "r", "d", {}, "p", "h")
        with pytest.raises(AttributeError):
            entry.actor = "modified"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# AuditEventType Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditEventType:
    """Audit event classification enum tests."""

    def test_data_lifecycle_events(self) -> None:
        assert AuditEventType.DATA_RECEIVED.value == "data_received"
        assert AuditEventType.DATA_TRANSFERRED.value == "data_transferred"
        assert AuditEventType.DATA_DELETED.value == "data_deleted"

    def test_security_events(self) -> None:
        assert AuditEventType.ENCRYPTION_APPLIED.value == "encryption_applied"
        assert AuditEventType.INTEGRITY_VIOLATION.value == "integrity_violation"

    def test_total_event_types(self) -> None:
        # 6 data-lifecycle + 3 access + 3 compliance + 5 security + 3 system
        assert len(AuditEventType) == 20


# ═══════════════════════════════════════════════════════════════════════════
# Merkle Tree Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMerkleTree:
    """Binary Merkle tree construction and proof verification."""

    def test_empty_tree(self) -> None:
        tree = MerkleTree([])
        assert tree.root == GENESIS_HASH
        assert tree.leaf_count == 0

    def test_single_leaf(self) -> None:
        tree = MerkleTree(["abcdef" * 10 + "abcd"])
        assert tree.leaf_count == 1
        assert tree.root != GENESIS_HASH

    def test_two_leaves(self) -> None:
        h1 = "a" * 64
        h2 = "b" * 64
        tree = MerkleTree([h1, h2])
        assert tree.leaf_count == 2
        assert tree.root == _hash_pair(h1, h2)

    def test_power_of_two_leaves(self) -> None:
        leaves = [f"{i:064x}" for i in range(8)]
        tree = MerkleTree(leaves)
        assert tree.leaf_count == 8
        assert tree.depth == 4  # 3 inner levels + leaf level

    def test_odd_number_of_leaves(self) -> None:
        leaves = [f"{i:064x}" for i in range(5)]
        tree = MerkleTree(leaves)
        assert tree.leaf_count == 5
        assert tree.root != GENESIS_HASH

    def test_proof_for_first_leaf(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(0)
        assert len(proof) > 0
        assert MerkleTree.verify_proof(leaves[0], proof, tree.root) is True

    def test_proof_for_last_leaf(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(3)
        assert MerkleTree.verify_proof(leaves[3], proof, tree.root) is True

    def test_proof_for_all_leaves(self) -> None:
        leaves = [f"{i:064x}" for i in range(8)]
        tree = MerkleTree(leaves)
        for i in range(8):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(leaves[i], proof, tree.root) is True

    def test_proof_invalid_leaf_rejected(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(0)
        fake_leaf = "f" * 64
        assert MerkleTree.verify_proof(fake_leaf, proof, tree.root) is False

    def test_proof_wrong_root_rejected(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(0)
        assert MerkleTree.verify_proof(leaves[0], proof, "0" * 64) is False

    def test_proof_index_out_of_range(self) -> None:
        leaves = [f"{i:064x}" for i in range(4)]
        tree = MerkleTree(leaves)
        with pytest.raises(IndexError):
            tree.get_proof(4)
        with pytest.raises(IndexError):
            tree.get_proof(-1)

    def test_depth_property(self) -> None:
        tree = MerkleTree(["a" * 64])
        assert tree.depth == 1


# ═══════════════════════════════════════════════════════════════════════════
# AuditLog Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditLog:
    """Hash-chained audit log tests."""

    def test_record_creates_entry(self) -> None:
        log = create_audit_log()
        entry = log.record(
            event_type=AuditEventType.DATA_RECEIVED,
            actor="system",
            resource="test.csv",
            action_detail="File ingested",
        )
        assert entry.sequence_number == 0
        assert entry.event_type == "data_received"
        assert entry.previous_hash == GENESIS_HASH
        assert len(entry.entry_hash) == 64

    def test_chain_links(self) -> None:
        log = create_audit_log()
        e1 = log.record(AuditEventType.DATA_RECEIVED, "sys", "a.csv", "Received")
        e2 = log.record(AuditEventType.DATA_VERIFIED, "sys", "a.csv", "Verified")
        assert e2.previous_hash == e1.entry_hash

    def test_verify_chain_valid(self) -> None:
        log = create_audit_log()
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"file_{i}", "Test")
        status, tampered = log.verify_chain()
        assert status == IntegrityStatus.VALID
        assert tampered == []

    def test_verify_chain_empty(self) -> None:
        log = create_audit_log()
        status, tampered = log.verify_chain()
        assert status == IntegrityStatus.VALID

    def test_verify_chain_detects_tamper(self) -> None:
        log = create_audit_log()
        for i in range(5):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"file_{i}", "Test")

        # Tamper with entry 2
        original = log.entries[2]
        tampered_entry = AuditEntry(
            entry_id=original.entry_id,
            sequence_number=original.sequence_number,
            timestamp=original.timestamp,
            event_type=original.event_type,
            actor="ATTACKER",
            resource=original.resource,
            action_detail=original.action_detail,
            metadata=original.metadata,
            previous_hash=original.previous_hash,
            entry_hash=original.entry_hash,  # hash no longer matches
        )
        log.entries[2] = tampered_entry

        status, tampered_list = log.verify_chain()
        assert status == IntegrityStatus.TAMPERED
        assert 2 in tampered_list

    def test_verify_entry(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "a.csv", "Test")
        assert log.verify_entry(0) is True

    def test_verify_entry_invalid_index(self) -> None:
        log = create_audit_log()
        assert log.verify_entry(0) is False
        assert log.verify_entry(-1) is False

    def test_checkpoint_creation(self) -> None:
        log = create_audit_log(checkpoint_interval=5)
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")
        assert len(log.checkpoints) >= 1

    def test_verify_checkpoint(self) -> None:
        log = create_audit_log(checkpoint_interval=5)
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")
        assert log.verify_checkpoint(0) is True

    def test_verify_checkpoint_invalid_index(self) -> None:
        log = create_audit_log()
        assert log.verify_checkpoint(0) is False
        assert log.verify_checkpoint(-1) is False

    def test_get_merkle_proof(self) -> None:
        log = create_audit_log(checkpoint_interval=5)
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")
        result = log.get_merkle_proof(2)
        assert result is not None
        checkpoint, proof = result
        leaf_hash = log.entries[2].entry_hash
        assert MerkleTree.verify_proof(leaf_hash, proof, checkpoint.merkle_root)

    def test_get_merkle_proof_no_checkpoint(self) -> None:
        log = create_audit_log(checkpoint_interval=1000)
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")
        assert log.get_merkle_proof(0) is None

    def test_filter_by_type(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "a", "Received")
        log.record(AuditEventType.DATA_VERIFIED, "sys", "a", "Verified")
        log.record(AuditEventType.DATA_RECEIVED, "sys", "b", "Received")
        filtered = log.filter_by_type(AuditEventType.DATA_RECEIVED)
        assert len(filtered) == 2

    def test_filter_by_resource(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "file_a.csv", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "sys", "file_b.csv", "Test")
        filtered = log.filter_by_resource("file_a.csv")
        assert len(filtered) == 1

    def test_filter_by_actor(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "alice", "f", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "bob", "f", "Test")
        assert len(log.filter_by_actor("alice")) == 1

    def test_filter_by_time_range(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")
        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(minutes=1)
        end = now + timedelta(minutes=1)
        filtered = log.filter_by_time_range(start, end)
        assert len(filtered) == 1

    def test_record_with_metadata(self) -> None:
        log = create_audit_log()
        entry = log.record(
            AuditEventType.ENCRYPTION_APPLIED,
            actor="system",
            resource="data.bin",
            action_detail="AES-256-GCM applied",
            metadata={"algorithm": "AES-256-GCM", "key_id": "K-001"},
        )
        assert entry.metadata["algorithm"] == "AES-256-GCM"


# ═══════════════════════════════════════════════════════════════════════════
# JSON Export / Import Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditLogPersistence:
    """JSON serialisation and integrity-verified import tests."""

    def test_export_import_roundtrip(self) -> None:
        log = create_audit_log(checkpoint_interval=5)
        for i in range(10):
            log.record(AuditEventType.DATA_RECEIVED, "sys", f"f_{i}", "Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json"
            log.export_json(path)
            assert path.exists()

            log2 = AuditLog()
            log2.import_json(path)
            assert len(log2.entries) == 10
            assert log2.log_id == log.log_id

    def test_import_verifies_integrity(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json"
            log.export_json(path)

            # Tamper with JSON
            with path.open("r") as f:
                data = json.load(f)
            data["entries"][0]["actor"] = "TAMPERED"
            with path.open("w") as f:
                json.dump(data, f)

            log2 = AuditLog()
            with pytest.raises(ValueError, match="integrity verification"):
                log2.import_json(path)

    def test_export_creates_directories(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "sys", "f", "Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "audit.json"
            log.export_json(path)
            assert path.exists()


# ═══════════════════════════════════════════════════════════════════════════
# Summary Generation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditLogSummary:
    """Audit log summary statistics tests."""

    def test_summary_structure(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.DATA_RECEIVED, "alice", "f1", "Test")
        log.record(AuditEventType.DATA_VERIFIED, "bob", "f2", "Test")
        log.record(AuditEventType.DATA_RECEIVED, "alice", "f1", "Test")

        summary = log.generate_summary()
        assert summary["total_entries"] == 3
        assert summary["integrity_status"] == "valid"
        assert summary["event_type_distribution"]["data_received"] == 2
        assert summary["actor_distribution"]["alice"] == 2
        assert summary["resource_distribution"]["f1"] == 2

    def test_summary_empty_log(self) -> None:
        log = create_audit_log()
        summary = log.generate_summary()
        assert summary["total_entries"] == 0
        assert summary["first_entry_time"] is None
        assert summary["last_entry_time"] is None

    def test_summary_timestamps(self) -> None:
        log = create_audit_log()
        log.record(AuditEventType.SESSION_START, "sys", "session", "Start")
        log.record(AuditEventType.SESSION_END, "sys", "session", "End")
        summary = log.generate_summary()
        assert summary["first_entry_time"] is not None
        assert summary["last_entry_time"] is not None


# ═══════════════════════════════════════════════════════════════════════════
# DataProvenance Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDataProvenance:
    """Data provenance custody chain and transformation tracking."""

    def test_factory_creates_provenance(self) -> None:
        prov = create_provenance_tracker(
            source="CDC SFTP",
            initial_custodian="system",
            initial_hash="abc123",
        )
        assert prov.source == "CDC SFTP"
        assert prov.current_custodian == "system"
        assert prov.integrity_hash == "abc123"

    def test_custody_transfer(self) -> None:
        prov = DataProvenance(source="CDC SFTP", current_custodian="system")
        prov.record_custody_transfer(
            from_custodian="system",
            to_custodian="researcher",
            reason="Data released for analysis",
            integrity_hash="hash_at_transfer",
        )
        assert prov.current_custodian == "researcher"
        assert len(prov.custody_chain) == 1
        assert prov.integrity_hash == "hash_at_transfer"

    def test_multiple_transfers(self) -> None:
        prov = DataProvenance(source="CDC", current_custodian="sys")
        prov.record_custody_transfer("sys", "alice", "Analysis", "h1")
        prov.record_custody_transfer("alice", "bob", "Review", "h2")
        prov.record_custody_transfer("bob", "archive", "Archival", "h3")
        assert len(prov.custody_chain) == 3
        assert prov.current_custodian == "archive"
        assert prov.integrity_hash == "h3"

    def test_custody_transfer_preserves_hash_when_none(self) -> None:
        prov = DataProvenance(
            source="CDC", current_custodian="sys", integrity_hash="original"
        )
        prov.record_custody_transfer("sys", "alice", "Transfer")
        # integrity_hash should remain "original" since none provided
        assert prov.custody_chain[-1]["integrity_hash"] == "original"

    def test_record_transformation(self) -> None:
        prov = DataProvenance(source="CDC", integrity_hash="input_hash")
        prov.record_transformation(
            transformation_type="de-identification",
            description="Safe Harbor + k-anonymity applied",
            input_hash="input_hash",
            output_hash="output_hash",
            parameters={"k": 5, "method": "safe_harbor"},
        )
        assert len(prov.transformations) == 1
        assert prov.integrity_hash == "output_hash"
        assert prov.transformations[0]["type"] == "de-identification"

    def test_multiple_transformations(self) -> None:
        prov = DataProvenance(source="CDC", integrity_hash="h0")
        prov.record_transformation("deidentify", "Safe Harbor", "h0", "h1")
        prov.record_transformation("anonymize", "k-Anonymity", "h1", "h2")
        prov.record_transformation("perturb", "Differential Privacy", "h2", "h3")
        assert len(prov.transformations) == 3
        assert prov.integrity_hash == "h3"

    def test_to_dict(self) -> None:
        prov = create_provenance_tracker("CDC SFTP", "system", "h0")
        prov.record_custody_transfer("system", "researcher", "Release", "h1")
        prov.record_transformation("deidentify", "Applied", "h0", "h1")

        d = prov.to_dict()
        assert d["source"] == "CDC SFTP"
        assert d["custody_events"] == 1
        assert d["transformation_events"] == 1
        assert "custody_chain" in d
        assert "transformations" in d

    def test_asset_id_auto_generated(self) -> None:
        prov = DataProvenance()
        assert prov.asset_id.startswith("ASSET-")

    def test_created_at_auto_set(self) -> None:
        prov = DataProvenance()
        assert prov.created_at != ""


# ═══════════════════════════════════════════════════════════════════════════
# IntegrityStatus Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrityStatus:
    """Integrity status enum tests."""

    def test_values(self) -> None:
        assert IntegrityStatus.VALID.value == "valid"
        assert IntegrityStatus.TAMPERED.value == "tampered"
        assert IntegrityStatus.INCOMPLETE.value == "incomplete"
        assert IntegrityStatus.UNKNOWN.value == "unknown"
        assert len(IntegrityStatus) == 4


# ═══════════════════════════════════════════════════════════════════════════
# MerkleCheckpoint Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMerkleCheckpoint:
    """MerkleCheckpoint NamedTuple tests."""

    def test_checkpoint_fields(self) -> None:
        cp = MerkleCheckpoint(
            checkpoint_id="CP-001",
            sequence_range=(0, 10),
            merkle_root="a" * 64,
            leaf_count=10,
            tree_depth=4,
            created_at="2025-01-01T00:00:00+00:00",
        )
        assert cp.checkpoint_id == "CP-001"
        assert cp.sequence_range == (0, 10)
        assert cp.leaf_count == 10

    def test_checkpoint_immutable(self) -> None:
        cp = MerkleCheckpoint("id", (0, 5), "root", 5, 3, "ts")
        with pytest.raises(AttributeError):
            cp.merkle_root = "modified"  # type: ignore[misc]
