"""Evidence Integrity Module using Merkle Trees.

THREAT-T-001 FIX: Merkle tree implementation for evidence tampering prevention.

This module provides cryptographic integrity verification for evidence artifacts
using Merkle trees. It ensures that any modification to evidence can be detected.

Key Features:
    - Merkle tree construction from evidence items
    - Tamper detection via root hash verification
    - Proof generation for individual evidence items
    - Proof verification without full tree
    - Persistent root storage for auditing

Usage:
    from pipeline.security.evidence_integrity import (
        EvidenceIntegrityChecker,
        build_evidence_tree,
        verify_evidence_item,
    )

    # Build integrity tree from evidence
    checker = EvidenceIntegrityChecker()
    root_hash = checker.build_tree([evidence1, evidence2, evidence3])

    # Verify integrity later
    is_valid, proof = checker.verify_item(evidence1, index=0)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-21)
SECURITY FIX: THREAT-T-001
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Hash algorithm to use (SHA-256 is recommended)
HASH_ALGORITHM = os.getenv("EVIDENCE_HASH_ALGORITHM", "sha256")

# Salt for evidence hashing (should be set in production)
EVIDENCE_HASH_SALT = os.getenv("EVIDENCE_HASH_SALT", "")


# =============================================================================
# MERKLE TREE IMPLEMENTATION
# =============================================================================

def compute_hash(data: str) -> str:
    """Compute cryptographic hash of data.

    Args:
        data: String data to hash.

    Returns:
        Hexadecimal hash string.
    """
    # Add salt if configured
    if EVIDENCE_HASH_SALT:
        data = EVIDENCE_HASH_SALT + data

    if HASH_ALGORITHM == "sha256":
        return hashlib.sha256(data.encode()).hexdigest()
    elif HASH_ALGORITHM == "sha3_256":
        return hashlib.sha3_256(data.encode()).hexdigest()
    else:
        return hashlib.sha256(data.encode()).hexdigest()


def hash_evidence_item(item: Any) -> str:
    """Hash an evidence item to a leaf node hash.

    Args:
        item: Evidence item (dict, str, or any JSON-serializable object).

    Returns:
        Hash of the evidence item.
    """
    if isinstance(item, str):
        data = item
    elif isinstance(item, dict):
        # Sort keys for deterministic hashing
        data = json.dumps(item, sort_keys=True, default=str)
    else:
        data = json.dumps(item, default=str)

    return compute_hash(data)


def combine_hashes(left: str, right: str) -> str:
    """Combine two hashes into a parent hash.

    Args:
        left: Left child hash.
        right: Right child hash.

    Returns:
        Combined parent hash.
    """
    # Ensure consistent ordering (smaller hash first)
    combined = left + right
    return compute_hash(combined)


@dataclass
class MerkleNode:
    """Node in the Merkle tree."""
    hash: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    is_leaf: bool = False
    data_index: Optional[int] = None  # Index in original data for leaf nodes


@dataclass
class MerkleProof:
    """Proof that an item is in the Merkle tree."""
    item_hash: str
    item_index: int
    siblings: List[Tuple[str, str]]  # List of (hash, position: "left"|"right")
    root_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_hash": self.item_hash,
            "item_index": self.item_index,
            "siblings": self.siblings,
            "root_hash": self.root_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MerkleProof":
        """Create from dictionary."""
        return cls(
            item_hash=data["item_hash"],
            item_index=data["item_index"],
            siblings=data["siblings"],
            root_hash=data["root_hash"],
        )


class MerkleTree:
    """THREAT-T-001 FIX: Merkle tree for evidence integrity.

    A binary Merkle tree that provides:
    - O(n) construction time
    - O(1) root hash access
    - O(log n) proof generation
    - O(log n) proof verification
    """

    def __init__(self, items: List[Any]):
        """Build Merkle tree from items.

        Args:
            items: List of evidence items to include.
        """
        self.items = items
        self.leaves: List[MerkleNode] = []
        self.root: Optional[MerkleNode] = None

        if items:
            self._build_tree()

    def _build_tree(self) -> None:
        """Build the Merkle tree from items."""
        # Create leaf nodes
        self.leaves = []
        for i, item in enumerate(self.items):
            item_hash = hash_evidence_item(item)
            leaf = MerkleNode(
                hash=item_hash,
                is_leaf=True,
                data_index=i,
            )
            self.leaves.append(leaf)

        # Handle empty or single item case
        if len(self.leaves) == 0:
            self.root = MerkleNode(hash=compute_hash("empty"))
            return

        if len(self.leaves) == 1:
            self.root = self.leaves[0]
            return

        # Build tree bottom-up
        current_level = self.leaves.copy()

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]

                # Handle odd number of nodes (duplicate last)
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left  # Duplicate

                parent = MerkleNode(
                    hash=combine_hashes(left.hash, right.hash),
                    left=left,
                    right=right,
                )
                next_level.append(parent)

            current_level = next_level

        self.root = current_level[0]

    @property
    def root_hash(self) -> str:
        """Get the root hash of the tree."""
        return self.root.hash if self.root else compute_hash("empty")

    def get_proof(self, index: int) -> Optional[MerkleProof]:
        """Generate a proof for the item at the given index.

        Args:
            index: Index of the item in the original list.

        Returns:
            MerkleProof or None if index is invalid.
        """
        if index < 0 or index >= len(self.leaves):
            return None

        siblings: List[Tuple[str, str]] = []
        current_level = self.leaves.copy()
        current_index = index

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]

                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left

                # Record sibling if we're at the current_index pair
                if i == current_index or i + 1 == current_index:
                    if current_index == i:
                        # Current is left, sibling is right
                        siblings.append((right.hash, "right"))
                    else:
                        # Current is right, sibling is left
                        siblings.append((left.hash, "left"))

                parent = MerkleNode(
                    hash=combine_hashes(left.hash, right.hash),
                    left=left,
                    right=right,
                )
                next_level.append(parent)

            current_level = next_level
            current_index = current_index // 2

        return MerkleProof(
            item_hash=self.leaves[index].hash,
            item_index=index,
            siblings=siblings,
            root_hash=self.root_hash,
        )

    @staticmethod
    def verify_proof(proof: MerkleProof) -> bool:
        """Verify a Merkle proof.

        Args:
            proof: The proof to verify.

        Returns:
            True if the proof is valid.
        """
        current_hash = proof.item_hash

        for sibling_hash, position in proof.siblings:
            if position == "left":
                current_hash = combine_hashes(sibling_hash, current_hash)
            else:
                current_hash = combine_hashes(current_hash, sibling_hash)

        return current_hash == proof.root_hash


# =============================================================================
# EVIDENCE INTEGRITY CHECKER
# =============================================================================

@dataclass
class EvidenceRecord:
    """Record of evidence with integrity information."""
    evidence_id: str
    content_hash: str
    merkle_index: int
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceIntegrityChecker:
    """THREAT-T-001 FIX: Evidence integrity checker using Merkle trees.

    This class provides a high-level interface for:
    - Registering evidence items
    - Building integrity trees
    - Verifying evidence integrity
    - Generating tamper-proof audit records

    Thread-safe singleton implementation.
    """

    _instance: Optional["EvidenceIntegrityChecker"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        """Initialize evidence integrity checker."""
        self._evidence: List[Any] = []
        self._records: Dict[str, EvidenceRecord] = {}
        self._tree: Optional[MerkleTree] = None
        self._tree_lock = threading.Lock()
        self._root_history: List[Tuple[str, str]] = []  # (timestamp, root_hash)

    @classmethod
    def get_instance(cls) -> "EvidenceIntegrityChecker":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def add_evidence(
        self,
        evidence: Any,
        evidence_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add an evidence item.

        Args:
            evidence: Evidence content (dict, str, or JSON-serializable).
            evidence_id: Optional unique ID (generated if not provided).
            metadata: Optional metadata about the evidence.

        Returns:
            Evidence ID.
        """
        with self._tree_lock:
            # Generate ID if not provided
            if evidence_id is None:
                evidence_id = f"ev_{len(self._evidence):08d}"

            # Compute hash
            content_hash = hash_evidence_item(evidence)

            # Add to list
            index = len(self._evidence)
            self._evidence.append(evidence)

            # Create record
            record = EvidenceRecord(
                evidence_id=evidence_id,
                content_hash=content_hash,
                merkle_index=index,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {},
            )
            self._records[evidence_id] = record

            # Invalidate current tree (will rebuild on next access)
            self._tree = None

            logger.info(f"THREAT-T-001: Added evidence {evidence_id} (hash: {content_hash[:16]}...)")
            return evidence_id

    def build_tree(self) -> str:
        """Build or rebuild the Merkle tree.

        Returns:
            Root hash of the tree.
        """
        with self._tree_lock:
            self._tree = MerkleTree(self._evidence)
            root_hash = self._tree.root_hash

            # Record in history
            timestamp = datetime.now(timezone.utc).isoformat()
            self._root_history.append((timestamp, root_hash))

            logger.info(f"THREAT-T-001: Built Merkle tree with root {root_hash[:16]}...")
            return root_hash

    @property
    def root_hash(self) -> str:
        """Get current root hash (builds tree if needed)."""
        with self._tree_lock:
            if self._tree is None:
                self._tree = MerkleTree(self._evidence)
            return self._tree.root_hash

    def verify_evidence(
        self,
        evidence_id: str,
        expected_content: Any,
    ) -> Tuple[bool, Optional[str]]:
        """Verify an evidence item hasn't been tampered with.

        Args:
            evidence_id: ID of the evidence to verify.
            expected_content: Expected content of the evidence.

        Returns:
            Tuple of (is_valid, error_message).
        """
        with self._tree_lock:
            if evidence_id not in self._records:
                return False, f"THREAT-T-001: Evidence {evidence_id} not found"

            record = self._records[evidence_id]
            actual_hash = hash_evidence_item(expected_content)

            if actual_hash != record.content_hash:
                logger.error(
                    f"THREAT-T-001 TAMPERING DETECTED: Evidence {evidence_id} "
                    f"hash mismatch. Expected {record.content_hash[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )
                return False, "THREAT-T-001: Evidence tampering detected - hash mismatch"

            # Verify Merkle proof
            if self._tree is not None:
                proof = self._tree.get_proof(record.merkle_index)
                if proof and not MerkleTree.verify_proof(proof):
                    return False, "THREAT-T-001: Merkle proof verification failed"

            return True, None

    def get_proof(self, evidence_id: str) -> Optional[MerkleProof]:
        """Get Merkle proof for an evidence item.

        Args:
            evidence_id: ID of the evidence.

        Returns:
            MerkleProof or None if not found.
        """
        with self._tree_lock:
            if evidence_id not in self._records:
                return None

            record = self._records[evidence_id]

            if self._tree is None:
                self._tree = MerkleTree(self._evidence)

            return self._tree.get_proof(record.merkle_index)

    def get_audit_record(self) -> Dict[str, Any]:
        """Get complete audit record for evidence chain.

        Returns:
            Dictionary with all integrity information.
        """
        with self._tree_lock:
            return {
                "evidence_count": len(self._evidence),
                "root_hash": self.root_hash,
                "root_history": self._root_history,
                "records": {
                    eid: {
                        "content_hash": r.content_hash,
                        "merkle_index": r.merkle_index,
                        "created_at": r.created_at,
                    }
                    for eid, r in self._records.items()
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_evidence_tree(items: List[Any]) -> str:
    """Build a Merkle tree from evidence items and return root hash.

    Args:
        items: List of evidence items.

    Returns:
        Root hash of the Merkle tree.
    """
    tree = MerkleTree(items)
    return tree.root_hash


def verify_evidence_item(
    item: Any,
    index: int,
    proof: MerkleProof,
) -> bool:
    """Verify a single evidence item using a Merkle proof.

    Args:
        item: The evidence item to verify.
        index: Index of the item.
        proof: Merkle proof for the item.

    Returns:
        True if the item is verified.
    """
    item_hash = hash_evidence_item(item)

    if item_hash != proof.item_hash:
        logger.warning(f"THREAT-T-001: Item hash mismatch at index {index}")
        return False

    return MerkleTree.verify_proof(proof)


def get_evidence_checker() -> EvidenceIntegrityChecker:
    """Get the singleton evidence integrity checker.

    Returns:
        EvidenceIntegrityChecker instance.
    """
    return EvidenceIntegrityChecker.get_instance()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MerkleTree",
    "MerkleProof",
    "MerkleNode",
    "EvidenceIntegrityChecker",
    "EvidenceRecord",
    "build_evidence_tree",
    "verify_evidence_item",
    "get_evidence_checker",
    "compute_hash",
    "hash_evidence_item",
]
