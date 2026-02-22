"""
AURORA Module 6 — Merkle Tree

Binary Merkle tree construction for tamper-proof audit certification.
Root hash provides cryptographic integrity over all leaf hashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MerkleNode:
    """A node in the Merkle tree."""
    hash_value: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    is_leaf: bool = False
    data_label: str = ""


class MerkleTree:
    """
    Binary Merkle tree for cryptographic integrity verification.

    Usage:
        tree = MerkleTree()
        tree.add_leaf("model_before", hash_before)
        tree.add_leaf("model_after", hash_after)
        tree.add_leaf("metrics", hash_metrics)
        tree.build()
        root_hash = tree.root_hash
    """

    def __init__(self):
        self.leaves: list[MerkleNode] = []
        self.root: Optional[MerkleNode] = None

    @property
    def root_hash(self) -> str:
        """Return the root hash of the tree."""
        if self.root is None:
            raise ValueError("Tree has not been built. Call build() first.")
        return self.root.hash_value

    def add_leaf(self, label: str, hash_value: str) -> None:
        """
        Add a leaf node to the tree.

        Args:
            label: Human-readable label for the data.
            hash_value: SHA-256 hash of the data.
        """
        self.leaves.append(MerkleNode(
            hash_value=hash_value,
            is_leaf=True,
            data_label=label,
        ))

    def build(self) -> None:
        """Build the Merkle tree from the current leaves."""
        if not self.leaves:
            raise ValueError("No leaves to build tree from.")

        nodes = list(self.leaves)

        # If odd number of nodes, duplicate the last one
        while len(nodes) > 1:
            next_level = []

            for i in range(0, len(nodes), 2):
                left = nodes[i]

                if i + 1 < len(nodes):
                    right = nodes[i + 1]
                else:
                    right = left  # Duplicate last node

                # Parent hash = SHA256(left_hash + right_hash)
                combined = left.hash_value + right.hash_value
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()

                parent = MerkleNode(
                    hash_value=parent_hash,
                    left=left,
                    right=right,
                )
                next_level.append(parent)

            nodes = next_level

        self.root = nodes[0]

    def verify_leaf(self, label: str, hash_value: str) -> bool:
        """
        Verify that a leaf with the given label and hash exists in the tree.

        Args:
            label: Leaf label.
            hash_value: Expected hash value.

        Returns:
            True if the leaf exists with the correct hash.
        """
        for leaf in self.leaves:
            if leaf.data_label == label and leaf.hash_value == hash_value:
                return True
        return False

    def get_proof(self, leaf_index: int) -> list[tuple[str, str]]:
        """
        Get the Merkle proof (authentication path) for a leaf.

        Args:
            leaf_index: Index of the leaf (0-based).

        Returns:
            List of (hash, position) tuples forming the proof path.
            Position is 'left' or 'right'.
        """
        if self.root is None:
            raise ValueError("Tree not built.")

        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise ValueError(f"Invalid leaf index: {leaf_index}")

        proof = []
        nodes = list(self.leaves)
        index = leaf_index

        while len(nodes) > 1:
            next_level = []

            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    if i == index or i + 1 == index:
                        # This pair contains our target
                        sibling_idx = i + 1 if i == index else i
                        position = "right" if i == index else "left"
                        proof.append((nodes[sibling_idx].hash_value, position))

                    combined = nodes[i].hash_value + nodes[i + 1].hash_value
                else:
                    combined = nodes[i].hash_value + nodes[i].hash_value

                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(MerkleNode(hash_value=parent_hash))

            index = index // 2
            nodes = next_level

        return proof

    def to_dict(self) -> dict:
        """Serialize the tree structure to a dict."""
        return {
            "root_hash": self.root_hash if self.root else None,
            "num_leaves": len(self.leaves),
            "leaves": [
                {"label": leaf.data_label, "hash": leaf.hash_value}
                for leaf in self.leaves
            ],
        }
