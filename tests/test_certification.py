"""
Tests for AURORA Module 6 — Cryptographic Certification
"""

import pytest
from aurora.certification.hasher import hash_data, hash_metrics
from aurora.certification.merkle_tree import MerkleTree
from aurora.certification.signer import generate_key_pair, sign, verify


class TestHashing:
    def test_hash_data(self):
        h = hash_data(b"hello")
        assert len(h) == 64  # SHA-256 hex digest

    def test_hash_deterministic(self):
        h1 = hash_data(b"test")
        h2 = hash_data(b"test")
        assert h1 == h2

    def test_hash_metrics(self):
        metrics = {"accuracy": 0.95, "loss": 0.05}
        h = hash_metrics(metrics)
        assert len(h) == 64

    def test_hash_metrics_order_independent(self):
        h1 = hash_metrics({"a": 1, "b": 2})
        h2 = hash_metrics({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys=True


class TestMerkleTree:
    def test_build_single_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("data", "abc123")
        tree.build()
        assert tree.root_hash is not None

    def test_build_multiple_leaves(self):
        tree = MerkleTree()
        tree.add_leaf("a", "hash_a")
        tree.add_leaf("b", "hash_b")
        tree.add_leaf("c", "hash_c")
        tree.add_leaf("d", "hash_d")
        tree.build()
        assert tree.root_hash is not None

    def test_root_hash_deterministic(self):
        t1 = MerkleTree()
        t1.add_leaf("a", "h1")
        t1.add_leaf("b", "h2")
        t1.build()

        t2 = MerkleTree()
        t2.add_leaf("a", "h1")
        t2.add_leaf("b", "h2")
        t2.build()

        assert t1.root_hash == t2.root_hash

    def test_verify_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("data", "myhash")
        tree.build()
        assert tree.verify_leaf("data", "myhash")
        assert not tree.verify_leaf("data", "wronghash")

    def test_to_dict(self):
        tree = MerkleTree()
        tree.add_leaf("a", "h1")
        tree.build()
        d = tree.to_dict()
        assert "root_hash" in d
        assert d["num_leaves"] == 1

    def test_get_proof(self):
        tree = MerkleTree()
        tree.add_leaf("a", "h1")
        tree.add_leaf("b", "h2")
        tree.add_leaf("c", "h3")
        tree.add_leaf("d", "h4")
        tree.build()
        proof = tree.get_proof(0)
        assert len(proof) > 0


class TestSigning:
    def test_generate_key_pair(self):
        private, public = generate_key_pair(key_size=2048)
        assert private is not None
        assert public is not None

    def test_sign_and_verify(self):
        private, public = generate_key_pair(key_size=2048)
        data = "test data"
        signature = sign(data, private)
        assert verify(data, signature, public)

    def test_verify_wrong_data(self):
        private, public = generate_key_pair(key_size=2048)
        signature = sign("correct data", private)
        assert not verify("wrong data", signature, public)

    def test_sign_produces_bytes(self):
        private, _ = generate_key_pair(key_size=2048)
        sig = sign("hello", private)
        assert isinstance(sig, bytes)
        assert len(sig) > 0
