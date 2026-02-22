"""
Tests for AURORA Math Operations
"""

import pytest
import numpy as np
import torch

from aurora.utils.math_ops import (
    kl_divergence,
    normalize_vector,
    top_k_indices,
)


class TestKLDivergence:
    def test_same_distribution(self):
        logits = torch.randn(1, 10)
        kl = kl_divergence(logits, logits)
        assert abs(kl.item()) < 1e-5

    def test_different_distributions(self):
        p = torch.tensor([[10.0, 0.0, 0.0]])
        q = torch.tensor([[0.0, 10.0, 0.0]])
        kl = kl_divergence(p, q)
        assert kl.item() > 0


class TestNormalize:
    def test_unit_vector(self):
        v = np.array([3.0, 4.0])
        n = normalize_vector(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_zero_vector(self):
        v = np.zeros(5)
        n = normalize_vector(v)
        assert np.linalg.norm(n) == 0.0


class TestTopK:
    def test_basic(self):
        scores = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        idx = top_k_indices(scores, 3)
        assert 3 in idx  # 5.0
        assert 4 in idx  # 4.0
        assert 1 in idx  # 3.0

    def test_k_larger_than_array(self):
        scores = np.array([1.0, 2.0])
        idx = top_k_indices(scores, 10)
        assert len(idx) == 2
