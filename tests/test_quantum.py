"""
Tests for AURORA Module 7 — Quantum Distinguishability
"""

import pytest
import numpy as np

from aurora.quantum_distinguishability.analyzer import (
    embedding_to_quantum_state,
    construct_density_matrix,
    trace_distance,
    fidelity,
    von_neumann_entropy,
    QuantumDistinguishabilityAnalyzer,
)
from aurora.config import AuroraConfig


class TestQuantumState:
    def test_normalization(self):
        v = np.array([3.0, 4.0])
        psi = embedding_to_quantum_state(v)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-6

    def test_zero_vector_fallback(self):
        v = np.zeros(10)
        psi = embedding_to_quantum_state(v)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-6

    def test_high_dim_truncation(self):
        v = np.random.randn(768)  # GPT-2 hidden dim
        psi = embedding_to_quantum_state(v)
        assert len(psi) == 128  # Truncated

    def test_unit_vector(self):
        v = np.array([1.0, 0.0, 0.0])
        psi = embedding_to_quantum_state(v)
        np.testing.assert_allclose(psi, v, atol=1e-6)


class TestDensityMatrix:
    def test_shape(self):
        psi = np.array([1.0, 0.0])
        rho = construct_density_matrix(psi)
        assert rho.shape == (2, 2)

    def test_trace_one(self):
        psi = embedding_to_quantum_state(np.array([1.0, 2.0, 3.0]))
        rho = construct_density_matrix(psi)
        assert abs(np.trace(rho) - 1.0) < 1e-6

    def test_positive_semidefinite(self):
        psi = embedding_to_quantum_state(np.random.randn(10))
        rho = construct_density_matrix(psi)
        eigenvalues = np.linalg.eigvalsh(rho)
        assert all(ev >= -1e-10 for ev in eigenvalues)

    def test_pure_state(self):
        psi = np.array([1.0, 0.0])
        rho = construct_density_matrix(psi)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(rho, expected)


class TestTraceDistance:
    def test_identical_states(self):
        rho = construct_density_matrix(np.array([1.0, 0.0]))
        d = trace_distance(rho, rho)
        assert abs(d) < 1e-6

    def test_orthogonal_states(self):
        rho1 = construct_density_matrix(np.array([1.0, 0.0]))
        rho2 = construct_density_matrix(np.array([0.0, 1.0]))
        d = trace_distance(rho1, rho2)
        assert abs(d - 1.0) < 1e-6

    def test_bounded(self):
        psi1 = embedding_to_quantum_state(np.random.randn(10))
        psi2 = embedding_to_quantum_state(np.random.randn(10))
        rho1 = construct_density_matrix(psi1)
        rho2 = construct_density_matrix(psi2)
        d = trace_distance(rho1, rho2)
        assert 0.0 <= d <= 1.0

    def test_symmetric(self):
        psi1 = embedding_to_quantum_state(np.array([1.0, 1.0]))
        psi2 = embedding_to_quantum_state(np.array([1.0, -1.0]))
        rho1 = construct_density_matrix(psi1)
        rho2 = construct_density_matrix(psi2)
        assert abs(trace_distance(rho1, rho2) - trace_distance(rho2, rho1)) < 1e-6


class TestFidelity:
    def test_same_state(self):
        rho = construct_density_matrix(np.array([1.0, 0.0]))
        f = fidelity(rho, rho)
        assert abs(f - 1.0) < 1e-3

    def test_orthogonal(self):
        rho1 = construct_density_matrix(np.array([1.0, 0.0]))
        rho2 = construct_density_matrix(np.array([0.0, 1.0]))
        f = fidelity(rho1, rho2)
        assert f < 0.1


class TestVonNeumannEntropy:
    def test_pure_state_zero_entropy(self):
        psi = np.array([1.0, 0.0])
        rho = construct_density_matrix(psi)
        s = von_neumann_entropy(rho)
        assert s < 0.01  # Pure states have ~zero entropy


class TestAnalyzer:
    def test_forget_vs_retain(self):
        config = AuroraConfig(distinguishability_threshold=0.1)
        analyzer = QuantumDistinguishabilityAnalyzer(config)

        # Simulate: forget embeddings change dramatically
        before_forget = np.random.randn(5, 64)
        after_forget = np.random.randn(5, 64)  # Completely different

        # Retain embeddings stay similar
        before_retain = np.random.randn(5, 64)
        after_retain = before_retain + np.random.randn(5, 64) * 0.01

        result = analyzer.analyze(
            before_forget, after_forget,
            before_retain, after_retain,
        )

        # D_forget should be much larger than D_retain
        assert result.trace_distance_forget > result.trace_distance_retain
        assert result.ratio > 1.0
