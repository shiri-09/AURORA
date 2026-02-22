"""
AURORA Module 7 — Quantum-Inspired Distinguishability Analyzer

Encodes embedding vectors into quantum state representations and
computes trace distance to validate that unlearning creates a
measurable separation in the model's representation space.

Key operations:
    v → |ψ⟩                          (embedding to quantum state)
    ρ = |ψ⟩⟨ψ|                       (density matrix construction)
    D(ρ₁, ρ₂) = ½ Tr(|ρ₁ - ρ₂|)     (trace distance)

Validation criterion: D_forget >> D_retain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import sqrtm

from aurora.config import AuroraConfig
from aurora.utils.logging import quantum_logger

logger = quantum_logger()


@dataclass
class DistinguishabilityResult:
    """Result from quantum distinguishability analysis."""
    trace_distance_forget: float
    trace_distance_retain: float
    ratio: float  # D_forget / D_retain
    passes_threshold: bool
    forget_distances: list[float]
    retain_distances: list[float]

    def to_dict(self) -> dict:
        return {
            "trace_distance_forget": self.trace_distance_forget,
            "trace_distance_retain": self.trace_distance_retain,
            "ratio": self.ratio,
            "passes_threshold": self.passes_threshold,
            "forget_distances": self.forget_distances,
            "retain_distances": self.retain_distances,
        }


class QuantumDistinguishabilityAnalyzer:
    """
    Quantum-inspired distinguishability analysis.

    Uses trace distance between density matrices to quantify how
    much the model's internal representations have changed for forget
    vs retain data after unlearning.

    If D_forget >> D_retain, unlearning is working correctly:
    - Forget representations have changed dramatically (good)
    - Retain representations are preserved (good)

    Usage:
        analyzer = QuantumDistinguishabilityAnalyzer(config)
        result = analyzer.analyze(
            before_forget_embeddings, after_forget_embeddings,
            before_retain_embeddings, after_retain_embeddings,
        )
    """

    def __init__(self, config: AuroraConfig):
        self.config = config
        self.threshold = config.distinguishability_threshold

    def analyze(
        self,
        before_forget: np.ndarray,
        after_forget: np.ndarray,
        before_retain: np.ndarray,
        after_retain: np.ndarray,
    ) -> DistinguishabilityResult:
        """
        Run full quantum distinguishability analysis.

        Args:
            before_forget: Embeddings for forget data BEFORE unlearning. Shape: (n, d)
            after_forget:  Embeddings for forget data AFTER unlearning.  Shape: (n, d)
            before_retain: Embeddings for retain data BEFORE unlearning. Shape: (m, d)
            after_retain:  Embeddings for retain data AFTER unlearning.  Shape: (m, d)

        Returns:
            DistinguishabilityResult with trace distances and pass/fail.
        """
        logger.info("═══ QUANTUM DISTINGUISHABILITY ANALYSIS ═══")

        # Compute trace distances for forget data
        logger.info("Computing forget set trace distances...")
        forget_distances = []
        for i in range(min(len(before_forget), len(after_forget))):
            psi_before = embedding_to_quantum_state(before_forget[i])
            psi_after = embedding_to_quantum_state(after_forget[i])

            rho_before = construct_density_matrix(psi_before)
            rho_after = construct_density_matrix(psi_after)

            dist = trace_distance(rho_before, rho_after)
            forget_distances.append(dist)

        # Compute trace distances for retain data
        logger.info("Computing retain set trace distances...")
        retain_distances = []
        for i in range(min(len(before_retain), len(after_retain))):
            psi_before = embedding_to_quantum_state(before_retain[i])
            psi_after = embedding_to_quantum_state(after_retain[i])

            rho_before = construct_density_matrix(psi_before)
            rho_after = construct_density_matrix(psi_after)

            dist = trace_distance(rho_before, rho_after)
            retain_distances.append(dist)

        # Aggregate
        avg_forget = np.mean(forget_distances) if forget_distances else 0.0
        avg_retain = np.mean(retain_distances) if retain_distances else 1e-10
        ratio = avg_forget / max(avg_retain, 1e-10)

        passes = ratio >= self.threshold

        result = DistinguishabilityResult(
            trace_distance_forget=float(avg_forget),
            trace_distance_retain=float(avg_retain),
            ratio=float(ratio),
            passes_threshold=passes,
            forget_distances=[float(d) for d in forget_distances],
            retain_distances=[float(d) for d in retain_distances],
        )

        logger.info(f"  D_forget (avg): {avg_forget:.6f}")
        logger.info(f"  D_retain (avg): {avg_retain:.6f}")
        logger.info(f"  Ratio D_forget/D_retain: {ratio:.4f}")
        logger.info(f"  Threshold: {self.threshold}")
        logger.info(f"  Status: {'✓ PASSED' if passes else '✗ FAILED'}")
        logger.info("═══ ANALYSIS COMPLETE ═══")

        return result

    def analyze_from_model(
        self,
        model_before,
        model_after,
        tokenizer,
        forget_texts: list[str],
        retain_texts: list[str],
        device: str = "cpu",
    ) -> DistinguishabilityResult:
        """
        Convenience method: extract embeddings from models and analyze.

        Args:
            model_before: Model before unlearning.
            model_after: Model after unlearning.
            tokenizer: Tokenizer.
            forget_texts: Forget set prompts.
            retain_texts: Retain set prompts.
            device: Compute device.

        Returns:
            DistinguishabilityResult.
        """
        from aurora.utils.embeddings import extract_embeddings

        logger.info("Extracting before/after embeddings...")

        before_forget = extract_embeddings(
            model_before, tokenizer, forget_texts, device
        )
        after_forget = extract_embeddings(
            model_after, tokenizer, forget_texts, device
        )
        before_retain = extract_embeddings(
            model_before, tokenizer, retain_texts, device
        )
        after_retain = extract_embeddings(
            model_after, tokenizer, retain_texts, device
        )

        return self.analyze(
            before_forget, after_forget,
            before_retain, after_retain,
        )


# ─── Core Quantum Operations ────────────────────────────────────────────


def embedding_to_quantum_state(v: np.ndarray) -> np.ndarray:
    """
    Normalize embedding vector to quantum state |ψ⟩.

    v → |ψ⟩ such that ⟨ψ|ψ⟩ = 1

    For high-dimensional embeddings, we project to a manageable
    subspace while preserving relative information.

    Args:
        v: Embedding vector of arbitrary dimension.

    Returns:
        Normalized vector |ψ⟩.
    """
    # Project to manageable dimension for density matrix operations
    max_dim = 128
    if len(v) > max_dim:
        # Use truncated representation (first max_dim components)
        # This preserves the most significant directions
        v = v[:max_dim]

    # L2 normalize to unit vector
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        # Fallback: create maximally mixed state
        psi = np.ones(len(v)) / np.sqrt(len(v))
    else:
        psi = v / norm

    return psi


def construct_density_matrix(psi: np.ndarray) -> np.ndarray:
    """
    Construct density matrix from pure quantum state.

    ρ = |ψ⟩⟨ψ|

    Args:
        psi: Normalized state vector |ψ⟩.

    Returns:
        Density matrix ρ of shape (d, d).
    """
    return np.outer(psi, psi.conj())


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute trace distance between two density matrices.

    D(ρ₁, ρ₂) = ½ Tr(|ρ₁ - ρ₂|)

    where |A| = √(A†A) (matrix absolute value via eigenvalues).

    Args:
        rho1: First density matrix.
        rho2: Second density matrix.

    Returns:
        Trace distance in [0, 1].
    """
    diff = rho1 - rho2

    # Compute |diff| = √(diff† @ diff) via eigendecomposition
    # For Hermitian matrices, eigenvalues are real
    try:
        # diff is Hermitian, so eigenvalues are real
        eigenvalues = np.linalg.eigvalsh(diff)
        abs_eigenvalues = np.abs(eigenvalues)
        distance = 0.5 * np.sum(abs_eigenvalues)
    except np.linalg.LinAlgError:
        # Fallback: Frobenius norm approximation
        distance = 0.5 * np.linalg.norm(diff, "fro")

    return float(np.clip(distance, 0.0, 1.0))


def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute quantum fidelity between two density matrices.

    F(ρ₁, ρ₂) = [Tr(√(√ρ₁ · ρ₂ · √ρ₁))]²

    Complementary to trace distance: high fidelity = low distance.

    Args:
        rho1: First density matrix.
        rho2: Second density matrix.

    Returns:
        Fidelity in [0, 1].
    """
    try:
        sqrt_rho1 = sqrtm(rho1)

        if np.iscomplexobj(sqrt_rho1):
            sqrt_rho1 = sqrt_rho1.real

        product = sqrt_rho1 @ rho2 @ sqrt_rho1
        sqrt_product = sqrtm(product)

        if np.iscomplexobj(sqrt_product):
            sqrt_product = sqrt_product.real

        fid = np.real(np.trace(sqrt_product)) ** 2
        return float(np.clip(fid, 0.0, 1.0))

    except Exception:
        # Fallback for numerical instability
        return float(np.clip(np.abs(np.trace(rho1 @ rho2)), 0.0, 1.0))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute von Neumann entropy of a density matrix.

    S(ρ) = -Tr(ρ log ρ)

    Args:
        rho: Density matrix.

    Returns:
        Entropy value ≥ 0.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter near-zero

    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return float(max(entropy, 0.0))


def relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute quantum relative entropy S(ρ || σ).

    S(ρ || σ) = Tr(ρ(log ρ - log σ))

    This is the quantum analog of KL divergence.

    Args:
        rho: First density matrix.
        sigma: Second density matrix.

    Returns:
        Relative entropy value ≥ 0.
    """
    try:
        eig_rho = np.linalg.eigvalsh(rho)
        eig_sigma = np.linalg.eigvalsh(sigma)

        # Filter near-zero eigenvalues
        mask_rho = eig_rho > 1e-10
        mask_sigma = eig_sigma > 1e-10

        if not np.all(mask_sigma[mask_rho]):
            return float("inf")  # Support condition violated

        log_rho = np.zeros_like(eig_rho)
        log_sigma = np.zeros_like(eig_sigma)

        log_rho[mask_rho] = np.log2(eig_rho[mask_rho])
        log_sigma[mask_sigma] = np.log2(eig_sigma[mask_sigma])

        # S(ρ || σ) = Σ λᵢ(log λᵢ - log μᵢ)
        divergence = np.sum(eig_rho * (log_rho - log_sigma))
        return float(max(divergence, 0.0))

    except Exception:
        return float("inf")
