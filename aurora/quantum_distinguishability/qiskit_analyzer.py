"""
AURORA Module 7b — Qiskit Quantum Distinguishability Analyzer

Uses IBM Qiskit to encode embedding vectors into quantum circuits
and compute distinguishability metrics via statevector simulation.

Pipeline:
    1. PCA-compress embeddings → 4–6 dimensions
    2. Angle-encode into qubit rotation gates (RY)
    3. Apply entanglement layer (CX chain)
    4. Extract statevector |ψ⟩
    5. Compute fidelity F and trace distance D

This is a VERIFICATION lens, not a computational engine.
It runs in simulator mode and is non-blocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aurora.config import AuroraConfig
from aurora.utils.logging import quantum_logger

logger = quantum_logger()

# Number of qubits for the quantum circuit
DEFAULT_NUM_QUBITS = 4


@dataclass
class QiskitDistinguishabilityResult:
    """Result from Qiskit-based quantum distinguishability analysis."""
    fidelity_forget: float              # F(|ψ_before⟩, |ψ_after⟩) for forget set
    fidelity_retain: float              # F(|ψ_before⟩, |ψ_after⟩) for retain set
    trace_distance_forget: float        # D(ρ_before, ρ_after) for forget set
    trace_distance_retain: float        # D(ρ_before, ρ_after) for retain set
    ratio: float                        # D_forget / D_retain
    passes_threshold: bool
    forget_fidelities: list[float] = field(default_factory=list)
    retain_fidelities: list[float] = field(default_factory=list)
    forget_distances: list[float] = field(default_factory=list)
    retain_distances: list[float] = field(default_factory=list)
    num_qubits: int = DEFAULT_NUM_QUBITS

    def to_dict(self) -> dict:
        return {
            "fidelity_forget": self.fidelity_forget,
            "fidelity_retain": self.fidelity_retain,
            "trace_distance_forget": self.trace_distance_forget,
            "trace_distance_retain": self.trace_distance_retain,
            "ratio": self.ratio,
            "passes_threshold": self.passes_threshold,
            "forget_fidelities": self.forget_fidelities,
            "retain_fidelities": self.retain_fidelities,
            "forget_distances": self.forget_distances,
            "retain_distances": self.retain_distances,
            "num_qubits": self.num_qubits,
        }


def _check_qiskit_available() -> bool:
    """Check if Qiskit is installed."""
    try:
        import qiskit  # noqa: F401
        return True
    except ImportError:
        return False


def _pca_compress(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """
    Compress embedding vectors to n_components dimensions using PCA.

    Args:
        vectors: Array of shape (n_samples, d).
        n_components: Target dimensions.

    Returns:
        Compressed array of shape (n_samples, n_components).
    """
    if vectors.shape[1] <= n_components:
        # Pad with zeros if fewer dimensions than qubits
        padded = np.zeros((vectors.shape[0], n_components))
        padded[:, :vectors.shape[1]] = vectors
        return padded

    # Center the data
    mean = vectors.mean(axis=0)
    centered = vectors - mean

    # SVD-based PCA (no sklearn dependency)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    compressed = centered @ components.T

    return compressed


def _normalize_to_angles(values: np.ndarray) -> np.ndarray:
    """
    Normalize values to [0, π] range for angle encoding.

    Args:
        values: Array of arbitrary values.

    Returns:
        Array of angles in [0, π].
    """
    v_min = values.min()
    v_max = values.max()
    if v_max - v_min < 1e-10:
        return np.full_like(values, np.pi / 2)
    normalized = (values - v_min) / (v_max - v_min)  # [0, 1]
    return normalized * np.pi  # [0, π]


def build_encoding_circuit(angles: np.ndarray, num_qubits: int):
    """
    Build a quantum circuit that angle-encodes embedding values.

    Architecture:
        1. RY(angle_i) on each qubit i  (rotation encoding)
        2. CX chain for entanglement     (qubit i → qubit i+1)

    Args:
        angles: Array of rotation angles, length = num_qubits.
        num_qubits: Number of qubits.

    Returns:
        Qiskit QuantumCircuit.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)

    # Layer 1: Angle encoding via RY rotations
    for i in range(num_qubits):
        qc.ry(float(angles[i]), i)

    # Layer 2: Entanglement via CX chain
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc


def get_statevector(circuit) -> np.ndarray:
    """
    Simulate the circuit and return the statevector.

    Args:
        circuit: Qiskit QuantumCircuit.

    Returns:
        Complex statevector array of length 2^num_qubits.
    """
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_instruction(circuit)
    return np.array(sv.data)


def quantum_fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """
    Compute fidelity between two pure quantum states.

    F = |⟨ψ₁|ψ₂⟩|²

    Args:
        psi1: First statevector.
        psi2: Second statevector.

    Returns:
        Fidelity in [0, 1].
    """
    inner_product = np.abs(np.vdot(psi1, psi2)) ** 2
    return float(np.clip(inner_product, 0.0, 1.0))


def quantum_trace_distance(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """
    Compute trace distance between two pure quantum states.

    D(ρ₁, ρ₂) = ½ Tr(|ρ₁ - ρ₂|)

    For pure states: D = √(1 - F)

    Args:
        psi1: First statevector.
        psi2: Second statevector.

    Returns:
        Trace distance in [0, 1].
    """
    fid = quantum_fidelity(psi1, psi2)
    return float(np.sqrt(max(1.0 - fid, 0.0)))


class QiskitDistinguishabilityAnalyzer:
    """
    Qiskit-based quantum distinguishability analysis.

    Encodes embedding vectors into quantum circuits using angle encoding,
    then computes fidelity and trace distance to measure how much the
    model's representations changed for forget vs retain data.

    Key insight:
        - Classical metrics check: "Did the output change?"
        - Quantum metrics check: "Did the representation geometry shift?"

    Usage:
        analyzer = QiskitDistinguishabilityAnalyzer(config)
        result = analyzer.analyze(
            before_forget, after_forget,
            before_retain, after_retain,
        )
    """

    def __init__(self, config: AuroraConfig, num_qubits: int = DEFAULT_NUM_QUBITS):
        self.config = config
        self.num_qubits = num_qubits
        self.threshold = config.distinguishability_threshold
        self._qiskit_available = _check_qiskit_available()

        if not self._qiskit_available:
            logger.warning(
                "Qiskit not installed. Install with: pip install qiskit\n"
                "Falling back to classical numpy-based analysis."
            )

    @property
    def is_available(self) -> bool:
        """Check if Qiskit backend is available."""
        return self._qiskit_available

    def _encode_and_get_statevector(self, embedding: np.ndarray) -> np.ndarray:
        """
        Encode a single embedding vector into a quantum statevector.

        Pipeline: embedding → PCA → angles → circuit → statevector
        """
        # Ensure correct shape
        if embedding.ndim == 1:
            angles = _normalize_to_angles(embedding[:self.num_qubits])
        else:
            angles = _normalize_to_angles(embedding.flatten()[:self.num_qubits])

        # Pad if needed
        if len(angles) < self.num_qubits:
            padded = np.full(self.num_qubits, np.pi / 2)
            padded[:len(angles)] = angles
            angles = padded

        circuit = build_encoding_circuit(angles, self.num_qubits)
        return get_statevector(circuit)

    def analyze(
        self,
        before_forget: np.ndarray,
        after_forget: np.ndarray,
        before_retain: np.ndarray,
        after_retain: np.ndarray,
    ) -> QiskitDistinguishabilityResult:
        """
        Run Qiskit quantum distinguishability analysis.

        Args:
            before_forget: Embeddings for forget data BEFORE unlearning. Shape: (n, d)
            after_forget:  Embeddings for forget data AFTER unlearning.  Shape: (n, d)
            before_retain: Embeddings for retain data BEFORE unlearning. Shape: (m, d)
            after_retain:  Embeddings for retain data AFTER unlearning.  Shape: (m, d)

        Returns:
            QiskitDistinguishabilityResult.
        """
        if not self._qiskit_available:
            logger.error("Qiskit not available. Cannot run quantum analysis.")
            return QiskitDistinguishabilityResult(
                fidelity_forget=0.0, fidelity_retain=0.0,
                trace_distance_forget=0.0, trace_distance_retain=0.0,
                ratio=0.0, passes_threshold=False, num_qubits=self.num_qubits,
            )

        logger.info(f"=== QISKIT QUANTUM DISTINGUISHABILITY ({self.num_qubits} qubits) ===")

        # PCA compress all embeddings to num_qubits dimensions
        all_before = np.vstack([before_forget, before_retain])
        all_after = np.vstack([after_forget, after_retain])
        all_embeddings = np.vstack([all_before, all_after])

        compressed = _pca_compress(all_embeddings, self.num_qubits)

        n_forget = len(before_forget)
        n_retain = len(before_retain)
        n_total = n_forget + n_retain

        comp_before_forget = compressed[:n_forget]
        comp_before_retain = compressed[n_forget:n_total]
        comp_after_forget = compressed[n_total:n_total + n_forget]
        comp_after_retain = compressed[n_total + n_forget:]

        # Compute forget set metrics
        logger.info("Computing forget set quantum metrics...")
        forget_fidelities = []
        forget_distances = []

        for i in range(n_forget):
            psi_before = self._encode_and_get_statevector(comp_before_forget[i])
            psi_after = self._encode_and_get_statevector(comp_after_forget[i])

            fid = quantum_fidelity(psi_before, psi_after)
            dist = quantum_trace_distance(psi_before, psi_after)

            forget_fidelities.append(fid)
            forget_distances.append(dist)

        # Compute retain set metrics
        logger.info("Computing retain set quantum metrics...")
        retain_fidelities = []
        retain_distances = []

        for i in range(n_retain):
            psi_before = self._encode_and_get_statevector(comp_before_retain[i])
            psi_after = self._encode_and_get_statevector(comp_after_retain[i])

            fid = quantum_fidelity(psi_before, psi_after)
            dist = quantum_trace_distance(psi_before, psi_after)

            retain_fidelities.append(fid)
            retain_distances.append(dist)

        # Aggregate
        avg_fid_forget = float(np.mean(forget_fidelities)) if forget_fidelities else 1.0
        avg_fid_retain = float(np.mean(retain_fidelities)) if retain_fidelities else 1.0
        avg_dist_forget = float(np.mean(forget_distances)) if forget_distances else 0.0
        avg_dist_retain = float(np.mean(retain_distances)) if retain_distances else 1e-10

        ratio = avg_dist_forget / max(avg_dist_retain, 1e-10)
        passes = ratio >= self.threshold

        result = QiskitDistinguishabilityResult(
            fidelity_forget=avg_fid_forget,
            fidelity_retain=avg_fid_retain,
            trace_distance_forget=avg_dist_forget,
            trace_distance_retain=avg_dist_retain,
            ratio=ratio,
            passes_threshold=passes,
            forget_fidelities=forget_fidelities,
            retain_fidelities=retain_fidelities,
            forget_distances=forget_distances,
            retain_distances=retain_distances,
            num_qubits=self.num_qubits,
        )

        logger.info(f"  Qubits:           {self.num_qubits}")
        logger.info(f"  F_forget (avg):   {avg_fid_forget:.6f}  (lower = more change)")
        logger.info(f"  F_retain (avg):   {avg_fid_retain:.6f}  (higher = preserved)")
        logger.info(f"  D_forget (avg):   {avg_dist_forget:.6f}")
        logger.info(f"  D_retain (avg):   {avg_dist_retain:.6f}")
        logger.info(f"  Ratio D_f/D_r:    {ratio:.4f}")
        logger.info(f"  Threshold:        {self.threshold}")
        logger.info(f"  Status:           {'PASSED' if passes else 'FAILED'}")
        logger.info("=== QISKIT ANALYSIS COMPLETE ===")

        return result

    def analyze_from_model(
        self,
        model_before,
        model_after,
        tokenizer,
        forget_texts: list[str],
        retain_texts: list[str],
        device: str = "cpu",
    ) -> QiskitDistinguishabilityResult:
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
            QiskitDistinguishabilityResult.
        """
        from aurora.utils.embeddings import extract_embeddings

        logger.info("Extracting before/after embeddings for Qiskit analysis...")

        before_forget = extract_embeddings(model_before, tokenizer, forget_texts, device)
        after_forget = extract_embeddings(model_after, tokenizer, forget_texts, device)
        before_retain = extract_embeddings(model_before, tokenizer, retain_texts, device)
        after_retain = extract_embeddings(model_after, tokenizer, retain_texts, device)

        return self.analyze(before_forget, after_forget, before_retain, after_retain)
