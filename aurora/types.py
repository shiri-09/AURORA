"""
AURORA Shared Data Types

Core dataclasses used across all modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np


# ─── Fact Formalization Types ────────────────────────────────────────────

@dataclass
class TargetFact:
    """A single knowledge fact to be unlearned."""
    subject: str
    relation: str
    obj: str  # 'object' is a Python builtin
    fact_id: str = ""

    def __post_init__(self):
        if not self.fact_id:
            content = f"{self.subject}|{self.relation}|{self.obj}"
            self.fact_id = hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_natural_language(self) -> str:
        return f"{self.subject} {self.relation} {self.obj}"

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
        }


@dataclass
class PromptTarget:
    """A prompt paired with its expected target answer."""
    prompt: str
    target_answer: str
    is_direct: bool = True  # True = direct query, False = indirect/multi-hop
    hop_distance: int = 0   # 0 = direct, 1+ = multi-hop


@dataclass
class ForgetSet:
    """Collection of prompts whose knowledge should be erased."""
    fact: TargetFact
    prompts: list[PromptTarget] = field(default_factory=list)

    @property
    def direct_prompts(self) -> list[PromptTarget]:
        return [p for p in self.prompts if p.is_direct]

    @property
    def indirect_prompts(self) -> list[PromptTarget]:
        return [p for p in self.prompts if not p.is_direct]


@dataclass
class RetainSet:
    """Collection of prompts whose knowledge must be preserved."""
    prompts: list[PromptTarget] = field(default_factory=list)


# ─── Knowledge Graph Types ───────────────────────────────────────────────

class EdgeType(str, Enum):
    SEMANTIC = "semantic"
    ATTENTION = "attention"
    RETRIEVAL = "retrieval"
    GRADIENT = "gradient"
    REASONING = "reasoning"


@dataclass
class KGNode:
    """A node in the Relational Knowledge Graph."""
    id: str
    label: str
    embedding_vector: Optional[np.ndarray] = None
    attention_importance_score: float = 0.0
    frequency_score: float = 0.0
    gradient_sensitivity_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "attention_importance_score": self.attention_importance_score,
            "frequency_score": self.frequency_score,
            "gradient_sensitivity_score": self.gradient_sensitivity_score,
        }


@dataclass
class KGEdge:
    """An edge in the Relational Knowledge Graph."""
    source: str
    target: str
    weight: float = 1.0
    edge_type: EdgeType = EdgeType.SEMANTIC
    conditional_probability_estimate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "type": self.edge_type.value,
            "conditional_probability_estimate": self.conditional_probability_estimate,
        }


# ─── Parameter Localization Types ────────────────────────────────────────

@dataclass
class RiskParameters:
    """Identified high-risk parameter subspace θ_risk."""
    parameter_names: list[str]
    parameter_indices: dict[str, np.ndarray]  # param_name -> index mask
    influence_scores: dict[str, float]        # param_name -> score
    total_params_selected: int = 0

    def __post_init__(self):
        self.total_params_selected = sum(
            len(idx) for idx in self.parameter_indices.values()
        )


# ─── Evaluation Types ───────────────────────────────────────────────────

@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics from unlearning + adversarial testing."""
    # Core metrics
    direct_forget_accuracy: float = 0.0     # Lower is better (0 = fully forgotten)
    indirect_leakage_rate: float = 0.0      # Lower is better
    retain_utility_drop: float = 0.0        # Lower is better (KL divergence)
    reconstruction_probability: float = 0.0  # Lower is better
    parameter_drift_norm: float = 0.0       # Lower is better

    # Adversarial metrics
    correlated_ft_leakage: float = 0.0
    prompt_injection_leakage: float = 0.0
    multihop_leakage: float = 0.0
    quantization_leakage: float = 0.0
    lora_recovery_leakage: float = 0.0

    # Quantum distinguishability
    trace_distance_forget: float = 0.0
    trace_distance_retain: float = 0.0

    def to_dict(self) -> dict:
        return {
            "direct_forget_accuracy": self.direct_forget_accuracy,
            "indirect_leakage_rate": self.indirect_leakage_rate,
            "retain_utility_drop": self.retain_utility_drop,
            "reconstruction_probability": self.reconstruction_probability,
            "parameter_drift_norm": self.parameter_drift_norm,
            "correlated_ft_leakage": self.correlated_ft_leakage,
            "prompt_injection_leakage": self.prompt_injection_leakage,
            "multihop_leakage": self.multihop_leakage,
            "quantization_leakage": self.quantization_leakage,
            "lora_recovery_leakage": self.lora_recovery_leakage,
            "trace_distance_forget": self.trace_distance_forget,
            "trace_distance_retain": self.trace_distance_retain,
        }

    def passes_bounds(self, alpha: float, epsilon: float) -> bool:
        """Check if metrics satisfy formal bounds."""
        return (
            self.indirect_leakage_rate <= epsilon
            and self.retain_utility_drop <= alpha
        )


# ─── Certificate Types ──────────────────────────────────────────────────

@dataclass
class UnlearningCertificate:
    """Cryptographic certificate of unlearning compliance."""
    certificate_id: str
    timestamp: str = ""
    root_hash: str = ""
    signature: bytes = b""
    model_hash_before: str = ""
    model_hash_after: str = ""
    metrics_hash: str = ""
    target_fact: Optional[dict] = None
    metrics: Optional[dict] = None
    compliance_tag: str = "AURORA-v0.1"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "certificate_id": self.certificate_id,
            "timestamp": self.timestamp,
            "root_hash": self.root_hash,
            "signature": self.signature.hex() if self.signature else "",
            "model_hash_before": self.model_hash_before,
            "model_hash_after": self.model_hash_after,
            "metrics_hash": self.metrics_hash,
            "target_fact": self.target_fact,
            "metrics": self.metrics,
            "compliance_tag": self.compliance_tag,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─── Pipeline Result ────────────────────────────────────────────────────

@dataclass
class UnlearningResult:
    """Complete result from the AURORA pipeline."""
    success: bool
    fact: TargetFact
    metrics: EvaluationMetrics
    certificate: Optional[UnlearningCertificate] = None
    graph_stats: dict = field(default_factory=dict)
    risk_params_count: int = 0
    cascade_iterations: int = 0
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "fact": self.fact.to_dict(),
            "metrics": self.metrics.to_dict(),
            "certificate": self.certificate.to_dict() if self.certificate else None,
            "graph_stats": self.graph_stats,
            "risk_params_count": self.risk_params_count,
            "cascade_iterations": self.cascade_iterations,
        }
