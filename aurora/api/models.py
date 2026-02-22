"""
AURORA API — Pydantic Request/Response Models
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class UnlearnRequest(BaseModel):
    """Request to unlearn a target fact."""
    subject: str = Field(..., description="Subject entity", examples=["Eiffel Tower"])
    relation: str = Field(..., description="Relation type", examples=["location"])
    obj: str = Field(..., description="Object value to forget", examples=["Paris"])
    retain_prompts: Optional[list[list[str]]] = Field(
        None, description="List of [prompt, answer] pairs to retain"
    )
    correlated_texts: Optional[list[str]] = Field(
        None, description="Related documents for adversarial testing"
    )


class MetricsResponse(BaseModel):
    """Evaluation metrics response."""
    direct_forget_accuracy: float
    indirect_leakage_rate: float
    retain_utility_drop: float
    reconstruction_probability: float
    parameter_drift_norm: float
    correlated_ft_leakage: float
    prompt_injection_leakage: float
    multihop_leakage: float
    quantization_leakage: float
    lora_recovery_leakage: float
    trace_distance_forget: float
    trace_distance_retain: float


class CertificateResponse(BaseModel):
    """Certificate response."""
    certificate_id: str
    timestamp: str
    root_hash: str
    compliance_tag: str
    target_fact: dict
    metrics: dict


class UnlearnResponse(BaseModel):
    """Full unlearning response."""
    success: bool
    fact: dict
    metrics: MetricsResponse
    certificate: Optional[CertificateResponse] = None
    graph_stats: dict
    risk_params_count: int
    cascade_iterations: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "0.1.0"
    model_loaded: bool = False
