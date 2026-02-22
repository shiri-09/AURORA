"""
AURORA Configuration Schema

Central configuration for all AURORA modules.
Uses Pydantic for validation and type safety.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    XPU = "xpu"
    AUTO = "auto"


class AuroraConfig(BaseModel):
    """Master configuration for the AURORA unlearning pipeline."""

    # --- Model ---
    model_name: str = Field(
        default="gpt2",
        description="HuggingFace model identifier. Use 'gpt2' for dev, 7B model for demo."
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Compute device. AUTO selects CUDA if available."
    )
    use_half_precision: bool = Field(
        default=False,
        description="Load model in float16 to halve memory. Recommended for 2B+ models on CPU."
    )

    # --- Formal Bounds ---
    alpha: float = Field(
        default=0.01,
        ge=0.0,
        description="Allowable utility drift — max KL(P_θ || P_θ') on retain set."
    )
    epsilon: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Acceptable leakage bound — max P(y_t | x_indirect, θ')."
    )

    # --- Optimizer Weights ---
    lambda_retain: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for retain loss in total objective."
    )
    gamma_stability: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight for L2 stability regularization."
    )

    # --- Parameter Localization ---
    top_k_params: int = Field(
        default=1000,
        ge=1,
        description="Number of highest-risk parameters to select for modification."
    )

    # --- Knowledge Graph ---
    knn_k: int = Field(
        default=10,
        ge=1,
        description="Number of nearest neighbors for kNN graph construction."
    )

    # --- Cascade ---
    cascade_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Influence score threshold to add node to cascade queue."
    )
    max_cascade_iterations: int = Field(
        default=50,
        ge=1,
        description="Maximum cascade expansion iterations."
    )

    # --- Unlearning Optimizer ---
    unlearn_lr: float = Field(
        default=1e-4,
        gt=0.0,
        description="Learning rate for unlearning optimizer."
    )
    unlearn_epochs: int = Field(
        default=10,
        ge=1,
        description="Maximum unlearning optimization epochs."
    )

    # --- Adversarial Suite ---
    adversarial_finetune_epochs: int = Field(
        default=3,
        ge=1,
        description="Epochs for correlated fine-tuning attack."
    )
    adversarial_finetune_lr: float = Field(
        default=5e-5,
        gt=0.0,
        description="Learning rate for adversarial fine-tuning."
    )
    num_prompt_perturbations: int = Field(
        default=20,
        ge=1,
        description="Number of adversarial prompt transformations to test."
    )
    multihop_max_depth: int = Field(
        default=3,
        ge=1,
        description="Maximum reasoning chain depth for multi-hop reconstruction."
    )
    quantization_bits: list[int] = Field(
        default=[8, 4],
        description="Bit widths to test in quantization attack."
    )
    lora_rank: int = Field(
        default=8,
        ge=1,
        description="LoRA rank for adaptation attack."
    )

    # --- Certificate ---
    certificate_output_dir: str = Field(
        default="certificates",
        description="Directory to store generated certificates."
    )
    rsa_key_size: int = Field(
        default=2048,
        ge=2048,
        description="RSA key size for certificate signing."
    )

    # --- Quantum Distinguishability ---
    distinguishability_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum trace distance ratio D_forget/D_retain for validation."
    )

    # --- Storage ---
    storage_dir: str = Field(
        default="aurora_storage",
        description="Base directory for logs, graphs, and artifacts."
    )

    def get_device(self) -> str:
        """Resolve device string."""
        if self.device == DeviceType.AUTO:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
            return "cpu"
        return self.device.value
