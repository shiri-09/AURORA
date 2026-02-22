"""
AURORA Module 3 — Fisher Information Selector

Approximate curvature: F = E[∇_θ L ∇_θ L^T]
Select top-k parameters by: |g_agg| / sqrt(diag(F))

This localizes the high-risk weight subspace θ_risk.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from aurora.config import AuroraConfig
from aurora.types import RiskParameters
from aurora.utils.math_ops import fisher_information_diagonal
from aurora.utils.logging import localization_logger

logger = localization_logger()


def select_risk_parameters(
    model,
    tokenizer,
    g_agg: dict[str, torch.Tensor],
    retain_texts: list[str],
    config: AuroraConfig,
) -> RiskParameters:
    """
    Select the top-k highest-risk parameters using Fisher-weighted gradient scores.

    Score for each parameter element:
        score = |g_agg| / sqrt(diag(F) + eps)

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        g_agg: Aggregated gradient dict from path_aggregator.
        retain_texts: Retain set texts for Fisher computation.
        config: AURORA configuration.

    Returns:
        RiskParameters identifying θ_risk.
    """
    device = config.get_device()

    logger.info("Computing Fisher Information diagonal...")
    fisher_diag = fisher_information_diagonal(
        model, tokenizer, retain_texts,
        device=device, max_samples=min(100, len(retain_texts)),
    )

    logger.info(f"Selecting top-{config.top_k_params} risk parameters...")

    # Memory-efficient top-k: instead of building list of ALL ~774M scores,
    # collect top-k candidates per parameter, then merge.
    k = config.top_k_params
    candidates: list[tuple[str, int, float]] = []  # (param_name, flat_index, score)

    for name, grad in g_agg.items():
        grad_np = grad.cpu().numpy().flatten()
        fisher_np = fisher_diag.get(name, np.ones_like(grad_np))
        fisher_np = fisher_np.flatten()

        # Ensure shapes match
        min_len = min(len(grad_np), len(fisher_np))
        grad_np = grad_np[:min_len]
        fisher_np = fisher_np[:min_len]

        # score = |g_agg| / sqrt(diag(F) + eps)
        scores = np.abs(grad_np) / np.sqrt(fisher_np + 1e-8)

        # Take per-parameter top-k using argpartition (O(n) instead of O(n log n))
        param_k = min(k, len(scores))
        if param_k > 0:
            top_indices = np.argpartition(scores, -param_k)[-param_k:]
            for idx in top_indices:
                candidates.append((name, int(idx), float(scores[idx])))

    # Sort all candidates and take global top-k
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Select top-k
    top_k = min(k, len(candidates))
    selected = candidates[:top_k]

    # Group by parameter name
    param_indices: dict[str, list[int]] = {}
    influence_scores: dict[str, float] = {}

    for name, idx, score in selected:
        if name not in param_indices:
            param_indices[name] = []
            influence_scores[name] = 0.0
        param_indices[name].append(idx)
        influence_scores[name] += score

    # Convert to numpy arrays
    param_indices_np = {
        name: np.array(indices, dtype=np.int64)
        for name, indices in param_indices.items()
    }

    risk_params = RiskParameters(
        parameter_names=list(param_indices.keys()),
        parameter_indices=param_indices_np,
        influence_scores=influence_scores,
    )

    logger.info(
        f"Selected θ_risk: {risk_params.total_params_selected} parameters "
        f"across {len(risk_params.parameter_names)} groups"
    )

    # Log top-5 most sensitive parameter groups
    sorted_groups = sorted(
        influence_scores.items(), key=lambda x: x[1], reverse=True
    )[:5]
    for name, score in sorted_groups:
        logger.info(f"  {name}: influence={score:.6f}, count={len(param_indices[name])}")

    return risk_params


def create_parameter_mask(
    model,
    risk_params: RiskParameters,
) -> dict[str, torch.Tensor]:
    """
    Create binary masks for each parameter, with 1s at risk indices.

    Used by the cascade optimizer to restrict gradient updates.

    Args:
        model: Model to create masks for.
        risk_params: Selected risk parameters.

    Returns:
        Dict mapping parameter names to binary mask tensors.
    """
    masks = {}

    for name, param in model.named_parameters():
        if name in risk_params.parameter_indices:
            mask = torch.zeros_like(param.data).flatten()
            indices = risk_params.parameter_indices[name]
            indices = indices[indices < len(mask)]  # bounds check
            mask[indices] = 1.0
            masks[name] = mask.reshape(param.data.shape)
        else:
            masks[name] = torch.zeros_like(param.data)

    return masks
