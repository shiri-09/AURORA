"""
AURORA Module 3 — Gradient Extraction Engine

Computes parameter gradients for target facts.

g_t = ∇_θ L_target where L_target = -log P(y_t | x_direct)
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np

from aurora.utils.logging import localization_logger

logger = localization_logger()


def compute_target_gradient(
    model,
    tokenizer,
    prompt: str,
    target_answer: str,
    device: str = "cpu",
    param_filter: Optional[list[str]] = None,
) -> dict[str, torch.Tensor]:
    """
    Compute gradient of the target loss w.r.t. model parameters.

    L_target = -log P(y_t | x_direct)
    g_t = ∇_θ L_target

    Args:
        model: HuggingFace language model.
        tokenizer: Corresponding tokenizer.
        prompt: Direct prompt for the target fact.
        target_answer: The answer that should be forgotten.
        device: Compute device.
        param_filter: Optional list of parameter name patterns to include.
            If None, all parameters with requires_grad=True are included.

    Returns:
        Dict mapping parameter names to their gradient tensors.
    """
    model.train()
    model.zero_grad()

    # Encode prompt + target as a sequence
    full_text = prompt + " " + target_answer
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    # Compute loss (language modeling loss on the full sequence)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    # Backpropagate
    loss.backward()

    logger.info(f"Target loss = {loss.item():.6f}")

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        # Apply filter if specified
        if param_filter is not None:
            if not any(f in name for f in param_filter):
                continue

        gradients[name] = param.grad.detach().clone()

    model.zero_grad()
    model.eval()

    logger.info(f"Extracted gradients for {len(gradients)} parameter groups")
    return gradients


def compute_gradient_norms(
    gradients: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compute L2 norm for each parameter's gradient.

    Args:
        gradients: Dict of parameter name → gradient tensor.

    Returns:
        Dict of parameter name → gradient norm.
    """
    norms = {}
    for name, grad in gradients.items():
        norms[name] = float(torch.norm(grad).item())
    return norms


def compute_gradient_sensitivity_per_layer(
    gradients: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Aggregate gradient sensitivity by transformer layer.

    Groups parameters by their layer number and computes
    total gradient norm per layer.

    Returns:
        Dict mapping layer identifier to sensitivity score.
    """
    layer_sensitivity = {}

    for name, grad in gradients.items():
        # Extract layer identifier (e.g., "transformer.h.0" or "model.layers.0")
        parts = name.split(".")
        layer_key = None
        for i, part in enumerate(parts):
            if part.isdigit():
                layer_key = ".".join(parts[:i + 1])
                break

        if layer_key is None:
            layer_key = "other"

        if layer_key not in layer_sensitivity:
            layer_sensitivity[layer_key] = 0.0
        layer_sensitivity[layer_key] += float(torch.norm(grad).item())

    return dict(sorted(layer_sensitivity.items(), key=lambda x: x[1], reverse=True))
