"""
AURORA Mathematical Operations

Core mathematical functions used across modules.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import Optional


def kl_divergence(
    p_logits: Tensor,
    q_logits: Tensor,
    dim: int = -1,
) -> Tensor:
    """
    Compute KL(P || Q) from logits.

    Args:
        p_logits: Logits from reference distribution P.
        q_logits: Logits from target distribution Q.
        dim: Dimension to compute softmax over.

    Returns:
        Scalar KL divergence tensor.
    """
    p = torch.softmax(p_logits, dim=dim)
    log_p = torch.log_softmax(p_logits, dim=dim)
    log_q = torch.log_softmax(q_logits, dim=dim)

    kl = (p * (log_p - log_q)).sum(dim=dim)
    return kl.mean()


def l2_norm(params1: dict[str, Tensor], params2: dict[str, Tensor]) -> Tensor:
    """
    Compute L2 norm ||θ - θ₀|| between two parameter dicts.

    Args:
        params1: Current parameters.
        params2: Original parameters.

    Returns:
        Scalar L2 norm tensor.
    """
    total = torch.tensor(0.0)
    for name in params1:
        if name in params2:
            diff = params1[name].float() - params2[name].float()
            total = total + (diff ** 2).sum()
    return torch.sqrt(total)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector. Returns zero vector if norm is zero.

    Args:
        v: Input vector.

    Returns:
        Unit vector or zero vector.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k highest values.

    Args:
        scores: 1D array of scores.
        k: Number of indices to return.

    Returns:
        Array of indices (sorted by descending score).
    """
    k = min(k, len(scores))
    return np.argsort(scores)[-k:][::-1]


def compute_leakage_probability(
    model,
    tokenizer,
    prompts: list[str],
    target: str,
    device: str = "cpu",
) -> float:
    """
    Compute maximum leakage probability across a set of prompts.

    max_x P(target | x, θ')

    Args:
        model: Modified model.
        tokenizer: Tokenizer.
        prompts: List of direct + indirect prompts.
        target: Target answer string.
        device: Compute device.

    Returns:
        Maximum probability of generating the target across all prompts.
    """
    from aurora.utils.embeddings import get_token_probabilities

    max_prob = 0.0
    for prompt in prompts:
        prob = get_token_probabilities(model, tokenizer, prompt, target, device)
        max_prob = max(max_prob, prob)
    return max_prob


def fisher_information_diagonal(
    model,
    tokenizer,
    data: list[str],
    device: str = "cpu",
    max_samples: int = 100,
) -> dict[str, np.ndarray]:
    """
    Compute diagonal Fisher Information Matrix approximation.

    F = E[∇θL ∇θL^T] ≈ diag of squared gradients

    Args:
        model: Model to analyze.
        tokenizer: Tokenizer.
        data: List of text samples.
        device: Compute device.
        max_samples: Maximum samples to use.

    Returns:
        Dict mapping parameter names to Fisher diagonal values.
    """
    model.eval()
    fisher_diag: dict[str, torch.Tensor] = {}

    # Initialize
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_diag[name] = torch.zeros_like(param.data)

    samples = data[:max_samples]
    n = len(samples)

    for text in samples:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        model.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_diag[name] += (param.grad.data ** 2) / n

    return {name: val.cpu().numpy() for name, val in fisher_diag.items()}
