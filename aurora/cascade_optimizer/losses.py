"""
AURORA Module 4 — Unlearning Loss Functions

L_total = L_forget + λ·L_retain + γ·‖θ−θ₀‖²

Where:
    L_forget = E_{x ∈ D_f}[-log(1 - P(y_t|x))]
    L_retain = E_{x ∈ D_r}[KL(P_θ || P_θ')]
    Stability = L2 regularization
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from aurora.utils.logging import optimizer_logger

logger = optimizer_logger()


def forget_loss(
    model,
    tokenizer,
    forget_prompts: list[tuple[str, str]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute forget loss: E[-log(1 - P(y_t|x))]

    Maximizes the probability of NOT generating the target.

    Args:
        model: Current model being unlearned.
        tokenizer: Tokenizer.
        forget_prompts: List of (prompt, target_answer) pairs.
        device: Compute device.

    Returns:
        Scalar loss tensor.
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0

    for prompt, target in forget_prompts:
        full_text = prompt + " " + target
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        target_ids = full_ids[len(prompt_ids):]

        if not target_ids:
            continue

        input_ids = torch.tensor([full_ids], device=device)
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Compute probability for target tokens
        loss_per_token = torch.tensor(0.0, device=device)
        num_target_tokens = 0

        for i, tid in enumerate(target_ids):
            pos = len(prompt_ids) + i - 1
            if 0 <= pos < logits.shape[1]:
                # P(target_token | context)
                probs = torch.softmax(logits[0, pos], dim=-1)
                p_target = probs[tid]

                # -log(1 - P(y_t|x)) — want to push P toward 0
                # Clamp to avoid log(0)
                loss_per_token = loss_per_token + (-torch.log(1.0 - p_target + 1e-8))
                num_target_tokens += 1

        if num_target_tokens > 0:
            total_loss = total_loss + loss_per_token / num_target_tokens
            count += 1

    if count > 0:
        total_loss = total_loss / count

    return total_loss


def retain_loss(
    model,
    original_model,
    tokenizer,
    retain_prompts: list[str],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute retain loss: E[KL(P_θ || P_θ')]

    Minimizes distribution shift on the retain set.

    Args:
        model: Current model (θ').
        original_model: Original model (θ).
        tokenizer: Tokenizer.
        retain_prompts: List of retain set texts.
        device: Compute device.

    Returns:
        Scalar KL divergence loss tensor.
    """
    total_kl = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0

    original_model.eval()

    for text in retain_prompts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        # Get logits from both models
        with torch.no_grad():
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits

        current_outputs = model(**inputs)
        current_logits = current_outputs.logits

        # KL(P_θ || P_θ') per position, averaged
        log_p = F.log_softmax(original_logits, dim=-1)
        p = F.softmax(original_logits, dim=-1)
        log_q = F.log_softmax(current_logits, dim=-1)

        kl = (p * (log_p - log_q)).sum(dim=-1).mean()
        total_kl = total_kl + kl
        count += 1

    if count > 0:
        total_kl = total_kl / count

    return total_kl


def stability_loss(
    model,
    original_params: dict[str, torch.Tensor],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute stability loss: ‖θ - θ₀‖²

    L2 regularization to prevent catastrophic parameter drift.

    Args:
        model: Current model.
        original_params: Snapshot of original parameters.
        device: Compute device.

    Returns:
        Scalar L2 norm tensor.
    """
    total = torch.tensor(0.0, device=device, requires_grad=True)

    for name, param in model.named_parameters():
        if name in original_params and param.requires_grad:
            diff = param - original_params[name].to(device)
            total = total + (diff ** 2).sum()

    return total


def compute_total_loss(
    model,
    original_model,
    tokenizer,
    forget_prompts: list[tuple[str, str]],
    retain_prompts: list[str],
    original_params: dict[str, torch.Tensor],
    lambda_retain: float,
    gamma_stability: float,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute L_total = L_forget + λ·L_retain + γ·‖θ−θ₀‖²

    Returns:
        Tuple of (total_loss, loss_breakdown_dict).
    """
    l_forget = forget_loss(model, tokenizer, forget_prompts, device)
    l_retain = retain_loss(model, original_model, tokenizer, retain_prompts, device)
    l_stability = stability_loss(model, original_params, device)

    total = l_forget + lambda_retain * l_retain + gamma_stability * l_stability

    breakdown = {
        "forget_loss": l_forget.item(),
        "retain_loss": l_retain.item(),
        "stability_loss": l_stability.item(),
        "total_loss": total.item(),
    }

    return total, breakdown
