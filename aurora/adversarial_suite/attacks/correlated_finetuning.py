"""
AURORA Attack 1 — Correlated Fine-Tuning

Fine-tune on related documents excluding the target fact.
Measure if the model re-learns the association.

P_reconstruct^after = P(y_t | x_direct, θ_finetuned)
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from aurora.config import AuroraConfig
from aurora.utils.embeddings import get_token_probabilities
from aurora.utils.logging import adversarial_logger

logger = adversarial_logger()


def correlated_finetuning_attack(
    model,
    tokenizer,
    correlated_texts: list[str],
    direct_prompt: str,
    target_answer: str,
    config: AuroraConfig,
) -> dict[str, float]:
    """
    Attack 1: Fine-tune on correlated data and measure reconstruction.

    Fine-tunes a copy of the unlearned model on texts related to (but
    not containing) the target fact. Checks if the association is recovered.

    Args:
        model: Unlearned model.
        tokenizer: Tokenizer.
        correlated_texts: Related documents (e.g., about the subject but
            not mentioning the specific fact).
        direct_prompt: Direct query for the target fact.
        target_answer: The forgotten answer.
        config: AURORA configuration.

    Returns:
        Dict with 'leakage_before', 'leakage_after', 'recovery_rate'.
    """
    device = config.get_device()

    logger.info("Running Attack 1: Correlated Fine-Tuning...")

    # Measure leakage before attack
    leakage_before = get_token_probabilities(
        model, tokenizer, direct_prompt, target_answer, device
    )

    # Create a copy to fine-tune
    attack_model = copy.deepcopy(model)
    attack_model.train()

    optimizer = torch.optim.Adam(
        attack_model.parameters(), lr=config.adversarial_finetune_lr
    )

    # Fine-tune on correlated data
    for epoch in range(config.adversarial_finetune_epochs):
        total_loss = 0.0
        for text in correlated_texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            ).to(device)

            optimizer.zero_grad()
            outputs = attack_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(correlated_texts), 1)
        logger.info(f"  Epoch {epoch + 1}/{config.adversarial_finetune_epochs}: loss={avg_loss:.4f}")

    # Measure leakage after attack
    attack_model.eval()
    leakage_after = get_token_probabilities(
        attack_model, tokenizer, direct_prompt, target_answer, device
    )

    recovery_rate = max(0, leakage_after - leakage_before)

    logger.info(
        f"  Before: {leakage_before:.6f}, After: {leakage_after:.6f}, "
        f"Recovery: {recovery_rate:.6f}"
    )

    # Clean up
    del attack_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "leakage_before": leakage_before,
        "leakage_after": leakage_after,
        "recovery_rate": recovery_rate,
    }
