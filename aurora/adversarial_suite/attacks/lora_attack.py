"""
AURORA Attack 5 — Low-Rank Adaptation (LoRA) Attack

Apply LoRA on a correlated corpus and measure recovery speed.
Tests if low-rank parameter updates can reconstruct forgotten knowledge.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from aurora.config import AuroraConfig
from aurora.utils.embeddings import get_token_probabilities
from aurora.utils.logging import adversarial_logger

logger = adversarial_logger()


class LoRALayer(nn.Module):
    """Simple LoRA adapter for a linear layer."""

    def __init__(self, original_layer: nn.Linear, rank: int = 8):
        super().__init__()
        self.original = original_layer
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Freeze original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        original_output = self.original(x)
        lora_output = x @ self.lora_A @ self.lora_B
        return original_output + lora_output


def lora_attack(
    model,
    tokenizer,
    correlated_texts: list[str],
    direct_prompt: str,
    target_answer: str,
    config: AuroraConfig,
) -> dict[str, float]:
    """
    Attack 5: Apply LoRA adapters on correlated corpus and measure
    how quickly the model recovers the forgotten association.

    Args:
        model: Unlearned model.
        tokenizer: Tokenizer.
        correlated_texts: Related documents for LoRA training.
        direct_prompt: Direct query for target.
        target_answer: Forgotten answer.
        config: AURORA configuration.

    Returns:
        Dict with recovery trajectory and final leakage.
    """
    device = config.get_device()
    logger.info(f"Running Attack 5: LoRA Adaptation (rank={config.lora_rank})...")

    # Measure baseline leakage
    baseline_leakage = get_token_probabilities(
        model, tokenizer, direct_prompt, target_answer, device
    )

    # Create a copy and apply LoRA adapters
    attack_model = copy.deepcopy(model)
    lora_params = _apply_lora_adapters(attack_model, config.lora_rank)
    attack_model.to(device)

    # Only optimize LoRA parameters
    optimizer = torch.optim.Adam(lora_params, lr=config.adversarial_finetune_lr)

    # Fine-tune with LoRA
    recovery_trajectory = [baseline_leakage]

    for epoch in range(config.adversarial_finetune_epochs):
        attack_model.train()
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

        # Measure leakage after each epoch
        attack_model.eval()
        current_leakage = get_token_probabilities(
            attack_model, tokenizer, direct_prompt, target_answer, device
        )
        recovery_trajectory.append(current_leakage)

        avg_loss = total_loss / max(len(correlated_texts), 1)
        logger.info(
            f"  Epoch {epoch + 1}/{config.adversarial_finetune_epochs}: "
            f"loss={avg_loss:.4f}, leakage={current_leakage:.6f}"
        )

    final_leakage = recovery_trajectory[-1]
    recovery_rate = max(0, final_leakage - baseline_leakage)

    # Compute recovery speed (how fast leakage increases)
    recovery_speed = 0.0
    for i in range(1, len(recovery_trajectory)):
        delta = recovery_trajectory[i] - recovery_trajectory[i - 1]
        if delta > 0:
            recovery_speed += delta

    logger.info(
        f"  Final leakage: {final_leakage:.6f}, "
        f"Recovery rate: {recovery_rate:.6f}, "
        f"Recovery speed: {recovery_speed:.6f}"
    )

    # Cleanup
    del attack_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "baseline_leakage": baseline_leakage,
        "final_leakage": final_leakage,
        "recovery_rate": recovery_rate,
        "recovery_speed": recovery_speed,
        "trajectory": recovery_trajectory,
    }


def _apply_lora_adapters(model, rank: int) -> list[nn.Parameter]:
    """
    Apply LoRA adapters to attention projection layers.

    Returns list of LoRA parameters for optimization.
    """
    lora_params = []

    for name, module in model.named_modules():
        # Target attention projection layers
        if isinstance(module, nn.Linear) and any(
            key in name for key in ["q_proj", "v_proj", "c_attn", "attn.c_proj"]
        ):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            # Get parent module
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            # Replace with LoRA layer
            lora_layer = LoRALayer(module, rank=rank)
            setattr(parent, child_name, lora_layer)

            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    logger.info(f"  Applied LoRA to {len(lora_params) // 2} layers (rank={rank})")
    return lora_params
