"""
AURORA Attack 4 — Quantization Attack

Quantize model to lower precision (8-bit, 4-bit).
Check if compressed noise reactivates target associations.
"""

from __future__ import annotations

from typing import Optional

import copy
import torch

from aurora.config import AuroraConfig
from aurora.utils.embeddings import get_token_probabilities
from aurora.utils.logging import adversarial_logger

logger = adversarial_logger()


def quantization_attack(
    model,
    tokenizer,
    direct_prompt: str,
    target_answer: str,
    indirect_prompts: list[str],
    config: AuroraConfig,
) -> dict[str, float]:
    """
    Attack 4: Quantize model and check if compression noise reactivates
    target associations.

    Simulates quantization by rounding weights to lower precision
    and measuring if the numerical noise brings back forgotten associations.

    Args:
        model: Unlearned model.
        tokenizer: Tokenizer.
        direct_prompt: Direct query for target.
        target_answer: Forgotten answer.
        indirect_prompts: Indirect queries.
        config: AURORA configuration.

    Returns:
        Dict with per-bitwidth leakage results.
    """
    device = config.get_device()
    logger.info("Running Attack 4: Quantization...")

    results = {}

    for bits in config.quantization_bits:
        logger.info(f"  Testing {bits}-bit quantization...")

        # Create quantized copy
        quantized_model = copy.deepcopy(model)
        _simulate_quantization(quantized_model, bits)
        quantized_model.eval()

        # Measure direct leakage
        direct_leakage = get_token_probabilities(
            quantized_model, tokenizer, direct_prompt, target_answer, device
        )

        # Measure indirect leakage
        indirect_leakages = []
        for prompt in indirect_prompts[:10]:  # Limit for speed
            prob = get_token_probabilities(
                quantized_model, tokenizer, prompt, target_answer, device
            )
            indirect_leakages.append(prob)

        max_indirect = max(indirect_leakages) if indirect_leakages else 0.0
        mean_indirect = sum(indirect_leakages) / max(len(indirect_leakages), 1)

        results[f"{bits}bit"] = {
            "direct_leakage": direct_leakage,
            "max_indirect_leakage": max_indirect,
            "mean_indirect_leakage": mean_indirect,
        }

        logger.info(
            f"    {bits}-bit: direct={direct_leakage:.6f}, "
            f"max_indirect={max_indirect:.6f}"
        )

        # Cleanup
        del quantized_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Overall max leakage across all bit widths
    overall_max = max(
        max(r["direct_leakage"], r["max_indirect_leakage"])
        for r in results.values()
    )

    return {
        "overall_max_leakage": overall_max,
        "per_bitwidth": results,
    }


def _simulate_quantization(model, bits: int) -> None:
    """
    Simulate quantization by rounding weights to specified bit precision.

    This is a simplified simulation — real quantization (e.g., GPTQ, AWQ)
    involves more sophisticated calibration, but this captures the core
    effect of precision loss on weight values.
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                # Compute quantization range
                w = param.data.float()
                w_min = w.min()
                w_max = w.max()

                # Number of quantization levels
                n_levels = 2 ** bits - 1
                if n_levels == 0:
                    continue

                # Scale and zero point
                scale = (w_max - w_min) / n_levels
                if scale == 0:
                    continue

                # Quantize: round to nearest level
                quantized = torch.round((w - w_min) / scale)
                quantized = quantized.clamp(0, n_levels)

                # Dequantize
                dequantized = quantized * scale + w_min

                param.data.copy_(dequantized.to(param.dtype))
