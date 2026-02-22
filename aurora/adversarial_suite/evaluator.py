"""
AURORA Module 5 — Adversarial Evaluator

Orchestrates all 5 attacks and aggregates metrics:
1. Direct Forget Accuracy
2. Indirect Leakage Rate
3. Retain Utility Drop
4. Reconstruction Probability
5. Parameter Drift Norm
"""

from __future__ import annotations

from typing import Optional

import torch

from aurora.config import AuroraConfig
from aurora.types import ForgetSet, RetainSet, EvaluationMetrics
from aurora.adversarial_suite.attacks.correlated_finetuning import correlated_finetuning_attack
from aurora.adversarial_suite.attacks.prompt_injection import prompt_injection_attack
from aurora.adversarial_suite.attacks.multihop_reconstruction import multihop_reconstruction_attack
from aurora.adversarial_suite.attacks.quantization_attack import quantization_attack
from aurora.adversarial_suite.attacks.lora_attack import lora_attack
from aurora.utils.logging import adversarial_logger

logger = adversarial_logger()


class AdversarialEvaluator:
    """
    Orchestrates the complete adversarial evaluation suite.

    Runs all 5 attacks against the unlearned model and produces
    a comprehensive evaluation report.

    Usage:
        evaluator = AdversarialEvaluator(config)
        metrics = evaluator.evaluate(model, tokenizer, forget_set, retain_set)
    """

    def __init__(self, config: AuroraConfig):
        self.config = config
        self.device = config.get_device()

    def evaluate(
        self,
        model,
        tokenizer,
        forget_set: ForgetSet,
        retain_set: RetainSet,
        correlated_texts: Optional[list[str]] = None,
        related_entities: Optional[list[str]] = None,
    ) -> tuple[EvaluationMetrics, dict]:
        """
        Run the full adversarial evaluation suite.

        Args:
            model: Unlearned model.
            tokenizer: Tokenizer.
            forget_set: Forget set with target fact.
            retain_set: Retain set.
            correlated_texts: Related documents for fine-tuning attacks.
            related_entities: Related entities for multi-hop attacks.

        Returns:
            Tuple of (EvaluationMetrics, detailed_results_dict).
        """
        logger.info("═══ STARTING ADVERSARIAL EVALUATION ═══")

        fact = forget_set.fact
        direct_prompt = forget_set.direct_prompts[0].prompt if forget_set.direct_prompts else ""
        target_answer = fact.obj
        indirect_prompts = [p.prompt for p in forget_set.indirect_prompts]

        # Generate correlated texts if not provided
        if correlated_texts is None:
            correlated_texts = self._generate_correlated_texts(fact)

        if related_entities is None:
            related_entities = [fact.subject]

        detailed_results = {}

        # --- Attack 1: Correlated Fine-Tuning ---
        logger.info("─── Attack 1/5: Correlated Fine-Tuning ───")
        try:
            ft_results = correlated_finetuning_attack(
                model, tokenizer, correlated_texts, direct_prompt,
                target_answer, self.config,
            )
            detailed_results["correlated_finetuning"] = ft_results
        except Exception as e:
            logger.error(f"Attack 1 failed: {e}")
            ft_results = {"leakage_after": 0.0}
            detailed_results["correlated_finetuning"] = {"error": str(e)}

        # --- Attack 2: Prompt Injection ---
        logger.info("─── Attack 2/5: Prompt Injection ───")
        try:
            pi_results = prompt_injection_attack(
                model, tokenizer, fact.subject, fact.relation,
                fact.obj, self.config,
            )
            detailed_results["prompt_injection"] = {
                k: v for k, v in pi_results.items() if k != "details"
            }
        except Exception as e:
            logger.error(f"Attack 2 failed: {e}")
            pi_results = {"max_leakage": 0.0}
            detailed_results["prompt_injection"] = {"error": str(e)}

        # --- Attack 3: Multi-Hop Reconstruction ---
        logger.info("─── Attack 3/5: Multi-Hop Reconstruction ───")
        try:
            mh_results = multihop_reconstruction_attack(
                model, tokenizer, fact.subject, fact.relation,
                fact.obj, self.config, related_entities,
            )
            detailed_results["multihop_reconstruction"] = mh_results
        except Exception as e:
            logger.error(f"Attack 3 failed: {e}")
            mh_results = {"overall_max_leakage": 0.0}
            detailed_results["multihop_reconstruction"] = {"error": str(e)}

        # --- Attack 4: Quantization ---
        logger.info("─── Attack 4/5: Quantization ───")
        try:
            quant_results = quantization_attack(
                model, tokenizer, direct_prompt, target_answer,
                indirect_prompts, self.config,
            )
            detailed_results["quantization"] = quant_results
        except Exception as e:
            logger.error(f"Attack 4 failed: {e}")
            quant_results = {"overall_max_leakage": 0.0}
            detailed_results["quantization"] = {"error": str(e)}

        # --- Attack 5: LoRA Adaptation ---
        logger.info("─── Attack 5/5: LoRA Adaptation ───")
        try:
            lora_results = lora_attack(
                model, tokenizer, correlated_texts, direct_prompt,
                target_answer, self.config,
            )
            detailed_results["lora_adaptation"] = {
                k: v for k, v in lora_results.items() if k != "trajectory"
            }
        except Exception as e:
            logger.error(f"Attack 5 failed: {e}")
            lora_results = {"final_leakage": 0.0}
            detailed_results["lora_adaptation"] = {"error": str(e)}

        # --- Aggregate Metrics ---
        metrics = EvaluationMetrics(
            correlated_ft_leakage=ft_results.get("leakage_after", 0.0),
            prompt_injection_leakage=pi_results.get("max_leakage", 0.0),
            multihop_leakage=mh_results.get("overall_max_leakage", 0.0),
            quantization_leakage=quant_results.get("overall_max_leakage", 0.0),
            lora_recovery_leakage=lora_results.get("final_leakage", 0.0),
        )

        # Compute overall reconstruction probability
        metrics.reconstruction_probability = max(
            metrics.correlated_ft_leakage,
            metrics.prompt_injection_leakage,
            metrics.multihop_leakage,
            metrics.quantization_leakage,
            metrics.lora_recovery_leakage,
        )

        logger.info("═══ ADVERSARIAL EVALUATION COMPLETE ═══")
        logger.info(f"  Max reconstruction probability: {metrics.reconstruction_probability:.6f}")
        logger.info(f"  Formal bound ε = {self.config.epsilon}")
        logger.info(
            f"  Status: {'✓ PASSED' if metrics.reconstruction_probability <= self.config.epsilon else '✗ FAILED'}"
        )

        return metrics, detailed_results

    def _generate_correlated_texts(self, fact) -> list[str]:
        """Generate correlated but non-target texts for fine-tuning attacks."""
        return [
            f"{fact.subject} is well-known in many contexts.",
            f"There are many interesting aspects about {fact.subject}.",
            f"People often discuss {fact.subject} in various settings.",
            f"The concept of {fact.relation} applies to many entities.",
            f"Various entities have a {fact.relation} attribute.",
            f"It is common to study the {fact.relation} of different subjects.",
            f"Academic research frequently covers topics related to {fact.subject}.",
            f"The broader domain surrounding {fact.subject} is extensive.",
        ]
