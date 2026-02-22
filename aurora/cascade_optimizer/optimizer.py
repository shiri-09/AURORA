"""
AURORA Module 4 — Cascade Unlearning Optimizer

The core engine that iteratively:
1. Updates parameters in θ_risk
2. Recomputes leakage probability
3. Expands cascade if leakage persists

Convergence: max_{x_indirect} P(y_t | x_indirect, θ') ≤ ε
"""

from __future__ import annotations

import copy
from typing import Optional

import networkx as nx
import numpy as np
import torch

from aurora.config import AuroraConfig
from aurora.types import (
    ForgetSet,
    RetainSet,
    RiskParameters,
    EvaluationMetrics,
)
from aurora.cascade_optimizer.losses import compute_total_loss
from aurora.parameter_localization.fisher_selector import create_parameter_mask
from aurora.utils.math_ops import compute_leakage_probability
from aurora.utils.logging import optimizer_logger

logger = optimizer_logger()


class CascadeOptimizer:
    """
    Relational Cascade Unlearning Optimizer.

    Performs iterative, parameter-masked gradient descent to suppress
    both direct and indirect knowledge leakage, expanding the unlearning
    scope through the RKG when indirect leakage persists.

    Usage:
        optimizer = CascadeOptimizer(model, tokenizer, config)
        result = optimizer.run(forget_set, retain_set, graph, risk_params)
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: AuroraConfig,
        original_model=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.get_device()

        # Reuse original model from orchestrator if provided (saves ~3GB)
        if original_model is not None:
            self.original_model = original_model
        else:
            self.original_model = copy.deepcopy(model)
        self.original_model.eval()
        for param in self.original_model.parameters():
            param.requires_grad = False

        # Snapshot original parameters for stability loss (only risk params)
        self.original_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

    def run(
        self,
        forget_set: ForgetSet,
        retain_set: RetainSet,
        graph: nx.DiGraph,
        risk_params: RiskParameters,
    ) -> tuple[EvaluationMetrics, int]:
        """
        Execute the cascade unlearning optimization.

        Args:
            forget_set: Target fact + prompts to forget.
            retain_set: Prompts to preserve.
            graph: Relational Knowledge Graph.
            risk_params: Localized high-risk parameters.

        Returns:
            Tuple of (EvaluationMetrics, cascade_iterations).
        """
        logger.info("═══ STARTING CASCADE UNLEARNING ═══")

        # Prepare data
        forget_prompts = [
            (p.prompt, p.target_answer) for p in forget_set.prompts
        ]
        retain_prompts = [p.prompt for p in retain_set.prompts]

        # Ensure we have retain prompts (use generic ones if empty)
        if not retain_prompts:
            retain_prompts = [
                "The capital of France is", "Water boils at",
                "The sun is a", "Mathematics is the study of",
            ]
            logger.warning("No retain prompts provided, using generic fallbacks")

        # Create parameter mask
        masks = create_parameter_mask(self.model, risk_params)

        # Cascade loop
        cascade_iteration = 0
        converged = False

        # Set up optimizer (only for risk parameters)
        self._setup_optimizer(risk_params)

        while cascade_iteration < self.config.max_cascade_iterations and not converged:
            cascade_iteration += 1
            logger.info(f"── Cascade Iteration {cascade_iteration} ──")

            # Run optimization epochs
            epoch_losses = self._optimize_epoch(
                forget_prompts, retain_prompts, masks
            )

            # Compute leakage
            direct_leakage = self._compute_direct_leakage(forget_set)
            indirect_leakage = self._compute_indirect_leakage(forget_set)
            retain_drift = self._compute_retain_drift(retain_prompts)

            logger.info(
                f"  Direct leakage:   {direct_leakage:.6f}\n"
                f"  Indirect leakage: {indirect_leakage:.6f}\n"
                f"  Retain drift:     {retain_drift:.6f}"
            )

            # Check convergence
            if indirect_leakage <= self.config.epsilon:
                logger.info(
                    f"✓ CONVERGED: indirect leakage {indirect_leakage:.6f} "
                    f"≤ ε={self.config.epsilon}"
                )
                converged = True
            else:
                # Expand cascade — add more parameters
                logger.info("Expanding cascade scope...")
                risk_params = self._expand_cascade(
                    risk_params, graph, forget_set, indirect_leakage
                )
                masks = create_parameter_mask(self.model, risk_params)
                self._setup_optimizer(risk_params)

        if not converged:
            logger.warning(
                f"⚠ Did not converge after {cascade_iteration} iterations. "
                f"Indirect leakage: {indirect_leakage:.6f} > ε={self.config.epsilon}"
            )

        # Final metrics
        metrics = self._compute_final_metrics(forget_set, retain_prompts)

        logger.info(f"═══ CASCADE COMPLETE ({cascade_iteration} iterations) ═══")
        return metrics, cascade_iteration

    def _setup_optimizer(self, risk_params: RiskParameters) -> None:
        """Configure optimizer for risk parameters only."""
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            if name in risk_params.parameter_names:
                param.requires_grad = True
                params_to_optimize.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            params_to_optimize, lr=self.config.unlearn_lr
        )

    def _optimize_epoch(
        self,
        forget_prompts: list[tuple[str, str]],
        retain_prompts: list[str],
        masks: dict[str, torch.Tensor],
    ) -> list[dict[str, float]]:
        """Run one epoch of unlearning optimization with masked gradients."""
        epoch_losses = []

        for epoch in range(self.config.unlearn_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Compute total loss
            loss, breakdown = compute_total_loss(
                model=self.model,
                original_model=self.original_model,
                tokenizer=self.tokenizer,
                forget_prompts=forget_prompts,
                retain_prompts=retain_prompts,
                original_params=self.original_params,
                lambda_retain=self.config.lambda_retain,
                gamma_stability=self.config.gamma_stability,
                device=self.device,
            )

            # Backpropagate
            loss.backward()

            # Apply parameter masks (zero out gradients outside θ_risk)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in masks:
                        param.grad *= masks[name].to(self.device)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            # Step
            self.optimizer.step()
            epoch_losses.append(breakdown)

            if (epoch + 1) % max(1, self.config.unlearn_epochs // 3) == 0:
                logger.info(
                    f"  Epoch {epoch + 1}/{self.config.unlearn_epochs}: "
                    f"total={breakdown['total_loss']:.4f} "
                    f"(forget={breakdown['forget_loss']:.4f}, "
                    f"retain={breakdown['retain_loss']:.4f}, "
                    f"stability={breakdown['stability_loss']:.4f})"
                )

        self.model.eval()
        return epoch_losses

    def _compute_direct_leakage(self, forget_set: ForgetSet) -> float:
        """Compute maximum leakage across direct prompts."""
        direct_prompts = [p.prompt for p in forget_set.direct_prompts]
        if not direct_prompts:
            return 0.0
        return compute_leakage_probability(
            self.model, self.tokenizer, direct_prompts,
            forget_set.fact.obj, self.device,
        )

    def _compute_indirect_leakage(self, forget_set: ForgetSet) -> float:
        """Compute maximum leakage across indirect/multi-hop prompts."""
        indirect_prompts = [p.prompt for p in forget_set.indirect_prompts]
        if not indirect_prompts:
            return 0.0
        return compute_leakage_probability(
            self.model, self.tokenizer, indirect_prompts,
            forget_set.fact.obj, self.device,
        )

    def _compute_retain_drift(self, retain_prompts: list[str]) -> float:
        """Compute average KL divergence on retain set."""
        if not retain_prompts:
            return 0.0

        total_kl = 0.0
        count = 0

        for text in retain_prompts[:20]:  # Limit for speed
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256,
            ).to(self.device)

            with torch.no_grad():
                original_logits = self.original_model(**inputs).logits
                current_logits = self.model(**inputs).logits

                p = torch.softmax(original_logits, dim=-1)
                log_p = torch.log_softmax(original_logits, dim=-1)
                log_q = torch.log_softmax(current_logits, dim=-1)

                kl = (p * (log_p - log_q)).sum(dim=-1).mean().item()
                total_kl += max(kl, 0.0)
                count += 1

        return total_kl / max(count, 1)

    def _expand_cascade(
        self,
        risk_params: RiskParameters,
        graph: nx.DiGraph,
        forget_set: ForgetSet,
        current_leakage: float,
    ) -> RiskParameters:
        """
        Expand the cascade by adding more parameters based on
        high-influence graph nodes.
        """
        # Increase top_k by 50% each iteration
        new_k = int(risk_params.total_params_selected * 1.5)
        new_k = min(new_k, sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        ))

        logger.info(
            f"  Expanding θ_risk: {risk_params.total_params_selected} → {new_k} params"
        )

        # Add parameters from high-influence nodes in the graph
        node_scores = []
        for node_id, data in graph.nodes(data=True):
            score = (
                data.get("attention_importance_score", 0) +
                data.get("gradient_sensitivity_score", 0)
            )
            node_scores.append((node_id, score))
        node_scores.sort(key=lambda x: x[1], reverse=True)

        # Update gradient sensitivity in graph
        for node_id, _ in node_scores[:5]:
            graph.nodes[node_id]["gradient_sensitivity_score"] = (
                graph.nodes[node_id].get("gradient_sensitivity_score", 0) + 0.1
            )

        # Create expanded risk parameters (add more from existing gradient info)
        expanded_indices = {}
        expanded_scores = {}

        for name in risk_params.parameter_names:
            expanded_indices[name] = risk_params.parameter_indices[name]
            expanded_scores[name] = risk_params.influence_scores[name]

        # Add parameters not yet in risk set
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in expanded_indices:
                # Add with small index set
                num_to_add = min(50, param.numel())
                indices = np.random.choice(param.numel(), num_to_add, replace=False)
                expanded_indices[name] = indices
                expanded_scores[name] = current_leakage * 0.1

        return RiskParameters(
            parameter_names=list(expanded_indices.keys()),
            parameter_indices=expanded_indices,
            influence_scores=expanded_scores,
        )

    def _compute_final_metrics(
        self,
        forget_set: ForgetSet,
        retain_prompts: list[str],
    ) -> EvaluationMetrics:
        """Compute comprehensive evaluation metrics after unlearning."""
        direct_leakage = self._compute_direct_leakage(forget_set)
        indirect_leakage = self._compute_indirect_leakage(forget_set)
        retain_drift = self._compute_retain_drift(retain_prompts)

        # Parameter drift norm
        drift_norm = 0.0
        for name, param in self.model.named_parameters():
            if name in self.original_params:
                diff = param.data.cpu() - self.original_params[name].cpu()
                drift_norm += (diff ** 2).sum().item()
        drift_norm = float(np.sqrt(drift_norm))

        return EvaluationMetrics(
            direct_forget_accuracy=direct_leakage,
            indirect_leakage_rate=indirect_leakage,
            retain_utility_drop=retain_drift,
            reconstruction_probability=max(direct_leakage, indirect_leakage),
            parameter_drift_norm=drift_norm,
        )
