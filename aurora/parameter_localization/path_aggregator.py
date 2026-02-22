"""
AURORA Module 3 — Path-Aggregated Gradient Computation

For each reasoning path p in P_target:
    g_p = Σ_{x ∈ p} ∇_θ L(x)

Aggregate:
    g_agg = Σ_p w_p * g_p
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np
import networkx as nx

from aurora.parameter_localization.gradient_engine import compute_target_gradient
from aurora.types import ForgetSet
from aurora.utils.logging import localization_logger

logger = localization_logger()


def compute_path_aggregated_gradient(
    model,
    tokenizer,
    forget_set: ForgetSet,
    graph: nx.DiGraph,
    device: str = "cpu",
    max_paths: int = 20,
    param_filter: Optional[list[str]] = None,
) -> dict[str, torch.Tensor]:
    """
    Compute path-aggregated gradient across all reasoning paths.

    g_agg = Σ_p w_p * g_p

    where:
        - Each path p is a sequence of prompts from the forget set
        - w_p is the normalized path contribution weight (from RKG edge weights)
        - g_p = Σ_{x ∈ p} ∇_θ L(x)

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        forget_set: ForgetSet with direct + indirect prompts.
        graph: RKG for extracting path weights.
        device: Compute device.
        max_paths: Maximum number of paths to consider.
        param_filter: Optional parameter name filter patterns.

    Returns:
        Dict mapping parameter names to aggregated gradient tensors.
    """
    logger.info("Computing path-aggregated gradients...")

    # Extract paths and their weights from the RKG
    paths_with_weights = _extract_reasoning_paths(graph, forget_set, max_paths)

    if not paths_with_weights:
        # Fallback: use all prompts with uniform weight
        logger.warning("No paths extracted from RKG, using uniform weights")
        all_prompts = [(p.prompt, p.target_answer) for p in forget_set.prompts]
        paths_with_weights = [(all_prompts, 1.0 / len(all_prompts))]

    # Aggregate gradients
    g_agg: dict[str, torch.Tensor] = {}
    total_weight = sum(w for _, w in paths_with_weights)

    for path_idx, (path_prompts, weight) in enumerate(paths_with_weights):
        w_normalized = weight / max(total_weight, 1e-8)

        logger.info(
            f"Path {path_idx + 1}/{len(paths_with_weights)}: "
            f"{len(path_prompts)} prompts, weight={w_normalized:.4f}"
        )

        # Compute g_p = Σ_{x ∈ p} ∇_θ L(x)
        for prompt_text, target_text in path_prompts:
            grads = compute_target_gradient(
                model, tokenizer, prompt_text, target_text,
                device=device, param_filter=param_filter,
            )

            for name, grad in grads.items():
                weighted_grad = grad * w_normalized
                if name in g_agg:
                    g_agg[name] = g_agg[name] + weighted_grad
                else:
                    g_agg[name] = weighted_grad.clone()

    logger.info(
        f"Aggregated gradients across {len(paths_with_weights)} paths, "
        f"{len(g_agg)} parameter groups"
    )
    return g_agg


def _extract_reasoning_paths(
    graph: nx.DiGraph,
    forget_set: ForgetSet,
    max_paths: int,
) -> list[tuple[list[tuple[str, str]], float]]:
    """
    Extract reasoning paths from the RKG.

    A reasoning path is a sequence of nodes (prompts) that connects
    an indirect query to the target fact through the graph.

    Returns:
        List of (path_prompts, path_weight) tuples.
        Each path_prompts is a list of (prompt, target) string pairs.
    """
    fact_label = f"fact_{forget_set.fact.fact_id}"
    target_answer = forget_set.fact.obj

    if fact_label not in graph.nodes:
        return []

    paths = []

    # For each prompt node, find path to fact node
    for i, prompt in enumerate(forget_set.prompts):
        source = f"prompt_{i}"
        if source not in graph.nodes:
            continue

        try:
            # Get shortest path
            path_nodes = nx.shortest_path(graph, source, fact_label)

            # Compute path weight as product of edge weights
            path_weight = 1.0
            for j in range(len(path_nodes) - 1):
                edge_data = graph.get_edge_data(path_nodes[j], path_nodes[j + 1])
                if edge_data:
                    path_weight *= edge_data.get("weight", 1.0)

            # Convert node path to (prompt, target) pairs
            path_prompts = [(prompt.prompt, target_answer)]

            # Add intermediate prompt nodes
            for node in path_nodes[1:-1]:
                if node.startswith("prompt_"):
                    idx = int(node.split("_")[1])
                    if idx < len(forget_set.prompts):
                        p = forget_set.prompts[idx]
                        path_prompts.append((p.prompt, p.target_answer))

            paths.append((path_prompts, path_weight))

        except nx.NetworkXNoPath:
            # No path exists; use direct connection with low weight
            paths.append(
                ([(prompt.prompt, target_answer)], 0.1)
            )

    # Sort by weight descending and limit
    paths.sort(key=lambda x: x[1], reverse=True)
    return paths[:max_paths]
