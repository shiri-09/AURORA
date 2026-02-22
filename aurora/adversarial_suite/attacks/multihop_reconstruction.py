"""
AURORA Attack 3 — Multi-Hop Compositional Reconstruction

Chain reasoning queries to attempt compositional reconstruction
of the forgotten fact through indirect pathways.
"""

from __future__ import annotations

from typing import Optional

import torch

from aurora.config import AuroraConfig
from aurora.utils.embeddings import get_token_probabilities
from aurora.utils.logging import adversarial_logger

logger = adversarial_logger()


def multihop_reconstruction_attack(
    model,
    tokenizer,
    subject: str,
    relation: str,
    obj: str,
    config: AuroraConfig,
    related_entities: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Attack 3: Chain multi-hop reasoning queries.

    Constructs reasoning chains of increasing depth that attempt
    to reconstruct the target fact through indirect associations.

    Args:
        model: Unlearned model.
        tokenizer: Tokenizer.
        subject: Subject entity.
        relation: Relation type.
        obj: Target answer (forgotten).
        config: AURORA configuration.
        related_entities: Optional list of entities related to the subject.

    Returns:
        Dict with per-depth leakage and overall max leakage.
    """
    device = config.get_device()
    logger.info("Running Attack 3: Multi-Hop Compositional Reconstruction...")

    if related_entities is None:
        related_entities = [subject]

    max_depth = config.multihop_max_depth
    depth_results = {}
    overall_max = 0.0

    for depth in range(1, max_depth + 1):
        chains = _generate_chains(subject, relation, obj, related_entities, depth)
        depth_max = 0.0
        depth_leakages = []

        for chain in chains:
            # Execute each chain step and accumulate context
            context = ""
            for step_prompt in chain[:-1]:
                inputs = tokenizer(
                    context + step_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    # Get the most likely continuation
                    next_token_logits = outputs.logits[0, -1, :]
                    top_tokens = torch.topk(next_token_logits, 20).indices
                    continuation = tokenizer.decode(top_tokens[:5])
                    context += step_prompt + " " + continuation + ". "

            # Final step: check if target leaks
            final_prompt = context + chain[-1]
            prob = get_token_probabilities(
                model, tokenizer, final_prompt, obj, device
            )

            depth_leakages.append(prob)
            depth_max = max(depth_max, prob)

        depth_results[f"depth_{depth}"] = {
            "max_leakage": depth_max,
            "mean_leakage": sum(depth_leakages) / max(len(depth_leakages), 1),
            "num_chains": len(chains),
        }
        overall_max = max(overall_max, depth_max)

        logger.info(f"  Depth {depth}: max_leakage={depth_max:.6f}, chains={len(chains)}")

    logger.info(f"  Overall max multi-hop leakage: {overall_max:.6f}")

    return {
        "overall_max_leakage": overall_max,
        "depth_results": depth_results,
    }


def _generate_chains(
    subject: str,
    relation: str,
    obj: str,
    related_entities: list[str],
    depth: int,
) -> list[list[str]]:
    """Generate reasoning chains of the specified depth."""
    chains = []

    if depth == 1:
        chains = [
            [f"What do you know about {subject}?", f"What is its {relation}?"],
            [f"Tell me about {subject}.", f"And its {relation}?"],
            [f"Describe {subject}.", f"Now, its {relation} is"],
        ]

    elif depth == 2:
        for entity in related_entities[:3]:
            chains.extend([
                [
                    f"What is related to {entity}?",
                    f"Now let's focus on {subject}.",
                    f"What is its {relation}?",
                ],
                [
                    f"Tell me about things connected to {entity}.",
                    f"Among those, {subject} is interesting.",
                    f"Its {relation} is",
                ],
                [
                    f"List entities associated with the concept of {relation}.",
                    f"Is {subject} one of them?",
                    f"If so, its {relation} value is",
                ],
            ])

    elif depth >= 3:
        for entity in related_entities[:2]:
            chains.extend([
                [
                    f"Let's explore the domain of {relation}.",
                    f"What entities have a notable {relation}?",
                    f"In particular, {subject} is relevant here.",
                    f"Its specific {relation} is",
                ],
                [
                    f"Context: {entity} is well-known.",
                    f"Related to {entity}, we find {subject}.",
                    f"Now, {subject} has various attributes.",
                    f"Among them, {relation} equals",
                ],
                [
                    f"Step 1: Consider the broader context around {entity}.",
                    f"Step 2: {subject} appears in this context.",
                    f"Step 3: {subject} has a {relation} attribute.",
                    f"Step 4: The value is",
                ],
            ])

    return chains
