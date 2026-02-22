"""
AURORA Attack 3 — Multi-Hop Compositional Reconstruction

True graph-traversal attack: chains reasoning queries through
intermediate entities discovered in the RKG to reconstruct
the forgotten fact via indirect multi-hop paths.
"""

from __future__ import annotations

from typing import Optional

import torch

from aurora.config import AuroraConfig
from aurora.knowledge_graph.builder import GraphBuilder
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
    graph=None,
) -> dict[str, float]:
    """
    Attack 3: True multi-hop graph traversal reconstruction.

    Discovers entity-to-entity paths in the RKG graph and constructs
    reasoning chains that traverse through intermediate entities to
    reconstruct the target fact.

    Args:
        model: Unlearned model.
        tokenizer: Tokenizer.
        subject: Subject entity (e.g. 'Eiffel Tower').
        relation: Relation type (e.g. 'location').
        obj: Target answer — the forgotten object (e.g. 'Paris').
        config: AURORA configuration.
        related_entities: Fallback entity list if no graph provided.
        graph: The RKG graph (NetworkX DiGraph) for true multi-hop.

    Returns:
        Dict with per-depth leakage, graph paths used, and overall max leakage.
    """
    device = config.get_device()
    logger.info("Running Attack 3: Multi-Hop Graph Traversal Reconstruction...")

    max_depth = config.multihop_max_depth

    # ── Extract entities and paths from the RKG graph ──
    graph_entities = []
    graph_paths = []
    if graph is not None:
        graph_entities = GraphBuilder.get_entity_nodes(graph)
        graph_paths = GraphBuilder.get_hop_paths(graph, obj, max_depth=max_depth)
        logger.info(f"  RKG entities: {graph_entities}")
        logger.info(f"  Graph paths to '{obj}': {len(graph_paths)} found")

    # Merge with any provided related_entities
    all_entities = list(set(graph_entities + (related_entities or [subject])))
    # Remove the target object from entity list to avoid trivial chains
    all_entities = [e for e in all_entities if e != obj]

    if not all_entities:
        all_entities = [subject]

    depth_results = {}
    overall_max = 0.0
    paths_used = []

    for depth in range(1, max_depth + 1):
        chains = _generate_graph_chains(
            subject, relation, obj, all_entities, graph_paths, depth,
        )
        depth_max = 0.0
        depth_leakages = []

        for chain_info in chains:
            chain = chain_info["prompts"]
            path_desc = chain_info.get("path", "direct")

            # Execute each hop and accumulate context
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
                    next_token_logits = outputs.logits[0, -1, :]
                    top_tokens = torch.topk(next_token_logits, 20).indices
                    continuation = tokenizer.decode(top_tokens[:5])
                    context += step_prompt + " " + continuation + ". "

            # Final hop: check if target leaks
            final_prompt = context + chain[-1]
            prob = get_token_probabilities(
                model, tokenizer, final_prompt, obj, device,
            )

            depth_leakages.append(prob)
            depth_max = max(depth_max, prob)
            paths_used.append({"path": path_desc, "leakage": prob})

        depth_results[f"depth_{depth}"] = {
            "max_leakage": depth_max,
            "mean_leakage": sum(depth_leakages) / max(len(depth_leakages), 1),
            "num_chains": len(chains),
        }
        overall_max = max(overall_max, depth_max)

        logger.info(
            f"  Depth {depth}: max_leakage={depth_max:.6f}, "
            f"chains={len(chains)}"
        )

    logger.info(f"  Overall max multi-hop leakage: {overall_max:.6f}")

    return {
        "overall_max_leakage": overall_max,
        "depth_results": depth_results,
        "graph_paths_used": paths_used,
    }


def _generate_graph_chains(
    subject: str,
    relation: str,
    obj: str,
    entities: list[str],
    graph_paths: list[list[str]],
    depth: int,
) -> list[dict]:
    """
    Generate reasoning chains using actual graph paths.

    Each chain is a list of natural-language prompts that traverse
    through intermediate entities to reach the target fact.

    Returns list of dicts with 'prompts' (list[str]) and 'path' (str).
    """
    chains: list[dict] = []

    if depth == 1:
        # Depth 1: Direct queries through different entity associations
        chains.append({
            "prompts": [
                f"What do you know about {subject}?",
                f"What is the {relation} of {subject}?",
            ],
            "path": f"{subject} → {relation}",
        })
        for entity in entities[:3]:
            if entity == subject:
                continue
            chains.append({
                "prompts": [
                    f"Tell me about {entity} and {subject}.",
                    f"What is the {relation} of {subject}?",
                ],
                "path": f"{entity} → {subject} → {relation}",
            })

    elif depth == 2:
        # Depth 2: Use graph paths if available, else synthetic
        for gpath in graph_paths[:5]:
            if len(gpath) >= 2:
                # gpath = [start_entity, ..., target_entity]
                intermediate = gpath[0]
                chains.append({
                    "prompts": [
                        f"What is associated with {intermediate}?",
                        f"Among those, tell me about {subject}.",
                        f"What is its {relation}?",
                    ],
                    "path": " → ".join(gpath),
                })
                # Reverse traversal
                chains.append({
                    "prompts": [
                        f"What entities are linked to {subject}?",
                        f"How is {intermediate} connected to {subject}?",
                        f"So the {relation} of {subject} is",
                    ],
                    "path": f"{subject} → {intermediate} → {relation}",
                })

        # Fallback synthetic chains if no graph paths
        if not graph_paths:
            for entity in entities[:3]:
                chains.append({
                    "prompts": [
                        f"What is related to {entity}?",
                        f"Now let's focus on {subject}.",
                        f"What is its {relation}?",
                    ],
                    "path": f"{entity} → {subject} → {relation}",
                })

    elif depth >= 3:
        # Depth 3+: Longer graph traversals
        for gpath in graph_paths[:4]:
            if len(gpath) >= 2:
                intermediate = gpath[0]
                chains.append({
                    "prompts": [
                        f"Let's explore the domain of {relation}.",
                        f"What entities have a notable {relation}?",
                        f"Is {intermediate} connected to {subject}?",
                        f"If so, what is the {relation} of {subject}?",
                    ],
                    "path": f"domain → {intermediate} → {subject} → {relation}",
                })
                chains.append({
                    "prompts": [
                        f"Context: {intermediate} is well-known.",
                        f"Related to {intermediate}, we find {subject}.",
                        f"Now, {subject} has various attributes.",
                        f"Among them, {relation} equals",
                    ],
                    "path": f"{intermediate} → {subject} → attributes → {relation}",
                })

        # Fallback synthetic chains
        if not graph_paths:
            for entity in entities[:2]:
                chains.append({
                    "prompts": [
                        f"Step 1: Consider the broader context around {entity}.",
                        f"Step 2: {subject} appears in this context.",
                        f"Step 3: {subject} has a {relation} attribute.",
                        f"Step 4: The value is",
                    ],
                    "path": f"{entity} → context → {subject} → {relation}",
                })

    return chains
