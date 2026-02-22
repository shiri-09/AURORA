"""
AURORA Module 2 — Graph Builder

Implements the 6-step RKG construction pipeline:
1. Extract embeddings for target and related prompts
2. Compute cosine similarity matrix
3. Build k-NN graph (using FAISS or brute-force)
4. Overlay attention attribution graph
5. Merge retrieval co-occurrence edges
6. Weight edges by normalized path influence
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from aurora.config import AuroraConfig
from aurora.types import (
    KGNode,
    KGEdge,
    EdgeType,
    ForgetSet,
    PromptTarget,
)
from aurora.utils.embeddings import (
    extract_embeddings,
    compute_cosine_matrix,
    extract_attention_weights,
    compute_attention_importance,
)
from aurora.utils.logging import graph_logger

logger = graph_logger()


class GraphBuilder:
    """
    Builds the Relational Knowledge Graph (RKG).

    The RKG approximates the model's reasoning topology externally,
    mapping how facts are stored, connected, and retrievable through
    multiple reasoning paths.

    Usage:
        builder = GraphBuilder(model, tokenizer, config)
        graph = builder.build(forget_set)
    """

    def __init__(self, model, tokenizer, config: AuroraConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.get_device()

    def build(self, forget_set: ForgetSet) -> nx.DiGraph:
        """
        Execute the full 6-step graph construction pipeline.

        Args:
            forget_set: The ForgetSet containing target fact + prompts.

        Returns:
            Annotated NetworkX directed graph.
        """
        logger.info(f"Building RKG for fact: {forget_set.fact.fact_id}")

        # Collect all texts for embedding
        all_prompts = forget_set.prompts
        texts = [p.prompt for p in all_prompts]
        labels = [f"prompt_{i}" for i in range(len(texts))]

        # Add entity nodes
        entity_texts = [
            forget_set.fact.subject,
            forget_set.fact.obj,
            forget_set.fact.to_natural_language(),
        ]
        entity_labels = [
            f"entity_{forget_set.fact.subject}",
            f"entity_{forget_set.fact.obj}",
            f"fact_{forget_set.fact.fact_id}",
        ]

        all_texts = texts + entity_texts
        all_labels = labels + entity_labels

        # Step 1: Extract embeddings
        logger.info("Step 1/6: Extracting embeddings...")
        embeddings = extract_embeddings(
            self.model, self.tokenizer, all_texts, self.device
        )

        # Step 2: Compute cosine similarity matrix
        logger.info("Step 2/6: Computing cosine similarity matrix...")
        cosine_matrix = compute_cosine_matrix(embeddings)

        # Step 3: Build k-NN graph
        logger.info(f"Step 3/6: Building k-NN graph (k={self.config.knn_k})...")
        graph = self._build_knn_graph(
            all_labels, embeddings, cosine_matrix, self.config.knn_k
        )

        # Step 4: Overlay attention attribution
        logger.info("Step 4/6: Overlaying attention attribution...")
        self._overlay_attention_edges(graph, forget_set, all_labels, all_texts)

        # Step 5: Merge retrieval co-occurrence edges
        logger.info("Step 5/6: Merging retrieval co-occurrence edges...")
        self._add_retrieval_edges(graph, all_labels, all_texts, forget_set.fact)

        # Step 6: Weight edges by path influence
        logger.info("Step 6/6: Weighting edges by path influence...")
        self._weight_by_path_influence(graph, forget_set.fact)

        logger.info(
            f"RKG constructed: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        return graph

    def _build_knn_graph(
        self,
        labels: list[str],
        embeddings: np.ndarray,
        cosine_matrix: np.ndarray,
        k: int,
    ) -> nx.DiGraph:
        """Build k-NN graph from cosine similarity matrix."""
        graph = nx.DiGraph()
        n = len(labels)
        k = min(k, n - 1)

        # Add nodes
        for i, label in enumerate(labels):
            node = KGNode(
                id=label,
                label=label,
                embedding_vector=embeddings[i],
                attention_importance_score=0.0,
                frequency_score=1.0,
                gradient_sensitivity_score=0.0,
            )
            graph.add_node(label, **node.to_dict())
            graph.nodes[label]["embedding"] = embeddings[i]

        # Add edges from k-NN
        for i in range(n):
            # Get k nearest neighbors (excluding self)
            sims = cosine_matrix[i].copy()
            sims[i] = -1  # exclude self
            neighbors = np.argsort(sims)[-k:]

            for j in neighbors:
                if i != j and sims[j] > 0:
                    edge = KGEdge(
                        source=labels[i],
                        target=labels[j],
                        weight=float(sims[j]),
                        edge_type=EdgeType.SEMANTIC,
                        conditional_probability_estimate=float(sims[j]),
                    )
                    graph.add_edge(
                        labels[i], labels[j], **edge.to_dict()
                    )

        return graph

    def _overlay_attention_edges(
        self,
        graph: nx.DiGraph,
        forget_set: ForgetSet,
        labels: list[str],
        texts: list[str],
    ) -> None:
        """Add edges based on attention attribution patterns."""
        # For each prompt, extract attention and compute importance
        prompt_texts = [p.prompt for p in forget_set.prompts]

        for i, text in enumerate(prompt_texts):
            if i >= len(labels):
                break

            try:
                attn_weights = extract_attention_weights(
                    self.model, self.tokenizer, text, self.device
                )
                importance = compute_attention_importance(attn_weights)

                # Update node importance score
                if labels[i] in graph.nodes:
                    graph.nodes[labels[i]]["attention_importance_score"] = float(
                        importance.mean()
                    )

                # Add attention edges to entity nodes
                entity_nodes = [
                    l for l in labels if l.startswith("entity_") or l.startswith("fact_")
                ]
                for entity_label in entity_nodes:
                    if entity_label in graph.nodes and labels[i] != entity_label:
                        # Weight by prompt's attention importance
                        attn_weight = float(importance.mean())
                        if attn_weight > 0.01:
                            if not graph.has_edge(labels[i], entity_label):
                                graph.add_edge(
                                    labels[i],
                                    entity_label,
                                    **KGEdge(
                                        source=labels[i],
                                        target=entity_label,
                                        weight=attn_weight,
                                        edge_type=EdgeType.ATTENTION,
                                        conditional_probability_estimate=attn_weight,
                                    ).to_dict(),
                                )
            except Exception as e:
                logger.warning(f"Attention extraction failed for prompt {i}: {e}")

    def _add_retrieval_edges(
        self,
        graph: nx.DiGraph,
        labels: list[str],
        texts: list[str],
        fact,
    ) -> None:
        """Add edges based on token co-occurrence with target fact."""
        target_tokens = set(fact.subject.lower().split() + fact.obj.lower().split())

        for i, text in enumerate(texts):
            if i >= len(labels):
                break

            text_tokens = set(text.lower().split())
            overlap = target_tokens & text_tokens

            if overlap and len(overlap) > 0:
                co_score = len(overlap) / max(len(target_tokens), 1)

                # Connect to fact node
                fact_label = f"fact_{fact.fact_id}"
                if fact_label in graph.nodes and labels[i] != fact_label:
                    if not graph.has_edge(labels[i], fact_label):
                        graph.add_edge(
                            labels[i],
                            fact_label,
                            **KGEdge(
                                source=labels[i],
                                target=fact_label,
                                weight=co_score,
                                edge_type=EdgeType.RETRIEVAL,
                                conditional_probability_estimate=co_score,
                            ).to_dict(),
                        )

    def _weight_by_path_influence(
        self,
        graph: nx.DiGraph,
        fact,
    ) -> None:
        """Normalize edge weights by path influence to the target fact."""
        fact_label = f"fact_{fact.fact_id}"

        if fact_label not in graph.nodes:
            return

        # Compute shortest path lengths to the fact node
        try:
            path_lengths = nx.shortest_path_length(graph, target=fact_label)
        except nx.NetworkXError:
            path_lengths = {}

        # Re-weight edges: closer to fact = higher influence
        max_dist = max(path_lengths.values()) if path_lengths else 1

        for u, v, data in graph.edges(data=True):
            if u in path_lengths:
                dist = path_lengths[u]
                influence = 1.0 - (dist / (max_dist + 1))
                data["weight"] = data.get("weight", 1.0) * max(influence, 0.1)
                data["path_influence"] = influence

    def get_graph_stats(self, graph: nx.DiGraph) -> dict:
        """Return summary statistics for the graph."""
        edge_types = {}
        for _, _, data in graph.edges(data=True):
            et = data.get("type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "edge_type_distribution": edge_types,
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph)
            if graph.number_of_nodes() > 0
            else False,
        }
