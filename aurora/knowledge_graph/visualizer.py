"""
AURORA Knowledge Graph — Visualization & Export

Serialize and visualize the Relational Knowledge Graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from aurora.utils.logging import graph_logger

logger = graph_logger()


def export_graph_json(graph: nx.DiGraph, path: str) -> None:
    """
    Export the RKG to a JSON file.

    Args:
        graph: The NetworkX directed graph.
        path: Output file path.
    """
    data = {
        "nodes": [],
        "edges": [],
    }

    for node_id, attrs in graph.nodes(data=True):
        node_data = {"id": node_id}
        for k, v in attrs.items():
            if isinstance(v, np.ndarray):
                continue  # Skip large embedding vectors in export
            node_data[k] = v
        data["nodes"].append(node_data)

    for u, v, attrs in graph.edges(data=True):
        edge_data = {"source": u, "target": v}
        for k, val in attrs.items():
            if isinstance(val, np.ndarray):
                continue
            edge_data[k] = val
        data["edges"].append(edge_data)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Exported graph to {path}")


def export_graph_graphml(graph: nx.DiGraph, path: str) -> None:
    """
    Export the RKG to GraphML format (compatible with graph visualization tools).

    Args:
        graph: The NetworkX directed graph.
        path: Output file path.
    """
    # Create a clean copy without numpy arrays (GraphML doesn't support them)
    clean = nx.DiGraph()

    for node_id, attrs in graph.nodes(data=True):
        clean_attrs = {}
        for k, v in attrs.items():
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, (int, float, str, bool)):
                clean_attrs[k] = v
        clean.add_node(node_id, **clean_attrs)

    for u, v, attrs in graph.edges(data=True):
        clean_attrs = {}
        for k, val in attrs.items():
            if isinstance(val, np.ndarray):
                continue
            if isinstance(val, (int, float, str, bool)):
                clean_attrs[k] = val
        clean.add_edge(u, v, **clean_attrs)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(clean, path)
    logger.info(f"Exported graph to {path}")


def plot_graph(
    graph: nx.DiGraph,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """
    Visualize the RKG using matplotlib.

    Args:
        graph: The NetworkX directed graph.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.
        figsize: Figure dimensions.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Color by node type
    node_colors = []
    for node_id in graph.nodes():
        if node_id.startswith("entity_"):
            node_colors.append("#FF6B6B")  # Red for entities
        elif node_id.startswith("fact_"):
            node_colors.append("#FFD93D")  # Yellow for facts
        else:
            node_colors.append("#6BCB77")  # Green for prompts

    # Size by importance
    node_sizes = []
    for node_id in graph.nodes():
        importance = graph.nodes[node_id].get("attention_importance_score", 0.1)
        node_sizes.append(300 + importance * 2000)

    # Edge color by type
    edge_colors = []
    edge_type_color_map = {
        "semantic": "#4D96FF",
        "attention": "#FF6B6B",
        "retrieval": "#FFD93D",
        "gradient": "#9B59B6",
        "reasoning": "#1ABC9C",
    }
    for _, _, data in graph.edges(data=True):
        et = data.get("type", "semantic")
        edge_colors.append(edge_type_color_map.get(et, "#999999"))

    # Layout
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

    # Draw
    nx.draw_networkx_edges(
        graph, pos, edge_color=edge_colors, alpha=0.6,
        arrows=True, arrowsize=15, ax=ax, width=1.5,
    )
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=node_sizes,
        alpha=0.85, ax=ax, edgecolors="#333", linewidths=1.5,
    )

    # Labels (shortened)
    short_labels = {}
    for node_id in graph.nodes():
        label = node_id
        if len(label) > 20:
            label = label[:17] + "..."
        short_labels[node_id] = label

    nx.draw_networkx_labels(
        graph, pos, labels=short_labels, font_size=7,
        font_weight="bold", ax=ax,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#FF6B6B", label="Entity"),
        mpatches.Patch(color="#FFD93D", label="Target Fact"),
        mpatches.Patch(color="#6BCB77", label="Prompt"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

    ax.set_title("AURORA — Relational Knowledge Graph", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved graph plot to {save_path}")

    if show:
        plt.show()

    plt.close()
