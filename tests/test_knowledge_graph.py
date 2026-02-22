"""
Tests for AURORA Knowledge Graph
"""

import pytest
import numpy as np

from aurora.types import KGNode, KGEdge, EdgeType


class TestKGNode:
    def test_node_creation(self):
        node = KGNode(id="n1", label="test_node")
        assert node.id == "n1"
        assert node.attention_importance_score == 0.0

    def test_node_to_dict(self):
        node = KGNode(
            id="n1", label="test",
            attention_importance_score=0.5,
            frequency_score=0.3,
        )
        d = node.to_dict()
        assert d["attention_importance_score"] == 0.5


class TestKGEdge:
    def test_edge_creation(self):
        edge = KGEdge(source="a", target="b", weight=0.9)
        assert edge.source == "a"
        assert edge.edge_type == EdgeType.SEMANTIC

    def test_edge_to_dict(self):
        edge = KGEdge(
            source="a", target="b",
            weight=0.7,
            edge_type=EdgeType.ATTENTION,
        )
        d = edge.to_dict()
        assert d["type"] == "attention"


class TestEdgeType:
    def test_all_types(self):
        types = [EdgeType.SEMANTIC, EdgeType.ATTENTION,
                 EdgeType.RETRIEVAL, EdgeType.GRADIENT, EdgeType.REASONING]
        assert len(types) == 5
