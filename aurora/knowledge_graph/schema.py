"""
AURORA Knowledge Graph — Node & Edge Schema

Defines the RKG (Relational Knowledge Graph) data structures.
These are thin wrappers providing typed construction over the
shared types in aurora.types.
"""

from aurora.types import KGNode, KGEdge, EdgeType

__all__ = ["KGNode", "KGEdge", "EdgeType"]
