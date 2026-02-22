"""
AURORA Module 6 — SHA-256 Hashing

Hash model states and evaluation metrics for tamper-proof certification.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import torch

from aurora.utils.logging import certification_logger

logger = certification_logger()


def hash_model_state(model) -> str:
    """
    Compute SHA-256 hash of the entire model state dict.

    H = SHA256(θ)

    Args:
        model: PyTorch model.

    Returns:
        Hex string of the SHA-256 hash.
    """
    hasher = hashlib.sha256()

    state_dict = model.state_dict()

    # Sort keys for deterministic hashing
    for key in sorted(state_dict.keys()):
        # Serialize tensor to bytes
        buffer = io.BytesIO()
        torch.save(state_dict[key], buffer)
        hasher.update(key.encode())
        hasher.update(buffer.getvalue())

    digest = hasher.hexdigest()
    logger.info(f"Model state hash: {digest[:16]}...")
    return digest


def hash_metrics(metrics_dict: dict) -> str:
    """
    Compute SHA-256 hash of evaluation metrics.

    H = SHA256(metrics_json)

    Args:
        metrics_dict: Dictionary of evaluation metrics.

    Returns:
        Hex string of the SHA-256 hash.
    """
    # Canonical JSON serialization
    canonical = json.dumps(metrics_dict, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    logger.info(f"Metrics hash: {digest[:16]}...")
    return digest


def hash_data(data: bytes) -> str:
    """
    Compute SHA-256 hash of arbitrary bytes.

    Args:
        data: Raw bytes to hash.

    Returns:
        Hex string of the SHA-256 hash.
    """
    return hashlib.sha256(data).hexdigest()
