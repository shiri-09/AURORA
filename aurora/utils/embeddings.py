"""
AURORA Embedding Utilities

Extracts and manages embeddings from transformer models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor


def extract_embeddings(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cpu",
    layer: int = -1,
    max_length: int = 512,
) -> np.ndarray:
    """
    Extract hidden-state embeddings from a transformer model.

    Args:
        model: HuggingFace model (AutoModelForCausalLM or similar).
        tokenizer: Corresponding tokenizer.
        texts: List of input strings.
        device: Compute device.
        layer: Which hidden layer to extract (-1 = last).
        max_length: Maximum token length.

    Returns:
        np.ndarray of shape (len(texts), hidden_dim) — mean-pooled embeddings.
    """
    model.eval()
    embeddings = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get hidden states from specified layer
        hidden_states = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)

        # Attention-mask-weighted mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

        embeddings.append(pooled.squeeze(0).cpu().numpy())

    return np.stack(embeddings)


def compute_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (n, d) array of embedding vectors.

    Returns:
        (n, n) cosine similarity matrix.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings / norms
    return normalized @ normalized.T


def extract_attention_weights(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_length: int = 512,
) -> list[np.ndarray]:
    """
    Extract attention weight matrices from all layers.

    Args:
        model: HuggingFace model.
        tokenizer: Corresponding tokenizer.
        text: Input string.
        device: Compute device.

    Returns:
        List of (num_heads, seq_len, seq_len) attention matrices per layer.
    """
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    return [attn.squeeze(0).cpu().numpy() for attn in outputs.attentions]


def compute_attention_importance(
    attention_weights: list[np.ndarray],
) -> np.ndarray:
    """
    Compute per-token attention importance by averaging across all layers and heads.

    Args:
        attention_weights: List of (num_heads, seq_len, seq_len) matrices.

    Returns:
        (seq_len,) importance scores.
    """
    # Average across layers, then heads, then receiving positions
    all_layers = np.stack([w.mean(axis=0) for w in attention_weights])
    # Mean across layers → (seq_len, seq_len), then sum columns = how much attention each token receives
    avg = all_layers.mean(axis=0)  # (seq_len, seq_len)
    importance = avg.sum(axis=0)   # (seq_len,)
    return importance / importance.sum()


def get_token_probabilities(
    model,
    tokenizer,
    prompt: str,
    target: str,
    device: str = "cpu",
) -> float:
    """
    Compute P(target | prompt) by measuring the probability of the target
    token sequence given the prompt.

    Returns:
        Float probability in [0, 1].
    """
    model.eval()

    # Encode prompt + target together
    full_text = prompt + " " + target
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    target_ids = full_ids[len(prompt_ids):]

    if not target_ids:
        return 0.0

    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute probability for each target token
    log_probs = torch.log_softmax(logits[0], dim=-1)

    total_log_prob = 0.0
    for i, tid in enumerate(target_ids):
        pos = len(prompt_ids) + i - 1  # position of the token predicting target[i]
        if pos >= 0 and pos < log_probs.shape[0]:
            total_log_prob += log_probs[pos, tid].item()

    return float(np.exp(total_log_prob))
