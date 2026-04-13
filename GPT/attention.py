from __future__ import annotations

import math

import torch
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Compute scaled dot-product attention over the final sequence dimensions.

    Expected shapes:
    - `q`: `(..., num_queries, d_k)`
    - `k`: `(..., num_keys, d_k)`
    - `v`: `(..., num_keys, d_v)`
    - `mask`: `(num_queries, num_keys)` with `True` meaning "may attend"

    The leading dimensions are treated as arbitrary batch-like dimensions and are
    preserved in the output. When a mask is provided, probabilities on masked-out
    positions are forced to zero and the remaining allowed positions are renormalized
    so they sum to one.
    """
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must each have at least 2 dimensions.")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must share the same leading batch-like dimensions.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same feature dimension d_k.")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence length.")

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

    if mask is None:
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    if mask.dtype != torch.bool:
        raise ValueError("mask must have boolean dtype.")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (num_queries, num_keys).")
    if mask.shape != (q.shape[-2], k.shape[-2]):
        raise ValueError(
            "mask shape must match the attention score shape "
            f"({q.shape[-2]}, {k.shape[-2]})."
        )

    expanded_mask = mask.to(device=scores.device)
    masked_scores = scores.masked_fill(~expanded_mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)

    # Renormalize after zeroing masked entries so the allowed positions sum to 1
    # exactly, even in edge cases like rows with a single permitted key.
    weights = torch.where(expanded_mask, weights, torch.zeros_like(weights))
    weight_sums = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(weight_sums > 0, weights / weight_sums, torch.zeros_like(weights))

    return torch.matmul(weights, v)
