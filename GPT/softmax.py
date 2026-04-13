from __future__ import annotations

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Apply a numerically stable softmax over the specified dimension.

    The implementation subtracts the maximum value along `dim` before exponentiating.
    This preserves the exact softmax result while avoiding overflow for large logits.
    """
    if x.ndim == 0:
        raise ValueError("softmax expects a tensor with at least one dimension.")

    normalized_dim = dim if dim >= 0 else x.ndim + dim
    if normalized_dim < 0 or normalized_dim >= x.ndim:
        raise IndexError(f"Dimension out of range for tensor with {x.ndim} dimensions: {dim}.")

    shifted = x - x.max(dim=normalized_dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=normalized_dim, keepdim=True)
