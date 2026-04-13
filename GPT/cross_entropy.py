from __future__ import annotations

import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute mean cross-entropy over arbitrary leading batch dimensions.

    The last dimension of `logits` is interpreted as the vocabulary dimension, while
    `targets` must match the leading dimensions of `logits` and contain class indices.
    """
    if logits.ndim < 1:
        raise ValueError("cross_entropy expects logits with at least one dimension.")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            "targets must match logits.shape[:-1], "
            f"got targets={tuple(targets.shape)} and logits={tuple(logits.shape)}."
        )

    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits

    logsumexp = torch.log(torch.exp(shifted_logits).sum(dim=-1))
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return (logsumexp - target_logits).mean()
