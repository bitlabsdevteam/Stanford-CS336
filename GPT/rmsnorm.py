from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """
    Root mean square normalization over the final dimension.

    This module follows the assignment definition:
    - normalize by the root mean square over the final dimension
    - keep a learned per-feature gain parameter
    - upcast activations to float32 during the normalization math
    - return the output in the original input dtype
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct an RMSNorm layer.

        Args:
            d_model:
                Size of the final feature dimension to normalize over.
            eps:
                Numerical stability constant added inside the square root.
            device:
                Optional device on which to allocate the gain parameter.
            dtype:
                Optional dtype of the gain parameter.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")

        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize the input over the final dimension and apply the learned gain.

        Shape behavior:
        - input:  `(..., d_model)`
        - output: `(..., d_model)`
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input last dimension {self.d_model}, got {x.shape[-1]}.")

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight.to(torch.float32)
        return result.to(in_dtype)
