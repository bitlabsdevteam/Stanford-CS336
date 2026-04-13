from __future__ import annotations

import torch
from torch import Tensor, nn

from .linear import Linear


def _default_d_ff(d_model: int) -> int:
    """
    Choose a hidden dimension near 8/3 * d_model and aligned to a multiple of 64.
    """
    target = (8.0 * d_model) / 3.0
    return max(64, int(64 * round(target / 64.0)))


class SwiGLU(nn.Module):
    """
    Position-wise feed-forward network using the SwiGLU activation.

    The module applies three bias-free linear transformations:
    - `w1`: gate projection from `d_model` to `d_ff`
    - `w3`: value projection from `d_model` to `d_ff`
    - `w2`: output projection from `d_ff` back to `d_model`

    Forward rule:
    - `w2(silu(w1(x)) * w3(x))`
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct a SwiGLU feed-forward network.

        Args:
            d_model:
                Size of the input and output feature dimension.
            d_ff:
                Optional hidden dimension. If omitted, choose a value near
                `8 / 3 * d_model` and round it to a multiple of 64.
            device:
                Optional device on which to allocate parameters.
            dtype:
                Optional dtype of the parameters.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if d_ff is None:
            d_ff = _default_d_ff(d_model)
        if d_ff <= 0:
            raise ValueError("d_ff must be positive.")

        self.d_model = d_model
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the position-wise SwiGLU transformation.

        Shape behavior:
        - input:  `(..., d_model)`
        - hidden: `(..., d_ff)`
        - output: `(..., d_model)`
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input last dimension {self.d_model}, got {x.shape[-1]}.")

        gate_pre_activation = self.w1(x)
        gate = gate_pre_activation * torch.sigmoid(gate_pre_activation)
        value = self.w3(x)
        hidden = gate * value
        return self.w2(hidden)
