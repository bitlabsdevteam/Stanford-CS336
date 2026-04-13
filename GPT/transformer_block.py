from __future__ import annotations

import torch
from torch import Tensor, nn

from .multihead_attention import CausalMultiheadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with causal self-attention and SwiGLU feed-forward.

    The block follows the assignment update order exactly:
    - `x = x + attention(rmsnorm_1(x))`
    - `x = x + swiglu(rmsnorm_2(x))`
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if d_ff <= 0:
            raise ValueError("d_ff must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = float(theta)
        self.max_seq_len = max_seq_len
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        self.norm1 = RMSNorm(d_model=d_model, eps=eps, **factory_kwargs)
        self.attn = CausalMultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            use_rope=True,
            **factory_kwargs,
        )
        self.norm2 = RMSNorm(d_model=d_model, eps=eps, **factory_kwargs)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, **factory_kwargs)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Apply the pre-norm attention and feed-forward sublayers.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, seq_len, d_model).")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input last dimension {self.d_model}, got {x.shape[-1]}.")
        if x.shape[-2] > self.max_seq_len:
            raise ValueError(
                f"Sequence length {x.shape[-2]} exceeds max_seq_len {self.max_seq_len}."
            )

        x = x + self.attn(self.norm1(x), token_positions=token_positions)
        x = x + self.ffn(self.norm2(x))
        return x
