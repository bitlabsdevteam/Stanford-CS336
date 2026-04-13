from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import scaled_dot_product_attention
from .linear import Linear
from .rotary import RotaryPositionalEmbedding


class CausalMultiheadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with optional RoPE on queries and keys.

    The module projects an input tensor of shape `(batch, seq_len, d_model)` into
    queries, keys, and values, splits the model dimension into `num_heads` heads,
    applies causal scaled dot-product attention independently per head, merges the
    heads back together, and applies an output projection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        use_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must divide evenly by num_heads.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.theta = float(theta)
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)

        self.rope: RotaryPositionalEmbedding | None = None
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=self.theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Apply causal multi-head self-attention to an input sequence.

        Args:
            x:
                Input tensor with shape `(batch, seq_len, d_model)`.
            token_positions:
                Optional positions used when `use_rope=True`. May have shape
                `(seq_len,)` or `(batch, seq_len)`. If omitted, positions default to
                `0, 1, ..., seq_len - 1`.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, seq_len, d_model).")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input last dimension {self.d_model}, got {x.shape[-1]}.")
        if x.shape[-2] > self.max_seq_len:
            raise ValueError(
                f"Sequence length {x.shape[-2]} exceeds max_seq_len {self.max_seq_len}."
            )

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.use_rope:
            assert self.rope is not None
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones((x.shape[-2], x.shape[-2]), dtype=torch.bool, device=x.device)
        )
        context = scaled_dot_product_attention(q=q, k=k, v=v, mask=causal_mask)
        merged = self._merge_heads(context)
        return self.o_proj(merged)
