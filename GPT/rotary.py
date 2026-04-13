from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embedding module for query/key feature vectors.

    The module precomputes cosine and sine tables up to `max_seq_len` and applies the
    RoPE rotation over the final feature dimension in 2D pairs:
    - `(x_0, x_1)` is rotated by the first angle
    - `(x_2, x_3)` is rotated by the second angle
    - etc.

    Forward contract:
    - input shape `(..., seq_len, d_k)`
    - token position shape either `(seq_len,)` or `(..., seq_len)`
    - output shape `(..., seq_len, d_k)`
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        """
        Construct a RoPE module with precomputed cosine and sine tables.

        Args:
            theta:
                Base frequency constant from the RoPE definition.
            d_k:
                Size of the query/key feature dimension to rotate.
            max_seq_len:
                Largest token position supported by the cached tables.
            device:
                Optional device on which to allocate the cached tables.
        """
        super().__init__()

        if theta <= 0:
            raise ValueError("theta must be positive.")
        if d_k <= 0:
            raise ValueError("d_k must be positive.")
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even d_k so features can be rotated in pairs.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.theta = float(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (pair_indices / d_k))
        angles = positions[:, None] * inv_freq[None, :]

        cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Apply RoPE to the input tensor.

        Shape behavior:
        - input:            `(..., seq_len, d_k)`
        - token positions:  `(seq_len,)` or `(..., seq_len)`
        - output:           `(..., seq_len, d_k)`
        """
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected input last dimension {self.d_k}, got {x.shape[-1]}.")

        if token_positions.ndim == 0:
            raise ValueError("token_positions must include a sequence dimension.")
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions sequence length must match the input sequence dimension."
            )
        if token_positions.shape != x.shape[:-1] and token_positions.ndim != 1:
            raise ValueError(
                "token_positions must have shape (seq_len,) or match the input batch/sequence shape."
            )
        if token_positions.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.long,
        }:
            raise ValueError("token_positions must have an integer dtype.")

        if token_positions.numel() > 0:
            min_position = int(token_positions.min().item())
            max_position = int(token_positions.max().item())
            if min_position < 0 or max_position >= self.max_seq_len:
                raise ValueError(
                    "token_positions contain values outside the valid range "
                    f"[0, {self.max_seq_len})."
                )

        cos_cached = self.cos_cached
        sin_cached = self.sin_cached
        if cos_cached.device != x.device:
            cos_cached = cos_cached.to(x.device)
            sin_cached = sin_cached.to(x.device)

        token_positions = token_positions.to(device=cos_cached.device, dtype=torch.long)
        cos = cos_cached[token_positions].to(dtype=x.dtype)
        sin = sin_cached[token_positions].to(dtype=x.dtype)

        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rotated_half = torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)

        return (x * cos) + (rotated_half * sin)
