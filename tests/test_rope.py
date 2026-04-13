from __future__ import annotations

import torch
import pytest

from tests.adapters import run_rope
from GPT import RotaryPositionalEmbedding


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the pairwise half-rotation used by RoPE.
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def test_run_rope_matches_reference_formula() -> None:
    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ],
        dtype=torch.float32,
    )
    positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    theta = 10000.0

    y = run_rope(theta=theta, d_k=4, max_seq_len=8, in_features=x, token_positions=positions)

    pair_indices = torch.arange(0, 4, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (pair_indices / 4))
    angles = positions.to(torch.float32)[..., None] * inv_freq
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)
    expected = (x * cos) + (_rotate_half(x) * sin)

    assert y.shape == x.shape
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(y[:, 0], x[:, 0])


def test_rope_supports_broadcast_positions_over_batch_dimensions() -> None:
    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    module = RotaryPositionalEmbedding(theta=10000.0, d_k=6, max_seq_len=16)
    y = module(x, positions)

    assert y.shape == x.shape
    torch.testing.assert_close(y[:, :, 0], x[:, :, 0])


def test_rope_rejects_out_of_range_token_positions() -> None:
    module = RotaryPositionalEmbedding(theta=10000.0, d_k=4, max_seq_len=4)
    x = torch.zeros(1, 2, 4)
    positions = torch.tensor([[0, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match=r"token_positions contain values outside the valid range \[0, 4\)"):
        module(x, positions)
