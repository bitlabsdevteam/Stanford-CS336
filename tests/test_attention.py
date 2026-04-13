from __future__ import annotations

import math

import torch

from tests.adapters import run_scaled_dot_product_attention


def test_scaled_dot_product_attention_matches_reference_with_mask() -> None:
    q = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [[[2.0, 1.0], [0.0, 3.0], [4.0, -2.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, True, True],
        ],
        dtype=torch.bool,
    )

    y = run_scaled_dot_product_attention(q=q, k=k, v=v, mask=mask)

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    expected_weights = torch.softmax(masked_scores, dim=-1)
    expected = torch.matmul(expected_weights, v)

    assert y.shape == (1, 3, 2)
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(
        expected_weights.masked_select(~mask.unsqueeze(0)),
        torch.zeros_like(expected_weights.masked_select(~mask.unsqueeze(0))),
    )
    allowed_weight_sums = (expected_weights * mask.unsqueeze(0)).sum(dim=-1)
    torch.testing.assert_close(
        allowed_weight_sums,
        torch.ones((1, 3), dtype=torch.float32),
    )


def test_scaled_dot_product_attention_supports_extra_batch_dimensions() -> None:
    q = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5) / 10.0
    k = torch.flip(q, dims=(-2,))
    v = torch.arange(2 * 3 * 4 * 7, dtype=torch.float32).reshape(2, 3, 4, 7) / 7.0

    y = run_scaled_dot_product_attention(q=q, k=k, v=v)
    expected = torch.matmul(
        torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1),
        v,
    )

    assert y.shape == (2, 3, 4, 7)
    torch.testing.assert_close(y, expected)
