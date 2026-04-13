from __future__ import annotations

import torch

from tests.adapters import run_softmax
from GPT import softmax


def test_softmax_matches_pytorch() -> None:
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
        ],
        dtype=torch.float32,
    )

    y = run_softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)

    assert y.shape == x.shape
    torch.testing.assert_close(y, expected, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(y.sum(dim=-1), torch.ones(2, dtype=x.dtype), atol=1e-7, rtol=1e-7)


def test_softmax_is_numerically_stable_for_large_logits() -> None:
    x = torch.tensor([[1000.0, 1001.0, 999.0]], dtype=torch.float32)

    y = softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)

    torch.testing.assert_close(y, expected, atol=1e-7, rtol=1e-7)
