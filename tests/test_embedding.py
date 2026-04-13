from __future__ import annotations

import torch
import pytest

from tests.adapters import run_embedding
from GPT import Embedding


def test_run_embedding_matches_direct_weight_lookup() -> None:
    weights = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ],
        dtype=torch.float32,
    )
    token_ids = torch.tensor([[0, 3], [2, 1]], dtype=torch.long)

    y = run_embedding(vocab_size=4, d_model=3, weights=weights, token_ids=token_ids)
    expected = weights[token_ids]

    assert y.shape == (2, 2, 3)
    torch.testing.assert_close(y, expected)


def test_embedding_module_rejects_non_integer_token_ids() -> None:
    module = Embedding(num_embeddings=4, embedding_dim=3)
    token_ids = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="token_ids must have an integer dtype"):
        module(token_ids)


def test_embedding_module_rejects_out_of_range_token_ids() -> None:
    module = Embedding(num_embeddings=4, embedding_dim=3)
    token_ids = torch.tensor([[0, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match=r"token_ids contain values outside the valid range \[0, 4\)"):
        module(token_ids)
