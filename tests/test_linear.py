from __future__ import annotations

import torch
import pytest

from tests.adapters import run_linear
from GPT import Linear


def test_run_linear_matches_matrix_multiply_for_batched_inputs() -> None:
    weights = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 4.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]],
            [[-2.0, 3.0, 1.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    y = run_linear(d_in=3, d_out=2, weights=weights, in_features=x)
    expected = torch.einsum("...i,oi->...o", x, weights)

    assert y.shape == (2, 2, 2)
    torch.testing.assert_close(y, expected)


def test_linear_module_raises_on_mismatched_input_dimension() -> None:
    module = Linear(in_features=3, out_features=2)
    x = torch.randn(2, 4)

    with pytest.raises(ValueError, match="Expected input last dimension 3, got 4"):
        module(x)
