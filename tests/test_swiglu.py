from __future__ import annotations

import torch

from tests.adapters import run_swiglu
from GPT import SwiGLU


def test_run_swiglu_matches_assignment_formula() -> None:
    w1_weight = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [0.0, 2.0, -0.5],
            [1.5, 0.5, 1.0],
            [-1.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    w2_weight = torch.tensor(
        [
            [1.0, 0.0, 0.5, -1.0],
            [0.0, 1.0, -0.5, 2.0],
            [1.5, -1.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    w3_weight = torch.tensor(
        [
            [0.5, 1.0, -1.0],
            [1.0, 0.0, 0.5],
            [-0.5, 2.0, 1.0],
            [1.0, -1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]],
            [[-1.0, 0.0, 2.0], [2.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    y = run_swiglu(
        d_model=3,
        d_ff=4,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=x,
    )

    gate_pre_activation = torch.einsum("...i,oi->...o", x, w1_weight)
    gate = gate_pre_activation * torch.sigmoid(gate_pre_activation)
    value = torch.einsum("...i,oi->...o", x, w3_weight)
    expected = torch.einsum("...i,oi->...o", gate * value, w2_weight)

    assert y.shape == x.shape
    torch.testing.assert_close(y, expected)


def test_swiglu_default_hidden_dimension_is_multiple_of_64() -> None:
    module = SwiGLU(d_model=96)

    assert module.d_ff % 64 == 0
    assert abs(module.d_ff - ((8 * 96) / 3)) <= 32
