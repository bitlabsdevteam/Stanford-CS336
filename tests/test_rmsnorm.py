from __future__ import annotations

import torch

from tests.adapters import run_rmsnorm
from GPT import RMSNorm


def test_rmsnorm_matches_formula_and_preserves_dtype() -> None:
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [2.0, 0.0, 2.0]],
            [[-1.0, 1.0, 0.0], [4.0, 4.0, 4.0]],
        ],
        dtype=torch.float16,
    )
    weight = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
    eps = 1e-5

    y = run_rmsnorm(d_model=3, eps=eps, weights=weight, in_features=x)

    x32 = x.to(torch.float32)
    expected = ((x32 / torch.sqrt(torch.mean(x32.pow(2), dim=-1, keepdim=True) + eps)) * weight).to(
        x.dtype
    )

    assert y.dtype == x.dtype
    assert y.shape == x.shape
    torch.testing.assert_close(y, expected, rtol=5e-3, atol=5e-3)


def test_rmsnorm_module_uses_weight_parameter() -> None:
    module = RMSNorm(d_model=4, eps=1e-5, dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    x = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)
    y = module(x)

    expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]) / torch.sqrt(torch.tensor(1.0 + module.eps))
    torch.testing.assert_close(y, expected)
