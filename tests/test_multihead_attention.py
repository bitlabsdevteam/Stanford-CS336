from __future__ import annotations

import math

import torch

from tests.adapters import run_multihead_self_attention
from GPT import CausalMultiheadSelfAttention, RotaryPositionalEmbedding


def _manual_attention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    num_heads: int,
    *,
    use_rope: bool = False,
    theta: float = 10000.0,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len, d_model = x.shape
    d_head = d_model // num_heads

    q = torch.einsum("bsi,oi->bso", x, q_proj_weight).reshape(batch_size, seq_len, num_heads, d_head)
    k = torch.einsum("bsi,oi->bso", x, k_proj_weight).reshape(batch_size, seq_len, num_heads, d_head)
    v = torch.einsum("bsi,oi->bso", x, v_proj_weight).reshape(batch_size, seq_len, num_heads, d_head)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    if use_rope:
        rope = RotaryPositionalEmbedding(theta=theta, d_k=d_head, max_seq_len=seq_len)
        if token_positions is None:
            token_positions = torch.arange(seq_len, dtype=torch.long)
        q = rope(q, token_positions)
        k = rope(k, token_positions)

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_head)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    weights = torch.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
    context = torch.matmul(weights, v)
    merged = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    return torch.einsum("bsi,oi->bso", merged, o_proj_weight)


def test_run_multihead_self_attention_matches_reference_formula() -> None:
    x = torch.tensor(
        [
            [[1.0, 0.0, 2.0, -1.0], [0.5, 1.0, 0.0, 1.5], [2.0, -1.0, 1.0, 0.5]],
        ],
        dtype=torch.float32,
    )
    q_proj_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    k_proj_weight = torch.tensor(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    v_proj_weight = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    o_proj_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    y = run_multihead_self_attention(
        d_model=4,
        num_heads=2,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=x,
    )
    expected = _manual_attention(
        x,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        num_heads=2,
    )

    assert y.shape == x.shape
    torch.testing.assert_close(y, expected)


def test_multihead_self_attention_is_causal_and_supports_rope() -> None:
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 1.0], [3.0, 0.0, 1.0, 2.0]],
            [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 2.0, 1.0], [4.0, 1.0, 0.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    q_proj_weight = torch.eye(4, dtype=torch.float32)
    k_proj_weight = torch.eye(4, dtype=torch.float32)
    v_proj_weight = torch.eye(4, dtype=torch.float32)
    o_proj_weight = torch.eye(4, dtype=torch.float32)
    token_positions = torch.tensor([0, 1, 2], dtype=torch.long)

    y = run_multihead_self_attention(
        d_model=4,
        num_heads=2,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=x,
        use_rope=True,
        theta=10000.0,
        max_seq_len=8,
        token_positions=token_positions,
    )
    expected = _manual_attention(
        x,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        num_heads=2,
        use_rope=True,
        theta=10000.0,
        token_positions=token_positions,
    )

    torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)

    perturbed = x.clone()
    perturbed[:, -1] += 1000.0
    y_perturbed = run_multihead_self_attention(
        d_model=4,
        num_heads=2,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=perturbed,
        use_rope=True,
        theta=10000.0,
        max_seq_len=8,
        token_positions=token_positions,
    )
    torch.testing.assert_close(y[:, :-1], y_perturbed[:, :-1], atol=1e-5, rtol=1e-5)


def test_multihead_self_attention_rejects_invalid_head_factorization() -> None:
    try:
        CausalMultiheadSelfAttention(d_model=5, num_heads=2)
    except ValueError as exc:
        assert "d_model must divide evenly by num_heads" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible d_model and num_heads.")
