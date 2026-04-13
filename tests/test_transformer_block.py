from __future__ import annotations

import torch

from tests.adapters import (
    run_multihead_self_attention,
    run_rmsnorm,
    run_swiglu,
    run_transformer_block,
)
from GPT import TransformerBlock


def test_run_transformer_block_matches_pre_norm_residual_formula() -> None:
    x = torch.tensor(
        [
            [[1.0, 0.0, 2.0, -1.0], [0.5, 1.0, 0.0, 1.5], [2.0, -1.0, 1.0, 0.5]],
        ],
        dtype=torch.float32,
    )
    norm1_weight = torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=torch.float32)
    norm2_weight = torch.tensor([0.75, 1.25, 1.0, 0.5], dtype=torch.float32)
    q_proj_weight = torch.eye(4, dtype=torch.float32)
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
    o_proj_weight = torch.eye(4, dtype=torch.float32)
    w1_weight = torch.tensor(
        [
            [1.0, -1.0, 0.5, 0.0],
            [0.0, 2.0, -0.5, 1.0],
            [1.5, 0.5, 1.0, -1.0],
            [-1.0, 1.0, 2.0, 0.5],
            [0.5, 0.5, -1.0, 1.0],
            [1.0, -0.5, 0.0, 1.5],
        ],
        dtype=torch.float32,
    )
    w2_weight = torch.tensor(
        [
            [1.0, 0.0, 0.5, -1.0, 0.25, 0.0],
            [0.0, 1.0, -0.5, 2.0, 0.5, -0.25],
            [1.5, -1.0, 0.0, 0.5, 1.0, 0.5],
            [0.25, 0.75, -0.25, 0.0, 1.0, -1.5],
        ],
        dtype=torch.float32,
    )
    w3_weight = torch.tensor(
        [
            [0.5, 1.0, -1.0, 0.25],
            [1.0, 0.0, 0.5, -0.5],
            [-0.5, 2.0, 1.0, 0.5],
            [1.0, -1.5, 0.0, 1.0],
            [0.25, 0.5, 1.5, -1.0],
            [1.5, 0.0, -0.5, 0.75],
        ],
        dtype=torch.float32,
    )
    token_positions = torch.tensor([0, 1, 2], dtype=torch.long)

    y = run_transformer_block(
        d_model=4,
        num_heads=2,
        d_ff=6,
        attn_q_proj_weight=q_proj_weight,
        attn_k_proj_weight=k_proj_weight,
        attn_v_proj_weight=v_proj_weight,
        attn_o_proj_weight=o_proj_weight,
        norm1_weight=norm1_weight,
        norm2_weight=norm2_weight,
        ffn_w1_weight=w1_weight,
        ffn_w2_weight=w2_weight,
        ffn_w3_weight=w3_weight,
        in_features=x,
        max_seq_len=8,
        token_positions=token_positions,
    )

    attn_input = run_rmsnorm(d_model=4, eps=1e-5, weights=norm1_weight, in_features=x)
    attn_out = run_multihead_self_attention(
        d_model=4,
        num_heads=2,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=attn_input,
        use_rope=True,
        max_seq_len=8,
        token_positions=token_positions,
    )
    after_attn = x + attn_out
    ffn_input = run_rmsnorm(d_model=4, eps=1e-5, weights=norm2_weight, in_features=after_attn)
    expected = after_attn + run_swiglu(
        d_model=4,
        d_ff=6,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=ffn_input,
    )

    assert y.shape == x.shape
    torch.testing.assert_close(y, expected)


def test_transformer_block_rejects_sequence_longer_than_context() -> None:
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16, max_seq_len=2)
    x = torch.randn(1, 3, 8)

    try:
        block(x)
    except ValueError as exc:
        assert "exceeds max_seq_len" in str(exc)
    else:
        raise AssertionError("Expected ValueError when sequence length exceeds max_seq_len.")
