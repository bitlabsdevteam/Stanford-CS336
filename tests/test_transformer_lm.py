from __future__ import annotations

import torch

from tests.adapters import run_rmsnorm, run_transformer_block, run_transformer_lm
from GPT import TransformerLM


def test_run_transformer_lm_matches_embedding_blocks_final_norm_and_tied_head() -> None:
    token_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
    token_embedding_weight = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.5],
            [0.0, 1.0, 0.5, -0.5],
            [1.5, -0.5, 0.0, 1.0],
            [-1.0, 0.5, 1.0, 0.0],
            [0.25, 1.25, -0.75, 0.75],
        ],
        dtype=torch.float32,
    )

    block_norm1_weights = torch.tensor(
        [
            [1.0, 1.25, 0.75, 1.5],
            [0.5, 1.0, 1.5, 0.75],
        ],
        dtype=torch.float32,
    )
    block_norm2_weights = torch.tensor(
        [
            [0.75, 1.0, 1.25, 0.5],
            [1.5, 0.75, 1.0, 1.25],
        ],
        dtype=torch.float32,
    )
    q_proj_weights = torch.stack(
        [
            torch.eye(4, dtype=torch.float32),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.5, 1.0, 0.0, 0.0],
                    [0.0, 0.5, 1.0, 0.0],
                    [0.0, 0.0, 0.5, 1.0],
                ],
                dtype=torch.float32,
            ),
        ]
    )
    k_proj_weights = torch.stack(
        [
            torch.tensor(
                [
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            torch.eye(4, dtype=torch.float32),
        ]
    )
    v_proj_weights = torch.stack(
        [
            torch.tensor(
                [
                    [1.0, 0.0, 0.5, 0.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.5, 0.0, 1.0, 0.0],
                    [0.0, 0.5, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [0.5, 0.5, 0.0, 0.0],
                    [0.0, 1.0, 0.5, 0.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.5, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        ]
    )
    o_proj_weights = torch.stack(
        [
            torch.eye(4, dtype=torch.float32),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.25, 0.0, 1.0, 0.0],
                    [0.0, -0.5, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        ]
    )
    ffn_w1_weights = torch.stack(
        [
            torch.tensor(
                [
                    [1.0, -1.0, 0.5, 0.0],
                    [0.0, 2.0, -0.5, 1.0],
                    [1.5, 0.5, 1.0, -1.0],
                    [-1.0, 1.0, 2.0, 0.5],
                    [0.5, 0.5, -1.0, 1.0],
                    [1.0, -0.5, 0.0, 1.5],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [0.5, 0.0, -1.0, 1.0],
                    [1.0, 1.5, 0.0, -0.5],
                    [-0.5, 0.75, 1.0, 0.25],
                    [1.25, -1.0, 0.5, 0.0],
                    [0.0, 0.5, 1.5, -1.0],
                    [1.0, -0.25, 0.75, 0.5],
                ],
                dtype=torch.float32,
            ),
        ]
    )
    ffn_w2_weights = torch.stack(
        [
            torch.tensor(
                [
                    [1.0, 0.0, 0.5, -1.0, 0.25, 0.0],
                    [0.0, 1.0, -0.5, 2.0, 0.5, -0.25],
                    [1.5, -1.0, 0.0, 0.5, 1.0, 0.5],
                    [0.25, 0.75, -0.25, 0.0, 1.0, -1.5],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [0.5, 1.0, 0.0, -0.5, 1.25, 0.0],
                    [1.0, -0.25, 0.5, 0.0, 0.75, -1.0],
                    [0.0, 0.5, 1.0, 0.25, -0.5, 1.5],
                    [-0.75, 0.0, 0.25, 1.0, 0.0, 0.5],
                ],
                dtype=torch.float32,
            ),
        ]
    )
    ffn_w3_weights = torch.stack(
        [
            torch.tensor(
                [
                    [0.5, 1.0, -1.0, 0.25],
                    [1.0, 0.0, 0.5, -0.5],
                    [-0.5, 2.0, 1.0, 0.5],
                    [1.0, -1.5, 0.0, 1.0],
                    [0.25, 0.5, 1.5, -1.0],
                    [1.5, 0.0, -0.5, 0.75],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [1.0, -0.5, 0.25, 0.0],
                    [0.5, 1.0, -1.0, 0.5],
                    [0.0, 0.75, 1.0, -0.5],
                    [1.25, 0.0, 0.5, 1.0],
                    [-0.5, 1.5, 0.0, 0.25],
                    [0.75, -1.0, 1.0, 0.5],
                ],
                dtype=torch.float32,
            ),
        ]
    )
    final_norm_weight = torch.tensor([1.0, 0.75, 1.25, 0.5], dtype=torch.float32)
    token_positions = torch.tensor([0, 1, 2], dtype=torch.long)

    logits = run_transformer_lm(
        vocab_size=5,
        context_length=8,
        num_layers=2,
        d_model=4,
        num_heads=2,
        d_ff=6,
        token_embedding_weight=token_embedding_weight,
        block_attn_q_proj_weights=q_proj_weights,
        block_attn_k_proj_weights=k_proj_weights,
        block_attn_v_proj_weights=v_proj_weights,
        block_attn_o_proj_weights=o_proj_weights,
        block_norm1_weights=block_norm1_weights,
        block_norm2_weights=block_norm2_weights,
        block_ffn_w1_weights=ffn_w1_weights,
        block_ffn_w2_weights=ffn_w2_weights,
        block_ffn_w3_weights=ffn_w3_weights,
        final_norm_weight=final_norm_weight,
        token_ids=token_ids,
        token_positions=token_positions,
    )

    x = token_embedding_weight[token_ids]
    for layer_idx in range(2):
        x = run_transformer_block(
            d_model=4,
            num_heads=2,
            d_ff=6,
            attn_q_proj_weight=q_proj_weights[layer_idx],
            attn_k_proj_weight=k_proj_weights[layer_idx],
            attn_v_proj_weight=v_proj_weights[layer_idx],
            attn_o_proj_weight=o_proj_weights[layer_idx],
            norm1_weight=block_norm1_weights[layer_idx],
            norm2_weight=block_norm2_weights[layer_idx],
            ffn_w1_weight=ffn_w1_weights[layer_idx],
            ffn_w2_weight=ffn_w2_weights[layer_idx],
            ffn_w3_weight=ffn_w3_weights[layer_idx],
            in_features=x,
            max_seq_len=8,
            token_positions=token_positions,
        )
    x = run_rmsnorm(d_model=4, eps=1e-5, weights=final_norm_weight, in_features=x)
    expected = torch.einsum("btd,vd->btv", x, token_embedding_weight)

    assert logits.shape == (2, 3, 5)
    torch.testing.assert_close(logits, expected)


def test_transformer_lm_ties_lm_head_to_embedding_weights() -> None:
    model = TransformerLM(
        vocab_size=11,
        context_length=7,
        num_layers=2,
        d_model=8,
        num_heads=2,
        d_ff=24,
    )

    expected_parameters = (
        (11 * 8)
        + 2 * ((4 * 8 * 8) + (3 * 24 * 8) + (2 * 8))
        + 8
    )
    actual_parameters = sum(param.numel() for param in model.parameters())

    assert actual_parameters == expected_parameters
