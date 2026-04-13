from __future__ import annotations

import torch
from torch import Tensor, nn

from tests.adapters import (
    run_decode,
    run_sample_next_token,
    run_temperature_scaled_softmax,
    run_top_p_filter,
)


class StepwiseToyLM(nn.Module):
    """
    Tiny deterministic language model used to exercise decoding behavior.
    """

    def __init__(self, transitions: dict[int, Tensor], vocab_size: int, context_length: int) -> None:
        super().__init__()
        self.transitions = transitions
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.anchor = nn.Parameter(torch.zeros(1))
        self.seen_lengths: list[int] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        self.seen_lengths.append(seq_len)
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            fill_value=-1000.0,
            device=token_ids.device,
        )

        for batch_idx in range(batch_size):
            next_token_logits = self.transitions[int(token_ids[batch_idx, -1].item())]
            logits[batch_idx, -1, :] = next_token_logits.to(device=token_ids.device)

        return logits


def test_temperature_scaled_softmax_becomes_greedy_at_zero_temperature() -> None:
    logits = torch.tensor([0.5, 2.0, -1.0], dtype=torch.float32)

    probs = run_temperature_scaled_softmax(logits, temperature=0.0)

    expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    torch.testing.assert_close(probs, expected)


def test_top_p_filter_keeps_smallest_prefix_reaching_target_mass() -> None:
    probs = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float32)

    filtered = run_top_p_filter(probs, top_p=0.7)

    expected = torch.tensor([0.625, 0.375, 0.0, 0.0], dtype=torch.float32)
    torch.testing.assert_close(filtered, expected, atol=1e-7, rtol=1e-7)


def test_sample_next_token_respects_top_p_when_sampling() -> None:
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)

    sampled = run_sample_next_token(
        logits,
        temperature=1.0,
        top_p=0.8,
        generator=generator,
    )

    assert int(sampled) in {0, 1}


def test_decode_generates_until_eos() -> None:
    model = StepwiseToyLM(
        transitions={
            0: torch.tensor([-5.0, 5.0, -5.0, -5.0], dtype=torch.float32),
            1: torch.tensor([-5.0, -5.0, -5.0, 6.0], dtype=torch.float32),
            3: torch.tensor([-5.0, -5.0, -5.0, 6.0], dtype=torch.float32),
        },
        vocab_size=4,
        context_length=6,
    )

    generated = run_decode(
        model,
        prompt_token_ids=torch.tensor([0], dtype=torch.long),
        max_new_tokens=5,
        end_of_text_token_id=3,
        temperature=0.0,
        top_p=1.0,
    )

    expected = torch.tensor([0, 1, 3], dtype=torch.long)
    torch.testing.assert_close(generated, expected)


def test_decode_crops_prompt_to_context_window() -> None:
    model = StepwiseToyLM(
        transitions={
            2: torch.tensor([-5.0, -5.0, 4.0], dtype=torch.float32),
        },
        vocab_size=3,
        context_length=3,
    )

    generated = run_decode(
        model,
        prompt_token_ids=torch.tensor([0, 1, 2, 2], dtype=torch.long),
        max_new_tokens=2,
        temperature=0.0,
        top_p=1.0,
    )

    expected = torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.long)
    torch.testing.assert_close(generated, expected)
    assert model.seen_lengths == [3, 3]
