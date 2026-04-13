from __future__ import annotations

import torch
from torch import Tensor, nn

from .softmax import softmax


def temperature_scaled_softmax(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Convert logits into probabilities after optional temperature scaling.

    A temperature of `0.0` is treated as greedy decoding and returns a one-hot
    distribution at the maximum-logit index.
    """
    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

    if temperature == 0:
        argmax_indices = logits.argmax(dim=-1, keepdim=True)
        probs = torch.zeros_like(logits)
        return probs.scatter(dim=-1, index=argmax_indices, value=1.0)

    return softmax(logits / temperature, dim=-1)


def top_p_filter(probs: Tensor, top_p: float = 1.0) -> Tensor:
    """
    Keep the smallest probability mass prefix whose cumulative weight reaches `top_p`.
    """
    if probs.ndim == 0:
        raise ValueError("probs must have at least one dimension.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")
    if top_p == 1.0:
        return probs

    squeeze_result = probs.ndim == 1
    working_probs = probs.unsqueeze(0) if squeeze_result else probs

    sorted_probs, sorted_indices = torch.sort(working_probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    keep_sorted = (cumulative_probs - sorted_probs) < top_p

    keep_mask = torch.zeros_like(working_probs, dtype=torch.bool)
    keep_mask.scatter_(dim=-1, index=sorted_indices, src=keep_sorted)

    filtered_probs = torch.where(keep_mask, working_probs, torch.zeros_like(working_probs))
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return filtered_probs.squeeze(0) if squeeze_result else filtered_probs


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Sample the next token index from logits using temperature scaling and top-p sampling.
    """
    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension.")

    squeeze_result = logits.ndim == 1
    working_logits = logits.unsqueeze(0) if squeeze_result else logits

    probs = temperature_scaled_softmax(working_logits, temperature=temperature)
    probs = top_p_filter(probs, top_p=top_p)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    return sampled.squeeze(0) if squeeze_result else sampled


def generate(
    model: nn.Module,
    prompt_token_ids: Tensor | list[int],
    *,
    max_new_tokens: int,
    end_of_text_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Generate a continuation from a prompt until EOS or `max_new_tokens` is reached.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    model_device = _get_model_device(model)
    generated = torch.as_tensor(prompt_token_ids, dtype=torch.long, device=model_device)
    if generated.ndim == 1:
        generated = generated.unsqueeze(0)
        squeeze_result = True
    elif generated.ndim == 2:
        squeeze_result = False
    else:
        raise ValueError("prompt_token_ids must have shape (seq_len,) or (batch, seq_len).")

    if generated.shape[-1] == 0:
        raise ValueError("prompt_token_ids must contain at least one token.")

    finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
    if end_of_text_token_id is not None:
        finished = generated[:, -1].eq(end_of_text_token_id)
        if bool(finished.all()) or max_new_tokens == 0:
            return generated.squeeze(0) if squeeze_result else generated

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                model_input = _crop_to_context_window(model, generated)
                logits = model(model_input)
                next_token = sample_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                )

                if end_of_text_token_id is not None:
                    eos_fill = torch.full_like(next_token, end_of_text_token_id)
                    next_token = torch.where(finished, eos_fill, next_token)

                generated = torch.cat((generated, next_token.unsqueeze(-1)), dim=-1)

                if end_of_text_token_id is not None:
                    finished = finished | next_token.eq(end_of_text_token_id)
                    if bool(finished.all()):
                        break
    finally:
        model.train(was_training)

    return generated.squeeze(0) if squeeze_result else generated


def decode(
    model: nn.Module,
    prompt_token_ids: Tensor | list[int],
    *,
    max_new_tokens: int,
    end_of_text_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Alias for `generate` using the assignment's decoding terminology.
    """
    return generate(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        end_of_text_token_id=end_of_text_token_id,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )


def _crop_to_context_window(model: nn.Module, token_ids: Tensor) -> Tensor:
    context_length = getattr(model, "context_length", None)
    if context_length is None or token_ids.shape[-1] <= context_length:
        return token_ids
    return token_ids[:, -context_length:]


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
