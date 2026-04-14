from __future__ import annotations

import torch
from torch import Tensor

from GPT import (
    AdamW,
    CausalMultiheadSelfAttention,
    decode,
    Embedding,
    FlashAttentionForwardAutogradFunctionPyTorch,
    FlashAttentionForwardAutogradFunctionTriton,
    generate,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    clip_gradients,
    cross_entropy,
    get_batch,
    load_checkpoint,
    lr_cosine_schedule,
    sample_next_token,
    save_checkpoint,
    scaled_dot_product_attention,
    softmax,
    temperature_scaled_softmax,
    top_p_filter,
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Run Linear with externally supplied weights for assignment testing.
    """
    linear = Linear(
        in_features=d_in,
        out_features=d_out,
        device=weights.device,
        dtype=weights.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(weights)
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Tensor,
    token_ids: Tensor,
) -> Tensor:
    """
    Run Embedding with externally supplied weights for assignment testing.
    """
    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=weights.device,
        dtype=weights.dtype,
    )
    with torch.no_grad():
        embedding.weight.copy_(weights)
    return embedding(token_ids)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Run RMSNorm with externally supplied weights for assignment testing.
    """
    rmsnorm = RMSNorm(
        d_model=d_model,
        eps=eps,
        device=weights.device,
        dtype=weights.dtype,
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weights)
    return rmsnorm(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Run SwiGLU with externally supplied weights for assignment testing.
    """
    swiglu = SwiGLU(
        d_model=d_model,
        d_ff=d_ff,
        device=w1_weight.device,
        dtype=w1_weight.dtype,
    )
    with torch.no_grad():
        swiglu.w1.weight.copy_(w1_weight)
        swiglu.w2.weight.copy_(w2_weight)
        swiglu.w3.weight.copy_(w3_weight)
    return swiglu(in_features)


def run_rope(
    theta: float,
    d_k: int,
    max_seq_len: int,
    in_features: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """
    Run RoPE with externally supplied activations and token positions for testing.
    """
    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_features.device,
    )
    return rope(in_features, token_positions)


def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    """
    Run numerically stable softmax for assignment testing.
    """
    return softmax(in_features, dim=dim)


def run_temperature_scaled_softmax(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Run temperature-scaled softmax for assignment testing.
    """
    return temperature_scaled_softmax(logits, temperature=temperature)


def run_top_p_filter(probs: Tensor, top_p: float = 1.0) -> Tensor:
    """
    Run top-p probability truncation for assignment testing.
    """
    return top_p_filter(probs, top_p=top_p)


def run_sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Run one-step next-token sampling for assignment testing.
    """
    return sample_next_token(
        logits,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )


def run_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Run scaled dot-product attention for assignment testing.
    """
    return scaled_dot_product_attention(q=q, k=k, v=v, mask=mask)


def run_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Run numerically stable mean cross-entropy for assignment testing.
    """
    return cross_entropy(logits=logits, targets=targets)


def run_get_batch(
    x,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """
    Run assignment batch sampling for testing.
    """
    return get_batch(
        x=x,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    *,
    theta: float = 10000.0,
    max_seq_len: int | None = None,
    use_rope: bool = False,
    token_positions: Tensor | None = None,
) -> Tensor:
    """
    Run causal multi-head self-attention with externally supplied weights.
    """
    attention = CausalMultiheadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len if max_seq_len is not None else in_features.shape[-2],
        use_rope=use_rope,
        device=q_proj_weight.device,
        dtype=q_proj_weight.dtype,
    )
    with torch.no_grad():
        attention.q_proj.weight.copy_(q_proj_weight)
        attention.k_proj.weight.copy_(k_proj_weight)
        attention.v_proj.weight.copy_(v_proj_weight)
        attention.o_proj.weight.copy_(o_proj_weight)
    return attention(in_features, token_positions=token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_q_proj_weight: Tensor,
    attn_k_proj_weight: Tensor,
    attn_v_proj_weight: Tensor,
    attn_o_proj_weight: Tensor,
    norm1_weight: Tensor,
    norm2_weight: Tensor,
    ffn_w1_weight: Tensor,
    ffn_w2_weight: Tensor,
    ffn_w3_weight: Tensor,
    in_features: Tensor,
    *,
    eps: float = 1e-5,
    theta: float = 10000.0,
    max_seq_len: int | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    """
    Run a pre-norm Transformer block with externally supplied weights.
    """
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        eps=eps,
        theta=theta,
        max_seq_len=max_seq_len if max_seq_len is not None else in_features.shape[-2],
        device=in_features.device,
        dtype=in_features.dtype,
    )
    with torch.no_grad():
        block.norm1.weight.copy_(norm1_weight)
        block.attn.q_proj.weight.copy_(attn_q_proj_weight)
        block.attn.k_proj.weight.copy_(attn_k_proj_weight)
        block.attn.v_proj.weight.copy_(attn_v_proj_weight)
        block.attn.o_proj.weight.copy_(attn_o_proj_weight)
        block.norm2.weight.copy_(norm2_weight)
        block.ffn.w1.weight.copy_(ffn_w1_weight)
        block.ffn.w2.weight.copy_(ffn_w2_weight)
        block.ffn.w3.weight.copy_(ffn_w3_weight)
    return block(in_features, token_positions=token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    token_embedding_weight: Tensor,
    block_attn_q_proj_weights: Tensor,
    block_attn_k_proj_weights: Tensor,
    block_attn_v_proj_weights: Tensor,
    block_attn_o_proj_weights: Tensor,
    block_norm1_weights: Tensor,
    block_norm2_weights: Tensor,
    block_ffn_w1_weights: Tensor,
    block_ffn_w2_weights: Tensor,
    block_ffn_w3_weights: Tensor,
    final_norm_weight: Tensor,
    token_ids: Tensor,
    *,
    eps: float = 1e-5,
    theta: float = 10000.0,
    token_positions: Tensor | None = None,
) -> Tensor:
    """
    Run the full Transformer LM with externally supplied weights.
    """
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        eps=eps,
        theta=theta,
        device=token_embedding_weight.device,
        dtype=token_embedding_weight.dtype,
    )
    with torch.no_grad():
        model.token_embedding.weight.copy_(token_embedding_weight)
        for layer_idx, block in enumerate(model.blocks):
            block.norm1.weight.copy_(block_norm1_weights[layer_idx])
            block.attn.q_proj.weight.copy_(block_attn_q_proj_weights[layer_idx])
            block.attn.k_proj.weight.copy_(block_attn_k_proj_weights[layer_idx])
            block.attn.v_proj.weight.copy_(block_attn_v_proj_weights[layer_idx])
            block.attn.o_proj.weight.copy_(block_attn_o_proj_weights[layer_idx])
            block.norm2.weight.copy_(block_norm2_weights[layer_idx])
            block.ffn.w1.weight.copy_(block_ffn_w1_weights[layer_idx])
            block.ffn.w2.weight.copy_(block_ffn_w2_weights[layer_idx])
            block.ffn.w3.weight.copy_(block_ffn_w3_weights[layer_idx])
        model.final_norm.weight.copy_(final_norm_weight)
    return model(token_ids=token_ids, token_positions=token_positions)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Return the assignment AdamW optimizer class for hidden test construction.
    """
    return AdamW


def get_flashattention_autograd_function_pytorch() -> type[torch.autograd.Function]:
    """
    Return the PyTorch FlashAttention forward autograd.Function subclass.
    """
    return FlashAttentionForwardAutogradFunctionPyTorch


def get_flash_autograd_function_triton() -> type[torch.autograd.Function]:
    """
    Return the Triton-backed FlashAttention forward autograd.Function subclass.
    """
    return FlashAttentionForwardAutogradFunctionTriton


def get_lr_cosine_schedule(
    it: int | None = None,
    *,
    max_lr: float | None = None,
    min_lr: float | None = None,
    warmup_iters: int | None = None,
    cosine_cycle_iters: int | None = None,
):
    """
    Return the cosine schedule function, or evaluate it directly when arguments are given.
    """
    if it is None:
        return lr_cosine_schedule

    if (
        max_lr is None
        or min_lr is None
        or warmup_iters is None
        or cosine_cycle_iters is None
    ):
        raise TypeError(
            "Direct evaluation requires max_lr, min_lr, warmup_iters, and cosine_cycle_iters."
        )

    return lr_cosine_schedule(
        it,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_gradient_clipping(
    parameters: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    max_l2_norm: float,
) -> Tensor:
    """
    Clip gradients in place and return the pre-clipping global L2 norm.
    """
    return clip_gradients(parameters, max_l2_norm)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
) -> None:
    """
    Save an assignment checkpoint for testing.
    """
    save_checkpoint(model=model, optimizer=optimizer, iteration=iteration, out=out)


def run_load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load an assignment checkpoint for testing.
    """
    return load_checkpoint(src=src, model=model, optimizer=optimizer)


def run_generate(
    model: torch.nn.Module,
    prompt_token_ids: Tensor | list[int],
    *,
    max_new_tokens: int,
    end_of_text_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Run iterative decoding for assignment testing.
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


def run_decode(
    model: torch.nn.Module,
    prompt_token_ids: Tensor | list[int],
    *,
    max_new_tokens: int,
    end_of_text_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Run iterative decoding using the assignment's naming.
    """
    return decode(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        end_of_text_token_id=end_of_text_token_id,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )


__all__ = [
    "run_linear",
    "run_embedding",
    "run_rmsnorm",
    "run_swiglu",
    "run_rope",
    "run_softmax",
    "run_temperature_scaled_softmax",
    "run_top_p_filter",
    "run_sample_next_token",
    "run_cross_entropy",
    "run_get_batch",
    "run_scaled_dot_product_attention",
    "run_multihead_self_attention",
    "run_transformer_block",
    "run_transformer_lm",
    "get_adamw_cls",
    "get_flashattention_autograd_function_pytorch",
    "get_flash_autograd_function_triton",
    "get_lr_cosine_schedule",
    "run_gradient_clipping",
    "run_save_checkpoint",
    "run_load_checkpoint",
    "run_generate",
    "run_decode",
]
