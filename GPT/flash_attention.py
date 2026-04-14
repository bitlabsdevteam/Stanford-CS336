from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


def _validate_flash_attention_inputs(q: Tensor, k: Tensor, v: Tensor) -> None:
    """
    Validate the tensor ranks and core attention shape relationships.
    """
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must each have at least 2 dimensions.")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must share the same leading batch-like dimensions.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same feature dimension.")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence length.")


def _batch_view(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """
    Collapse arbitrary leading dimensions into a single batch dimension.
    """
    leading_shape = x.shape[:-2]
    batch_size = math.prod(leading_shape) if leading_shape else 1
    return x.contiguous().reshape(batch_size, x.shape[-2], x.shape[-1]), leading_shape


def _select_tile_size(length: int) -> int:
    """
    Choose a power-of-two tile size no larger than the sequence length.
    """
    tile_size = 64
    while tile_size > length and tile_size > 16:
        tile_size //= 2
    return tile_size


def flash_attention_forward_pytorch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
    q_tile_size: int | None = None,
    k_tile_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute FlashAttention-2 forward pass with tiled online softmax in PyTorch.
    """
    _validate_flash_attention_inputs(q, k, v)

    q_flat, leading_shape = _batch_view(q)
    k_flat, _ = _batch_view(k)
    v_flat, _ = _batch_view(v)

    batch_size, n_queries, d_model = q_flat.shape
    _, n_keys, d_value = v_flat.shape
    q_tile_size = q_tile_size or _select_tile_size(n_queries)
    k_tile_size = k_tile_size or _select_tile_size(n_keys)

    scale = 1.0 / math.sqrt(d_model)
    output = torch.empty(
        (batch_size, n_queries, d_value),
        device=q.device,
        dtype=torch.float32,
    )
    logsumexp = torch.empty(
        (batch_size, n_queries),
        device=q.device,
        dtype=torch.float32,
    )

    for q_start in range(0, n_queries, q_tile_size):
        q_end = min(q_start + q_tile_size, n_queries)
        q_tile = q_flat[:, q_start:q_end, :].to(torch.float32)
        query_tile_size = q_tile.shape[1]

        acc = torch.zeros(
            (batch_size, query_tile_size, d_value),
            device=q.device,
            dtype=torch.float32,
        )
        l = torch.zeros(
            (batch_size, query_tile_size),
            device=q.device,
            dtype=torch.float32,
        )
        m = torch.full(
            (batch_size, query_tile_size),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        if is_causal:
            q_positions = torch.arange(
                q_start,
                q_end,
                device=q.device,
            ).view(1, query_tile_size, 1)

        for k_start in range(0, n_keys, k_tile_size):
            k_end = min(k_start + k_tile_size, n_keys)
            k_tile = k_flat[:, k_start:k_end, :].to(torch.float32)
            v_tile = v_flat[:, k_start:k_end, :].to(torch.float32)

            scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
            causal_mask = None
            if is_causal:
                k_positions = torch.arange(
                    k_start,
                    k_end,
                    device=q.device,
                ).view(1, 1, k_end - k_start)
                causal_mask = q_positions >= k_positions
                scores = scores.masked_fill(~causal_mask, -1.0e6)

            m_next = torch.maximum(m, scores.max(dim=-1).values)
            p_tilde = torch.exp(scores - m_next.unsqueeze(-1))
            if causal_mask is not None:
                p_tilde = torch.where(causal_mask, p_tilde, torch.zeros_like(p_tilde))

            alpha = torch.exp(m - m_next)
            l = alpha * l + p_tilde.sum(dim=-1)
            acc = alpha.unsqueeze(-1) * acc + torch.matmul(p_tilde, v_tile)
            m = m_next

        output[:, q_start:q_end, :] = acc / l.unsqueeze(-1)
        logsumexp[:, q_start:q_end] = m + torch.log(l)

    output = output.to(v.dtype).reshape(*leading_shape, n_queries, d_value)
    logsumexp = logsumexp.reshape(*leading_shape, n_queries)
    return output, logsumexp


def _validate_flash_attention_backward_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    d_o: Tensor,
    l: Tensor,
) -> None:
    """
    Validate the saved forward tensors and upstream gradient shapes for backward.
    """
    _validate_flash_attention_inputs(q, k, v)
    if o.shape[:-1] != q.shape[:-1]:
        raise ValueError("o must match q on all dimensions except the value dimension.")
    if d_o.shape != o.shape:
        raise ValueError("dO must have the same shape as O.")
    if l.shape != q.shape[:-1]:
        raise ValueError("L must have shape matching q without the feature dimension.")


def _flash_attention_backward_pytorch_impl(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    d_o: Tensor,
    l: Tensor,
    is_causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute the FlashAttention backward pass in PyTorch via recomputation.
    """
    _validate_flash_attention_backward_inputs(q, k, v, o, d_o, l)

    q_flat, leading_shape = _batch_view(q)
    k_flat, _ = _batch_view(k)
    v_flat, _ = _batch_view(v)
    o_flat, _ = _batch_view(o)
    d_o_flat, _ = _batch_view(d_o)
    l_flat = l.contiguous().reshape(-1, l.shape[-1]).to(torch.float32)

    _, n_queries, d_model = q_flat.shape
    _, n_keys, _ = k_flat.shape
    scale = 1.0 / math.sqrt(d_model)

    q_float = q_flat.to(torch.float32)
    k_float = k_flat.to(torch.float32)
    v_float = v_flat.to(torch.float32)
    o_float = o_flat.to(torch.float32)
    d_o_float = d_o_flat.to(torch.float32)

    scores = torch.matmul(q_float, k_float.transpose(-1, -2)) * scale
    attention_mask = None
    if is_causal:
        q_positions = torch.arange(n_queries, device=q.device).view(1, n_queries, 1)
        k_positions = torch.arange(n_keys, device=q.device).view(1, 1, n_keys)
        attention_mask = q_positions >= k_positions
        scores = scores.masked_fill(~attention_mask, -1.0e6)

    p = torch.exp(scores - l_flat.unsqueeze(-1))
    if attention_mask is not None:
        p = torch.where(attention_mask, p, torch.zeros_like(p))

    d_v = torch.matmul(p.transpose(-1, -2), d_o_float)
    d_p = torch.matmul(d_o_float, v_float.transpose(-1, -2))
    d = (o_float * d_o_float).sum(dim=-1)
    d_s = p * (d_p - d.unsqueeze(-1))
    if attention_mask is not None:
        d_s = torch.where(attention_mask, d_s, torch.zeros_like(d_s))

    d_q = torch.matmul(d_s, k_float) * scale
    d_k = torch.matmul(d_s.transpose(-1, -2), q_float) * scale

    d_q = d_q.to(q.dtype).reshape(*leading_shape, n_queries, d_model)
    d_k = d_k.to(k.dtype).reshape(*leading_shape, n_keys, d_model)
    d_v = d_v.to(v.dtype).reshape(*leading_shape, n_keys, v.shape[-1])
    return d_q, d_k, d_v


_compiled_flash_attention_backward_pytorch = None


def flash_attention_backward_pytorch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    d_o: Tensor,
    l: Tensor,
    *,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute the FlashAttention backward pass, using torch.compile when available.
    """
    global _compiled_flash_attention_backward_pytorch

    if _compiled_flash_attention_backward_pytorch is None:
        if hasattr(torch, "compile"):
            _compiled_flash_attention_backward_pytorch = torch.compile(
                _flash_attention_backward_pytorch_impl
            )
        else:
            _compiled_flash_attention_backward_pytorch = False

    if _compiled_flash_attention_backward_pytorch is False:
        return _flash_attention_backward_pytorch_impl(q, k, v, o, d_o, l, is_causal)

    return _compiled_flash_attention_backward_pytorch(q, k, v, o, d_o, l, is_causal)


if triton is not None:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """
        Triton kernel for a single (batch, query-tile) FlashAttention forward tile.
        """
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(D, N_KEYS),
            strides=(stride_kd, stride_kk),
            offsets=(0, 0),
            block_shape=(D, K_TILE_SIZE),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr)
        acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        for key_start in range(0, N_KEYS, K_TILE_SIZE):
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)

            scores = tl.dot(q, k) * scale
            if is_causal:
                k_offsets = key_start + tl.arange(0, K_TILE_SIZE)
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                scores = tl.where(causal_mask, scores, scores - 1.0e6)
            else:
                causal_mask = None

            m_next = tl.maximum(m, tl.max(scores, axis=1))
            p_tilde = tl.exp(scores - m_next[:, None])
            if is_causal:
                p_tilde = tl.where(causal_mask, p_tilde, 0.0)

            alpha = tl.exp(m - m_next)
            acc = acc * alpha[:, None]
            acc = tl.dot(p_tilde.to(v.dtype), v, acc=acc)
            l = alpha * l + tl.sum(p_tilde, axis=1)
            m = m_next

            K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        acc = acc / l[:, None]
        lse = m + tl.log(l)
        tl.store(O_block_ptr, acc.to(v.dtype))

        l_ptrs = (
            L_ptr
            + batch_index * stride_lb
            + (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) * stride_lq
        )
        tl.store(l_ptrs, lse)


def flash_attention_forward_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
    q_tile_size: int | None = None,
    k_tile_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute FlashAttention-2 forward pass with a Triton kernel when available.
    """
    _validate_flash_attention_inputs(q, k, v)

    if triton is None or not q.is_cuda:
        return flash_attention_forward_pytorch(
            q,
            k,
            v,
            is_causal=is_causal,
            q_tile_size=q_tile_size,
            k_tile_size=k_tile_size,
        )

    if v.shape[-1] != q.shape[-1]:
        raise ValueError("The Triton FlashAttention forward path requires v.shape[-1] == q.shape[-1].")

    q_flat, leading_shape = _batch_view(q)
    k_flat, _ = _batch_view(k)
    v_flat, _ = _batch_view(v)

    batch_size, n_queries, d_model = q_flat.shape
    _, n_keys, _ = k_flat.shape
    q_tile_size = q_tile_size or _select_tile_size(n_queries)
    k_tile_size = k_tile_size or _select_tile_size(n_keys)

    output = torch.empty_like(v_flat[:, :n_queries, :])
    logsumexp = torch.empty(
        (batch_size, n_queries),
        device=q.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(d_model)
    grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
    num_warps = 4 if q_tile_size <= 32 else 8

    flash_fwd_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        output,
        logsumexp,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        logsumexp.stride(0),
        logsumexp.stride(1),
        n_queries,
        n_keys,
        scale,
        D=d_model,
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
        is_causal=is_causal,
        num_warps=num_warps,
        num_stages=2,
    )

    output = output.reshape(*leading_shape, n_queries, d_model)
    logsumexp = logsumexp.reshape(*leading_shape, n_queries)
    return output, logsumexp


class FlashAttentionForwardAutogradFunctionPyTorch(torch.autograd.Function):
    """
    PyTorch autograd.Function implementing the tiled FlashAttention forward pass.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        output, logsumexp = flash_attention_forward_pytorch(
            q,
            k,
            v,
            is_causal=is_causal,
        )
        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        (d_o,) = grad_outputs
        l, q, k, v, o = ctx.saved_tensors
        d_q, d_k, d_v = flash_attention_backward_pytorch(
            q,
            k,
            v,
            o,
            d_o,
            l,
            is_causal=ctx.is_causal,
        )
        return d_q, d_k, d_v, None


class FlashAttentionForwardAutogradFunctionTriton(torch.autograd.Function):
    """
    Triton-backed autograd.Function implementing the fused FlashAttention forward pass.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        output, logsumexp = flash_attention_forward_triton(
            q,
            k,
            v,
            is_causal=is_causal,
        )
        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        (d_o,) = grad_outputs
        l, q, k, v, o = ctx.saved_tensors
        d_q, d_k, d_v = flash_attention_backward_pytorch(
            q,
            k,
            v,
            o,
            d_o,
            l,
            is_causal=ctx.is_causal,
        )
        return d_q, d_k, d_v, None


__all__ = [
    "FlashAttentionForwardAutogradFunctionPyTorch",
    "FlashAttentionForwardAutogradFunctionTriton",
    "flash_attention_backward_pytorch",
    "flash_attention_forward_pytorch",
    "flash_attention_forward_triton",
]
