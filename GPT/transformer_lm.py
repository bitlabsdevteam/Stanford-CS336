from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .embedding import Embedding
from . import nvtx
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """
    GPT-style Transformer language model with tied token embedding / LM head weights.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if d_ff <= 0:
            raise ValueError("d_ff must be positive.")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = float(theta)
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            **factory_kwargs,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=theta,
                    max_seq_len=context_length,
                    eps=eps,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model=d_model, eps=eps, **factory_kwargs)

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Embed token ids, apply Transformer blocks, normalize, and project to logits.
        """
        if token_ids.ndim < 1:
            raise ValueError("token_ids must include a sequence dimension.")
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context_length {self.context_length}."
            )

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)

        with nvtx.range("transformer_lm_forward"):
            with nvtx.range("token_embedding"):
                x = self.token_embedding(token_ids)
            with nvtx.range("transformer_blocks"):
                for block in self.blocks:
                    x = block(x, token_positions=token_positions)
            with nvtx.range("final_norm"):
                x = self.final_norm(x)
            with nvtx.range("lm_head"):
                return F.linear(x, self.token_embedding.weight)

    def generate(
        self,
        prompt_token_ids: Tensor | list[int],
        *,
        max_new_tokens: int,
        end_of_text_token_id: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """
        Convenience wrapper around the package-level decoding helper.
        """
        from .decoding import generate

        return generate(
            model=self,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            end_of_text_token_id=end_of_text_token_id,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
