"""Minimal masked diffusion language model components for the Colab notebook."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass
class DiffusionLMConfig:
    """Configuration for the notebook-friendly diffusion Transformer."""

    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1
    diffusion_steps: int = 128
    use_gradient_checkpointing: bool = False

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)


class SinusoidalTimestepEmbedding(nn.Module):
    """Embed discrete diffusion steps with a fixed sinusoidal basis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map integer timesteps to sinusoidal features."""
        half_dim = self.dim // 2
        device = timesteps.device
        steps = timesteps.float().unsqueeze(-1)
        freq_scale = math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -freq_scale
        )
        angles = steps * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class TransformerDenoiserBlock(nn.Module):
    """A standard full-attention Transformer block for masked-token denoising."""

    def __init__(self, config: DiffusionLMConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply full self-attention followed by an MLP residual block."""
        normed = self.ln_1(hidden_states)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class DiffusionTransformerLM(nn.Module):
    """A compact masked diffusion language model with timestep conditioning."""

    def __init__(self, config: DiffusionLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.time_emb = SinusoidalTimestepEmbedding(config.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerDenoiserBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with a GPT-style small normal distribution."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict token logits for every position in the partially masked sequence."""
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum {self.config.seq_len}."
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = self.token_emb(input_ids) + self.pos_emb(positions)
        hidden_states = hidden_states + self.time_mlp(self.time_emb(timesteps)).unsqueeze(1)
        hidden_states = self.dropout(hidden_states)
        key_padding_mask = ~attention_mask

        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    lambda x: block(x, key_padding_mask),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states, key_padding_mask)

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)
