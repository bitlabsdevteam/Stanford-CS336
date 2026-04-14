"""
Production-oriented Transformer language model package.

This package is the start of the repository's reusable GPT-style model code. The goal
is to move core model components out of notebooks and into importable modules that are
easy to test, compose, and extend as later assignment tasks add embeddings, RMSNorm,
attention, feed-forward layers, Transformer blocks, and the full language model.
"""

from .embedding import Embedding
from .attention import scaled_dot_product_attention
from .cross_entropy import cross_entropy
from .decoding import decode, generate, sample_next_token, temperature_scaled_softmax, top_p_filter
from .ddp import (
    average_gradients,
    average_gradients_flat,
    BucketedDistributedDataParallel,
    FlatDistributedDataParallel,
    IndividualParameterDistributedDataParallel,
)
from .flash_attention import (
    FlashAttentionForwardAutogradFunctionPyTorch,
    FlashAttentionForwardAutogradFunctionTriton,
    flash_attention_backward_pytorch,
    flash_attention_forward_pytorch,
    flash_attention_forward_triton,
)
from .linear import Linear
from .multihead_attention import CausalMultiheadSelfAttention
from .optimization import AdamW, ShardedOptimizer, clip_gradients, lr_cosine_schedule
from .rotary import RotaryPositionalEmbedding
from .rmsnorm import RMSNorm
from .softmax import softmax
from .swiglu import SwiGLU
from .training import (
    TrainingConfig,
    estimate_loss,
    format_metrics,
    get_batch,
    load_checkpoint,
    load_token_array,
    resolve_training_device,
    save_checkpoint,
    train_language_model,
)
from .transformer_block import TransformerBlock
from .transformer_lm import TransformerLM

__all__ = [
    "Embedding",
    "Linear",
    "AdamW",
    "ShardedOptimizer",
    "FlashAttentionForwardAutogradFunctionPyTorch",
    "FlashAttentionForwardAutogradFunctionTriton",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "CausalMultiheadSelfAttention",
    "TransformerBlock",
    "TransformerLM",
    "TrainingConfig",
    "clip_gradients",
    "scaled_dot_product_attention",
    "cross_entropy",
    "decode",
    "estimate_loss",
    "average_gradients",
    "average_gradients_flat",
    "BucketedDistributedDataParallel",
    "FlatDistributedDataParallel",
    "flash_attention_backward_pytorch",
    "flash_attention_forward_pytorch",
    "flash_attention_forward_triton",
    "format_metrics",
    "generate",
    "get_batch",
    "load_checkpoint",
    "load_token_array",
    "lr_cosine_schedule",
    "resolve_training_device",
    "sample_next_token",
    "save_checkpoint",
    "softmax",
    "IndividualParameterDistributedDataParallel",
    "temperature_scaled_softmax",
    "top_p_filter",
    "train_language_model",
]
