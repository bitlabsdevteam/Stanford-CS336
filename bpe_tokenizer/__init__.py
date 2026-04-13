"""
Production-oriented byte-level BPE tokenizer package.

This package consolidates the repository's BPE assignment work into reusable modules
that are easier to test, import, and deploy than notebook-local code. The top-level
exports intentionally cover the common workflows:
- training a tokenizer
- loading and saving vocabulary / merge artifacts
- encoding and decoding text
- serializing token-id datasets in `uint16` NumPy format
"""

from .experiments import (
    compression_ratio_bytes_per_token,
    encode_text_file_to_uint16,
    estimate_tokenization_time_seconds,
    estimate_tokenizer_throughput_bytes_per_second,
    sample_documents,
)
from .serialization import load_merges, load_vocab, save_merges, save_token_ids_uint16, save_vocab
from .tokenizer import Tokenizer
from .trainer import train_bpe, train_bpe_with_profile

__all__ = [
    "Tokenizer",
    "compression_ratio_bytes_per_token",
    "encode_text_file_to_uint16",
    "estimate_tokenization_time_seconds",
    "estimate_tokenizer_throughput_bytes_per_second",
    "load_merges",
    "load_vocab",
    "sample_documents",
    "save_merges",
    "save_token_ids_uint16",
    "save_vocab",
    "train_bpe",
    "train_bpe_with_profile",
]
