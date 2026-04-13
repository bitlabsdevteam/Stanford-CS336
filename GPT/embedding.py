from __future__ import annotations

import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    """
    Learned token embedding table for Transformer language models.

    This module mirrors the core behavior of `torch.nn.Embedding` while following the
    assignment requirements exactly:
    - it subclasses `nn.Module`
    - it stores the embedding matrix as an `nn.Parameter`
    - the embedding dimension is the final dimension of the parameter tensor
    - it does not call `nn.Embedding` or `torch.nn.functional.embedding`

    Parameter layout:
    - `weight` has shape `(num_embeddings, embedding_dim)`

    Forward contract:
    - input shape `(...,)` containing integer token ids
    - output shape `(..., embedding_dim)` containing the selected embedding vectors

    This layout matches the standard Transformer convention where token ids index rows
    of a vocabulary-sized table and produce dense vectors of size `d_model`.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct an embedding lookup table with assignment-specified initialization.

        Args:
            num_embeddings:
                Size of the discrete vocabulary, i.e. how many token ids are valid.
            embedding_dim:
                Dimension of each learned embedding vector, i.e. `d_model`.
            device:
                Optional device on which to allocate the embedding matrix.
            dtype:
                Optional dtype of the embedding matrix.

        Initialization rule from the assignment:
        - embeddings are drawn from `N(0, 1)`
        - values are truncated to the interval `[-3, 3]`

        The parameter is stored as `self.weight` with shape
        `(num_embeddings, embedding_dim)`, so indexing with token ids directly selects
        the appropriate row vectors.
        """
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Use explicit factory kwargs so device and dtype handling stays consistent
        # with the rest of the custom Transformer modules.
        factory_kwargs = {"device": device, "dtype": dtype}

        # The assignment requires storing the embedding table as an nn.Parameter with
        # the model dimension in the final axis.
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the embedding matrix with the assignment's truncated normal rule.

        Unlike the linear layer, the embedding initialization uses unit variance rather
        than a fan-in / fan-out scaling rule. The truncation bounds are fixed at
        `[-3, 3]`, which directly matches the assignment statement.
        """
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Look up embedding vectors for a tensor of integer token ids.

        Shape behavior:
        - input:  `(...,)` integer token ids
        - weight: `(num_embeddings, embedding_dim)`
        - output: `(..., embedding_dim)`

        Implementation detail:
        - Plain tensor indexing `self.weight[token_ids]` performs exactly the lookup we
          want, including support for arbitrary leading batch-like dimensions.

        Validation:
        - The input must use an integer dtype because floating-point values do not make
          sense as vocabulary indices.
        - The token ids must lie in `[0, num_embeddings)` so every index selects a
          valid embedding row.
        """
        if token_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.long,
        }:
            raise ValueError("token_ids must have an integer dtype.")

        if token_ids.numel() > 0:
            min_token_id = int(token_ids.min().item())
            max_token_id = int(token_ids.max().item())
            if min_token_id < 0 or max_token_id >= self.num_embeddings:
                raise ValueError(
                    "token_ids contain values outside the valid range "
                    f"[0, {self.num_embeddings})."
                )

        return self.weight[token_ids]
