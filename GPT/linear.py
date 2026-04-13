from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """
    Bias-free linear transformation used throughout the Transformer stack.

    This module intentionally mirrors the most relevant parts of `torch.nn.Linear`
    while following the assignment requirements exactly:
    - it subclasses `nn.Module`
    - it stores the parameter as `W`, not `W.T`
    - it does not expose or learn a bias parameter
    - it initializes weights with the assignment's truncated-normal scheme

    Parameter layout:
    - `weight` has shape `(out_features, in_features)`

    Forward contract:
    - input shape `(..., in_features)` produces output shape `(..., out_features)`

    The leading dimensions are treated as arbitrary batch-like dimensions. This lets
    the same layer operate on plain matrices, batched sequence activations, or other
    higher-rank tensors common in Transformer implementations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct a bias-free linear layer with assignment-specified initialization.

        Args:
            in_features:
                Size of the final input dimension consumed by the layer.
            out_features:
                Size of the final output dimension produced by the layer.
            device:
                Optional device on which to allocate the parameter tensor.
            dtype:
                Optional dtype of the parameter tensor.

        Initialization rule from the assignment:
        - weights are drawn from a zero-mean normal distribution with variance
          `2 / (in_features + out_features)`
        - the distribution is truncated to the interval `[-3 * std, 3 * std]`

        The parameter is stored as `self.weight` with shape
        `(out_features, in_features)`. During the forward pass we multiply by
        `self.weight.T` so the external behavior matches the usual linear layer API.
        """
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if out_features <= 0:
            raise ValueError("out_features must be positive.")

        self.in_features = in_features
        self.out_features = out_features

        # PyTorch uses a small factory kwargs pattern for parameter creation so device
        # and dtype handling stays explicit and uniform.
        factory_kwargs = {"device": device, "dtype": dtype}

        # The assignment explicitly asks for the learned parameter to be stored as W,
        # not W.T. That means the parameter shape is (out_features, in_features).
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the layer weights with the assignment's truncated normal scheme.

        Derivation:
        - the assignment specifies variance `2 / (d_in + d_out)`
        - standard deviation is therefore `sqrt(2 / (d_in + d_out))`

        Using a dedicated reset method keeps the module consistent with PyTorch module
        conventions and makes it easy to reinitialize the layer in future experiments.
        """
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        # `trunc_normal_` samples from a normal distribution and clips draws to the
        # provided interval. The assignment asks for truncation at +/- 3 standard
        # deviations, which we implement directly here.
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the linear transformation to the final dimension of the input tensor.

        Shape behavior:
        - input:  `(..., in_features)`
        - weight: `(out_features, in_features)`
        - output: `(..., out_features)`

        We use `torch.einsum` because it makes the batched dimension semantics explicit
        and matches the assignment's emphasis on readable tensor algebra. The equation
        says: for every batch-like prefix `...`, contract the input feature axis `i`
        with the corresponding feature axis of the weight matrix and produce an output
        feature axis `o`.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input last dimension {self.in_features}, got {x.shape[-1]}."
            )

        return torch.einsum("...i,oi->...o", x, self.weight)
