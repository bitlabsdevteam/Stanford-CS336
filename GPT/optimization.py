from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer that follows the assignment pseudocode.

    The optimizer keeps per-parameter first- and second-moment estimates and applies
    weight decay separately from the moment update, matching decoupled AdamW.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Construct an AdamW optimizer over the supplied parameters.

        Args:
            params:
                Iterable of parameters to optimize.
            lr:
                Base learning rate.
            betas:
                Exponential decay rates for first- and second-moment estimates.
            eps:
                Numerical stability term added to the denominator.
            weight_decay:
                Decoupled weight decay coefficient.
        """
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")

        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1).")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1).")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Update all parameters once using decoupled AdamW.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=param.device)
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"].add_(1.0)
                step = int(state["step"].item())

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                denom = exp_avg_sq_hat.sqrt().add_(eps)
                param.addcdiv_(exp_avg_hat, denom, value=-lr)

        return loss


def lr_cosine_schedule(
    it: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Compute the assignment's warmup + cosine annealing learning rate.
    """
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative.")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError("cosine_cycle_iters must be >= warmup_iters.")

    if warmup_iters > 0 and it < warmup_iters:
        return max_lr * it / warmup_iters

    if it <= cosine_cycle_iters:
        span = max(1, cosine_cycle_iters - warmup_iters)
        progress = (it - warmup_iters) / span
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (max_lr - min_lr)

    return min_lr


def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """
    Clip a collection of parameter gradients in place by their global L2 norm.
    """
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be non-negative.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    params_with_grads = [param for param in parameters if param.grad is not None]
    if not params_with_grads:
        return torch.tensor(0.0)

    reference_grad = params_with_grads[0].grad
    assert reference_grad is not None

    total_norm_sq = torch.zeros((), device=reference_grad.device, dtype=reference_grad.dtype)
    for param in params_with_grads:
        grad = param.grad
        assert grad is not None
        total_norm_sq.add_(torch.sum(grad * grad))

    total_norm = torch.sqrt(total_norm_sq)
    clip_scale = min(1.0, max_l2_norm / (float(total_norm.item()) + eps))

    if clip_scale < 1.0:
        for param in params_with_grads:
            grad = param.grad
            assert grad is not None
            grad.mul_(clip_scale)

    return total_norm
