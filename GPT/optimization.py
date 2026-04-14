from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.distributed as dist
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


def _distributed_sharding_enabled() -> bool:
    """
    Return whether optimizer state sharding collectives should be used.
    """
    return dist.is_initialized() and dist.get_world_size() > 1


class ShardedOptimizer(torch.optim.Optimizer):
    """
    Wrap an arbitrary optimizer class and shard optimizer state across ranks.

    Each rank owns a deterministic subset of parameters. The wrapped optimizer only
    tracks state for the local shard, and after each step the owning rank broadcasts
    its updated parameters so every replica stays synchronized.
    """

    def __init__(
        self,
        params,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs,
    ) -> None:
        """
        Initialize the sharded optimizer over ``params`` using ``optimizer_cls``.

        Args:
            params:
                Iterable of parameters or parameter-group dictionaries.
            optimizer_cls:
                Optimizer type to instantiate on the local shard.
            **kwargs:
                Keyword arguments forwarded to ``optimizer_cls``.
        """
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = dict(kwargs)
        self._wrapped_optimizer: torch.optim.Optimizer | None = None
        self._pending_local_param_groups: list[dict[str, object]] = []
        self._all_params: list[nn.Parameter] = []
        self._param_owners: dict[int, int] = {}
        self._known_param_ids: set[int] = set()
        self._next_param_index = 0
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(params, kwargs)

        local_groups = [group for group in self._pending_local_param_groups if group["params"]]
        if local_groups:
            self._wrapped_optimizer = optimizer_cls(local_groups, **kwargs)
            self.param_groups = self._wrapped_optimizer.param_groups
            self.state = self._wrapped_optimizer.state
        else:
            self.param_groups = []
            self.state = defaultdict(dict)

    def _assign_owner(self, param: nn.Parameter) -> int:
        """
        Assign a stable owner rank to ``param`` using round-robin sharding.
        """
        owner = self._next_param_index % self._world_size
        self._next_param_index += 1
        self._param_owners[id(param)] = owner
        self._all_params.append(param)
        return owner

    def _normalize_param_group(self, param_group: dict[str, object]) -> tuple[dict[str, object], list[nn.Parameter]]:
        """
        Validate and canonicalize a parameter-group dictionary.
        """
        group = dict(param_group)
        if "params" not in group:
            raise ValueError("parameter group must define 'params'")

        raw_params = group["params"]
        if isinstance(raw_params, torch.Tensor):
            params = [raw_params]
        else:
            params = list(raw_params)
        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        normalized_params: list[nn.Parameter] = []
        for param in params:
            if not isinstance(param, torch.nn.Parameter):
                raise TypeError("optimizer can only optimize torch.nn.Parameter instances")
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")
            param_id = id(param)
            if param_id in self._known_param_ids:
                raise ValueError("some parameters appear in more than one parameter group")
            self._known_param_ids.add(param_id)
            normalized_params.append(param)

        group["params"] = normalized_params
        return group, normalized_params

    def add_param_group(self, param_group: dict[str, object]) -> None:
        """
        Add a parameter group and assign its parameters across ranks.
        """
        group, params = self._normalize_param_group(param_group)
        local_params = [param for param in params if self._assign_owner(param) == self._rank]
        local_group = dict(group)
        local_group["params"] = local_params

        if self._wrapped_optimizer is None:
            self._pending_local_param_groups.append(local_group)
            return

        if local_params:
            self._wrapped_optimizer.add_param_group(local_group)
            self.param_groups = self._wrapped_optimizer.param_groups
            self.state = self._wrapped_optimizer.state

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clear gradients for all replicated parameters, not just the local shard.
        """
        for param in self._all_params:
            grad = param.grad
            if grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                if grad.grad_fn is not None:
                    grad.detach_()
                else:
                    grad.requires_grad_(False)
                grad.zero_()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """
        Step the wrapped optimizer on the local shard, then broadcast updates.
        """
        if self._wrapped_optimizer is None:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
        else:
            loss = self._wrapped_optimizer.step(closure=closure, **kwargs)
            self.param_groups = self._wrapped_optimizer.param_groups
            self.state = self._wrapped_optimizer.state

        if _distributed_sharding_enabled():
            for param in self._all_params:
                dist.broadcast(param.data, src=self._param_owners[id(param)])

        return loss

    def state_dict(self) -> dict[str, object]:
        """
        Return the wrapped optimizer state dictionary for the local shard.
        """
        if self._wrapped_optimizer is None:
            return super().state_dict()
        return self._wrapped_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """
        Load a local-shard optimizer state dictionary.
        """
        if self._wrapped_optimizer is None:
            super().load_state_dict(state_dict)
            return
        self._wrapped_optimizer.load_state_dict(state_dict)
        self.param_groups = self._wrapped_optimizer.param_groups
        self.state = self._wrapped_optimizer.state


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
