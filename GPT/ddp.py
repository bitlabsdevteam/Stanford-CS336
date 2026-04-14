from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .benchmarking import synchronize_device


def _broadcast_module_state(module: nn.Module, *, src: int = 0) -> None:
    """
    Broadcast parameters and buffers from the source rank to all other ranks.
    """

    if not dist.is_initialized():
        return

    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=src)


def _distributed_enabled() -> bool:
    """
    Return whether collective communication should be issued.
    """

    return dist.is_initialized() and dist.get_world_size() > 1


def average_gradients(module: nn.Module) -> float:
    """
    All-reduce each parameter gradient individually.
    """

    if not _distributed_enabled():
        return 0.0

    world_size = dist.get_world_size()
    reference_param = next(module.parameters(), None)
    if reference_param is None:
        return 0.0

    device = reference_param.device
    synchronize_device(device)
    start_time = time.perf_counter()
    for param in module.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
        param.grad.div_(world_size)
    synchronize_device(device)
    end_time = time.perf_counter()
    return end_time - start_time


def average_gradients_flat(module: nn.Module) -> float:
    """
    All-reduce one flattened gradient tensor across all parameters.
    """

    if not _distributed_enabled():
        return 0.0

    grads = [param.grad for param in module.parameters() if param.grad is not None]
    if not grads:
        return 0.0

    world_size = dist.get_world_size()
    device = grads[0].device
    synchronize_device(device)
    start_time = time.perf_counter()
    flat_grads = _flatten_dense_tensors(grads)
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)
    flat_grads.div_(world_size)
    for grad, synced_grad in zip(grads, _unflatten_dense_tensors(flat_grads, grads)):
        grad.copy_(synced_grad)
    synchronize_device(device)
    end_time = time.perf_counter()
    return end_time - start_time


class FlatDistributedDataParallel(nn.Module):
    """
    Minimal DDP wrapper that synchronizes gradients with one flattened all-reduce.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        _broadcast_module_state(self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        average_gradients_flat(self.module)


class IndividualParameterDistributedDataParallel(nn.Module):
    """
    DDP wrapper that asynchronously all-reduces each gradient when it becomes ready.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles: list[tuple[dist.Work, nn.Parameter]] = []
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._sync_in_flight = False
        _broadcast_module_state(self.module)
        if _distributed_enabled():
            self._register_hooks()

    def _register_hooks(self) -> None:
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            self._hook_handles.append(
                param.register_post_accumulate_grad_hook(self._make_post_accumulate_hook(param))
            )

    def _make_post_accumulate_hook(self, param: nn.Parameter):
        def hook(_: torch.Tensor) -> None:
            if not self.training or not _distributed_enabled():
                return
            if param.grad is None:
                return
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append((handle, param))
            self._sync_in_flight = True

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if not self._sync_in_flight or not _distributed_enabled():
            return

        world_size = dist.get_world_size()
        for handle, param in self._handles:
            handle.wait()
            if param.grad is not None:
                param.grad.div_(world_size)
        self._handles.clear()
        self._sync_in_flight = False


@dataclass(slots=True)
class _BucketState:
    params: list[nn.Parameter]
    ready_count: int = 0
    work: dist.Work | None = None
    flat_grads: torch.Tensor | None = None


class BucketedDistributedDataParallel(nn.Module):
    """
    DDP wrapper that asynchronously all-reduces gradient buckets as they become ready.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        if bucket_size_mb <= 0.0:
            raise ValueError("bucket_size_mb must be positive.")

        self.module = module
        self.bucket_size_mb = float(bucket_size_mb)
        self._bucket_bytes_limit = max(1, int(self.bucket_size_mb * 1024 * 1024))
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._buckets = self._build_buckets()
        self._param_to_bucket: dict[int, _BucketState] = {}
        for bucket in self._buckets:
            for param in bucket.params:
                self._param_to_bucket[id(param)] = bucket
        _broadcast_module_state(self.module)
        if _distributed_enabled():
            self._register_hooks()

    def _build_buckets(self) -> list[_BucketState]:
        trainable_params = [param for param in self.module.parameters() if param.requires_grad]
        reversed_params = list(reversed(trainable_params))
        buckets: list[_BucketState] = []
        current_params: list[nn.Parameter] = []
        current_bytes = 0

        for param in reversed_params:
            param_bytes = param.numel() * param.element_size()
            if current_params and current_bytes + param_bytes > self._bucket_bytes_limit:
                buckets.append(_BucketState(params=current_params))
                current_params = []
                current_bytes = 0
            current_params.append(param)
            current_bytes += param_bytes

        if current_params:
            buckets.append(_BucketState(params=current_params))
        return buckets

    def _register_hooks(self) -> None:
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            self._hook_handles.append(
                param.register_post_accumulate_grad_hook(self._make_post_accumulate_hook(param))
            )

    def _make_post_accumulate_hook(self, param: nn.Parameter):
        def hook(_: torch.Tensor) -> None:
            if not self.training or not _distributed_enabled():
                return
            if param.grad is None:
                return

            bucket = self._param_to_bucket[id(param)]
            bucket.ready_count += 1
            if bucket.ready_count != len(bucket.params):
                return

            grads = [bucket_param.grad for bucket_param in bucket.params if bucket_param.grad is not None]
            if len(grads) != len(bucket.params):
                raise RuntimeError("Bucket became ready before all bucket gradients were materialized.")
            bucket.flat_grads = _flatten_dense_tensors(grads)
            bucket.work = dist.all_reduce(bucket.flat_grads, op=dist.ReduceOp.SUM, async_op=True)

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if not _distributed_enabled():
            return

        world_size = dist.get_world_size()
        for bucket in self._buckets:
            if bucket.work is None:
                if bucket.ready_count == 0:
                    continue
                raise RuntimeError("Gradient bucket was partially populated but never scheduled for all-reduce.")
            assert bucket.flat_grads is not None
            bucket.work.wait()
            bucket.flat_grads.div_(world_size)
            synced_grads = _unflatten_dense_tensors(bucket.flat_grads, [param.grad for param in bucket.params])
            for param, synced_grad in zip(bucket.params, synced_grads):
                assert param.grad is not None
                param.grad.copy_(synced_grad)
            bucket.ready_count = 0
            bucket.work = None
            bucket.flat_grads = None


__all__ = [
    "FlatDistributedDataParallel",
    "IndividualParameterDistributedDataParallel",
    "BucketedDistributedDataParallel",
    "average_gradients",
    "average_gradients_flat",
]
