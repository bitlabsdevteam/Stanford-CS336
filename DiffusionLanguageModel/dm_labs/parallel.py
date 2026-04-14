"""Notebook-friendly 5D parallelism scaffolding for diffusion training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn


def find_repo_root() -> Path | None:
    """Walk likely parent paths until the repository root is found."""
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "GPT").exists():
            return candidate
    return None


REPO_ROOT = find_repo_root()
PROJECT_PARALLELISM_AVAILABLE = False
BucketedDistributedDataParallel = None
FlatDistributedDataParallel = None
IndividualParameterDistributedDataParallel = None
ProjectAdamW = None
ShardedOptimizer = None

if REPO_ROOT is not None:
    try:
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from GPT import (  # type: ignore
            BucketedDistributedDataParallel,
            FlatDistributedDataParallel,
            IndividualParameterDistributedDataParallel,
            AdamW as ProjectAdamW,
            ShardedOptimizer,
        )

        PROJECT_PARALLELISM_AVAILABLE = True
    except Exception:
        PROJECT_PARALLELISM_AVAILABLE = False


def resolve_device() -> tuple[torch.device, str]:
    """Pick the best notebook device in CUDA, MPS, CPU order."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


@dataclass
class FiveDParallelConfig:
    """A compact configuration surface matching a 5-axis training mesh."""

    data_parallel: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    context_parallel: int = 1
    sequence_parallel: int = 1
    ddp_strategy: str = "bucketed"
    bucket_size_mb: float = 8.0
    use_sharded_optimizer: bool = True

    @property
    def requested_world_size(self) -> int:
        """Return the total device count implied by the requested mesh."""
        return (
            self.data_parallel
            * self.tensor_parallel
            * self.pipeline_parallel
            * self.context_parallel
            * self.sequence_parallel
        )

    def as_dict(self) -> dict[str, int | float | bool | str]:
        """Return a compact topology summary."""
        return {
            "dp": self.data_parallel,
            "tp": self.tensor_parallel,
            "pp": self.pipeline_parallel,
            "cp": self.context_parallel,
            "sp": self.sequence_parallel,
            "ddp_strategy": self.ddp_strategy,
            "bucket_size_mb": self.bucket_size_mb,
            "use_sharded_optimizer": self.use_sharded_optimizer,
        }


def normalize_parallel_config(
    config: FiveDParallelConfig,
    *,
    device_name: str | None = None,
    cuda_device_count: int | None = None,
) -> FiveDParallelConfig:
    """Collapse unsupported axes while preserving the 5D planning vocabulary."""
    if device_name is None:
        _, device_name = resolve_device()
    if cuda_device_count is None:
        cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    values = [
        config.data_parallel,
        config.tensor_parallel,
        config.pipeline_parallel,
        config.context_parallel,
        config.sequence_parallel,
    ]
    if any(value <= 0 for value in values):
        raise ValueError("All parallelism dimensions must be positive integers.")

    if device_name != "cuda":
        return FiveDParallelConfig(
            data_parallel=1,
            tensor_parallel=1,
            pipeline_parallel=1,
            context_parallel=1,
            sequence_parallel=1,
            ddp_strategy=config.ddp_strategy,
            bucket_size_mb=config.bucket_size_mb,
            use_sharded_optimizer=False,
        )

    active = FiveDParallelConfig(**config.__dict__)
    unsupported_axes = []
    for axis_name in ("tensor_parallel", "pipeline_parallel", "context_parallel", "sequence_parallel"):
        axis_value = getattr(active, axis_name)
        if axis_value > 1:
            unsupported_axes.append(f"{axis_name}={axis_value}")
            setattr(active, axis_name, 1)

    if unsupported_axes:
        warnings.warn(
            "The notebook exposes a 5D topology config, but the in-notebook runtime executes "
            "the data-parallel axis directly. Collapsing unsupported axes to 1: "
            + ", ".join(unsupported_axes),
            stacklevel=2,
        )

    active.data_parallel = max(1, min(active.data_parallel, cuda_device_count))
    if active.data_parallel != config.data_parallel:
        warnings.warn(
            f"Requested data_parallel={config.data_parallel}, but only {cuda_device_count} CUDA device(s) are visible. "
            f"Using data_parallel={active.data_parallel} instead.",
            stacklevel=2,
        )

    if dist.is_initialized() and dist.get_world_size() != active.data_parallel:
        warnings.warn(
            f"torch.distributed is initialized with world_size={dist.get_world_size()}, so the active data-parallel axis "
            f"is forced to {dist.get_world_size()}.",
            stacklevel=2,
        )
        active.data_parallel = dist.get_world_size()

    if not dist.is_initialized():
        active.use_sharded_optimizer = False

    return active


def summarize_parallel_plan(
    requested: FiveDParallelConfig,
    active: FiveDParallelConfig,
    *,
    device_name: str,
) -> None:
    """Print a compact summary of the requested and active runtime mesh."""
    print("requested 5D mesh:", requested.as_dict())
    print("active 5D mesh:", active.as_dict())
    print("requested world size:", requested.requested_world_size)
    print("active world size:", active.requested_world_size)
    if dist.is_initialized():
        print("runtime: torch.distributed")
    elif active.data_parallel > 1 and device_name == "cuda":
        print("runtime: nn.DataParallel")
    else:
        print("runtime: single process")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model regardless of the runtime wrapper."""
    if hasattr(model, "module"):
        return model.module
    return model


def finalize_parallel_backward(model: nn.Module) -> None:
    """Finish gradient synchronization for custom DDP wrappers when present."""
    if hasattr(model, "finish_gradient_synchronization"):
        model.finish_gradient_synchronization()


def training_device_for_runtime(
    active: FiveDParallelConfig,
    *,
    base_device: torch.device,
    device_name: str,
) -> torch.device:
    """Choose the device batches should be moved to before each forward pass."""
    if device_name == "cuda" and active.data_parallel > 1:
        return torch.device("cuda:0")
    return base_device


def prepare_parallel_model(
    model: nn.Module,
    active: FiveDParallelConfig,
    *,
    base_device: torch.device,
    device_name: str,
) -> tuple[nn.Module, torch.device]:
    """Place the model onto the best available notebook parallel wrapper."""
    train_device = training_device_for_runtime(
        active,
        base_device=base_device,
        device_name=device_name,
    )
    model = model.to(train_device)

    if dist.is_initialized() and active.data_parallel > 1 and PROJECT_PARALLELISM_AVAILABLE:
        strategy = active.ddp_strategy.lower()
        if strategy == "flat":
            return FlatDistributedDataParallel(model), train_device
        if strategy == "individual":
            return IndividualParameterDistributedDataParallel(model), train_device
        return BucketedDistributedDataParallel(model, bucket_size_mb=active.bucket_size_mb), train_device

    if device_name == "cuda" and active.data_parallel > 1:
        device_ids = list(range(active.data_parallel))
        return nn.DataParallel(model, device_ids=device_ids), train_device

    return model, train_device


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    active: FiveDParallelConfig,
) -> torch.optim.Optimizer:
    """Build either the project sharded optimizer or a standard AdamW fallback."""
    params = unwrap_model(model).parameters()
    if active.use_sharded_optimizer and PROJECT_PARALLELISM_AVAILABLE and dist.is_initialized():
        return ShardedOptimizer(
            params,
            ProjectAdamW,
            lr=lr,
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    target_device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move all tensor batch members onto the requested device."""
    return {
        key: value.to(target_device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def normalize_loss(loss: torch.Tensor | None) -> torch.Tensor | None:
    """Reduce wrapper-specific loss shapes to a single scalar."""
    if loss is None:
        return None
    if loss.ndim == 0:
        return loss
    return loss.mean()
