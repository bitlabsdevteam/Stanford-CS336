from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from statistics import fmean

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from .benchmarking import MODEL_SIZE_SPECS, create_random_batch, resolve_model_dimensions, synchronize_device
from .cross_entropy import cross_entropy
from .optimization import AdamW
from .transformer_lm import TransformerLM
from .training import resolve_training_device


@dataclass(slots=True)
class NaiveDDPCheckConfig:
    """
    Configuration for validating naïve DDP against a single-process reference run.
    """

    world_size: int = 2
    backend: str = "gloo"
    device: str = "cpu"
    input_dim: int = 8
    hidden_dim: int = 16
    output_dim: int = 4
    global_batch_size: int = 8
    num_steps: int = 4
    learning_rate: float = 1e-2
    seed: int = 0


@dataclass(slots=True)
class NaiveDDPBenchmarkConfig:
    """
    Configuration for benchmarking per-parameter gradient all-reduce training.
    """

    model_size: str = "xl"
    vocab_size: int = 50_257
    global_batch_size: int = 8
    context_length: int = 1024
    num_layers: int | None = None
    d_model: int | None = None
    num_heads: int | None = None
    d_ff: int | None = None
    world_size: int = 2
    backend: str = "nccl"
    device: str = "cuda"
    dtype: str = "float32"
    mixed_precision_dtype: str | None = None
    warmup_steps: int = 2
    measure_steps: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    """
    Map a CLI dtype string to a torch dtype.
    """

    normalized = dtype_name.strip().lower()
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return dtype_map[normalized]


def _setup_process_group(
    *,
    rank: int,
    world_size: int,
    backend: str,
    rendezvous_path: str,
) -> None:
    """
    Initialize a local single-node process group.
    """

    if backend == "gloo" and "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=f"file://{rendezvous_path}",
    )


def _teardown_process_group() -> None:
    """
    Destroy the process group if it exists.
    """

    if dist.is_initialized():
        dist.destroy_process_group()


def _move_state_dict_to_cpu(state_dict: dict[str, object]) -> dict[str, object]:
    """
    Recursively move tensors inside a state dict onto CPU for comparison or serialization.
    """

    moved: dict[str, object] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            moved[key] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            moved[key] = _move_state_dict_to_cpu(value)
        elif isinstance(value, list):
            moved[key] = [
                _move_state_dict_to_cpu(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            moved[key] = value
    return moved


def _state_dict_max_abs_diff(left: object, right: object) -> float:
    """
    Compute the maximum absolute tensor difference across matching nested structures.
    """

    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.numel() == 0 and right.numel() == 0:
            return 0.0
        return float(torch.max(torch.abs(left.to(torch.float64) - right.to(torch.float64))).item())
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError("State dict keys do not match.")
        return max((_state_dict_max_abs_diff(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("State dict list lengths do not match.")
        return max((_state_dict_max_abs_diff(l_item, r_item) for l_item, r_item in zip(left, right)), default=0.0)
    if left != right:
        raise ValueError(f"State dict values do not match: {left!r} != {right!r}.")
    return 0.0


def _broadcast_module_state(module: nn.Module, *, src: int = 0) -> None:
    """
    Broadcast parameters and buffers from the source rank to all other ranks.
    """

    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=src)


def average_gradients(module: nn.Module) -> float:
    """
    All-reduce each parameter gradient and return the communication time in seconds.
    """

    if not dist.is_initialized():
        raise RuntimeError("average_gradients requires an initialized process group.")

    world_size = dist.get_world_size()
    if world_size <= 0:
        raise RuntimeError("Process group world_size must be positive.")

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


def shard_batch(
    tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Slice a batch tensor into equal-sized per-rank shards along the batch dimension.
    """

    if tensor.shape[0] % world_size != 0:
        raise ValueError(
            f"Global batch size {tensor.shape[0]} must be divisible by world_size {world_size}."
        )
    shard_size = tensor.shape[0] // world_size
    start = rank * shard_size
    end = start + shard_size
    return tensor[start:end]


def _autocast_context(device: torch.device, mixed_precision_dtype: torch.dtype | None):
    """
    Return the appropriate autocast context for the requested device and dtype.
    """

    if mixed_precision_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=mixed_precision_dtype)


def _build_toy_model(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    device: torch.device,
) -> nn.Module:
    """
    Construct a small MLP used for DDP correctness checks.
    """

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, device=device),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim, device=device),
    )


def _build_toy_batches(config: NaiveDDPCheckConfig) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Create deterministic random training batches shared by the reference and DDP runs.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(config.num_steps):
        inputs = torch.randn(
            config.global_batch_size,
            config.input_dim,
            generator=generator,
            dtype=torch.float32,
        )
        targets = torch.randn(
            config.global_batch_size,
            config.output_dim,
            generator=generator,
            dtype=torch.float32,
        )
        batches.append((inputs, targets))
    return batches


def run_single_process_reference(
    config: NaiveDDPCheckConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    """
    Train the toy model in a single process and return final model and optimizer states.
    """

    device = resolve_training_device(config.device)
    torch.manual_seed(config.seed)
    model = _build_toy_model(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    batches = _build_toy_batches(config)
    loss_fn = nn.MSELoss(reduction="mean")

    for inputs_cpu, targets_cpu in batches:
        optimizer.zero_grad(set_to_none=True)
        inputs = inputs_cpu.to(device)
        targets = targets_cpu.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    return _move_state_dict_to_cpu(model.state_dict()), _move_state_dict_to_cpu(optimizer.state_dict())


def run_simulated_naive_ddp_reference(
    config: NaiveDDPCheckConfig,
) -> tuple[dict[str, object], dict[str, object], float]:
    """
    Run the naïve DDP algorithm locally without torch.distributed for restricted environments.
    """

    device = resolve_training_device(config.device)
    batches = _build_toy_batches(config)
    loss_fn = nn.MSELoss(reduction="mean")

    models: list[nn.Module] = []
    optimizers: list[torch.optim.Optimizer] = []
    for rank in range(config.world_size):
        torch.manual_seed(config.seed + rank)
        model = _build_toy_model(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        models.append(model)
        optimizers.append(optimizer)

    reference_state = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict(reference_state)

    for inputs_cpu, targets_cpu in batches:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        for rank, model in enumerate(models):
            inputs = shard_batch(inputs_cpu, rank=rank, world_size=config.world_size).to(device)
            targets = shard_batch(targets_cpu, rank=rank, world_size=config.world_size).to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()

        parameter_groups = [list(model.parameters()) for model in models]
        for params_per_rank in zip(*parameter_groups):
            grads = [param.grad for param in params_per_rank]
            if any(grad is None for grad in grads):
                continue
            average_grad = sum(grad for grad in grads if grad is not None) / config.world_size
            for param in params_per_rank:
                assert param.grad is not None
                param.grad.copy_(average_grad)

        for optimizer in optimizers:
            optimizer.step()

    replica_model_states = [_move_state_dict_to_cpu(model.state_dict()) for model in models]
    replica_optimizer_states = [_move_state_dict_to_cpu(optimizer.state_dict()) for optimizer in optimizers]
    replica_model_diff = max(
        _state_dict_max_abs_diff(replica_model_states[0], other_state)
        for other_state in replica_model_states[1:]
    ) if len(replica_model_states) > 1 else 0.0

    return replica_model_states[0], replica_optimizer_states[0], replica_model_diff


def _naive_ddp_check_worker(
    rank: int,
    config: NaiveDDPCheckConfig,
    rendezvous_path: str,
    queue,
) -> None:
    """
    Spawned worker that runs the naïve DDP toy training loop.
    """

    try:
        _setup_process_group(
            rank=rank,
            world_size=config.world_size,
            backend=config.backend,
            rendezvous_path=rendezvous_path,
        )

        if config.device.startswith("cuda"):
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = resolve_training_device(config.device)

        torch.manual_seed(config.seed)
        model = _build_toy_model(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        _broadcast_module_state(model, src=0)
        batches = _build_toy_batches(config)
        loss_fn = nn.MSELoss(reduction="mean")

        for inputs_cpu, targets_cpu in batches:
            optimizer.zero_grad(set_to_none=True)
            inputs = shard_batch(inputs_cpu, rank=rank, world_size=config.world_size).to(device)
            targets = shard_batch(targets_cpu, rank=rank, world_size=config.world_size).to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            average_gradients(model)
            optimizer.step()

        if rank == 0:
            queue.put(
                {
                    "status": "ok",
                    "model_state_dict": _move_state_dict_to_cpu(model.state_dict()),
                    "optimizer_state_dict": _move_state_dict_to_cpu(optimizer.state_dict()),
                }
            )
    except Exception as exc:
        if rank == 0:
            queue.put({"status": "error", "error": repr(exc)})
        raise
    finally:
        _teardown_process_group()


def run_naive_ddp_check(
    config: NaiveDDPCheckConfig,
    *,
    spawn_processes: bool = True,
) -> dict[str, object]:
    """
    Compare single-process toy training with naïve DDP training.
    """

    if config.world_size <= 0:
        raise ValueError("world_size must be positive.")
    if config.global_batch_size % config.world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size.")
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    reference_model_state, reference_optimizer_state = run_single_process_reference(config)

    if spawn_processes:
        ctx = mp.get_context("spawn")
        queue = ctx.SimpleQueue()
        with tempfile.NamedTemporaryFile(prefix="naive_ddp_check_", dir="/tmp", delete=False) as handle:
            rendezvous_path = handle.name

        try:
            mp.spawn(
                _naive_ddp_check_worker,
                args=(config, rendezvous_path, queue),
                nprocs=config.world_size,
                join=True,
            )
        except Exception as exc:
            return {
                **asdict(config),
                "status": "error",
                "error": repr(exc),
                "execution_mode": "spawn",
            }
        finally:
            if os.path.exists(rendezvous_path):
                os.unlink(rendezvous_path)

        worker_result = queue.get()
        if worker_result["status"] != "ok":
            return {**asdict(config), **worker_result, "execution_mode": "spawn"}
        ddp_model_state = worker_result["model_state_dict"]
        ddp_optimizer_state = worker_result["optimizer_state_dict"]
        replica_model_diff = 0.0
        execution_mode = "spawn"
    else:
        ddp_model_state, ddp_optimizer_state, replica_model_diff = run_simulated_naive_ddp_reference(config)
        execution_mode = "simulated"

    model_diff = _state_dict_max_abs_diff(reference_model_state, ddp_model_state)
    optimizer_diff = _state_dict_max_abs_diff(reference_optimizer_state, ddp_optimizer_state)
    return {
        **asdict(config),
        "status": "ok",
        "execution_mode": execution_mode,
        "reference_model_state_dict": reference_model_state,
        "ddp_model_state_dict": ddp_model_state,
        "reference_optimizer_state_dict": reference_optimizer_state,
        "ddp_optimizer_state_dict": ddp_optimizer_state,
        "max_model_abs_diff": model_diff,
        "max_optimizer_abs_diff": optimizer_diff,
        "max_inter_replica_model_abs_diff": replica_model_diff,
    }


def _validate_benchmark_config(config: NaiveDDPBenchmarkConfig) -> str | None:
    """
    Return a skip reason when the benchmark cannot run on the current machine.
    """

    if config.world_size <= 0:
        return "world_size must be positive."
    if config.global_batch_size % config.world_size != 0:
        return "global_batch_size must be divisible by world_size."
    if config.backend == "nccl" and not config.device.startswith("cuda"):
        return "NCCL requires CUDA devices."
    if config.backend == "gloo" and config.device.startswith("cuda"):
        return "This benchmark uses Gloo only for CPU tensors."
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            return "CUDA is not available."
        if torch.cuda.device_count() < config.world_size:
            return (
                f"Requested {config.world_size} CUDA ranks, but only {torch.cuda.device_count()} GPU(s) are available."
            )
    return None


def _naive_ddp_benchmark_worker(
    rank: int,
    config: NaiveDDPBenchmarkConfig,
    dimensions: dict[str, int],
    rendezvous_path: str,
    queue,
) -> None:
    """
    Spawned worker that benchmarks naïve per-parameter gradient all-reduce training.
    """

    try:
        _setup_process_group(
            rank=rank,
            world_size=config.world_size,
            backend=config.backend,
            rendezvous_path=rendezvous_path,
        )

        if config.device.startswith("cuda"):
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = resolve_training_device(config.device)

        torch.manual_seed(config.seed)
        dtype = _torch_dtype_from_name(config.dtype)
        mixed_precision_dtype = (
            None if config.mixed_precision_dtype is None else _torch_dtype_from_name(config.mixed_precision_dtype)
        )
        model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=dimensions["num_layers"],
            d_model=dimensions["d_model"],
            num_heads=dimensions["num_heads"],
            d_ff=dimensions["d_ff"],
            device=device,
            dtype=dtype,
        )
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        _broadcast_module_state(model, src=0)

        local_batch_size = config.global_batch_size // config.world_size
        total_timings: list[float] = []
        communication_timings: list[float] = []

        total_steps = config.warmup_steps + config.measure_steps
        for step_idx in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = create_random_batch(
                batch_size=local_batch_size,
                context_length=config.context_length,
                vocab_size=config.vocab_size,
                device=device,
            )

            dist.barrier()
            synchronize_device(device)
            total_start = time.perf_counter()
            with _autocast_context(device, mixed_precision_dtype):
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
            loss.backward()
            communication_time = average_gradients(model)
            optimizer.step()
            synchronize_device(device)
            total_end = time.perf_counter()

            if step_idx >= config.warmup_steps:
                total_timings.append(total_end - total_start)
                communication_timings.append(communication_time)

        gathered_total: list[list[float] | None] = [None for _ in range(config.world_size)]
        gathered_comm: list[list[float] | None] = [None for _ in range(config.world_size)]
        dist.all_gather_object(gathered_total, total_timings)
        dist.all_gather_object(gathered_comm, communication_timings)

        if rank == 0:
            per_step_total_max = [
                max(float(rank_timings[step_idx]) for rank_timings in gathered_total if rank_timings is not None)
                for step_idx in range(config.measure_steps)
            ]
            per_step_comm_max = [
                max(float(rank_timings[step_idx]) for rank_timings in gathered_comm if rank_timings is not None)
                for step_idx in range(config.measure_steps)
            ]
            queue.put(
                {
                    "status": "ok",
                    "rank_total_step_seconds": gathered_total,
                    "rank_communication_seconds": gathered_comm,
                    "step_timings_seconds": per_step_total_max,
                    "communication_timings_seconds": per_step_comm_max,
                    "mean_step_seconds": fmean(per_step_total_max),
                    "mean_communication_seconds": fmean(per_step_comm_max),
                    "communication_fraction": (
                        fmean(per_step_comm_max) / fmean(per_step_total_max) if fmean(per_step_total_max) > 0.0 else 0.0
                    ),
                }
            )
    except Exception as exc:
        if rank == 0:
            queue.put({"status": "error", "error": repr(exc)})
        raise
    finally:
        _teardown_process_group()


def run_naive_ddp_benchmark(config: NaiveDDPBenchmarkConfig) -> dict[str, object]:
    """
    Benchmark Transformer LM training with naïve per-parameter gradient all-reduces.
    """

    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if config.measure_steps <= 0:
        raise ValueError("measure_steps must be positive.")

    skip_reason = _validate_benchmark_config(config)
    dimensions = resolve_model_dimensions(
        model_size=config.model_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )
    result: dict[str, object] = {
        **asdict(config),
        **dimensions,
    }
    if skip_reason is not None:
        return {**result, "status": "skipped", "skip_reason": skip_reason}

    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    with tempfile.NamedTemporaryFile(prefix="naive_ddp_benchmark_", dir="/tmp", delete=False) as handle:
        rendezvous_path = handle.name

    try:
        mp.spawn(
            _naive_ddp_benchmark_worker,
            args=(config, dimensions, rendezvous_path, queue),
            nprocs=config.world_size,
            join=True,
        )
    except Exception as exc:
        return {**result, "status": "error", "error": repr(exc)}
    finally:
        if os.path.exists(rendezvous_path):
            os.unlink(rendezvous_path)

    worker_result = queue.get()
    return {**result, **worker_result}


def format_naive_ddp_check_result(result: dict[str, object]) -> str:
    """
    Format the toy correctness check result as a compact human-readable string.
    """

    if result["status"] != "ok":
        return f"status={result['status']} error={result.get('error', result.get('skip_reason', 'unknown'))}"
    return (
        f"status=ok mode={result.get('execution_mode', 'spawn')} world_size={result['world_size']} device={result['device']} "
        f"steps={result['num_steps']} global_batch_size={result['global_batch_size']} "
        f"max_model_abs_diff={float(result['max_model_abs_diff']):.6e} "
        f"max_optimizer_abs_diff={float(result['max_optimizer_abs_diff']):.6e} "
        f"max_inter_replica_model_abs_diff={float(result.get('max_inter_replica_model_abs_diff', 0.0)):.6e}"
    )


def format_naive_ddp_benchmark_result(result: dict[str, object]) -> str:
    """
    Format the naïve DDP benchmark result as a compact human-readable string.
    """

    if result["status"] != "ok":
        detail = result.get("skip_reason", result.get("error", "unknown"))
        return f"status={result['status']} detail={detail}"
    return (
        f"status=ok model_size={result['model_size']} backend={result['backend']} device={result['device']} "
        f"world_size={result['world_size']} global_batch_size={result['global_batch_size']} "
        f"context_length={result['context_length']} "
        f"mean_step_ms={float(result['mean_step_seconds']) * 1000.0:.3f} "
        f"mean_gradient_comm_ms={float(result['mean_communication_seconds']) * 1000.0:.3f} "
        f"communication_fraction={float(result['communication_fraction']) * 100.0:.2f}%"
    )


def build_naive_ddp_check_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the toy DDP correctness check.
    """

    parser = argparse.ArgumentParser(
        description="Run a toy naïve DDP training loop and compare it against single-process training.",
    )
    parser.add_argument("--world-size", type=int, default=2, help="Number of ranks.")
    parser.add_argument("--backend", default="gloo", help="Distributed backend, e.g. gloo or nccl.")
    parser.add_argument("--device", default="cpu", help="Device specifier, e.g. cpu or cuda.")
    parser.add_argument("--input-dim", type=int, default=8, help="Toy model input width.")
    parser.add_argument("--hidden-dim", type=int, default=16, help="Toy model hidden width.")
    parser.add_argument("--output-dim", type=int, default=4, help="Toy model output width.")
    parser.add_argument("--global-batch-size", type=int, default=8, help="Global batch size.")
    parser.add_argument("--num-steps", type=int, default=4, help="Number of optimizer steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--simulate-locally",
        action="store_true",
        help="Run the correctness check without torch.distributed sockets.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full result dictionary.")
    return parser


def build_naive_ddp_benchmark_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the Transformer naïve DDP benchmark.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark Transformer LM training with naïve per-parameter DDP gradient all-reduces.",
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_SIZE_SPECS.keys()),
        default="xl",
        help="Named model preset.",
    )
    parser.add_argument("--vocab-size", type=int, default=50_257, help="Vocabulary size.")
    parser.add_argument("--global-batch-size", type=int, default=8, help="Global batch size across all ranks.")
    parser.add_argument("--context-length", type=int, default=1024, help="Sequence length.")
    parser.add_argument("--num-layers", type=int, default=None, help="Optional override for layer count.")
    parser.add_argument("--d-model", type=int, default=None, help="Optional override for hidden size.")
    parser.add_argument("--num-heads", type=int, default=None, help="Optional override for attention heads.")
    parser.add_argument("--d-ff", type=int, default=None, help="Optional override for feed-forward width.")
    parser.add_argument("--world-size", type=int, default=2, help="Number of ranks.")
    parser.add_argument("--backend", default="nccl", help="Distributed backend, e.g. nccl or gloo.")
    parser.add_argument("--device", default="cuda", help="Device family, e.g. cuda or cpu.")
    parser.add_argument("--dtype", default="float32", help="Model parameter dtype.")
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=("float16", "fp16", "bfloat16", "bf16"),
        default=None,
        help="Optional autocast dtype applied during forward/loss evaluation.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup training steps.")
    parser.add_argument("--measure-steps", type=int, default=5, help="Measured training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--json", action="store_true", help="Print the full result dictionary.")
    return parser


def naive_ddp_check_config_from_args(args: argparse.Namespace) -> NaiveDDPCheckConfig:
    """
    Convert parsed CLI arguments into a toy DDP check config.
    """

    return NaiveDDPCheckConfig(
        world_size=args.world_size,
        backend=args.backend,
        device=args.device,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        global_batch_size=args.global_batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


def naive_ddp_benchmark_config_from_args(args: argparse.Namespace) -> NaiveDDPBenchmarkConfig:
    """
    Convert parsed CLI arguments into a naïve DDP benchmark config.
    """

    return NaiveDDPBenchmarkConfig(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        global_batch_size=args.global_batch_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        world_size=args.world_size,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        mixed_precision_dtype=args.mixed_precision_dtype,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def _json_safe_result(result: dict[str, object]) -> dict[str, object]:
    """
    Drop large nested tensor payloads before CLI JSON printing.
    """

    blocked_keys = {
        "reference_model_state_dict",
        "ddp_model_state_dict",
        "reference_optimizer_state_dict",
        "ddp_optimizer_state_dict",
        "rank_total_step_seconds",
        "rank_communication_seconds",
    }
    return {key: value for key, value in result.items() if key not in blocked_keys}


__all__ = [
    "NaiveDDPBenchmarkConfig",
    "NaiveDDPCheckConfig",
    "average_gradients",
    "build_naive_ddp_benchmark_parser",
    "build_naive_ddp_check_parser",
    "format_naive_ddp_benchmark_result",
    "format_naive_ddp_check_result",
    "naive_ddp_benchmark_config_from_args",
    "naive_ddp_check_config_from_args",
    "run_naive_ddp_benchmark",
    "run_naive_ddp_check",
    "run_simulated_naive_ddp_reference",
    "run_single_process_reference",
    "shard_batch",
    "_json_safe_result",
]
