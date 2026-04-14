from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

from GPT import ShardedOptimizer, average_gradients_flat, cross_entropy
from GPT.benchmarking import MODEL_SIZE_SPECS, create_random_batch, synchronize_device
from GPT.transformer_lm import TransformerLM


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    """
    Translate a CLI dtype name into a torch dtype.
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


def _setup_process_group(backend: str) -> tuple[int, int, int]:
    """
    Initialize distributed execution from the torchrun environment.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """
    Sum the bytes occupied by tensor-valued optimizer state.
    """
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total


def _parameter_bytes(module: torch.nn.Module) -> int:
    """
    Sum the bytes occupied by model parameters.
    """
    return sum(param.numel() * param.element_size() for param in module.parameters())


def _gradient_bytes(module: torch.nn.Module) -> int:
    """
    Sum the bytes occupied by materialized parameter gradients.
    """
    total = 0
    for param in module.parameters():
        if param.grad is not None:
            total += param.grad.numel() * param.grad.element_size()
    return total


def _memory_snapshot(
    device: torch.device,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    """
    Capture current/peak CUDA memory stats plus a simple component breakdown.
    """
    return {
        "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "parameter_bytes": _parameter_bytes(module),
        "gradient_bytes": _gradient_bytes(module),
        "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
    }


def _build_model_and_optimizer(
    *,
    device: torch.device,
    dtype: torch.dtype,
    optimizer_name: str,
    vocab_size: int,
    context_length: int,
    model_size: str,
    learning_rate: float,
    weight_decay: float,
) -> tuple[TransformerLM, torch.optim.Optimizer]:
    """
    Construct the model and either baseline AdamW or the sharded wrapper.
    """
    dimensions = MODEL_SIZE_SPECS[model_size]
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=dimensions["num_layers"],
        d_model=dimensions["d_model"],
        num_heads=dimensions["num_heads"],
        d_ff=dimensions["d_ff"],
        device=device,
        dtype=dtype,
    )
    if optimizer_name == "baseline":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sharded":
        optimizer = ShardedOptimizer(
            model.parameters(),
            torch.optim.AdamW,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer_name {optimizer_name!r}.")
    return model, optimizer


def _run_case(
    *,
    optimizer_name: str,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    context_length: int,
    model_size: str,
    global_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    measure_steps: int,
    mixed_precision_dtype: torch.dtype | None,
    seed: int,
) -> dict[str, object]:
    """
    Run one optimizer configuration and return memory and timing results.
    """
    if global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size.")

    torch.manual_seed(seed)
    model, optimizer = _build_model_and_optimizer(
        device=device,
        dtype=dtype,
        optimizer_name=optimizer_name,
        vocab_size=vocab_size,
        context_length=context_length,
        model_size=model_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    local_batch_size = global_batch_size // world_size
    def autocast_context():
        return (
            torch.autocast(device_type=device.type, dtype=mixed_precision_dtype)
            if mixed_precision_dtype is not None
            else nullcontext()
        )

    synchronize_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    after_init = _memory_snapshot(device, model, optimizer)

    optimizer.zero_grad(set_to_none=True)
    inputs, targets = create_random_batch(
        batch_size=local_batch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        device=device,
    )
    with autocast_context():
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
    loss.backward()
    average_gradients_flat(model)
    synchronize_device(device)
    before_step = _memory_snapshot(device, model, optimizer)

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.step()
    synchronize_device(device)
    after_step = _memory_snapshot(device, model, optimizer)

    timings_ms: list[float] = []
    for step_idx in range(warmup_steps + measure_steps):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = create_random_batch(
            batch_size=local_batch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            device=device,
        )
        synchronize_device(device)
        start_time = time.perf_counter()
        with autocast_context():
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
        loss.backward()
        average_gradients_flat(model)
        optimizer.step()
        synchronize_device(device)
        end_time = time.perf_counter()
        if step_idx >= warmup_steps:
            timings_ms.append((end_time - start_time) * 1000.0)

    local_result = {
        "optimizer": optimizer_name,
        "rank": rank,
        "after_init": after_init,
        "before_step": before_step,
        "after_step": after_step,
        "timings_ms": timings_ms,
    }

    gathered_results: list[dict[str, object] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_result)
    assert all(result is not None for result in gathered_results)

    rank0_result = next(result for result in gathered_results if result["rank"] == 0)
    per_rank_peak_after_step = {
        str(result["rank"]): result["after_step"]["peak_allocated_bytes"]
        for result in gathered_results
    }

    return {
        "optimizer": optimizer_name,
        "world_size": world_size,
        "rank0_memory": {
            "after_init": rank0_result["after_init"],
            "before_step": rank0_result["before_step"],
            "after_step": rank0_result["after_step"],
        },
        "rank0_mean_iteration_ms": sum(rank0_result["timings_ms"]) / len(rank0_result["timings_ms"]),
        "rank0_timings_ms": rank0_result["timings_ms"],
        "per_rank_peak_after_step_allocated_bytes": per_rank_peak_after_step,
    }


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for optimizer-state sharding profiling.
    """
    parser = argparse.ArgumentParser(
        description="Profile memory and throughput for baseline vs sharded AdamW.",
    )
    parser.add_argument("--model-size", default="xl", choices=tuple(MODEL_SIZE_SPECS))
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--mixed-precision-dtype", default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        default="both",
        choices=("baseline", "sharded", "both"),
        help="Which optimizer configuration to profile.",
    )
    return parser


def main() -> None:
    """
    Profile the requested optimizer configurations and print JSON results.
    """
    parser = build_parser()
    args = parser.parse_args()
    rank, world_size, local_rank = _setup_process_group(args.backend)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This profiling script requires CUDA.")

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dtype = _torch_dtype_from_name(args.dtype)
        mixed_precision_dtype = (
            _torch_dtype_from_name(args.mixed_precision_dtype)
            if args.mixed_precision_dtype is not None
            else None
        )

        optimizer_names = ("baseline", "sharded") if args.optimizer == "both" else (args.optimizer,)
        results = [
            _run_case(
                optimizer_name=optimizer_name,
                rank=rank,
                world_size=world_size,
                device=device,
                dtype=dtype,
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                model_size=args.model_size,
                global_batch_size=args.global_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                mixed_precision_dtype=mixed_precision_dtype,
                seed=args.seed,
            )
            for optimizer_name in optimizer_names
        ]

        if rank == 0:
            print(json.dumps(results, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
