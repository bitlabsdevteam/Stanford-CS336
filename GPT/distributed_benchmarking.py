from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from statistics import fmean

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DEFAULT_BACKEND_DEVICE_PAIRS = (("gloo", "cpu"), ("nccl", "cuda"))
DEFAULT_DATA_SIZES_MB = (1, 10, 100, 1024)
DEFAULT_PROCESS_COUNTS = (2, 4, 6)


@dataclass(slots=True)
class DistributedAllReduceBenchmarkConfig:
    backend_device_pairs: tuple[tuple[str, str], ...] = DEFAULT_BACKEND_DEVICE_PAIRS
    data_sizes_mb: tuple[int, ...] = DEFAULT_DATA_SIZES_MB
    process_counts: tuple[int, ...] = DEFAULT_PROCESS_COUNTS
    warmup_iterations: int = 5
    measure_iterations: int = 20
    seed: int = 0


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_backend_device_csv(raw: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        backend, device = (part.strip().lower() for part in cleaned.split("+", maxsplit=1))
        if backend not in {"gloo", "nccl"}:
            raise ValueError(f"Unsupported backend '{backend}'.")
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device '{device}'.")
        pairs.append((backend, device))
    if not pairs:
        raise ValueError("Expected at least one backend+device pair.")
    return tuple(pairs)


def _choose_master_port(seed: int, world_size: int, size_mb: int) -> int:
    """
    Choose a likely-free local port without probing the OS first.
    """
    base_port = 29500
    port_span = 2000
    return base_port + ((seed + 31 * world_size + 17 * size_mb + os.getpid()) % port_span)


def _bytes_to_num_float32_elements(size_bytes: int) -> int:
    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive.")
    element_size = torch.tensor([], dtype=torch.float32).element_size()
    if size_bytes % element_size != 0:
        raise ValueError("size_bytes must be divisible by the float32 element size.")
    return size_bytes // element_size


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _setup_process_group(
    *,
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: int,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _teardown_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _benchmark_all_reduce_worker(
    rank: int,
    world_size: int,
    backend: str,
    device_type: str,
    size_bytes: int,
    warmup_iterations: int,
    measure_iterations: int,
    master_addr: str,
    master_port: int,
    seed: int,
    queue,
) -> None:
    try:
        _setup_process_group(
            rank=rank,
            world_size=world_size,
            backend=backend,
            master_addr=master_addr,
            master_port=master_port,
        )

        if device_type == "cuda":
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        torch.manual_seed(seed + rank)
        numel = _bytes_to_num_float32_elements(size_bytes)
        data = torch.randn(numel, device=device, dtype=torch.float32)

        for _ in range(warmup_iterations):
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            _synchronize_device(device)
        dist.barrier()
        _synchronize_device(device)

        timings_ms: list[float] = []
        for _ in range(measure_iterations):
            data.copy_(torch.randn_like(data))
            dist.barrier()
            _synchronize_device(device)
            start_time = time.perf_counter()
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            _synchronize_device(device)
            end_time = time.perf_counter()
            timings_ms.append((end_time - start_time) * 1000.0)

        gathered_timings: list[list[float] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_timings, timings_ms)

        if rank == 0:
            flat_timings = [timing for rank_timings in gathered_timings for timing in (rank_timings or [])]
            queue.put(
                {
                    "status": "ok",
                    "rank_timings_ms": gathered_timings,
                    "mean_latency_ms": fmean(flat_timings),
                    "min_latency_ms": min(flat_timings),
                    "max_latency_ms": max(flat_timings),
                    "per_rank_mean_latency_ms": [fmean(rank_timings or [0.0]) for rank_timings in gathered_timings],
                }
            )
    except Exception as exc:
        if rank == 0:
            queue.put({"status": "error", "error": repr(exc)})
        raise
    finally:
        _teardown_process_group()


def _validate_backend_device_pair(
    *,
    backend: str,
    device_type: str,
    world_size: int,
) -> str | None:
    if backend == "gloo" and device_type != "cpu":
        return "gloo benchmark is restricted to CPU tensors in this script."
    if backend == "nccl" and device_type != "cuda":
        return "nccl benchmark requires CUDA tensors."
    if device_type == "cuda":
        if not torch.cuda.is_available():
            return "CUDA is not available."
        if torch.cuda.device_count() < world_size:
            return (
                f"Requested {world_size} CUDA ranks, but only {torch.cuda.device_count()} GPU(s) are available."
            )
    return None


def benchmark_all_reduce_case(
    *,
    backend: str,
    device_type: str,
    world_size: int,
    size_mb: int,
    warmup_iterations: int,
    measure_iterations: int,
    seed: int,
) -> dict[str, object]:
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if size_mb <= 0:
        raise ValueError("size_mb must be positive.")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative.")
    if measure_iterations <= 0:
        raise ValueError("measure_iterations must be positive.")

    size_bytes = size_mb * 1024 * 1024
    skip_reason = _validate_backend_device_pair(
        backend=backend,
        device_type=device_type,
        world_size=world_size,
    )
    result: dict[str, object] = {
        "backend": backend,
        "device_type": device_type,
        "world_size": world_size,
        "size_mb": size_mb,
        "size_bytes": size_bytes,
        "dtype": "float32",
        "warmup_iterations": warmup_iterations,
        "measure_iterations": measure_iterations,
    }
    if skip_reason is not None:
        return {**result, "status": "skipped", "skip_reason": skip_reason}

    master_addr = "127.0.0.1"
    master_port = _choose_master_port(seed=seed, world_size=world_size, size_mb=size_mb)
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()

    try:
        mp.spawn(
            _benchmark_all_reduce_worker,
            args=(
                world_size,
                backend,
                device_type,
                size_bytes,
                warmup_iterations,
                measure_iterations,
                master_addr,
                master_port,
                seed,
                queue,
            ),
            nprocs=world_size,
            join=True,
        )
    except Exception as exc:
        return {**result, "status": "error", "error": repr(exc)}

    worker_result = queue.get()
    return {**result, **worker_result}


def run_all_reduce_benchmark(
    config: DistributedAllReduceBenchmarkConfig,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for backend, device_type in config.backend_device_pairs:
        for world_size in config.process_counts:
            for size_mb in config.data_sizes_mb:
                result = benchmark_all_reduce_case(
                    backend=backend,
                    device_type=device_type,
                    world_size=world_size,
                    size_mb=size_mb,
                    warmup_iterations=config.warmup_iterations,
                    measure_iterations=config.measure_iterations,
                    seed=config.seed,
                )
                results.append(result)
    return results


def format_all_reduce_benchmark_table(results: list[dict[str, object]]) -> str:
    headers = (
        "backend",
        "device_type",
        "world_size",
        "size_mb",
        "status",
        "mean_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "skip_reason",
    )
    lines = [" | ".join(headers), " | ".join("-" * len(header) for header in headers)]
    for row in results:
        values: list[str] = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append(" | ".join(values))
    return "\n".join(lines)


def summarize_all_reduce_results(results: list[dict[str, object]]) -> str:
    successful = [row for row in results if row.get("status") == "ok"]
    if not successful:
        return "No successful benchmark runs were recorded."

    fastest = min(successful, key=lambda row: float(row["mean_latency_ms"]))
    slowest = max(successful, key=lambda row: float(row["mean_latency_ms"]))
    return (
        "Latency increases with larger tensors and usually with more ranks because the collective moves more "
        "data and coordination work. GPU NCCL runs should outperform CPU Gloo at larger sizes when enough GPUs "
        "are available, while small transfers can be dominated by launch and synchronization overhead.\n"
        f"Fastest successful case: {fastest['backend']}+{fastest['device_type']} with world_size={fastest['world_size']} "
        f"and size_mb={fastest['size_mb']} at {float(fastest['mean_latency_ms']):.3f} ms.\n"
        f"Slowest successful case: {slowest['backend']}+{slowest['device_type']} with world_size={slowest['world_size']} "
        f"and size_mb={slowest['size_mb']} at {float(slowest['mean_latency_ms']):.3f} ms."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark single-node multi-process all-reduce across CPU Gloo and CUDA NCCL settings.",
    )
    parser.add_argument(
        "--backend-device-pairs",
        default="gloo+cpu,nccl+cuda",
        help="Comma-separated backend+device pairs such as gloo+cpu,nccl+cuda.",
    )
    parser.add_argument(
        "--data-sizes-mb",
        default="1,10,100,1024",
        help="Comma-separated float32 tensor sizes in megabytes.",
    )
    parser.add_argument(
        "--process-counts",
        default="2,4,6",
        help="Comma-separated world sizes to benchmark.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=5, help="Number of warmup all-reduce calls.")
    parser.add_argument(
        "--measure-iterations",
        type=int,
        default=20,
        help="Number of measured all-reduce calls per benchmark case.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = DistributedAllReduceBenchmarkConfig(
        backend_device_pairs=_parse_backend_device_csv(args.backend_device_pairs),
        data_sizes_mb=_parse_int_csv(args.data_sizes_mb),
        process_counts=_parse_int_csv(args.process_counts),
        warmup_iterations=args.warmup_iterations,
        measure_iterations=args.measure_iterations,
        seed=args.seed,
    )
    results = run_all_reduce_benchmark(config)
    print(format_all_reduce_benchmark_table(results))
    print()
    print(summarize_all_reduce_results(results))


__all__ = [
    "DistributedAllReduceBenchmarkConfig",
    "benchmark_all_reduce_case",
    "format_all_reduce_benchmark_table",
    "run_all_reduce_benchmark",
    "summarize_all_reduce_results",
    "main",
]
