from __future__ import annotations

import argparse
import json
import statistics
import timeit
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from .attention import scaled_dot_product_attention
from .training import resolve_training_device


DEFAULT_HEAD_DIMS = (16, 32, 64, 128)
DEFAULT_SEQUENCE_LENGTHS = (256, 1024, 4096, 8192, 16384)
ATTENTION_BENCHMARK_VARIANTS = ("eager", "compiled")


@dataclass(slots=True)
class AttentionBenchmarkConfig:
    batch_size: int = 8
    head_dims: tuple[int, ...] = DEFAULT_HEAD_DIMS
    sequence_lengths: tuple[int, ...] = DEFAULT_SEQUENCE_LENGTHS
    variants: tuple[str, ...] = ("eager",)
    warmup_steps: int = 10
    iterations: int = 100
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 0
    compile_backend: str | None = None
    compile_mode: str | None = None


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
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


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_variant_csv(raw: str) -> tuple[str, ...]:
    variants = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not variants:
        raise ValueError("Expected at least one execution variant.")
    invalid = tuple(variant for variant in variants if variant not in ATTENTION_BENCHMARK_VARIANTS)
    if invalid:
        raise ValueError(
            f"Unsupported execution variant(s): {', '.join(invalid)}. "
            f"Expected {', '.join(ATTENTION_BENCHMARK_VARIANTS)}."
        )
    return variants


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def is_out_of_memory_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def estimate_attention_tensor_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
) -> dict[str, int]:
    """
    Estimate the dominant tensor sizes for naïve single-head attention.

    The `scores` and `weights` tensors both have shape `(B, T, T)` and are the
    quadratic allocations that dominate memory growth. They are reported
    separately and together as a simple lower-bound for the tensors explicitly
    materialized by the implementation.
    """
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    qkv_elements = 3 * batch_size * sequence_length * head_dim
    output_elements = batch_size * sequence_length * head_dim
    quadratic_elements = batch_size * sequence_length * sequence_length
    return {
        "qkv_bytes": qkv_elements * bytes_per_element,
        "output_bytes": output_elements * bytes_per_element,
        "attention_scores_bytes": quadratic_elements * bytes_per_element,
        "attention_weights_bytes": quadratic_elements * bytes_per_element,
        "quadratic_saved_tensors_bytes": 2 * quadratic_elements * bytes_per_element,
    }


def _mib_or_none(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024**2)


def compile_attention_fn(
    attention_fn: Callable[..., torch.Tensor],
    *,
    backend: str | None,
    mode: str | None,
) -> Callable[..., torch.Tensor]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    compile_kwargs: dict[str, object] = {}
    if backend is not None:
        compile_kwargs["backend"] = backend
    if mode is not None:
        compile_kwargs["mode"] = mode
    return torch.compile(attention_fn, **compile_kwargs)


def benchmark_attention_case(
    *,
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    variant: str = "eager",
    warmup_steps: int,
    iterations: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    theoretical_bytes = estimate_attention_tensor_bytes(
        batch_size=batch_size,
        sequence_length=sequence_length,
        head_dim=head_dim,
        dtype=dtype,
    )
    result: dict[str, object] = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "variant": variant,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "status": "ok",
        "oom_stage": None,
        **theoretical_bytes,
        "qkv_mib": _mib_or_none(theoretical_bytes["qkv_bytes"]),
        "output_mib": _mib_or_none(theoretical_bytes["output_bytes"]),
        "attention_scores_mib": _mib_or_none(theoretical_bytes["attention_scores_bytes"]),
        "attention_weights_mib": _mib_or_none(theoretical_bytes["attention_weights_bytes"]),
        "quadratic_saved_tensors_mib": _mib_or_none(theoretical_bytes["quadratic_saved_tensors_bytes"]),
    }

    if variant not in ATTENTION_BENCHMARK_VARIANTS:
        raise ValueError(
            f"Unsupported execution variant '{variant}'. Expected one of {', '.join(ATTENTION_BENCHMARK_VARIANTS)}."
        )

    attention_fn = scaled_dot_product_attention
    if variant == "compiled":
        attention_fn = compile_attention_fn(
            scaled_dot_product_attention,
            backend=compile_backend,
            mode=compile_mode,
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        q_forward = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=dtype)
        k_forward = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=dtype)
        v_forward = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=dtype)
    except RuntimeError as exc:
        if not is_out_of_memory_error(exc):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result["status"] = "oom"
        result["oom_stage"] = "setup"
        return result

    forward_times: list[float] = []
    try:
        with torch.no_grad():
            for _ in range(warmup_steps):
                attention_fn(q=q_forward, k=k_forward, v=v_forward)
                synchronize_device(device)

            for _ in range(iterations):
                synchronize_device(device)
                start_time = timeit.default_timer()
                attention_fn(q=q_forward, k=k_forward, v=v_forward)
                synchronize_device(device)
                end_time = timeit.default_timer()
                forward_times.append(end_time - start_time)
    except RuntimeError as exc:
        if not is_out_of_memory_error(exc):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result["status"] = "oom"
        result["oom_stage"] = "forward"
        return result

    try:
        q_backward = torch.randn(
            batch_size,
            sequence_length,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        k_backward = torch.randn(
            batch_size,
            sequence_length,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        v_backward = torch.randn(
            batch_size,
            sequence_length,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
    except RuntimeError as exc:
        if not is_out_of_memory_error(exc):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result["status"] = "oom"
        result["oom_stage"] = "setup"
        return result

    backward_times: list[float] = []
    pre_backward_allocated_bytes: list[int] = []
    try:
        for _ in range(warmup_steps):
            out = attention_fn(q=q_backward, k=k_backward, v=v_backward)
            loss = out.sum()
            synchronize_device(device)
            loss.backward()
            synchronize_device(device)
            q_backward.grad = None
            k_backward.grad = None
            v_backward.grad = None

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(iterations):
            out = attention_fn(q=q_backward, k=k_backward, v=v_backward)
            loss = out.sum()
            synchronize_device(device)
            if device.type == "cuda":
                pre_backward_allocated_bytes.append(int(torch.cuda.memory_allocated(device)))
            start_time = timeit.default_timer()
            loss.backward()
            synchronize_device(device)
            end_time = timeit.default_timer()
            backward_times.append(end_time - start_time)
            q_backward.grad = None
            k_backward.grad = None
            v_backward.grad = None
    except RuntimeError as exc:
        if not is_out_of_memory_error(exc):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result["status"] = "oom"
        result["oom_stage"] = "backward"
        return result

    result["forward_mean_ms"] = statistics.fmean(forward_times) * 1000.0
    result["forward_std_ms"] = (statistics.stdev(forward_times) * 1000.0) if len(forward_times) > 1 else 0.0
    result["backward_mean_ms"] = statistics.fmean(backward_times) * 1000.0
    result["backward_std_ms"] = (
        statistics.stdev(backward_times) * 1000.0 if len(backward_times) > 1 else 0.0
    )
    result["pre_backward_allocated_bytes"] = (
        max(pre_backward_allocated_bytes) if pre_backward_allocated_bytes else None
    )
    result["pre_backward_allocated_mib"] = _mib_or_none(result["pre_backward_allocated_bytes"])
    result["peak_allocated_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    result["peak_allocated_mib"] = _mib_or_none(result["peak_allocated_bytes"])
    return result


def run_attention_benchmark(config: AttentionBenchmarkConfig) -> list[dict[str, object]]:
    torch.manual_seed(config.seed)
    device = resolve_training_device(config.device)
    dtype = _torch_dtype_from_name(config.dtype)

    results: list[dict[str, object]] = []
    for variant in config.variants:
        for head_dim in config.head_dims:
            for sequence_length in config.sequence_lengths:
                row = benchmark_attention_case(
                    batch_size=config.batch_size,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    variant=variant,
                    warmup_steps=config.warmup_steps,
                    iterations=config.iterations,
                    device=device,
                    dtype=dtype,
                    compile_backend=config.compile_backend,
                    compile_mode=config.compile_mode,
                )
                results.append(row)
    return results


def _format_optional_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def format_attention_benchmark_table(results: list[dict[str, object]]) -> str:
    header = (
        "| variant | d_model | seq_len | status | oom_stage | forward_ms | backward_ms | "
        "pre_backward_mib | quadratic_saved_mib | peak_allocated_mib |"
    )
    separator = (
        "| :--- | ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [header, separator]
    for result in results:
        rows.append(
            "| "
            f"{result['variant']} | "
            f"{result['head_dim']} | "
            f"{result['sequence_length']} | "
            f"{result['status']} | "
            f"{result['oom_stage'] or '-'} | "
            f"{_format_optional_float(result.get('forward_mean_ms'))} | "
            f"{_format_optional_float(result.get('backward_mean_ms'))} | "
            f"{_format_optional_float(result.get('pre_backward_allocated_mib'))} | "
            f"{_format_optional_float(result.get('quadratic_saved_tensors_mib'))} | "
            f"{_format_optional_float(result.get('peak_allocated_mib'))} |"
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the assignment scaled dot-product attention over a grid of sequence lengths and head sizes.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size. The assignment uses 8.")
    parser.add_argument(
        "--head-dims",
        default="16,32,64,128",
        help="Comma-separated head dimensions to benchmark.",
    )
    parser.add_argument(
        "--sequence-lengths",
        default="256,1024,4096,8192,16384",
        help="Comma-separated sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--variants",
        default="eager",
        help="Comma-separated execution variants: eager and/or compiled.",
    )
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup iterations before timing.")
    parser.add_argument("--iterations", type=int, default=100, help="Measured forward/backward iterations.")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, mps, cuda, or cuda:N.",
    )
    parser.add_argument("--dtype", default="float32", help="Input dtype.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--compile-backend",
        default=None,
        help="Optional backend forwarded to torch.compile when the compiled variant is enabled.",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="Optional mode forwarded to torch.compile when the compiled variant is enabled.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a markdown table.")
    return parser


def config_from_args(args: argparse.Namespace) -> AttentionBenchmarkConfig:
    return AttentionBenchmarkConfig(
        batch_size=args.batch_size,
        head_dims=_parse_int_csv(args.head_dims),
        sequence_lengths=_parse_int_csv(args.sequence_lengths),
        variants=_parse_variant_csv(args.variants),
        warmup_steps=args.warmup_steps,
        iterations=args.iterations,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    results = run_attention_benchmark(config)
    payload = {
        **asdict(config),
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(format_attention_benchmark_table(results))
