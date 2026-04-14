from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from .attention import scaled_dot_product_attention
from .flash_attention import FlashAttentionForwardAutogradFunctionTriton
from .training import resolve_training_device

try:
    import triton
except ModuleNotFoundError:
    triton = None


DEFAULT_SEQUENCE_LENGTHS = tuple(2**power for power in range(7, 17))
DEFAULT_HEAD_DIMS = tuple(2**power for power in range(4, 8))
DEFAULT_DTYPES = ("bfloat16", "float32")


@dataclass(slots=True)
class FlashAttentionBenchmarkConfig:
    sequence_lengths: tuple[int, ...] = DEFAULT_SEQUENCE_LENGTHS
    head_dims: tuple[int, ...] = DEFAULT_HEAD_DIMS
    dtypes: tuple[str, ...] = DEFAULT_DTYPES
    batch_size: int = 1
    warmup_ms: int = 100
    rep_ms: int = 300
    device: str = "cuda"
    seed: int = 0


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_str_csv(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return dtype_map[normalized]


def _causal_mask(n_queries: int, n_keys: int, device: torch.device) -> torch.Tensor:
    query_positions = torch.arange(n_queries, device=device).unsqueeze(-1)
    key_positions = torch.arange(n_keys, device=device).unsqueeze(0)
    return query_positions >= key_positions


def _naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    mask = _causal_mask(q.shape[-2], k.shape[-2], q.device)
    return scaled_dot_product_attention(q=q, k=k, v=v, mask=mask)


def _flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return FlashAttentionForwardAutogradFunctionTriton.apply(q, k, v, True)


def _reset_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad = None


def _bench_forward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    def step():
        with torch.no_grad():
            attention_fn(q, k, v)

    return float(triton.testing.do_bench(step, warmup=warmup_ms, rep=rep_ms))


def _bench_backward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_o: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    out = attention_fn(q, k, v)

    def step():
        _reset_grads(q, k, v)
        out.backward(d_o, retain_graph=True)

    return float(triton.testing.do_bench(step, warmup=warmup_ms, rep=rep_ms))


def _bench_end_to_end(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_o: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    def step():
        _reset_grads(q, k, v)
        out = attention_fn(q, k, v)
        out.backward(d_o)

    return float(triton.testing.do_bench(step, warmup=warmup_ms, rep=rep_ms))


def _oom_result(
    *,
    implementation: str,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    stage: str,
) -> dict[str, object]:
    return {
        "implementation": implementation,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "status": f"oom:{stage}",
        "forward_ms": None,
        "backward_ms": None,
        "end_to_end_ms": None,
    }


def benchmark_flash_attention_case(
    *,
    attention_fn,
    implementation: str,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_ms: int,
    rep_ms: int,
    batch_size: int,
) -> dict[str, object]:
    try:
        q_forward = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=dtype)
        k_forward = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=dtype)
        v_forward = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=dtype)
        forward_ms = _bench_forward(
            attention_fn,
            q_forward,
            k_forward,
            v_forward,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return _oom_result(
            implementation=implementation,
            sequence_length=sequence_length,
            head_dim=head_dim,
            dtype=dtype,
            stage="forward",
        )

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
        d_o = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=dtype)
        backward_ms = _bench_backward(
            attention_fn,
            q_backward,
            k_backward,
            v_backward,
            d_o,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        end_to_end_ms = _bench_end_to_end(
            attention_fn,
            q_backward,
            k_backward,
            v_backward,
            d_o,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return _oom_result(
            implementation=implementation,
            sequence_length=sequence_length,
            head_dim=head_dim,
            dtype=dtype,
            stage="backward",
        )

    return {
        "implementation": implementation,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "status": "ok",
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "end_to_end_ms": end_to_end_ms,
    }


def run_flash_attention_benchmark(
    config: FlashAttentionBenchmarkConfig,
) -> list[dict[str, object]]:
    if triton is None:
        raise RuntimeError("Triton is required to run FlashAttention benchmarking.")

    torch.manual_seed(config.seed)
    device = resolve_training_device(config.device)
    if device.type != "cuda":
        raise ValueError("FlashAttention benchmarking requires a CUDA device.")
    if config.batch_size != 1:
        raise ValueError("This benchmark is intended for batch size 1.")

    results: list[dict[str, object]] = []
    implementations = (
        ("pytorch", _naive_attention),
        ("flash", _flash_attention),
    )
    for dtype_name in config.dtypes:
        dtype = _torch_dtype_from_name(dtype_name)
        for sequence_length in config.sequence_lengths:
            for head_dim in config.head_dims:
                for implementation, attention_fn in implementations:
                    torch.cuda.empty_cache()
                    result = benchmark_flash_attention_case(
                        attention_fn=attention_fn,
                        implementation=implementation,
                        sequence_length=sequence_length,
                        head_dim=head_dim,
                        dtype=dtype,
                        device=device,
                        warmup_ms=config.warmup_ms,
                        rep_ms=config.rep_ms,
                        batch_size=config.batch_size,
                    )
                    results.append(result)
    return results


def format_flash_attention_benchmark_table(results: list[dict[str, object]]) -> str:
    headers = (
        "implementation",
        "sequence_length",
        "head_dim",
        "dtype",
        "status",
        "forward_ms",
        "backward_ms",
        "end_to_end_ms",
    )
    lines = [" | ".join(headers), " | ".join("-" * len(header) for header in headers)]
    for row in results:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append(" | ".join(values))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark FlashAttention against naïve PyTorch attention with triton.testing.do_bench.",
    )
    parser.add_argument(
        "--sequence-lengths",
        default=",".join(str(value) for value in DEFAULT_SEQUENCE_LENGTHS),
        help="Comma-separated sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--head-dims",
        default=",".join(str(value) for value in DEFAULT_HEAD_DIMS),
        help="Comma-separated head dimensions to benchmark.",
    )
    parser.add_argument(
        "--dtypes",
        default=",".join(DEFAULT_DTYPES),
        help="Comma-separated dtypes to benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Expected to stay at 1.")
    parser.add_argument("--warmup-ms", type=int, default=100, help="Warmup duration in milliseconds.")
    parser.add_argument("--rep-ms", type=int, default=300, help="Measurement duration in milliseconds.")
    parser.add_argument("--device", default="cuda", help="Benchmark device, typically cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = FlashAttentionBenchmarkConfig(
        sequence_lengths=_parse_int_csv(args.sequence_lengths),
        head_dims=_parse_int_csv(args.head_dims),
        dtypes=_parse_str_csv(args.dtypes),
        batch_size=args.batch_size,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        device=args.device,
        seed=args.seed,
    )
    results = run_flash_attention_benchmark(config)
    print(format_flash_attention_benchmark_table(results))


__all__ = [
    "FlashAttentionBenchmarkConfig",
    "benchmark_flash_attention_case",
    "format_flash_attention_benchmark_table",
    "run_flash_attention_benchmark",
    "main",
]
