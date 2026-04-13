from __future__ import annotations

import argparse
import json
import statistics
import timeit
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import torch

from .cross_entropy import cross_entropy
from .training import resolve_training_device
from .transformer_lm import TransformerLM


MODEL_SIZE_SPECS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

BENCHMARK_MODES = ("forward", "forward-backward")


@dataclass(slots=True)
class BenchmarkConfig:
    vocab_size: int = 10_000
    batch_size: int = 4
    context_length: int = 128
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    warmup_steps: int = 5
    measure_steps: int = 10
    device: str = "auto"
    dtype: str = "float32"
    mode: str = "forward-backward"
    seed: int = 0


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


def resolve_model_dimensions(
    *,
    model_size: str | None,
    d_model: int | None,
    d_ff: int | None,
    num_layers: int | None,
    num_heads: int | None,
) -> dict[str, int]:
    if model_size is None:
        if None in (d_model, d_ff, num_layers, num_heads):
            raise ValueError(
                "Provide either --model-size or all of --d-model, --d-ff, --num-layers, and --num-heads."
            )
        return {
            "d_model": int(d_model),
            "d_ff": int(d_ff),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
        }

    if model_size not in MODEL_SIZE_SPECS:
        raise ValueError(f"Unknown model size '{model_size}'.")

    dimensions = dict(MODEL_SIZE_SPECS[model_size])
    if d_model is not None:
        dimensions["d_model"] = int(d_model)
    if d_ff is not None:
        dimensions["d_ff"] = int(d_ff)
    if num_layers is not None:
        dimensions["num_layers"] = int(num_layers)
    if num_heads is not None:
        dimensions["num_heads"] = int(num_heads)
    return dimensions


def create_random_batch(
    *,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    return inputs, targets


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def benchmark_steps(
    *,
    step_fn: Callable[[], None],
    warmup_steps: int,
    measure_steps: int,
    synchronize_fn: Callable[[], None],
    timer_fn: Callable[[], float] = timeit.default_timer,
) -> list[float]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if measure_steps <= 0:
        raise ValueError("measure_steps must be positive.")

    for _ in range(warmup_steps):
        step_fn()
        synchronize_fn()

    timings: list[float] = []
    for _ in range(measure_steps):
        synchronize_fn()
        start_time = timer_fn()
        step_fn()
        synchronize_fn()
        end_time = timer_fn()
        timings.append(end_time - start_time)
    return timings


def summarize_timings(timings: Sequence[float]) -> dict[str, float | int]:
    if len(timings) == 0:
        raise ValueError("timings must not be empty.")

    mean_seconds = statistics.fmean(timings)
    std_seconds = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return {
        "mean_seconds": mean_seconds,
        "std_seconds": std_seconds,
        "num_measurements": len(timings),
    }


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    if config.mode not in BENCHMARK_MODES:
        raise ValueError(
            f"Unsupported mode '{config.mode}'. Expected one of {', '.join(BENCHMARK_MODES)}."
        )

    torch.manual_seed(config.seed)
    device = resolve_training_device(config.device)
    dtype = _torch_dtype_from_name(config.dtype)

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        device=device,
        dtype=dtype,
    )
    inputs, targets = create_random_batch(
        batch_size=config.batch_size,
        context_length=config.context_length,
        vocab_size=config.vocab_size,
        device=device,
    )

    if config.mode == "forward":
        model.eval()

        def step() -> None:
            with torch.no_grad():
                model(inputs)

    else:
        model.train()

        def step() -> None:
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()

    timings = benchmark_steps(
        step_fn=step,
        warmup_steps=config.warmup_steps,
        measure_steps=config.measure_steps,
        synchronize_fn=lambda: synchronize_device(device),
    )
    summary = summarize_timings(timings)
    return {
        **asdict(config),
        "device": str(device),
        "timings_seconds": timings,
        **summary,
    }


def format_benchmark_result(result: dict[str, object]) -> str:
    mean_ms = float(result["mean_seconds"]) * 1000.0
    std_ms = float(result["std_seconds"]) * 1000.0
    return (
        f"mode={result['mode']} device={result['device']} dtype={result['dtype']} "
        f"layers={result['num_layers']} d_model={result['d_model']} heads={result['num_heads']} "
        f"d_ff={result['d_ff']} batch_size={result['batch_size']} context_length={result['context_length']} "
        f"mean_ms={mean_ms:.3f} std_ms={std_ms:.3f} n={result['num_measurements']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark forward or forward-backward passes for the assignment GPT model.",
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_SIZE_SPECS.keys()),
        default="small",
        help="Named assignment model preset.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Vocabulary size.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--context-length", type=int, default=128, help="Sequence length.")
    parser.add_argument("--num-layers", type=int, default=None, help="Optional override for layer count.")
    parser.add_argument("--d-model", type=int, default=None, help="Optional override for hidden size.")
    parser.add_argument("--num-heads", type=int, default=None, help="Optional override for attention heads.")
    parser.add_argument("--d-ff", type=int, default=None, help="Optional override for feed-forward hidden size.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--measure-steps", type=int, default=10, help="Measured iterations.")
    parser.add_argument(
        "--mode",
        choices=BENCHMARK_MODES,
        default="forward-backward",
        help="Benchmark a pure forward pass or a training-style forward plus backward pass.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, mps, cuda, or cuda:0.",
    )
    parser.add_argument("--dtype", default="float32", help="Model parameter dtype.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a compact text line.")
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    dimensions = resolve_model_dimensions(
        model_size=args.model_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    return BenchmarkConfig(
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        num_layers=dimensions["num_layers"],
        d_model=dimensions["d_model"],
        num_heads=dimensions["num_heads"],
        d_ff=dimensions["d_ff"],
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        device=args.device,
        dtype=args.dtype,
        mode=args.mode,
        seed=args.seed,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_benchmark(config_from_args(args))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(format_benchmark_result(result))
