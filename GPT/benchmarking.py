from __future__ import annotations

import argparse
import json
import statistics
import timeit
from contextlib import nullcontext
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import torch

from .cross_entropy import cross_entropy
from . import nvtx
from .optimization import AdamW
from .training import resolve_training_device
from .transformer_lm import TransformerLM


EXECUTION_VARIANTS = ("eager", "compiled")
MODEL_SIZE_SPECS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

BENCHMARK_MODES = ("forward", "forward-backward", "train-step")


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
    mixed_precision_dtype: str | None = None
    mode: str = "forward-backward"
    seed: int = 0
    memory_profile_path: str | None = None
    memory_profile_max_entries: int = 1_000_000
    compile_model: bool = False
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


def _resolve_mixed_precision_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    return _torch_dtype_from_name(dtype_name)


def _parse_variant_csv(raw: str) -> tuple[str, ...]:
    variants = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not variants:
        raise ValueError("Expected at least one execution variant.")
    invalid = tuple(variant for variant in variants if variant not in EXECUTION_VARIANTS)
    if invalid:
        raise ValueError(
            f"Unsupported execution variant(s): {', '.join(invalid)}. Expected {', '.join(EXECUTION_VARIANTS)}."
        )
    return variants


def compile_module(
    module: torch.nn.Module,
    *,
    backend: str | None,
    mode: str | None,
) -> torch.nn.Module:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    compile_kwargs: dict[str, object] = {}
    if backend is not None:
        compile_kwargs["backend"] = backend
    if mode is not None:
        compile_kwargs["mode"] = mode
    return torch.compile(module, **compile_kwargs)


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
    after_warmup_fn: Callable[[], None] | None = None,
    after_measure_fn: Callable[[], None] | None = None,
) -> list[float]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if measure_steps <= 0:
        raise ValueError("measure_steps must be positive.")

    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            step_fn()
            synchronize_fn()

    if after_warmup_fn is not None:
        after_warmup_fn()

    timings: list[float] = []
    with nvtx.range("measured"):
        for _ in range(measure_steps):
            with nvtx.range("measured_step"):
                synchronize_fn()
                start_time = timer_fn()
                step_fn()
                synchronize_fn()
                end_time = timer_fn()
            timings.append(end_time - start_time)

    if after_measure_fn is not None:
        after_measure_fn()
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
    mixed_precision_dtype = _resolve_mixed_precision_dtype(config.mixed_precision_dtype)
    if config.memory_profile_max_entries <= 0:
        raise ValueError("memory_profile_max_entries must be positive.")
    if config.memory_profile_path is not None and device.type != "cuda":
        raise ValueError("CUDA memory profiling requires a CUDA device.")

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
    if config.compile_model:
        model = compile_module(
            model,
            backend=config.compile_backend,
            mode=config.compile_mode,
        )
    inputs, targets = create_random_batch(
        batch_size=config.batch_size,
        context_length=config.context_length,
        vocab_size=config.vocab_size,
        device=device,
    )

    def autocast_context():
        if mixed_precision_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=mixed_precision_dtype)

    if config.mode == "forward":
        model.eval()

        def step() -> None:
            with nvtx.range("forward"):
                with torch.no_grad():
                    with autocast_context():
                        model(inputs)
    else:
        model.train()
        optimizer = None
        if config.mode == "train-step":
            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        def step() -> None:
            model.zero_grad(set_to_none=True)
            with nvtx.range("forward"):
                with autocast_context():
                    logits = model(inputs)
                    with nvtx.range("loss"):
                        loss = cross_entropy(logits, targets)
            with nvtx.range("backward"):
                loss.backward()
            if optimizer is not None:
                with nvtx.range("optimizer_step"):
                    optimizer.step()

    peak_memory_allocated_bytes: int | None = None
    peak_memory_reserved_bytes: int | None = None

    def after_warmup() -> None:
        if device.type != "cuda":
            return
        torch.cuda.reset_peak_memory_stats(device)
        if config.memory_profile_path is not None:
            torch.cuda.memory._record_memory_history(max_entries=config.memory_profile_max_entries)

    def after_measure() -> None:
        nonlocal peak_memory_allocated_bytes, peak_memory_reserved_bytes
        if device.type != "cuda":
            return
        synchronize_device(device)
        peak_memory_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_memory_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        if config.memory_profile_path is not None:
            torch.cuda.memory._dump_snapshot(config.memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

    timings = benchmark_steps(
        step_fn=step,
        warmup_steps=config.warmup_steps,
        measure_steps=config.measure_steps,
        synchronize_fn=lambda: synchronize_device(device),
        after_warmup_fn=after_warmup,
        after_measure_fn=after_measure,
    )
    summary = summarize_timings(timings)
    return {
        **asdict(config),
        "device": str(device),
        "timings_seconds": timings,
        "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
        "peak_memory_reserved_bytes": peak_memory_reserved_bytes,
        "peak_memory_allocated_mib": (
            None if peak_memory_allocated_bytes is None else peak_memory_allocated_bytes / (1024**2)
        ),
        "peak_memory_reserved_mib": (
            None if peak_memory_reserved_bytes is None else peak_memory_reserved_bytes / (1024**2)
        ),
        "compile_model": config.compile_model,
        **summary,
    }


def format_benchmark_result(result: dict[str, object]) -> str:
    mean_ms = float(result["mean_seconds"]) * 1000.0
    std_ms = float(result["std_seconds"]) * 1000.0
    mixed_precision_dtype = result["mixed_precision_dtype"] or "none"
    base = (
        f"mode={result['mode']} device={result['device']} dtype={result['dtype']} "
        f"mixed_precision={mixed_precision_dtype} "
        f"layers={result['num_layers']} d_model={result['d_model']} heads={result['num_heads']} "
        f"d_ff={result['d_ff']} batch_size={result['batch_size']} context_length={result['context_length']} "
        f"mean_ms={mean_ms:.3f} std_ms={std_ms:.3f} n={result['num_measurements']}"
    )
    peak_allocated_mib = result.get("peak_memory_allocated_mib")
    peak_reserved_mib = result.get("peak_memory_reserved_mib")
    if peak_allocated_mib is not None and peak_reserved_mib is not None:
        base += (
            f" peak_allocated_mib={float(peak_allocated_mib):.2f}"
            f" peak_reserved_mib={float(peak_reserved_mib):.2f}"
        )
    memory_profile_path = result.get("memory_profile_path")
    if memory_profile_path is not None:
        base += f" memory_snapshot={memory_profile_path}"
    if result.get("compile_model"):
        base += " compiled=yes"
    return base


def run_benchmark_variants(
    config: BenchmarkConfig,
    execution_variants: Sequence[str],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for variant in execution_variants:
        variant_config = BenchmarkConfig(
            **{
                **asdict(config),
                "compile_model": variant == "compiled",
            }
        )
        result = run_benchmark(variant_config)
        result["variant"] = variant
        results.append(result)
    return results


def format_benchmark_comparison_table(results: Sequence[dict[str, object]]) -> str:
    header = (
        "| variant | mode | context_length | mean_ms | std_ms | "
        "peak_allocated_mib | peak_reserved_mib |"
    )
    separator = "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, separator]
    for result in results:
        peak_allocated = result.get("peak_memory_allocated_mib")
        peak_reserved = result.get("peak_memory_reserved_mib")
        rows.append(
            "| "
            f"{result['variant']} | "
            f"{result['mode']} | "
            f"{result['context_length']} | "
            f"{float(result['mean_seconds']) * 1000.0:.3f} | "
            f"{float(result['std_seconds']) * 1000.0:.3f} | "
            f"{('-' if peak_allocated is None else f'{float(peak_allocated):.2f}')} | "
            f"{('-' if peak_reserved is None else f'{float(peak_reserved):.2f}')} |"
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark forward, forward-backward, or full train-step passes for the assignment GPT model.",
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
        help="Benchmark a pure forward pass, forward plus backward, or a full train step including AdamW.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, mps, cuda, or cuda:0.",
    )
    parser.add_argument("--dtype", default="float32", help="Model parameter dtype.")
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=("float16", "fp16", "bfloat16", "bf16"),
        default=None,
        help="Optional autocast dtype. Model parameters remain in --dtype.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile the Transformer model with torch.compile before benchmarking.",
    )
    parser.add_argument(
        "--execution-variants",
        default="eager",
        help="Comma-separated execution variants: eager and/or compiled.",
    )
    parser.add_argument(
        "--compile-backend",
        default=None,
        help="Optional backend forwarded to torch.compile when compilation is enabled.",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="Optional mode forwarded to torch.compile when compilation is enabled.",
    )
    parser.add_argument(
        "--memory-profile-path",
        default=None,
        help="Optional CUDA memory snapshot output path, e.g. memory_snapshot.pickle.",
    )
    parser.add_argument(
        "--memory-profile-max-entries",
        type=int,
        default=1_000_000,
        help="Maximum CUDA allocation records to retain while memory history is enabled.",
    )
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
        mixed_precision_dtype=args.mixed_precision_dtype,
        mode=args.mode,
        seed=args.seed,
        memory_profile_path=args.memory_profile_path,
        memory_profile_max_entries=args.memory_profile_max_entries,
        compile_model=args.compile_model,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    execution_variants = _parse_variant_csv(args.execution_variants)
    if len(execution_variants) == 1:
        if args.compile_model:
            config.compile_model = True
        else:
            config.compile_model = execution_variants[0] == "compiled"
        result = run_benchmark(config)
        result["variant"] = execution_variants[0]
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        print(format_benchmark_result(result))
        return
    results = run_benchmark_variants(config, execution_variants)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    print(format_benchmark_comparison_table(results))
