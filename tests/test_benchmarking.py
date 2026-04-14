from __future__ import annotations

import math

import torch

from GPT.benchmarking import (
    BENCHMARK_MODES,
    EXECUTION_VARIANTS,
    MODEL_SIZE_SPECS,
    BenchmarkConfig,
    benchmark_steps,
    build_parser,
    compile_module,
    config_from_args,
    create_random_batch,
    format_benchmark_comparison_table,
    run_benchmark,
    run_benchmark_variants,
    summarize_timings,
)


def test_model_size_specs_include_assignment_presets() -> None:
    assert MODEL_SIZE_SPECS["small"] == {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    }
    assert MODEL_SIZE_SPECS["2.7b"] == {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    }
    assert "train-step" in BENCHMARK_MODES
    assert EXECUTION_VARIANTS == ("eager", "compiled")


def test_create_random_batch_matches_requested_shape_and_vocab_range() -> None:
    inputs, targets = create_random_batch(
        batch_size=4,
        context_length=16,
        vocab_size=100,
        device=torch.device("cpu"),
    )

    assert inputs.shape == (4, 16)
    assert targets.shape == (4, 16)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert int(inputs.min()) >= 0
    assert int(targets.min()) >= 0
    assert int(inputs.max()) < 100
    assert int(targets.max()) < 100


def test_benchmark_steps_runs_warmup_then_measured_iterations() -> None:
    calls: list[str] = []
    timer_values = iter([10.0, 10.2, 20.0, 20.5])

    def step() -> None:
        calls.append("step")

    def synchronize() -> None:
        calls.append("sync")

    def after_warmup() -> None:
        calls.append("after_warmup")

    def after_measure() -> None:
        calls.append("after_measure")

    timings = benchmark_steps(
        step_fn=step,
        warmup_steps=1,
        measure_steps=2,
        synchronize_fn=synchronize,
        timer_fn=lambda: next(timer_values),
        after_warmup_fn=after_warmup,
        after_measure_fn=after_measure,
    )

    assert timings == [0.1999999999999993, 0.5]
    assert calls == [
        "step",
        "sync",
        "after_warmup",
        "sync",
        "step",
        "sync",
        "sync",
        "step",
        "sync",
        "after_measure",
    ]


def test_summarize_timings_returns_mean_and_sample_standard_deviation() -> None:
    summary = summarize_timings([0.1, 0.2, 0.4])

    assert math.isclose(summary["mean_seconds"], 0.23333333333333336)
    assert math.isclose(summary["std_seconds"], 0.15275252316519466)
    assert summary["num_measurements"] == 3


def test_run_benchmark_reports_forward_and_backward_timings_on_cpu() -> None:
    config = BenchmarkConfig(
        vocab_size=32,
        batch_size=2,
        context_length=8,
        num_layers=1,
        d_model=8,
        num_heads=2,
        d_ff=16,
        warmup_steps=1,
        measure_steps=2,
        device="cpu",
        dtype="float32",
        mode="forward-backward",
    )

    result = run_benchmark(config)

    assert result["device"] == "cpu"
    assert result["mode"] == "forward-backward"
    assert result["num_measurements"] == 2
    assert result["mean_seconds"] >= 0.0
    assert result["std_seconds"] >= 0.0
    assert len(result["timings_seconds"]) == 2
    assert result["peak_memory_allocated_bytes"] is None
    assert result["peak_memory_reserved_bytes"] is None


def test_run_benchmark_reports_train_step_timings_on_cpu() -> None:
    config = BenchmarkConfig(
        vocab_size=32,
        batch_size=2,
        context_length=8,
        num_layers=1,
        d_model=8,
        num_heads=2,
        d_ff=16,
        warmup_steps=1,
        measure_steps=2,
        device="cpu",
        dtype="float32",
        mode="train-step",
    )

    result = run_benchmark(config)

    assert result["device"] == "cpu"
    assert result["mode"] == "train-step"
    assert result["num_measurements"] == 2
    assert result["mean_seconds"] >= 0.0
    assert result["std_seconds"] >= 0.0
    assert len(result["timings_seconds"]) == 2
    assert result["peak_memory_allocated_mib"] is None
    assert result["peak_memory_reserved_mib"] is None


def test_run_benchmark_supports_cpu_bfloat16_autocast() -> None:
    config = BenchmarkConfig(
        vocab_size=32,
        batch_size=2,
        context_length=8,
        num_layers=1,
        d_model=8,
        num_heads=2,
        d_ff=16,
        warmup_steps=1,
        measure_steps=2,
        device="cpu",
        dtype="float32",
        mixed_precision_dtype="bfloat16",
        mode="forward-backward",
    )

    result = run_benchmark(config)

    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert result["mixed_precision_dtype"] == "bfloat16"
    assert result["mode"] == "forward-backward"
    assert result["num_measurements"] == 2
    assert result["mean_seconds"] >= 0.0


def test_config_from_args_includes_memory_profile_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--model-size",
            "2.7b",
            "--mode",
            "train-step",
            "--execution-variants",
            "eager,compiled",
            "--compile-model",
            "--compile-mode",
            "reduce-overhead",
            "--memory-profile-path",
            "memory_snapshot.pickle",
            "--memory-profile-max-entries",
            "2048",
        ]
    )
    config = config_from_args(args)

    assert config.num_layers == MODEL_SIZE_SPECS["2.7b"]["num_layers"]
    assert config.d_model == MODEL_SIZE_SPECS["2.7b"]["d_model"]
    assert config.compile_model is True
    assert config.compile_mode == "reduce-overhead"
    assert config.memory_profile_path == "memory_snapshot.pickle"
    assert config.memory_profile_max_entries == 2048


def test_run_benchmark_rejects_memory_profile_on_cpu() -> None:
    config = BenchmarkConfig(
        vocab_size=32,
        batch_size=2,
        context_length=8,
        num_layers=1,
        d_model=8,
        num_heads=2,
        d_ff=16,
        warmup_steps=1,
        measure_steps=1,
        device="cpu",
        dtype="float32",
        mode="forward",
        memory_profile_path="memory_snapshot.pickle",
    )

    try:
        run_benchmark(config)
    except ValueError as exc:
        assert "CUDA memory profiling requires a CUDA device" in str(exc)
    else:
        raise AssertionError("Expected a ValueError when requesting CUDA memory profiling on CPU.")


def test_run_benchmark_variants_returns_variant_labels() -> None:
    config = BenchmarkConfig(
        vocab_size=32,
        batch_size=2,
        context_length=8,
        num_layers=1,
        d_model=8,
        num_heads=2,
        d_ff=16,
        warmup_steps=0,
        measure_steps=1,
        device="cpu",
        dtype="float32",
        mode="forward",
    )

    results = run_benchmark_variants(config, ("eager",))

    assert len(results) == 1
    assert results[0]["variant"] == "eager"
    assert results[0]["compile_model"] is False


def test_format_benchmark_comparison_table_includes_variants() -> None:
    table = format_benchmark_comparison_table(
        [
            {
                "variant": "eager",
                "mode": "forward",
                "context_length": 128,
                "mean_seconds": 0.001,
                "std_seconds": 0.0,
                "peak_memory_allocated_mib": None,
                "peak_memory_reserved_mib": None,
            },
            {
                "variant": "compiled",
                "mode": "forward",
                "context_length": 128,
                "mean_seconds": 0.0008,
                "std_seconds": 0.0,
                "peak_memory_allocated_mib": 42.5,
                "peak_memory_reserved_mib": 48.0,
            },
        ]
    )

    assert "| variant | mode | context_length | mean_ms |" in table
    assert "| eager | forward | 128 | 1.000 | 0.000 | - | - |" in table
    assert "| compiled | forward | 128 | 0.800 | 0.000 | 42.50 | 48.00 |" in table


def test_compile_module_returns_module_with_same_forward() -> None:
    module = torch.nn.Linear(4, 3)
    compiled = compile_module(module, backend="eager", mode=None)
    x = torch.randn(2, 4)

    torch.testing.assert_close(compiled(x), module(x))
