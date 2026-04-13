from __future__ import annotations

import math

import torch

from GPT.benchmarking import (
    MODEL_SIZE_SPECS,
    BenchmarkConfig,
    benchmark_steps,
    create_random_batch,
    run_benchmark,
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

    timings = benchmark_steps(
        step_fn=step,
        warmup_steps=1,
        measure_steps=2,
        synchronize_fn=synchronize,
        timer_fn=lambda: next(timer_values),
    )

    assert timings == [0.1999999999999993, 0.5]
    assert calls == [
        "step",
        "sync",
        "sync",
        "step",
        "sync",
        "sync",
        "step",
        "sync",
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
