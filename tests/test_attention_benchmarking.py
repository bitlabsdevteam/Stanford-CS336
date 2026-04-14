from __future__ import annotations

import torch

from GPT.attention_benchmarking import (
    ATTENTION_BENCHMARK_VARIANTS,
    AttentionBenchmarkConfig,
    benchmark_attention_case,
    build_parser,
    compile_attention_fn,
    config_from_args,
    estimate_attention_tensor_bytes,
    format_attention_benchmark_table,
)


def test_estimate_attention_tensor_bytes_matches_expected_fp32_sizes() -> None:
    sizes = estimate_attention_tensor_bytes(
        batch_size=8,
        sequence_length=256,
        head_dim=16,
        dtype=torch.float32,
    )

    assert sizes["qkv_bytes"] == 8 * 256 * 16 * 3 * 4
    assert sizes["output_bytes"] == 8 * 256 * 16 * 4
    assert sizes["attention_scores_bytes"] == 8 * 256 * 256 * 4
    assert sizes["attention_weights_bytes"] == 8 * 256 * 256 * 4
    assert sizes["quadratic_saved_tensors_bytes"] == 2 * 8 * 256 * 256 * 4


def test_config_from_args_parses_csv_sweep_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--batch-size",
            "8",
            "--head-dims",
            "16,64",
            "--sequence-lengths",
            "256,1024",
            "--variants",
            "eager,compiled",
            "--warmup-steps",
            "2",
            "--iterations",
            "3",
            "--compile-mode",
            "reduce-overhead",
        ]
    )
    config = config_from_args(args)

    assert config == AttentionBenchmarkConfig(
        batch_size=8,
        head_dims=(16, 64),
        sequence_lengths=(256, 1024),
        variants=("eager", "compiled"),
        warmup_steps=2,
        iterations=3,
        device="auto",
        dtype="float32",
        seed=0,
        compile_backend=None,
        compile_mode="reduce-overhead",
    )


def test_attention_benchmark_variants_include_compiled() -> None:
    assert ATTENTION_BENCHMARK_VARIANTS == ("eager", "compiled")


def test_benchmark_attention_case_runs_on_cpu_for_tiny_shape() -> None:
    result = benchmark_attention_case(
        batch_size=2,
        sequence_length=4,
        head_dim=8,
        warmup_steps=0,
        iterations=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result["status"] == "ok"
    assert result["oom_stage"] is None
    assert float(result["forward_mean_ms"]) >= 0.0
    assert float(result["backward_mean_ms"]) >= 0.0
    assert result["pre_backward_allocated_bytes"] is None
    assert result["peak_allocated_bytes"] is None


def test_format_attention_benchmark_table_includes_core_columns() -> None:
    table = format_attention_benchmark_table(
        [
            {
                "variant": "compiled",
                "head_dim": 16,
                "sequence_length": 256,
                "status": "ok",
                "oom_stage": None,
                "forward_mean_ms": 1.25,
                "backward_mean_ms": 2.5,
                "pre_backward_allocated_mib": 12.0,
                "quadratic_saved_tensors_mib": 4.0,
                "peak_allocated_mib": 18.0,
            }
        ]
    )

    assert "| variant | d_model | seq_len | status |" in table
    assert "| compiled | 16 | 256 | ok | - | 1.25 | 2.50 | 12.00 | 4.00 | 18.00 |" in table


def test_compile_attention_fn_returns_callable() -> None:
    compiled = compile_attention_fn(lambda q, k, v: q + k + v, backend="eager", mode=None)

    x = torch.ones(2, 3)
    y = compiled(x, x, x)

    torch.testing.assert_close(y, torch.full((2, 3), 3.0))
