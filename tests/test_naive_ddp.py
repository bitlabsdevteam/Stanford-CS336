from __future__ import annotations

import torch

from GPT.naive_ddp import NaiveDDPCheckConfig, run_naive_ddp_check, shard_batch


def test_shard_batch_returns_equal_contiguous_rank_slices() -> None:
    batch = torch.arange(24, dtype=torch.float32).view(6, 4)

    shard0 = shard_batch(batch, rank=0, world_size=3)
    shard1 = shard_batch(batch, rank=1, world_size=3)
    shard2 = shard_batch(batch, rank=2, world_size=3)

    torch.testing.assert_close(shard0, batch[0:2])
    torch.testing.assert_close(shard1, batch[2:4])
    torch.testing.assert_close(shard2, batch[4:6])


def test_run_naive_ddp_check_matches_single_process_training() -> None:
    config = NaiveDDPCheckConfig(
        world_size=2,
        backend="gloo",
        device="cpu",
        input_dim=6,
        hidden_dim=10,
        output_dim=3,
        global_batch_size=8,
        num_steps=3,
        learning_rate=5e-3,
        seed=7,
    )

    result = run_naive_ddp_check(config, spawn_processes=False)

    assert result["status"] == "ok"
    assert result["execution_mode"] == "simulated"
    assert float(result["max_model_abs_diff"]) <= 1e-7
    assert float(result["max_optimizer_abs_diff"]) <= 1e-7
    assert float(result["max_inter_replica_model_abs_diff"]) <= 1e-7
