from __future__ import annotations

import os
import tempfile
from collections.abc import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from GPT.ddp import (
    BucketedDistributedDataParallel,
    FlatDistributedDataParallel,
    IndividualParameterDistributedDataParallel,
)


WORLD_SIZE = 2
SEED = 17
NUM_STEPS = 3
GLOBAL_BATCH_SIZE = 8
INPUT_DIM = 6
HIDDEN_DIM = 10
OUTPUT_DIM = 4


def _setup_process_group(rank: int, rendezvous_path: str) -> None:
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=WORLD_SIZE,
        init_method=f"file://{rendezvous_path}",
    )


def _teardown_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    )


def _build_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(NUM_STEPS):
        inputs = torch.randn(GLOBAL_BATCH_SIZE, INPUT_DIM, generator=generator)
        targets = torch.randn(GLOBAL_BATCH_SIZE, OUTPUT_DIM, generator=generator)
        batches.append((inputs, targets))
    return batches


def _shard_batch(batch: torch.Tensor, rank: int) -> torch.Tensor:
    shard_size = batch.shape[0] // WORLD_SIZE
    start = rank * shard_size
    end = start + shard_size
    return batch[start:end]


def _copy_state_dict(module: nn.Module) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in module.state_dict().items()
    }


def _max_state_dict_diff(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> float:
    return max(
        float(np.max(np.abs(left[key].astype(np.float64) - right[key].astype(np.float64))))
        for key in left
    )


def _run_single_process_reference() -> dict[str, np.ndarray]:
    torch.manual_seed(SEED)
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss(reduction="mean")

    for inputs, targets in _build_batches():
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    return _copy_state_dict(model)


def _ddp_worker(
    rank: int,
    rendezvous_path: str,
    queue,
    wrapper_name: str,
    bucket_size_mb: float | None,
) -> None:
    try:
        _setup_process_group(rank, rendezvous_path)
        torch.manual_seed(SEED)
        model = _build_model()
        if wrapper_name == "flat":
            ddp_model = FlatDistributedDataParallel(model)
        elif wrapper_name == "individual":
            ddp_model = IndividualParameterDistributedDataParallel(model)
        elif wrapper_name == "bucketed":
            assert bucket_size_mb is not None
            ddp_model = BucketedDistributedDataParallel(model, bucket_size_mb=bucket_size_mb)
        else:
            raise ValueError(f"Unknown wrapper_name {wrapper_name!r}")

        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss(reduction="mean")

        for inputs, targets in _build_batches():
            optimizer.zero_grad(set_to_none=True)
            local_inputs = _shard_batch(inputs, rank)
            local_targets = _shard_batch(targets, rank)
            predictions = ddp_model(local_inputs)
            loss = loss_fn(predictions, local_targets)
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

        rank_states: list[dict[str, torch.Tensor] | None] = [None for _ in range(WORLD_SIZE)]
        local_state = _copy_state_dict(model)
        dist.all_gather_object(rank_states, local_state)
        if rank == 0:
            queue.put(rank_states)
    finally:
        _teardown_process_group()


def _run_wrapper(
    wrapper_name: str,
    *,
    bucket_size_mb: float | None = None,
) -> list[dict[str, np.ndarray]]:
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    with tempfile.NamedTemporaryFile(prefix="ddp_wrapper_test_", dir="/tmp", delete=False) as handle:
        rendezvous_path = handle.name
    try:
        mp.spawn(
            _ddp_worker,
            args=(rendezvous_path, queue, wrapper_name, bucket_size_mb),
            nprocs=WORLD_SIZE,
            join=True,
        )
        gathered_states = queue.get()
        return gathered_states
    finally:
        if os.path.exists(rendezvous_path):
            os.unlink(rendezvous_path)


def _assert_matches_reference(
    wrapped_run: Callable[[], list[dict[str, np.ndarray]]],
) -> None:
    reference_state = _run_single_process_reference()
    rank_states = wrapped_run()
    assert len(rank_states) == WORLD_SIZE
    for rank_state in rank_states:
        assert _max_state_dict_diff(reference_state, rank_state) <= 1e-6
    assert _max_state_dict_diff(rank_states[0], rank_states[1]) <= 1e-7


def test_flat_distributed_data_parallel_matches_single_process_reference() -> None:
    _assert_matches_reference(lambda: _run_wrapper("flat"))


def test_individual_parameter_ddp_matches_single_process_reference() -> None:
    _assert_matches_reference(lambda: _run_wrapper("individual"))


def test_bucketed_ddp_matches_single_process_reference() -> None:
    _assert_matches_reference(lambda: _run_wrapper("bucketed", bucket_size_mb=0.0001))
