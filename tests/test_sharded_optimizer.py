from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import adapters


WORLD_SIZE = 2
SEED = 23
NUM_STEPS = 3
GLOBAL_BATCH_SIZE = 8
INPUT_DIM = 6
HIDDEN_DIM = 10
OUTPUT_DIM = 4


def _setup_process_group(rank: int, rendezvous_path: str) -> None:
    """
    Initialize the distributed process group used by the spawned workers.
    """
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=WORLD_SIZE,
        init_method=f"file://{rendezvous_path}",
    )


def _teardown_process_group() -> None:
    """
    Tear down the process group after a worker finishes.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_model() -> nn.Module:
    """
    Construct the small MLP used for distributed optimizer testing.
    """
    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    )


def _build_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a deterministic sequence of training batches.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(NUM_STEPS):
        inputs = torch.randn(GLOBAL_BATCH_SIZE, INPUT_DIM, generator=generator)
        targets = torch.randn(GLOBAL_BATCH_SIZE, OUTPUT_DIM, generator=generator)
        batches.append((inputs, targets))
    return batches


def _shard_batch(batch: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Slice one rank's shard from a global batch tensor.
    """
    shard_size = batch.shape[0] // WORLD_SIZE
    start = rank * shard_size
    end = start + shard_size
    return batch[start:end]


def _copy_state_dict(module: nn.Module) -> dict[str, np.ndarray]:
    """
    Copy a module state dict into detached NumPy arrays for comparison.
    """
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in module.state_dict().items()
    }


def _max_state_dict_diff(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> float:
    """
    Return the maximum absolute difference across matching state-dict entries.
    """
    return max(
        float(np.max(np.abs(left[key].astype(np.float64) - right[key].astype(np.float64))))
        for key in left
    )


def _average_gradients(module: nn.Module) -> None:
    """
    Average gradients across all ranks to emulate standard DDP behavior.
    """
    world_size = dist.get_world_size()
    for param in module.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
        param.grad.div_(world_size)


def _optimizer_state_tensor_count(optimizer: torch.optim.Optimizer) -> int:
    """
    Count the tensors materialized inside an optimizer state dict.
    """
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += 1
    return total


def _run_single_process_reference() -> tuple[dict[str, np.ndarray], int]:
    """
    Train the model in one process and return the final parameters and state count.
    """
    torch.manual_seed(SEED)
    model = _build_model()
    optimizer = torch.optim.AdamW(
        [
            {"params": model[0].parameters(), "lr": 1e-2},
            {"params": model[2].parameters(), "lr": 2e-2},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    loss_fn = nn.MSELoss(reduction="mean")

    for inputs, targets in _build_batches():
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    return _copy_state_dict(model), _optimizer_state_tensor_count(optimizer)


def _worker(rank: int, rendezvous_path: str, queue) -> None:
    """
    Run distributed sharded-optimizer training on one rank.
    """
    try:
        _setup_process_group(rank, rendezvous_path)
        torch.manual_seed(SEED)
        model = _build_model()
        optimizer = adapters.get_sharded_optimizer(
            [
                {"params": model[0].parameters(), "lr": 1e-2},
                {"params": model[2].parameters(), "lr": 2e-2},
            ],
            torch.optim.AdamW,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )
        loss_fn = nn.MSELoss(reduction="mean")

        for inputs, targets in _build_batches():
            optimizer.zero_grad(set_to_none=True)
            for param in model.parameters():
                assert param.grad is None

            local_inputs = _shard_batch(inputs, rank)
            local_targets = _shard_batch(targets, rank)
            predictions = model(local_inputs)
            loss = loss_fn(predictions, local_targets)
            loss.backward()
            _average_gradients(model)
            optimizer.step()

        gathered_states: list[dict[str, np.ndarray] | None] = [None for _ in range(WORLD_SIZE)]
        local_state = _copy_state_dict(model)
        dist.all_gather_object(gathered_states, local_state)

        state_tensor_counts: list[int | None] = [None for _ in range(WORLD_SIZE)]
        local_state_tensor_count = _optimizer_state_tensor_count(optimizer)
        dist.all_gather_object(state_tensor_counts, local_state_tensor_count)

        if rank == 0:
            queue.put((gathered_states, state_tensor_counts))
    finally:
        _teardown_process_group()


def _run_distributed() -> tuple[list[dict[str, np.ndarray]], list[int]]:
    """
    Spawn two workers and collect their final model states and optimizer-state sizes.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    with tempfile.NamedTemporaryFile(prefix="sharded_optimizer_test_", dir="/tmp", delete=False) as handle:
        rendezvous_path = handle.name
    try:
        mp.spawn(
            _worker,
            args=(rendezvous_path, queue),
            nprocs=WORLD_SIZE,
            join=True,
        )
        return queue.get()
    finally:
        if os.path.exists(rendezvous_path):
            os.unlink(rendezvous_path)


def test_sharded_optimizer_matches_reference_and_shards_optimizer_state() -> None:
    """
    The sharded optimizer should match full AdamW updates while storing only a shard of state.
    """
    reference_state, reference_state_tensor_count = _run_single_process_reference()
    rank_states, state_tensor_counts = _run_distributed()

    assert len(rank_states) == WORLD_SIZE
    assert len(state_tensor_counts) == WORLD_SIZE
    for rank_state in rank_states:
        assert _max_state_dict_diff(reference_state, rank_state) <= 1e-6

    assert _max_state_dict_diff(rank_states[0], rank_states[1]) <= 1e-7
    assert sum(state_tensor_counts) == reference_state_tensor_count
    for local_count in state_tensor_counts:
        assert 0 < local_count < reference_state_tensor_count
