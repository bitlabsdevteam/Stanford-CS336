from __future__ import annotations

import math

import torch

import adapters


def test_get_adamw_cls_matches_torch_adamw_over_multiple_steps() -> None:
    adamw_cls = adapters.get_adamw_cls()

    param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64))
    ref_param = torch.nn.Parameter(param.detach().clone())

    optimizer = adamw_cls(
        [param],
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    reference = torch.optim.AdamW(
        [ref_param],
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    grads = [
        torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64),
        torch.tensor([-0.4, 0.3, 0.1], dtype=torch.float64),
        torch.tensor([0.0, -0.2, 0.5], dtype=torch.float64),
    ]

    for grad in grads:
        param.grad = grad.clone()
        ref_param.grad = grad.clone()
        optimizer.step()
        reference.step()

    torch.testing.assert_close(param, ref_param, atol=1e-12, rtol=1e-10)

    state = optimizer.state[param]
    assert state["step"].item() == 3
    assert state["exp_avg"].shape == param.shape
    assert state["exp_avg_sq"].shape == param.shape


def test_get_adamw_cls_rejects_sparse_gradients() -> None:
    adamw_cls = adapters.get_adamw_cls()
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    optimizer = adamw_cls([param], lr=1e-3)

    indices = torch.tensor([[0, 1]])
    values = torch.tensor([1.0, -1.0])
    param.grad = torch.sparse_coo_tensor(indices, values, size=(2,))

    try:
        optimizer.step()
    except RuntimeError as exc:
        assert "sparse gradients" in str(exc)
    else:
        raise AssertionError("Expected sparse gradients to raise RuntimeError.")


def test_get_lr_cosine_schedule_handles_warmup_cosine_and_floor() -> None:
    schedule = adapters.get_lr_cosine_schedule()

    max_lr = 1.0
    min_lr = 0.1
    warmup_iters = 4
    cosine_cycle_iters = 10

    assert schedule(
        0,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    ) == 0.0
    assert schedule(
        2,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    ) == 0.5
    assert schedule(
        4,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    ) == max_lr

    expected_mid = min_lr + 0.5 * (
        1.0 + math.cos(math.pi * ((7 - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    ) * (max_lr - min_lr)
    assert schedule(
        7,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    ) == expected_mid
    assert schedule(
        10,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    ) == min_lr
    assert schedule(
        11,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    ) == min_lr


def test_run_gradient_clipping_scales_all_gradients_with_one_global_factor() -> None:
    p1 = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float32))
    p2 = torch.nn.Parameter(torch.tensor([12.0], dtype=torch.float32))
    p3 = torch.nn.Parameter(torch.tensor([5.0], dtype=torch.float32))

    p1.grad = torch.tensor([3.0, 4.0], dtype=torch.float32)
    p2.grad = torch.tensor([12.0], dtype=torch.float32)
    p3.grad = None

    old_total_norm = adapters.run_gradient_clipping([p1, p2, p3], max_l2_norm=6.0)

    expected_scale = 6.0 / (13.0 + 1e-6)
    torch.testing.assert_close(old_total_norm, torch.tensor(13.0))
    torch.testing.assert_close(p1.grad, torch.tensor([3.0, 4.0]) * expected_scale)
    torch.testing.assert_close(p2.grad, torch.tensor([12.0]) * expected_scale)


def test_run_gradient_clipping_leaves_small_gradients_unchanged() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
    p.grad = torch.tensor([0.3, 0.4], dtype=torch.float32)

    old_total_norm = adapters.run_gradient_clipping([p], max_l2_norm=1.0)

    torch.testing.assert_close(old_total_norm, torch.tensor(0.5))
    torch.testing.assert_close(p.grad, torch.tensor([0.3, 0.4]))
