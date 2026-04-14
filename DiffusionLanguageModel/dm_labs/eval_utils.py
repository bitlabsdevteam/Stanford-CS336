"""Diffusion corruption, evaluation, and artifact helpers for the notebook."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .parallel import unwrap_model


def _to_float_tensor(value, *, device: torch.device | None = None) -> torch.Tensor:
    """Convert scalars or tensors into a float tensor for schedule math."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(value, device=device, dtype=torch.float32)


def mask_ratio_linear_schedule(t, T: int) -> torch.Tensor:
    """Return the linear mask ratio used by a simple absorbing-state schedule."""
    T = max(int(T), 1)
    return (_to_float_tensor(t) / float(T)).clamp(0.0, 1.0)


def mask_ratio_cosine_schedule(t, T: int) -> torch.Tensor:
    """Return a cosine mask ratio that starts gentle and finishes at full masking."""
    T = max(int(T), 1)
    scaled = (_to_float_tensor(t) / float(T)).clamp(0.0, 1.0)
    return (1.0 - torch.cos(0.5 * math.pi * scaled)).clamp(0.0, 1.0)


def _eligible_mask_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    excluded_token_ids: list[int] | tuple[int, ...] | None,
) -> torch.Tensor:
    """Return the positions that may be replaced by the absorbing mask token."""
    eligible = attention_mask.bool().clone()
    for token_id in excluded_token_ids or []:
        eligible &= input_ids.ne(token_id)
    return eligible


def corruption_factory(schedule_name: str):
    """Build a corruption function for the requested diffusion schedule."""
    normalized = schedule_name.strip().lower()
    if normalized == "cosine":
        schedule_fn = mask_ratio_cosine_schedule
    elif normalized == "linear":
        schedule_fn = mask_ratio_linear_schedule
    else:
        raise ValueError(f"Unsupported schedule: {schedule_name}")

    def corrupt_with_mask(
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t: torch.Tensor,
        mask_token_id: int,
        T: int,
        excluded_token_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply independent token masking according to the selected schedule."""
        ratios = schedule_fn(t.to(input_ids.device), T).to(input_ids.device)
        eligible = _eligible_mask_positions(input_ids, attention_mask, excluded_token_ids)
        random_values = torch.rand_like(input_ids, dtype=torch.float32)
        mask = eligible & (random_values < ratios.unsqueeze(1))

        # Force at least one supervised token whenever a sequence has eligible positions.
        eligible_counts = eligible.sum(dim=1)
        empty_rows = (mask.sum(dim=1) == 0) & (eligible_counts > 0)
        if empty_rows.any():
            for row_idx in torch.where(empty_rows)[0]:
                choices = torch.where(eligible[row_idx])[0]
                chosen = choices[torch.randint(0, choices.numel(), (1,), device=input_ids.device)]
                mask[row_idx, chosen] = True

        noisy_ids = input_ids.clone()
        noisy_ids[mask] = mask_token_id
        labels = input_ids.clone()
        labels[~mask] = -100
        return noisy_ids, labels, ratios

    return corrupt_with_mask


def build_eval_plan(
    dataloader,
    *,
    T: int,
    n_batches: int = 20,
    timestep_grid: list[int] | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Snapshot a small set of batches so repeated evals are directly comparable."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        batches.append({key: value.cpu().clone() for key, value in batch.items()})
    if timestep_grid is None:
        timestep_grid = [1, max(1, T // 4), max(1, T // 2), max(1, (3 * T) // 4), T]
    timestep_grid = sorted({int(step) for step in timestep_grid if step >= 1})
    return {
        "T": int(T),
        "n_batches": len(batches),
        "timestep_grid": timestep_grid,
        "seed": int(seed),
        "generator_seed": int(generator.initial_seed()),
        "batches": batches,
    }


def _compute_masked_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, int, int]:
    """Compute average masked-token NLL and accuracy from one batch."""
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    valid = flat_labels.ne(-100)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return 0.0, 0, 0
    token_nll = F.cross_entropy(flat_logits[valid], flat_labels[valid], reduction="sum")
    predictions = flat_logits[valid].argmax(dim=-1)
    correct = int(predictions.eq(flat_labels[valid]).sum().item())
    return float(token_nll.item()), valid_count, correct


@torch.no_grad()
def evaluate_diffusion_pseudo_perplexity_from_plan(
    *,
    model,
    eval_plan: dict[str, object],
    corruption_fn,
    mask_token_id: int,
    T: int,
    excluded_token_ids: list[int] | tuple[int, ...] | None,
    schedule_name: str,
    bootstrap_samples: int = 0,
) -> dict[str, object]:
    """Evaluate a diffusion checkpoint on a fixed timestep-aware masking plan."""
    del bootstrap_samples

    model.eval()
    base_model = unwrap_model(model)
    device = next(base_model.parameters()).device
    timestep_metrics = []
    total_nll = 0.0
    total_masked = 0
    total_correct = 0

    for timestep in eval_plan["timestep_grid"]:
        step_nll = 0.0
        step_masked = 0
        step_correct = 0
        for frozen_batch in eval_plan["batches"]:
            batch = {key: value.to(device) for key, value in frozen_batch.items()}
            batch_size = batch["input_ids"].size(0)
            t = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
            noisy_ids, labels, ratios = corruption_fn(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                t=t,
                mask_token_id=mask_token_id,
                T=T,
                excluded_token_ids=excluded_token_ids,
            )
            logits = model(noisy_ids, timesteps=t, attention_mask=batch["attention_mask"])
            batch_nll, batch_masked, batch_correct = _compute_masked_metrics(logits, labels)
            step_nll += batch_nll
            step_masked += batch_masked
            step_correct += batch_correct
            total_nll += batch_nll
            total_masked += batch_masked
            total_correct += batch_correct

        mean_nll = step_nll / max(step_masked, 1)
        timestep_metrics.append(
            {
                "timestep": int(timestep),
                "mask_ratio": float(ratios.mean().item()) if step_masked > 0 else 0.0,
                "mean_nll": mean_nll,
                "pseudo_perplexity": float(math.exp(min(mean_nll, 20.0))),
                "masked_accuracy": step_correct / max(step_masked, 1),
                "masked_tokens": int(step_masked),
            }
        )

    mean_nll = total_nll / max(total_masked, 1)
    result = {
        "schedule_name": schedule_name,
        "eval_protocol": {
            "n_batches": eval_plan["n_batches"],
            "timestep_grid": eval_plan["timestep_grid"],
            "T": int(T),
        },
        "mean_masked_nll": mean_nll,
        "pseudo_perplexity": float(math.exp(min(mean_nll, 20.0))),
        "masked_accuracy": total_correct / max(total_masked, 1),
        "quality_summary": {
            "masked_token_accuracy": total_correct / max(total_masked, 1),
            "mean_masked_nll": mean_nll,
            "pseudo_perplexity": float(math.exp(min(mean_nll, 20.0))),
        },
        "schedule_reweighted_effective_sample_size": int(total_masked),
        "schedule_reweighted_effective_sample_size_fraction": 1.0,
        "schedule_reweighted_nonzero_examples": int(total_masked > 0),
        "timestep_metrics": timestep_metrics,
    }
    model.train()
    return result


def export_eval_result(
    path: str | Path,
    artifact_name: str,
    result: dict[str, object],
) -> dict[str, str]:
    """Write one evaluation JSON artifact to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return {"artifact_name": artifact_name, "path": str(output_path)}


def _load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: torch.device,
    config_cls,
    model_cls,
):
    """Load a notebook checkpoint saved as model state plus JSON config."""
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.pt"
    config = config_cls(**json.loads(config_path.read_text()))
    model = model_cls(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def compare_schedule_checkpoints(
    *,
    cosine_dir: str | Path,
    linear_dir: str | Path | None,
    device: torch.device,
    config_cls,
    model_cls,
    dataloader,
    mask_token_id: int,
    excluded_token_ids: list[int] | tuple[int, ...] | None,
    n_batches: int,
    timestep_grid: list[int],
    seed: int,
    bootstrap_samples: int = 0,
) -> dict[str, object]:
    """Compare cosine and linear checkpoints on the same frozen evaluation plan."""
    cosine_model, cosine_cfg = _load_checkpoint(
        cosine_dir,
        device=device,
        config_cls=config_cls,
        model_cls=model_cls,
    )
    eval_plan = build_eval_plan(
        dataloader,
        T=cosine_cfg.diffusion_steps,
        n_batches=n_batches,
        timestep_grid=timestep_grid,
        seed=seed,
    )
    cosine_result = evaluate_diffusion_pseudo_perplexity_from_plan(
        model=cosine_model,
        eval_plan=eval_plan,
        corruption_fn=corruption_factory("cosine"),
        mask_token_id=mask_token_id,
        T=cosine_cfg.diffusion_steps,
        excluded_token_ids=excluded_token_ids,
        schedule_name="cosine",
        bootstrap_samples=bootstrap_samples,
    )

    output = {
        "comparison_protocol": eval_plan["timestep_grid"],
        "cosine": cosine_result,
        "linear": None,
        "winner": "cosine",
        "winner_confidence": {"note": "linear baseline missing"},
        "decision_summary": {
            "preferred_schedule": "cosine",
            "reason": "no linear baseline supplied",
        },
        "timestep_deltas": [],
    }

    if not linear_dir:
        return output

    linear_model, linear_cfg = _load_checkpoint(
        linear_dir,
        device=device,
        config_cls=config_cls,
        model_cls=model_cls,
    )
    linear_result = evaluate_diffusion_pseudo_perplexity_from_plan(
        model=linear_model,
        eval_plan=eval_plan,
        corruption_fn=corruption_factory("linear"),
        mask_token_id=mask_token_id,
        T=linear_cfg.diffusion_steps,
        excluded_token_ids=excluded_token_ids,
        schedule_name="linear",
        bootstrap_samples=bootstrap_samples,
    )
    deltas = []
    for linear_metric, cosine_metric in zip(
        linear_result["timestep_metrics"],
        cosine_result["timestep_metrics"],
    ):
        deltas.append(
            {
                "timestep": linear_metric["timestep"],
                "linear_minus_cosine_nll": linear_metric["mean_nll"] - cosine_metric["mean_nll"],
                "linear_minus_cosine_ppl": linear_metric["pseudo_perplexity"] - cosine_metric["pseudo_perplexity"],
            }
        )

    cosine_nll = cosine_result["mean_masked_nll"]
    linear_nll = linear_result["mean_masked_nll"]
    winner = "cosine" if cosine_nll <= linear_nll else "linear"
    output.update(
        {
            "linear": linear_result,
            "winner": winner,
            "winner_confidence": {
                "mean_masked_nll_delta": linear_nll - cosine_nll,
            },
            "decision_summary": {
                "preferred_schedule": winner,
                "reason": "lower mean masked-token negative log-likelihood",
            },
            "timestep_deltas": deltas,
        }
    )
    return output


def export_schedule_comparison(
    path: str | Path,
    result: dict[str, object],
) -> dict[str, str]:
    """Write a cosine-vs-linear comparison artifact to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return {"path": str(output_path)}
