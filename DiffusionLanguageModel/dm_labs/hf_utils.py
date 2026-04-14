"""Hugging Face export helpers for the diffusion notebook."""

from __future__ import annotations

import json
import os
from pathlib import Path


def build_eval_view_rows(eval_summary: dict | None) -> list[dict[str, object]]:
    """Flatten the evaluation summary into row-shaped records for notebook display."""
    if not eval_summary:
        return []
    rows = [
        {
            "schedule": eval_summary.get("schedule_name"),
            "metric": "mean_masked_nll",
            "value": eval_summary.get("mean_masked_nll"),
        },
        {
            "schedule": eval_summary.get("schedule_name"),
            "metric": "pseudo_perplexity",
            "value": eval_summary.get("pseudo_perplexity"),
        },
        {
            "schedule": eval_summary.get("schedule_name"),
            "metric": "masked_accuracy",
            "value": eval_summary.get("masked_accuracy"),
        },
    ]
    rows.extend(eval_summary.get("timestep_metrics", []))
    return rows


def build_schedule_comparison_rows(comparison_summary: dict | None) -> list[dict[str, object]]:
    """Flatten the paired schedule comparison into notebook-friendly rows."""
    if not comparison_summary:
        return []
    rows = [
        {
            "winner": comparison_summary.get("winner"),
            "winner_confidence": comparison_summary.get("winner_confidence"),
            "decision_summary": comparison_summary.get("decision_summary"),
        }
    ]
    rows.extend(comparison_summary.get("timestep_deltas", []))
    return rows


def validate_hf_export_bundle(
    local_artifact_dir: str | Path,
    *,
    repo_id: str,
) -> dict[str, object]:
    """Check that the minimum files for a notebook export are present."""
    artifact_dir = Path(local_artifact_dir)
    required = [
        artifact_dir / "model.pt",
        artifact_dir / "config.json",
        artifact_dir / "README.md",
    ]
    tokenizer_dir = artifact_dir / "tokenizer"
    missing = [str(path) for path in required if not path.exists()]
    if not tokenizer_dir.exists():
        missing.append(str(tokenizer_dir))
    return {
        "repo_id": repo_id,
        "valid": not missing,
        "missing": missing,
    }


def _build_model_card(
    *,
    repo_id: str,
    eval_summary: dict | None,
    comparison_summary: dict | None,
    eval_plan: dict | None,
) -> str:
    """Compose a concise model card for the exported diffusion checkpoint."""
    lines = [
        f"# {repo_id}",
        "",
        "TinyStories diffusion language model trained from scratch with a cosine masking schedule.",
        "",
        "## Training recipe",
        "",
        "- Architecture: full-attention Transformer denoiser",
        "- Objective: masked-token denoising under a discrete absorbing-state diffusion process",
        "- Noise schedule: cosine",
        "- Parallelism surface: 5D-ready topology with direct in-notebook data parallel execution",
        "- Deployment target: Colab/cloud runtime with up to 2x H100 GPUs",
        "",
    ]
    if eval_summary:
        lines.extend(
            [
                "## Evaluation snapshot",
                "",
                f"- Mean masked NLL: {eval_summary.get('mean_masked_nll')}",
                f"- Pseudo-perplexity: {eval_summary.get('pseudo_perplexity')}",
                f"- Masked-token accuracy: {eval_summary.get('masked_accuracy')}",
                "",
            ]
        )
    if comparison_summary:
        lines.extend(
            [
                "## Schedule comparison",
                "",
                f"- Preferred schedule: {comparison_summary.get('winner')}",
                f"- Decision summary: {comparison_summary.get('decision_summary')}",
                "",
            ]
        )
    if eval_plan:
        lines.extend(
            [
                "## Eval plan",
                "",
                f"- Timesteps: {eval_plan.get('timestep_grid')}",
                f"- Number of frozen batches: {eval_plan.get('n_batches')}",
                "",
            ]
        )
    return "\n".join(lines)


def write_hf_export_bundle(
    local_artifact_dir: str | Path,
    *,
    repo_id: str,
    eval_summary: dict | None = None,
    comparison_summary: dict | None = None,
    eval_plan: dict | None = None,
    overwrite_model_card: bool = True,
) -> dict[str, str]:
    """Write README and optional eval artifacts into the export directory."""
    artifact_dir = Path(local_artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    readme_path = artifact_dir / "README.md"
    if overwrite_model_card or not readme_path.exists():
        readme_path.write_text(
            _build_model_card(
                repo_id=repo_id,
                eval_summary=eval_summary,
                comparison_summary=comparison_summary,
                eval_plan=eval_plan,
            ),
            encoding="utf-8",
        )

    if eval_summary is not None:
        (artifact_dir / "eval_summary.json").write_text(
            json.dumps(eval_summary, indent=2),
            encoding="utf-8",
        )
    if comparison_summary is not None:
        (artifact_dir / "schedule_comparison.json").write_text(
            json.dumps(comparison_summary, indent=2),
            encoding="utf-8",
        )

    return {
        "artifact_dir": str(artifact_dir),
        "readme_path": str(readme_path),
    }


def upload_checkpoint_to_hub(
    *,
    local_artifact_dir: str | Path,
    eval_summary: dict | None = None,
    comparison_summary: dict | None = None,
    overwrite_model_card: bool = True,
    commit_message: str = "Upload diffusion LM artifacts",
    repo_id: str | None = None,
) -> str:
    """Create or update a Hugging Face repo with the local export bundle."""
    del eval_summary, comparison_summary, overwrite_model_card

    from huggingface_hub import HfApi

    repo_id = repo_id or os.getenv("HF_REPO_ID", "your-hf-username/tinystories-diffusion-lm-parallel")
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_artifact_dir),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"
