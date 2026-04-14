"""Small diffusion-LM utilities used by the Colab notebooks in this folder."""

from .data_utils import TokenBlockDataset, collate_blocks, format_as_chat
from .eval_utils import (
    build_eval_plan,
    compare_schedule_checkpoints,
    corruption_factory,
    evaluate_diffusion_pseudo_perplexity_from_plan,
    export_eval_result,
    export_schedule_comparison,
    mask_ratio_cosine_schedule,
    mask_ratio_linear_schedule,
)
from .hf_utils import (
    build_eval_view_rows,
    build_schedule_comparison_rows,
    upload_checkpoint_to_hub,
    validate_hf_export_bundle,
    write_hf_export_bundle,
)
from .modeling import DiffusionLMConfig, DiffusionTransformerLM
from .parallel import (
    FiveDParallelConfig,
    build_optimizer,
    finalize_parallel_backward,
    normalize_loss,
    normalize_parallel_config,
    prepare_parallel_model,
    resolve_device,
    summarize_parallel_plan,
    training_device_for_runtime,
    unwrap_model,
)

__all__ = [
    "DiffusionLMConfig",
    "DiffusionTransformerLM",
    "TokenBlockDataset",
    "collate_blocks",
    "format_as_chat",
    "build_eval_plan",
    "compare_schedule_checkpoints",
    "corruption_factory",
    "evaluate_diffusion_pseudo_perplexity_from_plan",
    "export_eval_result",
    "export_schedule_comparison",
    "mask_ratio_cosine_schedule",
    "mask_ratio_linear_schedule",
    "build_eval_view_rows",
    "build_schedule_comparison_rows",
    "upload_checkpoint_to_hub",
    "validate_hf_export_bundle",
    "write_hf_export_bundle",
    "FiveDParallelConfig",
    "build_optimizer",
    "finalize_parallel_backward",
    "normalize_loss",
    "normalize_parallel_config",
    "prepare_parallel_model",
    "resolve_device",
    "summarize_parallel_plan",
    "training_device_for_runtime",
    "unwrap_model",
]
