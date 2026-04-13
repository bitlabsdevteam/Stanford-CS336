from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, IO

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .cross_entropy import cross_entropy
from .optimization import AdamW, clip_gradients, lr_cosine_schedule
from .transformer_lm import TransformerLM


CheckpointDestination = str | os.PathLike | BinaryIO | IO[bytes]


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Sample a batch of fixed-length token windows and next-token targets.

    The input array is expected to be a one-dimensional numpy integer array or memmap.
    Returned tensors are shaped ``(batch_size, context_length)`` and moved onto the
    requested PyTorch device.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {x.shape}.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if x.shape[0] <= context_length:
        raise ValueError(
            "x must contain at least context_length + 1 tokens to form targets."
        )

    start_positions = np.random.randint(
        0,
        x.shape[0] - context_length,
        size=batch_size,
    )
    inputs = np.stack([np.asarray(x[pos : pos + context_length], dtype=np.int64) for pos in start_positions])
    targets = np.stack(
        [np.asarray(x[pos + 1 : pos + 1 + context_length], dtype=np.int64) for pos in start_positions]
    )

    x_batch = torch.as_tensor(inputs, dtype=torch.long, device=device)
    y_batch = torch.as_tensor(targets, dtype=torch.long, device=device)
    return x_batch, y_batch


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: CheckpointDestination,
) -> None:
    """
    Save model state, optimizer state, and iteration number to ``out``.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(checkpoint, out)


def _infer_model_device(model: nn.Module) -> torch.device:
    """
    Infer the device associated with the first parameter, defaulting to CPU.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_checkpoint(
    src: CheckpointDestination,
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    Load model state, optimizer state, and return the saved iteration number.
    """
    load_kwargs = {"map_location": _infer_model_device(model)}
    try:
        checkpoint = torch.load(src, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(src, **load_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])


def estimate_loss(
    model: TransformerLM,
    data: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    eval_batches: int,
) -> float:
    """
    Estimate mean next-token cross-entropy over a fixed number of random batches.
    """
    if eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")

    model_was_training = model.training
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(eval_batches):
            x_batch, y_batch = get_batch(data, batch_size, context_length, device)
            logits = model(x_batch)
            losses.append(float(cross_entropy(logits, y_batch).item()))
    if model_was_training:
        model.train()
    return sum(losses) / len(losses)


@dataclass(slots=True)
class TrainingConfig:
    """
    Hyperparameters and runtime configuration for a language-model training run.
    """

    train_data_path: str
    val_data_path: str | None
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    batch_size: int
    total_iters: int
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_clip: float
    device: str
    dtype: str
    checkpoint_path: str | None = None
    resume_from_checkpoint: str | None = None
    train_data_dtype: str = "uint16"
    val_data_dtype: str | None = None
    log_interval: int = 1


def load_token_array(path: str | os.PathLike[str], *, dtype: str = "uint16") -> np.ndarray:
    """
    Open tokenized data in a memory-efficient way.

    ``.npy`` files are opened with ``np.load(..., mmap_mode='r')``. Other paths are
    treated as raw contiguous token buffers and opened via ``np.memmap``.
    """
    resolved = Path(path)
    if resolved.suffix == ".npy":
        array = np.load(resolved, mmap_mode="r")
    else:
        array = np.memmap(resolved, mode="r", dtype=np.dtype(dtype))

    if array.ndim != 1:
        raise ValueError(f"Expected one-dimensional token data at {resolved}, got shape {array.shape}.")
    return array


def resolve_training_device(device_name: str | torch.device) -> torch.device:
    """
    Resolve a requested training device, supporting ``auto``, CUDA, and MPS.
    """
    if isinstance(device_name, torch.device):
        requested = str(device_name)
        original_value = str(device_name)
    else:
        requested = device_name.strip().lower()
        original_value = device_name

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device(original_value)

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{original_value}'. Expected auto, cpu, cuda, cuda:N, or mps."
    )


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    """
    Translate a CLI dtype name into a torch dtype object.
    """
    normalized = dtype_name.lower()
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return dtype_map[normalized]


def _set_learning_rate(optimizer: Optimizer, lr: float) -> None:
    """
    Apply a scalar learning rate to each optimizer parameter group.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_language_model(config: TrainingConfig) -> tuple[TransformerLM, Optimizer, list[dict[str, float | int]]]:
    """
    Train a Transformer LM from tokenized train/validation arrays.

    Returns the trained model, optimizer, and a compact history of logged metrics.
    """
    device = resolve_training_device(config.device)
    model_dtype = _torch_dtype_from_name(config.dtype)

    train_data = load_token_array(config.train_data_path, dtype=config.train_data_dtype)
    val_dtype = config.val_data_dtype if config.val_data_dtype is not None else config.train_data_dtype
    val_data = None
    if config.val_data_path is not None:
        val_data = load_token_array(config.val_data_path, dtype=val_dtype)

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        device=device,
        dtype=model_dtype,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    start_iteration = 0
    if config.resume_from_checkpoint is not None:
        start_iteration = load_checkpoint(config.resume_from_checkpoint, model, optimizer)
    model.train()

    for iteration in range(start_iteration, config.total_iters):
        current_lr = lr_cosine_schedule(
            iteration,
            max_lr=config.learning_rate,
            min_lr=config.min_learning_rate,
            warmup_iters=config.warmup_iters,
            cosine_cycle_iters=config.cosine_cycle_iters,
        )
        _set_learning_rate(optimizer, current_lr)

        x_batch, y_batch = get_batch(
            train_data,
            config.batch_size,
            config.context_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = cross_entropy(logits, y_batch)
        loss.backward()
        if config.grad_clip > 0.0:
            clip_gradients(model.parameters(), config.grad_clip)
        optimizer.step()

        if iteration % config.log_interval == 0:
            train_loss = float(loss.item())
            record: dict[str, float | int] = {
                "iteration": iteration,
                "lr": float(current_lr),
                "train_loss": train_loss,
            }
            if iteration % config.eval_interval == 0:
                record["train_eval_loss"] = estimate_loss(
                    model,
                    train_data,
                    batch_size=config.batch_size,
                    context_length=config.context_length,
                    device=device,
                    eval_batches=config.eval_batches,
                )
                if val_data is not None:
                    record["val_loss"] = estimate_loss(
                        model,
                        val_data,
                        batch_size=config.batch_size,
                        context_length=config.context_length,
                        device=device,
                        eval_batches=config.eval_batches,
                    )
            history.append(record)

        if (
            config.checkpoint_path is not None
            and config.checkpoint_interval > 0
            and ((iteration + 1) % config.checkpoint_interval == 0 or iteration + 1 == config.total_iters)
        ):
            save_checkpoint(model, optimizer, iteration + 1, config.checkpoint_path)

    return model, optimizer, history


def format_metrics(record: dict[str, float | int], ordered_keys: Sequence[str] | None = None) -> str:
    """
    Render a compact one-line metrics string for console logging.
    """
    keys = list(ordered_keys) if ordered_keys is not None else list(record.keys())
    pieces: list[str] = []
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            pieces.append(f"{key}={value:.6f}")
        else:
            pieces.append(f"{key}={value}")
    return " ".join(pieces)
