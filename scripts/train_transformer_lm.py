from __future__ import annotations

import argparse

from GPT import TrainingConfig, format_metrics, train_language_model


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI parser for Transformer LM training runs.
    """
    parser = argparse.ArgumentParser(
        description="Train the assignment Transformer LM on tokenized data.",
    )
    parser.add_argument("--train-data", required=True, help="Path to train token data (.npy or raw memmap).")
    parser.add_argument("--val-data", default=None, help="Optional validation token data path.")
    parser.add_argument("--train-data-dtype", default="uint16", help="Raw memmap dtype for train data.")
    parser.add_argument("--val-data-dtype", default=None, help="Raw memmap dtype for validation data.")
    parser.add_argument("--checkpoint", default=None, help="Path to save checkpoints.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, mps, cuda, or cuda:0.",
    )
    parser.add_argument("--dtype", default="float32", help="Model parameter dtype.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--context-length", type=int, default=128, help="Training context length.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of Transformer blocks.")
    parser.add_argument("--d-model", type=int, default=256, help="Hidden size.")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward hidden size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--total-iters", type=int, default=1000, help="Number of optimizer steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--min-learning-rate", type=float, default=3e-5, help="Final cosine floor.")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Warmup steps.")
    parser.add_argument("--cosine-cycle-iters", type=int, default=1000, help="Cosine decay horizon.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Global gradient clipping threshold.")
    parser.add_argument("--eval-interval", type=int, default=100, help="Training steps between evals.")
    parser.add_argument("--eval-batches", type=int, default=8, help="Random batches per evaluation.")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Training steps between checkpoints.")
    parser.add_argument("--log-interval", type=int, default=1, help="Training steps between log lines.")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """
    Translate parsed CLI arguments into a training configuration.
    """
    return TrainingConfig(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        total_iters=args.total_iters,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        grad_clip=args.grad_clip,
        device=args.device,
        dtype=args.dtype,
        checkpoint_path=args.checkpoint,
        resume_from_checkpoint=args.resume,
        train_data_dtype=args.train_data_dtype,
        val_data_dtype=args.val_data_dtype,
        log_interval=args.log_interval,
    )


def main() -> None:
    """
    Parse arguments, optionally resume from checkpoint, and run training.
    """
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    _, _, history = train_language_model(config)
    for record in history:
        print(format_metrics(record, ("iteration", "lr", "train_loss", "train_eval_loss", "val_loss")))


if __name__ == "__main__":
    main()
