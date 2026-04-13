from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import (
    compression_ratio_bytes_per_token,
    encode_text_file_to_uint16,
    estimate_tokenization_time_seconds,
    estimate_tokenizer_throughput_bytes_per_second,
    sample_documents,
)
from .serialization import save_merges, save_vocab
from .tokenizer import Tokenizer
from .trainer import train_bpe_with_profile


def _add_shared_tokenizer_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the common vocabulary, merges, and special-token arguments used by CLI commands.
    """
    parser.add_argument("--vocab", required=True, help="Path to the serialized vocabulary JSON.")
    parser.add_argument("--merges", required=True, help="Path to the serialized merges file.")
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=[],
        help="Special token to preserve during encoding. Pass multiple times as needed.",
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the package CLI with train, encode, and benchmark subcommands.
    """
    parser = argparse.ArgumentParser(description="Production-oriented byte-level BPE toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a byte-level BPE tokenizer.")
    train_parser.add_argument("--input", required=True, help="Path to the corpus text file.")
    train_parser.add_argument("--vocab-size", required=True, type=int, help="Maximum final vocabulary size.")
    train_parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=[],
        help="Special token to include and treat as a hard boundary.",
    )
    train_parser.add_argument("--num-processes", type=int, default=1, help="Worker processes for pre-tokenization.")
    train_parser.add_argument("--vocab-out", required=True, help="Output JSON path for the vocabulary.")
    train_parser.add_argument("--merges-out", required=True, help="Output text path for the merges.")

    encode_parser = subparsers.add_parser("encode", help="Encode a text file to uint16 token ids.")
    _add_shared_tokenizer_args(encode_parser)
    encode_parser.add_argument("--input", required=True, help="Path to the input text file.")
    encode_parser.add_argument("--output", required=True, help="Path to the output `.npy` file.")
    encode_parser.add_argument("--chunk-size", type=int, default=1 << 20, help="Streaming chunk size in characters.")

    benchmark_parser = subparsers.add_parser("benchmark", help="Sample docs and report compression / throughput.")
    _add_shared_tokenizer_args(benchmark_parser)
    benchmark_parser.add_argument("--input", required=True, help="Path to the corpus text file.")
    benchmark_parser.add_argument("--document-delimiter", required=True, help="Document boundary token in the corpus.")
    benchmark_parser.add_argument("--num-documents", type=int, default=10, help="Number of sampled documents.")
    benchmark_parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    benchmark_parser.add_argument("--repeats", type=int, default=1, help="Benchmark repeats.")
    benchmark_parser.add_argument(
        "--dataset-bytes",
        type=int,
        default=None,
        help="Optional dataset size for end-to-end tokenization time estimates.",
    )

    return parser


def _load_tokenizer_from_args(args) -> Tokenizer:
    """
    Construct a tokenizer from the file arguments shared by CLI subcommands.
    """
    return Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=args.special_tokens,
    )


def _run_train(args) -> None:
    """
    Execute the training command and print a compact metrics summary.
    """
    results = train_bpe_with_profile(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )

    save_vocab(results["vocab"], args.vocab_out)
    save_merges(results["merges"], args.merges_out)

    longest_token = results["longest_token"]
    assert isinstance(longest_token, bytes)

    print(f"input: {Path(args.input).resolve()}")
    print(f"vocab_size: {len(results['vocab'])}")
    print(f"num_merges: {len(results['merges'])}")
    print(f"elapsed_seconds: {results['elapsed_seconds']:.2f}")
    print(f"rss_before_mb: {results['rss_before_mb']}")
    print(f"rss_after_mb: {results['rss_after_mb']}")
    print(f"rss_delta_mb: {results['rss_delta_mb']}")
    print(f"longest_token_id: {results['longest_token_id']}")
    print(f"longest_token_bytes: {longest_token}")
    print(f"longest_token_utf8: {longest_token.decode('utf-8', errors='replace')}")
    print(f"vocab_out: {Path(args.vocab_out).resolve()}")
    print(f"merges_out: {Path(args.merges_out).resolve()}")
    print("\nTop profile summary:")
    print(results["profile_text"])


def _run_encode(args) -> None:
    """
    Execute the file-encoding command and write a uint16 `.npy` token-id array.
    """
    tokenizer = _load_tokenizer_from_args(args)
    output_path = encode_text_file_to_uint16(
        tokenizer=tokenizer,
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )
    print(f"encoded_input: {Path(args.input).resolve()}")
    print(f"token_ids_out: {output_path.resolve()}")


def _run_benchmark(args) -> None:
    """
    Execute the compression and throughput benchmark command on sampled documents.
    """
    tokenizer = _load_tokenizer_from_args(args)
    documents = sample_documents(
        input_path=args.input,
        num_documents=args.num_documents,
        document_delimiter=args.document_delimiter,
        seed=args.seed,
    )

    compression_ratio = compression_ratio_bytes_per_token(documents, tokenizer)
    throughput = estimate_tokenizer_throughput_bytes_per_second(documents, tokenizer, repeats=args.repeats)

    print(f"input: {Path(args.input).resolve()}")
    print(f"num_documents: {len(documents)}")
    print(f"compression_ratio_bytes_per_token: {compression_ratio:.4f}")
    print(f"throughput_bytes_per_second: {throughput:.2f}")

    if args.dataset_bytes is not None:
        estimated_seconds = estimate_tokenization_time_seconds(args.dataset_bytes, throughput)
        print(f"estimated_tokenization_seconds: {estimated_seconds:.2f}")


def main() -> None:
    """
    Dispatch the requested CLI subcommand.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _run_train(args)
        return
    if args.command == "encode":
        _run_encode(args)
        return
    if args.command == "benchmark":
        _run_benchmark(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")
