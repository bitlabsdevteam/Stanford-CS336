from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import time

import numpy as np

from .tokenizer import Tokenizer


def split_documents(text: str, document_delimiter: str) -> list[str]:
    """
    Split a corpus text into non-empty documents using the dataset's document delimiter.
    """
    return [document for document in text.split(document_delimiter) if document]


def sample_documents(
    input_path: str | Path,
    num_documents: int,
    document_delimiter: str,
    seed: int = 0,
) -> list[str]:
    """
    Sample a fixed number of documents from a corpus file for compression experiments.
    """
    text = Path(input_path).read_text(encoding="utf-8")
    documents = split_documents(text, document_delimiter)
    if num_documents > len(documents):
        raise ValueError("Requested more sampled documents than the corpus contains.")
    rng = random.Random(seed)
    return rng.sample(documents, num_documents)


def compression_ratio_bytes_per_token(documents: Iterable[str], tokenizer: Tokenizer) -> float:
    """
    Compute compression ratio as total UTF-8 bytes divided by total token count.
    """
    total_bytes = 0
    total_tokens = 0
    for document in documents:
        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(tokenizer.encode(document))
    if total_tokens == 0:
        raise ValueError("Compression ratio is undefined when the token count is zero.")
    return total_bytes / total_tokens


def estimate_tokenizer_throughput_bytes_per_second(
    documents: Iterable[str],
    tokenizer: Tokenizer,
    repeats: int = 1,
) -> float:
    """
    Estimate tokenizer throughput in bytes per second using repeated in-memory encoding.
    """
    document_list = list(documents)
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    total_bytes = sum(len(document.encode("utf-8")) for document in document_list) * repeats
    start = time.perf_counter()
    for _ in range(repeats):
        for document in document_list:
            tokenizer.encode(document)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        raise ValueError("Measured zero or negative elapsed time.")
    return total_bytes / elapsed


def estimate_tokenization_time_seconds(dataset_num_bytes: int, throughput_bytes_per_second: float) -> float:
    """
    Estimate total tokenization time from dataset size and measured throughput.
    """
    if dataset_num_bytes < 0:
        raise ValueError("dataset_num_bytes must be non-negative.")
    if throughput_bytes_per_second <= 0:
        raise ValueError("throughput_bytes_per_second must be positive.")
    return dataset_num_bytes / throughput_bytes_per_second


def iter_file_chunks(input_path: str | Path, chunk_size: int = 1 << 20):
    """
    Yield a UTF-8 text file in fixed-size chunks for streaming tokenization.
    """
    with Path(input_path).open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def count_encoded_tokens_for_file(
    tokenizer: Tokenizer,
    input_path: str | Path,
    chunk_size: int = 1 << 20,
) -> int:
    """
    Count how many token ids would be produced when encoding a text file.
    """
    return sum(1 for _ in tokenizer.encode_iterable(iter_file_chunks(input_path, chunk_size)))


def encode_text_file_to_uint16(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    chunk_size: int = 1 << 20,
) -> Path:
    """
    Encode a text file and serialize token ids as a NumPy `uint16` `.npy` array.

    Two-pass design:
    - first pass counts tokens so the output array shape is known
    - second pass writes token ids into a NumPy memory-mapped `.npy` file

    This keeps the final artifact deployment-friendly while avoiding a full Python list
    of token ids in memory.
    """
    total_tokens = count_encoded_tokens_for_file(tokenizer, input_path, chunk_size)
    if max(tokenizer.id_to_bytes, default=0) > 65535:
        raise ValueError("Vocabulary ids exceed uint16 range.")

    output = Path(output_path)
    token_array = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint16, shape=(total_tokens,))

    write_index = 0
    for token_id in tokenizer.encode_iterable(iter_file_chunks(input_path, chunk_size)):
        token_array[write_index] = token_id
        write_index += 1

    token_array.flush()
    return output
