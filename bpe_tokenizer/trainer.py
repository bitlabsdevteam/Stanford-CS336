from __future__ import annotations

import cProfile
from collections import Counter, defaultdict
import io
import multiprocessing as mp
from pathlib import Path
import pstats
import time

from .patterns import BYTE_TOKENS, PRETOKEN_PATTERN, compile_special_pattern, split_on_special_tokens


def pretoken_to_bytes(pretoken: str) -> tuple[bytes, ...]:
    """
    Convert one regex pre-token into a tuple of single-byte UTF-8 tokens.
    """
    return tuple(bytes([b]) for b in pretoken.encode("utf-8"))


def iter_pairs(tokens: tuple[bytes, ...]):
    """
    Yield every adjacent token pair in a sequence.
    """
    return zip(tokens, tokens[1:])


def merge_pair_in_sequence(
    tokens: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    """
    Replace every left-to-right non-overlapping occurrence of `pair` with `merged_token`.
    """
    merged: list[bytes] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(merged_token)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return tuple(merged)


def count_pretoken_sequences(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """
    Count unique byte-token sequences after splitting on special tokens and pre-tokenizing.
    """
    counts: Counter[tuple[bytes, ...]] = Counter()
    special_pattern = compile_special_pattern(special_tokens)

    for segment in split_on_special_tokens(text, special_pattern):
        for match in PRETOKEN_PATTERN.finditer(segment):
            pretoken = match.group(0)
            if pretoken:
                counts[pretoken_to_bytes(pretoken)] += 1
    return counts


def _count_pretoken_sequences_in_chunk(
    chunk_text: str,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    """
    Worker-safe wrapper for counting pre-token sequences inside one chunk.
    """
    return count_pretoken_sequences(chunk_text, special_tokens)


def merge_counters(counters: list[Counter[tuple[bytes, ...]]]) -> Counter[tuple[bytes, ...]]:
    """
    Merge many `Counter` objects into a single corpus-wide counter.
    """
    merged: Counter[tuple[bytes, ...]] = Counter()
    for counter in counters:
        merged.update(counter)
    return merged


def compute_chunk_boundaries(
    text: str,
    boundary_token: str,
    num_chunks: int,
) -> list[tuple[int, int]]:
    """
    Compute chunk boundaries whose starts align with a special token occurrence.

    This keeps document boundaries intact during multiprocessing pre-tokenization.
    """
    if num_chunks <= 1 or not text or not boundary_token:
        return [(0, len(text))]

    text_length = len(text)
    boundaries = [0]

    for chunk_index in range(1, num_chunks):
        target = (text_length * chunk_index) // num_chunks
        next_boundary = text.find(boundary_token, target)
        if next_boundary == -1 or next_boundary <= boundaries[-1]:
            break
        boundaries.append(next_boundary)

    boundaries.append(text_length)
    return list(zip(boundaries, boundaries[1:]))


def count_pretoken_sequences_parallel(
    text: str,
    special_tokens: list[str],
    num_processes: int,
) -> Counter[tuple[bytes, ...]]:
    """
    Count pre-token sequences using multiprocessing when safe document boundaries exist.
    """
    if num_processes <= 1 or not special_tokens:
        return count_pretoken_sequences(text, special_tokens)

    chunks = compute_chunk_boundaries(text, special_tokens[0], num_processes)
    if len(chunks) <= 1:
        return count_pretoken_sequences(text, special_tokens)

    chunk_payloads = [text[start:end] for start, end in chunks if start < end]
    if len(chunk_payloads) <= 1:
        return count_pretoken_sequences(text, special_tokens)

    with mp.Pool(processes=min(num_processes, len(chunk_payloads))) as pool:
        partial_counts = pool.starmap(
            _count_pretoken_sequences_in_chunk,
            [(chunk_text, special_tokens) for chunk_text in chunk_payloads],
        )
    return merge_counters(partial_counts)


def build_pair_statistics(sequence_counts: Counter[tuple[bytes, ...]]):
    """
    Build global pair counts plus a reverse index from pair to sequences containing it.
    """
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_sequences: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for tokens, seq_count in sequence_counts.items():
        local_pairs = Counter(iter_pairs(tokens))
        for pair, multiplicity in local_pairs.items():
            pair_counts[pair] += multiplicity * seq_count
            pair_to_sequences[pair].add(tokens)

    return pair_counts, pair_to_sequences


def remove_sequence_from_index(
    tokens: tuple[bytes, ...],
    seq_count: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_sequences: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    """
    Remove one sequence's pair contributions from the global indexes.
    """
    for pair, multiplicity in Counter(iter_pairs(tokens)).items():
        pair_counts[pair] -= multiplicity * seq_count
        if pair_counts[pair] <= 0:
            pair_counts.pop(pair, None)
        bucket = pair_to_sequences.get(pair)
        if bucket is not None:
            bucket.discard(tokens)
            if not bucket:
                pair_to_sequences.pop(pair, None)


def add_sequence_to_index(
    tokens: tuple[bytes, ...],
    seq_count: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_sequences: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    """
    Add one sequence's pair contributions back into the global indexes.
    """
    for pair, multiplicity in Counter(iter_pairs(tokens)).items():
        pair_counts[pair] += multiplicity * seq_count
        pair_to_sequences[pair].add(tokens)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer from a UTF-8 text file.

    The returned vocabulary contains, in order:
    - all special tokens supplied by the caller
    - the 256 single-byte base vocabulary
    - every merged token learned during training
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if num_processes <= 0:
        raise ValueError("num_processes must be positive.")

    text = Path(input_path).read_text(encoding="utf-8")
    sequence_counts = count_pretoken_sequences_parallel(text, special_tokens, num_processes)
    initial_vocab_size = len(special_tokens) + len(BYTE_TOKENS)

    if vocab_size < initial_vocab_size:
        raise ValueError(
            f"vocab_size={vocab_size} is too small; need at least {initial_vocab_size} "
            "to include special tokens and the 256 byte tokens."
        )

    vocab: dict[int, bytes] = {}
    next_id = 0

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    for token_bytes in BYTE_TOKENS:
        vocab[next_id] = token_bytes
        next_id += 1

    vocab_values = set(vocab.values())
    merges: list[tuple[bytes, bytes]] = []
    pair_counts, pair_to_sequences = build_pair_statistics(sequence_counts)

    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        if merged_token not in vocab_values:
            vocab[next_id] = merged_token
            vocab_values.add(merged_token)
            next_id += 1

        affected_sequences = list(pair_to_sequences.get(best_pair, ()))
        if not affected_sequences:
            pair_counts.pop(best_pair, None)
            continue

        for old_tokens in affected_sequences:
            seq_count = sequence_counts.pop(old_tokens)
            remove_sequence_from_index(old_tokens, seq_count, pair_counts, pair_to_sequences)
            new_tokens = merge_pair_in_sequence(old_tokens, best_pair, merged_token)
            sequence_counts[new_tokens] += seq_count
            add_sequence_to_index(new_tokens, seq_count, pair_counts, pair_to_sequences)

    return vocab, merges


def longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes]:
    """
    Return the longest token in a vocabulary, breaking ties lexicographically.
    """
    return max(vocab.items(), key=lambda item: (len(item[1]), item[1]))


def current_rss_mb() -> float | None:
    """
    Return current process resident memory in megabytes when `psutil` is available.
    """
    try:
        import psutil
    except ImportError:
        return None

    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def train_bpe_with_profile(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
) -> dict[str, object]:
    """
    Train BPE and return a metrics bundle suitable for reporting and deployment logs.
    """
    profiler = cProfile.Profile()
    rss_before_mb = current_rss_mb()
    start = time.perf_counter()

    profiler.enable()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes,
    )
    profiler.disable()

    elapsed_seconds = time.perf_counter() - start
    rss_after_mb = current_rss_mb()
    longest_token_id, longest_token_bytes = longest_token(vocab)

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)

    rss_delta_mb = None
    if rss_before_mb is not None and rss_after_mb is not None:
        rss_delta_mb = rss_after_mb - rss_before_mb

    return {
        "vocab": vocab,
        "merges": merges,
        "elapsed_seconds": elapsed_seconds,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "rss_delta_mb": rss_delta_mb,
        "longest_token_id": longest_token_id,
        "longest_token": longest_token_bytes,
        "profile_text": stream.getvalue(),
    }
