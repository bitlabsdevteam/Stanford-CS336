from __future__ import annotations

import regex as re


# GPT-2 style pre-tokenization pattern used throughout the assignment. It splits text
# into whitespace-aware pieces before byte-level BPE merges are replayed.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Compile once at import time because the same pattern is reused during training,
# encoding, throughput benchmarking, and iterable-based corpus processing.
PRETOKEN_PATTERN = re.compile(PAT)

# The initial byte vocabulary always contains exactly the 256 possible byte values.
BYTE_TOKENS: tuple[bytes, ...] = tuple(bytes([i]) for i in range(256))


def compile_special_pattern(special_tokens: list[str] | None) -> re.Pattern[str] | None:
    """
    Compile a regex that matches any special token exactly.

    Every token is escaped before insertion so metacharacters such as `|` are treated
    literally. Tokens are ordered by decreasing length so that, when one token is a
    prefix of another, the longer token is matched first.
    """
    if not special_tokens:
        return None
    escaped = sorted((re.escape(token) for token in special_tokens), key=lambda token: (-len(token), token))
    return re.compile("|".join(escaped))


def split_on_special_tokens(text: str, special_pattern: re.Pattern[str] | None) -> list[str]:
    """
    Split text into ordinary segments using special tokens as hard delimiters.

    The delimiters themselves are removed from the returned list. This helper is used
    during training because special tokens must prevent merges across their spans while
    contributing nothing to merge statistics.
    """
    if special_pattern is None:
        return [text]
    return [segment for segment in special_pattern.split(text) if segment]


def split_with_special_tokens(
    text: str,
    special_pattern: re.Pattern[str] | None,
) -> list[tuple[bool, str]]:
    """
    Split text while preserving which pieces are special tokens.

    Returned tuples are:
    - `(False, ordinary_text)` for spans that should be pre-tokenized and BPE-encoded
    - `(True, special_token_text)` for spans that should be emitted as atomic tokens
    """
    if special_pattern is None:
        return [(False, text)] if text else []

    parts: list[tuple[bool, str]] = []
    start = 0
    for match in special_pattern.finditer(text):
        if match.start() > start:
            parts.append((False, text[start:match.start()]))
        parts.append((True, match.group(0)))
        start = match.end()
    if start < len(text):
        parts.append((False, text[start:]))
    return parts


def longest_special_prefix_suffix(text: str, special_tokens: list[str]) -> int:
    """
    Return the longest suffix of `text` that is a prefix of some special token.

    This is used by streaming tokenization so a partial special token at the end of a
    chunk is not emitted prematurely before the next chunk arrives.
    """
    best = 0
    for token in special_tokens:
        max_candidate = min(len(text), len(token) - 1)
        for prefix_length in range(max_candidate, best, -1):
            if text.endswith(token[:prefix_length]):
                best = prefix_length
                break
    return best
