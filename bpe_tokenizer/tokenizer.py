from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .patterns import PRETOKEN_PATTERN, compile_special_pattern, longest_special_prefix_suffix, split_with_special_tokens
from .serialization import load_merges, load_vocab
from .trainer import iter_pairs, merge_pair_in_sequence, pretoken_to_bytes


class Tokenizer:
    """
    Byte-level BPE tokenizer that replays a trained merge list over a fixed vocabulary.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Initialize the tokenizer and prepare all lookup tables needed for fast encoding.
        """
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens or [])

        # Earlier merges have higher priority during encoding, so the merge rank is
        # simply the pair's position in the learned merge list.
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        next_id = max(self.vocab, default=-1) + 1
        existing_values = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_values:
                self.vocab[next_id] = token_bytes
                existing_values.add(token_bytes)
                next_id += 1

        self.id_to_bytes: dict[int, bytes] = dict(self.vocab)
        self.bytes_to_id: dict[bytes, int] = {
            token_bytes: token_id for token_id, token_bytes in self.id_to_bytes.items()
        }
        self.special_pattern = compile_special_pattern(self.special_tokens)

        # Caching repeated pre-token encodings materially improves throughput on real
        # corpora because the same space-prefixed words recur very often.
        self._encode_cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Create a tokenizer from serialized vocabulary and merge artifacts on disk.
        """
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_merges(self, pretoken: str) -> list[bytes]:
        """
        Apply the earliest possible learned merge repeatedly until no merge remains.
        """
        tokens = list(pretoken_to_bytes(pretoken))
        while len(tokens) >= 2:
            best_rank: int | None = None
            best_pair: tuple[bytes, bytes] | None = None

            for pair in iter_pairs(tuple(tokens)):
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            merged_token = best_pair[0] + best_pair[1]
            tokens = list(merge_pair_in_sequence(tuple(tokens), best_pair, merged_token))
        return tokens

    def _encode_ordinary_text(self, text: str) -> list[int]:
        """
        Encode ordinary text that does not contain special-token spans.
        """
        ids: list[int] = []
        for match in PRETOKEN_PATTERN.finditer(text):
            pretoken = match.group(0)
            if pretoken in self._encode_cache:
                ids.extend(self._encode_cache[pretoken])
                continue

            token_bytes = self._apply_merges(pretoken)
            token_ids = [self.bytes_to_id[token] for token in token_bytes]
            self._encode_cache[pretoken] = token_ids
            ids.extend(token_ids)
        return ids

    def encode(self, text: str) -> list[int]:
        """
        Encode one in-memory string into integer token ids.
        """
        ids: list[int] = []
        for is_special, chunk in split_with_special_tokens(text, self.special_pattern):
            if not chunk:
                continue
            if is_special:
                ids.append(self.bytes_to_id[chunk.encode("utf-8")])
            else:
                ids.extend(self._encode_ordinary_text(chunk))
        return ids

    def _encode_ordinary_prefix_safely(self, text: str) -> tuple[list[int], str]:
        """
        Emit only the prefix of an ordinary-text buffer whose final pre-token is settled.
        """
        matches = list(PRETOKEN_PATTERN.finditer(text))
        if len(matches) <= 1:
            return [], text

        safe_end = matches[-1].start()
        emitted_ids = self._encode_ordinary_text(text[:safe_end])
        remaining_text = text[safe_end:]
        return emitted_ids, remaining_text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode a stream while preserving exact equivalence with `encode`.
        """
        buffer = ""

        for chunk in iterable:
            if not chunk:
                continue

            buffer += chunk

            while True:
                if self.special_pattern is None:
                    break

                match = self.special_pattern.search(buffer)
                if match is None:
                    break

                ordinary_prefix = buffer[:match.start()]
                if ordinary_prefix:
                    yield from self._encode_ordinary_text(ordinary_prefix)

                special_text = match.group(0)
                yield self.bytes_to_id[special_text.encode("utf-8")]
                buffer = buffer[match.end() :]

            if not buffer:
                continue

            special_suffix_len = longest_special_prefix_suffix(buffer, self.special_tokens)
            if special_suffix_len > 0:
                ordinary_candidate = buffer[:-special_suffix_len]
                special_candidate = buffer[-special_suffix_len:]
            else:
                ordinary_candidate = buffer
                special_candidate = ""

            emitted_ids, ordinary_remainder = self._encode_ordinary_prefix_safely(ordinary_candidate)
            for token_id in emitted_ids:
                yield token_id

            buffer = ordinary_remainder + special_candidate

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        """
        Decode token ids to text, replacing malformed UTF-8 byte sequences with U+FFFD.
        """
        try:
            raw_bytes = b"".join(self.id_to_bytes[token_id] for token_id in ids)
        except KeyError as exc:
            raise KeyError(f"Unknown token id during decode: {exc.args[0]}") from exc
        return raw_bytes.decode("utf-8", errors="replace")
