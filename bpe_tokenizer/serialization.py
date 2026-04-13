from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def load_vocab(vocab_path: str | Path) -> dict[int, bytes]:
    """
    Load a vocabulary JSON file where values are stored as hexadecimal byte strings.
    """
    payload = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    return {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in payload.items()}


def save_vocab(vocab: dict[int, bytes], output_path: str | Path) -> None:
    """
    Save a vocabulary as JSON with hexadecimal byte-string payloads.

    Hex encoding makes arbitrary bytes round-trip safely through JSON.
    """
    payload = {str(token_id): token_bytes.hex() for token_id, token_bytes in sorted(vocab.items())}
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_merges(merges_path: str | Path) -> list[tuple[bytes, bytes]]:
    """
    Load merge pairs from a tab-separated text file in creation order.
    """
    merges: list[tuple[bytes, bytes]] = []
    for line in Path(merges_path).read_text(encoding="utf-8").splitlines():
        left_hex, right_hex = line.split("\t")
        merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
    return merges


def save_merges(merges: list[tuple[bytes, bytes]], output_path: str | Path) -> None:
    """
    Save merge pairs as tab-separated hexadecimal byte strings, one merge per line.
    """
    lines = [f"{left.hex()}\t{right.hex()}" for left, right in merges]
    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_token_ids_uint16(token_ids: list[int] | np.ndarray, output_path: str | Path) -> None:
    """
    Save token ids as a NumPy `uint16` array.

    `uint16` is appropriate when every token id is below 65536, which comfortably holds
    the assignment's 10K and 32K vocabularies while using half the space of `uint32`.
    """
    token_array = np.asarray(token_ids)
    if token_array.size == 0:
        np.save(output_path, token_array.astype(np.uint16))
        return
    if int(token_array.max()) > 65535:
        raise ValueError("Token ids exceed uint16 range.")
    np.save(output_path, token_array.astype(np.uint16, copy=False))
