"""Dataset helpers for TinyStories diffusion-language-model notebooks."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


def format_as_chat(story_text: str) -> str:
    """Wrap a TinyStories example in a simple chat-style training prompt."""
    story_text = story_text.strip()
    return f"<|user|>\nWrite a short story.\n<|assistant|>\n{story_text}\n<|end|>\n"


class TokenBlockDataset(Dataset):
    """Tokenize each example into one bounded training block."""

    def __init__(
        self,
        dataset,
        tokenizer,
        seq_len: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.indices = list(range(len(dataset)))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.indices)

    def __len__(self) -> int:
        """Return the number of examples available to the notebook."""
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenize one example and return tensors ready for collation."""
        example = self.dataset[self.indices[index]]
        text = format_as_chat(example["text"])
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.seq_len,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.bool),
        }


def collate_blocks(
    batch: list[dict[str, torch.Tensor]],
    *,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length token blocks to the local max length."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    attention_mask = []
    for item in batch:
        length = item["input_ids"].size(0)
        pad_len = max_len - length
        if pad_len > 0:
            input_ids.append(
                torch.cat(
                    [item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]
                )
            )
            attention_mask.append(
                torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.bool)])
            )
        else:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }
