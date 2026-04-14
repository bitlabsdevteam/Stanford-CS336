from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch


@contextmanager
def range(name: str) -> Iterator[None]:
    """
    Emit an NVTX range when CUDA profiling is available, otherwise act as a no-op.
    """
    if torch.cuda.is_available():
        with torch.cuda.nvtx.range(name):
            yield
        return

    yield
