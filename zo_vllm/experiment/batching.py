from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import torch


T = TypeVar("T")


def make_batches(
    items: Sequence[T],
    batch_size: int,
    *,
    sampler: str = "sequential",
    seed: int | None = None,
) -> list[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered_items = list(items)
    if sampler == "hf_random":
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        order = torch.randperm(len(ordered_items), generator=generator).tolist()
        ordered_items = [ordered_items[idx] for idx in order]
    elif sampler != "sequential":
        raise ValueError(f"unknown train sampler: {sampler}")
    num_batches = len(ordered_items) // batch_size
    return [
        ordered_items[idx * batch_size : (idx + 1) * batch_size]
        for idx in range(num_batches)
    ]
