from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TypeVar

import torch
from accelerate.data_loader import SeedableRandomSampler
from torch.utils.data import DataLoader, SequentialSampler


T = TypeVar("T")


def make_batches(
    items: Sequence[T],
    batch_size: int,
    *,
    sampler: str = "sequential",
    seed: int | None = None,
    drop_last: bool = True,
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
    num_batches = (
        len(ordered_items) // batch_size
        if drop_last
        else math.ceil(len(ordered_items) / batch_size)
    )
    return [
        ordered_items[idx * batch_size : (idx + 1) * batch_size]
        for idx in range(num_batches)
    ]


def batches_per_epoch(
    num_items: int, batch_size: int, *, drop_last: bool = False
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_items < 0:
        raise ValueError("num_items must be non-negative")
    if drop_last:
        return num_items // batch_size
    return math.ceil(num_items / batch_size)


def epoch_interval_to_steps(
    epoch_interval: float | None,
    *,
    num_items: int,
    batch_size: int,
    drop_last: bool = False,
) -> int:
    if epoch_interval is None or float(epoch_interval) <= 0.0:
        return 0
    per_epoch = batches_per_epoch(num_items, batch_size, drop_last=drop_last)
    if per_epoch <= 0:
        return 0
    return max(1, int(math.ceil(float(epoch_interval) * per_epoch)))


def make_train_dataloader(
    items: Sequence[T],
    batch_size: int,
    *,
    sampler: str = "sequential",
    seed: int | None = None,
    drop_last: bool = False,
) -> DataLoader[list[T]]:
    """Return row batches with HF Trainer-like sampler semantics."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = list(items)
    if sampler == "hf_random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        row_sampler = SeedableRandomSampler(
            dataset,
            generator=generator,
            data_seed=int(seed),
        )
    elif sampler == "sequential":
        row_sampler = SequentialSampler(dataset)
    else:
        raise ValueError(f"unknown train sampler: {sampler}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=row_sampler,
        collate_fn=lambda batch: batch,
        drop_last=drop_last,
    )
