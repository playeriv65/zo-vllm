"""Common task adapter interfaces for Phase 4 style experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from typing import Any, Protocol

import numpy as np
from datasets import Dataset

from zo_vllm.config import DEFAULT_ZO_TASK_SHUFFLE_IMPL

ZO_TASK_SHUFFLE_IMPL_ENV = "ZO_TASK_SHUFFLE_IMPL"


@dataclass(frozen=True)
class TaskConfig:
    """Dataset and task-level settings, separate from optimizer settings."""

    name: str
    num_train: int
    num_dev: int
    num_eval: int | None
    data_seed: int
    template: str = "default"
    max_length: int = 2048
    max_new_tokens: int = 50


@dataclass(frozen=True)
class TaskSplits:
    """HF datasets used by a training job."""

    train: Dataset
    dev: Dataset
    eval: Dataset


class ZOTaskAdapter(Protocol):
    """Task-owned dataset, objective, and metric contract."""

    name: str
    official_task_name: str
    vllm_train_objective: str
    metric_names: tuple[str, ...]
    primary_metric: str
    greater_is_better: bool
    supports_official_lozo: bool

    def load_splits(self, cfg: TaskConfig) -> TaskSplits:
        """Load and sample HF Dataset splits for this task."""

    def official_lozo_args(self, cfg: TaskConfig) -> list[str]:
        """Return extra official LOZO CLI flags for this task."""

    def vllm_args(self, cfg: TaskConfig) -> list[str]:
        """Return extra vLLM runner CLI flags for this task."""

    def data_collator(
        self,
        tokenizer: Any,
        cfg: TaskConfig | None = None,
    ) -> Callable[[list[Any]], Any]:
        """Return an HF-style collator for direct VLLMZOTrainer integration."""


def build_task_data_collator(
    tokenizer: Any,
    *,
    objective_name: str,
    row_converter: Callable[[Sequence[Mapping[str, Any]]], list[Any]],
    cfg: TaskConfig | None = None,
) -> Callable[[list[Any]], Any]:
    """Build a task-owned HF-row collator backed by the shared ZO encoder."""

    from zo_vllm.training.task_encoding import (  # noqa: PLC0415
        ZOTaskDataCollator,
        ZOTaskEncodingConfig,
    )

    task_cfg = cfg or TaskConfig(
        name=objective_name,
        num_train=0,
        num_dev=0,
        num_eval=0,
        data_seed=0,
    )
    collator = ZOTaskDataCollator(
        tokenizer=tokenizer,
        config=ZOTaskEncodingConfig(
            objective_name=objective_name,
            max_length=task_cfg.max_length,
            max_new_tokens=task_cfg.max_new_tokens,
        ),
    )

    def _collate(rows: list[Any]):
        if not rows:
            return collator([])
        if isinstance(rows[0], Mapping):
            return collator(row_converter(rows))
        return collator(rows)

    return _collate


def shuffled_select(dataset: Dataset, *, seed: int, num: int | None) -> Dataset:
    """Return a deterministic shuffled subset using the configured algorithm."""
    shuffle_impl = (
        os.environ.get(ZO_TASK_SHUFFLE_IMPL_ENV, DEFAULT_ZO_TASK_SHUFFLE_IMPL)
        .strip()
        .lower()
    )
    if shuffle_impl in {"hf", "datasets"}:
        shuffled = dataset.shuffle(seed=int(seed))
        if num is None:
            return shuffled
        return shuffled.select(range(min(int(num), len(shuffled))))
    if shuffle_impl not in {"numpy", "np", "lozo"}:
        raise ValueError(f"unknown {ZO_TASK_SHUFFLE_IMPL_ENV}: {shuffle_impl}")
    rng = np.random.RandomState(int(seed))
    indices = rng.permutation(len(dataset)).tolist()
    if num is None:
        return dataset.select(indices)
    return dataset.select(indices[: min(int(num), len(indices))])
