"""Shared SuperGLUE adapter primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from zo_vllm.core.binary_option_objective import multi_option_lengths
from zo_vllm.tasks.base import (
    TaskConfig,
    TaskSplits,
    build_task_data_collator,
    shuffled_select,
)


@dataclass(frozen=True)
class SuperGLUERow:
    task_name: str
    data: dict[str, Any]
    candidates: tuple[Any, ...]
    label_indices: tuple[int, ...]
    answers: tuple[str, ...] = ()
    idx: int | None = None

    @property
    def label(self) -> int:
        if not self.label_indices:
            raise ValueError(f"{self.task_name} row has no gold label")
        return int(self.label_indices[0])


@dataclass(frozen=True)
class SuperGLUEOptionEncoding:
    stem_ids: list[list[int]]
    option_ids: list[list[list[int]]]
    option_lens: list[list[int]]
    labels: list[int]
    label_sets: list[tuple[int, ...]]
    candidates: list[tuple[Any, ...]]
    answers: list[tuple[str, ...]]

    @property
    def option_counts(self) -> list[int]:
        return [len(options) for options in self.option_ids]

    @property
    def flat_option_ids(self) -> list[list[int]]:
        return [ids for row_options in self.option_ids for ids in row_options]

    @property
    def flat_option_lens(self) -> list[int]:
        return [length for row_lens in self.option_lens for length in row_lens]


@dataclass(frozen=True)
class SuperGLUETaskSpec:
    name: str
    dataset_config: str
    official_task_name: str
    objective_name: str
    row_builder: Callable[[dict[str, Any]], SuperGLUERow]
    stem: Callable[[SuperGLUERow], str]
    verbalized: Callable[[SuperGLUERow, Any], str]
    official_train_as_classification: bool = True


class SuperGLUETaskAdapter:
    metric_names = ("eval_loss", "eval_accuracy")
    primary_metric = "eval_loss"
    greater_is_better = False
    supports_official_lozo = True

    def __init__(self, spec: SuperGLUETaskSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.official_task_name = spec.official_task_name
        self.vllm_train_objective = spec.objective_name

    def load_splits(self, cfg: TaskConfig) -> TaskSplits:
        raw = load_dataset("super_glue", self.spec.dataset_config)
        train_dev = shuffled_select(
            raw["train"],
            seed=cfg.data_seed,
            num=int(cfg.num_train) + int(cfg.num_dev),
            shuffle_impl=cfg.shuffle_impl,
        )
        train, dev = split_train_dev(
            train_dev,
            num_train=int(cfg.num_train),
            num_dev=int(cfg.num_dev),
        )
        eval_split = shuffled_select(
            raw["validation"],
            seed=cfg.data_seed,
            num=cfg.num_eval,
            shuffle_impl=cfg.shuffle_impl,
        )
        return TaskSplits(train=train, dev=dev, eval=eval_split)

    def official_lozo_args(self, cfg: TaskConfig) -> list[str]:
        if self.spec.official_train_as_classification:
            return ["--train_as_classification"]
        return ["--train_as_classification", "False"]

    def vllm_args(self, cfg: TaskConfig) -> list[str]:
        return ["--train-objective", self.vllm_train_objective]

    def data_collator(self, tokenizer: Any, cfg: TaskConfig | None = None):
        return build_task_data_collator(
            tokenizer,
            objective_name=self.vllm_train_objective,
            row_converter=lambda rows: dataset_to_superglue_rows(
                rows,
                task_name=self.name,
                specs={self.name: self.spec},
            ),
            cfg=cfg,
        )


def split_train_dev(dataset: HFDataset, *, num_train: int, num_dev: int) -> tuple[HFDataset, HFDataset]:
    requested = int(num_train) + int(num_dev)
    if len(dataset) < requested:
        dev_count = min(int(num_dev), len(dataset))
        dev_start = max(0, len(dataset) - dev_count)
    else:
        dev_start = min(int(num_train), len(dataset))
    dev_end = min(dev_start + int(num_dev), len(dataset))
    train = select_range(dataset, 0, dev_start)
    dev = select_range(dataset, dev_start, dev_end)
    return train, dev


def select_range(dataset: HFDataset, start: int, end: int) -> HFDataset:
    if end <= start:
        return dataset.select([])
    return dataset.select(range(start, end))


def ensure_question_mark(question: str) -> str:
    question = question.strip()
    if not question.endswith("?"):
        question = question + "?"
    return question[0].upper() + question[1:] if question else question


def example_idx(example: dict[str, Any]) -> int | None:
    if "idx" not in example or example["idx"] is None:
        return None
    try:
        return int(example["idx"])
    except (TypeError, ValueError):
        return None


def label_row(
    task_name: str,
    example: dict[str, Any],
    *,
    candidates: Sequence[Any],
    label: int,
) -> SuperGLUERow:
    return SuperGLUERow(
        task_name=task_name,
        data=dict(example),
        candidates=tuple(candidates),
        label_indices=(int(label),),
        idx=example_idx(example),
    )


def yes_no(candidate: Any) -> str:
    return {0: "No", 1: "Yes"}[int(candidate)]


def dataset_to_superglue_rows(
    dataset: HFDataset,
    *,
    task_name: str,
    specs: dict[str, SuperGLUETaskSpec],
) -> list[SuperGLUERow]:
    spec = specs[task_name]
    return [spec.row_builder(dict(item)) for item in dataset]


def encode_superglue_option_prompts(
    rows: Sequence[SuperGLUERow],
    tokenizer: PreTrainedTokenizerBase,
    *,
    specs: dict[str, SuperGLUETaskSpec],
) -> SuperGLUEOptionEncoding:
    if not rows:
        return SuperGLUEOptionEncoding([], [], [], [], [], [], [])
    task_names = {row.task_name for row in rows}
    if len(task_names) != 1:
        raise ValueError(f"cannot encode mixed SuperGLUE tasks: {sorted(task_names)}")
    spec = specs[next(iter(task_names))]
    stem_ids = [
        tokenizer.encode(spec.stem(row).strip(" "), add_special_tokens=True)
        for row in rows
    ]
    option_ids = [
        [
            tokenizer.encode(
                spec.verbalized(row, candidate).strip(" "),
                add_special_tokens=True,
            )
            for candidate in row.candidates
        ]
        for row in rows
    ]
    option_lens = multi_option_lengths(stem_ids, option_ids)
    return SuperGLUEOptionEncoding(
        stem_ids=stem_ids,
        option_ids=option_ids,
        option_lens=option_lens,
        labels=[row.label for row in rows],
        label_sets=[tuple(int(value) for value in row.label_indices) for row in rows],
        candidates=[tuple(row.candidates) for row in rows],
        answers=[tuple(row.answers) for row in rows],
    )
