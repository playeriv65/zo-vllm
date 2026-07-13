"""BoolQ task adapter and prompt encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from .base import TaskConfig, TaskSplits, build_task_data_collator, shuffled_select


@dataclass(frozen=True)
class BoolQRow:
    passage: str
    question: str
    label: int


class BoolQTaskAdapter:
    name = "boolq"
    official_task_name = "BoolQ"
    vllm_train_objective = "boolq_classification"
    metric_names = ("eval_loss", "eval_accuracy")
    primary_metric = "eval_loss"
    greater_is_better = False
    supports_official_lozo = True

    def load_splits(self, cfg: TaskConfig) -> TaskSplits:
        raw = load_dataset("boolq")
        train_dev = shuffled_select(
            raw["train"],
            seed=cfg.data_seed,
            num=int(cfg.num_train) + int(cfg.num_dev),
        )
        train = train_dev.select(range(min(int(cfg.num_train), len(train_dev))))
        dev_start = min(int(cfg.num_train), len(train_dev))
        dev_end = min(dev_start + int(cfg.num_dev), len(train_dev))
        dev = train_dev.select(range(dev_start, dev_end))
        eval_split = shuffled_select(
            raw["validation"],
            seed=cfg.data_seed,
            num=cfg.num_eval,
        )
        return TaskSplits(train=train, dev=dev, eval=eval_split)

    def official_lozo_args(self, cfg: TaskConfig) -> list[str]:
        return ["--train_as_classification"]

    def vllm_args(self, cfg: TaskConfig) -> list[str]:
        return ["--train-objective", self.vllm_train_objective]

    def data_collator(self, tokenizer: Any, cfg: TaskConfig | None = None):
        return build_task_data_collator(
            tokenizer,
            objective_name=self.vllm_train_objective,
            row_converter=dataset_to_boolq_rows,
            cfg=cfg,
        )


def boolq_stem(row: BoolQRow) -> str:
    question = row.question.strip()
    if not question.endswith("?"):
        question = question + "?"
    if question:
        question = question[0].upper() + question[1:]
    return f"{row.passage.strip()} {question}\n"


def boolq_verbalized(row: BoolQRow, candidate: int) -> str:
    verbalizer = {0: "No", 1: "Yes"}
    return f"{boolq_stem(row)}{verbalizer[int(candidate)]}"


def dataset_to_boolq_rows(dataset: HFDataset) -> list[BoolQRow]:
    """Convert HF Dataset rows into BoolQ scoring rows."""

    return [
        BoolQRow(
            passage=str(item["passage"]),
            question=str(item["question"]),
            label=1 if bool(item["answer"]) else 0,
        )
        for item in dataset
    ]


def encode_boolq_vllm_prompts(
    rows: list[BoolQRow],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[int]]:
    stems = [tokenizer.encode(boolq_stem(row), add_special_tokens=True) for row in rows]
    no_ids = [
        tokenizer.encode(boolq_verbalized(row, 0), add_special_tokens=True)
        for row in rows
    ]
    yes_ids = [
        tokenizer.encode(boolq_verbalized(row, 1), add_special_tokens=True)
        for row in rows
    ]
    labels = [int(row.label) for row in rows]
    return stems, no_ids, yes_ids, labels
