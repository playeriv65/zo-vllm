"""SuperGLUE BoolQ prompt adapter."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from .base import (
    SuperGLUETaskAdapter,
    SuperGLUERow,
    SuperGLUETaskSpec,
    TaskConfig,
    TaskSplits,
    ensure_question_mark,
    label_row,
    shuffled_select,
    split_train_dev,
)


TASK_NAME = "superglue_boolq"


def row_from_example(example: dict[str, Any]) -> SuperGLUERow:
    label_value = example["answer"] if "answer" in example else example["label"]
    return label_row(
        TASK_NAME,
        example,
        candidates=("Yes", "No"),
        label=0 if bool(label_value) else 1,
    )


def stem(row: SuperGLUERow) -> str:
    question = ensure_question_mark(str(row.data["question"]))
    return f"{str(row.data['passage'])} {question}\n"


def verbalized(row: SuperGLUERow, candidate: Any) -> str:
    return f"{stem(row)}{candidate}"


TASK_SPEC = SuperGLUETaskSpec(
    name=TASK_NAME,
    dataset_config="boolq",
    official_task_name="BoolQ",
    objective_name="superglue_boolq_classification",
    row_builder=row_from_example,
    stem=stem,
    verbalized=verbalized,
)


class BoolQTaskAdapter(SuperGLUETaskAdapter):
    def load_splits(self, cfg: TaskConfig) -> TaskSplits:
        raw = load_dataset("boolq")
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


TASK_ADAPTER = BoolQTaskAdapter(TASK_SPEC)
