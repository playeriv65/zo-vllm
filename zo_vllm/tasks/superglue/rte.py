"""SuperGLUE RTE prompt adapter."""

from __future__ import annotations

from typing import Any

from .base import SuperGLUERow, SuperGLUETaskSpec, label_row


TASK_NAME = "superglue_rte"
VERBALIZER = {0: "Yes", 1: "No"}


def row_from_example(example: dict[str, Any]) -> SuperGLUERow:
    return label_row(
        TASK_NAME,
        example,
        candidates=(0, 1),
        label=example["label"],
    )


def stem(row: SuperGLUERow) -> str:
    return (
        f"{row.data['premise']}\nDoes this mean that "
        f"\"{row.data['hypothesis']}\" is true? Yes or No?\n"
    )


def verbalized(row: SuperGLUERow, candidate: Any) -> str:
    return f"{stem(row)}{VERBALIZER[int(candidate)]}"


TASK_SPEC = SuperGLUETaskSpec(
    name=TASK_NAME,
    dataset_config="rte",
    official_task_name="RTE",
    objective_name="superglue_rte_classification",
    row_builder=row_from_example,
    stem=stem,
    verbalized=verbalized,
)
