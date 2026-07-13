"""SuperGLUE MultiRC prompt adapter."""

from __future__ import annotations

from typing import Any

from .base import SuperGLUERow, SuperGLUETaskSpec, label_row, yes_no


TASK_NAME = "superglue_multirc"


def row_from_example(example: dict[str, Any]) -> SuperGLUERow:
    return label_row(
        TASK_NAME,
        example,
        candidates=(0, 1),
        label=example["label"],
    )


def stem(row: SuperGLUERow) -> str:
    return (
        f"{row.data['paragraph']}\n"
        f"Question: {row.data['question']}\n"
        f"I found this answer \"{row.data['answer']}\". Is that correct? "
        "Yes or No?\n"
    )


def verbalized(row: SuperGLUERow, candidate: Any) -> str:
    return f"{stem(row)}{yes_no(candidate)}"


TASK_SPEC = SuperGLUETaskSpec(
    name=TASK_NAME,
    dataset_config="multirc",
    official_task_name="MultiRC",
    objective_name="superglue_multirc_classification",
    row_builder=row_from_example,
    stem=stem,
    verbalized=verbalized,
)
