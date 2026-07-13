"""SuperGLUE CB prompt adapter."""

from __future__ import annotations

from typing import Any

from .base import SuperGLUERow, SuperGLUETaskSpec, label_row


TASK_NAME = "superglue_cb"
VERBALIZER = {0: "Yes", 1: "No", 2: "Maybe"}


def row_from_example(example: dict[str, Any]) -> SuperGLUERow:
    return label_row(
        TASK_NAME,
        example,
        candidates=(0, 1, 2),
        label=example["label"],
    )


def stem(row: SuperGLUERow) -> str:
    return (
        f"Suppose {row.data['premise']} Can we infer that "
        f"\"{row.data['hypothesis']}\"? Yes, No, or Maybe?\n"
    )


def verbalized(row: SuperGLUERow, candidate: Any) -> str:
    return f"{stem(row)}{VERBALIZER[int(candidate)]}"


TASK_SPEC = SuperGLUETaskSpec(
    name=TASK_NAME,
    dataset_config="cb",
    official_task_name="CB",
    objective_name="superglue_cb_classification",
    row_builder=row_from_example,
    stem=stem,
    verbalized=verbalized,
)
