"""SuperGLUE WiC prompt adapter."""

from __future__ import annotations

from typing import Any

from .base import SuperGLUERow, SuperGLUETaskSpec, label_row, yes_no


TASK_NAME = "superglue_wic"


def row_from_example(example: dict[str, Any]) -> SuperGLUERow:
    return label_row(
        TASK_NAME,
        example,
        candidates=(0, 1),
        label=example["label"],
    )


def stem(row: SuperGLUERow) -> str:
    return (
        f"Does the word \"{row.data['word']}\" have the same meaning in these "
        f"two sentences? Yes, No?\n{row.data['sentence1']}\n"
        f"{row.data['sentence2']}\n"
    )


def verbalized(row: SuperGLUERow, candidate: Any) -> str:
    return f"{stem(row)}{yes_no(candidate)}"


TASK_SPEC = SuperGLUETaskSpec(
    name=TASK_NAME,
    dataset_config="wic",
    official_task_name="WIC",
    objective_name="superglue_wic_classification",
    row_builder=row_from_example,
    stem=stem,
    verbalized=verbalized,
)
