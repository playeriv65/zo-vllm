"""SuperGLUE WSC.fixed prompt adapter."""

from __future__ import annotations

from typing import Any

from .base import SuperGLUERow, SuperGLUETaskSpec, label_row, yes_no


TASK_NAME = "superglue_wsc"


def row_from_example(example: dict[str, Any]) -> SuperGLUERow:
    return label_row(
        TASK_NAME,
        example,
        candidates=(0, 1),
        label=example["label"],
    )


def stem(row: SuperGLUERow) -> str:
    return (
        f"{row.data['text']}\nIn the previous sentence, does the pronoun "
        f"\"{str(row.data['span2_text']).lower()}\" refer to "
        f"{row.data['span1_text']}? Yes or No?\n"
    )


def verbalized(row: SuperGLUERow, candidate: Any) -> str:
    return f"{stem(row)}{yes_no(candidate)}"


TASK_SPEC = SuperGLUETaskSpec(
    name=TASK_NAME,
    dataset_config="wsc.fixed",
    official_task_name="WSC",
    objective_name="superglue_wsc_classification",
    row_builder=row_from_example,
    stem=stem,
    verbalized=verbalized,
)
