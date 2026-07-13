"""SuperGLUE COPA prompt adapter."""

from __future__ import annotations

from typing import Any

from .base import SuperGLUERow, SuperGLUETaskSpec, label_row


TASK_NAME = "superglue_copa"


def row_from_example(example: dict[str, Any]) -> SuperGLUERow:
    return label_row(
        TASK_NAME,
        example,
        candidates=(str(example["choice1"]), str(example["choice2"])),
        label=example["label"],
    )


def _conjunction(row: SuperGLUERow) -> str:
    question = row.data["question"]
    if question == "effect":
        return " so "
    if question == "cause":
        return " because "
    raise ValueError(f"unsupported COPA question type: {question}")


def stem(row: SuperGLUERow) -> str:
    premise = str(row.data["premise"]).rstrip()
    if premise.endswith("."):
        premise = premise[:-1]
    return premise + _conjunction(row)


def _candidate_text(candidate: Any) -> str:
    words = str(candidate).split(" ")
    if words and words[0] != "I":
        words[0] = words[0].lower()
    return " ".join(words)


def verbalized(row: SuperGLUERow, candidate: Any) -> str:
    return stem(row) + _candidate_text(candidate)


TASK_SPEC = SuperGLUETaskSpec(
    name=TASK_NAME,
    dataset_config="copa",
    official_task_name="Copa",
    objective_name="superglue_copa_classification",
    row_builder=row_from_example,
    stem=stem,
    verbalized=verbalized,
    official_train_as_classification=False,
)
