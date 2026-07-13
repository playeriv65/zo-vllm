"""Registry for independently trainable SuperGLUE tasks."""

from __future__ import annotations

from collections.abc import Sequence

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase

from . import boolq, cb, copa, multirc, record, rte, wic, wsc
from .base import (
    SuperGLUEOptionEncoding,
    SuperGLUERow,
    SuperGLUETaskSpec,
    dataset_to_superglue_rows as _dataset_to_superglue_rows,
    encode_superglue_option_prompts as _encode_superglue_option_prompts,
)


SUPERGLUE_TASK_SPECS: dict[str, SuperGLUETaskSpec] = {
    spec.name: spec
    for spec in (
        boolq.TASK_SPEC,
        cb.TASK_SPEC,
        copa.TASK_SPEC,
        multirc.TASK_SPEC,
        record.TASK_SPEC,
        rte.TASK_SPEC,
        wic.TASK_SPEC,
        wsc.TASK_SPEC,
    )
}

SUPERGLUE_OBJECTIVE_TO_TASK = {
    spec.objective_name: task_name for task_name, spec in SUPERGLUE_TASK_SPECS.items()
}


def dataset_to_superglue_rows(
    dataset: HFDataset,
    *,
    task_name: str,
) -> list[SuperGLUERow]:
    return _dataset_to_superglue_rows(
        dataset,
        task_name=task_name,
        specs=SUPERGLUE_TASK_SPECS,
    )


def encode_superglue_option_prompts(
    rows: Sequence[SuperGLUERow],
    tokenizer: PreTrainedTokenizerBase,
) -> SuperGLUEOptionEncoding:
    return _encode_superglue_option_prompts(
        rows,
        tokenizer,
        specs=SUPERGLUE_TASK_SPECS,
    )


def superglue_prediction_metric(
    rows: Sequence[SuperGLUERow],
    predictions: Sequence[int],
) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0}
    task_names = {row.task_name for row in rows}
    if len(task_names) != 1:
        raise ValueError(f"cannot evaluate mixed SuperGLUE tasks: {sorted(task_names)}")
    task_name = next(iter(task_names))
    if task_name == record.TASK_NAME:
        return record.prediction_metric(rows, predictions)
    correct = 0
    for row, prediction in zip(rows, predictions):
        correct += int(int(prediction) in set(row.label_indices))
    return {"accuracy": float(correct / len(rows))}
