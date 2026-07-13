"""Independently trainable SuperGLUE task adapters."""

from .base import (
    SuperGLUEOptionEncoding,
    SuperGLUERow,
    SuperGLUETaskAdapter,
    SuperGLUETaskSpec,
)
from .registry import (
    SUPERGLUE_OBJECTIVE_TO_TASK,
    SUPERGLUE_TASK_SPECS,
    dataset_to_superglue_rows,
    encode_superglue_option_prompts,
    superglue_prediction_metric,
)

__all__ = [
    "SUPERGLUE_OBJECTIVE_TO_TASK",
    "SUPERGLUE_TASK_SPECS",
    "SuperGLUEOptionEncoding",
    "SuperGLUERow",
    "SuperGLUETaskAdapter",
    "SuperGLUETaskSpec",
    "dataset_to_superglue_rows",
    "encode_superglue_option_prompts",
    "superglue_prediction_metric",
]
