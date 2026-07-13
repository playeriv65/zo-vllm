"""Task adapters for HF-dataset-backed ZO experiments."""

from .base import TaskConfig, TaskSplits, ZOTaskAdapter
from .boolq import BoolQRow, BoolQTaskAdapter, encode_boolq_vllm_prompts
from .registry import get_task, list_tasks
from .hf_preprocessing import (
    build_objective_hf_dataset,
    build_sst2_prompt_classification_preprocess,
)
from .squad import (
    SquadRow,
    SquadTaskAdapter,
    encode_squad_generation_prompts,
    encode_squad_train_prompts,
    squad_exact_match,
    squad_f1,
)
from .sst2 import SST2Row, SST2TaskAdapter, encode_sst2_vllm_prompts
from .superglue import (
    SUPERGLUE_OBJECTIVE_TO_TASK,
    SUPERGLUE_TASK_SPECS,
    SuperGLUEOptionEncoding,
    SuperGLUERow,
    SuperGLUETaskAdapter,
    dataset_to_superglue_rows,
    encode_superglue_option_prompts,
    superglue_prediction_metric,
)
from .tokenization import configure_opt_tokenizer, single_token_id

__all__ = [
    "BoolQRow",
    "BoolQTaskAdapter",
    "SUPERGLUE_OBJECTIVE_TO_TASK",
    "SUPERGLUE_TASK_SPECS",
    "SST2Row",
    "SST2TaskAdapter",
    "SquadRow",
    "SquadTaskAdapter",
    "SuperGLUEOptionEncoding",
    "SuperGLUERow",
    "SuperGLUETaskAdapter",
    "TaskConfig",
    "TaskSplits",
    "ZOTaskAdapter",
    "configure_opt_tokenizer",
    "build_objective_hf_dataset",
    "build_sst2_prompt_classification_preprocess",
    "dataset_to_superglue_rows",
    "encode_boolq_vllm_prompts",
    "encode_squad_generation_prompts",
    "encode_squad_train_prompts",
    "encode_sst2_vllm_prompts",
    "encode_superglue_option_prompts",
    "get_task",
    "list_tasks",
    "single_token_id",
    "squad_exact_match",
    "squad_f1",
    "superglue_prediction_metric",
]
