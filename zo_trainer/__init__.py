"""Hugging Face Trainer integration for ZO-vLLM."""

from .arguments import ZOTrainerArguments
from .callbacks import StopOnSignalCallback, ZOUSnapshotCallback
from .checkpointing import (
    ZO_CHECKPOINT_METADATA_NAME,
    ZO_CHECKPOINT_METADATA_VERSION,
    ZO_DIRECTION_STATE_NAME,
    ZOCheckpointHandler,
    ZOCheckpointMetadata,
    read_zo_checkpoint_metadata,
    write_zo_checkpoint_metadata,
)
from .modeling import (
    CompactCausalOutput,
    OptionClassificationOutput,
    ZOLogitsOutput,
    ZOTrainerModel,
    ZOTrainerRuntime,
)
from .optimizer import ZOSGDOptimizer
from .preprocessing import (
    IGNORE_INDEX,
    VLLMDataCollator,
    build_causal_lm_preprocess,
    build_target_lm_preprocess,
)
from .runtime import ZOVLLMCheckpointHandler
from .trainer import ZOTrainer

__all__ = [
    "IGNORE_INDEX",
    "CompactCausalOutput",
    "ZO_CHECKPOINT_METADATA_NAME",
    "ZO_CHECKPOINT_METADATA_VERSION",
    "ZO_DIRECTION_STATE_NAME",
    "ZOCheckpointHandler",
    "ZOCheckpointMetadata",
    "OptionClassificationOutput",
    "StopOnSignalCallback",
    "ZOLogitsOutput",
    "VLLMDataCollator",
    "ZOVLLMCheckpointHandler",
    "ZOTrainer",
    "ZOTrainerArguments",
    "ZOTrainerModel",
    "ZOTrainerRuntime",
    "ZOUSnapshotCallback",
    "ZOSGDOptimizer",
    "build_causal_lm_preprocess",
    "build_target_lm_preprocess",
    "read_zo_checkpoint_metadata",
    "write_zo_checkpoint_metadata",
]
