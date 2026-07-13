"""Hugging Face training arguments extended with ZO-vLLM controls."""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import TrainingArguments
from transformers.training_args import OptimizerNames


@dataclass
class ZOTrainerArguments(TrainingArguments):
    """TrainingArguments subclass for the Hugging Face ZO trainer adapter."""

    optim: str = field(
        default="sgd",
        metadata={"help": "ZO optimizer algorithm. Only exact ZO-SGD is supported."},
    )
    max_grad_norm: float = field(
        default=0.0,
        metadata={"help": "Autograd clipping is disabled for ZO estimates."},
    )

    use_cpu: bool = field(
        default=True,
        metadata={
            "help": "Keep Hugging Face orchestration on CPU; vLLM owns GPU devices."
        },
    )
    zo_checkpoint_mode: str = field(
        default="metadata",
        metadata={
            "help": (
                "ZO checkpoint payload mode. metadata writes trainer/runtime "
                "metadata only; native and lora are delegated to runtime "
                "checkpoint handlers."
            )
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Keep the complete ragged batch for the ZO runtime."},
    )
    zo_log_runtime_metrics: bool = field(
        default=True,
        metadata={"help": "Include numeric ZO runtime metrics in Hugging Face logs."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.optim != OptimizerNames.SGD:
            raise ValueError("ZOTrainer currently supports only optim='sgd'")
        if self.optim_args:
            raise ValueError("ZOTrainer does not support optim_args for ZO-SGD")
        if float(self.max_grad_norm) != 0.0:
            raise ValueError("ZOTrainer requires max_grad_norm=0")
        if self.zo_checkpoint_mode not in {"metadata", "native", "lora"}:
            raise ValueError(
                "zo_checkpoint_mode must be one of: metadata, native, lora"
            )
        if self.load_best_model_at_end and self.zo_checkpoint_mode != "native":
            raise ValueError(
                "load_best_model_at_end requires zo_checkpoint_mode='native' "
                "because metadata and LoRA-bank checkpoints are not full "
                "effective-model checkpoints"
            )
