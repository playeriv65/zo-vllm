"""HF-style training arguments for ZO-vLLM loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zo_vllm.config import (
    DEFAULT_ZO_BATCH_SIZE,
    DEFAULT_ZO_EVAL_INTERVAL,
    DEFAULT_ZO_LEARNING_RATE,
    DEFAULT_ZO_SEED,
    DEFAULT_ZO_STEPS,
)

from .scheduler import ConstantLR, CosineAfterLR, LRScheduler


@dataclass(frozen=True)
class ZOTrainingArguments:
    """Small HF-like argument container for external ZO training loops."""

    output_dir: str
    max_steps: int = DEFAULT_ZO_STEPS
    warmup_steps: int = 0
    per_device_train_batch_size: int = DEFAULT_ZO_BATCH_SIZE
    per_device_eval_batch_size: int = DEFAULT_ZO_BATCH_SIZE
    learning_rate: float = DEFAULT_ZO_LEARNING_RATE
    lr_scheduler_type: str = "constant"
    lr_decay_start_step: int | None = None
    lr_final_scale: float = 0.25
    logging_steps: int = 50
    eval_steps: int = DEFAULT_ZO_EVAL_INTERVAL
    save_steps: int = 0
    save_strategy: str = "steps"
    save_total_limit: int | None = None
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool | None = None
    seed: int = DEFAULT_ZO_SEED
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    report_to: tuple[str, ...] = ()
    run_name: str | None = None

    def __post_init__(self) -> None:
        if int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if int(self.per_device_train_batch_size) <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if int(self.per_device_eval_batch_size) <= 0:
            raise ValueError("per_device_eval_batch_size must be positive")
        if float(self.learning_rate) < 0.0:
            raise ValueError("learning_rate must be non-negative")
        if int(self.logging_steps) <= 0:
            raise ValueError("logging_steps must be positive")
        if int(self.eval_steps) <= 0:
            raise ValueError("eval_steps must be positive")
        if int(self.save_steps) < 0:
            raise ValueError("save_steps must be non-negative")
        if self.save_strategy not in {"no", "steps", "best"}:
            raise ValueError("save_strategy must be one of: no, steps, best")
        if self.save_total_limit is not None and int(self.save_total_limit) <= 0:
            raise ValueError("save_total_limit must be positive when set")
        if bool(self.load_best_model_at_end) and self.save_strategy == "no":
            raise ValueError("load_best_model_at_end requires checkpoint saving")
        if bool(self.load_best_model_at_end) and (
            self.save_strategy == "steps" and int(self.save_steps) <= 0
        ):
            raise ValueError(
                "load_best_model_at_end with save_strategy='steps' requires "
                "save_steps > 0"
            )
        if bool(self.load_best_model_at_end) and (
            self.save_strategy == "steps"
            and int(self.eval_steps) % max(1, int(self.save_steps)) != 0
        ):
            raise ValueError(
                "load_best_model_at_end with save_strategy='steps' requires "
                "eval_steps to be a multiple of save_steps"
            )
        if int(self.dataloader_num_workers) < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def create_scheduler(self) -> LRScheduler:
        scheduler_type = self.lr_scheduler_type.lower()
        if scheduler_type == "constant":
            return ConstantLR(self.learning_rate)
        if scheduler_type == "cosine_after":
            decay_start = (
                int(self.lr_decay_start_step)
                if self.lr_decay_start_step is not None
                else max(1, int(0.4 * self.max_steps))
            )
            return CosineAfterLR(
                learning_rate=self.learning_rate,
                decay_start_step=decay_start,
                max_steps=self.max_steps,
                final_scale=self.lr_final_scale,
            )
        raise ValueError(f"unsupported lr_scheduler_type: {self.lr_scheduler_type}")
