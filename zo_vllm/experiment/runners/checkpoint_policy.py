"""HF-like checkpoint policy helpers for runner checkpoint decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class BestMetricTracker:
    """Track the best eval row using HF Trainer-style metric semantics."""

    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool | None = None
    best_value: float | None = None
    best_record: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.greater_is_better is None:
            self.greater_is_better = infer_greater_is_better(self.metric_for_best_model)

    def update(
        self,
        eval_row: Mapping[str, Any],
        checkpoint_record: Mapping[str, Any] | None,
    ) -> bool:
        value = self.candidate_value(eval_row)
        if not self.is_better_value(value):
            return False
        self.best_value = float(value)
        self.best_record = {
            "metric": validate_metric_name(self.metric_for_best_model),
            "metric_value": float(value),
            "greater_is_better": bool(self.greater_is_better),
            "eval": dict(eval_row),
            "checkpoint": None
            if checkpoint_record is None
            else dict(checkpoint_record),
        }
        return True

    def candidate_value(self, eval_row: Mapping[str, Any]) -> float:
        return metric_value(eval_row, self.metric_for_best_model)

    def is_better(self, eval_row: Mapping[str, Any]) -> bool:
        return self.is_better_value(self.candidate_value(eval_row))

    def is_better_value(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if bool(self.greater_is_better):
            return float(value) > float(self.best_value)
        return float(value) < float(self.best_value)


@dataclass(frozen=True)
class RuntimeCheckpointSettings:
    """Resolved checkpoint settings for a training run."""

    load_best_model_at_end: bool
    effective_save_steps: int
    checkpoint_mode: str
    runtime_checkpoints_enabled: bool


def validate_metric_name(metric_name: str) -> str:
    name = str(metric_name)
    if not name:
        raise ValueError("metric_for_best_model must not be empty")
    return name


def infer_greater_is_better(metric_name: str) -> bool:
    name = validate_metric_name(metric_name).lower()
    return "loss" not in name


def metric_value(eval_row: Mapping[str, Any], metric_name: str) -> float:
    name = validate_metric_name(metric_name)
    if name not in eval_row:
        raise KeyError(f"evaluation result is missing configured metric: {name}")
    return float(eval_row[name])


def resolve_checkpoint_mode(
    *,
    save_checkpoint_mode: str,
    load_best_model_at_end: bool,
) -> str:
    if save_checkpoint_mode != "auto":
        return save_checkpoint_mode
    return "native" if load_best_model_at_end else "metadata"


def resolve_runtime_checkpoint_settings(
    *,
    args: Any,
    effective_eval_interval: int,
    effective_save_steps: int,
) -> RuntimeCheckpointSettings:
    """Validate and resolve runtime checkpoint policy settings."""

    load_best_model_at_end = bool(int(args.load_best_model_at_end))
    effective_save_steps = int(effective_save_steps)
    if effective_save_steps < 0:
        raise ValueError("--save-steps must be non-negative")
    if args.save_strategy == "best" and effective_eval_interval <= 0:
        raise ValueError("--save-strategy best requires evaluation to be enabled")
    if load_best_model_at_end and effective_eval_interval <= 0:
        raise ValueError("load_best_model_at_end requires evaluation to be enabled")
    if load_best_model_at_end and (
        args.save_strategy == "no"
        or (args.save_strategy == "steps" and effective_save_steps <= 0)
    ):
        raise ValueError(
            "load_best_model_at_end requires runtime checkpoints. Use "
            "--save-strategy steps with --save-steps > 0, or --save-strategy best."
        )
    if (
        load_best_model_at_end
        and args.save_strategy == "steps"
        and effective_save_steps > 0
        and effective_eval_interval > 0
        and effective_eval_interval % effective_save_steps != 0
    ):
        raise ValueError(
            "load_best_model_at_end with --save-strategy steps requires eval steps "
            "to also be checkpoint steps. Use --save-strategy best or choose "
            "--save-steps that divides the effective eval interval."
        )

    checkpoint_mode = resolve_checkpoint_mode(
        save_checkpoint_mode=args.save_checkpoint_mode,
        load_best_model_at_end=load_best_model_at_end,
    )
    if load_best_model_at_end and checkpoint_mode != "native":
        raise ValueError(
            "load_best_model_at_end requires a native checkpoint mode; "
            f"got {checkpoint_mode!r}"
        )
    runtime_checkpoints_enabled = args.save_strategy != "no" and (
        args.save_strategy == "best" or effective_save_steps > 0
    )
    return RuntimeCheckpointSettings(
        load_best_model_at_end=load_best_model_at_end,
        effective_save_steps=effective_save_steps,
        checkpoint_mode=checkpoint_mode,
        runtime_checkpoints_enabled=runtime_checkpoints_enabled,
    )
