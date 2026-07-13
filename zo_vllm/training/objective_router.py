"""Objective routing for training runner logging and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from zo_vllm.core.direct_worker_scorer import score_direct_worker_detailed
from zo_vllm.tasks.superglue import (
    SUPERGLUE_OBJECTIVE_TO_TASK,
    superglue_prediction_metric,
)
from zo_vllm.tasks.superglue.record import OBJECTIVE_NAME as RECORD_NLL_OBJECTIVE

from .generation_eval import eval_squad_f1
from .objective_scoring import (
    DirectWorkerObjectiveScorer,
    build_objective_batch,
    score_clean_objective,
)


OPTION_CLASSIFICATION_OBJECTIVES = {
    "sst2_classification",
    "boolq_classification",
    *(
        objective
        for objective in SUPERGLUE_OBJECTIVE_TO_TASK
        if objective != RECORD_NLL_OBJECTIVE
    ),
}
CLASSIFICATION_EVAL_OBJECTIVES = {
    "sst2_classification",
    "boolq_classification",
    *SUPERGLUE_OBJECTIVE_TO_TASK.keys(),
}


@dataclass(frozen=True)
class ObjectiveEvalMetrics:
    """Dev/validation metrics for one objective evaluation."""

    primary_name: str
    dev_value: float | None
    valid_value: float | None
    dev_metrics: dict[str, float] | None = None
    valid_metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class PeriodicEvalResult:
    """One periodic eval result for training-loop logging."""

    loss: float
    primary_name: str
    dev_value: float | None = None
    valid_value: float | None = None
    dev_metrics: dict[str, float] | None = None
    valid_metrics: dict[str, float] | None = None


def score_objective_loss(
    train_objective: str,
    llm: Any,
    rows: list[Any],
    tokenizer: Any,
    *,
    max_logits_tokens: int,
    loss_impl: str,
    max_length: int,
    max_new_tokens: int,
    lora_id: int | None = None,
) -> float | None:
    """Score a clean objective loss for eval/final reporting."""

    if is_objective_batch_loss(train_objective):
        score = _score_objective_rows(
            train_objective,
            llm,
            rows,
            tokenizer,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
            lora_id=lora_id,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        return None if score is None else score.loss
    return None


def score_periodic_eval(
    train_objective: str,
    llm: Any,
    tokenizer: Any,
    dev_rows: list[Any],
    valid_rows: list[Any],
    *,
    accuracy_eval_mode: str,
    max_logits_tokens: int,
    loss_impl: str,
    max_length: int,
    max_new_tokens: int,
    lora_id: int | None = None,
) -> PeriodicEvalResult | None:
    """Score the periodic eval payload used inside the training loop."""

    loss = score_objective_loss(
        train_objective,
        llm,
        dev_rows,
        tokenizer,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        lora_id=lora_id,
    )
    if loss is None:
        return None

    if is_classification_eval_objective(train_objective):
        if accuracy_eval_mode == "skip":
            return PeriodicEvalResult(loss=float(loss), primary_name="accuracy")
        metrics = eval_objective_metrics(
            train_objective,
            llm,
            tokenizer,
            dev_rows,
            valid_rows,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            lora_id=lora_id,
        )
        if metrics is None:
            return None
        return PeriodicEvalResult(
            loss=float(loss),
            primary_name=metrics.primary_name,
            dev_value=metrics.dev_value,
            valid_value=metrics.valid_value,
            dev_metrics=metrics.dev_metrics,
            valid_metrics=metrics.valid_metrics,
        )

    if train_objective == "squad_nll":
        dev_metrics = None
        if accuracy_eval_mode != "skip":
            dev_metrics = eval_squad_f1(
                llm,
                tokenizer,
                dev_rows,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                lora_id=lora_id,
            )
        return PeriodicEvalResult(
            loss=float(loss),
            primary_name="f1",
            dev_value=None if dev_metrics is None else dev_metrics["f1"],
            dev_metrics=dev_metrics,
        )

    return None


def score_current_batch_train_loss(
    train_objective: str,
    llm: Any,
    rows: list[Any],
    tokenizer: Any,
    *,
    max_logits_tokens: int,
    loss_impl: str,
    max_length: int,
    max_new_tokens: int,
    lora_id: int | None = None,
) -> tuple[float, float]:
    """Score current-batch clean train loss and return (loss, score_seconds)."""

    if train_objective == "prompt_nll":
        loss, detail = score_direct_worker_detailed(
            llm,
            rows,
            lora_ids=None if lora_id is None else [lora_id] * len(rows),
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
        )
        return float(loss), float(detail["score_direct_worker_s"])

    train_loss_t0 = time.perf_counter()
    loss = score_objective_loss(
        train_objective,
        llm,
        rows,
        tokenizer,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        lora_id=lora_id,
    )
    if loss is None:
        raise ValueError(f"cannot score train loss for objective: {train_objective}")
    return float(loss), time.perf_counter() - train_loss_t0


def eval_objective_metrics(
    train_objective: str,
    llm: Any,
    tokenizer: Any,
    dev_rows: list[Any],
    valid_rows: list[Any],
    *,
    max_logits_tokens: int,
    loss_impl: str,
    max_length: int,
    max_new_tokens: int,
    lora_id: int | None = None,
) -> ObjectiveEvalMetrics | None:
    """Evaluate objective metrics used by runner logs and summaries."""

    if is_classification_eval_objective(train_objective):
        dev_metrics = _classification_metrics(
            train_objective,
            llm,
            tokenizer,
            dev_rows,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
            lora_id=lora_id,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        valid_metrics = _classification_metrics(
            train_objective,
            llm,
            tokenizer,
            valid_rows,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
            lora_id=lora_id,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        return ObjectiveEvalMetrics(
            primary_name=_primary_metric_name(train_objective),
            dev_value=superglue_primary_metric(dev_metrics),
            valid_value=superglue_primary_metric(valid_metrics),
            dev_metrics=dev_metrics,
            valid_metrics=valid_metrics,
        )
    if train_objective == "squad_nll":
        dev_metrics = eval_squad_f1(
            llm,
            tokenizer,
            dev_rows,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            lora_id=lora_id,
        )
        valid_metrics = eval_squad_f1(
            llm,
            tokenizer,
            valid_rows,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            lora_id=lora_id,
        )
        return ObjectiveEvalMetrics(
            primary_name="f1",
            dev_value=None if dev_metrics is None else dev_metrics["f1"],
            valid_value=None if valid_metrics is None else valid_metrics["f1"],
            dev_metrics=dev_metrics,
            valid_metrics=valid_metrics,
        )
    return None


def is_option_classification_objective(train_objective: str) -> bool:
    return train_objective in OPTION_CLASSIFICATION_OBJECTIVES


def is_objective_batch_loss(train_objective: str) -> bool:
    return (
        train_objective in OPTION_CLASSIFICATION_OBJECTIVES
        or train_objective in {"squad_nll", RECORD_NLL_OBJECTIVE}
    )


def is_classification_eval_objective(train_objective: str) -> bool:
    return train_objective in CLASSIFICATION_EVAL_OBJECTIVES


def superglue_primary_metric(metrics: dict[str, float] | None) -> float | None:
    if metrics is None:
        return None
    if "f1" in metrics:
        return float(metrics["f1"])
    if "accuracy" in metrics:
        return float(metrics["accuracy"])
    if "em" in metrics:
        return float(metrics["em"])
    raise ValueError(f"unsupported SuperGLUE metric payload: {metrics}")


def _score_objective_rows(
    train_objective: str,
    llm: Any,
    rows: list[Any],
    tokenizer: Any,
    *,
    max_logits_tokens: int,
    loss_impl: str,
    lora_id: int | None,
    max_length: int,
    max_new_tokens: int,
):
    if not rows:
        return None
    batch = build_objective_batch(
        rows,
        tokenizer,
        objective_name=train_objective,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    return score_clean_objective(
        DirectWorkerObjectiveScorer(llm),
        batch,
        lora_id=lora_id,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
        include_rows=False,
    )


def _classification_metrics(
    train_objective: str,
    llm: Any,
    tokenizer: Any,
    rows: list[Any],
    *,
    max_logits_tokens: int,
    loss_impl: str,
    lora_id: int | None,
    max_length: int,
    max_new_tokens: int,
) -> dict[str, float] | None:
    if not rows:
        return None
    batch = build_objective_batch(
        rows,
        tokenizer,
        objective_name=train_objective,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    score = score_clean_objective(
        DirectWorkerObjectiveScorer(llm),
        batch,
        lora_id=lora_id,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
        include_rows=False,
    )
    predictions_fn = getattr(batch, "predictions")
    predictions = predictions_fn(score.score)
    if train_objective in {"sst2_classification", "boolq_classification"}:
        return {"accuracy": float(score.accuracy)}
    if train_objective in SUPERGLUE_OBJECTIVE_TO_TASK:
        return superglue_prediction_metric(rows, predictions)
    raise ValueError(f"not a classification objective: {train_objective}")


def _primary_metric_name(train_objective: str) -> str:
    if train_objective in {"sst2_classification", "boolq_classification"}:
        return "accuracy"
    if train_objective in SUPERGLUE_OBJECTIVE_TO_TASK:
        return "f1" if train_objective == RECORD_NLL_OBJECTIVE else "accuracy"
    return "accuracy"
