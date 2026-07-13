"""Objective-level scoring helpers above ``ZOVLLMEngine`` and below Trainer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from zo_vllm.core.direct_worker_scorer import (
    direct_worker_detail,
    score_token_id_groups,
)
from zo_vllm.core.token_groups import (
    borrow_or_copy_int_list,
    borrow_or_copy_token_groups,
)
from zo_vllm.core.token_scores import (
    merge_token_scores as merge_token_scores,
    score_clean_token_groups as _score_clean_token_groups,
)
from zo_vllm.engine import PlusMinusScoreResult, TokenScoreResult, ZOVLLMEngine
from zo_vllm.tasks.superglue import SUPERGLUE_OBJECTIVE_TO_TASK
from zo_vllm.tasks.superglue.record import OBJECTIVE_NAME as RECORD_NLL_OBJECTIVE

from .task_batches import (
    BinaryClassificationScoringBatch,
    MaskedLMScoringBatch,
    OptionClassificationScoringBatch,
    _build_classification_scoring_batch,
    _build_masked_lm_scoring_batch,
)

DEFAULT_CLEAN_SCORE_CHUNK_SIZE = 512


class ObjectiveBatch(Protocol):
    """Tokenized objective batch with task-specific metric aggregation."""

    token_id_groups: list[list[int]]
    loss_token_lens: list[int]
    token_labels: list[list[int]]

    def loss(self, score: TokenScoreResult) -> float:
        """Return the scalar objective loss for a token score."""

    def accuracy(self, score: TokenScoreResult) -> float:
        """Return the task accuracy for a token score."""


class DirectWorkerObjectiveScorer:
    """Adapter exposing ``score_token_groups`` for a raw vLLM LLM object."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def score_token_groups(
        self,
        token_id_groups,
        *,
        loss_token_lens=None,
        labels=None,
        lora_ids=None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
    ) -> TokenScoreResult:
        raw = score_token_id_groups(
            self.llm,
            borrow_or_copy_token_groups(token_id_groups),
            lora_ids=borrow_or_copy_int_list(lora_ids),
            loss_token_lens=borrow_or_copy_int_list(loss_token_lens),
            labels=borrow_or_copy_token_groups(labels),
            max_logits_tokens=int(max_logits_tokens),
            loss_impl=loss_impl,
        )
        return TokenScoreResult(
            loss=float(raw["loss"]),
            nll_sum=float(raw["nll_sum"]),
            num_tokens=int(raw["num_tokens"]),
            request_nll=[float(item) for item in raw["request_nll"]],
            request_num_tokens=[int(item) for item in raw["request_num_tokens"]],
            detail=direct_worker_detail(raw, 0.0, loss_impl=loss_impl),
            raw=raw,
        )


@dataclass(frozen=True)
class ObjectiveScore:
    """Clean objective score plus optional row-level analysis."""

    loss: float
    accuracy: float | None
    score: TokenScoreResult
    per_row_losses: list[float] | None = None
    row_request_nlls: list[list[float]] | None = None
    row_request_num_tokens: list[list[int]] | None = None


@dataclass(frozen=True)
class PlusMinusObjectiveScore:
    """Plus/minus objective score with per-side row analysis."""

    loss_plus: float
    loss_minus: float
    projected_grad: float
    plus: ObjectiveScore
    minus: ObjectiveScore
    raw: PlusMinusScoreResult
    per_row_coefficients: list[float] | None = None
    update_info: dict[str, Any] = field(default_factory=dict)


def build_objective_batch(
    rows: Sequence[Any],
    tokenizer: Any,
    *,
    objective_name: str,
    max_length: int | None = None,
    max_new_tokens: int = 50,
) -> (
    BinaryClassificationScoringBatch
    | OptionClassificationScoringBatch
    | MaskedLMScoringBatch
):
    """Build the public objective batch facade."""

    if objective_name in {"sst2_classification", "boolq_classification"}:
        return _build_classification_scoring_batch(
            rows,
            tokenizer,
            objective_name=objective_name,
        )
    if (
        objective_name in SUPERGLUE_OBJECTIVE_TO_TASK
        and objective_name != RECORD_NLL_OBJECTIVE
    ):
        return _build_classification_scoring_batch(
            rows,
            tokenizer,
            objective_name=objective_name,
        )
    if objective_name in {"squad_nll", RECORD_NLL_OBJECTIVE}:
        if max_length is None:
            raise ValueError(f"max_length is required for objective {objective_name}")
        return _build_masked_lm_scoring_batch(
            rows,
            tokenizer,
            objective_name=objective_name,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"unsupported objective: {objective_name}")


def score_clean_objective(
    engine: ZOVLLMEngine,
    batch: ObjectiveBatch,
    *,
    lora_id: int | None = None,
    max_logits_tokens: int = 8192,
    loss_impl: str = "logprobs",
    include_rows: bool = False,
    score_chunk_size: int = DEFAULT_CLEAN_SCORE_CHUNK_SIZE,
) -> ObjectiveScore:
    """Score a clean objective batch through ``ZOVLLMEngine``."""

    lora_ids = None
    if lora_id is not None:
        lora_ids = [int(lora_id)] * len(batch.token_id_groups)
    loss_token_lens = getattr(batch, "loss_token_lens", None)
    score = score_clean_token_groups(
        engine,
        batch.token_id_groups,
        loss_token_lens=loss_token_lens,
        labels=None if loss_token_lens is not None else batch.token_labels,
        lora_ids=lora_ids,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
        score_chunk_size=score_chunk_size,
    )
    return objective_score_from_token_score(
        batch,
        score,
        include_rows=include_rows,
    )


def score_clean_token_groups(
    engine: ZOVLLMEngine,
    token_id_groups: Sequence[Sequence[int]],
    *,
    loss_token_lens: Sequence[int] | None = None,
    labels: Sequence[Sequence[int]] | None = None,
    lora_ids: Sequence[int] | None,
    max_logits_tokens: int,
    loss_impl: str,
    score_chunk_size: int = DEFAULT_CLEAN_SCORE_CHUNK_SIZE,
) -> TokenScoreResult:
    """Score clean requests in chunks through the shared token-score helper."""

    return _score_clean_token_groups(
        engine,
        token_id_groups,
        loss_token_lens=loss_token_lens,
        labels=labels,
        lora_ids=lora_ids,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
        score_chunk_size=score_chunk_size,
    )


def score_plus_minus_objective(
    engine: ZOVLLMEngine,
    directions: Mapping[str, Mapping[str, torch.Tensor]],
    batch: ObjectiveBatch,
    *,
    eps: float,
    max_logits_tokens: int = 8192,
    loss_impl: str = "logprobs",
    score_chunk_size: int = 0,
    step: int = 0,
    include_rows: bool = False,
) -> PlusMinusObjectiveScore:
    """Score plus/minus directions for one objective batch."""

    loss_token_lens = getattr(batch, "loss_token_lens", None)
    raw = engine.score_plus_minus_directions(
        directions,
        batch.token_id_groups,
        eps=eps,
        loss_token_lens=loss_token_lens,
        labels=None if loss_token_lens is not None else batch.token_labels,
        objective=batch.loss,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
        score_chunk_size=score_chunk_size,
        step=step,
    )
    plus = objective_score_from_token_score(
        batch,
        raw.plus,
        include_rows=include_rows,
    )
    minus = objective_score_from_token_score(
        batch,
        raw.minus,
        include_rows=include_rows,
    )
    per_row_coefficients = None
    if plus.per_row_losses is not None and minus.per_row_losses is not None:
        per_row_coefficients = [
            (float(plus_loss) - float(minus_loss)) / (2.0 * float(eps))
            for plus_loss, minus_loss in zip(plus.per_row_losses, minus.per_row_losses)
        ]
    return PlusMinusObjectiveScore(
        loss_plus=float(raw.loss_plus),
        loss_minus=float(raw.loss_minus),
        projected_grad=float(raw.projected_grad),
        plus=plus,
        minus=minus,
        raw=raw,
        per_row_coefficients=per_row_coefficients,
        update_info=dict(raw.update_info),
    )


def objective_score_from_token_score(
    batch: ObjectiveBatch,
    score: TokenScoreResult,
    *,
    include_rows: bool = False,
) -> ObjectiveScore:
    """Aggregate a token score with one objective batch."""

    accuracy = None
    accuracy_fn = getattr(batch, "accuracy", None)
    if callable(accuracy_fn):
        accuracy = float(accuracy_fn(score))
    per_row_losses = None
    row_request_nlls = None
    row_request_num_tokens = None
    if include_rows:
        per_row_losses_fn = getattr(batch, "per_row_losses", None)
        row_request_nlls_fn = getattr(batch, "row_request_nlls", None)
        row_request_num_tokens_fn = getattr(batch, "row_request_num_tokens", None)
        if callable(per_row_losses_fn):
            per_row_losses = [float(value) for value in per_row_losses_fn(score)]
        if callable(row_request_nlls_fn):
            row_request_nlls = [
                [float(value) for value in row] for row in row_request_nlls_fn(score)
            ]
        if callable(row_request_num_tokens_fn):
            row_request_num_tokens = [
                [int(value) for value in row]
                for row in row_request_num_tokens_fn(score)
            ]
    return ObjectiveScore(
        loss=float(batch.loss(score)),
        accuracy=accuracy,
        score=score,
        per_row_losses=per_row_losses,
        row_request_nlls=row_request_nlls,
        row_request_num_tokens=row_request_num_tokens,
    )


def collect_per_row_coefficients(
    plus: ObjectiveScore,
    minus: ObjectiveScore,
    *,
    eps: float,
) -> list[float]:
    """Return per-row ZO coefficients from two row-analyzed objective scores."""

    if plus.per_row_losses is None or minus.per_row_losses is None:
        raise ValueError("per-row losses are required; call with include_rows=True")
    if len(plus.per_row_losses) != len(minus.per_row_losses):
        raise ValueError(
            "plus/minus row loss counts differ: "
            f"{len(plus.per_row_losses)} != {len(minus.per_row_losses)}"
        )
    return [
        (float(plus_loss) - float(minus_loss)) / (2.0 * float(eps))
        for plus_loss, minus_loss in zip(plus.per_row_losses, minus.per_row_losses)
    ]
