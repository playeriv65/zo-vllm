"""Shared utilities for per-request token score payloads."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .probe_results import ProbeLossResult, merge_probe_losses, slice_probe_loss
from .token_groups import borrow_or_copy_int_list, borrow_or_copy_token_groups


class TokenGroupScorer(Protocol):
    """Minimal scorer protocol used by token-score helpers."""

    def score_token_groups(
        self,
        token_id_groups,
        *,
        loss_token_lens=None,
        labels=None,
        lora_ids=None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
    ):
        """Return a token score for one batch of token id groups."""
        ...


def score_clean_token_groups(
    scorer: TokenGroupScorer,
    token_id_groups: Sequence[Sequence[int]],
    *,
    loss_token_lens: Sequence[int] | None = None,
    labels: Sequence[Sequence[int]] | None = None,
    lora_ids: Sequence[int] | None,
    max_logits_tokens: int,
    loss_impl: str,
    score_chunk_size: int = 0,
    result_factory: Callable[..., Any] | None = None,
):
    """Score token groups in chunks and merge them without changing order."""

    token_groups = borrow_or_copy_token_groups(token_id_groups)
    loss_lens = borrow_or_copy_int_list(loss_token_lens)
    label_rows = borrow_or_copy_token_groups(labels)
    lora_id_rows = borrow_or_copy_int_list(lora_ids)
    chunk_size = int(score_chunk_size)
    if chunk_size <= 0 or len(token_groups) <= chunk_size:
        return scorer.score_token_groups(
            token_groups,
            loss_token_lens=loss_lens,
            labels=label_rows,
            lora_ids=lora_id_rows,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
        )

    scores = []
    for start in range(0, len(token_groups), chunk_size):
        end = min(start + chunk_size, len(token_groups))
        scores.append(
            scorer.score_token_groups(
                token_groups[start:end],
                loss_token_lens=None if loss_lens is None else loss_lens[start:end],
                labels=None if label_rows is None else label_rows[start:end],
                lora_ids=None if lora_id_rows is None else lora_id_rows[start:end],
                max_logits_tokens=max_logits_tokens,
                loss_impl=loss_impl,
            )
        )
    return merge_score_results(
        scores,
        loss_impl=loss_impl,
        result_factory=result_factory,
    )


def merge_token_scores(
    scores: Sequence[Any],
    *,
    loss_impl: str,
    result_factory: Callable[..., Any] | None = None,
):
    """Merge score chunks into one per-request token score payload."""

    if not scores:
        raise ValueError("scores must not be empty")
    request_nll = [value for score in scores for value in score.request_nll]
    request_num_tokens = [
        value for score in scores for value in score.request_num_tokens
    ]
    nll_sum = float(
        sum(
            float(nll) * float(tokens)
            for nll, tokens in zip(request_nll, request_num_tokens)
        )
    )
    num_tokens = int(sum(request_num_tokens))
    if num_tokens <= 0:
        raise ValueError("merged score has zero loss tokens")
    raw = {
        **scores[-1].raw,
        "loss": nll_sum / float(num_tokens),
        "nll_sum": nll_sum,
        "num_tokens": num_tokens,
        "num_reqs": len(request_nll),
        "request_nll": request_nll,
        "request_num_tokens": request_num_tokens,
        "num_chunks": len(scores),
    }
    if result_factory is not None:
        return result_factory(raw, loss_impl=loss_impl)
    detail = dict(scores[-1].detail)
    detail.update(
        {
            "score_num_chunks": len(scores),
            "score_chunk_size": max(len(score.request_nll) for score in scores),
            "score_num_requests": len(request_nll),
            "score_num_loss_tokens": num_tokens,
            "score_loss_impl": loss_impl,
        }
    )
    return _token_score_like(scores[-1], raw, detail=detail)


def slice_token_score(
    score: Any,
    start: int,
    end: int,
    *,
    loss_impl: str,
    result_factory: Callable[..., Any] | None = None,
):
    """Slice a token score by request index and recompute aggregate loss."""

    request_nll = score.request_nll[start:end]
    request_num_tokens = score.request_num_tokens[start:end]
    if not request_nll:
        raise ValueError(f"empty score slice: start={start}, end={end}")
    nll_sum = float(
        sum(
            float(nll) * float(tokens)
            for nll, tokens in zip(request_nll, request_num_tokens)
        )
    )
    num_tokens = int(sum(request_num_tokens))
    if num_tokens <= 0:
        raise ValueError(f"score slice has zero loss tokens: start={start}, end={end}")
    raw = {
        **score.raw,
        "loss": nll_sum / float(num_tokens),
        "nll_sum": nll_sum,
        "num_tokens": num_tokens,
        "num_reqs": len(request_nll),
        "request_nll": request_nll,
        "request_num_tokens": request_num_tokens,
    }
    if result_factory is not None:
        return result_factory(raw, loss_impl=loss_impl)
    detail = dict(score.detail)
    detail.update(
        {
            "score_num_requests": len(request_nll),
            "score_num_loss_tokens": num_tokens,
            "score_loss_impl": loss_impl,
        }
    )
    return _token_score_like(score, raw, detail=detail)


def slice_score_result(
    score: Any,
    start: int,
    end: int,
    *,
    loss_impl: str,
    result_factory: Callable[..., Any] | None = None,
):
    """Slice either a token score or an objective-level probe result."""

    if isinstance(score, ProbeLossResult):
        return slice_probe_loss(score, start, end)
    return slice_token_score(
        score,
        start,
        end,
        loss_impl=loss_impl,
        result_factory=result_factory,
    )


def merge_score_results(
    scores: Sequence[Any],
    *,
    loss_impl: str,
    result_factory: Callable[..., Any] | None = None,
):
    """Merge homogeneous token-score or objective-level result chunks."""

    if scores and isinstance(scores[0], ProbeLossResult):
        if not all(isinstance(score, ProbeLossResult) for score in scores):
            raise TypeError("cannot merge token scores and probe losses")
        return merge_probe_losses(scores)
    return merge_token_scores(
        scores,
        loss_impl=loss_impl,
        result_factory=result_factory,
    )


def _token_score_like(template: Any, raw: dict[str, Any], *, detail: dict[str, Any]):
    score_type = type(template)
    return score_type(
        loss=float(raw["loss"]),
        nll_sum=float(raw["nll_sum"]),
        num_tokens=int(raw["num_tokens"]),
        request_nll=[float(item) for item in raw["request_nll"]],
        request_num_tokens=[int(item) for item in raw["request_num_tokens"]],
        detail=detail,
        raw=raw,
    )
