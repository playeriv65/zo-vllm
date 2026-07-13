from __future__ import annotations

from typing import Any

from zo_vllm.engine import PlusMinusScoreResult, TokenScoreResult
from zo_vllm.training.objective_scoring import (
    collect_per_row_coefficients,
    score_clean_objective,
    score_plus_minus_objective,
)
from zo_vllm.training.task_batches import BinaryClassificationScoringBatch
from zo_vllm.training.task_batches import MaskedLMScoringBatch


def _score(request_nll: list[float]) -> TokenScoreResult:
    return TokenScoreResult(
        loss=sum(request_nll) / len(request_nll),
        nll_sum=sum(request_nll),
        num_tokens=len(request_nll),
        request_nll=request_nll,
        request_num_tokens=[1] * len(request_nll),
        detail={},
        raw={},
    )


class FakeEngine:
    def __init__(self) -> None:
        self.clean_calls: list[dict[str, Any]] = []
        self.plus_minus_calls: list[dict[str, Any]] = []

    def score_token_groups(self, token_id_groups, **kwargs):
        self.clean_calls.append({"token_id_groups": token_id_groups, **kwargs})
        return _score([0.2, 2.0, 1.5, 0.3])

    def score_plus_minus_directions(self, directions, token_id_groups, **kwargs):
        self.plus_minus_calls.append(
            {"directions": directions, "token_id_groups": token_id_groups, **kwargs}
        )
        plus = _score([0.2, 2.0, 1.5, 0.3])
        minus = _score([0.4, 1.7, 1.2, 0.5])
        objective = kwargs["objective"]
        loss_plus = objective(plus)
        loss_minus = objective(minus)
        eps = float(kwargs["eps"])
        return PlusMinusScoreResult(
            loss_plus=loss_plus,
            loss_minus=loss_minus,
            projected_grad=(loss_plus - loss_minus) / (2.0 * eps),
            plus=plus,
            minus=minus,
            update_info={"profile_s": {"score": 1.0}},
        )


class EchoScoreEngine:
    def __init__(self, request_nll_by_token: dict[int, float]) -> None:
        self.request_nll_by_token = request_nll_by_token
        self.clean_calls: list[dict[str, Any]] = []

    def score_token_groups(self, token_id_groups, **kwargs):
        self.clean_calls.append({"token_id_groups": token_id_groups, **kwargs})
        request_nll = [
            self.request_nll_by_token[int(token_group[0])]
            for token_group in token_id_groups
        ]
        return _score(request_nll)


def test_score_clean_objective_aggregates_batch_metrics():
    batch = BinaryClassificationScoringBatch(
        token_id_groups=[[1], [2], [3], [4]],
        labels=[0, 1],
        loss_token_lens=[1, 1, 1, 1],
    )
    engine = FakeEngine()

    result = score_clean_objective(
        engine,  # type: ignore[arg-type]
        batch,
        lora_id=7,
        include_rows=True,
    )

    assert result.loss > 0.0
    assert result.accuracy == 1.0
    assert result.row_request_nlls == [[0.2, 1.5], [2.0, 0.3]]
    assert result.row_request_num_tokens == [[1, 1], [1, 1]]
    assert len(result.per_row_losses) == 2
    assert engine.clean_calls[0]["lora_ids"] == [7, 7, 7, 7]
    assert engine.clean_calls[0]["labels"] is None
    assert engine.clean_calls[0]["loss_token_lens"] == [1, 1, 1, 1]


def test_score_clean_objective_chunks_eval_requests_without_reordering():
    batch = BinaryClassificationScoringBatch(
        token_id_groups=[[0], [1], [2], [3], [4], [5]],
        labels=[0, 1, 0],
        loss_token_lens=[1, 1, 1, 1, 1, 1],
    )
    engine = EchoScoreEngine(
        {
            0: 0.2,
            1: 2.0,
            2: 0.1,
            3: 1.5,
            4: 0.3,
            5: 2.2,
        }
    )

    result = score_clean_objective(
        engine,  # type: ignore[arg-type]
        batch,
        lora_id=9,
        include_rows=True,
        score_chunk_size=2,
    )

    assert [call["token_id_groups"] for call in engine.clean_calls] == [
        [[0], [1]],
        [[2], [3]],
        [[4], [5]],
    ]
    assert [call["labels"] for call in engine.clean_calls] == [None, None, None]
    assert [call["loss_token_lens"] for call in engine.clean_calls] == [
        [1, 1],
        [1, 1],
        [1, 1],
    ]
    assert [call["lora_ids"] for call in engine.clean_calls] == [
        [9, 9],
        [9, 9],
        [9, 9],
    ]
    assert result.score.request_nll == [0.2, 2.0, 0.1, 1.5, 0.3, 2.2]
    assert result.accuracy == 1.0
    assert result.row_request_nlls == [[0.2, 1.5], [2.0, 0.3], [0.1, 2.2]]
    assert result.score.raw["num_chunks"] == 3


def test_score_plus_minus_objective_collects_row_coefficients():
    batch = BinaryClassificationScoringBatch(
        token_id_groups=[[1], [2], [3], [4]],
        labels=[0, 1],
        loss_token_lens=[1, 1, 1, 1],
    )
    engine = FakeEngine()

    result = score_plus_minus_objective(
        engine,  # type: ignore[arg-type]
        {"x": {}},
        batch,
        eps=0.5,
        include_rows=True,
        score_chunk_size=3,
        step=9,
    )

    assert result.loss_plus == result.plus.loss
    assert result.loss_minus == result.minus.loss
    assert result.per_row_coefficients == collect_per_row_coefficients(
        result.plus,
        result.minus,
        eps=0.5,
    )
    assert engine.plus_minus_calls[0]["labels"] is None
    assert engine.plus_minus_calls[0]["loss_token_lens"] == [1, 1, 1, 1]
    assert engine.plus_minus_calls[0]["score_chunk_size"] == 3
    assert engine.plus_minus_calls[0]["step"] == 9


def test_masked_lm_objective_uses_score_loss_and_row_nlls():
    batch = MaskedLMScoringBatch(
        token_id_groups=[[1, 2, 3], [4, 5]],
        loss_token_lens=[2, 1],
    )
    score = _score([0.5, 1.5])

    assert batch.loss(score) == score.loss
    assert batch.row_request_nlls(score) == [[0.5], [1.5]]
    assert batch.row_request_num_tokens(score) == [[1], [1]]
    assert batch.per_row_losses(score) == [0.5, 1.5]
