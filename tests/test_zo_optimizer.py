from __future__ import annotations

import pytest
import torch

from zo_trainer.optimizer import ZOSGDOptimizer
from zo_vllm.training.zo_step import ZOPendingStep, ZOStepResult


def _pending(step: int, calls: list[tuple[float, float]]) -> ZOPendingStep:
    def apply(learning_rate: float, weight_decay: float) -> ZOStepResult:
        calls.append((float(learning_rate), float(weight_decay)))
        return ZOStepResult(
            step=step,
            learning_rate=learning_rate,
            reported_loss=1.0,
            loss_plus=1.2,
            loss_minus=0.8,
            projected_grad=2.0,
            update_scale=2.0,
            direction_refreshed=False,
        )

    return ZOPendingStep(step=step, reported_loss=1.0, _apply_fn=apply)


def _optimizer(*, lr: float = 0.2, weight_decay: float = 0.1) -> ZOSGDOptimizer:
    return ZOSGDOptimizer(
        [torch.nn.Parameter(torch.zeros(()))],
        lr=lr,
        weight_decay=weight_decay,
    )


def test_zo_sgd_step_consumes_one_staged_estimate() -> None:
    calls: list[tuple[float, float]] = []
    optimizer = _optimizer()

    optimizer.stage(_pending(1, calls))
    optimizer.step()

    assert calls == [(0.2, 0.1)]
    assert optimizer.step_count == 1
    assert optimizer.last_step_result is not None
    assert optimizer.last_step_result.step == 1


def test_zo_sgd_rejects_missing_or_duplicate_staged_estimate() -> None:
    optimizer = _optimizer()
    pending = _pending(1, [])

    with pytest.raises(RuntimeError, match="requires one staged"):
        optimizer.step()

    optimizer.stage(pending)
    with pytest.raises(RuntimeError, match="already staged"):
        optimizer.stage(_pending(1, []))
    with pytest.raises(RuntimeError, match="cannot checkpoint"):
        optimizer.state_dict()


def test_zo_sgd_state_dict_restores_step_contract() -> None:
    first = _optimizer()
    first.stage(_pending(1, []))
    first.step()

    resumed = _optimizer()
    resumed.load_state_dict(first.state_dict())
    resumed.stage(_pending(2, []))
    resumed.step()

    assert resumed.step_count == 2
    with pytest.raises(RuntimeError, match="does not match optimizer state"):
        resumed.stage(_pending(4, []))


def test_pending_step_can_only_be_applied_once() -> None:
    pending = _pending(1, [])

    pending.apply(learning_rate=0.1, weight_decay=0.0)

    with pytest.raises(RuntimeError, match="already been applied"):
        pending.apply(learning_rate=0.1, weight_decay=0.0)
