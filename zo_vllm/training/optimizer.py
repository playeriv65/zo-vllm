"""Shared optimizer state machine for synchronous and asynchronous ZO loops."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math
from typing import Any, Generic, TypeVar

import torch

from .zo_step import ZOPendingStep, ZOStepResult


PendingT = TypeVar("PendingT")


class BaseZOSGDOptimizer(torch.optim.Optimizer, Generic[PendingT]):
    """Common staged-estimate and checkpoint semantics for ZO-SGD."""

    pending_type: type[Any]

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        lr_f = float(lr)
        decay_f = float(weight_decay)
        if lr_f < 0.0 or not math.isfinite(lr_f):
            raise ValueError("lr must be non-negative and finite")
        if decay_f < 0.0 or not math.isfinite(decay_f):
            raise ValueError("weight_decay must be non-negative and finite")
        super().__init__(params, {"lr": lr_f, "weight_decay": decay_f})
        if len(self.param_groups) != 1 or len(self.param_groups[0]["params"]) != 1:
            raise ValueError(
                f"{type(self).__name__} requires exactly one protocol parameter"
            )
        self._pending_step: PendingT | None = None
        self.state[self._protocol_parameter()]["step"] = 0

    def stage(self, pending_step: PendingT) -> None:
        if not isinstance(pending_step, self.pending_type):
            raise TypeError(
                f"{type(self).__name__}.stage requires {self.pending_type.__name__}"
            )
        if self._pending_step is not None:
            raise RuntimeError("a ZO optimizer step is already staged")
        expected_step = self.step_count + 1
        if int(pending_step.step) != expected_step:  # type: ignore[attr-defined]
            raise RuntimeError(
                "staged ZO step does not match optimizer state: "
                f"{pending_step.step} != {expected_step}"  # type: ignore[attr-defined]
            )
        self._pending_step = pending_step

    @property
    def step_count(self) -> int:
        value = self.state[self._protocol_parameter()].get("step", 0)
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    def _require_pending(self) -> PendingT:
        if self._pending_step is None:
            raise RuntimeError("optimizer.step() requires one staged ZO estimate")
        return self._pending_step

    def _finish_step(self) -> None:
        self._pending_step = None
        self.state[self._protocol_parameter()]["step"] = self.step_count + 1

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        del set_to_none

    def state_dict(self) -> dict[str, Any]:
        if self._pending_step is not None:
            raise RuntimeError("cannot checkpoint with a staged ZO estimate")
        return super().state_dict()

    def _protocol_parameter(self) -> torch.nn.Parameter:
        return self.param_groups[0]["params"][0]


class ZOSGDOptimizer(BaseZOSGDOptimizer[ZOPendingStep]):
    """Apply one staged runtime estimate at the optimizer boundary."""

    pending_type = ZOPendingStep

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.0,
        on_step_result: Callable[[ZOStepResult], None] | None = None,
    ) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self._on_step_result = on_step_result
        self.last_step_result: ZOStepResult | None = None

    def step(self, closure=None):  # type: ignore[override]
        if closure is not None:
            raise ValueError("ZOSGDOptimizer does not support closures")
        pending = self._require_pending()
        group = self.param_groups[0]
        result = pending.apply(
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
        )
        self._finish_step()
        self.last_step_result = result
        if self._on_step_result is not None:
            self._on_step_result(result)
        return None


__all__ = ["BaseZOSGDOptimizer", "ZOSGDOptimizer"]
