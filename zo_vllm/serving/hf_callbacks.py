"""Serving observations layered on Hugging Face Trainer callbacks."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any, Protocol

from transformers import TrainerCallback

from zo_vllm.training.optimizer import ZOSGDOptimizer

from .async_engine_service import validate_score_admission_policy
from .worker_client import WORKER_UPDATE_BANK_SOURCE


class RuntimeObservationSource(Protocol):
    def pop_step_observation(self, step: int) -> dict[str, Any] | None: ...

    def pop_clean_observation(self) -> dict[str, Any] | None: ...


class ServingObservationCallback(TrainerCallback):
    """Record scheduler observations without owning the training loop."""

    def __init__(
        self,
        *,
        runtime: RuntimeObservationSource,
        record: Callable[[dict[str, Any], int], None],
        batch_size: int,
        priority: int,
        score_admission_policy: str,
    ) -> None:
        self.runtime = runtime
        self.record = record
        self.batch_size = int(batch_size)
        self.priority = int(priority)
        self.score_admission_policy = validate_score_admission_policy(
            score_admission_policy
        )
        self._step_start_perf: float | None = None

    def on_step_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        self._step_start_perf = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        del args, kwargs
        step = int(state.global_step)
        step_end_perf = time.perf_counter()
        observation = self.runtime.pop_step_observation(step) or {}
        result = _last_step_result(optimizer)
        metrics: dict[str, Any] = {
            "event": "train_step",
            "step": step,
            "batch_size": self.batch_size,
            "priority": self.priority,
            "step_start_perf": self._step_start_perf,
            "step_end_perf": step_end_perf,
            "step_s": (
                None
                if self._step_start_perf is None
                else step_end_perf - self._step_start_perf
            ),
            "state_backend": WORKER_UPDATE_BANK_SOURCE,
            "score_admission_policy": self.score_admission_policy,
            **observation,
        }
        if result is not None:
            metrics.update(result.metrics())
            metrics["estimator"] = result.estimator_metrics.get("estimator")
            metrics["direction_info"] = dict(result.direction_info)
            metrics.setdefault("update_info", dict(result.update_info))
        self.record(metrics, step)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, kwargs
        values = {
            "event": "eval",
            "step": int(state.global_step),
            **dict(metrics or {}),
        }
        clean = self.runtime.pop_clean_observation()
        if clean is not None:
            values.update(clean)
        values["state_backend"] = WORKER_UPDATE_BANK_SOURCE
        self.record(values, int(state.global_step))
        return control


class InterStepDelayCallback(TrainerCallback):
    """Apply serving-only pacing without changing the Trainer loop."""

    def __init__(self, *, stop_event: threading.Event, delay_s: float) -> None:
        self.stop_event = stop_event
        self.delay_s = float(delay_s)

    def on_step_end(self, args, state, control, **kwargs):
        del args, state, kwargs
        if self.delay_s > 0:
            self.stop_event.wait(self.delay_s)
        if self.stop_event.is_set():
            control.should_training_stop = True
        return control


def _last_step_result(optimizer: Any) -> Any | None:
    current = optimizer
    while current is not None and not isinstance(current, ZOSGDOptimizer):
        current = getattr(current, "optimizer", None)
    return None if current is None else current.last_step_result


__all__ = [
    "InterStepDelayCallback",
    "RuntimeObservationSource",
    "ServingObservationCallback",
]
