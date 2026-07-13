"""Hugging Face callbacks for optional ZO runtime observations."""

from __future__ import annotations

from typing import Any, Protocol

from transformers import TrainerCallback


class ZOSnapshotRecorder(Protocol):
    def capture(
        self,
        step: int,
        eval_loss: float | None,
        eval_accuracy: float | None,
    ) -> Any: ...


class ZOUSnapshotCallback(TrainerCallback):
    """Persist optional U-update observations after completed optimizer steps."""

    def __init__(
        self,
        *,
        recorder: ZOSnapshotRecorder,
    ) -> None:
        self.recorder = recorder
        self._captured_steps: set[int] = set()

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if not control.should_evaluate:
            self._capture(int(state.global_step), None, None)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, kwargs
        values = dict(metrics or {})
        self._capture(
            int(state.global_step),
            _optional_float(values.get("eval_loss")),
            _first_metric(values, "eval_accuracy", "eval_acc", "eval_f1"),
        )
        return control

    def _capture(
        self,
        step: int,
        eval_loss: float | None,
        eval_accuracy: float | None,
    ) -> None:
        if step in self._captured_steps:
            return
        result = self.recorder.capture(step, eval_loss, eval_accuracy)
        if result is not None:
            self._captured_steps.add(step)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _first_metric(values: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if values.get(key) is not None:
            return float(values[key])
    return None


__all__ = ["ZOUSnapshotCallback"]
