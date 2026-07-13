"""Small learning-rate schedulers for zeroth-order training loops."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class LRScheduler(Protocol):
    """Callable step-indexed learning-rate schedule."""

    def __call__(self, step: int) -> float:
        """Return the learning rate for a one-indexed training step."""


@dataclass(frozen=True)
class ConstantLR:
    """Constant learning rate."""

    learning_rate: float

    def __post_init__(self) -> None:
        if float(self.learning_rate) < 0.0 or not math.isfinite(float(self.learning_rate)):
            raise ValueError("learning_rate must be non-negative and finite")

    def __call__(self, step: int) -> float:
        if int(step) <= 0:
            raise ValueError("step must be positive")
        return float(self.learning_rate)


@dataclass(frozen=True)
class CosineAfterLR:
    """Hold a peak LR, then cosine decay to a final scale."""

    learning_rate: float
    decay_start_step: int
    max_steps: int
    final_scale: float = 0.25

    def __post_init__(self) -> None:
        if float(self.learning_rate) < 0.0 or not math.isfinite(float(self.learning_rate)):
            raise ValueError("learning_rate must be non-negative and finite")
        if int(self.decay_start_step) <= 0:
            raise ValueError("decay_start_step must be positive")
        if int(self.max_steps) < int(self.decay_start_step):
            raise ValueError("max_steps must be >= decay_start_step")
        if float(self.final_scale) < 0.0 or not math.isfinite(float(self.final_scale)):
            raise ValueError("final_scale must be non-negative and finite")

    def __call__(self, step: int) -> float:
        step_i = int(step)
        if step_i <= 0:
            raise ValueError("step must be positive")
        if step_i <= int(self.decay_start_step):
            return float(self.learning_rate)
        span = max(1, int(self.max_steps) - int(self.decay_start_step))
        progress = min(1.0, (step_i - int(self.decay_start_step)) / float(span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = float(self.final_scale) + (1.0 - float(self.final_scale)) * cosine
        return float(self.learning_rate) * scale
