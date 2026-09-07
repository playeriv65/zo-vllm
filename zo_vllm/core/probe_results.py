"""Objective-level results returned by loss-backed ZO probe scorers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProbeTiming:
    """Typed host and device timing for one loss-backed probe call."""

    backend_profile_s: tuple[tuple[str, float], ...] = ()
    backend_cuda_ms: tuple[tuple[str, float], ...] = ()
    pending_cuda_events: tuple[tuple[str, Any, Any], ...] = ()
    batch_unpack_s: float = 0.0
    engine_call_s: float = 0.0
    output_postprocess_s: float = 0.0
    loss_dispatch_s: float = 0.0
    device_wait_s: float = 0.0
    loss_to_host_s: float = 0.0
    forward_unpack_s: float = 0.0
    forward_total_s: float = 0.0
    total_s: float = 0.0

    @classmethod
    def from_backend(
        cls,
        profile_s: Mapping[str, object] | None,
        profile_cuda_ms: Mapping[str, object] | None,
        cuda_events: Mapping[str, object] | None = None,
    ) -> "ProbeTiming":
        return cls(
            backend_profile_s=_numeric_items(profile_s),
            backend_cuda_ms=_numeric_items(profile_cuda_ms),
            pending_cuda_events=tuple(
                (str(key), value[0], value[1])
                for key, value in (cuda_events or {}).items()
                if isinstance(value, tuple) and len(value) == 2
            ),
        )

    def updated(self, **values: float) -> "ProbeTiming":
        return replace(self, **{key: float(value) for key, value in values.items()})

    def resolve_cuda_events(self) -> "ProbeTiming":
        if not self.pending_cuda_events:
            return self
        resolved = dict(self.backend_cuda_ms)
        for key, start, end in self.pending_cuda_events:
            resolved[key] = float(start.elapsed_time(end))
        return replace(
            self,
            backend_cuda_ms=tuple(resolved.items()),
            pending_cuda_events=(),
        )

    def profile_seconds(self) -> dict[str, float]:
        values = {
            f"worker_{key.removesuffix('_s')}": value
            for key, value in self.backend_profile_s
        }
        values.update(
            {
                f"worker_cuda_{key.removesuffix('_ms')}": value / 1000.0
                for key, value in self.backend_cuda_ms
            }
        )
        values.update(
            {
                "scorer_batch_unpack": self.batch_unpack_s,
                "scorer_engine_call": self.engine_call_s,
                "scorer_output_postprocess": self.output_postprocess_s,
                "scorer_loss_dispatch": self.loss_dispatch_s,
                "scorer_device_wait": self.device_wait_s,
                "scorer_loss_to_host": self.loss_to_host_s,
                "scorer_forward_unpack": self.forward_unpack_s,
                "scorer_forward_total": self.forward_total_s,
                "scorer_total": self.total_s,
            }
        )
        return {key: value for key, value in values.items() if value != 0.0}


def _numeric_items(
    values: Mapping[str, object] | None,
) -> tuple[tuple[str, float], ...]:
    if values is None:
        return ()
    return tuple(
        (str(key), float(value))
        for key, value in values.items()
        if isinstance(value, (int, float))
    )


@dataclass(frozen=True)
class ProbeLossResult:
    """Scalar objective losses for complete copies of the original batch."""

    group_losses: tuple[float, ...]
    requests_per_group: int
    timing: ProbeTiming = ProbeTiming()

    def __post_init__(self) -> None:
        if not self.group_losses:
            raise ValueError("ProbeLossResult requires at least one group loss")
        if int(self.requests_per_group) <= 0:
            raise ValueError("requests_per_group must be positive")

    @property
    def loss(self) -> float:
        return float(sum(self.group_losses) / len(self.group_losses))

    @property
    def num_requests(self) -> int:
        return int(len(self.group_losses) * self.requests_per_group)


def slice_probe_loss(
    result: ProbeLossResult,
    start: int,
    end: int,
) -> ProbeLossResult:
    """Slice complete objective groups by their request range."""

    width = int(result.requests_per_group)
    if start < 0 or end <= start or end > result.num_requests:
        raise ValueError(f"invalid probe loss slice: start={start}, end={end}")
    if start % width or end % width:
        raise ValueError("probe loss slices must align with complete objective groups")
    return ProbeLossResult(
        group_losses=result.group_losses[start // width : end // width],
        requests_per_group=width,
        timing=result.timing,
    )


def merge_probe_losses(results: Sequence[ProbeLossResult]) -> ProbeLossResult:
    """Merge adjacent objective-level probe results."""

    if not results:
        raise ValueError("probe loss results must not be empty")
    width = int(results[0].requests_per_group)
    if any(int(result.requests_per_group) != width for result in results):
        raise ValueError("probe loss results use different objective group widths")
    return ProbeLossResult(
        group_losses=tuple(loss for result in results for loss in result.group_losses),
        requests_per_group=width,
        timing=results[-1].timing,
    )


__all__ = ["ProbeLossResult", "ProbeTiming", "merge_probe_losses", "slice_probe_loss"]
