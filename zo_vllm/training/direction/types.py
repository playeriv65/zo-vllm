from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from zo_vllm.engine import ObjectiveFn

RewardFn = Callable[[str, Any], float | tuple[Any, float]]


@dataclass(frozen=True)
class TokenProbeBatch:
    """Token-scoring payload consumed by ZO estimators."""

    token_id_groups: Sequence[Sequence[int]]
    loss_token_lens: Sequence[int] | None = None
    labels: Sequence[Sequence[int]] | None = None
    objective: ObjectiveFn | None = None


@dataclass(frozen=True)
class RolloutProbeBatch:
    """Generation-reward payload consumed by evolution strategies."""

    rollout_prompts: Sequence[str]
    rollout_targets: Sequence[Any]
    rollout_reward_fn: RewardFn
    rollout_max_tokens: int = 512
    rollout_temperature: float = 0.0
    rollout_top_p: float = 1.0
    rollout_seed: int | None = None


@dataclass(frozen=True)
class SubspaceTokenProbeBatch(TokenProbeBatch):
    """Token payload with optional lazy batches for subspace estimation."""

    subspace_token_id_group_batches: Sequence[Sequence[Sequence[int]]] | None = None
    subspace_loss_token_lens_batches: Sequence[Sequence[int]] | None = None
    subspace_labels_batches: Sequence[Sequence[Sequence[int]]] | None = None
    subspace_num_rows: int | None = None
    subspace_token_id_group_batch_factory: (
        Callable[
            [],
            Sequence[Sequence[Sequence[int]]]
            | tuple[
                Sequence[Sequence[Sequence[int]]],
                Sequence[Sequence[Sequence[int]]],
            ],
        ]
        | None
    ) = None


ProbeBatch = TokenProbeBatch | SubspaceTokenProbeBatch | RolloutProbeBatch


@dataclass(frozen=True)
class DirectionSample:
    """One sampled low-rank direction and its provenance metadata."""

    directions: dict[str, dict[str, torch.Tensor]]
    refreshed: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DirectionSpec:
    """Tensor-free factorized direction specification."""

    name: str
    out_features: int
    in_features: int
    rank: int
    device: torch.device
    dtype: torch.dtype
    scale: float = 1.0
    v_energy_reference: str = "gaussian"


class DirectionProvider(Protocol):
    """Source of low-rank ZO directions."""

    def will_refresh(self, *, step: int) -> bool:
        """Return whether ``next`` will replace the current direction basis."""
        ...

    def next(self, batch: ProbeBatch, *, step: int) -> DirectionSample:
        """Return the direction used for one ZO plus/minus probe."""
        ...

    def direction_specs(self) -> Sequence[DirectionSpec]:
        """Return tensor-free specs for directions this provider samples."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable direction-generation state."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore direction-generation state."""
        ...


def clone_direction_map(
    directions: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Return a mutable shallow copy of a direction map."""

    return {name: dict(value) for name, value in directions.items()}


__all__ = [
    "DirectionProvider",
    "DirectionSample",
    "DirectionSpec",
    "ProbeBatch",
    "RolloutProbeBatch",
    "SubspaceTokenProbeBatch",
    "TokenProbeBatch",
    "clone_direction_map",
]
