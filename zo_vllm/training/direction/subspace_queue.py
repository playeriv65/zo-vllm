"""Fixed-shape AGZO subspace queue utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .types import DirectionSpec


DirectionMap = Mapping[str, Mapping[str, torch.Tensor]]


class SubspaceQueue:
    """Ring buffer of AGZO V subspaces with stable low-rank tensor shapes."""

    def __init__(self, queue_size: int = 1) -> None:
        if int(queue_size) <= 0:
            raise ValueError("queue_size must be positive")
        self.queue_size = int(queue_size)
        self._slots: list[dict[str, dict[str, torch.Tensor]] | None] = [
            None
        ] * self.queue_size
        self._victim_idx = 0

    @property
    def victim_idx(self) -> int:
        return int(self._victim_idx)

    @property
    def active_slots(self) -> int:
        return sum(slot is not None for slot in self._slots)

    @property
    def is_empty(self) -> bool:
        return self.active_slots == 0

    def should_refresh(self, step: int, reuse_steps: int) -> bool:
        if int(step) <= 0:
            raise ValueError("step must be positive")
        if int(reuse_steps) <= 0:
            raise ValueError("reuse_steps must be positive")
        return self.is_empty or ((int(step) - 1) % int(reuse_steps) == 0)

    def insert(self, directions: DirectionMap) -> int:
        """Overwrite the current victim slot and advance the ring pointer."""
        victim_idx = self._victim_idx
        self._slots[victim_idx] = {
            name: dict(value) for name, value in dict(directions).items()
        }
        self._victim_idx = (self._victim_idx + 1) % self.queue_size
        return int(victim_idx)

    def active_slot_items(self) -> list[tuple[int, dict[str, dict[str, torch.Tensor]]]]:
        """Return active cached slots with their queue indices."""
        return [(idx, slot) for idx, slot in enumerate(self._slots) if slot is not None]

    def state_dict(self) -> dict[str, Any]:
        slots = []
        for slot in self._slots:
            if slot is None:
                slots.append(None)
                continue
            slots.append(
                {
                    name: {
                        key: value.detach().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in direction.items()
                        if key != "V_T"
                    }
                    for name, direction in slot.items()
                }
            )
        return {
            "queue_size": int(self.queue_size),
            "victim_idx": int(self._victim_idx),
            "slots": slots,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        direction_specs: Sequence[DirectionSpec],
    ) -> None:
        if int(state["queue_size"]) != self.queue_size:
            raise ValueError("subspace queue size does not match checkpoint")
        raw_slots = list(state["slots"])
        if len(raw_slots) != self.queue_size:
            raise ValueError("subspace queue checkpoint has invalid slot count")
        specs = {spec.name: spec for spec in direction_specs}
        restored_slots = []
        for raw_slot in raw_slots:
            if raw_slot is None:
                restored_slots.append(None)
                continue
            restored_slot = {}
            for name, raw_direction in dict(raw_slot).items():
                if name not in specs:
                    raise KeyError(f"unknown direction in subspace queue: {name}")
                spec = specs[name]
                raw_v = raw_direction.get("V")
                if not isinstance(raw_v, torch.Tensor):
                    raise TypeError(f"subspace queue V must be a tensor: {name}")
                v = raw_v.to(device=spec.device, dtype=spec.dtype).contiguous()
                if int(v.shape[0]) != int(spec.in_features):
                    raise ValueError(f"subspace queue V shape mismatch: {name}")
                direction = {
                    key: value
                    for key, value in raw_direction.items()
                    if key not in {"U", "V", "V_T"}
                }
                direction.update(
                    {
                        "U": torch.empty(
                            (int(spec.out_features), int(v.shape[1])),
                            device=spec.device,
                            dtype=spec.dtype,
                        ),
                        "V": v,
                        "V_T": v.T.contiguous(),
                    }
                )
                restored_slot[name] = direction
            restored_slots.append(restored_slot)
        victim_idx = int(state["victim_idx"])
        if not 0 <= victim_idx < self.queue_size:
            raise ValueError("subspace queue victim index is invalid")
        self._slots = restored_slots
        self._victim_idx = victim_idx
