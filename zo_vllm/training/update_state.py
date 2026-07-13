"""Low-rank update state shared by LOZO and AGZO trainers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import time
from typing import Any, Protocol

import torch

from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.engine import DirectionMap, ZOVLLMEngine

from .direction import clone_direction_map
from .update_metrics import build_update_snapshot_metrics


class ZOUpdateState(Protocol):
    """State machine for applying projected ZO updates."""

    def prepare_for_score(
        self, directions: DirectionMap
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return directions that should be written into plus/minus slots."""

    def apply(
        self,
        directions: DirectionMap,
        *,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        step: int,
    ) -> dict[str, Any]:
        """Apply or accumulate one low-rank update."""


@dataclass
class ImmediateWeightUpdateState:
    """Apply every projected low-rank update directly to vLLM base weights."""

    weight_sync: WeightSync
    precision: str = "param"
    sync_device: bool = True
    qkv_update_mode: str = "batched"

    def prepare_for_score(
        self, directions: DirectionMap
    ) -> dict[str, dict[str, torch.Tensor]]:
        return clone_direction_map(directions)

    def apply(
        self,
        directions: DirectionMap,
        *,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        step: int,
    ) -> dict[str, Any]:
        del step
        return self.weight_sync.apply_lozo_update(
            dict(directions),
            c=float(projected_grad),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            precision=self.precision,
            sync_device=bool(self.sync_device),
            qkv_update_mode=self.qkv_update_mode,
        )


@dataclass
class AccumulatedLowRankUpdateState:
    """
    Accumulate projected updates in LoRA-B/U space and fold only at direction
    refresh boundaries or finalization.

    For a fixed V subspace this stores
    ``U_accum = u_beta * U_accum - lr * c * scale * U``. Plus/minus probes
    are scored as ``(U_accum +/- eps * scale * U) @ V.T`` by attaching
    U_accum to the direction map before it reaches LoRAUpdateRuntime.
    """

    weight_sync: WeightSync
    engine: ZOVLLMEngine | None = None
    precision: str = "param"
    sync_device: bool = True
    qkv_update_mode: str = "batched"
    u_beta: float = 1.0
    u_norm_cap: float | None = None
    gradient_accumulation_update_steps: int = 0
    accumulated_u: dict[str, torch.Tensor] = field(default_factory=dict)
    pending_u: dict[str, torch.Tensor] = field(default_factory=dict)
    v_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    vt_cache: dict[str, torch.Tensor | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.u_beta) <= 1.0):
            raise ValueError("u_beta must be in [0, 1]")
        if self.u_norm_cap is not None and float(self.u_norm_cap) <= 0.0:
            raise ValueError("u_norm_cap must be positive when set")
        if int(self.gradient_accumulation_update_steps) < 0:
            raise ValueError("gradient_accumulation_update_steps must be non-negative")

    def prepare_for_score(
        self, directions: DirectionMap
    ) -> dict[str, dict[str, torch.Tensor]]:
        prepared = clone_direction_map(directions)
        for name, direction in prepared.items():
            U = direction["U"]
            acc = self.accumulated_u.get(name)
            if acc is None:
                acc = torch.zeros_like(U)
                self.accumulated_u[name] = acc
            self._remember_basis(name, direction)
            direction["U_accum"] = acc
        return prepared

    def apply(
        self,
        directions: DirectionMap,
        *,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        step: int,
    ) -> dict[str, Any]:
        if float(weight_decay) != 0.0:
            raise ValueError(
                "AccumulatedLowRankUpdateState does not support weight_decay"
            )
        t0 = time.perf_counter()
        update_interval = int(self.gradient_accumulation_update_steps)
        update_target = self.pending_u if update_interval > 0 else self.accumulated_u
        for name, raw_direction in dict(directions).items():
            direction = dict(raw_direction)
            target_u = update_target.get(name)
            if target_u is None:
                target_u = torch.zeros_like(direction["U"])
                update_target[name] = target_u
            self._remember_basis(name, direction)
            scale = float(direction.get("scale", 1.0))
            if float(self.u_beta) != 1.0:
                target_u.mul_(float(self.u_beta))
            target_u.add_(
                direction["U"],
                alpha=-float(learning_rate) * float(projected_grad) * scale,
            )
        accumulate_s = time.perf_counter() - t0
        pending_flush_s = 0.0
        cap_scale = None
        capped_norm = None
        if update_interval > 0 and int(step) % update_interval == 0:
            pending_flush_s = self.flush_pending_to_accumulated(step=step)
            cap_scale, capped_norm = self._cap_accumulated_u_norm()
        elif update_interval <= 0:
            cap_scale, capped_norm = self._cap_accumulated_u_norm()
        return {
            "mode": "accumulate",
            "accumulate_s": accumulate_s,
            "pending_flush_s": pending_flush_s,
            "gradient_accumulation_update_steps": update_interval,
            "fold_s": 0.0,
            "num_accumulated_modules": len(self.accumulated_u),
            "num_pending_modules": len(self.pending_u),
            "u_beta": float(self.u_beta),
            "u_norm_cap": None if self.u_norm_cap is None else float(self.u_norm_cap),
            "u_cap_scale": cap_scale,
            "u_norm_after_cap": capped_norm,
        }

    def fold_before_direction_refresh(self, *, step: int) -> float:
        """Fold accumulated U before the provider replaces the current V basis."""
        return self.fold(step=step)

    def fold(self, *, step: int) -> float:
        self.flush_pending_to_accumulated(step=step)
        if not self.accumulated_u:
            return 0.0
        fold_directions = self.fold_directions()
        if not fold_directions:
            self.accumulated_u.clear()
            return 0.0
        t0 = time.perf_counter()
        self.weight_sync.apply_lozo_update(
            fold_directions,
            c=-1.0,
            lr=1.0,
            weight_decay=0.0,
            precision=self.precision,
            sync_device=bool(self.sync_device),
            qkv_update_mode=self.qkv_update_mode,
        )
        for acc in self.accumulated_u.values():
            acc.zero_()
        return time.perf_counter() - t0

    def flush_pending_to_accumulated(self, *, step: int) -> float:
        del step
        if not self.pending_u:
            return 0.0
        t0 = time.perf_counter()
        for name, pending in self.pending_u.items():
            acc = self.accumulated_u.get(name)
            if acc is None:
                acc = torch.zeros_like(pending)
                self.accumulated_u[name] = acc
            acc.add_(pending)
        self.pending_u.clear()
        self._cap_accumulated_u_norm()
        return time.perf_counter() - t0

    def fold_directions(self) -> dict[str, dict[str, torch.Tensor]]:
        result = {}
        for name, acc in self.accumulated_u.items():
            V = self.v_cache.get(name)
            if V is None:
                continue
            result[name] = {
                "U": acc,
                "V": V,
                "V_T": self.vt_cache.get(name),
            }
        return result

    def set_clean_lora_for_score(
        self,
        *,
        step: int,
        runtime: Any | None = None,
    ) -> tuple[int | None, float]:
        """Write W + U_accum V.T into the plus slot without folding base weights."""

        if not self.accumulated_u:
            return None, 0.0
        target_runtime = runtime
        plus_id = None
        if target_runtime is None:
            if self.engine is None:
                return None, 0.0
            self.engine.register_direction_slots()
            target_runtime = self.engine.runtime
            plus_id = self.engine.plus_id
        else:
            plus_id = target_runtime.plus_id
        clean_directions = self.clean_directions_for_score()
        if not clean_directions:
            return None, 0.0
        t0 = time.perf_counter()
        target_runtime.update_plus_minus_from_directions(
            clean_directions,
            eps=0.0,
            step=int(step),
        )
        return int(plus_id), time.perf_counter() - t0

    def clean_directions_for_score(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return the effective accumulated update without mutating base weights."""

        clean_directions = {}
        for name, acc in self.accumulated_u.items():
            V = self.v_cache.get(name)
            if V is None:
                continue
            clean_directions[name] = {
                "U": torch.zeros_like(acc),
                "U_accum": acc,
                "V": V,
                "V_T": self.vt_cache.get(name),
                "v_refreshed": True,
            }
        return clean_directions

    def snapshot_tensors(
        self,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
        """Return CPU copies of accumulated U tensors for durable snapshots."""

        return {
            name: acc.detach().to(device="cpu", dtype=dtype).clone()
            for name, acc in self.accumulated_u.items()
        }

    def snapshot_metrics(
        self,
        *,
        step: int,
        previous_flat: torch.Tensor | None = None,
        previous_delta: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any] | None, torch.Tensor | None, torch.Tensor | None]:
        """Return U_accum diagnostics without mutating training state."""

        if not self.accumulated_u:
            return None, previous_flat, previous_delta
        flat_parts = []
        module_rows = []
        total_u_sq = 0.0
        total_w_sq = 0.0
        for name in sorted(self.accumulated_u):
            acc = self.accumulated_u[name].detach().float()
            u_norm = float(torch.linalg.vector_norm(acc).item())
            V = self.v_cache.get(name)
            v_norm = None
            w_fro_est = None
            if V is not None:
                v_norm = float(torch.linalg.vector_norm(V.detach().float()).item())
                w_fro_est = u_norm * v_norm
                total_w_sq += w_fro_est * w_fro_est
            total_u_sq += u_norm * u_norm
            module_rows.append(
                {
                    "name": name,
                    "u_norm": u_norm,
                    "v_norm": v_norm,
                    "delta_w_fro_est": w_fro_est,
                }
            )
            flat_parts.append(acc.flatten().cpu())
        return build_update_snapshot_metrics(
            step=step,
            module_rows=module_rows,
            flat_parts=flat_parts,
            total_u_sq=total_u_sq,
            total_w_sq=total_w_sq,
            previous_flat=previous_flat,
            previous_delta=previous_delta,
        )

    def _remember_basis(self, name: str, direction: Mapping[str, torch.Tensor]) -> None:
        V = direction.get("V")
        if V is not None:
            self.v_cache[name] = V
        self.vt_cache[name] = direction.get("V_T")

    def _cap_accumulated_u_norm(self) -> tuple[float | None, float | None]:
        if self.u_norm_cap is None or not self.accumulated_u:
            return None, None
        accum_values = list(self.accumulated_u.values())
        first = accum_values[0]
        total_sq = torch.zeros((), device=first.device, dtype=torch.float32)
        for acc in accum_values:
            total_sq = total_sq + acc.detach().float().square().sum()
        total_norm = float(total_sq.sqrt().item())
        cap = float(self.u_norm_cap)
        if total_norm <= cap or total_norm <= 0.0:
            return 1.0, total_norm
        scale = cap / total_norm
        torch._foreach_mul_(accum_values, scale)
        return scale, cap
