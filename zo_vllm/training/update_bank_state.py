"""LoRA-bank update state for quantized/read-only base weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
import math
import os
import time
from typing import Any

import torch

from zo_vllm.training.u_optimizer import UCoefficientOptimizer

from zo_vllm.engine import DirectionMap, ZOVLLMEngine

from .update_debug import (
    direction_tensors_finite_summary,
    tensor_finite_summary,
    zo_bank_debug_enabled,
)
from .update_metrics import build_update_snapshot_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BankBlock:
    start: int
    rank: int


@dataclass
class BlockLoRAUpdateBankState:
    """
    Keep quantized base weights read-only and represent all updates as a
    high-rank LoRA bank.

    Each V refresh allocates one rank slice in the bank. Historical slices stay
    active in both plus/minus slots; the current slice carries the +/- U probe.
    """

    engine: ZOVLLMEngine | None = None
    update_bank_rank: int = 1
    u_beta: float = 1.0
    u_momentum: float = 0.0
    u_optimizer: str = "sgd"
    u_beta2: float = 0.9
    u_adam_eps: float = 1e-8
    u_opt: UCoefficientOptimizer | None = None
    # Direction tracking. With a frozen V* the u-direction estimated at step
    # t-1 is a good guess for step t, so instead of spending every probe on a
    # fresh random U the first probe of each step is spent on the previous
    # aggregate direction itself: it measures how strong that direction still
    # is (the correction), and the remaining probes measure how the direction
    # has swung. The multi-query estimator then averages the probes as usual.
    u_track: bool = False
    track_u: dict[str, torch.Tensor] = field(default_factory=dict)
    _queries_this_step: int = 0
    u_norm_cap: float | None = None
    gradient_accumulation_update_steps: int = 0
    bank_a: dict[str, torch.Tensor] = field(default_factory=dict)
    accumulated_u: dict[str, torch.Tensor] = field(default_factory=dict)
    pending_u: dict[str, torch.Tensor] = field(default_factory=dict)
    zero_u: dict[str, torch.Tensor] = field(default_factory=dict)
    current_blocks: dict[str, _BankBlock] = field(default_factory=dict)
    used_rank: dict[str, int] = field(default_factory=dict)
    debug_step: int | None = None

    def __post_init__(self) -> None:
        if int(self.update_bank_rank) <= 0:
            raise ValueError("update_bank_rank must be positive")
        if not (0.0 <= float(self.u_beta) <= 1.0):
            raise ValueError("u_beta must be in [0, 1]")
        self.u_opt = UCoefficientOptimizer(
            name=str(self.u_optimizer),
            momentum=float(self.u_momentum),
            beta2=float(self.u_beta2),
            eps=float(self.u_adam_eps),
        )
        if self.u_norm_cap is not None and float(self.u_norm_cap) <= 0.0:
            raise ValueError("u_norm_cap must be positive when set")
        if int(self.gradient_accumulation_update_steps) < 0:
            raise ValueError("gradient_accumulation_update_steps must be non-negative")

    def _accum_rms(self) -> float:
        """RMS of the live u coefficients: the displacement the bank carries."""

        total = 0.0
        count = 0
        for tensor in list(self.accumulated_u.values()) + list(self.pending_u.values()):
            total += float(tensor.pow(2).sum())
            count += int(tensor.numel())
        return math.sqrt(total / count) if count else 0.0

    def _accumulate(
        self,
        name: str,
        target: torch.Tensor,
        U: torch.Tensor,
        projected_grad: float,
        scale: float,
        learning_rate: float,
    ) -> None:
        """Plain accumulation; the optimizer paths batch their step instead."""

        target.add_(U, alpha=-float(learning_rate) * float(projected_grad) * scale)

    def prepare_for_score(
        self, directions: DirectionMap
    ) -> dict[str, dict[str, torch.Tensor]]:
        if zo_bank_debug_enabled():
            self._log_storage_finite_summary(
                "prepare_storage_before", step=self.debug_step
            )
            self._log_direction_finite_summary(
                "prepare_input",
                dict(directions),
                step=self.debug_step,
            )
        prepared: dict[str, dict[str, torch.Tensor]] = {}
        for name, raw_direction in dict(directions).items():
            direction = dict(raw_direction)
            U = direction["U"]
            block = self.current_blocks.get(name)
            # The v_refreshed flag is advisory: the multi-query aggregate sets it
            # unconditionally, and with a frozen V* re-allocating a block per
            # step would exhaust the bank by step 2. Allocate only when the V
            # factor actually differs from the one the current block holds.
            must_allocate = block is None or (
                bool(direction.get("v_refreshed", True))
                and not self._same_v(name, block, direction)
            )
            if must_allocate:
                block = self._allocate_block(name, direction)
            elif int(U.shape[1]) != int(block.rank):
                raise RuntimeError(
                    f"direction rank changed within an active bank block for {name}: "
                    f"{int(U.shape[1])} != {int(block.rank)}"
                )
            prepared[name] = self._bank_direction(
                name,
                direction,
                block=block,
                perturb=True,
                force_a_refresh=must_allocate,
            )
        if zo_bank_debug_enabled():
            self._log_direction_finite_summary(
                "prepare_output",
                prepared,
                step=self.debug_step,
            )
            self._log_storage_finite_summary(
                "prepare_storage_after", step=self.debug_step
            )
        return prepared

    @torch.no_grad()
    def shape_directions(self, directions: DirectionMap) -> None:
        """Spend the first probe of the step on the tracked direction."""

        if not self.u_track:
            return
        self._queries_this_step += 1
        if self._queries_this_step != 1 or not self.track_u:
            return
        for name, direction in dict(directions).items():
            prev = self.track_u.get(name)
            if prev is None or prev.shape != direction["U"].shape:
                continue
            direction["U"] = prev.to(direction["U"].dtype, copy=True)

    def _remember_track(self, directions: DirectionMap) -> None:
        for name, direction in dict(directions).items():
            U = direction["U"].detach().float()
            rms = U.pow(2).mean().sqrt()
            if float(rms) > 0.0:
                self.track_u[name] = (U / rms).clone()
        self._queries_this_step = 0

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
            raise ValueError("BlockLoRAUpdateBankState does not support weight_decay")
        if self.u_track:
            # the aggregate U handed to apply is the step's direction estimate
            self._remember_track(directions)
        t0 = time.perf_counter()
        self.u_opt.begin_step()
        update_interval = int(self.gradient_accumulation_update_steps)
        if zo_bank_debug_enabled():
            self._log_direction_finite_summary(
                "apply_input",
                dict(directions),
                step=step,
                extra={
                    "projected_grad": float(projected_grad),
                    "projected_grad_finite": math.isfinite(float(projected_grad)),
                    "learning_rate": float(learning_rate),
                    "learning_rate_finite": math.isfinite(float(learning_rate)),
                    "gradient_accumulation_update_steps": update_interval,
                },
            )
        batch_grads: dict[str, torch.Tensor] = {}
        batch_targets: dict[str, torch.Tensor] = {}
        use_torch_opt = self.u_opt.uses_torch
        # Magnitude bookkeeping. step_sq tracks the *plain-equivalent* update
        # -lr*pg*scale*U even when a torch optimizer is driving, because that is
        # the quantity an optimizer's lr has to be calibrated against; accum_sq
        # tracks the u coefficients themselves, i.e. the weight displacement.
        step_sq = 0.0
        step_n = 0
        for name, raw_direction in dict(directions).items():
            direction = dict(raw_direction)
            block = self.current_blocks.get(name)
            if block is None:
                raise RuntimeError(
                    f"cannot apply update before bank block allocation: {name}"
                )
            U = direction["U"]
            scale = float(direction.get("scale", 1.0))
            if self.u_optimizer == "adam_scalar":
                scale = self.u_opt.scalar_rate(
                    name, float(projected_grad) * scale
                )
                projected_grad_here = 1.0
            else:
                projected_grad_here = float(projected_grad)
            _alpha = float(learning_rate) * projected_grad_here * scale
            step_sq += (_alpha * _alpha) * float(U.pow(2).sum())
            step_n += int(U.numel())
            if use_torch_opt:
                if update_interval > 0:
                    tgt = self.pending_u.get(name)
                    if tgt is None:
                        tgt = torch.zeros_like(U)
                        self.pending_u[name] = tgt
                else:
                    tgt = self._current_accum_slice(name, block)
                if float(self.u_beta) != 1.0:
                    tgt.mul_(float(self.u_beta))
                batch_targets[name] = tgt
                batch_grads[name] = U * (float(projected_grad) * scale)
                continue
            if update_interval > 0:
                target = self.pending_u.get(name)
                if target is None:
                    target = torch.zeros_like(U)
                    self.pending_u[name] = target
                if float(self.u_beta) != 1.0:
                    target.mul_(float(self.u_beta))
                self._accumulate(
                    name, target, U, projected_grad_here, scale, learning_rate
                )
            else:
                target = self._current_accum_slice(name, block)
                if float(self.u_beta) != 1.0:
                    target.mul_(float(self.u_beta))
                self._accumulate(
                    name, target, U, projected_grad_here, scale, learning_rate
                )
            if zo_bank_debug_enabled():
                self._log_single_tensor_if_bad(
                    "apply_target_after_update",
                    name=name,
                    tensor=target,
                    step=step,
                    extra={
                        "projected_grad": float(projected_grad),
                        "learning_rate": float(learning_rate),
                        "scale": scale,
                        "update_interval": update_interval,
                        "block_start": int(block.start),
                        "block_rank": int(block.rank),
                    },
                )
        if use_torch_opt and batch_targets:
            self.u_opt.step(batch_grads, batch_targets, learning_rate)
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
            "mode": "lora_bank",
            "accumulate_s": accumulate_s,
            "pending_flush_s": pending_flush_s,
            "gradient_accumulation_update_steps": update_interval,
            "fold_s": 0.0,
            "num_accumulated_modules": len(self.accumulated_u),
            "num_pending_modules": len(self.pending_u),
            "update_bank_rank": int(self.update_bank_rank),
            "used_bank_rank": self.max_used_rank(),
            "u_beta": float(self.u_beta),
            "u_momentum": float(self.u_momentum),
            "u_optimizer": str(self.u_optimizer),
            "u_optimizer_code": self.u_opt.code,
            "u_track": 1.0 if self.u_track else 0.0,
            "u_step_rms": math.sqrt(step_sq / step_n) if step_n else 0.0,
            "u_accum_rms": self._accum_rms(),
            "u_norm_cap": None if self.u_norm_cap is None else float(self.u_norm_cap),
            "u_cap_scale": cap_scale,
            "u_norm_after_cap": capped_norm,
        }

    def fold_before_direction_refresh(self, *, step: int) -> float:
        """Commit pending U into the current bank block without touching base weights.

        The velocity survives the refresh: it lives in the U-space, whose
        dimension does not change with V, and V drifts a median 13 degrees over
        8000 steps.
        """

        return self.flush_pending_to_accumulated(step=step)

    def flush_pending_to_accumulated(self, *, step: int) -> float:
        del step
        if not self.pending_u:
            return 0.0
        t0 = time.perf_counter()
        for name, pending in self.pending_u.items():
            block = self.current_blocks.get(name)
            if block is None:
                raise RuntimeError(
                    f"cannot flush pending update without a bank block: {name}"
                )
            self._current_accum_slice(name, block).add_(pending)
            if zo_bank_debug_enabled():
                self._log_single_tensor_if_bad(
                    "flush_accum_after_pending",
                    name=name,
                    tensor=self._current_accum_slice(name, block),
                    step=self.debug_step,
                    extra={
                        "block_start": int(block.start),
                        "block_rank": int(block.rank),
                    },
                )
        self.pending_u.clear()
        self._cap_accumulated_u_norm()
        return time.perf_counter() - t0

    def set_clean_lora_for_score(
        self,
        *,
        step: int,
        runtime: Any | None = None,
    ) -> tuple[int | None, float]:
        """Write the active bank into the plus slot with no +/- perturbation."""

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
        clean_directions = {
            name: self._clean_bank_direction(name)
            for name in self.accumulated_u
            if name in self.bank_a
        }
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
        """Return active-bank directions for clean scoring without slot writes."""
        if not self.accumulated_u:
            return {}
        return {
            name: self._clean_bank_direction(name)
            for name in self.accumulated_u
            if name in self.bank_a
        }

    def snapshot_tensors(
        self,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
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
        if not self.accumulated_u:
            return None, previous_flat, previous_delta
        flat_parts = []
        module_rows = []
        total_u_sq = 0.0
        total_w_sq = 0.0
        for name in sorted(self.accumulated_u):
            acc = self.accumulated_u[name].detach().float()
            used_rank = int(self.used_rank.get(name, acc.shape[1]))
            active_acc = acc[:, :used_rank]
            u_norm = float(torch.linalg.vector_norm(active_acc).item())
            active_a = self.bank_a.get(name)
            v_norm = None
            w_fro_est = None
            if active_a is not None:
                active_v = active_a[:used_rank, :].detach().float()
                v_norm = float(torch.linalg.vector_norm(active_v).item())
                w_fro_est = u_norm * v_norm
                total_w_sq += w_fro_est * w_fro_est
            total_u_sq += u_norm * u_norm
            module_rows.append(
                {
                    "name": name,
                    "u_norm": u_norm,
                    "v_norm": v_norm,
                    "delta_w_fro_est": w_fro_est,
                    "used_bank_rank": used_rank,
                }
            )
            flat_parts.append(active_acc.flatten().cpu())
        return build_update_snapshot_metrics(
            step=step,
            module_rows=module_rows,
            flat_parts=flat_parts,
            total_u_sq=total_u_sq,
            total_w_sq=total_w_sq,
            previous_flat=previous_flat,
            previous_delta=previous_delta,
            extra_metrics={
                "update_bank_rank": int(self.update_bank_rank),
                "used_bank_rank": self.max_used_rank(),
            },
            require_matching_previous_shape=True,
        )

    def max_used_rank(self) -> int:
        return max((int(value) for value in self.used_rank.values()), default=0)

    def _same_v(
        self, name: str, block: _BankBlock, direction: Mapping[str, torch.Tensor]
    ) -> bool:
        bank = self.bank_a.get(name)
        if bank is None:
            return False
        V_T = direction.get("V_T")
        lora_a = V_T if V_T is not None else direction["V"].T
        held = bank[block.start : block.start + block.rank, :]
        if tuple(held.shape) != tuple(lora_a.shape):
            if os.environ.get("ZO_BANK_SAMEV_DEBUG"):
                print(f"[same_v] {name}: shape {tuple(held.shape)} vs {tuple(lora_a.shape)}", flush=True)
            return False
        cand = lora_a.to(held.device, held.dtype)
        same = bool(torch.equal(held, cand))
        if not same and os.environ.get("ZO_BANK_SAMEV_DEBUG"):
            h = held.float().reshape(-1); c = cand.float().reshape(-1)
            cos = float((h @ c) / (h.norm() * c.norm() + 1e-30))
            print(f"[same_v] {name}: maxdiff={float((h-c).abs().max()):.3e} "
                  f"|held|={float(h.norm()):.4g} |cand|={float(c.norm()):.4g} cos={cos:.6f} "
                  f"dtype held={held.dtype} cand={lora_a.dtype} V_T_given={direction.get('V_T') is not None}",
                  flush=True)
        return same

    def _allocate_block(
        self, name: str, direction: Mapping[str, torch.Tensor]
    ) -> _BankBlock:
        U = direction["U"]
        V = direction["V"]
        V_T = direction.get("V_T")
        lora_a = V_T if V_T is not None else V.T
        block_rank = int(U.shape[1])
        if int(lora_a.shape[0]) != block_rank:
            raise RuntimeError(f"direction U/V rank mismatch for {name}")
        if block_rank <= 0:
            raise RuntimeError(f"empty direction rank for {name}")
        self._ensure_bank_tensors(name, U=U, lora_a=lora_a)
        start = int(self.used_rank.get(name, 0))
        end = start + block_rank
        if end > int(self.update_bank_rank):
            raise RuntimeError(
                f"LoRA update bank exhausted for {name}: need rank {end}, "
                f"capacity is {int(self.update_bank_rank)}. Increase --update-bank-rank "
                "or reduce the number of V refreshes."
            )
        self.bank_a[name][start:end, :].copy_(lora_a)
        if zo_bank_debug_enabled():
            self._log_single_tensor_if_bad(
                "allocate_lora_a_after_copy",
                name=name,
                tensor=self.bank_a[name][start:end, :],
                step=self.debug_step,
                extra={
                    "block_start": start,
                    "block_rank": block_rank,
                    "source_lora_a": tensor_finite_summary(lora_a),
                    "source_U": tensor_finite_summary(U),
                    "source_V": tensor_finite_summary(V),
                },
            )
        block = _BankBlock(start=start, rank=block_rank)
        self.current_blocks[name] = block
        self.used_rank[name] = end
        return block

    def _ensure_bank_tensors(
        self,
        name: str,
        *,
        U: torch.Tensor,
        lora_a: torch.Tensor,
    ) -> None:
        bank_rank = int(self.update_bank_rank)
        if name in self.bank_a:
            return
        self.bank_a[name] = torch.zeros(
            (bank_rank, int(lora_a.shape[1])),
            device=lora_a.device,
            dtype=lora_a.dtype,
        )
        self.accumulated_u[name] = torch.zeros(
            (int(U.shape[0]), bank_rank),
            device=U.device,
            dtype=U.dtype,
        )
        self.zero_u[name] = torch.zeros_like(self.accumulated_u[name])
        self.used_rank[name] = 0

    def _current_accum_slice(self, name: str, block: _BankBlock) -> torch.Tensor:
        end = int(block.start + block.rank)
        return self.accumulated_u[name][:, int(block.start) : end]

    def _bank_direction(
        self,
        name: str,
        direction: Mapping[str, torch.Tensor],
        *,
        block: _BankBlock,
        perturb: bool,
        force_a_refresh: bool,
    ) -> dict[str, torch.Tensor]:
        acc = self.accumulated_u[name]
        probe_u = self.zero_u[name]
        scale = 1.0
        if perturb:
            scale = float(direction.get("scale", 1.0))
            U = direction["U"]
            start = int(block.start)
            end = start + int(block.rank)
            probe_u = self.zero_u[name].clone()
            probe_u[:, start:end].copy_(U)
            if zo_bank_debug_enabled():
                self._log_single_tensor_if_bad(
                    "probe_u_after_copy",
                    name=name,
                    tensor=probe_u,
                    step=self.debug_step,
                    extra={
                        "block_start": start,
                        "block_rank": int(block.rank),
                        "source_U": tensor_finite_summary(U),
                        "zero_u": tensor_finite_summary(self.zero_u[name]),
                        "accumulated_u": tensor_finite_summary(
                            self.accumulated_u[name]
                        ),
                    },
                )
        return {
            "U": probe_u,
            "U_accum": acc,
            "V": self.bank_a[name].T,
            "V_T": self.bank_a[name],
            "scale": scale if perturb else 1.0,
            "v_refreshed": bool(force_a_refresh),
        }

    def _log_direction_finite_summary(
        self,
        stage: str,
        directions: Mapping[str, Mapping[str, Any]],
        *,
        step: int | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        summary = direction_tensors_finite_summary(directions)
        if int(summary["num_bad_tensors"]) == 0:
            return
        payload = {
            "stage": stage,
            "step": step,
            "update_bank_rank": int(self.update_bank_rank),
            "used_bank_rank": self.max_used_rank(),
            "summary": summary,
        }
        if extra:
            payload["extra"] = extra
        logger.error("ZO bank state nonfinite tensors: %s", payload)

    def _log_single_tensor_if_bad(
        self,
        stage: str,
        *,
        name: str,
        tensor: torch.Tensor,
        step: int | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        summary = tensor_finite_summary(tensor)
        if bool(summary.get("finite", True)):
            return
        payload = {
            "stage": stage,
            "step": step,
            "name": name,
            "summary": summary,
            "update_bank_rank": int(self.update_bank_rank),
            "used_bank_rank": self.max_used_rank(),
        }
        if extra:
            payload["extra"] = extra
        logger.error("ZO bank state tensor became nonfinite: %s", payload)

    def _log_storage_finite_summary(self, stage: str, *, step: int | None) -> None:
        bad: list[dict[str, Any]] = []
        groups = {
            "bank_a": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
            "accumulated_u": {
                "num_tensors": 0,
                "num_bad_tensors": 0,
                "first_bad": None,
            },
            "pending_u": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
            "zero_u": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
        }
        containers = (
            ("bank_a", self.bank_a),
            ("accumulated_u", self.accumulated_u),
            ("pending_u", self.pending_u),
            ("zero_u", self.zero_u),
        )
        for group_name, tensors in containers:
            for name, tensor in tensors.items():
                groups[group_name]["num_tensors"] = (
                    int(groups[group_name]["num_tensors"]) + 1
                )
                summary = tensor_finite_summary(tensor)
                if bool(summary.get("finite", True)):
                    continue
                item = {
                    "group": group_name,
                    "name": name,
                    **summary,
                    "used_rank": int(self.used_rank.get(name, 0)),
                    "current_block": None,
                }
                block = self.current_blocks.get(name)
                if block is not None:
                    item["current_block"] = {
                        "start": int(block.start),
                        "rank": int(block.rank),
                    }
                groups[group_name]["num_bad_tensors"] = (
                    int(groups[group_name]["num_bad_tensors"]) + 1
                )
                if groups[group_name]["first_bad"] is None:
                    groups[group_name]["first_bad"] = item
                if len(bad) < 8:
                    bad.append(item)
        num_bad = sum(int(row["num_bad_tensors"]) for row in groups.values())
        if num_bad == 0:
            return
        logger.error(
            "ZO bank storage nonfinite tensors: %s",
            {
                "stage": stage,
                "step": step,
                "update_bank_rank": int(self.update_bank_rank),
                "used_bank_rank": self.max_used_rank(),
                "num_bad_tensors": num_bad,
                "bad": bad,
                "groups": groups,
            },
        )

    def _clean_bank_direction(self, name: str) -> dict[str, torch.Tensor]:
        acc = self.accumulated_u[name]
        return {
            "U": self.zero_u[name],
            "U_accum": acc,
            "V": self.bank_a[name].T,
            "V_T": self.bank_a[name],
            "v_refreshed": True,
        }

    def _cap_accumulated_u_norm(self) -> tuple[float | None, float | None]:
        if self.u_norm_cap is None or not self.accumulated_u:
            return None, None
        active_values = []
        for name, acc in self.accumulated_u.items():
            used_rank = int(self.used_rank.get(name, 0))
            if used_rank > 0:
                active_values.append(acc[:, :used_rank])
        if not active_values:
            return None, None
        first = active_values[0]
        total_sq = torch.zeros((), device=first.device, dtype=torch.float32)
        for acc in active_values:
            total_sq = total_sq + acc.detach().float().square().sum()
        total_norm = float(total_sq.sqrt().item())
        cap = float(self.u_norm_cap)
        if total_norm <= cap or total_norm <= 0.0:
            return 1.0, total_norm
        scale = cap / total_norm
        torch._foreach_mul_(active_values, scale)
        return scale, cap
