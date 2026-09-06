"""Low-rank update state shared by LOZO and AGZO trainers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import time
from typing import Any, Protocol

import torch

from zo_vllm.training.u_optimizer import UCoefficientOptimizer

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
    u_momentum: float = 0.0
    u_optimizer: str = "sgd"
    u_beta2: float = 0.9
    u_adam_eps: float = 1e-8
    velocity_u: dict[str, torch.Tensor] = field(default_factory=dict)
    second_moment_u: dict[str, torch.Tensor] = field(default_factory=dict)
    _adam_step: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.u_momentum) < 1.0):
            raise ValueError("u_momentum must be in [0, 1)")
        if self.u_optimizer not in {"sgd", "adam"}:
            raise ValueError("u_optimizer must be 'sgd' or 'adam'")
        if not (0.0 <= float(self.u_beta2) < 1.0):
            raise ValueError("u_beta2 must be in [0, 1)")

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
        payload = dict(directions)
        coefficient = float(projected_grad)
        if self.u_optimizer == "adam":
            # The reference ZO implementations write the estimate into
            # param.grad and hand it to a stock optimizer, so the second-moment
            # normalisation comes for free. This path applies the update itself,
            # so Adam has to be written out. Normalising by sqrt(v) is what lets
            # these methods run a learning rate ~50x the SGD one: the raw
            # coefficient here swings between 0.3 and 245 step to step, and
            # heavy-ball alone passes that swing straight through.
            self._adam_step += 1
            b1, b2 = float(self.u_momentum), float(self.u_beta2)
            bc1 = 1.0 - b1 ** self._adam_step
            bc2 = 1.0 - b2 ** self._adam_step
            carried = {}
            for name, direction in payload.items():
                U = direction["U"]
                # The moments are kept in fp32 regardless of the model dtype:
                # in fp16 both eps=1e-8 and (1-beta2)*g^2 for small g underflow
                # to exactly 0, so the denominator vanishes and the first step
                # is +-inf (0/0 -> NaN where the gradient is exactly 0).
                g = (U * (coefficient * float(direction.get("scale", 1.0)))).to(
                    torch.float32
                )
                m = self.velocity_u.get(name)
                v = self.second_moment_u.get(name)
                if m is None or m.shape != g.shape:
                    m = torch.zeros_like(g)
                    self.velocity_u[name] = m
                if v is None or v.shape != g.shape:
                    v = torch.zeros_like(g)
                    self.second_moment_u[name] = v
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                step = ((m / bc1) / ((v / bc2).sqrt() + float(self.u_adam_eps))).to(
                    U.dtype
                )
                moved = dict(direction)
                moved["U"] = step
                moved["scale"] = 1.0
                carried[name] = moved
            payload = carried
            coefficient = 1.0
        elif float(self.u_momentum) > 0.0:
            # This path writes each update straight into the weights and keeps
            # no U state, so momentum needs its own. Carrying a velocity turns
            # the per-step estimate into a running sum, which is what a residual
            # field whose estimates scatter step to step needs; the coefficient
            # is folded into the velocity, so the update applies it at c = 1.
            carried = {}
            for name, direction in payload.items():
                U = direction["U"]
                vel = self.velocity_u.get(name)
                if vel is None or vel.shape != U.shape:
                    vel = torch.zeros_like(U)
                    self.velocity_u[name] = vel
                vel.mul_(float(self.u_momentum)).add_(
                    U, alpha=coefficient * float(direction.get("scale", 1.0))
                )
                moved = dict(direction)
                moved["U"] = vel
                moved["scale"] = 1.0
                carried[name] = moved
            payload = carried
            coefficient = 1.0
        return self.weight_sync.apply_lozo_update(
            payload,
            c=coefficient,
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
    u_momentum: float = 0.0
    u_optimizer: str = "sgd"
    u_beta2: float = 0.9
    u_adam_eps: float = 1e-8
    u_opt: UCoefficientOptimizer | None = None
    u_norm_cap: float | None = None
    gradient_accumulation_update_steps: int = 0
    accumulated_u: dict[str, torch.Tensor] = field(default_factory=dict)
    pending_u: dict[str, torch.Tensor] = field(default_factory=dict)
    v_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    vt_cache: dict[str, torch.Tensor | None] = field(default_factory=dict)
    # Heavy-ball velocity kept in weight space as a queue of past rank-r
    # increments. Under random_full the V basis changes every step, so a
    # velocity kept in U-space (as the bank does with a frozen V) would be
    # expressed in a basis that no longer exists by the next step. Each entry
    # is (age, {name: (U_k, V_k)}); the fold applies sum_k beta^age_k U_k V_k^T,
    # which is exactly v_t = beta v_{t-1} + g_t d_t unrolled and truncated once
    # beta^age drops below momentum_tol.
    momentum_queue: list[tuple[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]] = field(
        default_factory=list
    )
    momentum_tol: float = 1e-3
    # ZO-AdaMU (Jiang et al., AAAI 2024), ported to the low-rank factors.
    # The paper relocates momentum from the gradient to the perturbation: the
    # next z is drawn centred on sign(g_t) * m_t, mixed with a fresh Gaussian by
    # an annealed beta_1, and the step is m / sqrt(v). Here z = U V^T, so the
    # same recipe is applied to each factor and the rank-1 product of the
    # mixed factors is the perturbation. Their schedule (warmup 1024, cosine
    # to 0.8 T, shared shrinking limit across the three varphi calls) is kept
    # verbatim, evaluated once per step rather than once per parameter.
    u_total_steps: int = 0
    u_adamu_seed: int = 0
    adamu_hist: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    adamu_mv: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    adamu_step: int = 0
    adamu_limit: float = 0.0
    adamu_gen: torch.Generator | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.u_beta) <= 1.0):
            raise ValueError("u_beta must be in [0, 1]")
        if str(self.u_optimizer) == "zo_adamu":
            if int(self.u_total_steps) <= 0:
                raise ValueError("zo_adamu needs u_total_steps for its anneal schedule")
            self.adamu_limit = float(self.u_total_steps)
            self.adamu_gen = torch.Generator().manual_seed(int(self.u_adamu_seed))
        # Momentum on this state lives in the weight-space queue, not in the
        # coefficient optimizer, so the optimizer runs momentum-free here.
        self.u_opt = UCoefficientOptimizer(
            name=str(self.u_optimizer),
            momentum=0.0 if str(self.u_optimizer) == "sgd" else float(self.u_momentum),
            beta2=float(self.u_beta2),
            eps=float(self.u_adam_eps),
        )
        if not (0.0 <= float(self.u_momentum) < 1.0):
            raise ValueError("u_momentum must be in [0, 1)")
        if self.u_norm_cap is not None and float(self.u_norm_cap) <= 0.0:
            raise ValueError("u_norm_cap must be positive when set")
        if int(self.gradient_accumulation_update_steps) < 0:
            raise ValueError("gradient_accumulation_update_steps must be non-negative")

    def _adamu_ema(self, varphi: float) -> float:
        """ZO-AdaMU's `_ema_weight`, with the shared shrinking limit."""

        t = int(self.adamu_step)
        T = int(self.u_total_steps)
        w1 = min(1024, max(1, T // 8))
        w2 = int(T * 0.8)
        if t < w1:
            return 1.0
        if t < w2:
            val = 0.5 * (1.0 + math.cos(math.pi * ((t - w1) / self.adamu_limit)))
            self.adamu_limit -= varphi * (T - w2) / max(1, (w2 - w1))
            return val
        return 0.5 * (1.0 + math.cos(math.pi * ((w2 - w1) / self.adamu_limit)))

    @torch.no_grad()
    def shape_directions(self, directions: DirectionMap) -> None:
        """Replace the sampled factors with their momentum-centred mixtures.

        Called once per step before the plus/minus scoring, and mutates the
        direction map in place so scoring and the update see the same z.
        """

        if str(self.u_optimizer) != "zo_adamu":
            return
        self.adamu_step += 1
        alpha = self._adamu_ema(1.0)
        b1 = self._adamu_ema(0.1)
        b2 = self._adamu_ema(1.5)
        gen = self.adamu_gen
        for name, direction in dict(directions).items():
            U = direction["U"].detach().float()
            V = direction["V"].detach().float()
            hU, hV = self.adamu_hist.get(name, (torch.zeros_like(U), torch.zeros_like(V)))
            noise_U = torch.randn(U.shape, generator=gen, dtype=torch.float32).to(U.device)
            noise_V = torch.randn(V.shape, generator=gen, dtype=torch.float32).to(V.device)
            Uh = hU.to(U.device) + math.sqrt(1.0 - alpha) * noise_U
            Vh = hV.to(V.device) + math.sqrt(1.0 - alpha) * noise_V
            Uc = math.sqrt(alpha) * U
            Vc = math.sqrt(alpha) * V
            mU = b1 * Uc + (1.0 - b1) * Uh
            vU = b2 * Uc.square() + (1.0 - b2) * Uh.square()
            mV = b1 * Vc + (1.0 - b1) * Vh
            vV = b2 * Vc.square() + (1.0 - b2) * Vh.square()
            self.adamu_mv[name] = (mU, vU, mV, vV)
            direction["U"] = mU.to(direction["U"].dtype)
            direction["V"] = mV.to(direction["V"].dtype)
            direction["V_T"] = None

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
        self.u_opt.begin_step()
        use_torch_opt = self.u_opt.uses_torch
        batch_grads: dict[str, torch.Tensor] = {}
        batch_targets: dict[str, torch.Tensor] = {}
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
            if use_torch_opt:
                batch_targets[name] = target_u
                batch_grads[name] = direction["U"] * (
                    float(projected_grad) * scale
                )
                continue
            if self.u_opt.name == "zo_adamu":
                mU, vU, mV, vV = self.adamu_mv[name]
                eps = float(self.u_adam_eps)
                dU = mU / (vU.sqrt() + eps)
                dV = mV / (vV.sqrt() + eps)
                target_u.add_(
                    dU.to(target_u.dtype),
                    alpha=-float(learning_rate) * float(projected_grad) * scale,
                )
                # the fold pairs accumulated_u with v_cache, so the normalised
                # V factor has to be what the fold sees
                self.v_cache[name] = dV.to(direction["V"].dtype)
                self.vt_cache[name] = None
                sgn = 1.0 if float(projected_grad) >= 0.0 else -1.0
                self.adamu_hist[name] = (sgn * mU, mV)
            elif self.u_opt.name == "adam_scalar":
                rate = self.u_opt.scalar_rate(
                    name, float(projected_grad) * scale
                )
                target_u.add_(direction["U"], alpha=-float(learning_rate) * rate)
            else:
                target_u.add_(
                    direction["U"],
                    alpha=-float(learning_rate) * float(projected_grad) * scale,
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
            "mode": "accumulate",
            "accumulate_s": accumulate_s,
            "pending_flush_s": pending_flush_s,
            "gradient_accumulation_update_steps": update_interval,
            "fold_s": 0.0,
            "num_accumulated_modules": len(self.accumulated_u),
            "num_pending_modules": len(self.pending_u),
            "u_beta": float(self.u_beta),
            "u_momentum": float(self.u_momentum),
            "u_optimizer": str(self.u_optimizer),
            "u_beta2": float(self.u_beta2),
            "u_optimizer_code": self.u_opt.code,
            "momentum_queue_len": len(self.momentum_queue),
            "u_norm_cap": None if self.u_norm_cap is None else float(self.u_norm_cap),
            "u_cap_scale": cap_scale,
            "u_norm_after_cap": capped_norm,
        }

    def fold_before_direction_refresh(self, *, step: int) -> float:
        """Fold accumulated U before the provider replaces the current V basis.

        The velocity survives the refresh. It lives in the U-space, whose
        dimension does not change with V, and the measured drift of V over 8000
        steps is a median 13 degrees on most targets -- small enough that the
        mixing this introduces costs less than throwing away the accumulation
        that a slow residual field needs.
        """

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
        beta = float(self.u_momentum)
        if beta > 0.0:
            fold_directions = self._with_momentum_queue(fold_directions, beta)
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

    def _with_momentum_queue(
        self,
        fresh: dict[str, dict[str, torch.Tensor]],
        beta: float,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Stack the fresh increment with beta^age-scaled past increments.

        The fresh increment enters the queue at age 0 with weight 1, matching
        the term ``g_t d_t`` of ``v_t = beta v_{t-1} + g_t d_t``; every older
        entry contributes ``beta^age U_k V_k^T``. Stacking along the rank axis
        turns the whole velocity into one rank-K fold instead of K rank-1 ones.
        """

        snapshot = {
            name: (d["U"].detach().clone(), d["V"].detach().clone())
            for name, d in fresh.items()
        }
        self.momentum_queue.append((0, snapshot))
        # age every entry, drop the ones whose weight is below tolerance
        kept: list[tuple[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]] = []
        for age, entry in self.momentum_queue:
            if beta ** age >= float(self.momentum_tol):
                kept.append((age, entry))
        self.momentum_queue = [(age + 1, entry) for age, entry in kept]

        stacked: dict[str, dict[str, torch.Tensor]] = {}
        for name in fresh:
            us = []
            vs = []
            for age, entry in kept:
                if name not in entry:
                    continue
                U_k, V_k = entry[name]
                us.append(U_k * (beta ** age))
                vs.append(V_k)
            stacked[name] = {
                "U": torch.cat(us, dim=1),
                "V": torch.cat(vs, dim=1),
                "V_T": None,
            }
        return stacked

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
