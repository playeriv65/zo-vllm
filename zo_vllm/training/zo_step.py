"""Generic zeroth-order stepper shared by LOZO and AGZO."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import math
import time

import torch
from typing import Any

from zo_vllm.engine import ZOVLLMEngine
from zo_vllm.core.probe_results import ProbeLossResult

from .direction import DirectionProvider, DirectionSample, ProbeBatch, TokenProbeBatch
from .estimator import (
    DirectionBundle,
    SingleDirectionAntitheticEstimator,
    ZOEstimate,
    ZOEstimator,
    ZOEstimatorConfig,
    _direction_info_from_specs,
    _sample_direction_from_specs,
)
from zo_vllm.core.token_scores import TokenGroupScorer
from .update_state import ZOUpdateState


@dataclass(frozen=True)
class ZOStepConfig:
    """Algorithm settings that are independent of the direction source."""

    eps: float = 1e-3
    max_logits_tokens: int = 8192
    loss_impl: str = "logprobs"
    score_chunk_size: int = 0

    def __post_init__(self) -> None:
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if int(self.score_chunk_size) < 0:
            raise ValueError("score_chunk_size must be non-negative")


@dataclass(frozen=True)
class ZOStepResult:
    """One generic ZO train-step result."""

    step: int
    learning_rate: float
    reported_loss: float
    loss_plus: float
    loss_minus: float
    projected_grad: float | None
    update_scale: float
    direction_refreshed: bool
    direction_info: dict[str, Any] = field(default_factory=dict)
    profile_s: dict[str, float] = field(default_factory=dict)
    probe_metrics: dict[str, Any] = field(default_factory=dict)
    estimator_metrics: dict[str, Any] = field(default_factory=dict)
    update_info: dict[str, Any] = field(default_factory=dict)

    def metrics(self) -> dict[str, float | int | bool]:
        values: dict[str, float | int | bool] = {
            "step": int(self.step),
            "applied_learning_rate": float(self.learning_rate),
            "loss": float(self.reported_loss),
            "loss_plus": float(self.loss_plus),
            "loss_minus": float(self.loss_minus),
            "update_scale": float(self.update_scale),
            "direction_refreshed": bool(self.direction_refreshed),
        }
        if self.projected_grad is not None:
            values["projected_grad"] = float(self.projected_grad)
        for key, value in self.direction_info.items():
            if isinstance(value, (float, int, bool)):
                values[key] = value
        for key, value in self.estimator_metrics.items():
            if isinstance(value, (float, int, bool)):
                values[f"estimator_{key}"] = value
        for key, value in self.probe_metrics.items():
            if isinstance(value, (float, int, bool)):
                values[f"probe_{key}"] = value
        if "perturbation_effective_rms" in values and "estimator_sigma" in values:
            values["perturbation_sigma_effective_rms"] = float(
                values["perturbation_effective_rms"]
            ) * float(values["estimator_sigma"])
        for key, value in self.profile_s.items():
            if isinstance(value, (float, int, bool)):
                values[f"profile_{key}_s"] = value
        for key, value in self.update_info.items():
            if isinstance(value, (float, int, bool)):
                values[f"update_{key}"] = value
        return values


def validate_zo_update_hparams(
    learning_rate: float,
    weight_decay: float,
) -> tuple[float, float]:
    """Validate optimizer-owned update hyperparameters once for every executor."""

    lr = float(learning_rate)
    decay = float(weight_decay)
    if lr < 0.0 or not math.isfinite(lr):
        raise ValueError("learning_rate must be non-negative and finite")
    if decay < 0.0 or not math.isfinite(decay):
        raise ValueError("weight_decay must be non-negative and finite")
    return lr, decay


def build_zo_step_result(
    *,
    step: int,
    learning_rate: float,
    estimate: ZOEstimate,
    direction_refreshed: bool,
    direction_info: dict[str, Any] | None = None,
    profile_s: dict[str, float] | None = None,
    update_info: dict[str, Any] | None = None,
) -> ZOStepResult:
    """Build the common step result after either sync or async mutation."""

    merged_direction_info = dict(direction_info or {})
    merged_direction_info.update(dict(estimate.direction_info))
    merged_profile = dict(estimate.profile_s)
    merged_profile.update(dict(profile_s or {}))
    return ZOStepResult(
        step=int(step),
        learning_rate=float(learning_rate),
        reported_loss=float(estimate.reported_loss),
        loss_plus=float(estimate.loss_plus),
        loss_minus=float(estimate.loss_minus),
        projected_grad=estimate.projected_grad,
        update_scale=float(estimate.gradient.scale),
        direction_refreshed=bool(direction_refreshed),
        direction_info=merged_direction_info,
        profile_s=merged_profile,
        probe_metrics=dict(estimate.probe_metrics),
        estimator_metrics=dict(estimate.estimator_metrics),
        update_info=dict(update_info or {}),
    )


@dataclass
class ZOPendingStep:
    """One estimated ZO step waiting for an optimizer update."""

    step: int
    reported_loss: float
    _apply_fn: Callable[[float, float], ZOStepResult] = field(repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def apply(self, *, learning_rate: float, weight_decay: float) -> ZOStepResult:
        if self._consumed:
            raise RuntimeError("pending ZO step has already been applied")
        lr, decay = validate_zo_update_hparams(learning_rate, weight_decay)
        self._consumed = True
        return self._apply_fn(lr, decay)


@dataclass(frozen=True)
class ZOProbeExecution:
    """Executor-independent observations for one completed probe plan."""

    losses: ProbeLossResult
    direction_refreshed: bool = False
    direction_info: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZOStepControl:
    """Mutable control flags for callbacks inside one ZO optimization step."""

    should_skip_update: bool = False


class ZOStepCallback:
    """Callbacks for algorithm-level ZO step events."""

    def on_direction_sampled(
        self,
        *,
        step: int,
        batch: ProbeBatch,
        sample: Any,
        control: ZOStepControl,
    ) -> ZOStepControl | None:
        return None

    def on_score_end(
        self,
        *,
        step: int,
        batch: ProbeBatch,
        sample: Any,
        score: ZOEstimate,
        control: ZOStepControl,
    ) -> ZOStepControl | None:
        return None

    def on_update_begin(
        self,
        *,
        step: int,
        batch: ProbeBatch,
        sample: Any,
        score: ZOEstimate,
        learning_rate: float,
        control: ZOStepControl,
    ) -> ZOStepControl | None:
        return None

    def on_update_end(
        self,
        *,
        step: int,
        batch: ProbeBatch,
        sample: Any,
        score: ZOEstimate,
        learning_rate: float,
        update_info: dict[str, Any],
        control: ZOStepControl,
    ) -> ZOStepControl | None:
        return None


@dataclass
class SVRGState:
    """MeZO-SVRG (Gautam et al., ICML 2024), Algorithm 1, on the rank-1 path.

    Every ``q`` steps the anchor step estimates the directional derivative on a
    large batch, snapshots the weights and steps with ``eta_1 = anchor_lr_ratio
    * eta_2``. In between, each mini step estimates the same direction at the
    current weights and at the snapshot, and steps along
    ``g_I(theta) - g_I(theta_bar) + g_anchor`` with ``eta_2``. Since the
    snapshot the weights differ by a handful of rank-1 increments, so the
    snapshot is reached by un-folding those increments in place, scoring, and
    folding them back: three rank-K folds instead of a copy of the weights.

    Both mini-batch estimates share one perturbation z, which is what makes
    their difference a control variate; the reference code draws a fresh seed
    for each, which we do not follow.
    """

    q: int = 2
    anchor_lr_ratio: float = 5.0
    # Algorithm 4 of the paper: once per window of ``anneal_window`` steps,
    # compare the mean training loss of the last window with the one before;
    # if the ratio exceeds ``anneal_kappa`` divide both learning rates by
    # ``anneal_alpha``. ``lr_scale`` carries the accumulated division.
    anneal_alpha: float = 5.0
    anneal_kappa: float = 1.05
    anneal_window: int = 0
    lr_scale: float = 1.0
    losses: list[float] = field(default_factory=list)
    anchor: dict[str, dict[str, torch.Tensor]] | None = None
    since_snapshot: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=list
    )
    last: dict[str, float] = field(default_factory=dict)

    def is_anchor(self, step: int) -> bool:
        return int(self.q) <= 1 or (int(step) - 1) % int(self.q) == 0

    def observe_loss(self, loss: float, step: int) -> None:
        """Record a training loss and apply the window-based anneal."""

        w = int(self.anneal_window)
        if w <= 0:
            return
        self.losses.append(float(loss))
        if len(self.losses) >= 2 * w and int(step) % w == 0:
            recent = sum(self.losses[-w:]) / w
            before = sum(self.losses[-2 * w : -w]) / w
            if before > 0 and recent / before > float(self.anneal_kappa):
                self.lr_scale /= float(self.anneal_alpha)
                self.last["svrg_annealed"] = 1.0


class _ReplaySampler:
    """Hand the estimator the bundle already sampled this step."""

    def __init__(self, bundle: "DirectionBundle") -> None:
        self._bundle = bundle

    def sample(
        self, *, seed: int | None = None, probe: int | None = None
    ) -> "DirectionBundle":
        del seed, probe
        return self._bundle


class _StepDirectionSampler:
    """Step-local direction sampler used by estimators."""

    def __init__(
        self,
        *,
        stepper: "ZOStepper",
        batch: ProbeBatch,
        step: int,
        perturbation_normalization: str,
    ) -> None:
        self.stepper = stepper
        self.batch = batch
        self.step = int(step)
        self.perturbation_normalization = perturbation_normalization
        self.first_sample: DirectionSample | None = None
        self.first_bundle: DirectionBundle | None = None
        self._refreshed = False
        self._direction_info: dict[str, Any] = {}
        self.profile_s: dict[str, float] = {}

    def sample(
        self, *, seed: int | None = None, probe: int | None = None
    ) -> DirectionBundle:
        sample_t0 = time.perf_counter()
        provider_t0 = time.perf_counter()
        provider = self.stepper.direction_provider
        if probe is not None and hasattr(provider, "u_provider"):
            # factorised provider: same V*, U re-drawn per probe
            sample = provider.next(self.batch, step=self.step, probe=int(probe))
        elif seed is None:
            sample = provider.next(self.batch, step=self.step)
            if self.stepper._force_direction_slot_sync:
                for direction in sample.directions.values():
                    direction["v_refreshed"] = True
                sample = DirectionSample(
                    directions=sample.directions,
                    refreshed=True,
                    info=dict(sample.info),
                )
                self.stepper._force_direction_slot_sync = False
        else:
            directions = self.sample_raw(seed=int(seed))
            sample = DirectionSample(
                directions=directions,
                refreshed=True,
                info={
                    "direction_provider": "spec",
                    "perturb_seed": int(seed),
                    **_direction_info_from_specs(
                        self.stepper.direction_provider.direction_specs(),
                        perturbation_normalization=self.perturbation_normalization,
                    ),
                },
            )
        self.profile_s["direction_sample_provider_next"] = (
            time.perf_counter() - provider_t0
        )
        record_t0 = time.perf_counter()
        self._record_sample(sample)
        self.profile_s["direction_sample_record"] = time.perf_counter() - record_t0
        prepare_t0 = time.perf_counter()
        shape = getattr(self.stepper.update_state, "shape_directions", None)
        if callable(shape):
            # perturbation-side transforms (ZO-AdaMU) must act before the
            # plus/minus scoring so the probe and the update share one z
            shape(sample.directions)
        score_directions = self.stepper.update_state.prepare_for_score(
            sample.directions
        )
        self.profile_s["direction_sample_prepare_for_score"] = (
            time.perf_counter() - prepare_t0
        )
        self.profile_s["direction_sample_total"] = time.perf_counter() - sample_t0
        bundle = DirectionBundle(
            directions=sample.directions,
            score_directions=score_directions,
        )
        if self.first_bundle is None:
            self.first_bundle = bundle
        return bundle

    def sample_raw(self, *, seed: int) -> dict[str, dict[str, Any]]:
        return _sample_direction_from_specs(
            self.stepper.direction_provider.direction_specs(),
            seed=int(seed),
            perturbation_normalization=self.perturbation_normalization,
        )

    @property
    def refreshed(self) -> bool:
        return self._refreshed

    @property
    def direction_info(self) -> dict[str, Any]:
        return dict(self._direction_info)

    def _record_sample(self, sample: DirectionSample) -> None:
        if self.first_sample is None:
            self.first_sample = sample
        self._refreshed = self._refreshed or bool(sample.refreshed)
        self._direction_info.update(sample.info)
        self.stepper._call_event(
            "on_direction_sampled",
            step=self.step,
            batch=self.batch,
            sample=sample,
        )


class ZOStepper:
    """Shared ZO update loop for any low-rank direction provider."""

    def __init__(
        self,
        *,
        engine: ZOVLLMEngine,
        direction_provider: DirectionProvider,
        update_state: ZOUpdateState,
        config: ZOStepConfig,
        estimator: ZOEstimator | None = None,
        callbacks: Sequence[ZOStepCallback] | None = None,
    ) -> None:
        self.engine = engine
        self.direction_provider = direction_provider
        self.update_state = update_state
        self.config = config
        self.estimator = estimator or SingleDirectionAntitheticEstimator()
        self.callbacks = list(callbacks or [])
        self.control = ZOStepControl()
        self._force_direction_slot_sync = False
        self.svrg: SVRGState | None = None

    def invalidate_direction_slot_state(self) -> None:
        """Force cached direction factors into newly created runtime slots."""

        self._force_direction_slot_sync = True

    def estimate(self, batch: ProbeBatch, *, step: int) -> ZOPendingStep:
        """Estimate one runtime-native ZO step without applying it."""

        return self._estimate_with_engine(batch, step=step, engine=self.engine)

    def estimate_with_score_fn(
        self,
        batch: TokenProbeBatch,
        *,
        step: int,
        score_fn: Callable[..., Any],
    ) -> ZOPendingStep:
        """Estimate one ZO step using caller-provided HF-native scoring.

        Estimator semantics stay unchanged: they still call engine-style methods
        such as ``score_plus_minus_directions`` and ``score_token_groups``. This
        adapter only replaces token scoring with a Hugging Face loss path while
        keeping LoRA slot selection and direction writes inside the runtime.
        """

        return self._estimate_with_engine(
            batch,
            step=step,
            engine=self.engine,
            scorer=_CallableProbeScorer(score_fn),
        )

    def _estimate_with_engine(
        self,
        batch: ProbeBatch,
        *,
        step: int,
        engine: Any,
        scorer: TokenGroupScorer | None = None,
    ) -> ZOPendingStep:
        profile_t0 = time.perf_counter()
        step_i = int(step)
        if step_i <= 0:
            raise ValueError("step must be positive")
        self.control = ZOStepControl()
        refresh_fold_s = 0.0
        refresh_t0 = time.perf_counter()
        will_refresh = getattr(self.direction_provider, "will_refresh", None)
        if callable(will_refresh) and bool(will_refresh(step=step_i)):
            fold = getattr(self.update_state, "fold_before_direction_refresh", None)
            if callable(fold):
                refresh_fold_s = float(fold(step=step_i))
        refresh_check_s = time.perf_counter() - refresh_t0
        sampler_t0 = time.perf_counter()
        sampler = _StepDirectionSampler(
            stepper=self,
            batch=batch,
            step=step_i,
            perturbation_normalization=str(
                getattr(self.estimator, "perturbation_normalization", "rms")
            ),
        )
        sampler_setup_s = time.perf_counter() - sampler_t0
        estimate_t0 = time.perf_counter()
        estimate = self._estimate(
            batch, sampler, step=step_i, engine=engine, scorer=scorer
        )
        estimate_s = time.perf_counter() - estimate_t0
        sample = sampler.first_sample
        if sample is None:
            sample = DirectionSample(directions={}, refreshed=False, info={})
        if self.svrg is not None and sampler.first_bundle is not None:
            estimate = self._svrg_adjust(
                estimate,
                sampler.first_bundle,
                batch,
                step=step_i,
                engine=engine,
                scorer=scorer,
            )
        self._call_event(
            "on_score_end",
            step=step_i,
            batch=batch,
            sample=sample,
            score=estimate,
        )
        pending_ready_t0 = time.perf_counter()
        return ZOPendingStep(
            step=step_i,
            reported_loss=float(estimate.reported_loss),
            _apply_fn=lambda learning_rate, weight_decay: self._apply_estimate(
                batch=batch,
                step=step_i,
                sample=sample,
                estimate=estimate,
                sampler=sampler,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                refresh_fold_s=refresh_fold_s,
                refresh_check_s=refresh_check_s,
                sampler_setup_s=sampler_setup_s,
                estimate_s=estimate_s,
                optimizer_wait_s=time.perf_counter() - pending_ready_t0,
                profile_t0=profile_t0,
            ),
        )

    def _svrg_fold(self, increments, *, sign: float) -> None:
        """Apply ``sign * sum_k U_k V_k^T`` for the recorded increments."""

        if not increments:
            return
        stacked: dict[str, dict[str, torch.Tensor]] = {}
        names = set().union(*(inc.keys() for inc in increments))
        for name in names:
            us = [inc[name][0] for inc in increments if name in inc]
            vs = [inc[name][1] for inc in increments if name in inc]
            stacked[name] = {
                "U": torch.cat(us, dim=1),
                "V": torch.cat(vs, dim=1),
                "V_T": None,
            }
        st = self.update_state
        # apply_lozo_update does W <- W - lr * c * U V^T
        st.weight_sync.apply_lozo_update(
            stacked,
            c=-float(sign),
            lr=1.0,
            weight_decay=0.0,
            precision=st.precision,
            sync_device=bool(st.sync_device),
            qkv_update_mode=st.qkv_update_mode,
        )

    def _svrg_adjust(self, estimate, bundle, batch, *, step, engine, scorer):
        svrg = self.svrg
        assert svrg is not None
        svrg.observe_loss(float(estimate.reported_loss), step)
        g = float(estimate.gradient.scale)
        if svrg.is_anchor(step):
            # snapshot: the large-batch estimate along this step's direction
            svrg.anchor = {
                name: {
                    "U": (d["U"] * (g * float(d.get("scale", 1.0)))).detach().clone(),
                    "V": d["V"].detach().clone(),
                }
                for name, d in bundle.directions.items()
            }
            svrg.since_snapshot = []
            svrg.last = {
                "svrg_anchor": 1.0,
                "svrg_g": g,
                "svrg_lr_scale": svrg.lr_scale,
            }
            # eta_1 = ratio * eta_2, folded into the coefficient, times the anneal
            return replace(
                estimate,
                gradient=replace(
                    estimate.gradient,
                    scale=g * float(svrg.anchor_lr_ratio) * float(svrg.lr_scale),
                ),
            )
        # mini step: same z, scored at the snapshot
        self._svrg_fold(svrg.since_snapshot, sign=-1.0)
        try:
            est_bar = self._estimate(
                batch, _ReplaySampler(bundle), step=step, engine=engine, scorer=scorer
            )
        finally:
            self._svrg_fold(svrg.since_snapshot, sign=+1.0)
        g_bar = float(est_bar.gradient.scale)
        svrg.last = {
            "svrg_anchor": 0.0,
            "svrg_g": g,
            "svrg_g_bar": g_bar,
            "svrg_g_diff": g - g_bar,
            "svrg_lr_scale": svrg.lr_scale,
        }
        return replace(
            estimate,
            gradient=replace(
                estimate.gradient, scale=(g - g_bar) * float(svrg.lr_scale)
            ),
        )

    def _svrg_after_apply(self, *, learning_rate: float, step: int) -> None:
        """Record what just entered the weights; add the anchor term on mini steps."""

        svrg = self.svrg
        assert svrg is not None
        st = self.update_state
        inc: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, acc in st.accumulated_u.items():
            V = st.v_cache.get(name)
            if V is None:
                continue
            inc[name] = (acc.detach().clone(), V.detach().clone())
        if inc:
            svrg.since_snapshot.append(inc)
        if svrg.is_anchor(step) or svrg.anchor is None:
            return
        anchor_inc = {
            name: ((-float(learning_rate) * float(svrg.lr_scale)) * d["U"], d["V"])
            for name, d in svrg.anchor.items()
        }
        self._svrg_fold([anchor_inc], sign=+1.0)
        svrg.since_snapshot.append(anchor_inc)

    def _apply_estimate(
        self,
        *,
        batch: ProbeBatch,
        step: int,
        sample: DirectionSample,
        estimate: ZOEstimate,
        sampler: "_StepDirectionSampler",
        learning_rate: float,
        weight_decay: float,
        refresh_fold_s: float,
        refresh_check_s: float,
        sampler_setup_s: float,
        estimate_s: float,
        optimizer_wait_s: float,
        profile_t0: float,
    ) -> ZOStepResult:
        gradient = estimate.gradient
        callbacks_pre_t0 = time.perf_counter()
        self._call_event(
            "on_update_begin",
            step=step,
            batch=batch,
            sample=sample,
            score=estimate,
            learning_rate=learning_rate,
        )
        callbacks_pre_update_s = time.perf_counter() - callbacks_pre_t0
        apply_t0 = time.perf_counter()
        if self.control.should_skip_update:
            update_info = {"skipped_by_callback": True}
        else:
            update_info = self._apply_update_directions(
                gradient.directions,
                projected_grad=gradient.scale,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                step=step,
            )
            if self.svrg is not None:
                self._svrg_after_apply(learning_rate=learning_rate, step=step)
                update_info = {**update_info, **self.svrg.last}
        apply_update_s = time.perf_counter() - apply_t0
        if refresh_fold_s:
            update_info = dict(update_info)
            update_info["refresh_fold_s"] = refresh_fold_s
            update_info["fold_s"] = (
                float(update_info.get("fold_s", 0.0)) + refresh_fold_s
            )
        update_info = update_info if isinstance(update_info, dict) else {}
        callbacks_post_t0 = time.perf_counter()
        self._call_event(
            "on_update_end",
            step=step,
            batch=batch,
            sample=sample,
            score=estimate,
            learning_rate=learning_rate,
            update_info=update_info,
        )
        callbacks_post_update_s = time.perf_counter() - callbacks_post_t0
        profile_s = dict(estimate.profile_s)
        profile_s.update(sampler.profile_s)
        profile_s.update(
            {
                "zo_stepper_refresh_check": refresh_check_s,
                "zo_stepper_sampler_setup": sampler_setup_s,
                "zo_stepper_estimate": estimate_s,
                "zo_stepper_optimizer_wait": optimizer_wait_s,
                "zo_stepper_callbacks_pre_update": callbacks_pre_update_s,
                "zo_stepper_apply_update": apply_update_s,
                "zo_stepper_callbacks_post_update": callbacks_post_update_s,
                "zo_stepper_total": time.perf_counter() - profile_t0,
            }
        )
        return build_zo_step_result(
            step=step,
            learning_rate=learning_rate,
            estimate=estimate,
            direction_refreshed=sampler.refreshed,
            direction_info=sampler.direction_info,
            profile_s=profile_s,
            update_info=update_info,
        )

    def _estimate(
        self,
        batch: ProbeBatch,
        direction_sampler: "_StepDirectionSampler",
        *,
        step: int,
        engine: Any | None = None,
        scorer: TokenGroupScorer | None = None,
    ) -> ZOEstimate:
        return self.estimator.estimate(
            engine=self.engine if engine is None else engine,
            batch=batch,
            direction_sampler=direction_sampler,
            config=ZOEstimatorConfig(
                eps=self.config.eps,
                max_logits_tokens=self.config.max_logits_tokens,
                loss_impl=self.config.loss_impl,
                score_chunk_size=self.config.score_chunk_size,
            ),
            step=step,
            scorer=scorer,
        )

    def _apply_update_directions(
        self,
        update_directions: Any,
        *,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        step: int,
    ) -> dict[str, Any]:
        is_sequence = isinstance(update_directions, list) or bool(
            getattr(update_directions, "is_multi_update_sequence", False)
        )
        if not is_sequence:
            return self.update_state.apply(
                update_directions,
                projected_grad=projected_grad,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                step=step,
            )

        update_infos = []
        fold_s_total = 0.0
        fold = getattr(self.update_state, "fold", None)
        num_updates = len(update_directions)
        for directions_i in update_directions:
            info_i = self.update_state.apply(
                directions_i,
                projected_grad=projected_grad,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                step=step,
            )
            if callable(fold):
                fold_s = float(fold(step=step))
                fold_s_total += fold_s
                info_i = dict(info_i)
                info_i["fold_s"] = fold_s
            update_infos.append(info_i)
            del directions_i
        return {
            "mode": "multi_update",
            "num_updates": num_updates,
            "fold_s": fold_s_total,
            "update_infos": update_infos,
        }

    def add_callback(self, callback: ZOStepCallback) -> None:
        self.callbacks.append(callback)

    def _call_event(self, event: str, **kwargs: Any) -> ZOStepControl:
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is None:
                continue
            new_control = method(control=self.control, **kwargs)
            if new_control is not None:
                self.control = new_control
        return self.control


class _CallableProbeScorer:
    """Expose one HF loss callback through the minimal scorer protocol."""

    def __init__(self, score_fn: Callable[..., Any]) -> None:
        self._score_fn = score_fn

    def score_token_groups(self, token_id_groups, **kwargs):
        return self._score_fn(token_id_groups=token_id_groups, **kwargs)
