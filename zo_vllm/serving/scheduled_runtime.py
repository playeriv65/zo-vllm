"""Synchronous HF runtime facade over scheduled serving engine operations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping

import torch

from zo_vllm.core.probe_results import ProbeTiming, merge_probe_losses
from zo_vllm.engine import TokenRequestNLLResult, TokenScoreResult
from zo_vllm.training.direction import TokenProbeBatch
from zo_vllm.training.estimator import (
    SingleDirectionAntitheticEstimator,
    ZOEstimatorConfig,
)
from zo_vllm.training.zo_step import ZOPendingStep, build_zo_step_result

from .async_engine_service import AsyncZOEngineService
from .thread_bridge import BlockingAsyncBridge


@dataclass(frozen=True)
class ServingStepObservation:
    """Serving-only timing and scheduler observations for one HF step."""

    step: int
    values: dict[str, Any]


class _WorkerDirectionState:
    """Marker for direction state owned by the worker-resident update bank."""

    def state_dict(self) -> dict[str, Any]:
        raise RuntimeError(
            "serving worker direction state is not driver-resident; "
            "use a worker-bank checkpoint handler"
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        del state
        raise RuntimeError(
            "serving worker direction state must be restored by the worker bank"
        )


class ScheduledNLLHFEngine:
    """Expose scheduled option NLL through the engine method used by HF."""

    def __init__(
        self,
        *,
        service: AsyncZOEngineService,
        bridge: BlockingAsyncBridge,
        current_step: Callable[[], int],
    ) -> None:
        self.service = service
        self.bridge = bridge
        self.current_step = current_step
        self._cached_score: TokenScoreResult | None = None
        self._last_clean_observation: dict[str, Any] | None = None

    @contextmanager
    def use_score(self, score: TokenScoreResult) -> Iterator[None]:
        if self._cached_score is not None:
            raise RuntimeError("scheduled score cache is already active")
        self._cached_score = score
        try:
            yield
        finally:
            self._cached_score = None

    def forward_token_request_nll(
        self,
        token_groups,
        *,
        loss_token_lens,
        lora_ids=None,
        max_logits_tokens=8192,
        loss_impl="logprobs",
    ) -> TokenRequestNLLResult:
        del lora_ids, max_logits_tokens, loss_impl
        score = self._cached_score
        if score is None:
            batch = TokenProbeBatch(
                token_id_groups=token_groups,
                loss_token_lens=loss_token_lens,
            )
            execution = self.bridge.call(
                self.service.score_clean(batch, step=int(self.current_step()))
            )
            score = execution.score
            self._last_clean_observation = {
                "eval_score_s": float(execution.score_s),
                "eval_lora_update_s": float(execution.lora_update_s),
                "eval_num_tokens": int(score.num_tokens),
                "eval_num_requests": len(score.request_nll),
                "clean_slot_info": execution.clean_slot_info,
            }
        if len(score.request_nll) != len(token_groups):
            raise RuntimeError(
                "scheduled request NLL count does not match the HF token batch"
            )
        return TokenRequestNLLResult(
            request_nll=torch.as_tensor(score.request_nll, dtype=torch.float32),
            timing=ProbeTiming(
                backend_profile_s=(("scheduled_score_s", 0.0),),
            ),
        )

    def forward_token_logits(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "the serving scheduler backend currently exposes option NLL only; "
            "causal-LM training requires scheduled compact logits"
        )

    def pop_clean_observation(self) -> dict[str, Any] | None:
        value = self._last_clean_observation
        self._last_clean_observation = None
        return value


class ScheduledServingRuntime:
    """HF-native ZO runtime whose only special case is scheduled engine I/O."""

    def __init__(
        self,
        *,
        service: AsyncZOEngineService,
        bridge: BlockingAsyncBridge,
        eps: float,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
    ) -> None:
        self.service = service
        self.bridge = bridge
        self.config = SimpleNamespace(
            max_logits_tokens=int(max_logits_tokens),
            loss_impl=str(loss_impl),
        )
        self.direction_provider = _WorkerDirectionState()
        self.estimator = SingleDirectionAntitheticEstimator()
        self.estimator_config = ZOEstimatorConfig(
            eps=float(eps),
            max_logits_tokens=int(max_logits_tokens),
            loss_impl=str(loss_impl),
        )
        self._current_step = 0
        self._last_observation: ServingStepObservation | None = None
        self.engine = ScheduledNLLHFEngine(
            service=service,
            bridge=bridge,
            current_step=lambda: self._current_step,
        )

    def clean_lora_id_for_score(self) -> None:
        return None

    def invalidate_direction_slot_state(self) -> None:
        raise RuntimeError(
            "serving checkpoint restore must rebuild and synchronize worker slots"
        )

    def estimate_with_score_fn(
        self,
        batch: TokenProbeBatch,
        *,
        step: int,
        score_fn: Callable[..., object],
    ) -> ZOPendingStep:
        step_i = int(step)
        if step_i <= 0:
            raise ValueError("step must be positive")
        if batch.loss_token_lens is None:
            raise RuntimeError(
                "scheduled serving training currently requires prompt-option "
                "classification features"
            )
        self._current_step = step_i
        plan = self.estimator.plan(self.estimator_config)
        execution = self.bridge.call(
            self.service.score_probe_plan(plan, batch, step=step_i)
        )
        probe_losses = []
        for score in execution.scores:
            with self.engine.use_score(score):
                loss_result = score_fn(
                    token_id_groups=batch.token_id_groups,
                    loss_token_lens=batch.loss_token_lens,
                    labels=batch.labels,
                    lora_ids=None,
                    max_logits_tokens=self.estimator_config.max_logits_tokens,
                    loss_impl=self.estimator_config.loss_impl,
                )
            probe_losses.append(loss_result)
        losses = merge_probe_losses(probe_losses)
        estimate = self.estimator.aggregate(
            plan,
            losses,
            directions=None,
            direction_info=execution.direction_info,
        )
        observation = dict(execution.observations)
        observation["direction_info"] = dict(execution.direction_info)
        self._last_observation = ServingStepObservation(step_i, observation)

        def apply(learning_rate: float, weight_decay: float):
            apply_t0 = time.perf_counter()
            update_info, worker_apply_s = self.bridge.call(
                self.service.apply_update(
                    step=step_i,
                    projected_grad=float(estimate.gradient.scale),
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                )
            )
            apply_s = time.perf_counter() - apply_t0
            if self._last_observation is not None:
                values = dict(self._last_observation.values)
                values["apply_s"] = float(apply_s)
                values["update_info"] = dict(update_info)
                self._last_observation = ServingStepObservation(step_i, values)
            return build_zo_step_result(
                step=step_i,
                learning_rate=float(learning_rate),
                estimate=estimate,
                direction_refreshed=execution.direction_refreshed,
                direction_info=dict(execution.direction_info),
                profile_s={"zo_stepper_apply_update": float(worker_apply_s)},
                update_info=dict(update_info),
            )

        return ZOPendingStep(
            step=step_i,
            reported_loss=float(estimate.reported_loss),
            _apply_fn=apply,
        )

    def pop_step_observation(self, step: int) -> dict[str, Any] | None:
        value = self._last_observation
        if value is None or int(value.step) != int(step):
            return None
        self._last_observation = None
        return dict(value.values)

    def pop_clean_observation(self) -> dict[str, Any] | None:
        return self.engine.pop_clean_observation()


__all__ = [
    "ScheduledNLLHFEngine",
    "ScheduledServingRuntime",
    "ServingStepObservation",
]
