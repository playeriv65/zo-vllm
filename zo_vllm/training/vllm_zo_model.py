"""Unified vLLM-backed ZO model wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from zo_vllm.config import (
    VLLMZOConfig,
)
from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.engine import ZOVLLMEngine

from .direction import (
    AGZODirectionProvider,
    LOZOFastDirectionProvider,
    LOZODirectionProvider,
    ProbeBatch,
    SUAGZODirectionProvider,
    UAGZODirectionProvider,
    TokenProbeBatch,
)
from .estimator import (
    EvolutionStrategyEstimator,
    MultiQueryZOEstimator,
    SingleDirectionAntitheticEstimator,
    ZOEstimator,
)
from .scheduler import ConstantLR, LRScheduler
from .update_state import ImmediateWeightUpdateState, ZOUpdateState
from .zo_step import ZOStepCallback, ZOStepConfig, ZOStepResult, ZOStepper


class VLLMZOModel:
    """Model-like vLLM-backed ZO step orchestrator."""

    def __init__(
        self,
        *,
        engine: ZOVLLMEngine,
        weight_sync: WeightSync,
        config: VLLMZOConfig,
        param_metadata: Mapping[str, ParamMetadata] | None = None,
        scheduler: LRScheduler | None = None,
        estimator: ZOEstimator | None = None,
        update_state: ZOUpdateState | None = None,
        seed_sampler: Callable[[int], int] | None = None,
        weight_update_precision: str = "param",
        sync_weight_update: bool = True,
        qkv_update_mode: str = "batched",
        step_callbacks: Sequence[ZOStepCallback] | None = None,
    ) -> None:
        self.engine = engine
        self.weight_sync = weight_sync
        self.config = config
        self.param_metadata = dict(
            param_metadata
            if param_metadata is not None
            else weight_sync.get_hf_param_metadata()
        )
        self.scheduler = scheduler or ConstantLR(config.learning_rate)
        self.direction_provider = self._build_direction_provider(seed_sampler)
        self.update_state = update_state or ImmediateWeightUpdateState(
            weight_sync=weight_sync,
            precision=weight_update_precision,
            sync_device=bool(sync_weight_update),
            qkv_update_mode=qkv_update_mode,
        )
        self.stepper = ZOStepper(
            engine=engine,
            direction_provider=self.direction_provider,
            update_state=self.update_state,
            config=ZOStepConfig(
                eps=config.eps,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                max_logits_tokens=config.max_logits_tokens,
                loss_impl=config.loss_impl,
                score_chunk_size=config.score_chunk_size,
            ),
            scheduler=self.scheduler,
            estimator=estimator or self._build_estimator(),
            callbacks=step_callbacks,
        )

    def _build_direction_provider(
        self,
        seed_sampler: Callable[[int], int] | None,
    ):
        config = self.config
        if config.direction_provider == "lozo":
            provider_kwargs = {
                "param_metadata": self.param_metadata,
                "rank": config.rank,
                "nu": config.nu,
                "random_device": config.random_device,
                "direction_sampling": config.direction_sampling,
                "direction_scale": config.direction_scale,
                "perturbation_normalization": config.perturbation_normalization,
                "v_normalization": config.v_normalization,
                "seed": config.seed,
                "perturb_seed_offset": config.perturb_seed_offset,
                "seed_sampler": seed_sampler,
            }
            if config.lozo_provider_mode == "scheduled":
                return LOZODirectionProvider(**provider_kwargs)
            return LOZOFastDirectionProvider(**provider_kwargs)

        provider_kwargs = {
            "engine": self.engine,
            "param_metadata": self.param_metadata,
            "rank": config.rank,
            "nu": config.nu,
            "power_iter_steps": config.power_iter_steps,
            "max_logits_tokens": config.max_logits_tokens,
            "loss_impl": config.loss_impl,
            "activation_force_eager": config.activation_force_eager,
            "low_rank_oversample": config.low_rank_oversample,
            "basis_method": config.basis_method,
            "direction_scale": config.direction_scale,
            "perturbation_normalization": config.perturbation_normalization,
            "basis_seed_offset": config.basis_seed_offset,
            "perturb_seed_offset": config.perturb_seed_offset,
            "seed": config.seed,
        }
        if config.direction_provider == "uagzo":
            return UAGZODirectionProvider(
                u_dim=int(config.u_dim),
                u_pool_seed_offset=config.u_pool_seed_offset,
                **provider_kwargs,
            )
        if config.direction_provider == "suagzo":
            return SUAGZODirectionProvider(
                u_dim=int(config.u_dim),
                u_pool_seed_offset=config.u_pool_seed_offset,
                **provider_kwargs,
            )
        return AGZODirectionProvider(**provider_kwargs)

    def _build_estimator(self) -> ZOEstimator:
        if self.config.estimator == "single_direction_antithetic":
            return SingleDirectionAntitheticEstimator()
        if self.config.estimator == "multi_query":
            return MultiQueryZOEstimator(
                num_queries=self.config.num_queries,
                perturbation_sides=self.config.perturbation_sides,
                query_microbatch_size=self.config.query_microbatch_size,
                direction_mode=self.config.multi_query_direction_mode,
                direction_specs=self.direction_provider.direction_specs(),
                seed=self.config.seed,
                perturbation_normalization=self.config.perturbation_normalization,
            )
        if self.config.estimator == "evolution_strategy":
            return EvolutionStrategyEstimator(
                population_size=self.config.population_size,
                sigma=self.config.sigma,
                reward_shaping=self.config.reward_shaping,
                query_microbatch_size=self.config.query_microbatch_size,
                direction_mode=self.config.multi_query_direction_mode,
                direction_specs=self.direction_provider.direction_specs(),
                seed=self.config.seed,
                perturbation_normalization=self.config.perturbation_normalization,
            )
        raise ValueError(f"unsupported estimator: {self.config.estimator}")

    def step(self, batch: ProbeBatch, *, step: int) -> ZOStepResult:
        """Run one ZO step and return generic metrics."""

        return self.stepper.step(batch, step=step)

    def estimate(self, batch: ProbeBatch, *, step: int):
        """Estimate one runtime-native ZO step without applying it."""

        return self.stepper.estimate(batch, step=step)

    def invalidate_direction_slot_state(self) -> None:
        """Mark runtime direction slots stale after rebuilding or loading workers."""

        self.stepper.invalidate_direction_slot_state()

    def estimate_with_score_fn(
        self,
        batch: TokenProbeBatch,
        *,
        step: int,
        score_fn: Callable[..., object],
    ):
        """Estimate one ZO step with scores computed by an external HF path."""

        return self.stepper.estimate_with_score_fn(
            batch,
            step=step,
            score_fn=score_fn,
        )
