"""Probe estimators for ZO optimizer steps."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
from zo_vllm.core.perturbation_normalization import (
    normalize_perturbation_energy_,
    normalize_perturbation_normalization,
    reference_v_column_norm_sq,
)
from zo_vllm.core.probe_results import ProbeLossResult
from zo_vllm.core.token_scores import slice_score_result
from zo_vllm.engine import ZOVLLMEngine

from .direction import (
    DirectionSpec,
    ProbeBatch,
    RolloutProbeBatch,
    TokenProbeBatch,
    clone_direction_map,
)


@dataclass(frozen=True)
class ZOEstimatorConfig:
    """Scoring settings shared by probe estimators."""

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
class ZOGradientEstimate:
    """Low-rank directions and scalar coefficient estimated from probe losses."""

    directions: Any
    scale: float


@dataclass(frozen=True)
class AntitheticProbePlan:
    """Executor-independent plan for one two-sided direction probe."""

    eps: float
    probe_names: tuple[str, str] = ("plus", "minus")

    def __post_init__(self) -> None:
        if float(self.eps) <= 0.0 or not math.isfinite(float(self.eps)):
            raise ValueError("eps must be positive and finite")


@dataclass(frozen=True)
class ZOEstimate:
    """Aggregated estimate produced after scoring one or more probes."""

    reported_loss: float
    loss_plus: float
    loss_minus: float
    projected_grad: float | None
    gradient: ZOGradientEstimate
    profile_s: dict[str, float] = field(default_factory=dict)
    probe_metrics: dict[str, Any] = field(default_factory=dict)
    estimator_metrics: dict[str, Any] = field(default_factory=dict)
    direction_info: dict[str, Any] = field(default_factory=dict)


def _split_probe_observation(
    values: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    profile = values.get("profile_s", {})
    profile_s = {
        str(key): float(value)
        for key, value in profile.items()
        if isinstance(profile, Mapping) and isinstance(value, (int, float))
    }
    metrics = {
        str(key): value
        for key, value in values.items()
        if key not in {"profile_s", "aggregate_directions"}
    }
    return profile_s, metrics


@dataclass(frozen=True)
class DirectionBundle:
    """Raw sampled direction plus the score-time view of that direction."""

    directions: dict[str, dict[str, Any]]
    score_directions: dict[str, dict[str, Any]]


class ZODirectionSampler(Protocol):
    """Sampler used by estimators to request exactly the probes they need."""

    def sample(
        self, *, seed: int | None = None, probe: int | None = None
    ) -> DirectionBundle:
        """Sample one direction and prepare its score-time representation."""
        ...

    def sample_raw(self, *, seed: int) -> dict[str, dict[str, Any]]:
        """Sample one raw direction without callbacks or score preparation."""
        ...


@dataclass(frozen=True)
class DirectionUpdateSequence:
    """Lazy low-rank update sequence for independent multi-query directions."""

    seeds: list[int]
    coeffs: list[float]
    sampler: Any

    is_multi_update_sequence: bool = True

    def __len__(self) -> int:
        return len(self.seeds)

    def __iter__(self):
        for seed, coeff in zip(self.seeds, self.coeffs):
            directions_i = self.sampler(seed=int(seed))
            for direction in directions_i.values():
                direction["U"].mul_(float(coeff))
                direction["v_refreshed"] = True
            yield directions_i


def _sample_direction_from_specs(
    specs: Sequence[DirectionSpec],
    *,
    seed: int,
    perturbation_normalization: str,
) -> dict[str, dict[str, torch.Tensor]]:
    rng_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        sampled: dict[str, dict[str, torch.Tensor]] = {}
        for spec in specs:
            u = torch.randn(
                (int(spec.out_features), int(spec.rank)),
                device=spec.device,
                dtype=spec.dtype,
            )
            v = torch.randn(
                (int(spec.in_features), int(spec.rank)),
                device=spec.device,
                dtype=spec.dtype,
            )
            sampled[spec.name] = {
                "U": u,
                "V": v,
                "scale": float(spec.scale),
                "v_refreshed": True,
                "perturbation_effective_rank": int(spec.rank),
                "perturbation_v_energy_reference": spec.v_energy_reference,
            }
        normalize_perturbation_energy_(
            sampled,
            mode=perturbation_normalization,
        )
        return sampled
    finally:
        torch.set_rng_state(rng_state)
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)


def _direction_info_from_specs(
    specs: Sequence[DirectionSpec],
    *,
    perturbation_normalization: str,
) -> dict[str, float | str]:
    if not specs:
        return {}
    expected_raw_fro_sq = 0.0
    target_fro_sq = 0.0
    for spec in specs:
        v_norm_sq = reference_v_column_norm_sq(
            in_features=int(spec.in_features),
            v_energy_reference=spec.v_energy_reference,
        )
        expected_raw_fro_sq += float(spec.out_features) * float(spec.rank) * v_norm_sq
        target_fro_sq += float(spec.out_features) * float(spec.in_features)
    mode = normalize_perturbation_normalization(perturbation_normalization)
    norm_scale = (
        1.0
        if mode == "none" or expected_raw_fro_sq <= 0.0 or target_fro_sq <= 0.0
        else float(np.sqrt(target_fro_sq / expected_raw_fro_sq))
    )
    raw_rms = (
        0.0
        if target_fro_sq <= 0.0
        else float(np.sqrt(expected_raw_fro_sq / target_fro_sq))
    )
    direction_scale = float(specs[0].scale) * norm_scale
    return {
        "perturbation_normalization": mode,
        "perturbation_expected_raw_fro_norm": float(np.sqrt(expected_raw_fro_sq)),
        "perturbation_target_fro_norm": float(np.sqrt(target_fro_sq)),
        "perturbation_expected_raw_rms": raw_rms,
        "perturbation_normalization_scale": norm_scale,
        "perturbation_direction_scale": direction_scale,
        "perturbation_effective_rms": raw_rms * direction_scale,
        "subspace_effective_rank": int(specs[0].rank),
    }


class ZOEstimator(Protocol):
    """Algorithm layer between sampled directions and weight updates."""

    def estimate(
        self,
        *,
        engine: ZOVLLMEngine,
        batch: ProbeBatch,
        direction_sampler: ZODirectionSampler,
        config: ZOEstimatorConfig,
        step: int,
        scorer: Any | None = None,
    ) -> ZOEstimate:
        """Score probes and aggregate them into an update estimate."""
        ...


def build_single_direction_antithetic_estimate(
    *,
    loss_plus: float,
    loss_minus: float,
    eps: float,
    directions: Any,
    projected_grad: float | None = None,
    profile_s: Mapping[str, float] | None = None,
    probe_metrics: Mapping[str, Any] | None = None,
    direction_info: Mapping[str, Any] | None = None,
) -> ZOEstimate:
    """Aggregate one antithetic probe pair independently of its executor."""

    eps_f = float(eps)
    if eps_f <= 0.0 or not math.isfinite(eps_f):
        raise ValueError("eps must be positive and finite")
    plus = float(loss_plus)
    minus = float(loss_minus)
    if not math.isfinite(plus) or not math.isfinite(minus):
        raise ValueError("antithetic probe losses must be finite")
    coefficient = (
        (plus - minus) / (2.0 * eps_f)
        if projected_grad is None
        else float(projected_grad)
    )
    if not math.isfinite(coefficient):
        raise ValueError("projected gradient must be finite")
    return ZOEstimate(
        reported_loss=(plus + minus) / 2.0,
        loss_plus=plus,
        loss_minus=minus,
        projected_grad=coefficient,
        gradient=ZOGradientEstimate(
            directions=directions,
            scale=coefficient,
        ),
        profile_s=dict(profile_s or {}),
        probe_metrics=dict(probe_metrics or {}),
        estimator_metrics={
            "estimator": SingleDirectionAntitheticEstimator.name,
            "query_count": SingleDirectionAntitheticEstimator.query_count,
        },
        direction_info=dict(direction_info or {}),
    )


class SingleDirectionAntitheticEstimator:
    """Default LOZO/AGZO estimator using one plus/minus LoRA slot pair."""

    name = "single_direction_antithetic"
    query_count = 2

    def plan(self, config: ZOEstimatorConfig) -> AntitheticProbePlan:
        """Describe the probes without choosing an execution backend."""

        return AntitheticProbePlan(eps=float(config.eps))

    def aggregate(
        self,
        plan: AntitheticProbePlan,
        losses: ProbeLossResult,
        *,
        directions: Any,
        projected_grad: float | None = None,
        profile_s: Mapping[str, float] | None = None,
        probe_metrics: Mapping[str, Any] | None = None,
        direction_info: Mapping[str, Any] | None = None,
    ) -> ZOEstimate:
        """Aggregate backend-produced objective losses into one ZO estimate."""

        if tuple(plan.probe_names) != ("plus", "minus"):
            raise ValueError("single-direction antithetic probes must be plus/minus")
        if len(losses.group_losses) != self.query_count:
            raise ValueError(
                "single-direction antithetic aggregation requires two probe losses"
            )
        loss_plus, loss_minus = losses.group_losses
        timing_profile = losses.timing.resolve_cuda_events().profile_seconds()
        timing_profile.update(dict(profile_s or {}))
        return build_single_direction_antithetic_estimate(
            loss_plus=float(loss_plus),
            loss_minus=float(loss_minus),
            eps=float(plan.eps),
            directions=directions,
            projected_grad=projected_grad,
            profile_s=timing_profile,
            probe_metrics=probe_metrics,
            direction_info=direction_info,
        )

    def estimate(
        self,
        *,
        engine: ZOVLLMEngine,
        batch: TokenProbeBatch,
        direction_sampler: ZODirectionSampler,
        config: ZOEstimatorConfig,
        step: int,
        scorer: Any | None = None,
    ) -> ZOEstimate:
        plan = self.plan(config)
        direction = direction_sampler.sample()
        score_method = (
            engine.score_plus_minus_directions
            if scorer is None
            else lambda *args, **kwargs: engine.score_plus_minus_directions_with_scorer(
                scorer, *args, **kwargs
            )
        )
        score = score_method(
            direction.score_directions,
            batch.token_id_groups,
            eps=config.eps,
            loss_token_lens=batch.loss_token_lens,
            labels=batch.labels,
            objective=batch.objective,
            max_logits_tokens=config.max_logits_tokens,
            loss_impl=config.loss_impl,
            score_chunk_size=config.score_chunk_size,
            step=step,
        )
        profile_s, probe_metrics = _split_probe_observation(score.update_info)
        return self.aggregate(
            plan,
            ProbeLossResult(
                group_losses=(float(score.loss_plus), float(score.loss_minus)),
                requests_per_group=len(batch.token_id_groups),
            ),
            directions=direction.directions,
            projected_grad=float(score.projected_grad),
            profile_s=profile_s,
            probe_metrics=probe_metrics,
        )


class MultiQueryZOEstimator:
    """Generic multi-query ZO estimator over token-scoring objectives."""

    name = "multi_query"

    def __init__(
        self,
        *,
        num_queries: int = 1,
        perturbation_sides: str = "two_sided",
        query_microbatch_size: int = 2,
        direction_mode: str = "shared_basis",
        direction_specs: Sequence[DirectionSpec] | None = None,
        seed: int = 42,
        perturbation_normalization: str = "rms",
    ) -> None:
        if int(num_queries) <= 0:
            raise ValueError("num_queries must be positive")
        if perturbation_sides not in {"two_sided", "one_sided"}:
            raise ValueError("perturbation_sides must be two_sided or one_sided")
        if int(query_microbatch_size) <= 0:
            raise ValueError("query_microbatch_size must be positive")
        direction_mode = str(direction_mode).replace("-", "_")
        if direction_mode not in {"shared_basis", "independent"}:
            raise ValueError("direction_mode must be shared_basis or independent")
        self.num_queries = int(num_queries)
        self.perturbation_sides = perturbation_sides
        self.query_microbatch_size = int(query_microbatch_size)
        self.direction_mode = direction_mode
        self.direction_specs = list(direction_specs or [])
        self.seed = int(seed)
        self.perturbation_normalization = str(perturbation_normalization)

    def estimate(
        self,
        *,
        engine: ZOVLLMEngine,
        batch: TokenProbeBatch,
        direction_sampler: ZODirectionSampler,
        config: ZOEstimatorConfig,
        step: int,
        scorer: Any | None = None,
    ) -> ZOEstimate:
        query_bundles = [
            direction_sampler.sample(seed=self.seed + int(step) * 100000 + i, probe=i)
            for i in range(self.num_queries)
        ]
        if self.perturbation_sides == "one_sided":
            return self._estimate_one_sided(
                engine=engine,
                batch=batch,
                query_bundles=query_bundles,
                config=config,
                scorer=scorer,
            )
        return self._estimate_two_sided(
            engine=engine,
            batch=batch,
            query_bundles=query_bundles,
            config=config,
            step=step,
            scorer=scorer,
        )

    def _estimate_two_sided(
        self,
        *,
        engine: ZOVLLMEngine,
        batch: TokenProbeBatch,
        query_bundles: list[DirectionBundle],
        config: ZOEstimatorConfig,
        step: int,
        scorer: Any | None,
    ) -> ZOEstimate:
        total_t0 = time.perf_counter()
        coeffs: list[float] = []
        losses_plus: list[float] = []
        losses_minus: list[float] = []
        update_infos = []
        for direction in query_bundles:
            score_method = (
                engine.score_plus_minus_directions
                if scorer is None
                else lambda *args, **kwargs: (
                    engine.score_plus_minus_directions_with_scorer(
                        scorer, *args, **kwargs
                    )
                )
            )
            score = score_method(
                direction.score_directions,
                batch.token_id_groups,
                eps=config.eps,
                loss_token_lens=batch.loss_token_lens,
                labels=batch.labels,
                objective=batch.objective,
                max_logits_tokens=config.max_logits_tokens,
                loss_impl=config.loss_impl,
                score_chunk_size=config.score_chunk_size,
                step=step,
            )
            coeffs.append(float(score.projected_grad))
            losses_plus.append(float(score.loss_plus))
            losses_minus.append(float(score.loss_minus))
            update_infos.append(dict(score.update_info))
        aggregate = self._aggregate_directions(
            [item.directions for item in query_bundles],
            np.array(coeffs, dtype=np.float32),
        )
        mean_plus = float(np.mean(losses_plus))
        mean_minus = float(np.mean(losses_minus))
        mean_coeff = float(np.mean(coeffs))
        update_info = {
            "profile_s": {
                "multi_query_total": time.perf_counter() - total_t0,
                "query_microbatch_size": 1,
            },
            "query_projected_grads": coeffs,
            "query_losses_plus": losses_plus,
            "query_losses_minus": losses_minus,
            "multi_query_direction_mode": self.direction_mode,
            "slot_update_infos": update_infos,
            "aggregate_directions": aggregate,
        }
        profile_s, probe_metrics = _split_probe_observation(update_info)
        return ZOEstimate(
            reported_loss=(mean_plus + mean_minus) / 2.0,
            loss_plus=mean_plus,
            loss_minus=mean_minus,
            projected_grad=mean_coeff,
            gradient=ZOGradientEstimate(
                directions=aggregate,
                scale=1.0,
            ),
            profile_s=profile_s,
            probe_metrics=probe_metrics,
            estimator_metrics={
                "estimator": self.name,
                "query_count": len(query_bundles) * 2,
                "num_queries": len(query_bundles),
                "perturbation_sides": "two_sided",
                "direction_mode": self.direction_mode,
                "mean_projected_grad": mean_coeff,
            },
        )

    def _estimate_one_sided(
        self,
        *,
        engine: ZOVLLMEngine,
        batch: TokenProbeBatch,
        query_bundles: list[DirectionBundle],
        config: ZOEstimatorConfig,
        scorer: Any | None,
    ) -> ZOEstimate:
        total_t0 = time.perf_counter()
        token_scorer = engine if scorer is None else scorer
        clean = token_scorer.score_token_groups(
            batch.token_id_groups,
            loss_token_lens=batch.loss_token_lens,
            labels=batch.labels,
            max_logits_tokens=config.max_logits_tokens,
            loss_impl=config.loss_impl,
        )
        clean_loss = float(
            clean.loss if batch.objective is None else batch.objective(clean)
        )
        slot_capacity = min(
            int(self.query_microbatch_size), int(engine.probe_slot_capacity)
        )
        if slot_capacity <= 0:
            raise ValueError("MultiQueryZOEstimator requires a runtime probe slot")
        coeffs: list[float] = []
        perturbed_losses: list[float] = []
        update_infos = []
        for start in range(0, len(query_bundles), slot_capacity):
            chunk = query_bundles[start : start + slot_capacity]
            slots, slot_update_infos = engine.set_probe_directions(
                [direction.score_directions for direction in chunk],
                eps=config.eps,
                sign=1.0,
            )
            update_infos.extend(slot_update_infos)
            chunk_score = engine.score_probe_slots_with_scorer(
                token_scorer,
                slots,
                batch.token_id_groups,
                loss_token_lens=batch.loss_token_lens,
                labels=batch.labels,
                max_logits_tokens=config.max_logits_tokens,
                loss_impl=config.loss_impl,
            )
            for idx in range(len(chunk)):
                start_row = idx * len(batch.token_id_groups)
                end_row = (idx + 1) * len(batch.token_id_groups)
                sliced = slice_score_result(
                    chunk_score,
                    start_row,
                    end_row,
                    loss_impl=config.loss_impl,
                )
                loss_i = float(
                    sliced.loss if batch.objective is None else batch.objective(sliced)
                )
                perturbed_losses.append(loss_i)
                coeffs.append((loss_i - clean_loss) / float(config.eps))
        aggregate = self._aggregate_directions(
            [item.directions for item in query_bundles],
            np.array(coeffs, dtype=np.float32),
        )
        mean_perturbed = float(np.mean(perturbed_losses))
        mean_coeff = float(np.mean(coeffs))
        update_info = {
            "profile_s": {
                "multi_query_total": time.perf_counter() - total_t0,
            },
            "query_microbatch_size": slot_capacity,
            "query_projected_grads": coeffs,
            "query_losses_plus": perturbed_losses,
            "clean_loss": clean_loss,
            "multi_query_direction_mode": self.direction_mode,
            "slot_update_infos": update_infos,
            "aggregate_directions": aggregate,
        }
        profile_s, probe_metrics = _split_probe_observation(update_info)
        return ZOEstimate(
            reported_loss=clean_loss,
            loss_plus=mean_perturbed,
            loss_minus=clean_loss,
            projected_grad=mean_coeff,
            gradient=ZOGradientEstimate(
                directions=aggregate,
                scale=1.0,
            ),
            profile_s=profile_s,
            probe_metrics=probe_metrics,
            estimator_metrics={
                "estimator": self.name,
                "query_count": len(query_bundles) + 1,
                "num_queries": len(query_bundles),
                "query_microbatch_size": slot_capacity,
                "perturbation_sides": "one_sided",
                "direction_mode": self.direction_mode,
                "mean_projected_grad": mean_coeff,
            },
        )

    def _aggregate_directions(
        self,
        direction_list: list[dict[str, dict[str, torch.Tensor]]],
        coeffs: np.ndarray,
    ) -> Any:
        if self.direction_mode == "independent":
            scale = 1.0 / float(max(1, len(direction_list)))
            updates = []
            for directions_i, coeff in zip(direction_list, coeffs):
                coeff_f = float(coeff) * scale
                for direction in directions_i.values():
                    direction["U"].mul_(coeff_f)
                    direction["v_refreshed"] = True
                updates.append(directions_i)
            return updates
        aggregate = clone_direction_map(direction_list[0])
        for direction in aggregate.values():
            direction["U"] = torch.zeros_like(direction["U"])
            direction["v_refreshed"] = True
        scale = 1.0 / float(max(1, len(direction_list)))
        for directions_i, coeff in zip(direction_list, coeffs):
            coeff_f = float(coeff) * scale
            for name, direction in directions_i.items():
                aggregate[name]["U"].add_(direction["U"], alpha=coeff_f)
        return aggregate


class EvolutionStrategyEstimator:
    """Single-sided ES estimator over generation reward rollouts."""

    name = "evolution_strategy"

    def __init__(
        self,
        *,
        population_size: int = 30,
        sigma: float = 1e-3,
        reward_shaping: str = "z_score",
        query_microbatch_size: int = 2,
        direction_mode: str = "shared_basis",
        direction_specs: Sequence[DirectionSpec] | None = None,
        seed: int = 42,
        perturbation_normalization: str = "rms",
    ) -> None:
        if int(population_size) <= 0:
            raise ValueError("population_size must be positive")
        if float(sigma) <= 0.0:
            raise ValueError("sigma must be positive")
        if int(query_microbatch_size) <= 0:
            raise ValueError("query_microbatch_size must be positive")
        direction_mode = str(direction_mode).replace("-", "_")
        if direction_mode not in {"shared_basis", "independent"}:
            raise ValueError("direction_mode must be shared_basis or independent")
        reward_shaping = str(reward_shaping).replace("-", "_")
        if reward_shaping == "z_scores":
            reward_shaping = "z_score"
        if reward_shaping not in {"z_score", "none"}:
            raise ValueError("reward_shaping must be z_score or none")
        self.population_size = int(population_size)
        self.sigma = float(sigma)
        self.reward_shaping = reward_shaping
        self.query_microbatch_size = int(query_microbatch_size)
        self.direction_mode = direction_mode
        self.direction_specs = list(direction_specs or [])
        self.seed = int(seed)
        self.perturbation_normalization = str(perturbation_normalization)

    def estimate(
        self,
        *,
        engine: ZOVLLMEngine,
        batch: RolloutProbeBatch,
        direction_sampler: ZODirectionSampler,
        config: ZOEstimatorConfig,
        step: int,
        scorer: Any | None = None,
    ) -> ZOEstimate:
        del scorer
        del config
        prompts = list(batch.rollout_prompts)
        targets = list(batch.rollout_targets)
        if not prompts:
            raise ValueError("rollout_prompts must not be empty")
        if len(prompts) != len(targets):
            raise ValueError(
                "rollout_prompts and rollout_targets must have equal length"
            )

        total_t0 = time.perf_counter()
        query_seeds = [
            self.seed + int(step) * 100000 + i for i in range(self.population_size)
        ]
        slot_capacity = min(
            int(self.query_microbatch_size), int(engine.probe_slot_capacity)
        )
        if slot_capacity <= 0:
            raise ValueError("EvolutionStrategyEstimator requires a runtime probe slot")
        rewards: list[float] = []
        response_lengths: list[float] = []
        update_infos = []
        profile = {
            "sample_directions": 0.0,
            "set_slot_directions": 0.0,
            "generate": 0.0,
            "reward_postprocess": 0.0,
            "aggregate_directions": 0.0,
        }
        for start in range(0, len(query_seeds), slot_capacity):
            seed_chunk = query_seeds[start : start + slot_capacity]
            sample_t0 = time.perf_counter()
            chunk = [direction_sampler.sample(seed=seed_i) for seed_i in seed_chunk]
            profile["sample_directions"] += time.perf_counter() - sample_t0
            slot_t0 = time.perf_counter()
            slots, slot_update_infos = engine.set_probe_directions(
                [direction.score_directions for direction in chunk],
                eps=self.sigma,
                sign=1.0,
            )
            update_infos.extend(slot_update_infos)
            profile["set_slot_directions"] += time.perf_counter() - slot_t0
            del chunk
            chunk_rewards, chunk_lengths, chunk_profile = self._generate_slot_rewards(
                engine=engine,
                prompts=prompts,
                targets=targets,
                reward_fn=batch.rollout_reward_fn,
                slots=slots,
                max_tokens=int(batch.rollout_max_tokens),
                temperature=float(batch.rollout_temperature),
                top_p=float(batch.rollout_top_p),
                seed=batch.rollout_seed,
            )
            profile["generate"] += float(chunk_profile["generate"])
            profile["reward_postprocess"] += float(chunk_profile["reward_postprocess"])
            rewards.extend(chunk_rewards)
            response_lengths.extend(chunk_lengths)

        reward_array = np.array(rewards, dtype=np.float32)
        coeffs = self._shape_rewards(reward_array)
        aggregate_t0 = time.perf_counter()
        aggregate = self._aggregate_seeded_directions(
            direction_sampler,
            seeds=query_seeds,
            coeffs=coeffs,
        )
        profile["aggregate_directions"] += time.perf_counter() - aggregate_t0
        reward_mean = float(np.mean(reward_array))
        reward_std = float(np.std(reward_array))
        update_info = {
            "profile_s": {
                "evolution_strategy_total": time.perf_counter() - total_t0,
                **profile,
            },
            "query_microbatch_size": slot_capacity,
            "query_rewards": [float(x) for x in rewards],
            "query_coeffs": [float(x) for x in coeffs],
            "query_seeds": [int(x) for x in query_seeds],
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_shaping": self.reward_shaping,
            "multi_query_direction_mode": self.direction_mode,
            "slot_update_infos": update_infos,
            "aggregate_directions": aggregate,
        }
        profile_s, probe_metrics = _split_probe_observation(update_info)
        return ZOEstimate(
            reported_loss=-reward_mean,
            loss_plus=-reward_mean,
            loss_minus=-reward_mean,
            projected_grad=None,
            gradient=ZOGradientEstimate(
                directions=aggregate,
                scale=-1.0,
            ),
            profile_s=profile_s,
            probe_metrics=probe_metrics,
            estimator_metrics={
                "estimator": self.name,
                "query_count": int(self.population_size),
                "num_queries": int(self.population_size),
                "query_microbatch_size": slot_capacity,
                "perturbation_sides": "one_sided",
                "sigma": float(self.sigma),
                "direction_mode": self.direction_mode,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_shaping": self.reward_shaping,
            },
        )

    @staticmethod
    def _generate_slot_rewards(
        *,
        engine: ZOVLLMEngine,
        prompts: list[str],
        targets: list[Any],
        reward_fn,
        slots: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> tuple[list[float], list[float], dict[str, float]]:
        generate_t0 = time.perf_counter()
        output_groups = engine.generate_probe_slots(
            slots,
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        generate_s = time.perf_counter() - generate_t0
        slot_rewards: list[float] = []
        slot_response_lengths: list[float] = []
        post_t0 = time.perf_counter()
        for outputs in output_groups:
            rewards_i: list[float] = []
            lengths_i: list[int] = []
            rows = zip(outputs, targets)
            for output, target in rows:
                choice = output.outputs[0]
                reward_result = reward_fn(choice.text, target)
                reward = (
                    float(reward_result[1])
                    if isinstance(reward_result, tuple)
                    else float(reward_result)
                )
                rewards_i.append(reward)
                lengths_i.append(len(choice.token_ids))
            slot_rewards.append(float(np.mean(rewards_i)))
            slot_response_lengths.append(float(np.mean(lengths_i)))
        return (
            slot_rewards,
            slot_response_lengths,
            {
                "generate": generate_s,
                "reward_postprocess": time.perf_counter() - post_t0,
            },
        )

    def _shape_rewards(self, rewards: np.ndarray) -> np.ndarray:
        if self.reward_shaping == "none":
            return rewards.astype(np.float32)
        return ((rewards - rewards.mean()) / (rewards.std() + 1e-8)).astype(np.float32)

    def _aggregate_directions(
        self,
        direction_list: list[dict[str, dict[str, torch.Tensor]]],
        coeffs: np.ndarray,
    ) -> dict[str, dict[str, torch.Tensor]]:
        aggregate = clone_direction_map(direction_list[0])
        for direction in aggregate.values():
            direction["U"] = torch.zeros_like(direction["U"])
            direction["v_refreshed"] = True
        scale = 1.0 / float(max(1, len(direction_list)))
        for directions_i, coeff in zip(direction_list, coeffs):
            coeff_f = float(coeff) * scale
            for name, direction in directions_i.items():
                aggregate[name]["U"].add_(direction["U"], alpha=coeff_f)
        return aggregate

    def _aggregate_seeded_directions(
        self,
        direction_sampler: ZODirectionSampler,
        *,
        seeds: list[int],
        coeffs: np.ndarray,
    ) -> dict[str, dict[str, torch.Tensor]]:
        scale = 1.0 / float(max(1, len(seeds)))
        return DirectionUpdateSequence(
            seeds=[int(seed) for seed in seeds],
            coeffs=[float(coeff) * scale for coeff in coeffs],
            sampler=direction_sampler.sample_raw,
        )


__all__ = [
    "AntitheticProbePlan",
    "DirectionBundle",
    "EvolutionStrategyEstimator",
    "MultiQueryZOEstimator",
    "SingleDirectionAntitheticEstimator",
    "build_single_direction_antithetic_estimate",
    "ZODirectionSampler",
    "ZOEstimate",
    "ZOGradientEstimate",
    "ZOEstimator",
    "ZOEstimatorConfig",
]
