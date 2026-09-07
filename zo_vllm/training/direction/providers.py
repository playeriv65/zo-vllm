from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch

from zo_vllm.config import DEFAULT_ZO_PERTURBATION_NORMALIZATION
from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.core.perturbation_normalization import (
    PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
    normalize_perturbation_energy_,
    normalize_perturbation_normalization,
    reference_v_column_norm_sq,
)

from .components import (
    AGZOVProvider,
    GaussianUProvider,
    GaussianVProvider,
    PoolUProvider,
    QueuedAGZOVProvider,
    SubspaceUProvider,
    UProvider,
    VProvider,
)
from .types import DirectionSample, ProbeBatch, TokenProbeBatch
from .types import DirectionSpec


_DIRECTION_NORMALIZATION_STAT_KEYS = {
    "perturbation_normalization",
    "perturbation_normalization_scale",
    "perturbation_expected_raw_fro_norm",
    "perturbation_target_fro_norm",
    "perturbation_expected_raw_rms",
    "perturbation_target_rms",
}


def direction_specs_from_param_metadata(
    param_metadata: Mapping[str, ParamMetadata],
    *,
    rank: int,
    direction_scale: float,
    v_energy_reference: str,
) -> list[DirectionSpec]:
    return [
        DirectionSpec(
            name=name,
            out_features=int(metadata.shape[0]),
            in_features=int(metadata.shape[1]),
            rank=int(rank),
            device=metadata.device,
            dtype=GaussianVProvider.direction_dtype_for(metadata.dtype),
            scale=float(direction_scale),
            v_energy_reference=v_energy_reference,
        )
        for name, metadata in dict(param_metadata).items()
        if metadata.ndim >= 2
    ]


def _perturbation_normalization_info_from_specs(
    specs: list[DirectionSpec],
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
        expected_raw_fro_sq += float(spec.out_features * spec.rank) * v_norm_sq
        target_fro_sq += float(spec.out_features * spec.in_features)

    mode = normalize_perturbation_normalization(perturbation_normalization)
    norm_scale = (
        1.0
        if mode == "none" or expected_raw_fro_sq <= 0.0 or target_fro_sq <= 0.0
        else math.sqrt(target_fro_sq / expected_raw_fro_sq)
    )
    raw_fro_norm = math.sqrt(max(expected_raw_fro_sq, 0.0))
    target_fro_norm = math.sqrt(max(target_fro_sq, 0.0))
    raw_rms = raw_fro_norm / target_fro_norm if target_fro_norm > 0.0 else 0.0
    direction_scale = float(specs[0].scale) * float(norm_scale)
    return {
        "perturbation_normalization": mode,
        "perturbation_normalization_scale": float(norm_scale),
        "perturbation_expected_raw_fro_norm": float(raw_fro_norm),
        "perturbation_target_fro_norm": float(target_fro_norm),
        "perturbation_expected_raw_rms": float(raw_rms),
        "perturbation_target_rms": 1.0 if target_fro_norm > 0.0 else 0.0,
        "perturbation_direction_scale": float(direction_scale),
        "perturbation_effective_rms": float(raw_rms * direction_scale),
        "perturbation_effective_fro_norm": float(raw_fro_norm * direction_scale),
        "subspace_effective_rank": int(specs[0].rank),
    }


def _direction_normalization_stats(
    info: Mapping[str, float | str],
) -> dict[str, float | str]:
    return {
        key: value
        for key, value in info.items()
        if key in _DIRECTION_NORMALIZATION_STAT_KEYS
    }


class FactorizedDirectionProvider:
    """Compose one U provider with one V provider."""

    def __init__(
        self,
        *,
        direction_provider_name: str,
        direction_specs: list[DirectionSpec],
        u_provider: UProvider,
        v_provider: VProvider,
        rank: int,
        basis_seed_offset: int = 100000,
        perturb_seed_offset: int = 200000,
        seed: int = 0,
        seed_sampler: Callable[[int], int] | None = None,
        perturbation_normalization: str = DEFAULT_ZO_PERTURBATION_NORMALIZATION,
    ) -> None:
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        self.direction_provider_name = direction_provider_name
        self._direction_specs = list(direction_specs)
        self.u_provider = u_provider
        self.v_provider = v_provider
        self.rank = int(rank)
        self.basis_seed_offset = int(basis_seed_offset)
        self.perturb_seed_offset = int(perturb_seed_offset)
        self.seed = int(seed)
        self.seed_sampler = seed_sampler
        self.perturbation_normalization = perturbation_normalization

    def will_refresh(self, *, step: int) -> bool:
        will_refresh = getattr(self.v_provider, "will_refresh", None)
        if will_refresh is None:
            return True
        return bool(will_refresh(step=int(step)))

    def direction_specs(self) -> list[DirectionSpec]:
        return list(self._direction_specs)

    @torch.no_grad()
    def prime_v(self, batch: ProbeBatch, *, step: int) -> dict[str, Any]:
        """Build V once before an HF-native training batch reaches the estimator."""

        step_i = int(step)
        if step_i <= 0:
            raise ValueError("step must be positive")
        prime = getattr(self.v_provider, "prime", None)
        if not callable(prime):
            raise RuntimeError(
                "direction V provider does not support preinitialization"
            )
        basis_seed = self.basis_seed_offset + step_i + self.seed * 1_000_000
        perturb_seed = self.perturb_seed_offset + step_i + self.seed * 1_000_000
        raw = prime(
            batch,
            targets=self.direction_specs(),
            step=step_i,
            basis_seed=basis_seed,
            perturb_seed=perturb_seed,
        )
        provider_info = {}
        info_fn = getattr(self.v_provider, "info", None)
        if callable(info_fn):
            provider_info = dict(info_fn())
        return {
            "direction_provider": self.direction_provider_name,
            "basis_seed": basis_seed,
            "perturb_seed": perturb_seed,
            **provider_info,
            **{
                str(key): value
                for key, value in dict(raw).items()
                if isinstance(value, (str, int, float, bool))
            },
            "subspace_effective_rank": int(self.rank),
        }

    def state_dict(self) -> dict[str, Any]:
        state_dict = getattr(self.v_provider, "state_dict", None)
        if not callable(state_dict):
            raise RuntimeError("direction V provider does not support checkpointing")
        state = {
            "type": "factorized_direction_provider",
            "direction_provider_name": self.direction_provider_name,
            "v_provider": state_dict(),
        }
        u_state_dict = getattr(self.u_provider, "state_dict", None)
        if callable(u_state_dict):
            state["u_provider"] = u_state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("type") != "factorized_direction_provider":
            raise ValueError("invalid factorized direction provider checkpoint")
        if state.get("direction_provider_name") != self.direction_provider_name:
            raise ValueError("direction provider type does not match checkpoint")
        load_state_dict = getattr(self.v_provider, "load_state_dict", None)
        if not callable(load_state_dict):
            raise RuntimeError("direction V provider cannot load checkpoints")
        if isinstance(self.v_provider, QueuedAGZOVProvider):
            load_state_dict(
                state["v_provider"],
                direction_specs=self.direction_specs(),
            )
        else:
            load_state_dict(state["v_provider"])
        if "u_provider" in state:
            load_u_state_dict = getattr(self.u_provider, "load_state_dict", None)
            if not callable(load_u_state_dict):
                raise RuntimeError("direction U provider cannot load checkpoints")
            load_u_state_dict(state["u_provider"])

    def next(self, batch: ProbeBatch, *, step: int, probe: int = 0) -> DirectionSample:
        """One direction for ``step``; ``probe`` > 0 re-draws U under the same V.

        The V provider is keyed on the step and basis seed, so with a frozen
        V* every probe of a step shares the same V and only the U seed moves.
        This is what a multi-query estimate over the u-space needs; the
        spec-based sampler it used before drew a fresh V per probe as well.
        """

        step_i = int(step)
        if step_i <= 0:
            raise ValueError("step must be positive")
        basis_seed = self.basis_seed_offset + step_i + self.seed * 1_000_000
        if self.seed_sampler is None:
            perturb_seed = self.perturb_seed_offset + step_i + self.seed * 1_000_000
        else:
            perturb_seed = int(self.seed_sampler(step_i))
        if int(probe) > 0:
            perturb_seed += int(probe) * 7_919_000

        rng_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            torch.manual_seed(int(perturb_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(perturb_seed))
            raw: dict[str, Any]
            targets = self.direction_specs()
            v_map, v_raw = self.v_provider.collect(
                batch,
                targets=targets,
                step=step_i,
                basis_seed=basis_seed,
                perturb_seed=perturb_seed,
            )
            targets = self._targets_with_effective_v_rank(targets, v_map)
            u_map, u_raw = self.u_provider.collect(
                batch,
                targets=targets,
                step=step_i,
                basis_seed=basis_seed,
                perturb_seed=perturb_seed,
            )
            raw = {**v_raw, **u_raw}
            refreshed = bool(
                raw.get(
                    "v_refreshed",
                    not bool(raw.get("basis_reused", False)),
                )
            )
            directions = self._compose_direction_maps(
                targets=targets,
                u_map=u_map,
                v_map=v_map,
                v_refreshed=refreshed,
            )
            normalization_info = normalize_perturbation_energy_(
                directions,
                mode=self.perturbation_normalization,
            )
        finally:
            torch.set_rng_state(rng_state)
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
        provider_info = {}
        info_fn = getattr(self.v_provider, "info", None)
        if info_fn is not None:
            provider_info = dict(info_fn())
        info = {
            "direction_provider": self.direction_provider_name,
            "basis_seed": basis_seed,
            "perturb_seed": perturb_seed,
            **provider_info,
            **self.u_provider.info(),
            **normalization_info,
            "subspace_effective_rank": self._effective_rank(directions),
        }
        info.update(
            {
                str(key): value
                for key, value in raw.items()
                if isinstance(value, (str, int, float, bool))
            }
        )
        if "kappa" in raw:
            info["kappa"] = int(raw["kappa"])
        elif getattr(batch, "subspace_num_rows", None) is not None:
            info["kappa"] = int(batch.subspace_num_rows)
        elif getattr(batch, "subspace_token_id_group_batches", None) is not None:
            info["subspace_num_chunks"] = len(batch.subspace_token_id_group_batches)
        if "subspace_num_chunks" in raw:
            info["subspace_num_chunks"] = int(raw["subspace_num_chunks"])
        return DirectionSample(
            directions=directions,
            refreshed=refreshed,
            info=info,
        )

    def _targets_with_effective_v_rank(
        self,
        targets: list[DirectionSpec],
        v_map: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> list[DirectionSpec]:
        adjusted: list[DirectionSpec] = []
        changed = False
        for spec in targets:
            rank = int(v_map[spec.name]["V"].shape[1])
            if rank == int(spec.rank):
                adjusted.append(spec)
                continue
            changed = True
            adjusted.append(
                DirectionSpec(
                    name=spec.name,
                    out_features=spec.out_features,
                    in_features=spec.in_features,
                    rank=rank,
                    device=spec.device,
                    dtype=spec.dtype,
                    scale=spec.scale,
                    v_energy_reference=spec.v_energy_reference,
                )
            )
        return adjusted if changed else targets

    @torch.no_grad()
    def _compose_direction_maps(
        self,
        *,
        targets: list[DirectionSpec],
        u_map: Mapping[str, Mapping[str, torch.Tensor]],
        v_map: Mapping[str, Mapping[str, torch.Tensor]],
        v_refreshed: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        directions: dict[str, dict[str, torch.Tensor]] = {}
        for spec in targets:
            name = spec.name
            cached = v_map[name]
            u_entry = u_map[name]
            u = u_entry["U"].contiguous()
            cached_v = cached["V"]
            if tuple(u.shape) != (int(spec.out_features), int(cached_v.shape[1])):
                raise RuntimeError(f"direction U shape mismatch for {name}")
            direction = {
                "U": u,
                "V": cached_v,
                "v_refreshed": bool(v_refreshed),
            }
            if "V_T" in cached:
                direction["V_T"] = cached["V_T"]
            if "scale" in cached:
                direction["scale"] = cached["scale"]
            for key, value in cached.items():
                if key.startswith("perturbation_"):
                    direction[key] = value
            directions[name] = direction
        return directions

    def _effective_rank(
        self, directions: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> int:
        for direction in directions.values():
            return int(direction["V"].shape[1])
        return 0


class AGZODirectionProvider(FactorizedDirectionProvider):
    """Activation-guided V provider with Gaussian U sampling."""

    def __init__(self, **kwargs: Any) -> None:
        rank = int(kwargs["rank"])
        param_metadata = kwargs.pop("param_metadata")
        nu = int(kwargs.pop("nu", 1))
        subspace_queue_size = int(kwargs.pop("subspace_queue_size", 1))
        basis_seed_offset = int(kwargs.pop("basis_seed_offset", 100000))
        perturb_seed_offset = int(kwargs.pop("perturb_seed_offset", 200000))
        perturbation_normalization = kwargs.pop(
            "perturbation_normalization",
            DEFAULT_ZO_PERTURBATION_NORMALIZATION,
        )
        seed = int(kwargs.pop("seed", 0))
        direction_scale = float(kwargs.get("direction_scale", 1.0))
        super().__init__(
            direction_provider_name="agzo",
            direction_specs=direction_specs_from_param_metadata(
                param_metadata,
                rank=rank,
                direction_scale=direction_scale,
                v_energy_reference=PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
            ),
            u_provider=GaussianUProvider(random_device="cuda"),
            v_provider=QueuedAGZOVProvider(
                AGZOVProvider(**kwargs),
                nu=nu,
                queue_size=subspace_queue_size,
            ),
            rank=rank,
            basis_seed_offset=basis_seed_offset,
            perturb_seed_offset=perturb_seed_offset,
            seed=seed,
            perturbation_normalization=perturbation_normalization,
        )


class LOZODirectionProvider(FactorizedDirectionProvider):
    """Unified Gaussian-V LOZO provider built from explicit U/V providers."""

    def __init__(
        self,
        *,
        param_metadata: Mapping[str, ParamMetadata],
        rank: int,
        nu: int,
        random_device: str = "cuda",
        direction_sampling: str = "exact",
        direction_scale: float = 1.0,
        perturbation_normalization: str = DEFAULT_ZO_PERTURBATION_NORMALIZATION,
        v_normalization: str = "none",
        seed: int = 0,
        perturb_seed_offset: int = 200000,
        seed_sampler: Callable[[int], int] | None = None,
    ) -> None:
        v_provider = GaussianVProvider(
            param_metadata=param_metadata,
            rank=rank,
            nu=nu,
            random_device=random_device,
            direction_sampling=direction_sampling,
            direction_scale=direction_scale,
            v_normalization=v_normalization,
        )
        super().__init__(
            direction_provider_name="lozo",
            direction_specs=v_provider.direction_specs(),
            u_provider=GaussianUProvider(
                random_device=random_device,
                direction_sampling=direction_sampling,
            ),
            v_provider=v_provider,
            rank=rank,
            basis_seed_offset=perturb_seed_offset,
            perturb_seed_offset=perturb_seed_offset,
            seed=seed,
            seed_sampler=seed_sampler,
            perturbation_normalization=perturbation_normalization,
        )


class LOZOFastDirectionProvider:
    """Fast plain-LOZO provider that directly samples Gaussian U/V."""

    def __init__(
        self,
        *,
        param_metadata: Mapping[str, ParamMetadata],
        rank: int,
        nu: int,
        random_device: str = "cuda",
        direction_sampling: str = "exact",
        direction_scale: float = 1.0,
        perturbation_normalization: str = DEFAULT_ZO_PERTURBATION_NORMALIZATION,
        v_normalization: str = "none",
        seed: int = 0,
        perturb_seed_offset: int = 200000,
        seed_sampler: Callable[[int], int] | None = None,
    ) -> None:
        self.v_provider = GaussianVProvider(
            param_metadata=param_metadata,
            rank=rank,
            nu=nu,
            random_device=random_device,
            direction_sampling=direction_sampling,
            direction_scale=direction_scale,
            v_normalization=v_normalization,
        )
        self.u_provider = GaussianUProvider(
            random_device=random_device,
            direction_sampling=direction_sampling,
        )
        self._direction_specs = self.v_provider.direction_specs()
        self.param_metadata = self.v_provider.param_metadata
        self.rank = int(rank)
        self.nu = self.v_provider.nu
        self.direction_sampling = direction_sampling
        self.direction_scale = float(direction_scale)
        self.perturbation_normalization = perturbation_normalization
        self.v_normalization = v_normalization
        self.seed = int(seed)
        self.perturb_seed_offset = int(perturb_seed_offset)
        self.seed_sampler = seed_sampler
        self.step = 0
        self._active_perturb_seed: int | None = None
        self._active_directions: dict[str, dict[str, torch.Tensor]] | None = None
        self._normalization_info = _perturbation_normalization_info_from_specs(
            self._direction_specs,
            perturbation_normalization=self.perturbation_normalization,
        )
        self._normalization_direction_stats = _direction_normalization_stats(
            self._normalization_info
        )
        self._effective_direction_scale = float(
            self._normalization_info.get(
                "perturbation_direction_scale",
                self.direction_scale,
            )
        )

    def will_refresh(self, *, step: int) -> bool:
        if self.seed_sampler is not None:
            return self._active_perturb_seed != self._seed_for_step(step)
        return self.v_provider.will_refresh(step=int(step))

    def direction_specs(self) -> list[DirectionSpec]:
        return list(self._direction_specs)

    def state_dict(self) -> dict[str, Any]:
        state = {
            "type": "lozo_fast_direction_provider",
            "step": int(self.step),
            "active_perturb_seed": self._active_perturb_seed,
            "v_provider": self.v_provider.state_dict(),
        }
        sampler_state_dict = getattr(self.seed_sampler, "state_dict", None)
        if callable(sampler_state_dict):
            state["seed_sampler"] = sampler_state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("type") != "lozo_fast_direction_provider":
            raise ValueError("invalid fast LOZO direction provider checkpoint")
        self.step = int(state["step"])
        raw_active_seed = state.get("active_perturb_seed")
        self._active_perturb_seed = (
            None if raw_active_seed is None else int(raw_active_seed)
        )
        self._active_directions = None
        self.v_provider.load_state_dict(state["v_provider"])
        if "seed_sampler" in state:
            load_sampler_state = getattr(self.seed_sampler, "load_state_dict", None)
            if not callable(load_sampler_state):
                raise RuntimeError(
                    "LOZO checkpoint contains seed-sampler state but no compatible "
                    "sampler is installed"
                )
            load_sampler_state(state["seed_sampler"])

    def next(self, batch: ProbeBatch, *, step: int) -> DirectionSample:
        del batch
        step_i = int(step)
        if step_i <= 0:
            raise ValueError("step must be positive")
        perturb_seed = self._seed_for_step(step_i)
        if self.seed_sampler is None:
            directions = self.sample_direction(perturb_seed, step=step_i)
            refreshed = any(
                bool(item.get("v_refreshed", False)) for item in directions.values()
            )
        else:
            refreshed = self._active_perturb_seed != perturb_seed
            if refreshed or self._active_directions is None:
                directions = self._sample_complete_direction(
                    perturb_seed,
                    step=step_i,
                )
                self._active_perturb_seed = perturb_seed
                self._active_directions = directions
            else:
                directions = self._active_directions
            for direction in directions.values():
                direction["v_refreshed"] = bool(refreshed)
        normalization_info = dict(self._normalization_info)
        return DirectionSample(
            directions=directions,
            refreshed=refreshed,
            info={
                "direction_provider": "lozo",
                "provider_mode": "fast",
                "perturb_seed": perturb_seed,
                **self.v_provider.info(),
                **self.u_provider.info(),
                **normalization_info,
                "subspace_effective_rank": int(self.rank),
            },
        )

    def _seed_for_step(self, step: int) -> int:
        step_i = int(step)
        if step_i <= 0:
            raise ValueError("step must be positive")
        if self.seed_sampler is None:
            return self.perturb_seed_offset + step_i + self.seed * 1_000_000
        return int(self.seed_sampler(step_i))

    @torch.no_grad()
    def _sample_complete_direction(
        self,
        perturb_seed: int,
        *,
        step: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        # A lifecycle seed identifies the entire low-rank direction. Clear the
        # Gaussian V cache so replaying that seed consumes the same RNG stream
        # for both V and U, independent of the provider's ordinary nu cadence.
        self.v_provider.v_cache.clear()
        self.v_provider.vt_cache.clear()
        return self.sample_direction(int(perturb_seed), step=int(step))

    @torch.no_grad()
    def sample_direction(
        self,
        random_seed: int,
        *,
        step: int | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        rng_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            torch.manual_seed(int(random_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(random_seed))
            directions = self._sample_direction_with_current_rng(
                step=int(step) if step is not None else self.step + 1
            )
        finally:
            torch.set_rng_state(rng_state)
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
        self.step += 1
        return directions

    def _sample_direction_with_current_rng(
        self,
        *,
        step: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        batch = TokenProbeBatch(token_id_groups=())
        step_i = int(step)
        targets = self.direction_specs()
        v_map, raw = self.v_provider.collect(
            batch,
            targets=targets,
            step=step_i,
            basis_seed=0,
            perturb_seed=0,
        )
        u_map, _ = self.u_provider.collect(
            batch,
            targets=targets,
            step=step_i,
            basis_seed=0,
            perturb_seed=0,
        )
        v_refreshed = bool(raw.get("v_refreshed", True))
        directions: dict[str, dict[str, torch.Tensor]] = {}
        for spec in targets:
            name = spec.name
            cached = v_map[name]
            directions[name] = {
                "U": u_map[name]["U"].contiguous(),
                "V": cached["V"],
                "V_T": cached["V_T"],
                "v_refreshed": v_refreshed,
                "scale": self._effective_direction_scale,
            }
            for key, value in cached.items():
                if key.startswith("perturbation_"):
                    directions[name][key] = value
            directions[name].update(self._normalization_direction_stats)
        return directions


class UAGZODirectionProvider(FactorizedDirectionProvider):
    """AGZO V-subspace provider with U sampled from fixed orthogonal pools."""

    def __init__(
        self,
        *,
        u_dim: int,
        u_pool_seed_offset: int = 300000,
        **kwargs: Any,
    ) -> None:
        if int(u_dim) <= 0:
            raise ValueError("u_dim must be positive")
        rank = int(kwargs["rank"])
        param_metadata = kwargs.pop("param_metadata")
        nu = int(kwargs.pop("nu", 1))
        subspace_queue_size = int(kwargs.pop("subspace_queue_size", 1))
        basis_seed_offset = int(kwargs.pop("basis_seed_offset", 100000))
        perturb_seed_offset = int(kwargs.pop("perturb_seed_offset", 200000))
        perturbation_normalization = kwargs.pop(
            "perturbation_normalization",
            DEFAULT_ZO_PERTURBATION_NORMALIZATION,
        )
        seed = int(kwargs.get("seed", 0))
        seed = int(kwargs.pop("seed", seed))
        direction_scale = float(kwargs.get("direction_scale", 1.0))
        if int(u_dim) < rank:
            raise ValueError("u_dim must be greater than or equal to rank")
        self.u_provider = PoolUProvider(
            rank=rank,
            u_dim=int(u_dim),
            seed=seed,
            u_pool_seed_offset=int(u_pool_seed_offset),
        )
        super().__init__(
            direction_provider_name="uagzo",
            direction_specs=direction_specs_from_param_metadata(
                param_metadata,
                rank=rank,
                direction_scale=direction_scale,
                v_energy_reference=PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
            ),
            u_provider=self.u_provider,
            v_provider=QueuedAGZOVProvider(
                AGZOVProvider(**kwargs),
                nu=nu,
                queue_size=subspace_queue_size,
            ),
            rank=rank,
            basis_seed_offset=basis_seed_offset,
            perturb_seed_offset=perturb_seed_offset,
            seed=seed,
            perturbation_normalization=perturbation_normalization,
        )

    def _u_pool_for(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.u_provider._u_pool_for(*args, **kwargs)

    def set_u_index_selector(self, selector: Any | None) -> None:
        """Install an external per-target U-pool selector for future steps."""

        self.u_provider.set_index_selector(selector)


class SUAGZODirectionProvider(FactorizedDirectionProvider):
    """AGZO V-subspace provider with continuous U subspace sampling."""

    def __init__(
        self,
        *,
        u_dim: int,
        u_pool_seed_offset: int = 300000,
        **kwargs: Any,
    ) -> None:
        if int(u_dim) <= 0:
            raise ValueError("u_dim must be positive")
        rank = int(kwargs["rank"])
        param_metadata = kwargs.pop("param_metadata")
        nu = int(kwargs.pop("nu", 1))
        subspace_queue_size = int(kwargs.pop("subspace_queue_size", 1))
        basis_seed_offset = int(kwargs.pop("basis_seed_offset", 100000))
        perturb_seed_offset = int(kwargs.pop("perturb_seed_offset", 200000))
        perturbation_normalization = kwargs.pop(
            "perturbation_normalization",
            DEFAULT_ZO_PERTURBATION_NORMALIZATION,
        )
        seed = int(kwargs.get("seed", 0))
        seed = int(kwargs.pop("seed", seed))
        direction_scale = float(kwargs.get("direction_scale", 1.0))
        self.u_provider = SubspaceUProvider(
            u_dim=int(u_dim),
            seed=seed,
            u_pool_seed_offset=int(u_pool_seed_offset),
        )
        super().__init__(
            direction_provider_name="suagzo",
            direction_specs=direction_specs_from_param_metadata(
                param_metadata,
                rank=rank,
                direction_scale=direction_scale,
                v_energy_reference=PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
            ),
            u_provider=self.u_provider,
            v_provider=QueuedAGZOVProvider(
                AGZOVProvider(**kwargs),
                nu=nu,
                queue_size=subspace_queue_size,
            ),
            rank=rank,
            basis_seed_offset=basis_seed_offset,
            perturb_seed_offset=perturb_seed_offset,
            seed=seed,
            perturbation_normalization=perturbation_normalization,
        )

    def _u_pool_for(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.u_provider._u_pool_for(*args, **kwargs)


__all__ = [
    "AGZODirectionProvider",
    "FactorizedDirectionProvider",
    "LOZODirectionProvider",
    "LOZOFastDirectionProvider",
    "SUAGZODirectionProvider",
    "UAGZODirectionProvider",
    "direction_specs_from_param_metadata",
]
