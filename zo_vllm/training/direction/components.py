from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any, Protocol

import torch

from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.core.lora_scope import should_refresh_for_nu, validate_nu
from zo_vllm.core.perturbation_normalization import (
    PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
    attach_factorized_perturbation_spec_,
    v_energy_reference_from_v_normalization,
)
from zo_vllm.engine import ZOVLLMEngine

from .subspace_queue import SubspaceQueue
from .types import (
    DirectionSpec,
    ProbeBatch,
    SubspaceTokenProbeBatch,
    TokenProbeBatch,
)


class UProvider(Protocol):
    """Provider that collects U matrices for a full target set."""

    def collect(
        self,
        batch: ProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Return U factors for the requested targets."""
        ...

    def info(self) -> dict[str, Any]:
        """Return metadata specific to this U provider."""
        ...


class UPoolIndexSelector(Protocol):
    """Select local orthogonal-pool columns for every target in a ZO step."""

    def select_indices(
        self,
        *,
        step: int,
        u_dim: int,
        targets: Sequence[DirectionSpec],
    ) -> Mapping[str, Sequence[int]]:
        """Return one local index sequence for each requested target."""
        ...

    def info(self) -> Mapping[str, Any]:
        """Return selector provenance for direction logging."""
        ...


class VProvider(Protocol):
    """Provider that generates V matrices on request."""

    def collect(
        self,
        batch: TokenProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Return a V subspace for the requested step."""
        ...


class GaussianUProvider:
    """Collect Gaussian U matrices from target specs."""

    def __init__(
        self,
        *,
        random_device: str = "cuda",
        direction_sampling: str = "exact",
    ) -> None:
        if random_device not in {"cpu", "cuda"}:
            raise ValueError(f"unknown random_device: {random_device}")
        if direction_sampling not in {"exact", "flat"}:
            raise ValueError(f"unknown direction_sampling: {direction_sampling}")
        self.random_device = random_device
        self.direction_sampling = direction_sampling

    def collect(
        self,
        batch: TokenProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        del batch, step, basis_seed, perturb_seed
        target_list = list(targets)
        if self.direction_sampling == "flat" and target_list:
            return self._collect_flat(target_list), {
                "u_provider": "gaussian",
                "u_direction_sampling": "flat",
            }
        return {
            spec.name: {
                "U": self._randn(
                    (int(spec.out_features), int(spec.rank)),
                    spec.device,
                    spec.dtype,
                )
            }
            for spec in target_list
        }, {
            "u_provider": "gaussian",
            "u_direction_sampling": "exact",
        }

    def _collect_flat(
        self,
        targets: Sequence[DirectionSpec],
    ) -> dict[str, dict[str, Any]]:
        sample_device = targets[0].device
        sample_dtype = targets[0].dtype
        if any(
            spec.device != sample_device or spec.dtype != sample_dtype
            for spec in targets
        ):
            raise RuntimeError(
                "flat U sampling requires uniform target device and dtype"
            )
        total_u = sum(int(spec.out_features) * int(spec.rank) for spec in targets)
        flat_u = self._randn_flat(total_u, sample_device, sample_dtype)
        offset = 0
        u_map: dict[str, dict[str, Any]] = {}
        for spec in targets:
            numel = int(spec.out_features) * int(spec.rank)
            u_map[spec.name] = {
                "U": flat_u[offset : offset + numel].view(
                    int(spec.out_features),
                    int(spec.rank),
                )
            }
            offset += numel
        return u_map

    def _sample_device_for(self, target_device: torch.device) -> torch.device:
        if self.random_device == "cpu":
            return torch.device("cpu")
        if self.random_device == "cuda":
            return target_device
        raise ValueError(f"unknown random_device: {self.random_device}")

    def _randn(
        self,
        shape: tuple[int, ...],
        target_device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(
            shape,
            device=self._sample_device_for(target_device),
            dtype=dtype,
        ).to(target_device)

    def _randn_flat(
        self,
        numel: int,
        target_device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._randn((int(numel),), target_device, dtype)

    def info(self) -> dict[str, Any]:
        return {
            "u_provider": "gaussian",
            "u_direction_sampling": self.direction_sampling,
        }


class _OrthogonalUPoolBase:
    """Shared fixed orthogonal U basis storage."""

    def __init__(
        self,
        *,
        u_dim: int,
        seed: int = 0,
        u_pool_seed_offset: int = 300000,
    ) -> None:
        if int(u_dim) <= 0:
            raise ValueError("u_dim must be positive")
        self.u_dim = int(u_dim)
        self.seed = int(seed)
        self.u_pool_seed_offset = int(u_pool_seed_offset)
        self._u_pool_generation = 0
        self._u_pools: dict[
            tuple[str, int, torch.device, torch.dtype],
            torch.Tensor,
        ] = {}

    def info(self) -> dict[str, Any]:
        return {
            "u_dim": self.u_dim,
            "u_pool_seed_offset": self.u_pool_seed_offset,
            "u_pool_generation": self._u_pool_generation,
        }

    def _set_u_pool_generation(self, generation: int) -> None:
        generation_i = int(generation)
        if generation_i < 0:
            raise ValueError("U pool generation must be non-negative")
        if generation_i == self._u_pool_generation:
            return
        self._u_pool_generation = generation_i
        self._u_pools.clear()

    @torch.no_grad()
    def _u_pool_for(
        self,
        name: str,
        *,
        out_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.u_dim > int(out_features):
            raise ValueError(
                f"u_dim={self.u_dim} exceeds output dimension {out_features} for {name}"
            )
        key = (name, int(out_features), device, dtype)
        pool = self._u_pools.get(key)
        if pool is not None:
            return pool

        generator = torch.Generator(device=device)
        generator.manual_seed(self._pool_seed_for(name))
        raw = torch.randn(
            (int(out_features), self.u_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        q, _ = torch.linalg.qr(raw, mode="reduced")
        pool = (q * math.sqrt(out_features)).to(dtype=dtype).contiguous()
        self._u_pools[key] = pool
        return pool

    def _pool_seed_for(self, name: str) -> int:
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
        name_seed = int.from_bytes(digest, byteorder="little", signed=False)
        generation_seed = self._u_pool_generation * 10_000_000_000
        return int(
            self.u_pool_seed_offset
            + self.seed * 1_000_000
            + generation_seed
            + name_seed
        ) % (2**63 - 1)


class PoolUProvider(_OrthogonalUPoolBase):
    """Sample U columns from fixed per-parameter orthogonal pools."""

    def __init__(
        self,
        *,
        rank: int,
        u_dim: int,
        seed: int = 0,
        u_pool_seed_offset: int = 300000,
        index_selector: UPoolIndexSelector | None = None,
    ) -> None:
        if int(u_dim) < int(rank):
            raise ValueError("u_dim must be greater than or equal to rank")
        super().__init__(
            u_dim=int(u_dim),
            seed=int(seed),
            u_pool_seed_offset=int(u_pool_seed_offset),
        )
        self.rank = int(rank)
        self.index_selector = index_selector

    def set_index_selector(self, selector: UPoolIndexSelector | None) -> None:
        """Install a per-target selector without changing the stored U pools."""

        self.index_selector = selector

    def info(self) -> dict[str, Any]:
        values = {
            **super().info(),
            "u_provider": "pool",
            "u_pool_index_mode": (
                "per_target_random"
                if self.index_selector is None
                else "per_target_selector"
            ),
        }
        selector_info = getattr(self.index_selector, "info", None)
        if callable(selector_info):
            values.update(dict(selector_info()))
        return values

    def state_dict(self) -> dict[str, Any]:
        selector_state = None
        state_dict = getattr(self.index_selector, "state_dict", None)
        if callable(state_dict):
            selector_state = state_dict()
        return {
            "type": "pool_u_provider",
            "rank": self.rank,
            "u_dim": self.u_dim,
            "seed": self.seed,
            "u_pool_seed_offset": self.u_pool_seed_offset,
            "u_pool_generation": self._u_pool_generation,
            "index_selector": selector_state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("type") != "pool_u_provider":
            raise ValueError("invalid pool U provider checkpoint")
        expected = {
            "rank": self.rank,
            "u_dim": self.u_dim,
            "seed": self.seed,
            "u_pool_seed_offset": self.u_pool_seed_offset,
        }
        for key, value in expected.items():
            if int(state.get(key, -1)) != int(value):
                raise ValueError(f"pool U provider checkpoint mismatch: {key}")
        self._set_u_pool_generation(int(state.get("u_pool_generation", 0)))
        selector_state = state.get("index_selector")
        if selector_state is None:
            return
        load_state_dict = getattr(self.index_selector, "load_state_dict", None)
        if not callable(load_state_dict):
            raise RuntimeError(
                "pool U checkpoint contains selector state but no compatible "
                "selector is installed"
            )
        load_state_dict(selector_state)

    @torch.no_grad()
    def collect(
        self,
        batch: TokenProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        del batch, basis_seed, perturb_seed
        target_list = list(targets)
        generation_for_step = getattr(
            self.index_selector, "pool_generation_for_step", None
        )
        if callable(generation_for_step):
            self._set_u_pool_generation(generation_for_step(step=int(step)))
        selected_indices = self._selected_indices_for_step(target_list, step=int(step))
        scale_for_selection = getattr(
            self.index_selector, "u_scale_for_selection", None
        )
        selected_scales: list[float] = []
        u_map: dict[str, dict[str, Any]] = {}
        for spec in target_list:
            indices = None if selected_indices is None else selected_indices[spec.name]
            u_value = self.sample(
                name=spec.name,
                out_features=int(spec.out_features),
                rank=int(spec.rank),
                device=spec.device,
                dtype=spec.dtype,
                indices=indices,
            )
            if indices is not None and callable(scale_for_selection):
                scale = float(
                    scale_for_selection(
                        step=int(step),
                        target_name=spec.name,
                        indices=indices,
                    )
                )
                if not math.isfinite(scale) or scale <= 0.0:
                    raise ValueError("U selector importance scale must be positive")
                u_value.mul_(scale)
                selected_scales.append(scale)
            u_map[spec.name] = {"U": u_value}
        info = {
            "u_provider": "pool",
            "u_pool_generation": self._u_pool_generation,
            "u_direction_sampling": (
                "pool_columns"
                if selected_indices is None
                else "selected_per_target_pool_columns"
            ),
        }
        if selected_indices is not None:
            serialized = "|".join(
                f"{name}:{','.join(str(value) for value in indices)}"
                for name, indices in selected_indices.items()
            )
            info["u_pool_action_digest"] = hashlib.blake2b(
                serialized.encode("utf-8"), digest_size=8
            ).hexdigest()
            info["u_pool_action_num_targets"] = len(selected_indices)
            info["u_pool_action_unique_indices"] = len(
                {
                    int(value)
                    for indices in selected_indices.values()
                    for value in indices
                }
            )
        if selected_scales:
            info["u_pool_importance_scale_mean"] = sum(selected_scales) / len(
                selected_scales
            )
            info["u_pool_importance_scale_max"] = max(selected_scales)
        return u_map, info

    def _selected_indices_for_step(
        self,
        targets: Sequence[DirectionSpec],
        *,
        step: int,
    ) -> dict[str, tuple[int, ...]] | None:
        if self.index_selector is None:
            return None
        raw_by_name = self.index_selector.select_indices(
            step=int(step),
            u_dim=self.u_dim,
            targets=targets,
        )
        expected_names = [spec.name for spec in targets]
        if set(raw_by_name) != set(expected_names):
            missing = sorted(set(expected_names) - set(raw_by_name))
            extra = sorted(set(raw_by_name) - set(expected_names))
            raise ValueError(
                f"U pool selector target mismatch: missing={missing}, extra={extra}"
            )
        selected: dict[str, tuple[int, ...]] = {}
        for spec in targets:
            indices = tuple(int(value) for value in raw_by_name[spec.name])
            effective_rank = int(spec.rank)
            if effective_rank > self.u_dim:
                raise ValueError(f"target rank exceeds U pool dimension: {spec.name}")
            if len(indices) != effective_rank:
                raise ValueError(
                    "U pool selector must return one distinct index per target "
                    f"rank column: {spec.name}"
                )
            if len(set(indices)) != len(indices):
                raise ValueError(
                    f"U pool selector indices must be distinct: {spec.name}"
                )
            if any(index < 0 or index >= self.u_dim for index in indices):
                raise ValueError(
                    f"U pool selector index is outside the configured pool: {spec.name}"
                )
            selected[spec.name] = indices
        return selected

    @torch.no_grad()
    def sample(
        self,
        *,
        name: str,
        out_features: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
        indices: Sequence[int] | None = None,
    ) -> torch.Tensor:
        rank_i = int(rank)
        if rank_i <= 0:
            raise ValueError(f"direction has empty rank: {name}")
        pool = self._u_pool_for(
            name,
            out_features=int(out_features),
            device=device,
            dtype=dtype,
        )
        if indices is not None:
            if len(indices) != rank_i:
                raise ValueError("provided U pool indices do not match rank")
            index_tensor = torch.tensor(
                list(indices),
                device=pool.device,
                dtype=torch.long,
            )
            return pool.index_select(1, index_tensor).contiguous()
        if rank_i <= self.u_dim:
            return self._sample_pool_columns(pool, rank_i)

        parts = []
        remaining = rank_i
        while remaining > 0:
            chunk_rank = min(self.rank, remaining)
            if chunk_rank > self.u_dim:
                raise ValueError(f"rank={rank} exceeds u_dim={self.u_dim} for {name}")
            parts.append(self._sample_pool_columns(pool, chunk_rank))
            remaining -= chunk_rank
        return torch.cat(parts, dim=1).contiguous()

    def _sample_pool_columns(self, pool: torch.Tensor, rank: int) -> torch.Tensor:
        if int(rank) == 1:
            indices = torch.randint(self.u_dim, (1,), device=pool.device)
        else:
            indices = torch.randperm(self.u_dim, device=pool.device)[: int(rank)]
        return pool.index_select(1, indices)


class SubspaceUProvider(_OrthogonalUPoolBase):
    """Sample continuous U directions from a fixed orthogonal U subspace."""

    @torch.no_grad()
    def collect(
        self,
        batch: TokenProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        del batch, step, basis_seed, perturb_seed
        return {
            spec.name: {
                "U": self.sample(
                    name=spec.name,
                    out_features=int(spec.out_features),
                    rank=int(spec.rank),
                    device=spec.device,
                    dtype=spec.dtype,
                )
            }
            for spec in targets
        }, {
            "u_provider": "subspace",
            "u_direction_sampling": "subspace_coefficients",
        }

    @torch.no_grad()
    def sample(
        self,
        *,
        name: str,
        out_features: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if int(rank) <= 0:
            raise ValueError(f"direction has empty rank: {name}")
        pool = self._u_pool_for(
            name,
            out_features=int(out_features),
            device=device,
            dtype=dtype,
        )
        coefficients = torch.randn(
            (self.u_dim, int(rank)),
            device=pool.device,
            dtype=dtype,
        )
        coefficients = coefficients / math.sqrt(self.u_dim)
        return pool.matmul(coefficients).contiguous()

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "u_provider": "subspace",
            "u_normalization": "coefficients/sqrt(u_dim)",
        }


class GaussianVProvider:
    """Sample and cache Gaussian V matrices from parameter metadata."""

    def __init__(
        self,
        *,
        param_metadata: Mapping[str, ParamMetadata],
        rank: int,
        nu: int,
        random_device: str = "cuda",
        direction_sampling: str = "exact",
        direction_scale: float = 1.0,
        v_normalization: str = "none",
    ) -> None:
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        if random_device not in {"cpu", "cuda"}:
            raise ValueError(f"unknown random_device: {random_device}")
        if direction_sampling not in {"exact", "flat"}:
            raise ValueError(f"unknown direction_sampling: {direction_sampling}")
        if v_normalization not in {"none", "unit"}:
            raise ValueError(f"unknown v_normalization: {v_normalization}")
        self.param_metadata = dict(param_metadata)
        self.rank = int(rank)
        self.nu = validate_nu(nu)
        self.random_device = random_device
        self.direction_sampling = direction_sampling
        self.direction_scale = float(direction_scale)
        self.v_normalization = v_normalization
        self.v_cache: dict[str, torch.Tensor] = {}
        self.vt_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def direction_dtype_for(base_dtype: torch.dtype) -> torch.dtype:
        if str(base_dtype).startswith("torch.float8"):
            return torch.float16
        return base_dtype

    def _sample_device_for(self, target_device: torch.device) -> torch.device:
        if self.random_device == "cpu":
            return torch.device("cpu")
        if self.random_device == "cuda":
            if target_device.type != "cuda":
                raise ValueError("random_device='cuda' requires CUDA metadata device")
            return target_device
        raise ValueError(f"unknown random_device: {self.random_device}")

    def _randn(
        self, shape: tuple[int, ...], target_device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        dtype = self.direction_dtype_for(dtype)
        return torch.randn(
            shape,
            dtype=dtype,
            device=self._sample_device_for(target_device),
        ).to(target_device)

    def _randn_flat(
        self,
        numel: int,
        target_device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        dtype = self.direction_dtype_for(dtype)
        return torch.randn(
            (int(numel),),
            dtype=dtype,
            device=self._sample_device_for(target_device),
        ).to(target_device)

    def _normalize_v(self, v: torch.Tensor) -> torch.Tensor:
        if self.v_normalization == "none":
            return v
        if self.v_normalization == "unit":
            denom = v.float().norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
            return (v.float() / denom).to(device=v.device, dtype=v.dtype).contiguous()
        raise ValueError(f"unknown v_normalization: {self.v_normalization}")

    def will_refresh(self, *, step: int) -> bool:
        return self._needs_refresh(self.direction_specs(), step=int(step))

    def direction_specs(self) -> list[DirectionSpec]:
        v_energy_reference = v_energy_reference_from_v_normalization(
            self.v_normalization
        )
        return [
            DirectionSpec(
                name=name,
                out_features=int(metadata.shape[0]),
                in_features=int(metadata.shape[1]),
                rank=int(self.rank),
                device=metadata.device,
                dtype=self.direction_dtype_for(metadata.dtype),
                scale=float(self.direction_scale),
                v_energy_reference=v_energy_reference,
            )
            for name, metadata in self.param_metadata.items()
            if metadata.ndim >= 2
        ]

    def collect(
        self,
        batch: TokenProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        del batch, basis_seed, perturb_seed
        target_list = list(targets)
        refreshed = self._needs_refresh(target_list, step=int(step))
        directions: dict[str, dict[str, Any]] = {}
        if self.direction_sampling == "flat" and target_list:
            sample_device = target_list[0].device
            sample_dtype = target_list[0].dtype
            if any(
                spec.device != sample_device or spec.dtype != sample_dtype
                for spec in target_list
            ):
                raise RuntimeError(
                    "flat V sampling requires uniform target device and dtype"
                )
            if refreshed:
                total_v = sum(
                    int(spec.in_features) * int(spec.rank) for spec in target_list
                )
                flat_v = self._randn_flat(total_v, sample_device, sample_dtype)
                offset = 0
                for spec in target_list:
                    numel = int(spec.in_features) * int(spec.rank)
                    v = flat_v[offset : offset + numel].view(
                        int(spec.in_features),
                        int(spec.rank),
                    )
                    v = self._normalize_v(v)
                    self._cache_v(spec.name, v)
                    offset += numel
            for spec in target_list:
                directions[spec.name] = self._direction_metadata(
                    spec,
                    v_refreshed=refreshed,
                )
            return directions, {
                "v_provider": "gaussian",
                "direction_sampling": "flat",
                "v_refreshed": bool(refreshed),
            }

        for spec in target_list:
            if refreshed:
                v = self._randn(
                    (int(spec.in_features), int(spec.rank)),
                    spec.device,
                    spec.dtype,
                )
                v = self._normalize_v(v)
                self._cache_v(spec.name, v)
            directions[spec.name] = self._direction_metadata(
                spec,
                v_refreshed=refreshed,
            )
        return directions, {
            "v_provider": "gaussian",
            "direction_sampling": "exact",
            "v_refreshed": bool(refreshed),
        }

    def _needs_refresh(
        self,
        targets: Sequence[DirectionSpec],
        *,
        step: int,
    ) -> bool:
        if int(step) <= 0:
            raise ValueError("step must be positive")
        if not targets:
            return False
        if any(spec.name not in self.v_cache for spec in targets):
            return True
        return should_refresh_for_nu(step=int(step), nu=self.nu)

    def _cache_v(self, name: str, v: torch.Tensor) -> None:
        self.v_cache[name] = v
        self.vt_cache[name] = v.T.contiguous()

    def _direction_metadata(
        self,
        spec: DirectionSpec,
        *,
        v_refreshed: bool,
    ) -> dict[str, Any]:
        v = self.v_cache[spec.name]
        direction = {
            "U": torch.empty(
                (int(spec.out_features), int(v.shape[1])),
                device=spec.device,
                dtype=spec.dtype,
            ),
            "V": v,
            "V_T": self.vt_cache[spec.name],
            "scale": float(spec.scale),
            "v_refreshed": bool(v_refreshed),
        }
        attach_factorized_perturbation_spec_(
            direction,
            v_energy_reference=spec.v_energy_reference,
        )
        return direction

    def info(self) -> dict[str, Any]:
        return {
            "v_provider": "gaussian",
            "direction_sampling": self.direction_sampling,
            "v_normalization": self.v_normalization,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "gaussian_v",
            "v_cache": {
                name: value.detach().cpu() for name, value in self.v_cache.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("type") != "gaussian_v":
            raise ValueError("invalid Gaussian V provider checkpoint")
        specs = {spec.name: spec for spec in self.direction_specs()}
        restored = dict(state.get("v_cache", {}))
        self.v_cache.clear()
        self.vt_cache.clear()
        for name, raw_v in restored.items():
            if name not in specs:
                raise KeyError(f"unknown Gaussian V cache parameter: {name}")
            if not isinstance(raw_v, torch.Tensor):
                raise TypeError(f"Gaussian V cache must contain tensors: {name}")
            spec = specs[name]
            expected_shape = (int(spec.in_features), int(spec.rank))
            if tuple(raw_v.shape) != expected_shape:
                raise ValueError(f"Gaussian V cache shape mismatch: {name}")
            v = raw_v.to(device=spec.device, dtype=spec.dtype).contiguous()
            self._cache_v(name, v)


class AGZOVProvider:
    """Generate activation-guided V matrices."""

    def __init__(
        self,
        *,
        engine: ZOVLLMEngine,
        rank: int,
        power_iter_steps: int = 5,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
        activation_force_eager: bool = True,
        low_rank_oversample: int = 4,
        basis_method: str = "power_iter",
        direction_scale: float = 1.0,
    ) -> None:
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        if int(power_iter_steps) <= 0:
            raise ValueError("power_iter_steps must be positive")
        if basis_method not in {"power_iter", "svd", "low_rank_svd"}:
            raise ValueError(f"unsupported basis_method: {basis_method}")
        if float(direction_scale) < 0.0:
            raise ValueError("direction_scale must be non-negative")
        self.engine = engine
        self.rank = int(rank)
        self.power_iter_steps = int(power_iter_steps)
        self.max_logits_tokens = int(max_logits_tokens)
        self.loss_impl = loss_impl
        self.activation_force_eager = bool(activation_force_eager)
        self.low_rank_oversample = int(low_rank_oversample)
        self.basis_method = basis_method
        self.direction_scale = float(direction_scale)

    def collect(
        self,
        batch: TokenProbeBatch | SubspaceTokenProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        del targets, step
        batches = getattr(batch, "subspace_token_id_group_batches", None)
        labels_batches = getattr(batch, "subspace_labels_batches", None)
        batch_factory = getattr(batch, "subspace_token_id_group_batch_factory", None)
        if batches is None and batch_factory is not None:
            factory_result = batch_factory()
            if isinstance(factory_result, tuple):
                batches, labels_batches = factory_result
            else:
                batches = factory_result
        if batches is None:
            directions, raw = self.engine.collect_agzo_directions(
                batch.token_id_groups,
                loss_token_lens=batch.loss_token_lens,
                labels=batch.labels,
                max_logits_tokens=self.max_logits_tokens,
                loss_impl=self.loss_impl,
                activation_force_eager=self.activation_force_eager,
                agzo_rank=self.rank,
                agzo_power_iter_steps=self.power_iter_steps,
                agzo_low_rank_oversample=self.low_rank_oversample,
                agzo_basis_seed=basis_seed,
                agzo_perturb_seed=perturb_seed,
                agzo_basis_method=self.basis_method,
            )
            self._attach_direction_scale(directions)
            return directions, raw
        del labels_batches
        directions, raw = self.engine.collect_agzo_directions_chunked(
            batches,
            activation_force_eager=self.activation_force_eager,
            agzo_rank=self.rank,
            agzo_power_iter_steps=self.power_iter_steps,
            agzo_low_rank_oversample=self.low_rank_oversample,
            agzo_basis_seed=basis_seed,
            agzo_perturb_seed=perturb_seed,
            agzo_basis_method=self.basis_method,
        )
        self._attach_direction_scale(directions)
        raw = dict(raw)
        raw["subspace_num_chunks"] = len(batches)
        if getattr(batch, "subspace_num_rows", None) is not None:
            raw["kappa"] = int(batch.subspace_num_rows)
        return directions, raw

    def _attach_direction_scale(self, directions: dict[str, dict[str, Any]]) -> None:
        for direction in directions.values():
            if "U" in direction:
                direction["U"] = torch.empty_like(direction["U"])
            direction["scale"] = self.direction_scale
            if "V" in direction and "V_T" not in direction:
                direction["V_T"] = direction["V"].T.contiguous()
            if "V" in direction:
                attach_factorized_perturbation_spec_(
                    direction,
                    v_energy_reference=PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
                )


class QueuedAGZOVProvider:
    """AGZO V provider with a small FIFO ring over collected V subspaces."""

    def __init__(
        self,
        inner: AGZOVProvider,
        *,
        nu: int = 1,
        queue_size: int = 1,
    ) -> None:
        validate_nu(nu)
        if int(queue_size) <= 0:
            raise ValueError("queue_size must be positive")
        self.inner = inner
        self.nu = int(nu)
        self.subspace_queue = SubspaceQueue(queue_size)
        self._preinitialized_step: int | None = None
        self._last_refresh_step: int | None = None

    def will_refresh(self, *, step: int) -> bool:
        if self._preinitialized_step == int(step) and not self.subspace_queue.is_empty:
            return False
        if should_refresh_for_nu(step=int(step), nu=self.nu):
            return True
        return bool(self.subspace_queue.is_empty)

    def prime(
        self,
        batch: SubspaceTokenProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> dict[str, Any]:
        """Initialize V before an HF-native step consumes a plain token batch."""

        step_i = int(step)
        if step_i <= 0:
            raise ValueError("preinitialized V step must be positive")
        if not self.subspace_queue.is_empty:
            raise RuntimeError("cannot preinitialize a non-empty V queue")
        cached_directions, raw = self.inner.collect(
            batch,
            targets=targets,
            step=step_i,
            basis_seed=int(basis_seed),
            perturb_seed=int(perturb_seed),
        )
        victim_idx = self.subspace_queue.insert(cached_directions)
        self._preinitialized_step = step_i
        return {
            **dict(raw),
            "basis_preinitialized": True,
            "subspace_queue_victim_idx": int(victim_idx),
        }

    def collect(
        self,
        batch: SubspaceTokenProbeBatch,
        *,
        targets: Sequence[DirectionSpec],
        step: int,
        basis_seed: int,
        perturb_seed: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        step_i = int(step)
        preinitialized = (
            self._preinitialized_step == step_i and not self.subspace_queue.is_empty
        )
        if preinitialized:
            # consuming the primed V counts as this step's refresh
            self._last_refresh_step = step_i
        refreshed = self.will_refresh(step=step_i)
        if refreshed and self._last_refresh_step == step_i:
            # A refresh step is refreshed once. Several collects within the
            # same step (one per probe of a multi-query estimate) must all see
            # the V computed by the first, not re-run the seeded power
            # iteration and hand every probe a slightly different V.
            refreshed = False
        if refreshed:
            self._last_refresh_step = step_i
            cached_directions, raw = self.inner.collect(
                batch,
                targets=targets,
                step=step,
                basis_seed=basis_seed,
                perturb_seed=perturb_seed,
            )
            victim_idx = self.subspace_queue.insert(cached_directions)
            raw = dict(raw)
            raw["subspace_queue_victim_idx"] = int(victim_idx)
        else:
            raw = {"basis_reused": True}
            if preinitialized:
                raw["basis_preinitialized"] = True

        directions = self._combine_v_slots(
            self.subspace_queue.active_slot_items(),
            queue_size=self.subspace_queue.queue_size,
            v_refreshed=refreshed,
        )
        if preinitialized:
            self._preinitialized_step = None
        return directions, raw

    def info(self) -> dict[str, Any]:
        return {
            "v_provider": "queued_agzo",
            "subspace_queue_active_slots": self.subspace_queue.active_slots,
            "subspace_queue_size": self.subspace_queue.queue_size,
            "subspace_queue_next_victim_idx": self.subspace_queue.victim_idx,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "queued_agzo_v",
            "nu": int(self.nu),
            "subspace_queue": self.subspace_queue.state_dict(),
            "preinitialized_step": self._preinitialized_step,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        direction_specs: Sequence[DirectionSpec],
    ) -> None:
        if state.get("type") != "queued_agzo_v":
            raise ValueError("invalid queued AGZO V provider checkpoint")
        if int(state["nu"]) != self.nu:
            raise ValueError("queued AGZO nu does not match checkpoint")
        self.subspace_queue.load_state_dict(
            state["subspace_queue"],
            direction_specs=direction_specs,
        )
        raw_step = state.get("preinitialized_step")
        self._preinitialized_step = None if raw_step is None else int(raw_step)

    @torch.no_grad()
    def _combine_v_slots(
        self,
        slots: Sequence[tuple[int, Mapping[str, Mapping[str, Any]]]],
        *,
        queue_size: int,
        v_refreshed: bool,
    ) -> dict[str, dict[str, Any]]:
        if not slots:
            raise ValueError("cannot combine empty V slots")
        first_slot = slots[0][1]
        if int(queue_size) == 1 and len(slots) == 1:
            return {
                name: self._clone_v_metadata(direction, v_refreshed=v_refreshed)
                for name, direction in dict(first_slot).items()
            }

        active_count = len(slots)
        combined: dict[str, dict[str, Any]] = {}
        for name, first_direction in dict(first_slot).items():
            rank = int(first_direction["V"].shape[1])
            v_parts = []
            for slot_idx in range(int(queue_size)):
                slot_direction = next(
                    (directions[name] for idx, directions in slots if idx == slot_idx),
                    None,
                )
                if slot_direction is None:
                    v_parts.append(torch.zeros_like(first_direction["V"]))
                    continue
                if int(slot_direction["V"].shape[1]) != rank:
                    raise ValueError(f"subspace queue rank mismatch for {name}")
                v_parts.append(slot_direction["V"])
            v = torch.cat(v_parts, dim=1).contiguous()
            combined[name] = self._clone_v_metadata(
                first_direction,
                v=v,
                v_refreshed=v_refreshed,
                effective_rank=active_count * rank,
            )
        return combined

    def _clone_v_metadata(
        self,
        direction: Mapping[str, Any],
        *,
        v: torch.Tensor | None = None,
        v_refreshed: bool,
        effective_rank: int | None = None,
    ) -> dict[str, Any]:
        cached_u = direction["U"]
        v = direction["V"] if v is None else v
        cloned: dict[str, Any] = {
            "U": torch.empty(
                (int(cached_u.shape[0]), int(v.shape[1])),
                device=cached_u.device,
                dtype=cached_u.dtype,
            ),
            "V": v,
            "V_T": v.T.contiguous(),
            "v_refreshed": bool(v_refreshed),
        }
        attach_factorized_perturbation_spec_(
            cloned,
            v_energy_reference=str(
                direction.get(
                    "perturbation_v_energy_reference",
                    PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
                )
            ),
            effective_rank=(
                effective_rank
                if effective_rank is not None
                else int(direction.get("perturbation_effective_rank", int(v.shape[1])))
            ),
        )
        if "scale" in direction:
            cloned["scale"] = direction["scale"]
        return cloned


__all__ = [
    "AGZOVProvider",
    "GaussianUProvider",
    "PoolUProvider",
    "QueuedAGZOVProvider",
    "SubspaceUProvider",
    "UPoolIndexSelector",
    "GaussianVProvider",
    "UProvider",
    "VProvider",
]
