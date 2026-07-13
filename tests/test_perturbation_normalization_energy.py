from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.core.perturbation_normalization import (
    PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
    PERTURBATION_V_ENERGY_REFERENCE_UNIT,
    perturbation_normalization_info,
)
from zo_vllm.training.direction import (
    AGZODirectionProvider,
    LOZOFastDirectionProvider,
    SUAGZODirectionProvider,
    UAGZODirectionProvider,
    TokenProbeBatch,
)


_RANKS = (1, 2, 4)
_LOZO_SHAPES = {
    "model.decoder.layers.0.fc1.weight": (384, 320),
    "model.decoder.layers.0.fc2.weight": (320, 384),
}
_AGZO_SHAPES = {
    "model.layers.0.fc1.weight": (384, 320),
    "model.layers.0.fc2.weight": (320, 384),
}


class UnitVFakeAGZOEngine:
    def __init__(self, shapes: dict[str, tuple[int, int]]) -> None:
        self.shapes = shapes

    def collect_agzo_directions(self, token_id_groups, **kwargs):
        del token_id_groups
        rank = int(kwargs.get("agzo_rank", 1))
        return self._directions(rank), {"basis_path": "direct"}

    def collect_agzo_directions_chunked(self, batches, **kwargs):
        del batches
        rank = int(kwargs.get("agzo_rank", 1))
        return self._directions(rank), {"basis_path": "chunked"}

    def _directions(self, rank: int) -> dict[str, dict[str, torch.Tensor]]:
        directions = {}
        for name, (out_features, in_features) in self.shapes.items():
            if rank > in_features:
                raise ValueError("rank exceeds in_features")
            v = torch.eye(in_features, rank, dtype=torch.float32)
            directions[name] = {
                "U": torch.empty((out_features, rank), dtype=torch.float32),
                "V": v,
            }
        return directions


def _lozo_metadata(shapes: dict[str, tuple[int, int]]) -> dict[str, ParamMetadata]:
    return {
        name: ParamMetadata(
            name=name,
            shape=shape,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        for name, shape in shapes.items()
    }


def _global_realized_rms(directions: dict[str, dict[str, Any]]) -> float:
    fro_sq = 0.0
    numel = 0
    for direction in directions.values():
        u = direction["U"].float()
        v = direction["V"].float()
        scale = float(direction.get("scale", 1.0))
        perturbation = scale * u.matmul(v.T)
        fro_sq += float(perturbation.square().sum())
        numel += int(u.shape[0]) * int(v.shape[0])
    return math.sqrt(fro_sq / float(numel))


def _global_u_rms(directions: dict[str, dict[str, Any]]) -> float:
    sq_sum = 0.0
    numel = 0
    for direction in directions.values():
        u = direction["U"].float()
        sq_sum += float(u.square().sum())
        numel += int(u.numel())
    return math.sqrt(sq_sum / float(numel))


def _first_normalization_scale(directions: dict[str, dict[str, Any]]) -> float:
    return float(next(iter(directions.values()))["perturbation_normalization_scale"])


def _lozo_directions(*, rank: int, seed: int, normalization: str = "rms"):
    provider = LOZOFastDirectionProvider(
        param_metadata=_lozo_metadata(_LOZO_SHAPES),
        rank=rank,
        nu=10,
        random_device="cpu",
        perturbation_normalization=normalization,
    )
    return provider.sample_direction(seed)


def _lozo_sample_info(
    *,
    rank: int,
    seed: int,
    direction_scale: float,
    normalization: str,
) -> dict[str, Any]:
    provider = LOZOFastDirectionProvider(
        param_metadata=_lozo_metadata(_LOZO_SHAPES),
        rank=rank,
        nu=10,
        random_device="cpu",
        direction_scale=direction_scale,
        perturbation_normalization=normalization,
    )
    sample = provider.next(TokenProbeBatch(token_id_groups=[[10, 11]]), step=seed)
    return sample.info


def _agzo_family_directions(*, provider_name: str, rank: int, step: int):
    engine = UnitVFakeAGZOEngine(_AGZO_SHAPES)
    common_kwargs = {
        "engine": engine,
        "param_metadata": _lozo_metadata(_AGZO_SHAPES),
        "rank": rank,
        "nu": 20000,
        "power_iter_steps": 1,
        "seed": 11,
        "perturbation_normalization": "rms",
    }
    if provider_name == "agzo":
        provider = AGZODirectionProvider(**common_kwargs)
    elif provider_name == "uagzo":
        provider = UAGZODirectionProvider(**common_kwargs, u_dim=16)
    elif provider_name == "suagzo":
        provider = SUAGZODirectionProvider(**common_kwargs, u_dim=64)
    else:
        raise ValueError(f"unknown provider: {provider_name}")
    sample = provider.next(TokenProbeBatch(token_id_groups=[[10, 11]]), step=step)
    assert sample.info["perturbation_normalization"] == "rms"
    return sample.directions


@pytest.mark.parametrize("rank", _RANKS)
def test_lozo_perturbation_normalization_keeps_realized_rms_near_one(
    rank: int,
) -> None:
    directions = _lozo_directions(rank=rank, seed=12345 + rank)
    realized_rms = _global_realized_rms(directions)

    assert realized_rms == pytest.approx(1.0, rel=0.12)


@pytest.mark.parametrize("rank", (2, 4))
def test_lozo_without_perturbation_normalization_keeps_rank_dependent_energy(
    rank: int,
) -> None:
    directions = _lozo_directions(
        rank=rank,
        seed=12345 + rank,
        normalization="none",
    )
    realized_rms = _global_realized_rms(directions)

    assert realized_rms == pytest.approx(math.sqrt(rank), rel=0.15)


@pytest.mark.parametrize("provider_name", ("agzo", "uagzo", "suagzo"))
@pytest.mark.parametrize("rank", _RANKS)
def test_agzo_family_perturbation_normalization_is_rank_only(
    provider_name: str,
    rank: int,
) -> None:
    directions = _agzo_family_directions(
        provider_name=provider_name,
        rank=rank,
        step=rank,
    )
    expected_scale = 1.0 / math.sqrt(rank)

    assert _first_normalization_scale(directions) == pytest.approx(
        expected_scale,
        rel=1e-6,
    )
    first = next(iter(directions.values()))
    assert (
        first["perturbation_v_energy_reference"]
        == PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN
    )
    assert "perturbation_v_expected_column_norm_sq" not in first


def test_lozo_unit_v_normalization_uses_unit_energy_reference() -> None:
    provider = LOZOFastDirectionProvider(
        param_metadata=_lozo_metadata({"layer.weight": (8, 4)}),
        rank=1,
        nu=10,
        random_device="cpu",
        v_normalization="unit",
        perturbation_normalization="rms",
    )
    directions = provider.sample_direction(123)
    first = next(iter(directions.values()))

    assert (
        first["perturbation_v_energy_reference"] == PERTURBATION_V_ENERGY_REFERENCE_UNIT
    )
    assert first["perturbation_normalization_scale"] == pytest.approx(2.0)
    assert "perturbation_v_expected_column_norm_sq" not in first


def test_perturbation_info_reports_effective_rms_after_direction_scale() -> None:
    rank = 4
    info = _lozo_sample_info(
        rank=rank,
        seed=17,
        direction_scale=1.0 / math.sqrt(rank),
        normalization="none",
    )

    assert info["perturbation_normalization"] == "none"
    assert info["perturbation_expected_raw_rms"] == pytest.approx(math.sqrt(rank))
    assert info["perturbation_direction_scale"] == pytest.approx(1.0 / math.sqrt(rank))
    assert info["perturbation_effective_rms"] == pytest.approx(1.0)


def test_perturbation_info_reports_effective_rms_after_rms_normalization() -> None:
    info = _lozo_sample_info(
        rank=4,
        seed=18,
        direction_scale=1.0,
        normalization="rms",
    )

    assert info["perturbation_normalization"] == "rms"
    assert info["perturbation_direction_scale"] == pytest.approx(
        info["perturbation_normalization_scale"]
    )
    assert info["perturbation_effective_rms"] == pytest.approx(1.0)


def test_perturbation_info_can_be_read_directly_from_direction_map() -> None:
    directions = _lozo_directions(rank=4, seed=19)
    info = perturbation_normalization_info(directions)

    assert info["perturbation_effective_rms"] == pytest.approx(1.0)


@pytest.mark.parametrize("rank", _RANKS)
def test_all_direction_providers_have_aligned_u_energy(rank: int) -> None:
    provider_to_u_rms = {
        "lozo": _global_u_rms(_lozo_directions(rank=rank, seed=54321 + rank)),
        "agzo": _global_u_rms(
            _agzo_family_directions(provider_name="agzo", rank=rank, step=rank)
        ),
        "uagzo": _global_u_rms(
            _agzo_family_directions(provider_name="uagzo", rank=rank, step=rank)
        ),
        "suagzo": _global_u_rms(
            _agzo_family_directions(provider_name="suagzo", rank=rank, step=rank)
        ),
    }
    spread = max(provider_to_u_rms.values()) - min(provider_to_u_rms.values())

    assert spread <= 0.15, provider_to_u_rms
