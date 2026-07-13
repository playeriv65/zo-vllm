"""Utilities for analytic factorized perturbation energy normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from typing import Any


PERTURBATION_NORMALIZATION_NONE = "none"
PERTURBATION_NORMALIZATION_RMS = "rms"
PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN = "gaussian"
PERTURBATION_V_ENERGY_REFERENCE_UNIT = "unit"
SUPPORTED_PERTURBATION_NORMALIZATIONS = {
    PERTURBATION_NORMALIZATION_NONE,
    PERTURBATION_NORMALIZATION_RMS,
}
SUPPORTED_PERTURBATION_V_ENERGY_REFERENCES = {
    PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
    PERTURBATION_V_ENERGY_REFERENCE_UNIT,
}
AGZO_DIRECTION_PROVIDER_NAMES = {"agzo", "uagzo", "suagzo"}


def normalize_perturbation_normalization(value: str | bool | None) -> str:
    """Parse a user-facing perturbation normalization value."""

    if value is None:
        return PERTURBATION_NORMALIZATION_RMS
    if isinstance(value, bool):
        return (
            PERTURBATION_NORMALIZATION_RMS if value else PERTURBATION_NORMALIZATION_NONE
        )
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"1", "true", "yes", "on", "rms", "rms_per_weight"}:
        return PERTURBATION_NORMALIZATION_RMS
    if normalized in {"0", "false", "no", "off", "none", "disabled"}:
        return PERTURBATION_NORMALIZATION_NONE
    raise ValueError(f"unknown perturbation_normalization: {value}")


def normalize_perturbation_energy_(
    directions: MutableMapping[str, MutableMapping[str, Any]],
    *,
    mode: str | bool | None,
) -> dict[str, float | str]:
    """Normalize expected global ``U @ V.T`` RMS and multiply into scale.

    ``direction["scale"]`` remains the user-facing amplitude knob. This helper
    uses analytic metadata such as rank and expected V column norm, avoiding
    hot-path tensor norm computation.
    """

    normalized_mode = normalize_perturbation_normalization(mode)
    if normalized_mode == PERTURBATION_NORMALIZATION_NONE:
        stats = _normalization_stats(
            mode=normalized_mode,
            expected_raw_fro_sq=_expected_raw_perturbation_fro_sq(directions),
            target_fro_sq=_target_fro_sq(directions),
            scale=1.0,
        )
        _attach_stats(directions, stats)
        return stats

    raw_fro_sq = _expected_raw_perturbation_fro_sq(directions)
    target_fro_sq = _target_fro_sq(directions)
    if raw_fro_sq <= 0.0 or target_fro_sq <= 0.0:
        scale = 1.0
    else:
        scale = math.sqrt(target_fro_sq / raw_fro_sq)

    for direction in directions.values():
        direction["scale"] = float(direction.get("scale", 1.0)) * scale

    stats = _normalization_stats(
        mode=normalized_mode,
        expected_raw_fro_sq=raw_fro_sq,
        target_fro_sq=target_fro_sq,
        scale=scale,
    )
    _attach_stats(directions, stats)
    return stats


def perturbation_normalization_info(
    directions: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | str]:
    """Read normalization metadata from a direction map."""

    for direction in directions.values():
        info = {
            key: value
            for key, value in direction.items()
            if key.startswith("perturbation_") and isinstance(value, (float, int, str))
        }
        scale = float(direction.get("scale", 1.0))
        info["perturbation_direction_scale"] = scale
        raw_rms = info.get("perturbation_expected_raw_rms")
        raw_fro_norm = info.get("perturbation_expected_raw_fro_norm")
        if isinstance(raw_rms, (float, int)):
            info["perturbation_effective_rms"] = float(raw_rms) * scale
        if isinstance(raw_fro_norm, (float, int)):
            info["perturbation_effective_fro_norm"] = float(raw_fro_norm) * scale
        return info
    return {}


def v_energy_reference_from_v_normalization(
    v_normalization: str | None,
) -> str:
    """Map a V generation policy to the normalization energy reference."""

    return (
        PERTURBATION_V_ENERGY_REFERENCE_UNIT
        if v_normalization == "unit"
        else PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN
    )


def v_energy_reference_for_direction_provider(
    *,
    direction_provider_name: str,
    v_normalization: str | None,
) -> str:
    """Return the V energy reference implied by a direction provider."""

    if direction_provider_name in AGZO_DIRECTION_PROVIDER_NAMES:
        return PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN
    return v_energy_reference_from_v_normalization(v_normalization)


def reference_v_column_norm_sq(
    *,
    in_features: int,
    v_energy_reference: str,
) -> float:
    """Return the reference expected squared norm for one V column."""

    if v_energy_reference == PERTURBATION_V_ENERGY_REFERENCE_UNIT:
        return 1.0
    if v_energy_reference == PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN:
        return float(in_features)
    raise ValueError(f"unknown perturbation V energy reference: {v_energy_reference}")


def attach_factorized_perturbation_spec_(
    direction: MutableMapping[str, Any],
    *,
    v_energy_reference: str,
    effective_rank: int | None = None,
) -> None:
    """Attach the normalization spec for a factorized direction."""

    v = direction["V"]
    if v_energy_reference not in SUPPORTED_PERTURBATION_V_ENERGY_REFERENCES:
        raise ValueError(
            f"unknown perturbation V energy reference: {v_energy_reference}"
        )
    direction["perturbation_effective_rank"] = int(
        int(v.shape[1]) if effective_rank is None else effective_rank
    )
    direction["perturbation_v_energy_reference"] = v_energy_reference


def _raw_perturbation_fro_sq(
    directions: Mapping[str, Mapping[str, Any]],
) -> float:
    return _expected_raw_perturbation_fro_sq(directions)


def _expected_raw_perturbation_fro_sq(
    directions: Mapping[str, Mapping[str, Any]],
) -> float:
    total = 0.0
    for direction in directions.values():
        u = direction["U"]
        v = direction["V"]
        out_features = int(u.shape[0])
        in_features = int(v.shape[0])
        effective_rank = int(
            direction.get("perturbation_effective_rank", int(v.shape[1]))
        )
        v_column_norm_sq = _direction_v_column_norm_sq(
            direction,
            in_features=in_features,
        )
        total += float(out_features * effective_rank) * v_column_norm_sq
    return total


def _direction_v_column_norm_sq(
    direction: Mapping[str, Any],
    *,
    in_features: int,
) -> float:
    if "perturbation_v_energy_reference" in direction:
        return reference_v_column_norm_sq(
            in_features=in_features,
            v_energy_reference=str(direction["perturbation_v_energy_reference"]),
        )
    if "perturbation_v_expected_column_norm_sq" in direction:
        return float(direction["perturbation_v_expected_column_norm_sq"])
    return float(in_features)


def _target_fro_sq(directions: Mapping[str, Mapping[str, Any]]) -> float:
    total = 0
    for direction in directions.values():
        u = direction["U"]
        v = direction["V"]
        total += int(u.shape[0]) * int(v.shape[0])
    return float(total)


def _normalization_stats(
    *,
    mode: str,
    expected_raw_fro_sq: float,
    target_fro_sq: float,
    scale: float,
) -> dict[str, float | str]:
    raw_fro_norm = math.sqrt(max(expected_raw_fro_sq, 0.0))
    target_fro_norm = math.sqrt(max(target_fro_sq, 0.0))
    return {
        "perturbation_normalization": mode,
        "perturbation_normalization_scale": float(scale),
        "perturbation_expected_raw_fro_norm": float(raw_fro_norm),
        "perturbation_target_fro_norm": float(target_fro_norm),
        "perturbation_expected_raw_rms": (
            float(raw_fro_norm / target_fro_norm) if target_fro_norm > 0.0 else 0.0
        ),
        "perturbation_target_rms": 1.0 if target_fro_norm > 0.0 else 0.0,
    }


def _attach_stats(
    directions: MutableMapping[str, MutableMapping[str, Any]],
    stats: Mapping[str, float | str],
) -> None:
    for direction in directions.values():
        direction.update(stats)
