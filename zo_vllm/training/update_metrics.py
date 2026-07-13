"""Shared metrics for ZO update-state snapshots."""

from __future__ import annotations

import math
from typing import Any

import torch


def build_update_snapshot_metrics(
    *,
    step: int,
    module_rows: list[dict[str, Any]],
    flat_parts: list[torch.Tensor],
    total_u_sq: float,
    total_w_sq: float,
    previous_flat: torch.Tensor | None,
    previous_delta: torch.Tensor | None,
    extra_metrics: dict[str, Any] | None = None,
    require_matching_previous_shape: bool = False,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    flat = torch.cat(flat_parts)
    delta = None
    delta_norm = None
    cosine = None
    angle = None
    delta_cosine = None
    delta_angle = None
    can_compare_previous = previous_flat is not None and (
        not require_matching_previous_shape or previous_flat.numel() == flat.numel()
    )
    if can_compare_previous and previous_flat is not None:
        delta = flat - previous_flat
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        cosine, angle = cosine_and_angle(flat, previous_flat)
        can_compare_delta = previous_delta is not None and (
            not require_matching_previous_shape
            or previous_delta.numel() == delta.numel()
        )
        if can_compare_delta and previous_delta is not None:
            delta_cosine, delta_angle = cosine_and_angle(delta, previous_delta)
    next_delta = torch.zeros_like(flat) if delta is None else delta
    metrics = {
        "step": int(step),
        "u_total_norm": float(total_u_sq**0.5),
        "u_delta_norm_since_prev": delta_norm,
        "u_cosine_to_prev": cosine,
        "u_angle_deg_to_prev": angle,
        "u_delta_cosine_to_prev_delta": delta_cosine,
        "u_delta_angle_deg_to_prev_delta": delta_angle,
        "delta_w_fro_est": float(total_w_sq**0.5),
        "num_modules": len(module_rows),
        "top_modules_by_u_norm": sorted(
            module_rows,
            key=lambda row: row["u_norm"],
            reverse=True,
        )[:10],
    }
    if extra_metrics:
        metrics |= dict(extra_metrics)
    return metrics, flat, next_delta


def cosine_and_angle(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float | None, float | None]:
    denom = float(
        torch.linalg.vector_norm(left).item() * torch.linalg.vector_norm(right).item()
    )
    if denom <= 0.0:
        return None, None
    cosine = float(torch.dot(left, right).item() / denom)
    clamped = max(-1.0, min(1.0, cosine))
    return cosine, float(math.degrees(math.acos(clamped)))
