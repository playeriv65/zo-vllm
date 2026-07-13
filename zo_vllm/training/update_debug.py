"""Debug helpers for ZO update-state tensors."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import torch


def zo_bank_debug_enabled() -> bool:
    return os.environ.get("VLLM_ZO_DEBUG_BANK_FINITE", "0") != "0"


def tensor_finite_summary(tensor: torch.Tensor) -> dict[str, Any]:
    item: dict[str, Any] = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "data_ptr": int(tensor.data_ptr()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "floating": bool(tensor.is_floating_point()),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0 or not tensor.is_floating_point():
        item["finite"] = True
        return item
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    bad_count = int(tensor.numel() - finite_count)
    item.update(
        {
            "finite": bad_count == 0,
            "finite_count": finite_count,
            "bad_count": bad_count,
        }
    )
    if bad_count:
        bad_values = tensor[~finite]
        item.update(
            {
                "has_nan": bool(torch.isnan(bad_values).any().item()),
                "has_posinf": bool(torch.isposinf(bad_values).any().item()),
                "has_neginf": bool(torch.isneginf(bad_values).any().item()),
            }
        )
        if tensor.ndim >= 2:
            bad_rows = (~finite).reshape(int(tensor.shape[0]), -1).any(dim=1)
            item["bad_rows_head"] = (
                bad_rows.nonzero(as_tuple=False).flatten()[:16].detach().cpu().tolist()
            )
    return item


def direction_tensors_finite_summary(
    directions: Mapping[str, Mapping[str, Any]],
    *,
    max_bad: int = 8,
) -> dict[str, Any]:
    groups = {
        "U": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
        "V": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
        "V_T": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
        "U_accum": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
        "lora_B_plus": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
        "lora_B_minus": {"num_tensors": 0, "num_bad_tensors": 0, "first_bad": None},
    }
    bad: list[dict[str, Any]] = []
    for name, direction in dict(directions).items():
        for key in groups:
            tensor = direction.get(key)
            if not isinstance(tensor, torch.Tensor):
                continue
            groups[key]["num_tensors"] = int(groups[key]["num_tensors"]) + 1
            summary = tensor_finite_summary(tensor)
            if bool(summary.get("finite", True)):
                continue
            item = {
                "name": name,
                "tensor": key,
                **summary,
                "scale": direction.get("scale"),
                "v_refreshed": direction.get("v_refreshed"),
            }
            groups[key]["num_bad_tensors"] = int(groups[key]["num_bad_tensors"]) + 1
            if groups[key]["first_bad"] is None:
                groups[key]["first_bad"] = item
            if len(bad) < max_bad:
                bad.append(item)
    return {
        "checked": True,
        "num_bad_tensors": sum(int(row["num_bad_tensors"]) for row in groups.values()),
        "bad": bad,
        "groups": groups,
    }
