"""Debug helpers for direct LoRA slot updates."""

import logging
import os
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)


def _debug_env(name: str, default: str = "0") -> str:
    return os.environ.get(name, default)


def _debug_slot_tensor_finite_summary(
    named_groups: Sequence[
        tuple[str, Sequence[torch.Tensor]]
        | tuple[str, Sequence[torch.Tensor], Sequence[str]]
    ],
) -> dict[str, Any] | None:
    if _debug_env("VLLM_ZO_DEBUG_SLOT_FINITE") == "0":
        return None
    summaries: dict[str, Any] = {}
    bad: list[dict[str, Any]] = []
    for group in named_groups:
        if len(group) == 2:
            group_name, tensors = group
            tensor_names: Sequence[str] = ()
        else:
            group_name, tensors, tensor_names = group
        group_summary = {
            "num_tensors": len(tensors),
            "num_bad_tensors": 0,
            "first_bad": None,
        }
        for idx, tensor in enumerate(tensors):
            if tensor.numel() == 0 or not tensor.is_floating_point():
                continue
            finite = torch.isfinite(tensor)
            if bool(finite.all().item()):
                continue
            bad_count = int(tensor.numel() - int(finite.sum().item()))
            item = {
                "group": group_name,
                "index": idx,
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "storage_offset": int(tensor.storage_offset()),
                "data_ptr": int(tensor.data_ptr()),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "device": str(tensor.device),
                "bad_count": bad_count,
                "has_nan": bool(torch.isnan(tensor[~finite]).any().item()),
                "has_posinf": bool(torch.isposinf(tensor[~finite]).any().item()),
                "has_neginf": bool(torch.isneginf(tensor[~finite]).any().item()),
            }
            if idx < len(tensor_names):
                item["name"] = tensor_names[idx]
            group_summary["num_bad_tensors"] = int(group_summary["num_bad_tensors"]) + 1
            if group_summary["first_bad"] is None:
                group_summary["first_bad"] = item
            if len(bad) < 8:
                bad.append(item)
        summaries[group_name] = group_summary
    result = {
        "checked": True,
        "bad": bad,
        "num_bad_tensors": sum(
            int(item["num_bad_tensors"]) for item in summaries.values()
        ),
        "groups": summaries,
    }
    if result["num_bad_tensors"]:
        logger.error("ZO LoRA slot write produced nonfinite tensors: %s", result)
    return result


def _debug_base_weight_sentinel(model, stage: str) -> list[dict[str, Any]]:
    if (
        _debug_env(
            "VLLM_ZO_DEBUG_BASE_SENTINEL",
        )
        == "0"
    ):
        return []
    filters_env = _debug_env(
        "VLLM_ZO_DEBUG_BASE_SENTINEL_FILTERS",
        "input_layernorm,post_attention_layernorm,norm",
    )
    filters = [item.strip() for item in filters_env.split(",") if item.strip()]
    limit = int(
        _debug_env(
            "VLLM_ZO_DEBUG_BASE_SENTINEL_LIMIT",
            "4",
        )
    )
    targets_env = _debug_env(
        "VLLM_ZO_DEBUG_BASE_SENTINEL_TARGETS",
        "",
    )
    targets = [item.strip() for item in targets_env.split(",") if item.strip()]
    log_transitions = (
        _debug_env(
            "VLLM_ZO_DEBUG_BASE_SENTINEL_TRANSITIONS",
        )
        != "0"
    )
    sync_before = (
        _debug_env(
            "VLLM_ZO_DEBUG_BASE_SENTINEL_SYNC",
        )
        != "0"
    )
    if sync_before and torch.cuda.is_available():
        torch.cuda.synchronize()
    raw_model = getattr(model, "model", model)
    found: list[dict[str, Any]] = []
    for name, param in raw_model.named_parameters():
        if filters and not any(item in name for item in filters):
            continue
        if not param.is_floating_point() or param.numel() == 0:
            continue
        finite = torch.isfinite(param)
        finite_count = int(finite.sum().item())
        summary: dict[str, Any] = {
            "name": name,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "device": str(param.device),
            "numel": int(param.numel()),
            "finite_count": finite_count,
            "bad_count": int(param.numel() - finite_count),
            **_debug_tensor_storage_summary(param),
        }
        if finite_count != param.numel():
            bad_values = param[~finite]
            summary.update(
                {
                    "has_nan": bool(torch.isnan(bad_values).any().item()),
                    "has_posinf": bool(torch.isposinf(bad_values).any().item()),
                    "has_neginf": bool(torch.isneginf(bad_values).any().item()),
                }
            )
        if targets and any(item in name for item in targets) and log_transitions:
            last_states = getattr(raw_model, "_zo_base_sentinel_last_states", None)
            if last_states is None:
                last_states = {}
                setattr(raw_model, "_zo_base_sentinel_last_states", last_states)
            state = {
                "finite_count": finite_count,
                "bad_count": int(param.numel() - finite_count),
                "data_ptr": summary.get("data_ptr"),
                "storage_data_ptr": summary.get("storage_data_ptr"),
            }
            last_state = last_states.get(name)
            if last_state != state:
                log_fn = logger.error if state["bad_count"] else logger.info
                log_fn(
                    "ZO base weight sentinel transition during slot write: "
                    "stage=%s name=%s state=%s previous=%s summary=%s",
                    stage,
                    name,
                    state,
                    last_state,
                    summary,
                )
                last_states[name] = state
        if finite_count == param.numel():
            continue
        found.append(summary)
        if len(found) >= limit:
            break
    if found:
        logger.error(
            "ZO base weight sentinel nonfinite during slot write: stage=%s "
            "bad_params=%s",
            stage,
            found,
        )
    return found


def _debug_tensor_storage_summary(tensor: torch.Tensor) -> dict[str, Any]:
    element_size = int(tensor.element_size())
    numel = int(tensor.numel())
    data_ptr = int(tensor.data_ptr()) if numel else 0
    summary: dict[str, Any] = {
        "data_ptr": data_ptr,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "view_nbytes": int(numel * element_size),
        "view_end_ptr": int(data_ptr + numel * element_size),
        "stride": [int(item) for item in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
    }
    try:
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
        summary.update(
            {
                "storage_data_ptr": storage_ptr,
                "storage_nbytes": storage_nbytes,
                "storage_end_ptr": int(storage_ptr + storage_nbytes),
            }
        )
    except RuntimeError as exc:
        summary["storage_error"] = str(exc)
    return summary


def _debug_direction_slot_storage(
    entries: Sequence[dict[str, Any]],
    *,
    stage: str,
) -> None:
    if (
        _debug_env(
            "VLLM_ZO_DEBUG_LORA_STORAGE",
        )
        == "0"
    ):
        return
    targets_env = _debug_env(
        "VLLM_ZO_DEBUG_LORA_STORAGE_TARGETS",
        "",
    )
    targets = [item.strip() for item in targets_env.split(",") if item.strip()]
    limit = int(
        _debug_env(
            "VLLM_ZO_DEBUG_LORA_STORAGE_LIMIT",
            "8",
        )
    )
    rows: list[dict[str, Any]] = []
    for entry in entries:
        key = str(entry.get("direction_key", ""))
        if targets and not any(item in key for item in targets):
            continue
        rows.append(
            {
                "direction_key": key,
                "plus_a": _debug_tensor_storage_summary(entry["plus_a"]),
                "minus_a": _debug_tensor_storage_summary(entry["minus_a"]),
                "plus_b": _debug_tensor_storage_summary(entry["plus_b"]),
                "minus_b": _debug_tensor_storage_summary(entry["minus_b"]),
            }
        )
        if len(rows) >= limit:
            break
    if rows:
        logger.warning("ZO LoRA slot storage: stage=%s entries=%s", stage, rows)
