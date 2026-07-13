"""Validation helpers for direct vLLM LoRA slot metadata."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def _adapter_slot_manager(manager):
    """Return the manager object that owns active LoRA slot tensors."""
    return getattr(manager, "_adapter_manager", manager)


def validate_direct_lora_slot_structure(
    manager,
    *,
    direction_shapes: dict[str, Sequence[int]] | None = None,
    direction_keys: Sequence[str] | None = None,
    source: str = "metadata",
) -> dict[str, Any]:
    """Validate direct-write LoRA metadata against the active vLLM LoRA slots.

    The safetensors loader rejects adapter tensors for unsupported modules before
    the adapter becomes active. Direct GPU slot writes bypass that loader, so we
    perform the equivalent structural check once at bank initialization.
    """

    slot_manager = _adapter_slot_manager(manager)
    modules = getattr(slot_manager, "modules", None)
    if not modules:
        return {
            "source": source,
            "modules_expected": 0,
            "keys_available": 0,
            "keys_consumed": 0,
            "missing_modules": [],
            "missing_direction_keys": [],
            "unexpected_direction_keys": [],
            "shape_mismatches": [],
        }

    available = set(direction_shapes or {})
    if direction_keys is not None:
        available.update(str(key) for key in direction_keys)
    consumed: set[str] = set()
    missing_modules: list[str] = []
    missing_direction_keys: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    packed_modules = getattr(slot_manager, "packed_modules", {}) or {}

    def consume_key(
        module_name: str, module, direction_key: str, slice_idx: int | None
    ):
        consumed.add(direction_key)
        if direction_shapes is None or direction_key not in direction_shapes:
            return
        expected_shape = _slot_base_weight_shape(module, slice_idx=slice_idx)
        actual_shape = tuple(int(dim) for dim in direction_shapes[direction_key])
        if expected_shape is not None and actual_shape != expected_shape:
            shape_mismatches.append(
                {
                    "module": module_name,
                    "direction_key": direction_key,
                    "expected": list(expected_shape),
                    "actual": list(actual_shape),
                }
            )

    for module_name, module in modules.items():
        if module_name in packed_modules:
            direct_key = f"{module_name}.weight"
            if direct_key in available:
                consume_key(module_name, module, direct_key, None)
                continue

            replacements = list(packed_modules[module_name])
            required_keys = [f"{replacement}.weight" for replacement in replacements]
            missing = [key for key in required_keys if key not in available]
            if missing:
                missing_modules.append(module_name)
                missing_direction_keys.extend(missing)
                continue
            for slice_idx, direction_key in enumerate(required_keys):
                consume_key(module_name, module, direction_key, slice_idx)
            continue

        direction_key = f"{module_name}.weight"
        if direction_key not in available:
            missing_modules.append(module_name)
            missing_direction_keys.append(direction_key)
            continue
        consume_key(module_name, module, direction_key, 0)

    unexpected_direction_keys = sorted(available - consumed)
    if missing_direction_keys or unexpected_direction_keys or shape_mismatches:
        parts = [
            "direct LoRA slot metadata does not match the vLLM LoRA manager",
            f"source={source}",
        ]
        if missing_direction_keys:
            parts.append(
                "missing LoRA direction metadata for manager modules: "
                + ", ".join(sorted(missing_direction_keys)[:12])
            )
        if unexpected_direction_keys:
            parts.append(
                "unexpected LoRA direction metadata not registered in manager: "
                + ", ".join(unexpected_direction_keys[:12])
            )
        if shape_mismatches:
            parts.append(f"shape mismatches: {shape_mismatches[:4]}")
        parts.append(
            "Restrict vLLM with --lora-target-modules or include matching "
            "metadata for every active LoRA module."
        )
        raise RuntimeError("; ".join(parts))

    return {
        "source": source,
        "modules_expected": len(modules),
        "keys_available": len(available),
        "keys_consumed": len(consumed),
        "missing_modules": [],
        "missing_direction_keys": [],
        "unexpected_direction_keys": [],
        "shape_mismatches": [],
    }


def _slot_base_weight_shape(module, *, slice_idx: int | None) -> tuple[int, int] | None:
    lora_a_stacked = getattr(module, "lora_a_stacked", None)
    lora_b_stacked = getattr(module, "lora_b_stacked", None)
    if lora_a_stacked is None or lora_b_stacked is None:
        return None

    if isinstance(lora_a_stacked, torch.Tensor) and isinstance(
        lora_b_stacked, torch.Tensor
    ):
        return _tensor_stacked_base_weight_shape(lora_a_stacked, lora_b_stacked)

    if slice_idx is None:
        input_dims = {int(tensor.shape[-1]) for tensor in lora_a_stacked}
        if len(input_dims) != 1:
            return None
        output_dim = sum(int(tensor.shape[-2]) for tensor in lora_b_stacked)
        return output_dim, input_dims.pop()

    if slice_idx >= len(lora_a_stacked) or slice_idx >= len(lora_b_stacked):
        return None
    input_dim = int(lora_a_stacked[slice_idx].shape[-1])
    output_dim = int(lora_b_stacked[slice_idx].shape[-2])
    return output_dim, input_dim


def _tensor_stacked_base_weight_shape(
    lora_a_stacked: torch.Tensor,
    lora_b_stacked: torch.Tensor,
) -> tuple[int, int] | None:
    if lora_a_stacked.ndim == 4:
        input_dim = int(lora_a_stacked.shape[-1])
    elif lora_a_stacked.ndim == 3:
        input_dim = int(lora_a_stacked.shape[-2])
    else:
        return None
    if lora_b_stacked.ndim < 3:
        return None
    output_dim = int(lora_b_stacked.shape[-2])
    return output_dim, input_dim
