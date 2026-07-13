"""PEFT-format LoRA tensor name and shape helpers."""

from __future__ import annotations

from typing import Dict

import torch


def peft_tensor_base_shapes(
    tensors: Dict[str, torch.Tensor],
) -> dict[str, tuple[int, int]]:
    """Recover HF-style base weight shapes from PEFT LoRA tensors."""

    partials: dict[str, dict[str, tuple[int, ...]]] = {}
    for tensor_name, tensor in tensors.items():
        module_name, kind = parse_peft_lora_tensor_name(tensor_name)
        if module_name is None or kind is None:
            continue
        partials.setdefault(module_name, {})[kind] = tuple(
            int(dim) for dim in tensor.shape
        )

    shapes: dict[str, tuple[int, int]] = {}
    for module_name, parts in partials.items():
        lora_a = parts.get("A")
        lora_b = parts.get("B")
        if lora_a is None or lora_b is None:
            continue
        shapes[f"{module_name}.weight"] = (int(lora_b[0]), int(lora_a[1]))
    return shapes


def parse_peft_lora_tensor_name(
    tensor_name: str,
) -> tuple[str | None, str | None]:
    prefix = "base_model.model."
    name = tensor_name[len(prefix) :] if tensor_name.startswith(prefix) else tensor_name
    if name.endswith(".lora_A.weight"):
        return name[: -len(".lora_A.weight")], "A"
    if name.endswith(".lora_B.weight"):
        return name[: -len(".lora_B.weight")], "B"
    if name.endswith(".lora_embedding_A"):
        return name[: -len(".lora_embedding_A")], "A"
    if name.endswith(".lora_embedding_B"):
        return name[: -len(".lora_embedding_B")], "B"
    return None, None
