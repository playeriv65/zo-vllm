"""Convert sampled low-rank ZO directions into LoRA runtime tensors."""

from __future__ import annotations

from typing import Mapping

import torch


LINEAR_LORA_MODULE_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "embed_tokens",
    "lm_head",
)


def is_factorized_direction(direction: Mapping[str, torch.Tensor]) -> bool:
    return "U" in direction and "V" in direction


def build_lora_runtime_tensors(
    directions: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    eps: float,
    sign: int,
    output_device: str | torch.device | None = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    layer_to_a: dict[str, torch.Tensor] = {}
    layer_to_b: dict[str, torch.Tensor] = {}
    for name, direction in directions.items():
        if not is_factorized_direction(direction):
            continue
        if not _is_lora_linear_name(name):
            continue
        u = direction["U"]
        v = direction["V"]
        scale = float(direction.get("scale", 1.0))
        lora_a = v.T.contiguous().half()
        lora_b = (int(sign) * float(eps) * scale * u).contiguous().half()
        if output_device is not None:
            lora_a = lora_a.to(output_device).contiguous()
            lora_b = lora_b.to(output_device).contiguous()
        layer_to_a[name] = lora_a
        layer_to_b[name] = lora_b
    return layer_to_a, layer_to_b


def build_lora_runtime_pair_tensors(
    directions: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    eps: float,
    output_device: str | torch.device | None = "cpu",
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    plus_a: dict[str, torch.Tensor] = {}
    plus_b: dict[str, torch.Tensor] = {}
    minus_a: dict[str, torch.Tensor] = {}
    minus_b: dict[str, torch.Tensor] = {}
    for name, direction in directions.items():
        if not is_factorized_direction(direction):
            continue
        if not _is_lora_linear_name(name):
            continue
        u = direction["U"]
        v = direction["V"]
        scale = float(direction.get("scale", 1.0))
        lora_a = v.T.contiguous().half()
        lora_b = (float(eps) * scale * u).contiguous().half()
        if output_device is not None:
            lora_a = lora_a.to(output_device).contiguous()
            lora_b = lora_b.to(output_device).contiguous()
        plus_a[name] = lora_a
        minus_a[name] = lora_a
        plus_b[name] = lora_b
        minus_b[name] = lora_b.neg()
    return plus_a, plus_b, minus_a, minus_b


def compute_projected_grad(loss_plus: float, loss_minus: float, *, eps: float) -> float:
    return (float(loss_plus) - float(loss_minus)) / (2.0 * float(eps))


def _is_lora_linear_name(name: str) -> bool:
    return any(module in name for module in LINEAR_LORA_MODULE_NAMES)
