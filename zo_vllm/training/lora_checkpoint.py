"""Lightweight LoRA-bank checkpoint payload helpers."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from zo_vllm.core.perturbation_normalization import (
    attach_factorized_perturbation_spec_,
    v_energy_reference_for_direction_provider,
)


def restore_direction_provider_v_cache_from_lora_bank(
    *,
    param_metadata: Mapping[str, Any],
    accumulated_update_state: Any,
    direction_provider: Any,
    direction_provider_name: str,
    direction_scale: float,
    v_normalization: str = "none",
) -> dict[str, int]:
    restored = 0
    queued_slot: dict[str, dict[str, torch.Tensor]] = {}
    v_provider = getattr(direction_provider, "v_provider", None)
    if v_provider is None:
        v_provider = getattr(direction_provider, "v_provider", None)
    for name, block in accumulated_update_state.current_blocks.items():
        if name not in accumulated_update_state.bank_a:
            continue
        metadata = param_metadata.get(name)
        if metadata is None:
            continue
        start = int(block.start)
        end = start + int(block.rank)
        v_t = accumulated_update_state.bank_a[name][start:end, :].contiguous()
        v = v_t.T.contiguous()
        if v_provider is not None and hasattr(v_provider, "v_cache"):
            v_provider.v_cache[name] = v
            v_provider.vt_cache[name] = v_t
        direction_dtype_for = getattr(v_provider, "direction_dtype_for", None)
        if callable(direction_dtype_for):
            direction_dtype = direction_dtype_for(metadata.dtype)
        elif str(metadata.dtype).startswith("torch.float8"):
            direction_dtype = torch.float16
        else:
            direction_dtype = metadata.dtype
        direction = {
            "U": torch.empty(
                (int(metadata.shape[0]), int(v.shape[1])),
                device=metadata.device,
                dtype=direction_dtype,
            ),
            "V": v,
            "V_T": v_t,
            "scale": float(direction_scale),
            "v_refreshed": False,
        }
        attach_factorized_perturbation_spec_(
            direction,
            v_energy_reference=v_energy_reference_for_direction_provider(
                direction_provider_name=direction_provider_name,
                v_normalization=v_normalization,
            ),
        )
        queued_slot[name] = direction
        restored += 1

    subspace_queue = getattr(v_provider, "subspace_queue", None)
    if subspace_queue is not None and queued_slot:
        subspace_queue.insert(queued_slot)
    return {
        "restored_v_cache_modules": restored,
        "restored_subspace_queue": int(
            subspace_queue is not None and bool(queued_slot)
        ),
    }


def save_lora_bank_checkpoint(
    *,
    checkpoint_dir: str,
    accumulated_update_state: Any,
    step: int,
    raw_step: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Save only LoRA bank state for resumable lightweight checkpoints."""

    raw_step_i = int(step if raw_step is None else raw_step)
    flush = getattr(accumulated_update_state, "flush_pending_to_accumulated", None)
    if not callable(flush):
        raise RuntimeError("LoRA checkpoint mode requires a LoRA bank update state")
    flush_s = float(flush(step=raw_step_i))

    bank_a = getattr(accumulated_update_state, "bank_a", None)
    accumulated_u = getattr(accumulated_update_state, "accumulated_u", None)
    if not isinstance(bank_a, dict) or not isinstance(accumulated_u, dict):
        raise RuntimeError("LoRA checkpoint mode requires bank_a and accumulated_u")

    os.makedirs(checkpoint_dir, exist_ok=True)
    payload_path = os.path.join(checkpoint_dir, "zo_lora_bank.pt")
    current_blocks = getattr(accumulated_update_state, "current_blocks", {})
    used_rank = getattr(accumulated_update_state, "used_rank", {})
    pending_u = getattr(accumulated_update_state, "pending_u", {})

    payload = {
        "checkpoint_type": "vllm_zo_lora_bank",
        "step": int(step),
        "raw_step": raw_step_i,
        "dtype": str(dtype).replace("torch.", ""),
        "update_bank_rank": int(accumulated_update_state.update_bank_rank),
        "u_beta": float(accumulated_update_state.u_beta),
        "u_norm_cap": accumulated_update_state.u_norm_cap,
        "gradient_accumulation_update_steps": int(
            accumulated_update_state.gradient_accumulation_update_steps
        ),
        "bank_a": _cpu_tensor_dict(bank_a, dtype=dtype),
        "accumulated_u": _cpu_tensor_dict(accumulated_u, dtype=dtype),
        "pending_u": _cpu_tensor_dict(pending_u, dtype=dtype),
        "used_rank": {str(name): int(value) for name, value in dict(used_rank).items()},
        "current_blocks": {
            str(name): {
                "start": int(getattr(block, "start")),
                "rank": int(getattr(block, "rank")),
            }
            for name, block in dict(current_blocks).items()
        },
    }
    torch.save(payload, payload_path)
    return {
        "lora_payload_path": payload_path,
        "flush_s": flush_s,
        "folded_accumulated_update": False,
        "materialized_effective_delta": False,
        "num_bank_modules": len(payload["bank_a"]),
        "num_accumulated_modules": len(payload["accumulated_u"]),
        "max_used_rank": max(payload["used_rank"].values(), default=0),
    }


def load_lora_bank_checkpoint(
    checkpoint_path: str,
    *,
    accumulated_update_state: Any,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Load a lightweight LoRA bank checkpoint into an existing bank state."""

    payload_path = _resolve_lora_bank_payload_path(checkpoint_path)
    payload = torch.load(payload_path, map_location="cpu")
    if payload["checkpoint_type"] != "vllm_zo_lora_bank":
        raise ValueError(
            f"not a vLLM ZO LoRA bank checkpoint: {payload['checkpoint_type']!r}"
        )
    if int(payload["update_bank_rank"]) > int(
        accumulated_update_state.update_bank_rank
    ):
        raise ValueError(
            "checkpoint update_bank_rank exceeds configured capacity: "
            f"{payload['update_bank_rank']} > "
            f"{accumulated_update_state.update_bank_rank}"
        )
    _require_matching_optional(
        "u_beta",
        payload["u_beta"],
        accumulated_update_state.u_beta,
    )
    _require_matching_optional(
        "u_norm_cap",
        payload["u_norm_cap"],
        accumulated_update_state.u_norm_cap,
    )
    _require_matching_optional(
        "gradient_accumulation_update_steps",
        payload["gradient_accumulation_update_steps"],
        accumulated_update_state.gradient_accumulation_update_steps,
    )

    target_device = torch.device(device)
    accumulated_update_state.bank_a.clear()
    accumulated_update_state.accumulated_u.clear()
    accumulated_update_state.pending_u.clear()
    accumulated_update_state.zero_u.clear()
    accumulated_update_state.current_blocks.clear()
    accumulated_update_state.used_rank.clear()

    bank_a = _load_tensor_dict(
        payload["bank_a"],
        device=target_device,
        dtype=dtype,
    )
    accumulated_u = _load_tensor_dict(
        payload["accumulated_u"],
        device=target_device,
        dtype=dtype,
    )
    pending_u = _load_tensor_dict(
        payload["pending_u"],
        device=target_device,
        dtype=dtype,
    )
    accumulated_update_state.bank_a.update(bank_a)
    accumulated_update_state.accumulated_u.update(accumulated_u)
    accumulated_update_state.pending_u.update(pending_u)
    accumulated_update_state.zero_u.update(
        {name: torch.zeros_like(value) for name, value in accumulated_u.items()}
    )
    accumulated_update_state.used_rank.update(
        {str(name): int(value) for name, value in dict(payload["used_rank"]).items()}
    )
    accumulated_update_state.current_blocks.update(
        {
            str(name): SimpleNamespace(
                start=int(block["start"]),
                rank=int(block["rank"]),
            )
            for name, block in dict(payload["current_blocks"]).items()
        }
    )

    missing_bank = sorted(set(accumulated_u) - set(bank_a))
    if missing_bank:
        raise ValueError(f"checkpoint missing bank_a tensors for: {missing_bank[:5]}")
    missing_zero = sorted(set(bank_a) - set(accumulated_u))
    if missing_zero:
        raise ValueError(
            f"checkpoint missing accumulated_u tensors for: {missing_zero[:5]}"
        )
    missing_blocks = sorted(set(bank_a) - set(accumulated_update_state.current_blocks))
    if missing_blocks:
        raise ValueError(
            f"checkpoint missing current block metadata for: {missing_blocks[:5]}"
        )
    for name, block in accumulated_update_state.current_blocks.items():
        if name not in bank_a:
            raise ValueError(
                f"checkpoint current block references missing bank: {name}"
            )
        start = int(block.start)
        rank = int(block.rank)
        bank_rank = int(bank_a[name].shape[0])
        if start < 0 or rank <= 0 or start + rank > bank_rank:
            raise ValueError(
                f"checkpoint current block out of range for {name}: "
                f"start={start}, rank={rank}, bank_rank={bank_rank}"
            )
        used_rank = int(accumulated_update_state.used_rank.get(name, 0))
        if used_rank < start + rank or used_rank > bank_rank:
            raise ValueError(
                f"checkpoint used rank out of range for {name}: "
                f"used_rank={used_rank}, block_end={start + rank}, "
                f"bank_rank={bank_rank}"
            )
    return {
        "lora_payload_path": payload_path,
        "step": int(payload["step"]),
        "raw_step": int(payload["raw_step"]),
        "dtype": payload["dtype"],
        "update_bank_rank": int(payload["update_bank_rank"]),
        "num_bank_modules": len(bank_a),
        "num_accumulated_modules": len(accumulated_u),
        "num_pending_modules": len(pending_u),
        "max_used_rank": max(
            (int(value) for value in payload["used_rank"].values()),
            default=0,
        ),
    }


def _resolve_lora_bank_payload_path(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "zo_lora_bank.pt"
    if not path.exists():
        raise FileNotFoundError(f"LoRA bank checkpoint payload not found: {path}")
    return str(path)


def _require_matching_optional(name: str, saved: Any, configured: Any) -> None:
    if saved is None and configured is None:
        return
    if saved is None or configured is None:
        raise ValueError(
            f"checkpoint {name}={saved!r} does not match configured {configured!r}"
        )
    if isinstance(saved, float) or isinstance(configured, float):
        if abs(float(saved) - float(configured)) > 1e-12:
            raise ValueError(
                f"checkpoint {name}={saved!r} does not match configured {configured!r}"
            )
        return
    if int(saved) != int(configured):
        raise ValueError(
            f"checkpoint {name}={saved!r} does not match configured {configured!r}"
        )


def _load_tensor_dict(
    values: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    return {
        str(name): tensor.detach()
        .to(device=device, dtype=(dtype or tensor.dtype))
        .contiguous()
        for name, tensor in dict(values).items()
    }


def _cpu_tensor_dict(
    values: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        str(name): tensor.detach().to(device="cpu", dtype=dtype).clone()
        for name, tensor in dict(values).items()
    }
