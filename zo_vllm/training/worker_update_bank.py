"""Worker-resident update bank state for ZO training."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from zo_vllm.core.lora_runtime import (
    _update_lora_slots_from_directions_in_vllm_model,
    validate_direct_lora_slot_structure,
)
from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.config import VLLMZOConfig
from zo_vllm.training.direction import (
    DirectionSample,
    LOZODirectionProvider,
    TokenProbeBatch,
)
from zo_vllm.training.update_bank_state import BlockLoRAUpdateBankState

SOURCE_NAME = "worker_update_bank"


@dataclass
class _WorkerUpdateBankState:
    direction_provider: LOZODirectionProvider
    bank_state: BlockLoRAUpdateBankState
    plus_id: int
    minus_id: int
    structure_info: dict[str, Any]
    last_step: int = 0
    last_sample: DirectionSample | None = None


def worker_update_bank_key(*, plus_id: int, minus_id: int) -> str:
    return f"_zo_worker_update_bank_{int(plus_id)}_{int(minus_id)}"


def init_worker_update_bank(
    model: Any,
    *,
    state_key: str,
    metadata_specs: list[dict[str, Any]],
    lozo_config: dict[str, Any],
    bank_config: dict[str, Any],
    plus_id: int,
    minus_id: int,
) -> dict[str, Any]:
    """Initialize worker-local ZO direction and update bank state."""

    manager = getattr(model, "lora_manager", None)
    if manager is None:
        raise RuntimeError(
            "vLLM model has no lora_manager; enable_lora=True is required"
        )
    device = torch.device(getattr(manager, "device", "cuda"))
    dtype = _torch_dtype(str(lozo_config.get("direction_dtype", "float16")))
    metadata = _metadata_from_specs(metadata_specs, device=device, dtype=dtype)
    structure_info = validate_direct_lora_slot_structure(
        manager,
        direction_shapes={name: meta.shape for name, meta in metadata.items()},
        source="worker_update_bank_metadata",
    )

    zo_config = VLLMZOConfig(
        rank=int(lozo_config["rank"]),
        eps=float(lozo_config["eps"]),
        nu=int(lozo_config["nu"]),
        seed=int(lozo_config.get("seed", 42)),
        random_device=str(lozo_config.get("random_device", "cuda")),
        direction_sampling=str(lozo_config.get("direction_sampling", "exact")),
        direction_scale=lozo_config.get("direction_scale"),
        perturbation_normalization=str(
            lozo_config.get("perturbation_normalization", "rms")
        ),
        v_normalization=str(lozo_config.get("v_normalization", "none")),
    )
    state = _WorkerUpdateBankState(
        direction_provider=LOZODirectionProvider(
            param_metadata=metadata,
            rank=zo_config.rank,
            nu=zo_config.nu,
            random_device=zo_config.random_device,
            direction_sampling=zo_config.direction_sampling,
            direction_scale=zo_config.direction_scale,
            perturbation_normalization=zo_config.perturbation_normalization,
            v_normalization=zo_config.v_normalization,
            seed=zo_config.seed,
        ),
        bank_state=BlockLoRAUpdateBankState(
            update_bank_rank=int(bank_config["update_bank_rank"]),
            u_beta=float(bank_config.get("u_beta", 1.0)),
            u_momentum=float(bank_config.get("u_momentum", 0.0)),
            u_norm_cap=bank_config.get("u_norm_cap"),
            gradient_accumulation_update_steps=int(
                bank_config.get("gradient_accumulation_update_steps", 0)
            ),
        ),
        plus_id=int(plus_id),
        minus_id=int(minus_id),
        structure_info=structure_info,
    )
    setattr(model, state_key, state)
    return {
        "state_key": state_key,
        "num_modules": len(metadata),
        "device": str(device),
        "direction_dtype": str(dtype).replace("torch.", ""),
        "source": SOURCE_NAME,
        "structure_info": structure_info,
    }


def prepare_worker_update_bank_slots(
    model: Any,
    *,
    state_key: str,
    step: int,
    eps: float,
    copy_stream: str = "default",
) -> dict[str, Any]:
    """Sample the next direction, update bank slots, and keep state in worker."""

    state = _worker_state(model, state_key)
    step_i = int(step)
    refresh_fold_s = 0.0
    if state.direction_provider.will_refresh(step=step_i):
        refresh_fold_s = state.bank_state.fold_before_direction_refresh(step=step_i)

    direction_t0 = time.perf_counter()
    sample = state.direction_provider.next(
        TokenProbeBatch(token_id_groups=()),
        step=step_i,
    )
    direction_s = time.perf_counter() - direction_t0

    prepare_t0 = time.perf_counter()
    state.bank_state.debug_step = step_i
    prepared = state.bank_state.prepare_for_score(sample.directions)
    prepare_s = time.perf_counter() - prepare_t0

    slot_t0 = time.perf_counter()
    slot_info = _update_lora_slots_from_directions_in_vllm_model(
        model,
        plus_id=state.plus_id,
        minus_id=state.minus_id,
        directions_2d=prepared,
        eps=float(eps),
        copy_stream=copy_stream,
    )
    slot_write_s = time.perf_counter() - slot_t0

    state.last_step = step_i
    state.last_sample = sample
    return {
        "source": SOURCE_NAME,
        "direction_refreshed": bool(sample.refreshed),
        "direction_info": sample.info,
        "refresh_fold_s": refresh_fold_s,
        "direction_s": direction_s,
        "bank_prepare_s": prepare_s,
        "worker_slot_write_s": slot_write_s,
        "slot_info": slot_info,
    }


def apply_worker_update_bank_update(
    model: Any,
    *,
    state_key: str,
    step: int,
    projected_grad: float,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Apply one ZO projected update to the worker-resident update bank."""

    state = _worker_state(model, state_key)
    step_i = int(step)
    if state.last_step != step_i or state.last_sample is None:
        raise RuntimeError(
            "worker update bank update called before matching prepare call: "
            f"last_step={state.last_step}, step={step_i}"
        )
    t0 = time.perf_counter()
    state.bank_state.debug_step = step_i
    update_info = state.bank_state.apply(
        state.last_sample.directions,
        projected_grad=float(projected_grad),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        step=step_i,
    )
    return {
        "source": SOURCE_NAME,
        "apply_s": time.perf_counter() - t0,
        "update_info": update_info,
    }


def write_clean_worker_update_bank(
    model: Any,
    *,
    state_key: str,
    step: int,
    copy_stream: str = "default",
) -> dict[str, Any]:
    """Write the clean accumulated bank to the plus/minus slots."""

    del step
    state = _worker_state(model, state_key)
    clean = state.bank_state.clean_directions_for_score()
    if not clean:
        return {
            "source": SOURCE_NAME,
            "has_update": False,
            "slot_info": None,
            "worker_slot_write_s": 0.0,
        }
    t0 = time.perf_counter()
    slot_info = _update_lora_slots_from_directions_in_vllm_model(
        model,
        plus_id=state.plus_id,
        minus_id=state.minus_id,
        directions_2d=clean,
        eps=0.0,
        copy_stream=copy_stream,
    )
    return {
        "source": SOURCE_NAME,
        "has_update": True,
        "slot_info": slot_info,
        "worker_slot_write_s": time.perf_counter() - t0,
    }


def _worker_state(model: Any, state_key: str) -> _WorkerUpdateBankState:
    state = getattr(model, state_key, None)
    if state is None:
        raise RuntimeError(f"worker update bank is not initialized: {state_key}")
    return state


def _metadata_from_specs(
    specs: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, ParamMetadata]:
    metadata: dict[str, ParamMetadata] = {}
    for spec in specs:
        name = str(spec["name"])
        shape = tuple(int(dim) for dim in spec["shape"])
        metadata[name] = ParamMetadata(
            name=name,
            shape=shape,
            dtype=dtype,
            device=device,
        )
    return metadata


def _torch_dtype(value: str) -> torch.dtype:
    normalized = value.lower().replace("torch.", "")
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported direction dtype: {value}")
