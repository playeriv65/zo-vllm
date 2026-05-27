"""
Temporary LoRA Runtime - Manages plus/minus LoRA slots for LOZO perturbation.

Uses either the in-memory CPU LoRA interface from zo_vllm.core.memory_lora_loader.py,
the GPU-resident manager path, or the training-only direct slot updater that
writes fixed vLLM LoRA slots in place.
Supports all 2D trainable parameters (Linear layers).
"""

from functools import partial
import time
from typing import Any, Dict, List
import torch

from .memory_lora_loader import register_memory_lora_cpu, unregister_memory_lora


GPU_MEMORY_PATH_PREFIX = "/gpu_lora_loaded"


def _load_lora_tensors_into_vllm_model(
    model,
    *,
    lora_id: int,
    config: dict,
    tensors: Dict[str, torch.Tensor],
) -> dict:
    """Load PEFT-format LoRA tensors into vLLM's GPU LoRA slots."""
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    manager = getattr(model, "lora_manager", None)
    if manager is None:
        raise RuntimeError("vLLM model has no lora_manager; enable_lora=True is required")

    peft_config = dict(config)
    peft_config.setdefault("vllm_max_position_embeddings", None)
    peft_helper = PEFTHelper.from_dict(peft_config)
    peft_helper.validate_legal(manager.lora_config)

    hf_to_vllm_mapper = getattr(manager.model, "hf_to_vllm_mapper", None)
    lora_skip_prefixes = getattr(manager.model, "lora_skip_prefixes", None)
    lora = LoRAModel.from_lora_tensors(
        lora_model_id=lora_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device=str(manager.device),
        dtype=manager.lora_config.lora_dtype,
        model_vocab_size=manager.vocab_size,
        weights_mapper=hf_to_vllm_mapper,
        skip_prefixes=lora_skip_prefixes,
    )

    manager.remove_adapter(lora.id)
    loaded = manager.add_adapter(lora)
    activated = manager.activate_adapter(lora.id)
    return {
        "lora_id": int(lora.id),
        "loaded": bool(loaded),
        "activated": bool(activated),
        "device": str(manager.device),
        "num_tensors": len(tensors),
    }


def _load_two_lora_tensors_into_vllm_model(
    model,
    *,
    plus_id: int,
    minus_id: int,
    config: dict,
    plus_tensors: Dict[str, torch.Tensor],
    minus_tensors: Dict[str, torch.Tensor],
) -> dict:
    """Load and activate both training LoRA slots in one model callback."""
    plus = _load_lora_tensors_into_vllm_model(
        model,
        lora_id=plus_id,
        config=config,
        tensors=plus_tensors,
    )
    minus = _load_lora_tensors_into_vllm_model(
        model,
        lora_id=minus_id,
        config=dict(config),
        tensors=minus_tensors,
    )
    return {"plus": plus, "minus": minus}


def _remove_lora_from_vllm_model(model, *, lora_id: int) -> bool:
    manager = getattr(model, "lora_manager", None)
    if manager is None:
        return False
    return bool(manager.remove_adapter(lora_id))


def _lora_slot_index(manager, lora_id: int) -> int:
    try:
        return manager.lora_index_to_id.index(lora_id)
    except ValueError as exc:
        raise RuntimeError(f"LoRA id {lora_id} is not active in a GPU slot") from exc


def _read_lora_pair(
    layer_to_A: Dict[str, torch.Tensor],
    layer_to_B: Dict[str, torch.Tensor],
    key: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    a = layer_to_A.get(key)
    b = layer_to_B.get(key)
    if (a is None) != (b is None):
        raise RuntimeError(f"incomplete LoRA pair for {key}")
    return a, b


def _normalize_lora_slices(
    module,
    lora_a: torch.Tensor | list[torch.Tensor | None],
    lora_b: torch.Tensor | list[torch.Tensor | None],
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    """Normalize LoRA tensors to one A/B pair per vLLM slice."""
    n_slices = int(getattr(module, "n_slices", len(module.lora_a_stacked)))

    if isinstance(lora_b, list) and len(lora_b) != n_slices:
        expand = getattr(module, "expand_packed_lora", None)
        if expand is not None:
            lora_a, lora_b = expand(lora_a, lora_b)

    if isinstance(lora_a, torch.Tensor):
        lora_a = [lora_a] * n_slices
    if isinstance(lora_b, torch.Tensor):
        if n_slices == 1:
            lora_b = [lora_b]
        else:
            output_sizes = getattr(module, "output_sizes", None)
            if output_sizes is None:
                output_sizes = getattr(module.base_layer, "output_sizes")
            start = 0
            split_b = []
            for output_size in output_sizes:
                end = start + output_size
                split_b.append(lora_b[start:end, :])
                start = end
            lora_b = split_b

    if int(getattr(module, "tp_size", 1)) > 1:
        lora_a = module.slice_lora_a(lora_a)
        lora_b = module.slice_lora_b(lora_b)

    return list(lora_a), list(lora_b)


def _set_lora_noreset_if_full(
    module,
    index: int,
    lora_a: torch.Tensor | list[torch.Tensor | None],
    lora_b: torch.Tensor | list[torch.Tensor | None],
) -> bool:
    """
    Overwrite a slot without zeroing it first when every stored slice is full.

    This is a training-only fast path for fixed-rank temporary adapters. If a
    tensor is partial or missing, the caller falls back to vLLM's set_lora().
    """
    lora_a_slices, lora_b_slices = _normalize_lora_slices(module, lora_a, lora_b)
    if not (
        len(lora_a_slices)
        == len(lora_b_slices)
        == len(module.lora_a_stacked)
        == len(module.lora_b_stacked)
    ):
        return False

    for slice_idx, (lora_a_i, lora_b_i) in enumerate(zip(lora_a_slices, lora_b_slices)):
        if lora_a_i is None or lora_b_i is None:
            return False
        target_a = module.lora_a_stacked[slice_idx][index, 0]
        target_b = module.lora_b_stacked[slice_idx][index, 0]
        if tuple(lora_a_i.shape) != tuple(target_a.shape):
            return False
        if tuple(lora_b_i.shape) != tuple(target_b.shape):
            return False

    for slice_idx, (lora_a_i, lora_b_i) in enumerate(zip(lora_a_slices, lora_b_slices)):
        module.lora_a_stacked[slice_idx][index, 0].copy_(lora_a_i, non_blocking=True)
        module.lora_b_stacked[slice_idx][index, 0].copy_(lora_b_i, non_blocking=True)
    return True


def _copy_direction_to_slot(
    module,
    *,
    index: int,
    slice_idx: int,
    U: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    V_T: torch.Tensor | None = None,
    write_a: bool = True,
) -> bool:
    target_a = module.lora_a_stacked[slice_idx][index, 0]
    target_b = module.lora_b_stacked[slice_idx][index, 0]
    lora_a = V_T if V_T is not None else V.T
    if tuple(lora_a.shape) != tuple(target_a.shape):
        return False
    if tuple(U.shape) != tuple(target_b.shape):
        return False
    if write_a:
        target_a.copy_(lora_a, non_blocking=True)
    torch.mul(U, scale, out=target_b)
    return True


def _copy_lora_b_to_slot(
    module,
    *,
    index: int,
    slice_idx: int,
    B: torch.Tensor,
    V: torch.Tensor,
    V_T: torch.Tensor | None = None,
    write_a: bool = True,
) -> bool:
    target_a = module.lora_a_stacked[slice_idx][index, 0]
    target_b = module.lora_b_stacked[slice_idx][index, 0]
    lora_a = V_T if V_T is not None else V.T
    if tuple(lora_a.shape) != tuple(target_a.shape):
        return False
    if tuple(B.shape) != tuple(target_b.shape):
        return False
    if write_a:
        target_a.copy_(lora_a, non_blocking=True)
    target_b.copy_(B, non_blocking=True)
    return True


def _copy_accumulated_direction_to_slot(
    module,
    *,
    index: int,
    slice_idx: int,
    U: torch.Tensor,
    U_accum: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    V_T: torch.Tensor | None = None,
    write_a: bool = True,
) -> bool:
    target_a = module.lora_a_stacked[slice_idx][index, 0]
    target_b = module.lora_b_stacked[slice_idx][index, 0]
    lora_a = V_T if V_T is not None else V.T
    if tuple(lora_a.shape) != tuple(target_a.shape):
        return False
    if tuple(U.shape) != tuple(target_b.shape):
        return False
    if tuple(U_accum.shape) != tuple(target_b.shape):
        return False
    if write_a:
        target_a.copy_(lora_a, non_blocking=True)
    target_b.copy_(U_accum, non_blocking=True)
    target_b.add_(U, alpha=scale)
    return True


def _write_direction_to_plus_minus(
    module,
    *,
    plus_index: int,
    minus_index: int,
    slice_idx: int,
    direction: dict[str, torch.Tensor],
    eps: float,
) -> bool:
    U = direction["U"]
    V = direction["V"]
    V_T = direction.get("V_T")
    write_a = bool(direction.get("v_refreshed", True))
    U_accum = direction.get("U_accum")
    if U_accum is not None:
        return _copy_accumulated_direction_to_slot(
            module,
            index=plus_index,
            slice_idx=slice_idx,
            U=U,
            U_accum=U_accum,
            V=V,
            scale=eps,
            V_T=V_T,
            write_a=write_a,
        ) and _copy_accumulated_direction_to_slot(
            module,
            index=minus_index,
            slice_idx=slice_idx,
            U=U,
            U_accum=U_accum,
            V=V,
            scale=-eps,
            V_T=V_T,
            write_a=write_a,
        )
    plus_b = direction.get("lora_B_plus")
    minus_b = direction.get("lora_B_minus")
    if plus_b is not None or minus_b is not None:
        if plus_b is None or minus_b is None:
            return False
        return _copy_lora_b_to_slot(
            module,
            index=plus_index,
            slice_idx=slice_idx,
            B=plus_b,
            V=V,
            V_T=V_T,
            write_a=write_a,
        ) and _copy_lora_b_to_slot(
            module,
            index=minus_index,
            slice_idx=slice_idx,
            B=minus_b,
            V=V,
            V_T=V_T,
            write_a=write_a,
        )
    return _copy_direction_to_slot(
        module,
        index=plus_index,
        slice_idx=slice_idx,
        U=U,
        V=V,
        scale=eps,
        V_T=V_T,
        write_a=write_a,
    ) and _copy_direction_to_slot(
        module,
        index=minus_index,
        slice_idx=slice_idx,
        U=U,
        V=V,
        scale=-eps,
        V_T=V_T,
        write_a=write_a,
    )


def _queue_direction_to_plus_minus(
    module,
    *,
    plus_index: int,
    minus_index: int,
    slice_idx: int,
    direction: dict[str, torch.Tensor],
    plus_a_targets: list[torch.Tensor],
    minus_a_targets: list[torch.Tensor],
    a_sources: list[torch.Tensor],
    plus_b_targets: list[torch.Tensor],
    minus_b_targets: list[torch.Tensor],
    u_sources: list[torch.Tensor],
    custom_plus_b_targets: list[torch.Tensor],
    custom_minus_b_targets: list[torch.Tensor],
    plus_b_sources: list[torch.Tensor],
    minus_b_sources: list[torch.Tensor],
    accum_plus_b_targets: list[torch.Tensor],
    accum_minus_b_targets: list[torch.Tensor],
    accum_sources: list[torch.Tensor],
    accum_u_sources: list[torch.Tensor],
) -> bool:
    U = direction["U"]
    V = direction["V"]
    V_T = direction.get("V_T")
    lora_a = V_T if V_T is not None else V.T

    plus_a = module.lora_a_stacked[slice_idx][plus_index, 0]
    minus_a = module.lora_a_stacked[slice_idx][minus_index, 0]
    plus_b = module.lora_b_stacked[slice_idx][plus_index, 0]
    minus_b = module.lora_b_stacked[slice_idx][minus_index, 0]

    if tuple(lora_a.shape) != tuple(plus_a.shape):
        return False
    if tuple(lora_a.shape) != tuple(minus_a.shape):
        return False
    if tuple(U.shape) != tuple(plus_b.shape):
        return False
    if tuple(U.shape) != tuple(minus_b.shape):
        return False

    if bool(direction.get("v_refreshed", True)):
        plus_a_targets.append(plus_a)
        minus_a_targets.append(minus_a)
        a_sources.append(lora_a)
    custom_plus_b = direction.get("lora_B_plus")
    custom_minus_b = direction.get("lora_B_minus")
    U_accum = direction.get("U_accum")
    if U_accum is not None:
        accum_plus_b_targets.append(plus_b)
        accum_minus_b_targets.append(minus_b)
        accum_sources.append(U_accum)
        accum_u_sources.append(U)
    elif custom_plus_b is not None or custom_minus_b is not None:
        if custom_plus_b is None or custom_minus_b is None:
            return False
        custom_plus_b_targets.append(plus_b)
        custom_minus_b_targets.append(minus_b)
        plus_b_sources.append(custom_plus_b)
        minus_b_sources.append(custom_minus_b)
    else:
        plus_b_targets.append(plus_b)
        minus_b_targets.append(minus_b)
        u_sources.append(U)
    return True


def _direction_slot_entry(
    module,
    *,
    plus_index: int,
    minus_index: int,
    slice_idx: int,
    direction_key: str,
    direction: dict[str, torch.Tensor],
) -> dict[str, Any] | None:
    U = direction["U"]
    V = direction["V"]
    V_T = direction.get("V_T")
    lora_a = V_T if V_T is not None else V.T

    plus_a = module.lora_a_stacked[slice_idx][plus_index, 0]
    minus_a = module.lora_a_stacked[slice_idx][minus_index, 0]
    plus_b = module.lora_b_stacked[slice_idx][plus_index, 0]
    minus_b = module.lora_b_stacked[slice_idx][minus_index, 0]

    if tuple(lora_a.shape) != tuple(plus_a.shape):
        return None
    if tuple(lora_a.shape) != tuple(minus_a.shape):
        return None
    if tuple(U.shape) != tuple(plus_b.shape):
        return None
    if tuple(U.shape) != tuple(minus_b.shape):
        return None
    return {
        "direction_key": direction_key,
        "plus_a": plus_a,
        "minus_a": minus_a,
        "plus_b": plus_b,
        "minus_b": minus_b,
    }


def _build_direction_slot_plan(
    manager,
    *,
    plus_index: int,
    minus_index: int,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    modules_written = 0
    packed_written = 0
    missing: list[str] = []
    fallback_required: list[str] = []

    for module_name, module in manager.modules.items():
        if module_name in manager.packed_modules:
            replacements = list(manager.packed_modules[module_name])
            if len(replacements) != len(module.lora_a_stacked):
                fallback_required.append(module_name)
                continue
            wrote_any = False
            ok = True
            for slice_idx, replacement in enumerate(replacements):
                direction_key = f"{replacement}.weight"
                direction = directions_2d.get(direction_key)
                if direction is None:
                    ok = False
                    break
                wrote_any = True
                entry = _direction_slot_entry(
                    module,
                    plus_index=plus_index,
                    minus_index=minus_index,
                    slice_idx=slice_idx,
                    direction_key=direction_key,
                    direction=direction,
                )
                if entry is None:
                    ok = False
                    break
                entries.append(entry)
            if ok and wrote_any:
                packed_written += 1
            elif wrote_any:
                fallback_required.append(module_name)
            else:
                module.reset_lora(plus_index)
                module.reset_lora(minus_index)
                missing.append(module_name)
            continue

        direction_key = f"{module_name}.weight"
        direction = directions_2d.get(direction_key)
        if direction is None:
            module.reset_lora(plus_index)
            module.reset_lora(minus_index)
            missing.append(module_name)
            continue
        if len(module.lora_a_stacked) != 1:
            fallback_required.append(module_name)
            continue
        entry = _direction_slot_entry(
            module,
            plus_index=plus_index,
            minus_index=minus_index,
            slice_idx=0,
            direction_key=direction_key,
            direction=direction,
        )
        if entry is None:
            fallback_required.append(module_name)
        else:
            entries.append(entry)
            modules_written += 1

    if fallback_required:
        raise RuntimeError(
            "direct direction slot update does not support modules: "
            + ", ".join(fallback_required[:8])
        )
    return {
        "entries": entries,
        "modules_written": int(modules_written),
        "packed_written": int(packed_written),
        "missing": missing,
    }


def _flush_queued_direction_writes(
    *,
    eps: float,
    plus_a_targets: list[torch.Tensor],
    minus_a_targets: list[torch.Tensor],
    a_sources: list[torch.Tensor],
    plus_b_targets: list[torch.Tensor],
    minus_b_targets: list[torch.Tensor],
    u_sources: list[torch.Tensor],
    custom_plus_b_targets: list[torch.Tensor],
    custom_minus_b_targets: list[torch.Tensor],
    plus_b_sources: list[torch.Tensor],
    minus_b_sources: list[torch.Tensor],
    accum_plus_b_targets: list[torch.Tensor],
    accum_minus_b_targets: list[torch.Tensor],
    accum_sources: list[torch.Tensor],
    accum_u_sources: list[torch.Tensor],
) -> None:
    if a_sources:
        torch._foreach_copy_(plus_a_targets, a_sources)
        torch._foreach_copy_(minus_a_targets, a_sources)
    if u_sources:
        torch._foreach_copy_(plus_b_targets, u_sources)
        torch._foreach_mul_(plus_b_targets, eps)
        torch._foreach_copy_(minus_b_targets, u_sources)
        torch._foreach_mul_(minus_b_targets, -eps)
    if plus_b_sources:
        torch._foreach_copy_(custom_plus_b_targets, plus_b_sources)
        torch._foreach_copy_(custom_minus_b_targets, minus_b_sources)
    if accum_sources:
        torch._foreach_copy_(accum_plus_b_targets, accum_sources)
        torch._foreach_add_(accum_plus_b_targets, accum_u_sources, alpha=eps)
        torch._foreach_copy_(accum_minus_b_targets, accum_sources)
        torch._foreach_add_(accum_minus_b_targets, accum_u_sources, alpha=-eps)


def _write_plus_minus_slots_from_directions(
    manager,
    *,
    plus_id: int,
    minus_id: int,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
    eps: float,
) -> dict:
    """Write plus/minus slots directly from LOZO U/V directions."""
    profile_t0 = time.perf_counter()
    plus_index = _lora_slot_index(manager, plus_id)
    minus_index = _lora_slot_index(manager, minus_id)
    profile_after_index = time.perf_counter()
    plus_a_targets: list[torch.Tensor] = []
    minus_a_targets: list[torch.Tensor] = []
    a_sources: list[torch.Tensor] = []
    plus_b_targets: list[torch.Tensor] = []
    minus_b_targets: list[torch.Tensor] = []
    u_sources: list[torch.Tensor] = []
    custom_plus_b_targets: list[torch.Tensor] = []
    custom_minus_b_targets: list[torch.Tensor] = []
    plus_b_sources: list[torch.Tensor] = []
    minus_b_sources: list[torch.Tensor] = []
    accum_plus_b_targets: list[torch.Tensor] = []
    accum_minus_b_targets: list[torch.Tensor] = []
    accum_sources: list[torch.Tensor] = []
    accum_u_sources: list[torch.Tensor] = []

    cache_key = (int(plus_index), int(minus_index), int(len(directions_2d)))
    cache = getattr(manager, "_zo_direction_slot_plan_cache", None)
    if cache is None:
        cache = {}
        setattr(manager, "_zo_direction_slot_plan_cache", cache)
    plan = cache.get(cache_key)
    plan_cache_hit = plan is not None
    if plan is None:
        plan = _build_direction_slot_plan(
            manager,
            plus_index=plus_index,
            minus_index=minus_index,
            directions_2d=directions_2d,
        )
        cache[cache_key] = plan

    missing_direction_keys: list[str] = []
    for entry in plan["entries"]:
        direction = directions_2d.get(entry["direction_key"])
        if direction is None:
            missing_direction_keys.append(entry["direction_key"])
            continue
        U = direction["U"]
        V = direction["V"]
        V_T = direction.get("V_T")
        lora_a = V_T if V_T is not None else V.T
        if bool(direction.get("v_refreshed", True)):
            plus_a_targets.append(entry["plus_a"])
            minus_a_targets.append(entry["minus_a"])
            a_sources.append(lora_a)
        custom_plus_b = direction.get("lora_B_plus")
        custom_minus_b = direction.get("lora_B_minus")
        U_accum = direction.get("U_accum")
        if U_accum is not None:
            accum_plus_b_targets.append(entry["plus_b"])
            accum_minus_b_targets.append(entry["minus_b"])
            accum_sources.append(U_accum)
            accum_u_sources.append(U)
        elif custom_plus_b is not None or custom_minus_b is not None:
            if custom_plus_b is None or custom_minus_b is None:
                cache.pop(cache_key, None)
                raise RuntimeError("direction provides only one custom LoRA B tensor")
            custom_plus_b_targets.append(entry["plus_b"])
            custom_minus_b_targets.append(entry["minus_b"])
            plus_b_sources.append(custom_plus_b)
            minus_b_sources.append(custom_minus_b)
        else:
            plus_b_targets.append(entry["plus_b"])
            minus_b_targets.append(entry["minus_b"])
            u_sources.append(U)

    if missing_direction_keys:
        cache.pop(cache_key, None)
        raise RuntimeError(
            "cached direct direction slot plan is stale; missing directions: "
            + ", ".join(missing_direction_keys[:8])
        )
    profile_after_traversal = time.perf_counter()
    _flush_queued_direction_writes(
        eps=eps,
        plus_a_targets=plus_a_targets,
        minus_a_targets=minus_a_targets,
        a_sources=a_sources,
        plus_b_targets=plus_b_targets,
        minus_b_targets=minus_b_targets,
        u_sources=u_sources,
        custom_plus_b_targets=custom_plus_b_targets,
        custom_minus_b_targets=custom_minus_b_targets,
        plus_b_sources=plus_b_sources,
        minus_b_sources=minus_b_sources,
        accum_plus_b_targets=accum_plus_b_targets,
        accum_minus_b_targets=accum_minus_b_targets,
        accum_sources=accum_sources,
        accum_u_sources=accum_u_sources,
    )
    profile_after_flush = time.perf_counter()

    return {
        "plus": {"lora_id": int(plus_id), "slot_index": int(plus_index)},
        "minus": {"lora_id": int(minus_id), "slot_index": int(minus_index)},
        "modules_written": int(plan["modules_written"]),
        "packed_written": int(plan["packed_written"]),
        "missing": list(plan["missing"]),
        "source": "directions",
        "profile_s": {
            "index": profile_after_index - profile_t0,
            "traversal": profile_after_traversal - profile_after_index,
            "flush": profile_after_flush - profile_after_traversal,
            "total_worker": profile_after_flush - profile_t0,
            "plan_cache_hit": float(plan_cache_hit),
        },
    }


def _write_plus_minus_slots(
    manager,
    *,
    plus_id: int,
    minus_id: int,
    plus_A: Dict[str, torch.Tensor],
    plus_B: Dict[str, torch.Tensor],
    minus_A: Dict[str, torch.Tensor],
    minus_B: Dict[str, torch.Tensor],
) -> dict:
    """Write both training adapters while walking vLLM LoRA modules once."""
    plus_index = _lora_slot_index(manager, plus_id)
    minus_index = _lora_slot_index(manager, minus_id)
    modules_written = 0
    packed_written = 0
    missing: list[str] = []

    for module_name, module in manager.modules.items():
        if module_name in manager.packed_modules:
            plus_lora_a = []
            plus_lora_b = []
            minus_lora_a = []
            minus_lora_b = []
            plus_has_any = False
            minus_has_any = False

            for replacement in manager.packed_modules[module_name]:
                key = f"{replacement}.weight"
                plus_a, plus_b = _read_lora_pair(plus_A, plus_B, key)
                minus_a, minus_b = _read_lora_pair(minus_A, minus_B, key)
                plus_lora_a.append(plus_a)
                plus_lora_b.append(plus_b)
                minus_lora_a.append(minus_a)
                minus_lora_b.append(minus_b)
                plus_has_any = plus_has_any or plus_a is not None
                minus_has_any = minus_has_any or minus_a is not None

            if plus_has_any:
                if not _set_lora_noreset_if_full(
                    module, plus_index, plus_lora_a, plus_lora_b
                ):
                    module.set_lora(plus_index, plus_lora_a, plus_lora_b)
            else:
                module.reset_lora(plus_index)
            if minus_has_any:
                if not _set_lora_noreset_if_full(
                    module, minus_index, minus_lora_a, minus_lora_b
                ):
                    module.set_lora(minus_index, minus_lora_a, minus_lora_b)
            else:
                module.reset_lora(minus_index)

            if plus_has_any or minus_has_any:
                packed_written += 1
            else:
                missing.append(module_name)
            continue

        key = f"{module_name}.weight"
        plus_a, plus_b = _read_lora_pair(plus_A, plus_B, key)
        minus_a, minus_b = _read_lora_pair(minus_A, minus_B, key)

        if plus_a is None:
            module.reset_lora(plus_index)
        else:
            if not _set_lora_noreset_if_full(module, plus_index, plus_a, plus_b):
                module.set_lora(plus_index, plus_a, plus_b)

        if minus_a is None:
            module.reset_lora(minus_index)
        else:
            if not _set_lora_noreset_if_full(module, minus_index, minus_a, minus_b):
                module.set_lora(minus_index, minus_a, minus_b)

        if plus_a is None and minus_a is None:
            missing.append(module_name)
        else:
            modules_written += 1

    return {
        "plus": {"lora_id": int(plus_id), "slot_index": int(plus_index)},
        "minus": {"lora_id": int(minus_id), "slot_index": int(minus_index)},
        "modules_written": int(modules_written),
        "packed_written": int(packed_written),
        "missing": missing,
    }


def _update_lora_slots_in_vllm_model(
    model,
    *,
    plus_id: int,
    minus_id: int,
    plus_A: Dict[str, torch.Tensor],
    plus_B: Dict[str, torch.Tensor],
    minus_A: Dict[str, torch.Tensor],
    minus_B: Dict[str, torch.Tensor],
) -> dict:
    """Training-only fast path: overwrite fixed plus/minus LoRA slots in place."""
    manager = getattr(model, "lora_manager", None)
    if manager is None:
        raise RuntimeError("vLLM model has no lora_manager; enable_lora=True is required")

    return _write_plus_minus_slots(
        manager,
        plus_id=plus_id,
        minus_id=minus_id,
        plus_A=plus_A,
        plus_B=plus_B,
        minus_A=minus_A,
        minus_B=minus_B,
    )


def _update_lora_slots_from_directions_in_vllm_model(
    model,
    *,
    plus_id: int,
    minus_id: int,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
    eps: float,
) -> dict:
    """Training-only fast path: build and write fixed slots inside vLLM."""
    manager = getattr(model, "lora_manager", None)
    if manager is None:
        raise RuntimeError("vLLM model has no lora_manager; enable_lora=True is required")

    return _write_plus_minus_slots_from_directions(
        manager,
        plus_id=plus_id,
        minus_id=minus_id,
        directions_2d=directions_2d,
        eps=eps,
    )


class TempLoRARuntime:
    """
    Manages two in-memory LoRA slots for LOZO plus/minus perturbation.
    
    Slots:
    - plus_id: Perturbation +eps*U @ V^T
    - minus_id: Perturbation -eps*U @ V^T
    
    Supports all 2D trainable parameters (Linear layers).
    """
    
    def __init__(
        self,
        rank: int,
        num_layers: int = 32,
        plus_id: int = 9001,
        minus_id: int = 9002,
        stable_lora_ids: bool = True,
        residency: str = "cpu",
        injection: str = "auto",
        llm: Any | None = None,
        base_model_name: str = "facebook/opt-2.7b",
        hidden_size: int = 2560,
        ffn_dim: int = 10240,
    ):
        if residency not in {"cpu", "gpu"}:
            raise ValueError(f"unknown LoRA residency: {residency}")
        if injection == "auto":
            injection = "direct" if residency == "gpu" else "manager"
        if injection not in {"direct", "manager"}:
            raise ValueError(f"unknown LoRA injection mode: {injection}")
        if residency == "cpu" and injection != "manager":
            raise ValueError("direct LoRA injection requires residency='gpu'")
        if injection == "direct" and not stable_lora_ids:
            raise ValueError("direct LoRA injection requires stable LoRA IDs")
        self.rank = rank
        self.num_layers = num_layers
        self.plus_id = plus_id
        self.minus_id = minus_id
        self._base_plus_id = plus_id
        self._base_minus_id = minus_id
        self.stable_lora_ids = stable_lora_ids
        self.residency = residency
        self.injection = injection
        self.llm = llm
        self.base_model_name = base_model_name
        self.hidden_size = int(hidden_size)
        self.ffn_dim = int(ffn_dim)
        self.plus_name = "lozo_plus"
        self.minus_name = "lozo_minus"
        self.last_update_info: dict[str, Any] = {}
        
        self.plus_path: str = ""
        self.minus_path: str = ""
        self._registered = False

    @property
    def request_load_inplace(self) -> bool:
        """Whether vLLM should reload tensors from the request path."""
        return self.residency == "cpu"

    def attach_llm(self, llm: Any) -> None:
        self.llm = llm

    def _memory_path(self, lora_id: int) -> str:
        if self.residency == "gpu":
            return f"{GPU_MEMORY_PATH_PREFIX}/{lora_id}"
        return ""
    
    def _build_target_modules(self) -> List[str]:
        """
        Build list of target modules for LoRA.
        
        Returns short names (vLLM expects this format).
        """
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    
    def _build_empty_tensors(self) -> Dict[str, torch.Tensor]:
        """Build empty tensors for registration (PEFT format, only Linear modules)."""
        tensors = {}
        
        linear_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        
        for layer_idx in range(self.num_layers):
            for module_name in linear_modules:
                if module_name in ["fc1", "fc2"]:
                    module_path = f"model.decoder.layers.{layer_idx}.{module_name}"
                else:
                    module_path = f"model.decoder.layers.{layer_idx}.self_attn.{module_name}"
                
                # Determine dimensions
                if module_name == "fc1":
                    out_features, in_features = self.ffn_dim, self.hidden_size
                elif module_name == "fc2":
                    out_features, in_features = self.hidden_size, self.ffn_dim
                else:
                    out_features, in_features = self.hidden_size, self.hidden_size
                
                lora_A = torch.zeros(self.rank, in_features, dtype=torch.float16)
                lora_B = torch.zeros(out_features, self.rank, dtype=torch.float16)
                
                # PEFT format keys
                tensors[f"base_model.model.{module_path}.lora_A.weight"] = lora_A
                tensors[f"base_model.model.{module_path}.lora_B.weight"] = lora_B
        
        return tensors

    def _build_config(self) -> dict:
        target_modules = self._build_target_modules()
        return {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": self.base_model_name,
            "bias": "none",
            "exclude_modules": [],
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": float(self.rank),
            "lora_dropout": 0.0,
            "megatron_core": "megatron.core",
            "megatron_config": None,
            "modules_to_save": None,
            "r": self.rank,
            "rank_pattern": {},
            "revision": None,
            "target_modules": target_modules,
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }

    def _set_ids_for_step(self, step: int) -> None:
        if self.stable_lora_ids:
            self.plus_id = self._base_plus_id
            self.minus_id = self._base_minus_id
            self.plus_name = "lozo_plus"
            self.minus_name = "lozo_minus"
        else:
            # Compatibility path for vLLM versions without load_inplace support.
            self.plus_id = 9000 + 2 * step + 1
            self.minus_id = 9000 + 2 * step + 2
            self.plus_name = f"lozo_plus_step_{step}"
            self.minus_name = f"lozo_minus_step_{step}"

        if self.residency == "gpu":
            self.plus_path = self._memory_path(self.plus_id)
            self.minus_path = self._memory_path(self.minus_id)

    def _load_gpu_lora(
        self,
        lora_id: int,
        config: dict,
        tensors: Dict[str, torch.Tensor],
    ) -> dict:
        if self.llm is None:
            raise RuntimeError("GPU LoRA residency requires TempLoRARuntime(llm=...)")
        fn = partial(
            _load_lora_tensors_into_vllm_model,
            lora_id=lora_id,
            config=config,
            tensors=tensors,
        )
        results = self.llm.apply_model(fn)
        if not results:
            raise RuntimeError("vLLM apply_model returned no results while loading LoRA")
        return {"workers": results}

    def _load_gpu_lora_pair(
        self,
        config: dict,
        plus_tensors: Dict[str, torch.Tensor],
        minus_tensors: Dict[str, torch.Tensor],
    ) -> dict:
        if self.llm is None:
            raise RuntimeError("GPU LoRA residency requires TempLoRARuntime(llm=...)")
        fn = partial(
            _load_two_lora_tensors_into_vllm_model,
            plus_id=self.plus_id,
            minus_id=self.minus_id,
            config=config,
            plus_tensors=plus_tensors,
            minus_tensors=minus_tensors,
        )
        results = self.llm.apply_model(fn)
        if not results:
            raise RuntimeError("vLLM apply_model returned no results while loading LoRA")
        return {"workers": results}

    def _update_gpu_lora_slots_direct(
        self,
        plus_A: Dict[str, torch.Tensor],
        plus_B: Dict[str, torch.Tensor],
        minus_A: Dict[str, torch.Tensor],
        minus_B: Dict[str, torch.Tensor],
    ) -> dict:
        if self.llm is None:
            raise RuntimeError("GPU LoRA residency requires TempLoRARuntime(llm=...)")
        fn = partial(
            _update_lora_slots_in_vllm_model,
            plus_id=self.plus_id,
            minus_id=self.minus_id,
            plus_A=plus_A,
            plus_B=plus_B,
            minus_A=minus_A,
            minus_B=minus_B,
        )
        results = self.llm.apply_model(fn)
        if not results:
            raise RuntimeError("vLLM apply_model returned no results while updating LoRA")
        return {"workers": results}

    def _update_gpu_lora_slots_from_directions(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        eps: float,
    ) -> dict:
        if self.llm is None:
            raise RuntimeError("GPU LoRA residency requires TempLoRARuntime(llm=...)")
        fn = partial(
            _update_lora_slots_from_directions_in_vllm_model,
            plus_id=self.plus_id,
            minus_id=self.minus_id,
            directions_2d=directions_2d,
            eps=eps,
        )
        results = self.llm.apply_model(fn)
        if not results:
            raise RuntimeError("vLLM apply_model returned no results while updating LoRA")
        return {"workers": results}

    def _remove_gpu_lora(self, lora_id: int) -> None:
        if self.llm is None:
            return
        fn = partial(_remove_lora_from_vllm_model, lora_id=lora_id)
        self.llm.apply_model(fn)

    def register_slots(self):
        """
        Register two empty LoRA slots.

        Note: Actual tensors will be updated via update_plus_minus().
        """
        self._set_ids_for_step(step=0)
        if self.residency == "gpu":
            self.plus_path = self._memory_path(self.plus_id)
            self.minus_path = self._memory_path(self.minus_id)
            if self.injection == "direct":
                config = self._build_config()
                empty_tensors = self._build_empty_tensors()
                self.last_update_info = self._load_gpu_lora_pair(
                    config,
                    empty_tensors,
                    empty_tensors.copy(),
                )
            self._registered = True
            return

        config = self._build_config()
        
        empty_tensors = self._build_empty_tensors()
        
        self.plus_path = register_memory_lora_cpu(
            self.plus_id,
            config,
            empty_tensors,
        )
        
        self.minus_path = register_memory_lora_cpu(
            self.minus_id,
            config.copy(),
            empty_tensors.copy(),
        )
        
        self._registered = True
    
    def build_peft_tensors(
        self,
        layer_to_A: Dict[str, torch.Tensor],
        layer_to_B: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert layer_to_A/B to PEFT format tensor dict.
        
        PEFT format: base_model.model.{module_path}.lora_A.weight
        
        Args:
            layer_to_A: {module_name: [rank, in_features]}  (HF weight names with .weight suffix)
            layer_to_B: {module_name: [out_features, rank]}
        
        Returns:
            tensors: {base_model.model.{module_path}.lora_A.weight: tensor, ...}
        """
        tensors = {}
        
        for name, A in layer_to_A.items():
            # Strip ".weight" suffix to get module path
            # e.g., "model.decoder.layers.0.self_attn.k_proj.weight" -> "model.decoder.layers.0.self_attn.k_proj"
            module_path = name
            if module_path.endswith(".weight"):
                module_path = module_path[:-7]  # Remove ".weight"
            
            tensors[f"base_model.model.{module_path}.lora_A.weight"] = A
            tensors[f"base_model.model.{module_path}.lora_B.weight"] = layer_to_B[name]
        
        return tensors
    
    def update_plus_minus(
        self,
        plus_A: Dict[str, torch.Tensor],
        plus_B: Dict[str, torch.Tensor],
        minus_A: Dict[str, torch.Tensor],
        minus_B: Dict[str, torch.Tensor],
        step: int = 0,
    ):
        """
        Update both plus/minus LoRA slots.
        
        In CPU residency this re-registers memory paths and relies on
        LoRARequest(load_inplace=True). In GPU manager mode this replaces
        adapters through vLLM's LoRA manager. In GPU direct mode this overwrites
        the fixed plus/minus slots in place, so scoring requests only select
        already-loaded adapter IDs.
        """
        old_plus_id = self.plus_id
        old_minus_id = self.minus_id
        if self.residency == "cpu" and self._registered:
            unregister_memory_lora(self.plus_id)
            unregister_memory_lora(self.minus_id)
        
        self._set_ids_for_step(step)
        if (
            self.residency == "gpu"
            and self._registered
            and {old_plus_id, old_minus_id} != {self.plus_id, self.minus_id}
        ):
            self._remove_gpu_lora(old_plus_id)
            self._remove_gpu_lora(old_minus_id)
        if self.residency == "gpu":
            if self.injection == "direct":
                if not self._registered:
                    config = self._build_config()
                    plus_tensors = self.build_peft_tensors(plus_A, plus_B)
                    minus_tensors = self.build_peft_tensors(minus_A, minus_B)
                    self.last_update_info = self._load_gpu_lora_pair(
                        config,
                        plus_tensors,
                        minus_tensors,
                    )
                    self._registered = True
                else:
                    self.last_update_info = self._update_gpu_lora_slots_direct(
                        plus_A,
                        plus_B,
                        minus_A,
                        minus_B,
                    )
            else:
                config = self._build_config()
                plus_tensors = self.build_peft_tensors(plus_A, plus_B)
                minus_tensors = self.build_peft_tensors(minus_A, minus_B)
                self.last_update_info = {
                    "plus": self._load_gpu_lora(self.plus_id, config, plus_tensors),
                    "minus": self._load_gpu_lora(
                        self.minus_id,
                        config.copy(),
                        minus_tensors,
                    ),
                }
        else:
            config = self._build_config()
            plus_tensors = self.build_peft_tensors(plus_A, plus_B)
            minus_tensors = self.build_peft_tensors(minus_A, minus_B)
            self.plus_path = register_memory_lora_cpu(self.plus_id, config, plus_tensors)
            self.minus_path = register_memory_lora_cpu(
                self.minus_id,
                config.copy(),
                minus_tensors,
            )
            self.last_update_info = {}

        self._registered = True

    def update_plus_minus_from_directions(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        *,
        eps: float,
        step: int = 0,
    ):
        """Update plus/minus GPU slots directly from U/V directions."""
        if self.residency != "gpu" or self.injection != "direct":
            raise RuntimeError("direction slot update requires GPU direct injection")
        old_plus_id = self.plus_id
        old_minus_id = self.minus_id
        self._set_ids_for_step(step)
        if (
            self._registered
            and {old_plus_id, old_minus_id} != {self.plus_id, self.minus_id}
        ):
            self._remove_gpu_lora(old_plus_id)
            self._remove_gpu_lora(old_minus_id)
        if not self._registered:
            config = self._build_config()
            empty_tensors = self._build_empty_tensors()
            self.last_update_info = self._load_gpu_lora_pair(
                config,
                empty_tensors,
                empty_tensors.copy(),
            )
            self._registered = True
        self.last_update_info = self._update_gpu_lora_slots_from_directions(
            directions_2d,
            eps,
        )

    def get_plus_request_info(self) -> tuple[str, int, str]:
        """Get (name, id, path) for plus LoRA."""
        return self.plus_name, self.plus_id, self.plus_path
    
    def get_minus_request_info(self) -> tuple[str, int, str]:
        """Get (name, id, path) for minus LoRA."""
        return self.minus_name, self.minus_id, self.minus_path
    
    def cleanup(self):
        """Unregister LoRA slots."""
        if self._registered:
            if self.residency == "gpu":
                self._remove_gpu_lora(self.plus_id)
                self._remove_gpu_lora(self.minus_id)
            else:
                unregister_memory_lora(self.plus_id)
                unregister_memory_lora(self.minus_id)
            self._registered = False
