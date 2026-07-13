"""Direct vLLM LoRA slot write helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, Sequence

import torch

from zo_vllm.core.lora_runtime.debug import (
    _debug_base_weight_sentinel,
    _debug_direction_slot_storage,
    _debug_slot_tensor_finite_summary,
)
from zo_vllm.core.lora_runtime.slot_validation import _adapter_slot_manager
from zo_vllm.core.lora_runtime.streams import (
    _copy_stream_device,
    _get_background_lora_copy_stream,
    _lora_slot_use_events,
    _normalize_slot_write_stream,
    _record_lora_slot_write_event,
    _record_tensors_on_stream,
)


def _lora_slot_index(manager, lora_id: int) -> int:
    manager = _adapter_slot_manager(manager)
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

    This is a training-only fast path for fixed-rank direct update adapters. If a
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


def _direction_lora_pair(
    direction: dict[str, torch.Tensor],
    *,
    eps: float,
    sign: float,
    output_projection: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    U = direction["U"]
    V = direction["V"]
    V_T = direction.get("V_T")
    scale = float(direction.get("scale", 1.0))

    if output_projection:
        U_accum = direction.get("U_accum")
        if U_accum is not None:
            lora_a = (U_accum + (eps * sign * scale) * U).T
            lora_b = V
        else:
            lora_a = U.T
            lora_b = (eps * sign * scale) * V
        return lora_a, lora_b

    lora_a = V_T if V_T is not None else V.T

    custom_plus_b = direction.get("lora_B_plus")
    custom_minus_b = direction.get("lora_B_minus")
    if custom_plus_b is not None or custom_minus_b is not None:
        if custom_plus_b is None or custom_minus_b is None:
            raise RuntimeError("direction provides only one custom LoRA B tensor")
        return lora_a, custom_plus_b if sign > 0 else custom_minus_b

    U_accum = direction.get("U_accum")
    if U_accum is not None:
        return lora_a, U_accum + (eps * sign * scale) * U
    return lora_a, (eps * sign * scale) * U


def _is_output_projection_lora(module_name: str, module) -> bool:
    return (
        module_name.endswith("lm_head")
        or type(module).__name__ == "LogitsProcessorWithLoRA"
    )


def _direction_key_for_module(
    module_name: str,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
) -> str | None:
    direct_direction_key = f"{module_name}.weight"
    if direct_direction_key in directions_2d:
        return direct_direction_key
    if module_name.endswith("lm_head"):
        for tied_embedding_key in (
            "model.decoder.embed_tokens.weight",
            "model.embed_tokens.weight",
        ):
            if tied_embedding_key in directions_2d:
                return tied_embedding_key
    return None


def _write_direction_fallback_slot(
    *,
    module,
    slot_index: int,
    direction_keys: list[str],
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
    eps: float,
    sign: float,
) -> list[str]:
    lora_a: list[torch.Tensor | None] = []
    lora_b: list[torch.Tensor | None] = []
    missing: list[str] = []
    has_any = False
    for direction_key in direction_keys:
        direction = directions_2d.get(direction_key)
        if direction is None:
            missing.append(direction_key)
            lora_a.append(None)
            lora_b.append(None)
            continue
        a_i, b_i = _direction_lora_pair(
            direction,
            eps=eps,
            sign=sign,
            output_projection=_is_output_projection_lora(
                str(getattr(module, "_zo_vllm_module_name", "")),
                module,
            ),
        )
        lora_a.append(a_i)
        lora_b.append(b_i)
        has_any = True

    if missing:
        return missing
    if not has_any:
        module.reset_lora(slot_index)
        return []
    if not _set_lora_noreset_if_full(module, slot_index, lora_a, lora_b):
        if (
            len(lora_a) == 1
            and len(lora_b) == 1
            and isinstance(lora_a[0], torch.Tensor)
            and isinstance(lora_b[0], torch.Tensor)
        ):
            module.set_lora(slot_index, lora_a[0], lora_b[0])
        else:
            module.set_lora(slot_index, lora_a, lora_b)
    return []


def _build_direction_slot_plan(
    manager,
    *,
    plus_index: int,
    minus_index: int,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    fallback_entries: list[dict[str, Any]] = []
    modules_written = 0
    packed_written = 0
    missing: list[str] = []

    for module_name, module in manager.modules.items():
        if module_name in manager.packed_modules:
            direct_direction_key = f"{module_name}.weight"
            if direct_direction_key in directions_2d:
                fallback_entries.append(
                    {
                        "module": module,
                        "module_name": module_name,
                        "direction_keys": [direct_direction_key],
                    }
                )
                packed_written += 1
                continue

            replacements = list(manager.packed_modules[module_name])
            if len(replacements) != len(module.lora_a_stacked):
                fallback_entries.append(
                    {
                        "module": module,
                        "module_name": module_name,
                        "direction_keys": [f"{name}.weight" for name in replacements],
                    }
                )
                packed_written += 1
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
                fallback_entries.append(
                    {
                        "module": module,
                        "module_name": module_name,
                        "direction_keys": [f"{name}.weight" for name in replacements],
                    }
                )
                packed_written += 1
            else:
                module.reset_lora(plus_index)
                module.reset_lora(minus_index)
                missing.append(module_name)
            continue

        direction_key = _direction_key_for_module(module_name, directions_2d)
        if direction_key is None:
            module.reset_lora(plus_index)
            module.reset_lora(minus_index)
            missing.append(module_name)
            continue
        if _is_output_projection_lora(module_name, module):
            module._zo_vllm_module_name = module_name
            fallback_entries.append(
                {
                    "module": module,
                    "module_name": module_name,
                    "direction_keys": [direction_key],
                }
            )
            modules_written += 1
            continue
        direction = directions_2d.get(direction_key)
        if direction is None:
            module.reset_lora(plus_index)
            module.reset_lora(minus_index)
            missing.append(module_name)
            continue
        if len(module.lora_a_stacked) != 1:
            fallback_entries.append(
                {
                    "module": module,
                    "module_name": module_name,
                    "direction_keys": [direction_key],
                }
            )
            modules_written += 1
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
            fallback_entries.append(
                {
                    "module": module,
                    "module_name": module_name,
                    "direction_keys": [direction_key],
                }
            )
            modules_written += 1
        else:
            entries.append(entry)
            modules_written += 1

    return {
        "entries": entries,
        "fallback_entries": fallback_entries,
        "modules_written": int(modules_written),
        "packed_written": int(packed_written),
        "missing": missing,
    }


def _flush_queued_direction_writes(
    *,
    manager=None,
    lora_ids: Sequence[int] = (),
    copy_stream: str = "default",
    debug_tensor_names: dict[str, Sequence[str]] | None = None,
    eps: float,
    plus_a_targets: list[torch.Tensor],
    minus_a_targets: list[torch.Tensor],
    a_sources: list[torch.Tensor],
    plus_b_targets: list[torch.Tensor],
    minus_b_targets: list[torch.Tensor],
    u_sources: list[torch.Tensor],
    u_scales: list[float],
    custom_plus_b_targets: list[torch.Tensor],
    custom_minus_b_targets: list[torch.Tensor],
    plus_b_sources: list[torch.Tensor],
    minus_b_sources: list[torch.Tensor],
    accum_plus_b_targets: list[torch.Tensor],
    accum_minus_b_targets: list[torch.Tensor],
    accum_sources: list[torch.Tensor],
    accum_u_sources: list[torch.Tensor],
    accum_u_scales: list[float],
) -> dict[str, Any]:
    debug_tensor_names = debug_tensor_names or {}

    def uniform_scale(scales: list[float]) -> float | None:
        if not scales:
            return 1.0
        first = float(scales[0])
        if all(float(item) == first for item in scales):
            return first
        return None

    def flush() -> None:
        if a_sources:
            torch._foreach_copy_(plus_a_targets, a_sources)
            torch._foreach_copy_(minus_a_targets, a_sources)
        if u_sources:
            torch._foreach_copy_(plus_b_targets, u_sources)
            torch._foreach_copy_(minus_b_targets, u_sources)
            scale = uniform_scale(u_scales)
            if scale is None:
                for target, scale_i in zip(plus_b_targets, u_scales):
                    target.mul_(float(eps) * float(scale_i))
                for target, scale_i in zip(minus_b_targets, u_scales):
                    target.mul_(-float(eps) * float(scale_i))
            else:
                torch._foreach_mul_(plus_b_targets, float(eps) * float(scale))
                torch._foreach_mul_(minus_b_targets, -float(eps) * float(scale))
        if plus_b_sources:
            torch._foreach_copy_(custom_plus_b_targets, plus_b_sources)
            torch._foreach_copy_(custom_minus_b_targets, minus_b_sources)
        if accum_sources:
            torch._foreach_copy_(accum_plus_b_targets, accum_sources)
            torch._foreach_copy_(accum_minus_b_targets, accum_sources)
            scale = uniform_scale(accum_u_scales)
            if scale is None:
                for target, source, scale_i in zip(
                    accum_plus_b_targets, accum_u_sources, accum_u_scales
                ):
                    target.add_(source, alpha=float(eps) * float(scale_i))
                for target, source, scale_i in zip(
                    accum_minus_b_targets, accum_u_sources, accum_u_scales
                ):
                    target.add_(source, alpha=-float(eps) * float(scale_i))
            else:
                torch._foreach_add_(
                    accum_plus_b_targets,
                    accum_u_sources,
                    alpha=float(eps) * float(scale),
                )
                torch._foreach_add_(
                    accum_minus_b_targets,
                    accum_u_sources,
                    alpha=-float(eps) * float(scale),
                )

    mode = _normalize_slot_write_stream(copy_stream)
    device = _copy_stream_device(
        (
            plus_a_targets,
            minus_a_targets,
            a_sources,
            plus_b_targets,
            minus_b_targets,
            u_sources,
            custom_plus_b_targets,
            custom_minus_b_targets,
            plus_b_sources,
            minus_b_sources,
            accum_plus_b_targets,
            accum_minus_b_targets,
            accum_sources,
            accum_u_sources,
        )
    )
    if (
        mode in {"background", "background_sync"}
        and manager is not None
        and device is not None
    ):
        stream = _get_background_lora_copy_stream(manager, device)
        stream.wait_stream(torch.cuda.current_stream(device))
        use_events = _lora_slot_use_events(manager, lora_ids=lora_ids)
        sync_start = time.perf_counter() if mode == "background_sync" else None
        with torch.cuda.stream(stream):
            for event in use_events:
                stream.wait_event(event)
            flush()
            _record_tensors_on_stream(
                (
                    plus_a_targets,
                    minus_a_targets,
                    a_sources,
                    plus_b_targets,
                    minus_b_targets,
                    u_sources,
                    custom_plus_b_targets,
                    custom_minus_b_targets,
                    plus_b_sources,
                    minus_b_sources,
                    accum_plus_b_targets,
                    accum_minus_b_targets,
                    accum_sources,
                    accum_u_sources,
                ),
                stream,
            )
            event = torch.cuda.Event(blocking=False)
            event.record(stream)
        if mode == "background_sync":
            event.synchronize()
            _record_lora_slot_write_event(manager, lora_ids=lora_ids, event=None)
            sync_s = time.perf_counter() - float(sync_start)
        else:
            _record_lora_slot_write_event(manager, lora_ids=lora_ids, event=event)
            sync_s = 0.0
        result = {
            "copy_stream": mode,
            "copy_event_recorded": mode == "background",
            "copy_event_device": str(device),
            "use_events_waited": len(use_events),
            "host_synchronized": mode == "background_sync",
            "host_sync_s": sync_s,
        }
        finite_summary = _debug_slot_tensor_finite_summary(
            (
                ("plus_a_targets", plus_a_targets),
                ("minus_a_targets", minus_a_targets),
                ("a_sources", a_sources, debug_tensor_names.get("a", ())),
                ("plus_b_targets", plus_b_targets),
                ("minus_b_targets", minus_b_targets),
                ("u_sources", u_sources, debug_tensor_names.get("u", ())),
                ("custom_plus_b_targets", custom_plus_b_targets),
                ("custom_minus_b_targets", custom_minus_b_targets),
                (
                    "plus_b_sources",
                    plus_b_sources,
                    debug_tensor_names.get("custom_b", ()),
                ),
                (
                    "minus_b_sources",
                    minus_b_sources,
                    debug_tensor_names.get("custom_b", ()),
                ),
                ("accum_plus_b_targets", accum_plus_b_targets),
                ("accum_minus_b_targets", accum_minus_b_targets),
                (
                    "accum_sources",
                    accum_sources,
                    debug_tensor_names.get("accum", ()),
                ),
                (
                    "accum_u_sources",
                    accum_u_sources,
                    debug_tensor_names.get("accum", ()),
                ),
            )
        )
        if finite_summary is not None:
            result["finite_summary"] = finite_summary
        return result

    if a_sources:
        flush()
    elif u_sources or plus_b_sources or accum_sources:
        flush()
    host_synchronized = False
    host_sync_s = 0.0
    if mode == "default_sync" and device is not None and torch.cuda.is_available():
        sync_start = time.perf_counter()
        torch.cuda.synchronize(device)
        host_synchronized = True
        host_sync_s = time.perf_counter() - sync_start
    if manager is not None and lora_ids:
        _record_lora_slot_write_event(manager, lora_ids=lora_ids, event=None)
    result = {
        "copy_stream": mode,
        "copy_event_recorded": False,
        "copy_event_device": None if device is None else str(device),
        "use_events_waited": 0,
        "host_synchronized": host_synchronized,
        "host_sync_s": host_sync_s,
    }
    finite_summary = _debug_slot_tensor_finite_summary(
        (
            ("plus_a_targets", plus_a_targets),
            ("minus_a_targets", minus_a_targets),
            ("a_sources", a_sources, debug_tensor_names.get("a", ())),
            ("plus_b_targets", plus_b_targets),
            ("minus_b_targets", minus_b_targets),
            ("u_sources", u_sources, debug_tensor_names.get("u", ())),
            ("custom_plus_b_targets", custom_plus_b_targets),
            ("custom_minus_b_targets", custom_minus_b_targets),
            (
                "plus_b_sources",
                plus_b_sources,
                debug_tensor_names.get("custom_b", ()),
            ),
            (
                "minus_b_sources",
                minus_b_sources,
                debug_tensor_names.get("custom_b", ()),
            ),
            ("accum_plus_b_targets", accum_plus_b_targets),
            ("accum_minus_b_targets", accum_minus_b_targets),
            ("accum_sources", accum_sources, debug_tensor_names.get("accum", ())),
            ("accum_u_sources", accum_u_sources, debug_tensor_names.get("accum", ())),
        )
    )
    if finite_summary is not None:
        result["finite_summary"] = finite_summary
    return result


def _write_plus_minus_slots_from_directions(
    manager,
    *,
    plus_id: int,
    minus_id: int,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
    eps: float,
    copy_stream: str = "default",
) -> dict:
    """Write plus/minus slots directly from LOZO U/V directions."""
    event_owner = manager
    manager = _adapter_slot_manager(manager)
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
    u_scales: list[float] = []
    custom_plus_b_targets: list[torch.Tensor] = []
    custom_minus_b_targets: list[torch.Tensor] = []
    plus_b_sources: list[torch.Tensor] = []
    minus_b_sources: list[torch.Tensor] = []
    accum_plus_b_targets: list[torch.Tensor] = []
    accum_minus_b_targets: list[torch.Tensor] = []
    accum_sources: list[torch.Tensor] = []
    accum_u_sources: list[torch.Tensor] = []
    accum_u_scales: list[float] = []
    a_names: list[str] = []
    u_names: list[str] = []
    custom_b_names: list[str] = []
    accum_names: list[str] = []

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
    _debug_direction_slot_storage(plan["entries"], stage="direction_plan")

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
            a_names.append(entry["direction_key"])
        custom_plus_b = direction.get("lora_B_plus")
        custom_minus_b = direction.get("lora_B_minus")
        U_accum = direction.get("U_accum")
        if U_accum is not None:
            scale = float(direction.get("scale", 1.0))
            accum_plus_b_targets.append(entry["plus_b"])
            accum_minus_b_targets.append(entry["minus_b"])
            accum_sources.append(U_accum)
            accum_u_sources.append(U)
            accum_u_scales.append(scale)
            accum_names.append(entry["direction_key"])
        elif custom_plus_b is not None or custom_minus_b is not None:
            if custom_plus_b is None or custom_minus_b is None:
                cache.pop(cache_key, None)
                raise RuntimeError("direction provides only one custom LoRA B tensor")
            custom_plus_b_targets.append(entry["plus_b"])
            custom_minus_b_targets.append(entry["minus_b"])
            plus_b_sources.append(custom_plus_b)
            minus_b_sources.append(custom_minus_b)
            custom_b_names.append(entry["direction_key"])
        else:
            scale = float(direction.get("scale", 1.0))
            plus_b_targets.append(entry["plus_b"])
            minus_b_targets.append(entry["minus_b"])
            u_sources.append(U)
            u_scales.append(scale)
            u_names.append(entry["direction_key"])

    for entry in plan.get("fallback_entries", []):
        missing_direction_keys.extend(
            _write_direction_fallback_slot(
                module=entry["module"],
                slot_index=plus_index,
                direction_keys=entry["direction_keys"],
                directions_2d=directions_2d,
                eps=eps,
                sign=1.0,
            )
        )
        missing_direction_keys.extend(
            _write_direction_fallback_slot(
                module=entry["module"],
                slot_index=minus_index,
                direction_keys=entry["direction_keys"],
                directions_2d=directions_2d,
                eps=eps,
                sign=-1.0,
            )
        )

    if missing_direction_keys:
        cache.pop(cache_key, None)
        raise RuntimeError(
            "cached direct direction slot plan is stale; missing directions: "
            + ", ".join(missing_direction_keys[:8])
        )
    profile_after_traversal = time.perf_counter()
    stream_info = _flush_queued_direction_writes(
        manager=None if plan.get("fallback_entries") else event_owner,
        lora_ids=(int(plus_id), int(minus_id)),
        copy_stream="default" if plan.get("fallback_entries") else copy_stream,
        debug_tensor_names={
            "a": a_names,
            "u": u_names,
            "custom_b": custom_b_names,
            "accum": accum_names,
        },
        eps=eps,
        plus_a_targets=plus_a_targets,
        minus_a_targets=minus_a_targets,
        a_sources=a_sources,
        plus_b_targets=plus_b_targets,
        minus_b_targets=minus_b_targets,
        u_sources=u_sources,
        u_scales=u_scales,
        custom_plus_b_targets=custom_plus_b_targets,
        custom_minus_b_targets=custom_minus_b_targets,
        plus_b_sources=plus_b_sources,
        minus_b_sources=minus_b_sources,
        accum_plus_b_targets=accum_plus_b_targets,
        accum_minus_b_targets=accum_minus_b_targets,
        accum_sources=accum_sources,
        accum_u_sources=accum_u_sources,
        accum_u_scales=accum_u_scales,
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
        "slot_write_stream": stream_info,
    }


def _write_single_slot_from_directions(
    manager,
    *,
    lora_id: int,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
    eps: float,
    sign: float = 1.0,
) -> dict:
    """Write one direct LoRA slot from one direction map."""
    manager = _adapter_slot_manager(manager)
    profile_t0 = time.perf_counter()
    slot_index = _lora_slot_index(manager, lora_id)
    profile_after_index = time.perf_counter()
    missing_direction_keys: list[str] = []
    modules_written = 0
    packed_written = 0
    for module_name, module in manager.modules.items():
        if module_name in manager.packed_modules:
            direct_direction_key = f"{module_name}.weight"
            if direct_direction_key in directions_2d:
                direction_keys = [direct_direction_key]
            else:
                direction_keys = [
                    f"{name}.weight" for name in manager.packed_modules[module_name]
                ]
            missing_direction_keys.extend(
                _write_direction_fallback_slot(
                    module=module,
                    slot_index=slot_index,
                    direction_keys=direction_keys,
                    directions_2d=directions_2d,
                    eps=eps,
                    sign=sign,
                )
            )
            packed_written += 1
            continue

        direction_key = _direction_key_for_module(module_name, directions_2d)
        if direction_key is None:
            module.reset_lora(slot_index)
            missing_direction_keys.append(module_name)
            continue
        module._zo_vllm_module_name = module_name
        missing_direction_keys.extend(
            _write_direction_fallback_slot(
                module=module,
                slot_index=slot_index,
                direction_keys=[direction_key],
                directions_2d=directions_2d,
                eps=eps,
                sign=sign,
            )
        )
        modules_written += 1

    if missing_direction_keys:
        raise RuntimeError(
            "direct single-slot direction write is missing directions: "
            + ", ".join(missing_direction_keys[:8])
        )
    return {
        "slot": {"lora_id": int(lora_id), "slot_index": int(slot_index)},
        "modules_written": int(modules_written),
        "packed_written": int(packed_written),
        "missing": [],
        "source": "single_direction",
        "profile_s": {
            "slot_index": profile_after_index - profile_t0,
            "single_slot_total": time.perf_counter() - profile_t0,
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
    manager = _adapter_slot_manager(manager)
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
        raise RuntimeError(
            "vLLM model has no lora_manager; enable_lora=True is required"
        )

    before = _debug_base_weight_sentinel(model, "slot_write_before_direct")
    result = _write_plus_minus_slots(
        manager,
        plus_id=plus_id,
        minus_id=minus_id,
        plus_A=plus_A,
        plus_B=plus_B,
        minus_A=minus_A,
        minus_B=minus_B,
    )
    _raise_on_missing_direct_lora_modules(result, source="direct_lora_ab_update")
    after = _debug_base_weight_sentinel(model, "slot_write_after_direct")
    if before or after:
        result["base_weight_sentinel"] = {"before": before, "after": after}
    return result


def _update_lora_slots_from_directions_in_vllm_model(
    model,
    *,
    plus_id: int,
    minus_id: int,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
    eps: float,
    copy_stream: str = "default",
) -> dict:
    """Training-only fast path: build and write fixed slots inside vLLM."""
    manager = getattr(model, "lora_manager", None)
    if manager is None:
        raise RuntimeError(
            "vLLM model has no lora_manager; enable_lora=True is required"
        )

    before = _debug_base_weight_sentinel(model, "slot_write_before_directions")
    result = _write_plus_minus_slots_from_directions(
        manager,
        plus_id=plus_id,
        minus_id=minus_id,
        directions_2d=directions_2d,
        eps=eps,
        copy_stream=copy_stream,
    )
    _raise_on_missing_direct_lora_modules(result, source="direct_lora_direction_update")
    after = _debug_base_weight_sentinel(model, "slot_write_after_directions")
    if before or after:
        result["base_weight_sentinel"] = {"before": before, "after": after}
    return result


def _update_lora_slot_from_direction_in_vllm_model(
    model,
    *,
    lora_id: int,
    directions_2d: Dict[str, Dict[str, torch.Tensor]],
    eps: float,
    sign: float = 1.0,
) -> dict:
    """Training-only fast path: write one fixed slot inside vLLM."""
    manager = getattr(model, "lora_manager", None)
    if manager is None:
        raise RuntimeError(
            "vLLM model has no lora_manager; enable_lora=True is required"
        )
    before = _debug_base_weight_sentinel(model, "slot_write_before_single_direction")
    result = _write_single_slot_from_directions(
        manager,
        lora_id=lora_id,
        directions_2d=directions_2d,
        eps=eps,
        sign=sign,
    )
    _raise_on_missing_direct_lora_modules(
        result,
        source="direct_lora_single_direction_update",
    )
    after = _debug_base_weight_sentinel(model, "slot_write_after_single_direction")
    if before or after:
        result["base_weight_sentinel"] = {"before": before, "after": after}
    return result


def _raise_on_missing_direct_lora_modules(
    result: dict[str, Any], *, source: str
) -> None:
    missing = list(result.get("missing") or [])
    if not missing:
        return
    raise RuntimeError(
        "direct LoRA update did not provide tensors for active vLLM LoRA "
        f"modules; source={source}; missing={missing[:12]}"
    )
