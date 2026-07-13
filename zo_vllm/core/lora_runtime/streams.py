"""CUDA stream and event helpers for LoRA slot writes."""

from typing import Sequence

import torch


def _copy_stream_device(
    tensor_groups: Sequence[Sequence[torch.Tensor]],
) -> torch.device | None:
    for tensors in tensor_groups:
        for tensor in tensors:
            if tensor.is_cuda:
                return tensor.device
    return None


def _normalize_slot_write_stream(value: str | None) -> str:
    normalized = (value or "default").strip().lower().replace("-", "_")
    if normalized in {"default", "current"}:
        return "default"
    if normalized in {"sync", "synchronous", "default_sync", "current_sync"}:
        return "default_sync"
    if normalized in {"background", "async", "low_priority", "lowpriority"}:
        return "background"
    if normalized in {
        "background_sync",
        "async_sync",
        "low_priority_sync",
        "lowpriority_sync",
        "conservative",
    }:
        return "background_sync"
    raise ValueError(
        "slot_write_stream must be one of: default, default_sync, "
        "background, background_sync, low_priority"
    )


def _get_background_lora_copy_stream(
    manager, device: torch.device
) -> torch.cuda.Stream:
    streams = getattr(manager, "_zo_lora_copy_streams", None)
    if streams is None:
        streams = {}
        setattr(manager, "_zo_lora_copy_streams", streams)
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = streams.get(device_index)
    if stream is None:
        # CUDA exposes smaller numeric priorities as higher priority. The
        # least-priority value is usually 0, so this is mainly a separate stream
        # on common NVIDIA devices; when lower-priority streams are available, use
        # the lowest priority.
        priority_range = getattr(torch.cuda.Stream, "priority_range", None)
        if priority_range is None:
            least_priority = 0
        else:
            least_priority, _ = priority_range()
        stream = torch.cuda.Stream(device=device, priority=least_priority)
        streams[device_index] = stream
    return stream


def _record_lora_slot_write_event(
    manager,
    *,
    lora_ids: Sequence[int],
    event: torch.cuda.Event | None,
) -> None:
    owners = [manager, getattr(manager, "_adapter_manager", None)]
    seen: set[int] = set()
    for owner in owners:
        if owner is None:
            continue
        owner_id = id(owner)
        if owner_id in seen:
            continue
        seen.add(owner_id)
        events = getattr(owner, "_zo_lora_slot_write_events", None)
        if events is None:
            events = {}
            setattr(owner, "_zo_lora_slot_write_events", events)
        for lora_id in lora_ids:
            if event is None:
                events.pop(int(lora_id), None)
            else:
                events[int(lora_id)] = event


def _lora_slot_event_owners(manager) -> list[object]:
    owners = [manager, getattr(manager, "_adapter_manager", None)]
    result: list[object] = []
    seen: set[int] = set()
    for owner in owners:
        if owner is None:
            continue
        owner_id = id(owner)
        if owner_id in seen:
            continue
        seen.add(owner_id)
        result.append(owner)
    return result


def _lora_slot_use_events(
    manager,
    *,
    lora_ids: Sequence[int],
) -> list[torch.cuda.Event]:
    events_to_wait: list[torch.cuda.Event] = []
    seen_events: set[int] = set()
    active_ids = {int(lora_id) for lora_id in lora_ids if int(lora_id) > 0}
    if not active_ids:
        return events_to_wait
    for owner in _lora_slot_event_owners(manager):
        events = getattr(owner, "_zo_lora_slot_use_events", None)
        if not events:
            continue
        for lora_id in active_ids:
            event = events.get(lora_id)
            if event is None:
                continue
            event_id = id(event)
            if event_id in seen_events:
                continue
            seen_events.add(event_id)
            events_to_wait.append(event)
    return events_to_wait


def _record_tensors_on_stream(
    tensor_groups: Sequence[Sequence[torch.Tensor]],
    stream: torch.cuda.Stream,
) -> None:
    for tensors in tensor_groups:
        for tensor in tensors:
            if tensor.is_cuda:
                tensor.record_stream(stream)
