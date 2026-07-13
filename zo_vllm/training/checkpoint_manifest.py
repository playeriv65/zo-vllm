"""Exact runtime manifests shared by native and LoRA checkpoints."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


LAYER_MAPPING_FORMAT_VERSION = 1


def build_layer_mapping_manifest(weight_sync: Any) -> dict[str, Any]:
    """Serialize the exact HF-to-vLLM parameter mapping used by a runtime."""

    hf_to_vllm = getattr(weight_sync, "hf_to_vllm_mapping", None)
    hf_to_slice = getattr(weight_sync, "hf_to_slice", None)
    if not isinstance(hf_to_vllm, Mapping) or not isinstance(hf_to_slice, Mapping):
        raise TypeError(
            "checkpoint weight_sync must expose hf_to_vllm_mapping and hf_to_slice"
        )
    mapping = {
        str(name): str(target)
        for name, target in sorted(hf_to_vllm.items(), key=lambda item: str(item[0]))
    }
    slices: dict[str, list[int]] = {}
    for name, bounds in sorted(hf_to_slice.items(), key=lambda item: str(item[0])):
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise TypeError(f"invalid HF-to-vLLM slice for {name!r}: {bounds!r}")
        slices[str(name)] = [int(bounds[0]), int(bounds[1])]
    model_config = getattr(weight_sync, "model_config", None)
    hidden_size = (
        None if model_config is None else getattr(model_config, "hidden_size", None)
    )
    for name, target in mapping.items():
        if name in slices or "qkv_proj.weight" not in target:
            continue
        if hidden_size is None:
            raise TypeError(
                "checkpoint weight_sync must expose model_config.hidden_size for "
                f"implicit packed QKV mapping {name!r}"
            )
        width = int(hidden_size)
        if name.endswith(".q_proj.weight"):
            slices[name] = [0, width]
        elif name.endswith(".k_proj.weight"):
            slices[name] = [width, 2 * width]
        elif name.endswith(".v_proj.weight"):
            slices[name] = [2 * width, 3 * width]
        else:
            raise ValueError(f"unknown implicit packed QKV mapping: {name!r}")
    canonical = {
        "format_version": LAYER_MAPPING_FORMAT_VERSION,
        "hf_to_vllm_mapping": mapping,
        "hf_to_slice": slices,
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        **canonical,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def validate_layer_mapping_manifest(
    recorded: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Require an exact mapping match before restoring runtime state."""

    if dict(recorded) != dict(expected):
        raise ValueError(
            "checkpoint layer mapping does not match the current vLLM runtime"
        )


__all__ = [
    "LAYER_MAPPING_FORMAT_VERSION",
    "build_layer_mapping_manifest",
    "validate_layer_mapping_manifest",
]
