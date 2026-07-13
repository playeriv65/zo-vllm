"""Runtime checkpoint adapters for the Hugging Face ZO trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from zo_vllm.training.checkpoint_manifest import (
    build_layer_mapping_manifest,
    validate_layer_mapping_manifest,
)
from zo_vllm.training.native_checkpoint import (
    clear_update_state_for_loaded_checkpoint,
    load_native_checkpoint_into_workers,
    save_effective_native_checkpoint,
)
from zo_vllm.training.lora_checkpoint import (
    load_lora_bank_checkpoint,
    save_lora_bank_checkpoint,
)

from .checkpointing import read_zo_checkpoint_metadata


@dataclass
class ZOVLLMCheckpointHandler:
    """Checkpoint handler that reuses the existing ZO-vLLM payload helpers."""

    checkpoint_mode: str = "metadata"
    llm: Any | None = None
    weight_sync: Any | None = None
    update_state: Any | None = None
    precision: str = "param"
    lora_dtype: str | torch.dtype = torch.float16
    lora_load_device: str | torch.device | None = None
    lora_load_dtype: str | torch.dtype | None = None
    runtime_manifest: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_mode not in {"metadata", "native", "lora"}:
            raise ValueError("checkpoint_mode must be one of: metadata, native, lora")
        if self.checkpoint_mode == "native" and self.llm is None:
            raise ValueError("native checkpoint mode requires llm and weight_sync")
        if self.checkpoint_mode in {"native", "lora"} and self.weight_sync is None:
            raise ValueError(
                f"{self.checkpoint_mode} checkpoint mode requires weight_sync"
            )
        if self.checkpoint_mode == "lora" and self.update_state is None:
            raise ValueError("lora checkpoint mode requires update_state")

    def save_checkpoint(
        self,
        output_dir: str,
        *,
        step: int,
        metrics: Mapping[str, Any] | None,
        reason: str,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.checkpoint_mode,
            "loadable": self.checkpoint_mode != "metadata",
            "step": int(step),
            "reason": str(reason),
        }
        if self.checkpoint_mode != "metadata":
            runtime_manifest = dict(self.runtime_manifest or {})
            if "layer_mapping" in runtime_manifest:
                raise ValueError("runtime_manifest.layer_mapping is runtime-owned")
            runtime_manifest["layer_mapping"] = build_layer_mapping_manifest(
                self.weight_sync
            )
            payload["runtime_manifest"] = runtime_manifest
        elif self.runtime_manifest is not None:
            payload["runtime_manifest"] = dict(self.runtime_manifest)
        if metrics is not None:
            payload["metrics"] = dict(metrics)
        if self.checkpoint_mode == "metadata":
            return payload
        if self.checkpoint_mode == "native":
            uses_lora_bank = _is_lora_bank_update_state(self.update_state)
            save_info = save_effective_native_checkpoint(
                llm=self.llm,
                checkpoint_dir=output_dir,
                accumulated_update_state=(
                    self.update_state if uses_lora_bank else None
                ),
                weight_sync=self.weight_sync,
                step=int(step),
                use_lora_bank_update=uses_lora_bank,
                precision=self.precision,
            )
            payload.update(save_info)
            return payload
        save_info = save_lora_bank_checkpoint(
            checkpoint_dir=output_dir,
            accumulated_update_state=self.update_state,
            step=int(step),
            raw_step=int(step),
            dtype=_resolve_dtype(self.lora_dtype),
            layer_mapping=runtime_manifest["layer_mapping"],
        )
        payload.update(save_info)
        return payload

    def load_checkpoint(self, checkpoint_dir: str) -> Mapping[str, Any]:
        if self.checkpoint_mode == "metadata":
            raise RuntimeError("metadata-only ZO checkpoints are not loadable")
        metadata = read_zo_checkpoint_metadata(checkpoint_dir)
        payload = metadata.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint metadata payload must be a mapping")
        runtime_manifest = payload.get("runtime_manifest")
        if not isinstance(runtime_manifest, Mapping):
            raise ValueError("checkpoint payload is missing runtime_manifest")
        recorded_mapping = runtime_manifest.get("layer_mapping")
        if not isinstance(recorded_mapping, Mapping):
            raise ValueError("checkpoint runtime manifest is missing layer_mapping")
        current_mapping = build_layer_mapping_manifest(self.weight_sync)
        validate_layer_mapping_manifest(recorded_mapping, current_mapping)
        if self.checkpoint_mode == "native":
            results = load_native_checkpoint_into_workers(self.llm, checkpoint_dir)
            clear_update_state_for_loaded_checkpoint(self.update_state)
            return {
                "mode": "native",
                "loadable": True,
                "load_results": results,
            }
        device = (
            self.lora_load_device
            if self.lora_load_device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        result = load_lora_bank_checkpoint(
            checkpoint_dir,
            accumulated_update_state=self.update_state,
            expected_layer_mapping=current_mapping,
            device=device,
            dtype=(
                None
                if self.lora_load_dtype is None
                else _resolve_dtype(self.lora_load_dtype)
            ),
        )
        return {"mode": "lora", "loadable": True, **dict(result)}


def _resolve_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = str(value).replace("torch.", "")
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {value!r}")


def _is_lora_bank_update_state(update_state: Any | None) -> bool:
    return update_state is not None and callable(
        getattr(update_state, "clean_directions_for_score", None)
    )


__all__ = [
    "ZOVLLMCheckpointHandler",
]
