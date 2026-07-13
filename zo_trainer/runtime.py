"""Runtime checkpoint adapters for the Hugging Face ZO trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from zo_vllm.training.native_checkpoint import (
    clear_update_state_for_loaded_checkpoint,
    load_native_checkpoint_into_workers,
    save_effective_native_checkpoint,
)
from zo_vllm.training.lora_checkpoint import (
    load_lora_bank_checkpoint,
    save_lora_bank_checkpoint,
)


@dataclass
class ZOVLLMCheckpointHandler:
    """Checkpoint handler that reuses the existing ZO-vLLM payload helpers."""

    checkpoint_mode: str = "metadata"
    llm: Any | None = None
    weight_sync: Any | None = None
    update_state: Any | None = None
    use_lora_bank_update: bool = False
    precision: str = "param"
    lora_dtype: str | torch.dtype = torch.float16
    lora_load_device: str | torch.device | None = None
    lora_load_dtype: str | torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_mode not in {"metadata", "native", "lora"}:
            raise ValueError("checkpoint_mode must be one of: metadata, native, lora")
        if self.checkpoint_mode == "native" and (
            self.llm is None or self.weight_sync is None
        ):
            raise ValueError("native checkpoint mode requires llm and weight_sync")
        if (
            self.checkpoint_mode == "native"
            and self.use_lora_bank_update
            and self.update_state is None
        ):
            raise ValueError(
                "native LoRA-bank checkpoint mode requires an accumulated update_state"
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
        if metrics is not None:
            payload["metrics"] = dict(metrics)
        if self.checkpoint_mode == "metadata":
            return payload
        if self.checkpoint_mode == "native":
            save_info = save_effective_native_checkpoint(
                llm=self.llm,
                checkpoint_dir=output_dir,
                accumulated_update_state=(
                    self.update_state if self.use_lora_bank_update else None
                ),
                weight_sync=self.weight_sync,
                step=int(step),
                use_lora_bank_update=bool(self.use_lora_bank_update),
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
        )
        payload.update(save_info)
        return payload

    def load_checkpoint(self, checkpoint_dir: str) -> Mapping[str, Any]:
        if self.checkpoint_mode == "metadata":
            raise RuntimeError("metadata-only ZO checkpoints are not loadable")
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
