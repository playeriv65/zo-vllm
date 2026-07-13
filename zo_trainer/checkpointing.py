"""Checkpoint metadata helpers for the Hugging Face ZO trainer adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch


ZO_CHECKPOINT_METADATA_NAME = "zo_checkpoint_metadata.json"
ZO_CHECKPOINT_METADATA_VERSION = 1
ZO_DIRECTION_STATE_NAME = "zo_direction_state.pt"


class ZOCheckpointHandler(Protocol):
    """Runtime-aware save/load hooks owned by :class:`ZOTrainer`."""

    checkpoint_mode: str

    def save_checkpoint(
        self,
        output_dir: str,
        *,
        step: int,
        metrics: Mapping[str, Any] | None,
        reason: str,
    ) -> Mapping[str, Any]: ...

    def load_checkpoint(self, checkpoint_dir: str) -> Mapping[str, Any]: ...


def save_direction_provider_state(output_dir: str | Path, provider: Any) -> None:
    state_dict = getattr(provider, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeError("direction provider does not support checkpointing")
    torch.save(state_dict(), Path(output_dir) / ZO_DIRECTION_STATE_NAME)


def load_direction_provider_state(checkpoint_dir: str | Path, provider: Any) -> None:
    path = Path(checkpoint_dir) / ZO_DIRECTION_STATE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"direction provider checkpoint not found: {path}")
    load_state_dict = getattr(provider, "load_state_dict", None)
    if not callable(load_state_dict):
        raise RuntimeError("direction provider cannot load checkpoints")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("direction provider checkpoint must contain a mapping")
    load_state_dict(state)


@dataclass(frozen=True)
class ZOCheckpointMetadata:
    """Small JSON-serializable record for one ZO trainer checkpoint."""

    checkpoint_type: str
    checkpoint_mode: str
    global_step: int
    reason: str = "checkpoint"
    metrics: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.checkpoint_type not in {
            "zo_trainer_checkpoint",
            "zo_trainer_model",
        }:
            raise ValueError(f"unknown ZO checkpoint type: {self.checkpoint_type!r}")
        if self.checkpoint_mode not in {"metadata", "native", "lora"}:
            raise ValueError(f"unknown ZO checkpoint mode: {self.checkpoint_mode!r}")
        if int(self.global_step) < 0:
            raise ValueError("global_step must be non-negative")
        if self.reason not in {"checkpoint", "save_model"}:
            raise ValueError(f"unknown ZO checkpoint reason: {self.reason!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_version": ZO_CHECKPOINT_METADATA_VERSION,
            "checkpoint_type": self.checkpoint_type,
            "checkpoint_mode": self.checkpoint_mode,
            "global_step": int(self.global_step),
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "payload": dict(self.payload),
        }


def write_zo_checkpoint_metadata(
    output_dir: str | Path,
    metadata: ZOCheckpointMetadata,
) -> Path:
    """Write ZO checkpoint metadata into ``output_dir``."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = metadata.to_dict()
    metadata_path = path / ZO_CHECKPOINT_METADATA_NAME
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def read_zo_checkpoint_metadata(output_dir: str | Path) -> dict[str, Any]:
    """Read ZO checkpoint metadata from ``output_dir``."""

    metadata_path = Path(output_dir) / ZO_CHECKPOINT_METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"ZO checkpoint metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("ZO checkpoint metadata must be a JSON object")
    required = {
        "metadata_version",
        "checkpoint_type",
        "checkpoint_mode",
        "global_step",
        "reason",
        "metrics",
        "payload",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise KeyError(f"ZO checkpoint metadata is missing fields: {missing}")
    if int(payload["metadata_version"]) != ZO_CHECKPOINT_METADATA_VERSION:
        raise ValueError(
            "unsupported ZO checkpoint metadata version: "
            f"{payload['metadata_version']!r}"
        )
    if payload["checkpoint_mode"] not in {"metadata", "native", "lora"}:
        raise ValueError(f"unknown ZO checkpoint mode: {payload['checkpoint_mode']!r}")
    if payload["checkpoint_type"] not in {
        "zo_trainer_checkpoint",
        "zo_trainer_model",
    }:
        raise ValueError(f"unknown ZO checkpoint type: {payload['checkpoint_type']!r}")
    if int(payload["global_step"]) < 0:
        raise ValueError("ZO checkpoint global_step must be non-negative")
    if payload["reason"] not in {"checkpoint", "save_model"}:
        raise ValueError(f"unknown ZO checkpoint reason: {payload['reason']!r}")
    if not isinstance(payload["metrics"], dict) or not isinstance(
        payload["payload"], dict
    ):
        raise TypeError("ZO checkpoint metrics and payload must be JSON objects")
    return payload
