"""Read-only discovery for ZO training and study artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .checkpoint_manifest import (
    build_layer_mapping_manifest,
    validate_layer_mapping_manifest,
)


@dataclass(frozen=True)
class TrainingArtifact:
    """Normalized description of one persisted training or study artifact."""

    path: Path
    kind: str
    format_name: str
    step: int | None = None
    raw_step: int | None = None
    loadable: bool = False
    payload_path: Path | None = None
    metadata_path: Path | None = None
    trainer_state_path: Path | None = None
    direction_state_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def inspect_training_artifact(path: str | Path) -> TrainingArtifact:
    """Inspect a checkpoint, U snapshot, or V cache without loading tensors."""

    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"training artifact not found: {artifact_path}")
    if artifact_path.is_file():
        return _inspect_file(artifact_path)
    return _inspect_directory(artifact_path)


def require_native_checkpoint(path: str | Path) -> TrainingArtifact:
    """Resolve a study input that requires materialized native model weights."""

    artifact = inspect_training_artifact(path)
    if (
        artifact.kind != "native_checkpoint"
        or not artifact.loadable
        or artifact.payload_path is None
    ):
        raise ValueError(
            "study requires a loadable native sharded checkpoint, got "
            f"{artifact.kind!r} ({artifact.format_name}) at {artifact.path}"
        )
    return artifact


def native_checkpoint_model_kwargs(
    path: str | Path | TrainingArtifact,
) -> dict[str, Any]:
    """Build vLLM model-loader kwargs from a typed native checkpoint."""

    artifact = (
        path if isinstance(path, TrainingArtifact) else require_native_checkpoint(path)
    )
    if artifact.kind != "native_checkpoint" or artifact.payload_path is None:
        raise ValueError("native checkpoint model kwargs require native weights")
    kwargs: dict[str, Any] = {
        "load_format": "sharded_state",
        "model_weights": str(artifact.payload_path),
    }
    return kwargs


def validate_checkpoint_layer_mapping(
    artifact: TrainingArtifact,
    weight_sync: Any,
) -> None:
    """Validate a checkpoint before using HF-to-vLLM training semantics."""

    recorded = artifact.runtime_manifest.get("layer_mapping")
    if not isinstance(recorded, dict):
        raise ValueError("native checkpoint runtime manifest is missing layer_mapping")
    validate_layer_mapping_manifest(
        recorded,
        build_layer_mapping_manifest(weight_sync),
    )


def _inspect_file(path: Path) -> TrainingArtifact:
    if path.name == "zo_lora_bank.pt":
        return TrainingArtifact(
            path=path,
            kind="lora_bank_checkpoint",
            format_name="vllm_zo_lora_bank",
            step=_step_from_parent(path),
            raw_step=_step_from_parent(path),
            loadable=True,
            payload_path=path,
            metadata_path=_first_existing(path.parent, "zo_checkpoint_metadata.json"),
            trainer_state_path=_trainer_state_path(path.parent),
        )
    if path.name.startswith("u_step_") and path.suffix == ".pt":
        return TrainingArtifact(
            path=path,
            kind="u_snapshot",
            format_name="zo_u_snapshot",
            step=_step_from_name(path.name),
            raw_step=_step_from_name(path.name),
            loadable=False,
            payload_path=path,
        )
    if path.name == "fullkappa_agzo_v_cache.pt":
        return TrainingArtifact(
            path=path,
            kind="v_cache",
            format_name="fullkappa_agzo_v_cache",
            loadable=False,
            payload_path=path,
            metadata_path=_first_existing(path.parent, "metadata.json"),
        )
    if path.name == "zo_checkpoint_metadata.json":
        return _inspect_metadata_file(path)
    raise ValueError(f"unrecognized training artifact file: {path}")


def _inspect_directory(path: Path) -> TrainingArtifact:
    hf_metadata = path / "zo_checkpoint_metadata.json"
    if hf_metadata.is_file():
        return _inspect_metadata_file(hf_metadata)
    lora_payload = path / "zo_lora_bank.pt"
    if lora_payload.is_file():
        return _inspect_file(lora_payload)
    if list(path.glob("*.safetensors")):
        return TrainingArtifact(
            path=path,
            kind="native_checkpoint",
            format_name="vllm_sharded_state",
            step=_step_from_name(path.name),
            raw_step=_step_from_name(path.name),
            loadable=True,
            payload_path=path,
            metadata_path=None,
            trainer_state_path=_trainer_state_path(path),
            direction_state_path=_first_existing(path, "zo_direction_state.pt"),
        )
    v_cache = path / "fullkappa_agzo_v_cache.pt"
    if v_cache.is_file():
        return _inspect_file(v_cache)
    raise ValueError(f"unrecognized training artifact directory: {path}")


def _inspect_metadata_file(path: Path) -> TrainingArtifact:
    metadata = _read_json(path)
    if int(metadata.get("metadata_version", 0) or 0) > 0:
        mode = str(metadata.get("checkpoint_mode", ""))
        payload = metadata.get("payload", {})
        if not isinstance(payload, dict):
            raise TypeError(f"checkpoint payload must be a mapping: {path}")
        kind = {
            "native": "native_checkpoint",
            "lora": "lora_bank_checkpoint",
            "metadata": "metadata",
        }.get(mode)
        if kind is None:
            raise ValueError(f"unknown checkpoint mode {mode!r}: {path}")
        root = path.parent
        payload_path = _local_payload(root, kind)
        return TrainingArtifact(
            path=root,
            kind=kind,
            format_name="hf_zo_checkpoint_v1",
            step=_optional_int(metadata.get("global_step")),
            raw_step=_optional_int(payload.get("raw_step", metadata.get("global_step"))),
            loadable=bool(payload.get("loadable", mode != "metadata"))
            and (kind != "native_checkpoint" or payload_path is not None),
            payload_path=payload_path,
            metadata_path=path,
            trainer_state_path=_trainer_state_path(root),
            direction_state_path=_first_existing(root, "zo_direction_state.pt"),
            metadata=metadata,
            runtime_manifest=_runtime_manifest(root, metadata),
        )

    raise ValueError(f"unsupported checkpoint metadata format: {path}")


def _local_payload(root: Path, kind: str) -> Path | None:
    if kind == "lora_bank_checkpoint":
        path = root / "zo_lora_bank.pt"
        return path if path.is_file() else None
    if kind == "native_checkpoint" and list(root.glob("*.safetensors")):
        return root
    return None


def _runtime_manifest(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("payload", {})
    if isinstance(payload, dict) and isinstance(payload.get("runtime_manifest"), dict):
        return dict(payload["runtime_manifest"])
    return {}


def _trainer_state_path(root: Path) -> Path | None:
    return _first_existing(root, "trainer_state.json", "zo_trainer_state.json")


def _first_existing(root: Path, *names: str) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"metadata must be a JSON object: {path}")
    return payload


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _step_from_parent(path: Path) -> int | None:
    return _step_from_name(path.parent.name)


def _step_from_name(name: str) -> int | None:
    match = re.search(r"(?:checkpoint-|step_)(\d+)", name)
    return None if match is None else int(match.group(1))


__all__ = [
    "TrainingArtifact",
    "inspect_training_artifact",
    "native_checkpoint_model_kwargs",
    "require_native_checkpoint",
    "validate_checkpoint_layer_mapping",
]
