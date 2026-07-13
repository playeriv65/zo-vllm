import json
from types import SimpleNamespace

import pytest

from zo_vllm.training.artifacts import (
    inspect_training_artifact,
    native_checkpoint_model_kwargs,
    require_native_checkpoint,
    validate_checkpoint_layer_mapping,
)
from zo_vllm.training.checkpoint_manifest import build_layer_mapping_manifest


def test_inspects_hf_native_checkpoint(tmp_path) -> None:
    weight_sync = SimpleNamespace(
        hf_to_vllm_mapping={"layer.q_proj.weight": "layer.qkv_proj.weight"},
        hf_to_slice={"layer.q_proj.weight": (0, 8)},
    )
    layer_mapping = build_layer_mapping_manifest(weight_sync)
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    (checkpoint / "model-rank-0-part-0.safetensors").touch()
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "zo_direction_state.pt").touch()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_name": "facebook/opt-125m",
                "target_modules": ["q_proj", "lm_head", "embed_tokens"],
                "rank": 8,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "zo_checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "checkpoint_type": "zo_trainer_checkpoint",
                "checkpoint_mode": "native",
                "global_step": 12,
                "reason": "checkpoint",
                "metrics": {},
                "payload": {
                    "mode": "native",
                    "loadable": True,
                    "runtime_manifest": {
                        "target_modules": ["q_proj", "lm_head", "embed_tokens"],
                        "layer_mapping": layer_mapping,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = require_native_checkpoint(checkpoint)

    assert artifact.kind == "native_checkpoint"
    assert artifact.format_name == "hf_zo_checkpoint_v1"
    assert artifact.step == 12
    assert artifact.payload_path == checkpoint
    assert artifact.trainer_state_path == checkpoint / "trainer_state.json"
    assert artifact.direction_state_path == checkpoint / "zo_direction_state.pt"
    assert artifact.runtime_manifest["target_modules"] == [
        "q_proj",
        "lm_head",
        "embed_tokens",
    ]
    assert native_checkpoint_model_kwargs(artifact) == {
        "load_format": "sharded_state",
        "model_weights": str(checkpoint),
    }
    validate_checkpoint_layer_mapping(artifact, weight_sync)


def test_rejects_legacy_checkpoint_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-0020500"
    checkpoint.mkdir()
    payload = checkpoint / "zo_lora_bank.pt"
    payload.touch()
    metadata = checkpoint / "zo_checkpoint_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "checkpoint_type": "vllm_zo_lora",
                "step": 20500,
                "raw_step": 20501,
                "loadable": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported checkpoint metadata format"):
        inspect_training_artifact(checkpoint)


def test_native_metadata_without_weight_shards_is_not_loadable(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    (checkpoint / "zo_checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "checkpoint_type": "zo_trainer_checkpoint",
                "checkpoint_mode": "native",
                "global_step": 3,
                "reason": "checkpoint",
                "metrics": {},
                "payload": {"mode": "native", "loadable": True},
            }
        ),
        encoding="utf-8",
    )

    artifact = inspect_training_artifact(checkpoint)

    assert artifact.kind == "native_checkpoint"
    assert not artifact.loadable
    assert artifact.payload_path is None
    with pytest.raises(ValueError, match="requires a loadable native"):
        require_native_checkpoint(checkpoint)


def test_inspects_u_snapshot_and_v_cache_by_explicit_schema_name(tmp_path) -> None:
    snapshot = tmp_path / "u_step_0000500.pt"
    snapshot.touch()
    cache = tmp_path / "fullkappa_agzo_v_cache.pt"
    cache.touch()

    snapshot_artifact = inspect_training_artifact(snapshot)
    cache_artifact = inspect_training_artifact(cache)

    assert snapshot_artifact.kind == "u_snapshot"
    assert snapshot_artifact.step == 500
    assert cache_artifact.kind == "v_cache"
