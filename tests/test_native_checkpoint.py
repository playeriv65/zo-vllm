from __future__ import annotations

import pytest
import torch

from zo_vllm.training.native_checkpoint import (
    _apply_effective_direction_to_target_,
    _save_tensor_parts,
    normalize_native_checkpoint_key,
    normalize_native_checkpoint_state,
)


def test_tensor_parts_respect_shard_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ZO_CHECKPOINT_MAX_CPU_SHARD_BYTES", "16")
    saved = []

    def save_file(part, path):
        saved.append((path, list(part)))

    result = _save_tensor_parts(
        {
            "a": torch.zeros(4),
            "b": torch.zeros(4),
            "c": torch.zeros(1),
        },
        checkpoint_path=str(tmp_path),
        rank=2,
        save_file=save_file,
    )

    assert saved == [
        (str(tmp_path / "model-rank-2-part-0.safetensors"), ["a"]),
        (str(tmp_path / "model-rank-2-part-1.safetensors"), ["b"]),
        (str(tmp_path / "model-rank-2-part-2.safetensors"), ["c"]),
    ]
    assert result["num_parts"] == 3


def test_normalizes_runtime_lora_wrapper_keys() -> None:
    assert normalize_native_checkpoint_key("lm_head.base_layer.weight") == (
        "lm_head.weight"
    )
    assert (
        normalize_native_checkpoint_key(
            "model.layers.0.mlp.linear_method.base_layer.weight"
        )
        == "model.layers.0.mlp.weight"
    )
    assert normalize_native_checkpoint_key("model.norm.weight") == "model.norm.weight"


def test_normalized_state_keeps_shared_tensor_once() -> None:
    tensor = torch.zeros(2, 3)
    state = normalize_native_checkpoint_state(
        {
            "lm_head.weight": tensor,
            "lm_head.base_layer.weight": tensor,
        }
    )

    assert state == {"lm_head.weight": tensor}


def test_normalized_state_rejects_distinct_tensor_collision() -> None:
    with pytest.raises(RuntimeError, match="normalization collision"):
        normalize_native_checkpoint_state(
            {
                "lm_head.weight": torch.zeros(2, 3),
                "lm_head.base_layer.weight": torch.ones(2, 3),
            }
        )


def test_materializes_tied_embedding_direction_in_vocab_first_layout() -> None:
    target = torch.zeros((5, 3), dtype=torch.float32)
    accumulated_u = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    v = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    _apply_effective_direction_to_target_(
        target,
        hf_name="model.embed_tokens.weight",
        raw_direction={
            "U": torch.zeros_like(accumulated_u),
            "U_accum": accumulated_u,
            "V": v,
        },
        precision="float32",
    )

    assert torch.equal(target, v @ accumulated_u.T)
