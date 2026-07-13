from __future__ import annotations

from types import SimpleNamespace

import pytest

from zo_vllm.training.checkpoint_manifest import (
    build_layer_mapping_manifest,
    validate_layer_mapping_manifest,
)


def _weight_sync(*, q_end: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        hf_to_vllm_mapping={
            "model.layers.0.self_attn.v_proj.weight": (
                "model.layers.0.self_attn.qkv_proj.weight"
            ),
            "model.layers.0.self_attn.q_proj.weight": (
                "model.layers.0.self_attn.qkv_proj.weight"
            ),
        },
        hf_to_slice={
            "model.layers.0.self_attn.q_proj.weight": (0, q_end),
            "model.layers.0.self_attn.v_proj.weight": (12, 20),
        },
    )


def test_layer_mapping_manifest_is_complete_and_deterministic() -> None:
    first = build_layer_mapping_manifest(_weight_sync())
    second = build_layer_mapping_manifest(_weight_sync())

    assert first == second
    assert first["format_version"] == 1
    assert first["hf_to_slice"][
        "model.layers.0.self_attn.q_proj.weight"
    ] == [0, 8]
    assert len(first["fingerprint"]) == 64


def test_layer_mapping_validation_rejects_slice_drift() -> None:
    recorded = build_layer_mapping_manifest(_weight_sync())
    current = build_layer_mapping_manifest(_weight_sync(q_end=7))

    with pytest.raises(ValueError, match="layer mapping does not match"):
        validate_layer_mapping_manifest(recorded, current)


def test_layer_mapping_expands_implicit_opt_qkv_slices() -> None:
    manifest = build_layer_mapping_manifest(
        SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=8),
            hf_to_vllm_mapping={
                "model.decoder.layers.0.self_attn.q_proj.weight": (
                    "model.decoder.layers.0.self_attn.qkv_proj.weight"
                ),
                "model.decoder.layers.0.self_attn.k_proj.weight": (
                    "model.decoder.layers.0.self_attn.qkv_proj.weight"
                ),
                "model.decoder.layers.0.self_attn.v_proj.weight": (
                    "model.decoder.layers.0.self_attn.qkv_proj.weight"
                ),
            },
            hf_to_slice={},
        )
    )

    assert manifest["hf_to_slice"] == {
        "model.decoder.layers.0.self_attn.q_proj.weight": [0, 8],
        "model.decoder.layers.0.self_attn.k_proj.weight": [8, 16],
        "model.decoder.layers.0.self_attn.v_proj.weight": [16, 24],
    }
