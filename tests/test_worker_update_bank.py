from types import SimpleNamespace

import pytest
import torch

from zo_vllm.training import worker_update_bank


class _FakeModel:
    def __init__(self, modules=None, packed_modules=None):
        self.lora_manager = SimpleNamespace(
            device=torch.device("cpu"),
            modules=modules or {},
            packed_modules=packed_modules or {},
        )


class _FakeLoRAModule:
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=4,
        rank=2,
        max_loras=2,
        slices=1,
    ):
        self.lora_a_stacked = [
            torch.empty(max_loras, 1, rank, input_dim, dtype=torch.float32)
            for _ in range(slices)
        ]
        self.lora_b_stacked = [
            torch.empty(max_loras, 1, output_dim, rank, dtype=torch.float32)
            for _ in range(slices)
        ]


class _FakeLogitsLoRAModule:
    def __init__(self, *, input_dim=3, output_dim=8, rank=2, max_loras=2):
        self.lora_a_stacked = torch.empty(
            max_loras,
            1,
            rank,
            input_dim,
            dtype=torch.float32,
        )
        self.lora_b_stacked = torch.empty(
            max_loras,
            1,
            output_dim,
            rank,
            dtype=torch.float32,
        )


class _FakeEmbeddingLoRAModule:
    def __init__(self, *, vocab_size=8, output_dim=3, rank=2, max_loras=2):
        self.lora_a_stacked = torch.empty(
            max_loras,
            vocab_size,
            rank,
            dtype=torch.float32,
        )
        self.lora_b_stacked = torch.empty(
            max_loras,
            1,
            output_dim,
            rank,
            dtype=torch.float32,
        )


def _metadata_specs():
    return [
        {
            "name": "model.layers.0.self_attn.q_proj.weight",
            "shape": [4, 3],
        }
    ]


def _lozo_config():
    return {
        "rank": 2,
        "eps": 1e-3,
        "nu": 10,
        "seed": 7,
        "random_device": "cpu",
        "direction_dtype": "float32",
        "direction_sampling": "exact",
        "direction_scale": 1.0,
        "v_normalization": "none",
    }


def _bank_config():
    return {
        "update_bank_rank": 4,
        "u_beta": 1.0,
        "u_norm_cap": None,
        "gradient_accumulation_update_steps": 0,
    }


def test_worker_bank_prepare_apply_and_clean(monkeypatch):
    calls = []

    def fake_write(
        model, *, plus_id, minus_id, directions_2d, eps, copy_stream="default"
    ):
        del model
        calls.append(
            {
                "plus_id": plus_id,
                "minus_id": minus_id,
                "eps": eps,
                "copy_stream": copy_stream,
                "directions": directions_2d,
            }
        )
        return {
            "plus": {"lora_id": plus_id, "slot_index": 0},
            "minus": {"lora_id": minus_id, "slot_index": 1},
            "modules_written": len(directions_2d),
            "packed_written": 0,
            "missing": [],
            "profile_s": {"total_worker": 0.0},
        }

    monkeypatch.setattr(
        worker_update_bank,
        "_update_lora_slots_from_directions_in_vllm_model",
        fake_write,
    )

    model = _FakeModel()
    state_key = worker_update_bank.worker_update_bank_key(plus_id=11, minus_id=12)
    init_info = worker_update_bank.init_worker_update_bank(
        model,
        state_key=state_key,
        metadata_specs=_metadata_specs(),
        lozo_config=_lozo_config(),
        bank_config=_bank_config(),
        plus_id=11,
        minus_id=12,
    )

    assert init_info["source"] == "worker_update_bank"
    assert init_info["num_modules"] == 1

    prepare_info = worker_update_bank.prepare_worker_update_bank_slots(
        model,
        state_key=state_key,
        step=1,
        eps=1e-3,
    )

    assert prepare_info["source"] == "worker_update_bank"
    assert prepare_info["direction_refreshed"] is True
    assert calls[-1]["plus_id"] == 11
    assert calls[-1]["minus_id"] == 12
    prepared = calls[-1]["directions"]["model.layers.0.self_attn.q_proj.weight"]
    assert prepared["U"].shape == (4, 4)
    assert prepared["U_accum"].shape == (4, 4)
    assert prepared["V_T"].shape == (4, 3)

    apply_info = worker_update_bank.apply_worker_update_bank_update(
        model,
        state_key=state_key,
        step=1,
        projected_grad=2.0,
        learning_rate=0.1,
        weight_decay=0.0,
    )

    assert apply_info["source"] == "worker_update_bank"
    assert apply_info["update_info"]["mode"] == "lora_bank"
    assert apply_info["update_info"]["used_bank_rank"] == 2

    clean_info = worker_update_bank.write_clean_worker_update_bank(
        model,
        state_key=state_key,
        step=1,
    )

    assert clean_info["has_update"] is True
    assert calls[-1]["eps"] == 0.0
    clean = calls[-1]["directions"]["model.layers.0.self_attn.q_proj.weight"]
    assert torch.count_nonzero(clean["U"]).item() == 0
    assert torch.count_nonzero(clean["U_accum"]).item() > 0


def test_worker_bank_rejects_apply_without_matching_prepare():
    model = _FakeModel()
    state_key = worker_update_bank.worker_update_bank_key(plus_id=11, minus_id=12)
    worker_update_bank.init_worker_update_bank(
        model,
        state_key=state_key,
        metadata_specs=_metadata_specs(),
        lozo_config=_lozo_config(),
        bank_config=_bank_config(),
        plus_id=11,
        minus_id=12,
    )

    with pytest.raises(RuntimeError, match="before matching prepare"):
        worker_update_bank.apply_worker_update_bank_update(
            model,
            state_key=state_key,
            step=1,
            projected_grad=1.0,
            learning_rate=0.1,
            weight_decay=0.0,
        )


def test_worker_bank_validates_lora_manager_metadata_match():
    model = _FakeModel(
        modules={
            "model.layers.0.self_attn.q_proj": _FakeLoRAModule(
                input_dim=3,
                output_dim=4,
            )
        }
    )
    state_key = worker_update_bank.worker_update_bank_key(plus_id=11, minus_id=12)
    init_info = worker_update_bank.init_worker_update_bank(
        model,
        state_key=state_key,
        metadata_specs=_metadata_specs(),
        lozo_config=_lozo_config(),
        bank_config=_bank_config(),
        plus_id=11,
        minus_id=12,
    )

    assert init_info["structure_info"]["modules_expected"] == 1
    assert init_info["structure_info"]["keys_consumed"] == 1


def test_worker_bank_rejects_missing_lora_manager_module():
    model = _FakeModel(
        modules={
            "model.layers.0.self_attn.q_proj": _FakeLoRAModule(),
            "lm_head": _FakeLoRAModule(input_dim=3, output_dim=8),
        }
    )
    state_key = worker_update_bank.worker_update_bank_key(plus_id=11, minus_id=12)

    with pytest.raises(RuntimeError, match="missing LoRA direction metadata"):
        worker_update_bank.init_worker_update_bank(
            model,
            state_key=state_key,
            metadata_specs=_metadata_specs(),
            lozo_config=_lozo_config(),
            bank_config=_bank_config(),
            plus_id=11,
            minus_id=12,
        )


def test_worker_bank_rejects_unexpected_metadata_module():
    model = _FakeModel(
        modules={
            "model.layers.0.self_attn.q_proj": _FakeLoRAModule(),
        }
    )
    state_key = worker_update_bank.worker_update_bank_key(plus_id=11, minus_id=12)

    with pytest.raises(RuntimeError, match="unexpected LoRA direction metadata"):
        worker_update_bank.init_worker_update_bank(
            model,
            state_key=state_key,
            metadata_specs=[
                *_metadata_specs(),
                {"name": "lm_head.weight", "shape": [8, 3]},
            ],
            lozo_config=_lozo_config(),
            bank_config=_bank_config(),
            plus_id=11,
            minus_id=12,
        )


def test_worker_bank_validates_packed_replacement_metadata():
    model = _FakeModel(
        modules={
            "model.layers.0.self_attn.qkv_proj": _FakeLoRAModule(
                input_dim=3,
                output_dim=4,
                slices=3,
            )
        },
        packed_modules={
            "model.layers.0.self_attn.qkv_proj": [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ]
        },
    )
    state_key = worker_update_bank.worker_update_bank_key(plus_id=11, minus_id=12)
    init_info = worker_update_bank.init_worker_update_bank(
        model,
        state_key=state_key,
        metadata_specs=[
            {"name": "model.layers.0.self_attn.q_proj.weight", "shape": [4, 3]},
            {"name": "model.layers.0.self_attn.k_proj.weight", "shape": [4, 3]},
            {"name": "model.layers.0.self_attn.v_proj.weight", "shape": [4, 3]},
        ],
        lozo_config=_lozo_config(),
        bank_config=_bank_config(),
        plus_id=11,
        minus_id=12,
    )

    assert init_info["structure_info"]["modules_expected"] == 1
    assert init_info["structure_info"]["keys_consumed"] == 3


def test_worker_bank_validates_tensor_stacked_lm_head_and_embedding_shapes():
    model = _FakeModel(
        modules={
            "lm_head": _FakeLogitsLoRAModule(input_dim=3, output_dim=8),
            "model.embed_tokens": _FakeEmbeddingLoRAModule(
                vocab_size=8,
                output_dim=3,
            ),
        }
    )
    state_key = worker_update_bank.worker_update_bank_key(plus_id=11, minus_id=12)
    init_info = worker_update_bank.init_worker_update_bank(
        model,
        state_key=state_key,
        metadata_specs=[
            {"name": "lm_head.weight", "shape": [8, 3]},
            {"name": "model.embed_tokens.weight", "shape": [3, 8]},
        ],
        lozo_config=_lozo_config(),
        bank_config=_bank_config(),
        plus_id=11,
        minus_id=12,
    )

    assert init_info["structure_info"]["keys_consumed"] == 2
