import pytest
import torch

from zo_vllm.config import ZOVLLMEngineConfig, ZOVLLMSlotConfig
from zo_vllm.engine import ZOVLLMEngine
from zo_vllm.training.update_bank_state import BlockLoRAUpdateBankState


class _FakeModelConfig:
    model_type = "opt"
    num_hidden_layers = 1
    hidden_size = 4
    ffn_dim = 8
    num_attention_heads = 1


def _direction(*, value: float, v_value: float, v_refreshed: bool = True):
    U = torch.full((3, 2), value, dtype=torch.float32)
    V = torch.full((4, 2), v_value, dtype=torch.float32)
    return {
        "layer.weight": {
            "U": U,
            "V": V,
            "V_T": V.T.contiguous(),
            "v_refreshed": v_refreshed,
        }
    }


def test_lora_bank_reuses_current_block_when_v_is_not_refreshed():
    state = BlockLoRAUpdateBankState(update_bank_rank=4)
    first = _direction(value=1.0, v_value=2.0, v_refreshed=True)
    state.prepare_for_score(first)
    assert state.max_used_rank() == 2
    first_a = state.bank_a["layer.weight"].clone()

    second = _direction(value=3.0, v_value=9.0, v_refreshed=False)
    prepared = state.prepare_for_score(second)

    assert state.max_used_rank() == 2
    assert torch.equal(state.bank_a["layer.weight"], first_a)
    assert torch.equal(
        prepared["layer.weight"]["U"][:, :2], second["layer.weight"]["U"]
    )
    assert torch.count_nonzero(prepared["layer.weight"]["U"][:, 2:]).item() == 0


def test_lora_bank_allocates_new_block_on_v_refresh():
    state = BlockLoRAUpdateBankState(update_bank_rank=4)
    state.prepare_for_score(_direction(value=1.0, v_value=2.0, v_refreshed=True))

    refreshed = _direction(value=3.0, v_value=5.0, v_refreshed=True)
    prepared = state.prepare_for_score(refreshed)

    assert state.max_used_rank() == 4
    assert torch.equal(
        state.bank_a["layer.weight"][2:4],
        refreshed["layer.weight"]["V_T"],
    )
    assert prepared["layer.weight"]["v_refreshed"] is True
    assert torch.equal(
        prepared["layer.weight"]["U"][:, 2:4], refreshed["layer.weight"]["U"]
    )


def test_lora_bank_capacity_hard_fails():
    state = BlockLoRAUpdateBankState(update_bank_rank=3)
    state.prepare_for_score(_direction(value=1.0, v_value=2.0, v_refreshed=True))

    with pytest.raises(RuntimeError, match="LoRA update bank exhausted"):
        state.prepare_for_score(_direction(value=1.0, v_value=3.0, v_refreshed=True))


def test_lora_bank_pending_flush_and_plus_minus_math():
    state = BlockLoRAUpdateBankState(
        update_bank_rank=4,
        gradient_accumulation_update_steps=2,
    )
    directions = _direction(value=2.0, v_value=1.0, v_refreshed=True)
    state.prepare_for_score(directions)
    state.apply(
        directions,
        projected_grad=3.0,
        learning_rate=0.1,
        weight_decay=0.0,
        step=1,
    )
    assert state.pending_u
    assert torch.count_nonzero(state.accumulated_u["layer.weight"]).item() == 0

    state.flush_pending_to_accumulated(step=1)
    prepared = state.prepare_for_score(
        _direction(value=4.0, v_value=7.0, v_refreshed=False)
    )["layer.weight"]
    eps = 1e-3
    plus_b = prepared["U_accum"] + eps * prepared["U"]
    minus_b = prepared["U_accum"] - eps * prepared["U"]
    bank_a = prepared["V_T"]

    expected_acc = torch.zeros((3, 4), dtype=torch.float32)
    expected_acc[:, :2] = -0.6
    expected_probe = torch.zeros((3, 4), dtype=torch.float32)
    expected_probe[:, :2] = 4.0
    assert torch.allclose(prepared["U_accum"], expected_acc)
    assert torch.allclose(prepared["U"], expected_probe)
    assert torch.allclose(
        plus_b @ bank_a, (expected_acc + eps * expected_probe) @ bank_a
    )
    assert torch.allclose(
        minus_b @ bank_a, (expected_acc - eps * expected_probe) @ bank_a
    )


def test_lora_bank_rejects_unsupported_weight_decay():
    state = BlockLoRAUpdateBankState(update_bank_rank=4)
    directions = _direction(value=2.0, v_value=1.0, v_refreshed=True)
    state.prepare_for_score(directions)

    with pytest.raises(ValueError, match="does not support weight_decay"):
        state.apply(
            directions,
            projected_grad=3.0,
            learning_rate=0.1,
            weight_decay=0.1,
            step=1,
        )


def test_lora_bank_clean_directions_for_score_returns_active_bank():
    state = BlockLoRAUpdateBankState(update_bank_rank=4)
    directions = _direction(value=2.0, v_value=1.0, v_refreshed=True)
    state.prepare_for_score(directions)
    state.apply(
        directions,
        projected_grad=3.0,
        learning_rate=0.1,
        weight_decay=0.0,
        step=1,
    )

    clean = state.clean_directions_for_score()["layer.weight"]

    assert torch.count_nonzero(clean["U"]).item() == 0
    assert torch.equal(clean["U_accum"], state.accumulated_u["layer.weight"])
    assert torch.equal(clean["V_T"], state.bank_a["layer.weight"])
    assert clean["v_refreshed"] is True


def test_engine_decouples_direction_rank_from_lora_slot_rank():
    engine = ZOVLLMEngine(
        model="fake",
        rank=2,
        config=ZOVLLMEngineConfig(lora_rank=4),
        llm=object(),
        model_config=_FakeModelConfig(),
    )

    assert engine.rank == 2
    assert engine.lora_rank == 4
    assert engine.runtime.rank == 4


def test_engine_uses_runtime_config_defaults():
    engine = ZOVLLMEngine(
        model="fake",
        rank=2,
        config=ZOVLLMEngineConfig(
            lora_rank=4,
            target_modules="q_proj,v_proj",
            slot=ZOVLLMSlotConfig(plus_id=21, minus_id=22, max_loras=3),
        ),
        llm=object(),
        model_config=_FakeModelConfig(),
    )

    assert engine.config.slot.plus_id == 21
    assert engine.config.slot.minus_id == 22
    assert engine.config.slot.max_loras == 3
    assert engine.lora_rank == 4
    assert engine.plus_id == 21
    assert engine.minus_id == 22
    assert engine.target_modules == ["q_proj", "v_proj"]


def test_engine_uses_config_as_the_only_runtime_knob_source():
    engine = ZOVLLMEngine(
        model="fake",
        rank=2,
        config=ZOVLLMEngineConfig(
            lora_rank=6,
            target_modules=("k_proj", "v_proj"),
            slot=ZOVLLMSlotConfig(plus_id=31, minus_id=32, max_loras=4),
        ),
        llm=object(),
        model_config=_FakeModelConfig(),
    )

    assert engine.config.slot.plus_id == 31
    assert engine.config.slot.minus_id == 32
    assert engine.config.slot.max_loras == 4
    assert engine.lora_rank == 6
    assert engine.plus_id == 31
    assert engine.minus_id == 32
    assert engine.target_modules == ["k_proj", "v_proj"]
