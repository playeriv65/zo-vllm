from types import SimpleNamespace

import torch

from zo_vllm.core.weight_sync import WeightSync


def test_weight_sync_uses_qwen3_model_config_for_mapping():
    model_config = SimpleNamespace(
        model_type="qwen3",
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )

    sync = WeightSync(object(), num_layers=1, model_config=model_config)

    q_name = "model.layers.0.self_attn.q_proj.weight"
    k_name = "model.layers.0.self_attn.k_proj.weight"
    gate_name = "model.layers.0.mlp.gate_proj.weight"

    assert sync.model_type == "qwen3"
    assert "model.layers.0.self_attn.qkv_proj.weight" not in sync.hf_to_vllm_mapping
    assert "model.layers.0.mlp.gate_up_proj.weight" not in sync.hf_to_vllm_mapping
    assert sync.hf_to_vllm_mapping[q_name] == "model.layers.0.self_attn.qkv_proj.weight"
    assert sync.hf_to_slice[q_name] == (0, 8)
    assert sync.hf_to_slice[k_name] == (8, 12)
    assert (
        sync.hf_to_vllm_mapping[gate_name] == "model.layers.0.mlp.gate_up_proj.weight"
    )
    assert sync.hf_to_slice[gate_name] == (0, 16)


def test_weight_sync_uses_qwen2_model_layers_mapping():
    model_config = SimpleNamespace(
        model_type="qwen2",
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )

    sync = WeightSync(object(), num_layers=1, model_config=model_config)

    q_name = "model.layers.0.self_attn.q_proj.weight"
    o_name = "model.layers.0.self_attn.o_proj.weight"
    down_name = "model.layers.0.mlp.down_proj.weight"

    assert sync.model_type == "qwen2"
    assert sync.hf_to_vllm_mapping[q_name] == "model.layers.0.self_attn.qkv_proj.weight"
    assert sync.hf_to_vllm_mapping[o_name] == o_name
    assert sync.hf_to_vllm_mapping[down_name] == down_name
    assert sync.hf_to_slice[q_name] == (0, 8)


class _QuantizedLayer:
    def __init__(self):
        self.qweight = torch.empty((1, 1), device="cpu", dtype=torch.int32)


class _FakeModel:
    def get_submodule(self, path):
        return _QuantizedLayer()


class _FakeWorker:
    def __init__(self):
        self.model_runner = SimpleNamespace(model=_FakeModel())


class _FakeLLM:
    def collective_rpc(self, fn):
        return [fn(_FakeWorker())]


def test_weight_sync_infers_opt_metadata_for_quantized_layers_without_weight():
    model_config = SimpleNamespace(
        model_type="opt",
        hidden_size=8,
        ffn_dim=32,
    )
    sync = WeightSync(_FakeLLM(), num_layers=1, model_config=model_config)

    metadata = sync.get_hf_param_metadata()

    assert metadata["model.decoder.layers.0.self_attn.q_proj.weight"].shape == (8, 8)
    assert metadata["model.decoder.layers.0.self_attn.k_proj.weight"].shape == (8, 8)
    assert metadata["model.decoder.layers.0.self_attn.v_proj.weight"].shape == (8, 8)
    assert metadata["model.decoder.layers.0.self_attn.out_proj.weight"].shape == (8, 8)
    assert metadata["model.decoder.layers.0.fc1.weight"].shape == (32, 8)
    assert metadata["model.decoder.layers.0.fc2.weight"].shape == (8, 32)
    assert metadata["model.decoder.layers.0.fc1.weight"].dtype is torch.float16
    assert metadata["model.decoder.layers.0.fc1.weight"].device == torch.device("cpu")
