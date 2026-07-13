import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from zo_vllm.serving.compact_nll_scorer import (
    QueuedCompactNLLSubmitter,
    ScheduledCompactNLLScorer,
)
from zo_vllm.serving.async_engine_service import _ScoreAdmissionController
import zo_vllm.serving.async_engine_service as async_engine_service_module
from zo_vllm.serving.zo_training_api import ServingZOStartRequest
import zo_vllm.serving.zo_training_api as zo_training_api_module
from zo_vllm.core.lora_scope import (
    ceil_update_bank_rank_to_vllm_lora_rank,
    resolve_lora_target_modules,
    resolve_update_bank_rank,
)
from zo_vllm.core.lora_runtime import LoRAUpdateRuntime
from zo_vllm.training.model_metadata import build_lora_param_metadata_from_config
from zo_vllm.training.model_metadata import estimate_lora_bank_state_memory_bytes
from zo_vllm.training.model_metadata import resolve_direction_dtype
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
from zo_vllm.experiment.runners.serving_load import ServingLoadConfig


def _causal_labels(token_groups):
    return [[-100, *list(token_ids[1:])] for token_ids in token_groups]


def test_causal_label_helper_masks_first_token():
    assert _causal_labels([[1, 2, 3], [4, 5]]) == [
        [-100, 2, 3],
        [-100, 5],
    ]


def test_auto_update_bank_rank_estimates_finite_training_horizon():
    resolved, auto = resolve_update_bank_rank(
        "auto",
        rank=8,
        steps=300,
        nu=50,
    )

    assert auto is True
    assert resolved == 64


def test_auto_update_bank_rank_rounds_up_to_vllm_supported_rank():
    resolved, auto = resolve_update_bank_rank(
        "auto",
        rank=8,
        steps=100,
        nu=50,
    )

    assert auto is True
    assert resolved == 32


def test_auto_update_bank_rank_uses_one_block_when_nu_covers_horizon():
    resolved, auto = resolve_update_bank_rank(
        "auto",
        rank=8,
        steps=100,
        nu=1000,
    )

    assert auto is True
    assert resolved == 8


def test_auto_update_bank_rank_treats_negative_one_nu_as_infinite():
    resolved, auto = resolve_update_bank_rank(
        "auto",
        rank=8,
        steps=100,
        nu=-1,
    )

    assert auto is True
    assert resolved == 8


def test_update_bank_rank_ceil_matches_vllm_supported_ranks():
    assert ceil_update_bank_rank_to_vllm_lora_rank(24) == 32
    assert ceil_update_bank_rank_to_vllm_lora_rank(56) == 64
    assert ceil_update_bank_rank_to_vllm_lora_rank(320) == 320
    assert ceil_update_bank_rank_to_vllm_lora_rank(321) == 512
    with pytest.raises(ValueError, match="largest vLLM-supported"):
        ceil_update_bank_rank_to_vllm_lora_rank(1025)


def test_manual_update_bank_rank_overrides_auto_estimate():
    resolved, auto = resolve_update_bank_rank(
        1024,
        rank=8,
        steps=300,
        nu=50,
    )

    assert auto is False
    assert resolved == 1024


def test_manual_update_bank_rank_rounds_up_to_vllm_supported_rank():
    resolved, auto = resolve_update_bank_rank(
        24,
        rank=8,
        steps=300,
        nu=50,
    )

    assert auto is False
    assert resolved == 32


def test_auto_update_bank_rank_rejects_open_ended_runs():
    with pytest.raises(ValueError, match="finite positive step horizon"):
        resolve_update_bank_rank(
            "auto",
            rank=8,
            steps=0,
            nu=50,
        )


def test_manual_update_bank_rank_allows_open_ended_runs():
    resolved, auto = resolve_update_bank_rank(
        128,
        rank=8,
        steps=0,
        nu=50,
    )

    assert auto is False
    assert resolved == 128


def test_serving_load_config_defaults_update_bank_rank_to_auto(monkeypatch):
    monkeypatch.delenv("UPDATE_BANK_RANK", raising=False)
    monkeypatch.setenv("VLLM_ZO_RESERVED_LORA_BANK_BYTES", "1234")
    monkeypatch.setenv("TRAIN_STEPS", "300")
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("NU", "50")

    config = ServingLoadConfig("server")

    assert config.update_bank_rank_auto is True
    assert config.update_bank_rank_requested == "auto"
    assert config.update_bank_rank == "64"
    assert config.server_env()["VLLM_ZO_RESERVED_LORA_BANK_BYTES"] == "1234"


def test_serving_load_config_manual_update_bank_rank(monkeypatch):
    monkeypatch.setenv("UPDATE_BANK_RANK", "512")
    monkeypatch.setenv("VLLM_ZO_RESERVED_LORA_BANK_BYTES", "5678")
    monkeypatch.setenv("TRAIN_STEPS", "300")
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("NU", "50")

    config = ServingLoadConfig("server")

    assert config.update_bank_rank_auto is False
    assert config.update_bank_rank == "512"


def test_serving_server_forwards_scheduler_capacity(monkeypatch):
    monkeypatch.setenv("UPDATE_BANK_RANK", "16")
    monkeypatch.setenv("VLLM_ZO_RESERVED_LORA_BANK_BYTES", "5678")
    monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "16384")
    monkeypatch.setenv("MAX_NUM_SEQS", "32")
    monkeypatch.setenv("ENABLE_PREFIX_CACHING", "0")

    cmd = ServingLoadConfig("server").server_cmd()

    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "16384"
    assert cmd[cmd.index("--max-num-seqs") + 1] == "32"
    assert "--no-enable-prefix-caching" in cmd


def test_serving_load_config_keeps_unset_optional_values_null(monkeypatch):
    monkeypatch.setenv("UPDATE_BANK_RANK", "64")
    monkeypatch.setenv("VLLM_ZO_RESERVED_LORA_BANK_BYTES", "5678")
    monkeypatch.delenv("DATA_SEED", raising=False)
    monkeypatch.delenv("U_NORM_CAP", raising=False)

    payload = ServingLoadConfig("server").zo_payload()

    assert payload["data_seed"] is None
    assert payload["u_norm_cap"] is None
    assert "task_shuffle_impl" not in payload
    assert "num_eval" not in payload


def test_serving_load_config_forwards_worker_bank_algorithm_settings(monkeypatch):
    monkeypatch.setenv("UPDATE_BANK_RANK", "64")
    monkeypatch.setenv("VLLM_ZO_RESERVED_LORA_BANK_BYTES", "5678")
    monkeypatch.setenv("DATA_SEED", "9")
    monkeypatch.setenv("GRADIENT_ACCUMULATION_UPDATE_STEPS", "25")
    monkeypatch.setenv("U_BETA", "0.75")
    monkeypatch.setenv("U_NORM_CAP", "3.5")
    monkeypatch.setenv("ZO_RANDOM_DEVICE", "cpu")
    monkeypatch.setenv("DIRECTION_DEVICE", "cpu")
    monkeypatch.setenv("DIRECTION_SAMPLING", "flat")
    monkeypatch.setenv("DIRECTION_SCALE", "2.0")
    monkeypatch.setenv("PERTURBATION_NORMALIZATION", "none")
    monkeypatch.setenv("V_NORMALIZATION", "unit")
    monkeypatch.setenv("WANDB_PROJECT", "phase7-test")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")

    payload = ServingLoadConfig("server").zo_payload()

    assert payload["data_seed"] == 9
    assert payload["gradient_accumulation_update_steps"] == 25
    assert payload["u_beta"] == 0.75
    assert payload["u_norm_cap"] == 3.5
    assert payload["random_device"] == "cpu"
    assert payload["direction_device"] == "cpu"
    assert payload["direction_sampling"] == "flat"
    assert payload["direction_scale"] == 2.0
    assert payload["perturbation_normalization"] == "none"
    assert payload["v_normalization"] == "unit"
    assert payload["wandb_project"] == "phase7-test"
    assert payload["wandb_entity"] == "test-entity"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("random_device", "tpu"),
        ("direction_device", "xpu"),
        ("direction_sampling", "approximate"),
        ("perturbation_normalization", "layerwise"),
        ("v_normalization", "rms"),
        ("direction_scale", -1.0),
    ],
)
def test_serving_request_rejects_invalid_algorithm_settings(field, value):
    with pytest.raises(ValueError):
        ServingZOStartRequest(**{field: value})


class _Qwen3Config:
    model_type = "qwen3"
    num_hidden_layers = 2
    hidden_size = 16
    intermediate_size = 32
    vocab_size = 128
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 4
    torch_dtype = torch.bfloat16
    tie_word_embeddings = False


class _Qwen2Config(_Qwen3Config):
    model_type = "qwen2"


class _OPTConfig:
    model_type = "opt"
    num_hidden_layers = 2
    hidden_size = 16
    ffn_dim = 64
    vocab_size = 128
    word_embed_proj_dim = 16
    torch_dtype = torch.float16
    tie_word_embeddings = True


def test_serving_zo_qwen3_metadata_shapes():
    metadata = build_lora_param_metadata_from_config(
        _Qwen3Config(),
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert metadata["model.layers.0.self_attn.q_proj.weight"].shape == (16, 16)
    assert metadata["model.layers.0.self_attn.k_proj.weight"].shape == (8, 16)
    assert metadata["model.layers.0.mlp.down_proj.weight"].shape == (16, 32)
    assert len(metadata) == 14
    assert "lm_head.weight" not in metadata
    assert "model.embed_tokens.weight" not in metadata


def test_lora_bank_state_memory_estimate_counts_worker_bank_tensors():
    metadata = build_lora_param_metadata_from_config(
        _Qwen3Config(),
        device=torch.device("cpu"),
        dtype=torch.float16,
        target_modules=["q_proj"],
    )

    estimated = estimate_lora_bank_state_memory_bytes(
        metadata,
        update_bank_rank=8,
        include_probe_u=True,
    )

    # Two q_proj layers, each shape (out=16, in=16). The worker bank keeps
    # bank_a plus accumulated_u, zero_u, and the prepare-time probe_u clone.
    assert estimated == 2 * 8 * (16 + 3 * 16) * 2


def test_serving_zo_opt_metadata_shapes():
    metadata = build_lora_param_metadata_from_config(
        _OPTConfig(),
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert metadata["model.decoder.layers.0.self_attn.out_proj.weight"].shape == (
        16,
        16,
    )
    assert metadata["model.decoder.layers.0.fc1.weight"].shape == (64, 16)
    assert len(metadata) == 12


def test_serving_zo_metadata_optionally_includes_lm_head_and_embeddings():
    target_modules = resolve_lora_target_modules(
        None,
        include_lm_head=True,
        include_embeddings=True,
    )
    metadata = build_lora_param_metadata_from_config(
        _Qwen3Config(),
        device=torch.device("cpu"),
        dtype=torch.float16,
        target_modules=target_modules,
    )

    assert metadata["lm_head.weight"].shape == (128, 16)
    assert metadata["model.embed_tokens.weight"].shape == (16, 128)


def test_serving_zo_metadata_respects_explicit_target_modules():
    target_modules = resolve_lora_target_modules(
        ["q_proj"],
        include_lm_head=True,
        include_embeddings=True,
    )
    metadata = build_lora_param_metadata_from_config(
        _Qwen3Config(),
        device=torch.device("cpu"),
        dtype=torch.float16,
        target_modules=target_modules,
    )

    assert sorted(metadata) == [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ]


def test_serving_zo_metadata_uses_target_modules_list_only_internally():
    metadata = build_lora_param_metadata_from_config(
        _Qwen3Config(),
        device=torch.device("cpu"),
        dtype=torch.float16,
        target_modules=["q_proj", "lm_head"],
    )

    assert sorted(metadata) == [
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ]


def test_serving_zo_target_modules_flags_are_shared_with_vllm_scope():
    assert resolve_lora_target_modules(
        ["q_proj"],
        include_lm_head=True,
        include_embeddings=True,
    ) == ["q_proj", "lm_head", "embed_tokens"]


def test_direct_lora_runtime_empty_tensors_follow_optional_scope():
    runtime = LoRAUpdateRuntime.from_model_config(
        _Qwen3Config(),
        rank=4,
        base_model_name="Qwen/Qwen3-8B",
        target_modules=resolve_lora_target_modules(
            None,
            include_lm_head=True,
            include_embeddings=True,
        ),
    )
    tensors = runtime._build_empty_tensors()

    assert "base_model.model.lm_head.lora_A.weight" in tensors
    assert "base_model.model.lm_head.lora_B.weight" in tensors
    assert "base_model.model.model.embed_tokens.lora_embedding_A" in tensors
    assert "base_model.model.model.embed_tokens.lora_embedding_B" in tensors
    assert tensors["base_model.model.lm_head.lora_A.weight"].shape == (4, 16)
    assert tensors["base_model.model.lm_head.lora_B.weight"].shape == (128, 4)
    assert tensors["base_model.model.model.embed_tokens.lora_embedding_A"].shape == (
        4,
        128,
    )


def test_direct_lora_runtime_builds_tied_opt_lm_head_tensors():
    runtime = LoRAUpdateRuntime.from_model_config(
        _OPTConfig(),
        rank=4,
        base_model_name="facebook/opt-125m",
        target_modules=resolve_lora_target_modules(
            None,
            include_lm_head=True,
            include_embeddings=True,
        ),
    )
    tensors = runtime._build_empty_tensors()

    assert "base_model.model.lm_head.lora_A.weight" in tensors
    assert "base_model.model.lm_head.lora_B.weight" in tensors
    assert "base_model.model.model.decoder.embed_tokens.lora_embedding_A" in tensors
    assert "base_model.model.model.decoder.embed_tokens.lora_embedding_B" in tensors
    assert tensors["base_model.model.lm_head.lora_A.weight"].shape == (4, 16)
    assert tensors["base_model.model.lm_head.lora_B.weight"].shape == (128, 4)


def test_direct_lora_runtime_empty_tensors_support_qwen2_layout():
    runtime = LoRAUpdateRuntime.from_model_config(
        _Qwen2Config(),
        rank=2,
        base_model_name="Qwen/Qwen2.5-0.5B",
    )
    tensors = runtime._build_empty_tensors()

    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in tensors
    assert "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight" in tensors
    assert "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight" in tensors
    assert tensors[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    ].shape == (2, 16)
    assert tensors[
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"
    ].shape == (8, 2)


def test_direct_lora_runtime_peft_builder_uses_embedding_key_format():
    runtime = LoRAUpdateRuntime(rank=4)
    tensors = runtime.build_peft_tensors(
        {"model.embed_tokens.weight": torch.zeros(4, 128)},
        {"model.embed_tokens.weight": torch.zeros(16, 4)},
    )

    assert "base_model.model.model.embed_tokens.lora_embedding_A" in tensors
    assert "base_model.model.model.embed_tokens.lora_embedding_B" in tensors
    assert "base_model.model.model.embed_tokens.lora_A.weight" not in tensors


def test_serving_zo_direction_dtype_auto_follows_model_config():
    assert resolve_direction_dtype("auto", _Qwen3Config()) == torch.bfloat16
    assert resolve_direction_dtype("auto", _OPTConfig()) == torch.float16


def test_serving_zo_direction_dtype_auto_treats_fp32_config_as_vllm_fp16():
    config = SimpleNamespace(torch_dtype=torch.float32)

    assert resolve_direction_dtype("auto", config) == torch.float16


def test_serving_zo_start_request_has_only_explicit_score_admission_policy():
    request = ServingZOStartRequest()
    assert request.slot_write_stream == "background_sync"
    assert request.score_admission_policy == "idle_gap"
    assert request.score_admission_timeout_s == 0.0
    fields = getattr(ServingZOStartRequest, "model_fields", {})
    assert "wait_for_foreground_idle" not in fields
    assert "max_foreground_load_for_slot_write" not in fields
    assert "foreground_idle_poll_s" not in fields
    assert "foreground_idle_timeout_s" not in fields
    assert "score_wait_for_foreground_idle" not in fields
    assert "max_foreground_load_for_score" not in fields
    assert "max_inflight_zo_pairs" not in fields
    assert not hasattr(zo_training_api_module, "ServingZOController")
    assert not hasattr(zo_training_api_module, "ServingZOTrainer")


def _serving_zo_compact_nll_runner(
    *,
    lora_mapping: list[int],
    extra_args_by_req: dict[str, dict[str, object]],
):
    req_ids = list(extra_args_by_req)
    runner = object.__new__(GPUModelRunner)
    runner.lora_config = object()
    runner.input_batch = SimpleNamespace(
        req_ids=req_ids,
        num_reqs=len(req_ids),
        request_lora_mapping=np.array(lora_mapping, dtype=np.int32),
    )
    runner.requests = {
        req_id: SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args=dict(extra_args))
        )
        for req_id, extra_args in extra_args_by_req.items()
    }
    return runner


def test_serving_zo_compact_nll_with_lora_forces_eager_detection(monkeypatch):
    monkeypatch.delenv("VLLM_ZO_FORCE_EAGER_SCORING", raising=False)
    runner = _serving_zo_compact_nll_runner(
        lora_mapping=[0, 9001],
        extra_args_by_req={
            "foreground": {},
            "zo": {"zo_direct_prompt_nll": True},
        },
    )

    assert runner._has_serving_zo_compact_nll_with_lora()


def test_serving_zo_compact_nll_without_lora_keeps_graph_detection(monkeypatch):
    monkeypatch.delenv("VLLM_ZO_FORCE_EAGER_SCORING", raising=False)
    runner = _serving_zo_compact_nll_runner(
        lora_mapping=[0],
        extra_args_by_req={"clean": {"zo_direct_prompt_nll": True}},
    )

    assert not runner._has_serving_zo_compact_nll_with_lora()


def test_serving_zo_compact_nll_eager_detection_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_ZO_FORCE_EAGER_SCORING", "0")
    runner = _serving_zo_compact_nll_runner(
        lora_mapping=[9001],
        extra_args_by_req={"zo": {"zo_direct_prompt_nll": True}},
    )

    assert not runner._has_serving_zo_compact_nll_with_lora()


def test_serving_zo_lora_slot_event_waits_only_for_matching_lora(monkeypatch):
    waits = []

    class _FakeStream:
        def wait_event(self, event):
            waits.append(event)

    event = object()
    runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_zo_lora_slot_write_events={9001: event}),
        input_batch=SimpleNamespace(
            request_lora_mapping=np.array([9001, 0], dtype=np.int64),
            num_reqs=2,
        ),
        device=torch.device("cuda", 0),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gpu_model_runner_module, "current_stream", lambda: _FakeStream()
    )

    GPUModelRunner._wait_for_pending_zo_lora_slot_writes(runner)

    assert waits == [event]


def test_serving_zo_lora_slot_event_does_not_wait_for_unmatched_lora(monkeypatch):
    waits = []

    class _FakeStream:
        def wait_event(self, event):
            waits.append(event)

    runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_zo_lora_slot_write_events={9001: object()}),
        input_batch=SimpleNamespace(
            request_lora_mapping=np.array([0, 1234], dtype=np.int64),
            num_reqs=2,
        ),
        device=torch.device("cuda", 0),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gpu_model_runner_module, "current_stream", lambda: _FakeStream()
    )

    GPUModelRunner._wait_for_pending_zo_lora_slot_writes(runner)

    assert waits == []


def test_serving_zo_lora_slot_use_records_matching_lora(monkeypatch):
    recorded = []

    class _FakeStream:
        pass

    class _FakeEvent:
        def __init__(self, blocking=False):
            self.blocking = blocking

        def record(self, stream):
            recorded.append(stream)

    stream = _FakeStream()
    manager = SimpleNamespace(
        _zo_lora_slot_write_events={9001: object(), 9002: object()}
    )
    runner = SimpleNamespace(
        lora_manager=manager,
        input_batch=SimpleNamespace(
            request_lora_mapping=np.array([9001, 0, 9002], dtype=np.int64),
            num_reqs=3,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(gpu_model_runner_module, "current_stream", lambda: stream)

    GPUModelRunner._record_zo_lora_slot_use(runner)

    events = manager._zo_lora_slot_use_events
    assert sorted(events.keys()) == [9001, 9002]
    assert events[9001] is events[9002]
    assert recorded == [stream]


def _runtime_validation_inputs(
    *,
    policy="priority",
    enable_lora=True,
    max_loras=2,
    max_lora_rank=1024,
    enable_server_load_tracking=True,
):
    state = SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(policy=policy),
        ),
        enable_server_load_tracking=enable_server_load_tracking,
        server_load_metrics=0,
    )
    server_args = SimpleNamespace(
        enable_lora=enable_lora,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
    )
    return state, server_args


def test_serving_zo_runtime_validation_accepts_required_serving_config():
    state, server_args = _runtime_validation_inputs()

    zo_training_api_module._validate_runtime_config(
        state, server_args, ServingZOStartRequest(update_bank_rank=1024)
    )


def test_serving_zo_runtime_validation_rejects_invalid_scheduler():
    state, server_args = _runtime_validation_inputs()

    with pytest.raises(ValueError, match="unsupported serving scheduler"):
        zo_training_api_module._validate_runtime_config(
            state, server_args, ServingZOStartRequest(lr_scheduler_type="surprise")
        )
    with pytest.raises(ValueError, match="open-ended serving training"):
        zo_training_api_module._validate_runtime_config(
            state,
            server_args,
            ServingZOStartRequest(
                steps=0,
                update_bank_rank=1024,
                lr_scheduler_type="linear",
            ),
        )
    with pytest.raises(ValueError, match="metric-driven plateau"):
        zo_training_api_module._validate_runtime_config(
            state,
            server_args,
            ServingZOStartRequest(lr_scheduler_type="reduce_lr_on_plateau"),
        )


def test_serving_zo_runtime_validation_rejects_unsupported_weight_decay():
    state, server_args = _runtime_validation_inputs()

    with pytest.raises(ValueError, match="does not support weight_decay"):
        zo_training_api_module._validate_runtime_config(
            state, server_args, ServingZOStartRequest(weight_decay=0.1)
        )


def test_serving_zo_runtime_validation_accepts_scheduler_only_admission():
    state, server_args = _runtime_validation_inputs()

    zo_training_api_module._validate_runtime_config(
        state,
        server_args,
        ServingZOStartRequest(score_admission_policy="scheduler_only"),
    )
    with pytest.raises(RuntimeError, match="score_admission_policy"):
        zo_training_api_module._validate_runtime_config(
            state, server_args, ServingZOStartRequest(score_admission_policy="none")
        )


def test_serving_zo_runtime_validation_accepts_queued_admission():
    state, server_args = _runtime_validation_inputs()

    zo_training_api_module._validate_runtime_config(
        state, server_args, ServingZOStartRequest(score_admission_policy="queued")
    )
    with pytest.raises(RuntimeError, match="score_admission_policy"):
        zo_training_api_module._validate_runtime_config(
            state,
            server_args,
            ServingZOStartRequest(score_admission_policy="rate_limited"),
        )


def test_serving_zo_runtime_validation_accepts_gpu_utilization_admission():
    state, server_args = _runtime_validation_inputs()

    zo_training_api_module._validate_runtime_config(
        state,
        server_args,
        ServingZOStartRequest(score_admission_policy="gpu_utilization"),
    )
    with pytest.raises(RuntimeError, match="score_admission_policy"):
        zo_training_api_module._validate_runtime_config(
            state, server_args, ServingZOStartRequest(score_admission_policy="gpu")
        )


def test_serving_zo_runtime_validation_accepts_enum_like_priority_policy():
    state, server_args = _runtime_validation_inputs(policy="SchedulingPolicy.PRIORITY")

    zo_training_api_module._validate_runtime_config(
        state, server_args, ServingZOStartRequest(update_bank_rank=1024)
    )


def test_serving_zo_runtime_validation_rejects_non_priority_scheduler():
    state, server_args = _runtime_validation_inputs(policy="fcfs")

    with pytest.raises(RuntimeError, match="scheduling-policy priority"):
        zo_training_api_module._validate_runtime_config(
            state, server_args, ServingZOStartRequest()
        )


def test_serving_zo_runtime_validation_rejects_small_lora_bank_rank():
    state, server_args = _runtime_validation_inputs(max_lora_rank=16)

    with pytest.raises(RuntimeError, match="max-lora-rank"):
        zo_training_api_module._validate_runtime_config(
            state, server_args, ServingZOStartRequest(update_bank_rank=32)
        )


def test_serving_zo_runtime_validation_rejects_unknown_admission_policy():
    state, server_args = _runtime_validation_inputs()

    with pytest.raises(RuntimeError, match="score_admission_policy"):
        zo_training_api_module._validate_runtime_config(
            state, server_args, ServingZOStartRequest(score_admission_policy="surprise")
        )


def test_serving_zo_runtime_validation_allows_no_load_tracking_dependency():
    state, server_args = _runtime_validation_inputs(enable_server_load_tracking=False)

    zo_training_api_module._validate_runtime_config(
        state, server_args, ServingZOStartRequest()
    )


def test_serving_zo_idle_gap_wait_exits_when_stop_requested():
    state, _ = _runtime_validation_inputs()
    stop_event = threading.Event()
    stop_event.set()
    state.server_load_metrics = 13
    admission = _ScoreAdmissionController(
        state=state,
        request=ServingZOStartRequest(
            score_admission_poll_s=0.001,
            score_admission_timeout_s=30.0,
            max_score_admission_foreground_load=0,
        ),
        stop_requested=stop_event.is_set,
    )

    wait_s, samples, load = asyncio.run(admission.wait())

    assert wait_s < 0.1
    assert samples == 1
    assert load == 13


def test_serving_zo_scheduler_only_admission_does_not_wait_for_gap():
    state, _ = _runtime_validation_inputs()
    state.server_load_metrics = 13
    admission = _ScoreAdmissionController(
        state=state,
        request=ServingZOStartRequest(
            score_admission_policy="scheduler_only",
            score_admission_poll_s=30.0,
            score_admission_timeout_s=30.0,
            max_score_admission_foreground_load=0,
        ),
        stop_requested=lambda: False,
    )

    wait_s, samples, load = asyncio.run(admission.wait())

    assert wait_s == 0.0
    assert samples == 1
    assert load == 13


def test_serving_zo_gpu_utilization_admission_waits_for_low_utilization(monkeypatch):
    state, _ = _runtime_validation_inputs()
    state.server_load_metrics = 7
    samples = iter([91.0, 83.0, 42.0])

    monkeypatch.setattr(
        async_engine_service_module,
        "query_gpu_utilization_percent",
        lambda device: next(samples),
    )
    monkeypatch.setattr(
        async_engine_service_module,
        "resolve_gpu_utilization_device",
        lambda request_device: request_device or "6",
    )
    admission = _ScoreAdmissionController(
        state=state,
        request=ServingZOStartRequest(
            score_admission_policy="gpu_utilization",
            score_admission_poll_s=0.001,
            score_admission_timeout_s=1.0,
            max_score_admission_gpu_utilization=50.0,
            score_admission_gpu_device="6",
        ),
        stop_requested=lambda: False,
    )

    wait_s, sample_count, load = asyncio.run(admission.wait())

    assert wait_s >= 0.0
    assert sample_count == 3
    assert load == 7
    assert admission.extra() == {
        "score_admission_gpu_utilization": 42.0,
        "score_admission_gpu_utilization_threshold": 50.0,
        "score_admission_gpu_device": "6",
    }


class _FakeCompactNLLEngineClient:
    def __init__(self):
        self.calls = []

    async def generate(
        self,
        prompt,
        sampling_params,
        request_id,
        *,
        lora_request=None,
        priority=0,
    ):
        del lora_request
        self.calls.append(
            {
                "request_id": request_id,
                "priority": priority,
                "prompt_token_ids": list(prompt["prompt_token_ids"]),
                "perf_time": asyncio.get_running_loop().time(),
            }
        )
        labels = list(sampling_params.extra_args["zo_loss_labels"])
        active_labels = [int(item) for item in labels if int(item) != -100]
        yield SimpleNamespace(
            prompt_logprobs={
                "__zo_prompt_nll__": True,
                "nll_sum": float(sum(active_labels)),
                "num_tokens": len(active_labels),
            }
        )


def test_queued_compact_nll_scorer_preserves_results_and_priority():
    async def run():
        engine = _FakeCompactNLLEngineClient()
        submitter = QueuedCompactNLLSubmitter(
            token_rate=100.0,
            burst_tokens=5,
            max_inflight=8,
            max_admitted_tokens=10,
            poll_s=0.001,
        )
        scorer = ScheduledCompactNLLScorer(
            engine_client=engine,
            priority=1000,
            submitter=submitter,
        )
        try:
            t0 = asyncio.get_running_loop().time()
            result = await scorer.score(
                token_groups := [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                ],
                labels=_causal_labels(token_groups),
                lora_request=None,
                tag="queue-test",
            )
            elapsed = asyncio.get_running_loop().time() - t0
        finally:
            await submitter.close()
        return engine, result, elapsed

    engine, result, elapsed = asyncio.run(run())

    expected_nll = float(sum(range(1, 16)) - 1 - 6 - 11)
    assert result.nll_sum == pytest.approx(expected_nll)
    assert result.num_tokens == 12
    assert result.loss == pytest.approx(expected_nll / 12.0)
    assert [call["priority"] for call in engine.calls] == [1000, 1000, 1000]
    assert all(
        call["request_id"].startswith("serving-zo-queue-test-") for call in engine.calls
    )
    assert elapsed >= 0.07


@pytest.mark.parametrize(
    ("max_admitted_tokens", "expected_max_concurrent"),
    [(5, 1), (10, 2), (15, 3)],
)
def test_queued_compact_nll_scorer_limits_admitted_tokens(
    max_admitted_tokens: int,
    expected_max_concurrent: int,
):
    class SlowEngine(_FakeCompactNLLEngineClient):
        def __init__(self) -> None:
            super().__init__()
            self.active_tokens = 0
            self.max_active_tokens = 0
            self.max_concurrent = 0
            self.active_requests = 0

        async def generate(
            self,
            prompt,
            sampling_params,
            request_id,
            *,
            lora_request=None,
            priority=0,
        ):
            token_ids = list(prompt["prompt_token_ids"])
            self.active_tokens += len(token_ids)
            self.active_requests += 1
            self.max_active_tokens = max(self.max_active_tokens, self.active_tokens)
            self.max_concurrent = max(self.max_concurrent, self.active_requests)
            try:
                await asyncio.sleep(0.02)
                async for output in super().generate(
                    prompt,
                    sampling_params,
                    request_id,
                    lora_request=lora_request,
                    priority=priority,
                ):
                    yield output
            finally:
                self.active_tokens -= len(token_ids)
                self.active_requests -= 1

    async def run():
        engine = SlowEngine()
        submitter = QueuedCompactNLLSubmitter(
            token_rate=1000.0,
            burst_tokens=1000,
            max_inflight=8,
            max_admitted_tokens=max_admitted_tokens,
            poll_s=0.001,
        )
        scorer = ScheduledCompactNLLScorer(
            engine_client=engine,
            priority=1000,
            submitter=submitter,
        )
        try:
            result = await scorer.score(
                token_groups := [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                ],
                labels=_causal_labels(token_groups),
                lora_request=None,
                tag=f"admitted-{max_admitted_tokens}",
            )
        finally:
            await submitter.close()
        return engine, submitter, result

    engine, submitter, result = asyncio.run(run())

    assert result.num_tokens == 12
    assert engine.max_active_tokens <= max_admitted_tokens
    assert engine.max_concurrent == expected_max_concurrent
    assert submitter.stats()["admitted_tokens"] == 0


def test_queued_compact_nll_scorer_rejects_oversized_request():
    async def run():
        submitter = QueuedCompactNLLSubmitter(
            token_rate=1000.0,
            burst_tokens=1000,
            max_inflight=8,
            max_admitted_tokens=4,
            poll_s=0.001,
        )
        try:
            with pytest.raises(RuntimeError, match="exceeds max_admitted_tokens"):
                await submitter.submit(
                    lambda: asyncio.sleep(0),
                    estimated_tokens=5,
                )
        finally:
            await submitter.close()

    asyncio.run(run())
