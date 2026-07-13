from types import SimpleNamespace

import torch
import pytest

from zo_vllm.experiment.runners.checkpoint_policy import (
    BestMetricTracker,
    infer_greater_is_better,
    metric_value,
    resolve_checkpoint_mode,
    resolve_runtime_checkpoint_settings,
)
from zo_vllm.training.lora_checkpoint import (
    load_lora_bank_checkpoint,
    restore_direction_provider_v_cache_from_lora_bank,
    save_lora_bank_checkpoint,
)
from zo_vllm.experiment.runners.output import resolve_vllm_output_paths
from zo_vllm.experiment.runners.trainer_state import (
    build_zo_trainer_state,
    load_zo_trainer_state,
    merge_zo_trainer_state,
    save_zo_trainer_state,
)
from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.core.perturbation_normalization import (
    PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN,
)


def test_metric_value_uses_hf_like_eval_prefixes():
    row = {
        "step": 10,
        "eval_loss": 1.25,
        "eval_accuracy": 0.6,
        "eval_valid_accuracy": 0.7,
    }

    assert metric_value(row, "eval_loss") == 1.25
    assert metric_value(row, "eval_accuracy") == 0.6
    assert metric_value(row, "eval_valid_accuracy") == 0.7


def test_best_metric_tracker_minimizes_loss_by_default():
    tracker = BestMetricTracker(metric_for_best_model="eval_loss")

    assert tracker.is_better({"step": 1, "eval_loss": 2.0}) is True
    assert tracker.update({"step": 1, "eval_loss": 2.0}, {"path": "a"}) is True
    assert tracker.is_better({"step": 2, "eval_loss": 2.5}) is False
    assert tracker.update({"step": 2, "eval_loss": 2.5}, {"path": "b"}) is False
    assert tracker.is_better({"step": 3, "eval_loss": 1.5}) is True
    assert tracker.update({"step": 3, "eval_loss": 1.5}, {"path": "c"}) is True

    assert tracker.best_value == 1.5
    assert tracker.best_record is not None
    assert tracker.best_record["checkpoint"]["path"] == "c"


def test_best_metric_tracker_maximizes_accuracy_by_default():
    tracker = BestMetricTracker(metric_for_best_model="eval_accuracy")

    assert tracker.update({"step": 1, "eval_accuracy": 0.7}, {"path": "a"}) is True
    assert tracker.update({"step": 2, "eval_accuracy": 0.6}, {"path": "b"}) is False
    assert tracker.update({"step": 3, "eval_accuracy": 0.8}, {"path": "c"}) is True

    assert tracker.best_value == 0.8
    assert tracker.best_record is not None
    assert tracker.best_record["checkpoint"]["path"] == "c"


def test_best_metric_tracker_rejects_unprefixed_metric_name():
    tracker = BestMetricTracker(metric_for_best_model="eval_loss")

    with pytest.raises(KeyError, match="eval_loss"):
        tracker.update({"step": 1, "loss": 2.0}, {"path": "a"})


def test_checkpoint_mode_auto_matches_load_best():
    assert infer_greater_is_better("eval_loss") is False
    assert infer_greater_is_better("eval_accuracy") is True
    assert (
        resolve_checkpoint_mode(
            save_checkpoint_mode="auto",
            load_best_model_at_end=False,
        )
        == "metadata"
    )
    assert (
        resolve_checkpoint_mode(
            save_checkpoint_mode="auto",
            load_best_model_at_end=True,
        )
        == "native"
    )


def test_runtime_checkpoint_settings_validate_load_best_schedule():
    args = SimpleNamespace(
        save_strategy="steps",
        save_checkpoint_mode="auto",
        load_best_model_at_end=True,
    )

    try:
        resolve_runtime_checkpoint_settings(
            args=args,
            effective_eval_interval=10,
            effective_save_steps=3,
        )
    except ValueError as exc:
        assert "requires eval steps to also be checkpoint steps" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_runtime_checkpoint_settings_resolve_native_load_best():
    args = SimpleNamespace(
        save_strategy="steps",
        save_checkpoint_mode="auto",
        load_best_model_at_end=True,
    )

    settings = resolve_runtime_checkpoint_settings(
        args=args,
        effective_eval_interval=10,
        effective_save_steps=5,
    )

    assert settings.load_best_model_at_end is True
    assert settings.effective_save_steps == 5
    assert settings.checkpoint_mode == "native"
    assert settings.runtime_checkpoints_enabled is True


def test_resolve_vllm_output_paths_keeps_runner_defaults_outside_main_loop():
    base_args = SimpleNamespace(
        output_dir=None,
        output_root=None,
        experiment_name=None,
        direction_provider="lozo",
        train_objective="sst2",
        rank=2,
        nu=50,
    )

    lozo_paths = resolve_vllm_output_paths(
        args=base_args,
        project_root="/repo",
        model_name="facebook/opt-2.7b",
        timestamp="20260624_120000",
    )
    assert lozo_paths.output_root == "/repo/phase3/results"
    assert lozo_paths.output_dir.endswith("/vllm")
    assert "facebook__opt-2.7b" in lozo_paths.experiment_name

    agzo_args = SimpleNamespace(**vars(base_args) | {"direction_provider": "agzo"})
    agzo_paths = resolve_vllm_output_paths(
        args=agzo_args,
        project_root="/repo",
        model_name="facebook/opt-2.7b",
        timestamp="20260624_120000",
    )
    assert agzo_paths.output_root == "/repo/zo_post/results"

    explicit_args = SimpleNamespace(**vars(base_args) | {"output_dir": "/tmp/run"})
    explicit_paths = resolve_vllm_output_paths(
        args=explicit_args,
        project_root="/repo",
        model_name="facebook/opt-2.7b",
        timestamp="20260624_120000",
    )
    assert explicit_paths.output_dir == "/tmp/run"
    assert explicit_paths.output_root is None
    assert explicit_paths.experiment_name is None


def test_lora_bank_checkpoint_saves_lightweight_payload(tmp_path):
    state = SimpleNamespace(
        update_bank_rank=4,
        u_beta=1.0,
        u_norm_cap=None,
        gradient_accumulation_update_steps=0,
        bank_a={"layer.weight": torch.ones(2, 3)},
        accumulated_u={"layer.weight": torch.full((4, 2), 2.0)},
        pending_u={},
        used_rank={"layer.weight": 2},
        current_blocks={"layer.weight": SimpleNamespace(start=0, rank=2)},
        flush_pending_to_accumulated=lambda *, step: 0.125,
    )

    info = save_lora_bank_checkpoint(
        checkpoint_dir=str(tmp_path),
        accumulated_update_state=state,
        step=7,
        raw_step=9,
        dtype=torch.float16,
    )

    payload = torch.load(info["lora_payload_path"], map_location="cpu")
    assert info["flush_s"] == 0.125
    assert info["folded_accumulated_update"] is False
    assert payload["checkpoint_type"] == "vllm_zo_lora_bank"
    assert payload["step"] == 7
    assert payload["raw_step"] == 9
    assert payload["bank_a"]["layer.weight"].dtype == torch.float16
    assert payload["current_blocks"]["layer.weight"] == {"start": 0, "rank": 2}


def test_lora_bank_checkpoint_loads_into_existing_state(tmp_path):
    state = SimpleNamespace(
        update_bank_rank=4,
        u_beta=1.0,
        u_norm_cap=None,
        gradient_accumulation_update_steps=0,
        bank_a={"layer.weight": torch.arange(12).reshape(4, 3).float()},
        accumulated_u={"layer.weight": torch.full((5, 4), 2.0)},
        pending_u={"layer.weight": torch.ones(5, 2)},
        used_rank={"layer.weight": 2},
        current_blocks={"layer.weight": SimpleNamespace(start=0, rank=2)},
        flush_pending_to_accumulated=lambda *, step: 0.0,
    )
    save_lora_bank_checkpoint(
        checkpoint_dir=str(tmp_path),
        accumulated_update_state=state,
        step=11,
        dtype=torch.float32,
    )
    target = SimpleNamespace(
        update_bank_rank=4,
        u_beta=1.0,
        u_norm_cap=None,
        gradient_accumulation_update_steps=0,
        bank_a={},
        accumulated_u={},
        pending_u={},
        zero_u={},
        used_rank={},
        current_blocks={},
    )

    info = load_lora_bank_checkpoint(
        str(tmp_path),
        accumulated_update_state=target,
        device="cpu",
    )

    assert info["step"] == 11
    assert info["raw_step"] == 11
    assert info["num_bank_modules"] == 1
    torch.testing.assert_close(
        target.bank_a["layer.weight"], state.bank_a["layer.weight"]
    )
    torch.testing.assert_close(
        target.accumulated_u["layer.weight"],
        state.accumulated_u["layer.weight"],
    )
    assert torch.count_nonzero(target.zero_u["layer.weight"]) == 0
    assert target.used_rank["layer.weight"] == 2
    assert target.current_blocks["layer.weight"].start == 0
    assert target.current_blocks["layer.weight"].rank == 2


def test_zo_trainer_state_roundtrip_and_merge(tmp_path):
    tracker = BestMetricTracker(metric_for_best_model="eval_loss")
    best = {
        "metric": "loss",
        "metric_value": 0.4,
        "checkpoint": {"path": "checkpoint-0000010"},
    }
    state = build_zo_trainer_state(
        global_step=10,
        raw_step=12,
        log_history=[{"step": 0, "loss": 1.0}, {"step": 10, "loss": 0.4}],
        eval_losses=[{"step": 0, "loss": 1.0}, {"step": 10, "loss": 0.4}],
        eval_metrics=[{"step": 10, "loss": 0.4, "accuracy": 0.7}],
        checkpoint_records=[{"step": 10, "path": "checkpoint-0000010"}],
        best_checkpoint=best,
        wandb_run_id="abc123",
        wandb_run_name="run-a",
    )

    path = save_zo_trainer_state(tmp_path, state)
    loaded = load_zo_trainer_state(tmp_path)
    assert path.endswith("zo_trainer_state.json")
    assert loaded is not None
    assert loaded["global_step"] == 10
    assert loaded["wandb"]["run_id"] == "abc123"

    history = [{"step": 0, "loss": 9.0}, {"step": 11, "loss": 0.3}]
    eval_losses = [{"step": 0, "loss": 9.0}]
    eval_metrics = [{"step": 11, "loss": 0.3}]
    checkpoint_records = []
    checkpoint_paths = []
    info = merge_zo_trainer_state(
        loaded,
        history=history,
        eval_losses=eval_losses,
        eval_metrics=eval_metrics,
        checkpoint_records=checkpoint_records,
        checkpoint_paths=checkpoint_paths,
        best_tracker=tracker,
    )

    assert info["restored"] is True
    assert history == state["log_history"] + [{"step": 11, "loss": 0.3}]
    assert eval_losses == state["eval_losses"]
    assert eval_metrics == state["eval_metrics"] + [{"step": 11, "loss": 0.3}]
    assert checkpoint_records == state["checkpoint_records"]
    assert checkpoint_paths == ["checkpoint-0000010"]
    assert tracker.best_value == 0.4
    assert tracker.best_record == best


def test_agzo_lora_bank_resume_restores_rank_only_normalization_metadata():
    metadata = {
        "model.layers.0.fc1.weight": ParamMetadata(
            name="model.layers.0.fc1.weight",
            shape=(4, 3),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    }
    state = SimpleNamespace(
        bank_a={"model.layers.0.fc1.weight": torch.ones((1, 3))},
        current_blocks={"model.layers.0.fc1.weight": SimpleNamespace(start=0, rank=1)},
    )
    queue = SimpleNamespace(insert=lambda slot: setattr(queue, "slot", slot))
    provider = SimpleNamespace(v_provider=SimpleNamespace(subspace_queue=queue))

    info = restore_direction_provider_v_cache_from_lora_bank(
        param_metadata=metadata,
        accumulated_update_state=state,
        direction_provider=provider,
        direction_provider_name="uagzo",
        direction_scale=1.0,
    )
    direction = queue.slot["model.layers.0.fc1.weight"]

    assert info == {"restored_v_cache_modules": 1, "restored_subspace_queue": 1}
    assert direction["perturbation_effective_rank"] == 1
    assert (
        direction["perturbation_v_energy_reference"]
        == PERTURBATION_V_ENERGY_REFERENCE_GAUSSIAN
    )
    assert "perturbation_v_expected_column_norm_sq" not in direction
