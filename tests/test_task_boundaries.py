from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1] / "zo_vllm" / "tasks"
PACKAGE_DIR = Path(__file__).resolve().parents[1] / "zo_vllm"


def test_tasks_do_not_depend_on_experiment_modules():
    for path in TASK_DIR.glob("*.py"):
        source = path.read_text()
        assert "zo_vllm.experiment" not in source, path


def test_experiment_implementations_are_grouped_by_role():
    experiment_dir = PACKAGE_DIR / "experiment"
    for dirname in ("infra", "scoring", "runners"):
        assert (experiment_dir / dirname / "__init__.py").exists()
    assert not (experiment_dir / "compat").exists()

    assert (experiment_dir / "runners" / "vllm_zo_task.py").exists()
    assert (experiment_dir / "infra" / "naming.py").exists()


def test_experiment_project_root_resolves_repo_root():
    from zo_vllm.experiment.infra.paths import project_root

    assert project_root() == PACKAGE_DIR.parent


def test_experiment_flat_modules_are_removed():
    experiment_dir = PACKAGE_DIR / "experiment"
    flat_modules = {
        "batching.py",
        "binary_classification_direct_worker.py",
        "boolq_direct_worker.py",
        "boolq_official.py",
        "env.py",
        "io.py",
        "launch.py",
        "manifest.py",
        "naming.py",
        "paths.py",
        "phase4_summary.py",
        "run_state.py",
        "runner_args.py",
        "runner_output.py",
        "serving_load_runner.py",
        "squad_direct_worker.py",
        "squad_official.py",
        "sst2_direct_worker.py",
        "sst2_official.py",
        "stats.py",
        "vllm_zo_task_runner.py",
        "zo_task_batches.py",
        "task_batches.py",
    }

    for filename in flat_modules:
        assert not (experiment_dir / filename).exists(), filename


def test_training_direction_implementations_are_grouped():
    from zo_vllm.training.direction.components import (
        GaussianVProvider as GroupedGaussianVProvider,
    )
    from zo_vllm.training.direction.providers import (
        LOZOFastDirectionProvider as GroupedLOZOFastDirectionProvider,
    )
    from zo_vllm.training.direction import LOZOFastDirectionProvider

    assert LOZOFastDirectionProvider is GroupedLOZOFastDirectionProvider
    assert GroupedGaussianVProvider.__name__ == "GaussianVProvider"

    training_dir = PACKAGE_DIR / "training"
    flat_modules = {
        "direction_components.py",
        "direction_providers.py",
        "direction_types.py",
        "directions.py",
        "subspace_queue.py",
    }
    for filename in flat_modules:
        assert not (training_dir / filename).exists(), filename


def test_legacy_cpu_memory_lora_mock_is_removed():
    assert not (PACKAGE_DIR / "core" / "compat" / "__init__.py").exists()
    assert not (PACKAGE_DIR / "core" / "compat" / "memory_lora_loader.py").exists()
    assert not (PACKAGE_DIR / "core" / "memory_lora_loader.py").exists()


def test_generate_scorer_is_not_in_core():
    assert not (PACKAGE_DIR / "core" / "vllm_scorer.py").exists()
    assert (PACKAGE_DIR / "experiment" / "scoring" / "generate_scorer.py").exists()


def test_legacy_classification_scoring_wrappers_are_removed():
    scoring_dir = PACKAGE_DIR / "experiment" / "scoring"
    for filename in (
        "binary_classification.py",
        "option_classification.py",
        "sst2.py",
        "boolq.py",
        "superglue.py",
    ):
        assert not (scoring_dir / filename).exists(), filename
    assert not (scoring_dir / "record_nll.py").exists()
    assert not (scoring_dir / "router.py").exists()
    assert not (scoring_dir / "squad.py").exists()


def test_serving_package_does_not_import_experiment_package():
    for path in (PACKAGE_DIR / "serving").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "zo_vllm.experiment" not in text, path
        assert "phase7" not in text, path


def test_task_batches_are_training_owned():
    assert not (PACKAGE_DIR / "experiment" / "scoring" / "task_batches.py").exists()
    assert (PACKAGE_DIR / "training" / "task_batches.py").exists()
    assert (PACKAGE_DIR / "training" / "objective_scoring.py").exists()
    assert (PACKAGE_DIR / "training" / "objective_router.py").exists()
    assert (PACKAGE_DIR / "training" / "generation_eval.py").exists()
    task_batches_text = (PACKAGE_DIR / "training" / "task_batches.py").read_text(
        encoding="utf-8"
    )
    assert "def build_binary_classification_zo_batch" not in task_batches_text
    assert "def build_option_classification_zo_batch" not in task_batches_text
    assert "def build_squad_nll_zo_batch" not in task_batches_text
    assert "def build_record_nll_zo_batch" not in task_batches_text
    assert "def build_binary_classification_scoring_batch" not in task_batches_text
    assert "def build_option_classification_scoring_batch" not in task_batches_text
    assert "def build_classification_scoring_batch" not in task_batches_text
    assert "def build_masked_lm_scoring_batch" not in task_batches_text


def test_objective_scoring_facade_is_public():
    import zo_vllm
    from zo_vllm import training

    for name in (
        "build_objective_batch",
        "score_clean_objective",
        "score_plus_minus_objective",
        "collect_per_row_coefficients",
    ):
        assert hasattr(training, name)
        assert hasattr(zo_vllm, name)


def test_clean_objective_eval_routing_is_not_in_main_runner():
    runner_text = (
        PACKAGE_DIR / "experiment" / "runners" / "vllm_zo_task.py"
    ).read_text(encoding="utf-8")
    router_text = (PACKAGE_DIR / "training" / "objective_router.py").read_text(
        encoding="utf-8"
    )
    initial_loss_block = runner_text.split(
        "subspace_row_stream = list(train_items)", maxsplit=1
    )[1].split("np.random.seed(args.seed)", maxsplit=1)[0]
    final_loss_block = runner_text.split("final_dev_acc = None", maxsplit=1)[0].rsplit(
        'if args.base_eval_mode == "skip":', maxsplit=1
    )[-1]
    initial_metric_block = runner_text.split("np.random.seed(args.seed)", maxsplit=1)[
        1
    ].split("eval_metrics = (", maxsplit=1)[0]
    final_metric_block = runner_text.split("final_dev_acc = None", maxsplit=1)[1].split(
        "if not eval_metrics", maxsplit=1
    )[0]

    assert "score_objective_loss" in router_text
    assert "eval_objective_metrics" in router_text
    assert "score_objective_loss" in initial_loss_block
    assert "score_objective_loss" in final_loss_block
    assert "eval_objective_metrics" in initial_metric_block
    assert "eval_objective_metrics" in final_metric_block
    assert "score_record_nll(" not in initial_loss_block
    assert "score_squad_nll_direct_worker" not in initial_loss_block
    assert "score_record_nll(" not in final_loss_block
    assert "score_squad_nll_direct_worker" not in final_loss_block
    assert "classification_eval_fn(" not in initial_metric_block
    assert "eval_squad_f1_direct_worker(" not in initial_metric_block
    assert "classification_eval_fn(" not in final_metric_block
    assert "eval_squad_f1_direct_worker(" not in final_metric_block
    assert "classification_score_fn" not in runner_text
    assert "classification_eval_fn" not in runner_text
    assert "score_record_nll(" not in runner_text
    assert "score_squad_nll_direct_worker" not in runner_text
    assert "eval_squad_f1_direct_worker" not in runner_text


def test_update_debug_helpers_are_split_from_update_state():
    training_dir = PACKAGE_DIR / "training"
    assert (training_dir / "update_bank_state.py").exists()
    assert (training_dir / "update_debug.py").exists()
    assert (training_dir / "update_metrics.py").exists()

    update_state_text = (training_dir / "update_state.py").read_text(encoding="utf-8")
    assert "def tensor_finite_summary" not in update_state_text
    assert "def direction_tensors_finite_summary" not in update_state_text
    assert "def zo_bank_debug_enabled" not in update_state_text
    assert "def cosine_and_angle" not in update_state_text
    assert "def build_update_snapshot_metrics" not in update_state_text
    assert "class BlockLoRAUpdateBankState" not in update_state_text
    assert "class _BankBlock" not in update_state_text


def test_async_lora_registry_is_split_from_runtime():
    lora_runtime_dir = PACKAGE_DIR / "core" / "lora_runtime"
    assert (lora_runtime_dir / "runtime.py").exists()
    assert (lora_runtime_dir / "registry.py").exists()

    runtime_text = (lora_runtime_dir / "runtime.py").read_text(encoding="utf-8")
    assert "class AsyncLoRASlotRegistry" not in runtime_text


def test_lora_slot_validation_is_split_from_runtime():
    lora_runtime_dir = PACKAGE_DIR / "core" / "lora_runtime"
    assert (lora_runtime_dir / "slot_validation.py").exists()

    runtime_text = (lora_runtime_dir / "runtime.py").read_text(encoding="utf-8")
    assert "def validate_direct_lora_slot_structure" not in runtime_text
    assert "def _slot_base_weight_shape" not in runtime_text


def test_peft_tensor_helpers_are_split_from_runtime():
    lora_runtime_dir = PACKAGE_DIR / "core" / "lora_runtime"
    assert (lora_runtime_dir / "peft_tensors.py").exists()

    runtime_text = (lora_runtime_dir / "runtime.py").read_text(encoding="utf-8")
    assert "def _peft_tensor_base_shapes" not in runtime_text
    assert "def _parse_peft_lora_tensor_name" not in runtime_text


def test_lora_slot_writer_is_split_from_runtime():
    lora_runtime_dir = PACKAGE_DIR / "core" / "lora_runtime"
    assert (lora_runtime_dir / "slot_writer.py").exists()

    runtime_text = (lora_runtime_dir / "runtime.py").read_text(encoding="utf-8")
    assert "def _write_plus_minus_slots_from_directions" not in runtime_text
    assert "def _flush_queued_direction_writes" not in runtime_text
    assert "def _set_lora_noreset_if_full" not in runtime_text


def test_trainer_state_is_split_from_checkpoint_tensor_io():
    runner_dir = PACKAGE_DIR / "experiment" / "runners"
    training_dir = PACKAGE_DIR / "training"
    assert (runner_dir / "checkpoint_policy.py").exists()
    assert (runner_dir / "trainer_state.py").exists()
    assert (training_dir / "lora_checkpoint.py").exists()
    assert (training_dir / "native_checkpoint.py").exists()

    checkpointing_text = (runner_dir / "checkpointing.py").read_text(encoding="utf-8")
    assert "class BestMetricTracker" not in checkpointing_text
    assert "def metric_value" not in checkpointing_text
    assert "def resolve_checkpoint_mode" not in checkpointing_text
    assert "def save_lora_bank_checkpoint" not in checkpointing_text
    assert "def load_lora_bank_checkpoint" not in checkpointing_text
    assert (
        "def restore_direction_provider_v_cache_from_lora_bank"
        not in checkpointing_text
    )
    assert "def build_zo_trainer_state" not in checkpointing_text
    assert "def merge_zo_trainer_state" not in checkpointing_text
    assert "def load_zo_trainer_state" not in checkpointing_text

    runtime_text = (PACKAGE_DIR.parent / "zo_trainer" / "runtime.py").read_text(
        encoding="utf-8"
    )
    assert "zo_vllm.experiment.runners" not in runtime_text


def test_wandb_glue_is_split_from_main_runner():
    runner_dir = PACKAGE_DIR / "experiment" / "runners"
    assert (runner_dir / "wandb_logging.py").exists()

    runner_text = (runner_dir / "vllm_zo_task.py").read_text(encoding="utf-8")
    assert "import wandb" not in runner_text
    assert "def log_wandb" not in runner_text
    assert "init_wandb_logger" in runner_text


def test_interval_and_checkpoint_policy_are_split_from_main_runner():
    runner_dir = PACKAGE_DIR / "experiment" / "runners"
    assert (runner_dir / "intervals.py").exists()
    assert (runner_dir / "checkpoint_policy.py").exists()

    runner_text = (runner_dir / "vllm_zo_task.py").read_text(encoding="utf-8")
    assert "epoch_interval_to_steps" not in runner_text
    assert "resolve_runner_intervals" in runner_text
    assert "resolve_runtime_checkpoint_settings" in runner_text
    assert "resolve_checkpoint_mode(" not in runner_text


def test_run_console_logging_is_split_from_main_runner():
    runner_dir = PACKAGE_DIR / "experiment" / "runners"
    assert (runner_dir / "run_logging.py").exists()

    runner_text = (runner_dir / "vllm_zo_task.py").read_text(encoding="utf-8")
    assert "print_vllm_run_header" in runner_text
    assert "print_interval_summary" in runner_text
    assert "direction_scale_requested=" not in runner_text
    assert "eval_interval_steps=" not in runner_text


def test_agzo_subspace_queue_is_not_a_runner_config_knob():
    from dataclasses import fields

    from zo_vllm.config import VLLMZOConfig
    from zo_vllm.experiment.runners.args import build_vllm_zo_task_arg_parser

    config_fields = {field.name for field in fields(VLLMZOConfig)}
    assert "subspace_queue_size" not in config_fields

    help_text = build_vllm_zo_task_arg_parser().format_help()
    assert "--agzo-subspace-queue-size" not in help_text


def test_vllm_task_runner_delegates_output_path_policy():
    runner_text = (
        PACKAGE_DIR / "experiment" / "runners" / "vllm_zo_task.py"
    ).read_text(encoding="utf-8")
    assert "resolve_vllm_output_paths" in runner_text
    assert '"phase3", "results"' not in runner_text
    assert '"zo_post", "results"' not in runner_text


def test_vllm_task_runner_uses_public_trainer_loop():
    runner_text = (
        PACKAGE_DIR / "experiment" / "runners" / "vllm_zo_task.py"
    ).read_text(encoding="utf-8")
    assert "VLLMZOTrainer" in runner_text
    assert "VLLMZOTrainerCallback" in runner_text
    callback_class = "class VLLMTaskRunnerCallback"
    assert callback_class in runner_text
    assert "trainer.train()" in runner_text
    assert "for _epoch in range(" not in runner_text

    step_body = runner_text.split("class RunnerStepModel:", maxsplit=1)[1].split(
        callback_class, maxsplit=1
    )[0]
    assert "maybe_save_runtime_checkpoint" not in step_body
    assert "score_periodic_eval(" not in step_body
    assert "score_current_batch_train_loss(" not in step_body


def test_phase_training_scripts_use_public_trainer_loop():
    phase2_runner = PACKAGE_DIR.parent / "phase2" / "runners" / "train_convergence.py"
    runner_text = phase2_runner.read_text(encoding="utf-8")

    assert "VLLMZOTrainer" in runner_text
    assert "VLLMZOTrainerCallback" in runner_text
    assert "trainer.train()" in runner_text
    assert "for epoch in range(" not in runner_text
    assert "maybe_save_runtime_checkpoint" not in runner_text


def test_runtime_checkpoint_manager_does_not_own_save_cadence():
    checkpointing_text = (
        PACKAGE_DIR / "experiment" / "runners" / "checkpointing.py"
    ).read_text(encoding="utf-8")

    assert "def maybe_save_runtime_checkpoint" not in checkpointing_text
    assert "def save_runtime_checkpoint" in checkpointing_text


def test_training_layer_boundaries_are_documented():
    root = PACKAGE_DIR.parent
    readme_text = (root / "README.md").read_text(encoding="utf-8")
    execution_graph_text = (root / "docs" / "execution_graph.md").read_text(
        encoding="utf-8"
    )
    agents_text = (root / "AGENTS.md").read_text(encoding="utf-8")

    for text in (readme_text, execution_graph_text, agents_text):
        assert "ZOTrainer" in text
        assert "ZOStepper" in text
        assert "ZOEstimator" in text
    assert "Training Layering" in readme_text
    assert "VLLMZOTrainer" not in readme_text
    assert "VLLMZOTrainer" not in execution_graph_text
    assert "Trainer._save_checkpoint" in execution_graph_text
    assert "multi-query" in agents_text
