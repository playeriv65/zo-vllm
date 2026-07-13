"""GPU smoke tests for the Hugging Face ZO trainer integration."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, TrainerCallback
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from zo_trainer import (
    VLLMDataCollator,
    ZO_DIRECTION_STATE_NAME,
    ZOVLLMCheckpointHandler,
    ZOSGDOptimizer,
    ZOTrainer,
    ZOTrainerArguments,
    read_zo_checkpoint_metadata,
)
from zo_vllm.config import VLLMZOConfig, ZOVLLMEngineConfig
from zo_vllm.core.lora_scope import resolve_lora_target_modules
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.engine import ZOVLLMEngine
from zo_vllm.training import (
    RolloutProbeBatch,
    VLLMZOModel,
    build_tokenizer,
)
from zo_vllm.tasks.hf_preprocessing import (
    build_sst2_prompt_classification_preprocess,
)

OPT_125M = "facebook/opt-125m"


def _e2e_enabled() -> bool:
    return os.environ.get("ZO_VLLM_RUN_E2E") == "1"


def _prepare_e2e_environment() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("WANDB_MODE", "offline")


def _build_real_vllm_zo_model(
    **zo_config_overrides: object,
) -> tuple[ZOVLLMEngine, VLLMZOModel, object]:
    model_config = AutoConfig.from_pretrained(OPT_125M)
    tokenizer = build_tokenizer(OPT_125M, use_fast=False)
    target_modules = resolve_lora_target_modules(None)
    engine = ZOVLLMEngine(
        model=OPT_125M,
        rank=1,
        model_config=model_config,
        config=ZOVLLMEngineConfig(
            max_model_len=128,
            gpu_memory_utilization=0.25,
            enforce_eager=True,
            lora_rank=1,
            target_modules=target_modules,
            llm_kwargs={
                "max_num_batched_tokens": 256,
                "max_num_seqs": 4,
            },
        ),
    )
    weight_sync = WeightSync(
        engine.llm,
        num_layers=model_config.num_hidden_layers,
        model_config=model_config,
    )
    zo_config_values = {
        "direction_provider": "lozo",
        "lozo_provider_mode": "fast",
        "rank": 1,
        "eps": 1e-3,
        "learning_rate": 1e-7,
        "nu": 10,
        "random_device": "cuda",
        "direction_sampling": "flat",
        "direction_scale": 1.0,
        "perturbation_normalization": "rms",
        "max_logits_tokens": 1024,
        "loss_impl": "logprobs",
        "seed": 0,
    }
    zo_config_values.update(zo_config_overrides)
    zo_model = VLLMZOModel(
        engine=engine,
        weight_sync=weight_sync,
        config=VLLMZOConfig(**zo_config_values),
        param_metadata=weight_sync.get_hf_param_metadata(include_embeddings=False),
        sync_weight_update=False,
        qkv_update_mode="batched",
    )
    return engine, zo_model, tokenizer


def _trainer_args(
    tmp_path: Path, *, name: str, max_steps: int = 1
) -> ZOTrainerArguments:
    return ZOTrainerArguments(
        output_dir=str(tmp_path / name),
        max_steps=int(max_steps),
        per_device_train_batch_size=1,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=1,
        zo_checkpoint_mode="native",
        learning_rate=1e-7,
        disable_tqdm=True,
        report_to=[],
        remove_unused_columns=False,
    )


def _train_dataset_and_collator(tokenizer: object) -> tuple[object, object]:
    raw = Dataset.from_list(
        [
            {"sentence": "A good movie.", "label": 1},
            {"sentence": "A bad movie.", "label": 0},
        ]
    )
    train_dataset = raw.map(
        build_sst2_prompt_classification_preprocess(
            tokenizer,
            max_length=64,
        ),
        batched=True,
        remove_columns=raw.column_names,
    )
    return train_dataset, VLLMDataCollator()


def _cleanup_engine(engine: ZOVLLMEngine | None) -> None:
    if engine is not None:
        engine.cleanup()
    cleanup_dist_env_and_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _skip_without_e2e_gpu() -> None:
    if not _e2e_enabled():
        pytest.skip("set ZO_VLLM_RUN_E2E=1 to run GPU e2e smoke tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU e2e smoke tests")


@pytest.mark.e2e
@pytest.mark.gpu
def test_zo_trainer_resumes_real_runtime_optimizer_and_scheduler(
    tmp_path: Path,
) -> None:
    class StopAfterOneStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step == 1:
                control.should_training_stop = True
            return control

    _skip_without_e2e_gpu()
    _prepare_e2e_environment()
    engine: ZOVLLMEngine | None = None
    try:
        engine, zo_model, tokenizer = _build_real_vllm_zo_model()
        train_rows, data_collator = _train_dataset_and_collator(tokenizer)
        trainer = ZOTrainer(
            model=zo_model,
            args=_trainer_args(tmp_path, name="zo_trainer_vllm", max_steps=2),
            train_dataset=train_rows,
            data_collator=data_collator,
            checkpoint_handler=ZOVLLMCheckpointHandler(
                checkpoint_mode="native",
                llm=engine.llm,
                weight_sync=zo_model.weight_sync,
                update_state=zo_model.update_state,
            ),
            callbacks=[StopAfterOneStep()],
        )

        output = trainer.train()

        assert output.global_step == 1
        assert trainer.state.log_history
        assert "zo_projected_grad" in trainer.state.log_history[0]
        assert trainer.state.log_history[0][
            "zo_applied_learning_rate"
        ] == pytest.approx(1e-7)
        checkpoint = tmp_path / "zo_trainer_vllm" / "checkpoint-1"
        metadata = read_zo_checkpoint_metadata(checkpoint)
        assert metadata["checkpoint_mode"] == "native"
        assert metadata["payload"]["loadable"] is True
        assert (checkpoint / "optimizer.pt").exists()
        assert (checkpoint / "scheduler.pt").exists()
        assert (checkpoint / ZO_DIRECTION_STATE_NAME).exists()
        assert (checkpoint / "rng_state.pth").exists()

        _cleanup_engine(engine)
        engine = None
        del trainer, zo_model

        engine, resumed_zo_model, resumed_tokenizer = _build_real_vllm_zo_model()
        resumed_rows, resumed_collator = _train_dataset_and_collator(resumed_tokenizer)
        resumed = ZOTrainer(
            model=resumed_zo_model,
            args=_trainer_args(tmp_path, name="zo_trainer_vllm", max_steps=2),
            train_dataset=resumed_rows,
            data_collator=resumed_collator,
            checkpoint_handler=ZOVLLMCheckpointHandler(
                checkpoint_mode="native",
                llm=engine.llm,
                weight_sync=resumed_zo_model.weight_sync,
                update_state=resumed_zo_model.update_state,
            ),
        )

        resumed_output = resumed.train(resume_from_checkpoint=str(checkpoint))

        assert resumed_output.global_step == 2
        assert resumed._zo_optimizer().step_count == 2
        zo_logs = [
            row
            for row in resumed.state.log_history
            if "zo_applied_learning_rate" in row
        ]
        resumed_step_two = zo_logs[-1]
        assert resumed_step_two["zo_applied_learning_rate"] == pytest.approx(5e-8)
        assert (tmp_path / "zo_trainer_vllm" / "checkpoint-2").exists()

        _cleanup_engine(engine)
        engine = None
        del resumed, resumed_zo_model

        engine, continuous_zo_model, continuous_tokenizer = _build_real_vllm_zo_model()
        continuous_rows, continuous_collator = _train_dataset_and_collator(
            continuous_tokenizer
        )
        continuous = ZOTrainer(
            model=continuous_zo_model,
            args=_trainer_args(
                tmp_path,
                name="zo_trainer_vllm_continuous",
                max_steps=2,
            ),
            train_dataset=continuous_rows,
            data_collator=continuous_collator,
            checkpoint_handler=ZOVLLMCheckpointHandler(
                checkpoint_mode="native",
                llm=engine.llm,
                weight_sync=continuous_zo_model.weight_sync,
                update_state=continuous_zo_model.update_state,
            ),
        )

        continuous.train()

        continuous_step_two = next(
            row for row in continuous.state.log_history if row.get("zo_step") == 2
        )
        for key in (
            "zo_loss",
            "zo_loss_plus",
            "zo_loss_minus",
            "zo_projected_grad",
            "zo_applied_learning_rate",
        ):
            assert resumed_step_two[key] == pytest.approx(
                continuous_step_two[key], abs=1e-6
            )
    finally:
        _cleanup_engine(engine)


@pytest.mark.e2e
@pytest.mark.gpu
def test_zo_sgd_optimizer_applies_real_es_rollout_estimate() -> None:
    _skip_without_e2e_gpu()
    _prepare_e2e_environment()
    engine: ZOVLLMEngine | None = None
    try:
        engine, zo_model, _ = _build_real_vllm_zo_model(
            estimator="evolution_strategy",
            population_size=2,
            query_microbatch_size=2,
            sigma=1e-3,
            reward_shaping="none",
        )
        pending = zo_model.estimate(
            RolloutProbeBatch(
                rollout_prompts=["Review sentiment:"],
                rollout_targets=[None],
                rollout_reward_fn=lambda text, target: float(len(text)),
                rollout_max_tokens=1,
                rollout_seed=0,
            ),
            step=1,
        )
        optimizer = ZOSGDOptimizer(
            [torch.nn.Parameter(torch.zeros(()))],
            lr=1e-7,
        )

        optimizer.stage(pending)
        optimizer.step()

        result = optimizer.last_step_result
        assert result is not None
        assert result.estimator_metrics["estimator"] == "evolution_strategy"
        assert result.estimator_metrics["query_count"] == 2
        assert result.learning_rate == pytest.approx(1e-7)
    finally:
        _cleanup_engine(engine)
