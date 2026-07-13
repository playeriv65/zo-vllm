"""GPU smoke tests for the Hugging Face ZO trainer integration."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from safetensors import safe_open
from transformers import AutoConfig, TrainerCallback
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from zo_trainer import (
    VLLMDataCollator,
    ZO_DIRECTION_STATE_NAME,
    ZOVLLMCheckpointHandler,
    ZOTrainer,
    ZOTrainerArguments,
    ZORolloutTrainerModel,
    read_zo_checkpoint_metadata,
)
from zo_vllm.config import VLLMZOConfig, ZOVLLMEngineConfig
from zo_vllm.core.direction_digest import digest_named_uv
from zo_vllm.core.lora_scope import resolve_lora_target_modules
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.engine import ZOVLLMEngine
from zo_vllm.training import (
    AccumulatedLowRankUpdateState,
    VLLMZOModel,
    build_tokenizer,
)
from zo_vllm.training.zo_step import ZOStepCallback
from zo_vllm.tasks.hf_preprocessing import (
    build_sst2_prompt_classification_preprocess,
)
from zo_vllm.training.native_checkpoint import normalize_native_checkpoint_state

OPT_125M = "facebook/opt-125m"


def _e2e_enabled() -> bool:
    return os.environ.get("ZO_VLLM_RUN_E2E") == "1"


def _prepare_e2e_environment() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("WANDB_MODE", "offline")


def _build_real_vllm_zo_model(
    *,
    include_embeddings: bool = False,
    accumulate_updates: bool = False,
    **zo_config_overrides: object,
) -> tuple[ZOVLLMEngine, VLLMZOModel, object]:
    model_config = AutoConfig.from_pretrained(OPT_125M)
    tokenizer = build_tokenizer(OPT_125M, use_fast=False)
    target_modules = resolve_lora_target_modules(
        None,
        include_lm_head=include_embeddings
        and bool(getattr(model_config, "tie_word_embeddings", False)),
        include_embeddings=include_embeddings,
    )
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
    update_state = (
        AccumulatedLowRankUpdateState(
            weight_sync=weight_sync,
            engine=engine,
            precision="param",
            sync_device=False,
            qkv_update_mode="batched",
        )
        if accumulate_updates
        else None
    )
    zo_model = VLLMZOModel(
        engine=engine,
        weight_sync=weight_sync,
        config=VLLMZOConfig(**zo_config_values),
        param_metadata=weight_sync.get_hf_param_metadata(
            include_embeddings=include_embeddings
        ),
        update_state=update_state,
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


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = (
        tensor.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def _live_native_state_digests(engine: ZOVLLMEngine) -> dict[str, str]:
    def read_on_worker(worker):
        import hashlib as worker_hashlib

        from vllm.model_executor.model_loader import ShardedStateLoader

        state = ShardedStateLoader._filter_subtensors(
            worker.model_runner.model.state_dict()
        )
        state = normalize_native_checkpoint_state(state)
        return {
            key: worker_hashlib.sha256(
                value.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
            for key, value in state.items()
        }

    return dict(engine.llm.collective_rpc(read_on_worker)[0])


def _checkpoint_native_state_digests(checkpoint: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for shard in sorted(checkpoint.glob("model-rank-*-part-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise AssertionError(f"duplicate checkpoint tensor: {key}")
                result[key] = _tensor_digest(handle.get_tensor(key))
    return result


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
    class CaptureDirectionDigests(ZOStepCallback):
        def __init__(self) -> None:
            self.by_step: dict[int, str] = {}
            self.batch_by_step: dict[int, str] = {}

        def on_direction_sampled(self, *, step, batch, sample, control, **kwargs):
            del kwargs
            self.by_step[int(step)] = digest_named_uv(
                (name, value["U"], value["V"])
                for name, value in sorted(sample.directions.items())
            )
            self.batch_by_step[int(step)] = hashlib.sha256(
                repr(
                    (
                        batch.token_id_groups,
                        batch.loss_token_lens,
                        batch.labels,
                    )
                ).encode("utf-8")
            ).hexdigest()
            return control

    class StopAfterOneStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step == 1:
                control.should_training_stop = True
            return control

    class AssertRestoredDirectionState(TrainerCallback):
        def __init__(self, direction_provider, engine, expected_state_digests) -> None:
            self.direction_provider = direction_provider
            self.engine = engine
            self.expected_state_digests = expected_state_digests

        def on_train_begin(self, args, state, control, **kwargs):
            del args, state, kwargs
            assert not self.direction_provider.will_refresh(step=2)
            assert (
                _live_native_state_digests(self.engine) == self.expected_state_digests
            )
            return control

    class RehydrateSlotsAfterFirstStep(TrainerCallback):
        def __init__(self, zo_model, engine) -> None:
            self.zo_model = zo_model
            self.engine = engine
            self.state_digests: dict[int, dict[str, str]] = {}

        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step == 1:
                self.state_digests[1] = _live_native_state_digests(self.engine)
                self.zo_model.invalidate_direction_slot_state()
            return control

    _skip_without_e2e_gpu()
    _prepare_e2e_environment()
    engine: ZOVLLMEngine | None = None
    try:
        engine, zo_model, tokenizer = _build_real_vllm_zo_model()
        stopped_digests = CaptureDirectionDigests()
        zo_model.stepper.callbacks.append(stopped_digests)
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
        stopped_step_one = next(
            row for row in trainer.state.log_history if row.get("zo_step") == 1
        )
        assert "zo_projected_grad" in trainer.state.log_history[0]
        assert trainer.state.log_history[0][
            "zo_applied_learning_rate"
        ] == pytest.approx(1e-7)
        checkpoint = tmp_path / "zo_trainer_vllm" / "checkpoint-1"
        metadata = read_zo_checkpoint_metadata(checkpoint)
        assert metadata["checkpoint_mode"] == "native"
        assert metadata["payload"]["loadable"] is True
        layer_mapping = metadata["payload"]["runtime_manifest"]["layer_mapping"]
        assert len(layer_mapping["hf_to_vllm_mapping"]) > 0
        assert len(layer_mapping["hf_to_slice"]) > 0
        assert len(layer_mapping["fingerprint"]) == 64
        assert (checkpoint / "optimizer.pt").exists()
        assert (checkpoint / "scheduler.pt").exists()
        assert (checkpoint / ZO_DIRECTION_STATE_NAME).exists()
        assert (checkpoint / "rng_state.pth").exists()
        shard = next(checkpoint.glob("model-rank-*-part-*.safetensors"))
        with safe_open(shard, framework="pt", device="cpu") as handle:
            assert not any(".base_layer." in key for key in handle.keys())
        checkpoint_state_digests = _checkpoint_native_state_digests(checkpoint)
        assert _live_native_state_digests(engine) == checkpoint_state_digests

        _cleanup_engine(engine)
        engine = None
        del trainer, zo_model

        engine, resumed_zo_model, resumed_tokenizer = _build_real_vllm_zo_model()
        resumed_digests = CaptureDirectionDigests()
        resumed_zo_model.stepper.callbacks.append(resumed_digests)
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
            callbacks=[
                AssertRestoredDirectionState(
                    resumed_zo_model.direction_provider,
                    engine,
                    checkpoint_state_digests,
                )
            ],
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
        continuous_digests = CaptureDirectionDigests()
        continuous_zo_model.stepper.callbacks.append(continuous_digests)
        continuous_rows, continuous_collator = _train_dataset_and_collator(
            continuous_tokenizer
        )
        continuous_state = RehydrateSlotsAfterFirstStep(
            continuous_zo_model,
            engine,
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
            callbacks=[continuous_state],
        )

        continuous.train()

        assert stopped_digests.by_step[1] == continuous_digests.by_step[1]
        assert resumed_digests.by_step[2] == continuous_digests.by_step[2]
        assert resumed_digests.batch_by_step[2] == continuous_digests.batch_by_step[2]
        assert continuous_state.state_digests[1] == checkpoint_state_digests

        continuous_step_one = next(
            row for row in continuous.state.log_history if row.get("zo_step") == 1
        )
        for key in (
            "zo_loss",
            "zo_loss_plus",
            "zo_loss_minus",
            "zo_projected_grad",
        ):
            assert stopped_step_one[key] == pytest.approx(
                continuous_step_one[key], abs=1e-6
            )
        continuous_step_two = next(
            row for row in continuous.state.log_history if row.get("zo_step") == 2
        )
        for key in ("zo_loss", "zo_loss_plus", "zo_loss_minus"):
            assert resumed_step_two[key] == pytest.approx(
                continuous_step_two[key], abs=5e-3
            )
        assert resumed_step_two["zo_projected_grad"] == pytest.approx(
            continuous_step_two["zo_projected_grad"],
            rel=6e-2,
            abs=2.0,
        )
        assert resumed_step_two["zo_applied_learning_rate"] == pytest.approx(
            continuous_step_two["zo_applied_learning_rate"], abs=0.0
        )
    finally:
        _cleanup_engine(engine)


@pytest.mark.e2e
@pytest.mark.gpu
def test_hf_trainer_runs_real_es_rollout_end_to_end(tmp_path: Path) -> None:
    _skip_without_e2e_gpu()
    _prepare_e2e_environment()
    engine: ZOVLLMEngine | None = None
    try:
        engine, zo_model, _ = _build_real_vllm_zo_model(
            include_embeddings=True,
            accumulate_updates=True,
            estimator="evolution_strategy",
            population_size=2,
            query_microbatch_size=2,
            sigma=1e-3,
            reward_shaping="none",
        )
        dataset = Dataset.from_list(
            [{"prompt": "Review sentiment:", "target": "nonempty"}]
        )

        def collate(rows):
            return {
                "rollout_prompts": [row["prompt"] for row in rows],
                "rollout_targets": [row["target"] for row in rows],
            }

        def reward_fn(text, target):
            del target
            return float(len(text) > 0)

        trainer = ZOTrainer(
            model=ZORolloutTrainerModel(
                zo_model,
                reward_fn=reward_fn,
                max_tokens=1,
                seed=0,
            ),
            args=ZOTrainerArguments(
                output_dir=str(tmp_path / "es_trainer"),
                max_steps=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=1e-7,
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="steps",
                save_steps=1,
                zo_checkpoint_mode="native",
                disable_tqdm=True,
                report_to=[],
                remove_unused_columns=False,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=collate,
            compute_metrics=lambda prediction: {
                "reward": float(prediction.predictions[:, 0].mean())
            },
            checkpoint_handler=ZOVLLMCheckpointHandler(
                checkpoint_mode="native",
                llm=engine.llm,
                weight_sync=zo_model.weight_sync,
                update_state=zo_model.update_state,
            ),
        )

        train_result = trainer.train()

        assert train_result.global_step == 1
        train_metrics = next(
            row for row in trainer.state.log_history if row.get("zo_step") == 1
        )
        assert train_metrics["zo_estimator_query_count"] == 2
        assert train_metrics["zo_applied_learning_rate"] == pytest.approx(1e-7)
        eval_metrics = next(
            row for row in trainer.state.log_history if "eval_reward" in row
        )
        assert "eval_loss" in eval_metrics
        checkpoint = tmp_path / "es_trainer" / "checkpoint-1"
        metadata = read_zo_checkpoint_metadata(checkpoint)
        assert metadata["payload"]["materialized_effective_delta"] is True
        assert _checkpoint_native_state_digests(checkpoint)

        _cleanup_engine(engine)
        engine = None
        del trainer, zo_model

        engine, resumed_zo_model, _ = _build_real_vllm_zo_model(
            include_embeddings=True,
            accumulate_updates=True,
            estimator="evolution_strategy",
            population_size=2,
            query_microbatch_size=2,
            sigma=1e-3,
            reward_shaping="none",
        )
        resumed = ZOTrainer(
            model=ZORolloutTrainerModel(
                resumed_zo_model,
                reward_fn=reward_fn,
                max_tokens=1,
                seed=0,
            ),
            args=ZOTrainerArguments(
                output_dir=str(tmp_path / "es_trainer"),
                max_steps=2,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=1e-7,
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="steps",
                save_steps=1,
                zo_checkpoint_mode="native",
                disable_tqdm=True,
                report_to=[],
                remove_unused_columns=False,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=collate,
            compute_metrics=lambda prediction: {
                "reward": float(prediction.predictions[:, 0].mean())
            },
            checkpoint_handler=ZOVLLMCheckpointHandler(
                checkpoint_mode="native",
                llm=engine.llm,
                weight_sync=resumed_zo_model.weight_sync,
                update_state=resumed_zo_model.update_state,
            ),
        )

        resumed_result = resumed.train(resume_from_checkpoint=str(checkpoint))

        assert resumed_result.global_step == 2
        assert (tmp_path / "es_trainer" / "checkpoint-2").exists()
    finally:
        _cleanup_engine(engine)
