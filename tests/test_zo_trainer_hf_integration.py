from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, HfArgumentParser, TrainerCallback

from zo_trainer import (
    IGNORE_INDEX,
    VLLMDataCollator,
    ZO_CHECKPOINT_METADATA_NAME,
    ZOLogitsOutput,
    ZOTrainer,
    ZOTrainerArguments,
    ZOTrainerModel,
    ZORolloutTrainerModel,
    ZOVLLMCheckpointHandler,
    build_causal_lm_preprocess,
    build_target_lm_preprocess,
    read_zo_checkpoint_metadata,
)
from zo_trainer.modeling import hf_batch_to_token_groups
from zo_vllm.core.probe_results import ProbeTiming
from zo_vllm.core.token_scores import slice_score_result
from zo_vllm.engine import TokenScoreResult
from zo_vllm.training.direction import RolloutProbeBatch, TokenProbeBatch
from zo_vllm.training.zo_step import ZOPendingStep, ZOStepResult
from zo_vllm.tasks.boolq import BoolQRow
from zo_vllm.tasks.hf_preprocessing import (
    build_objective_hf_dataset,
    build_sst2_prompt_classification_preprocess,
)
from zo_vllm.tasks.squad import SquadRow
from zo_vllm.tasks.sst2 import SST2Row
from zo_vllm.tasks.superglue import dataset_to_superglue_rows


class TinyTokenizer:
    pad_token_id = 0
    padding_side = "right"
    model_input_names = ["input_ids", "attention_mask"]
    def __call__(
        self,
        texts,
        *,
        max_length=None,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    ):
        del padding
        if isinstance(texts, str):
            return {
                "input_ids": self.encode(
                    texts,
                    max_length=max_length,
                    truncation=truncation,
                    add_special_tokens=add_special_tokens,
                ),
                "attention_mask": [
                    1
                    for _ in self.encode(
                        texts,
                        max_length=max_length,
                        truncation=truncation,
                        add_special_tokens=add_special_tokens,
                    )
                ],
            }
        rows = [
            self.encode(
                text,
                max_length=max_length,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
            )
            for text in texts
        ]
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }

    def encode(
        self,
        text: str,
        *,
        max_length=None,
        truncation=True,
        add_special_tokens=True,
    ) -> list[int]:
        ids = [ord(ch) % 50 + 3 for ch in text if ch != " "]
        if add_special_tokens:
            ids = [2, *ids]
        if max_length is not None and truncation:
            ids = ids[: int(max_length)]
        return ids

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **_: Any,
    ):
        del padding
        rows = [dict(item) for item in encoded_inputs]
        target_length = max(len(row["input_ids"]) for row in rows)
        if max_length is not None:
            target_length = max(target_length, int(max_length))
        if pad_to_multiple_of:
            multiple = int(pad_to_multiple_of)
            target_length = ((target_length + multiple - 1) // multiple) * multiple

        padded: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
        }
        if "labels" in rows[0]:
            padded["labels"] = []
        for row in rows:
            pad_len = target_length - len(row["input_ids"])
            padded["input_ids"].append(
                list(row["input_ids"]) + [self.pad_token_id] * pad_len
            )
            padded["attention_mask"].append(list(row["attention_mask"]) + [0] * pad_len)
            if "labels" in padded:
                padded["labels"].append(list(row["labels"]) + [IGNORE_INDEX] * pad_len)
        if return_tensors == "pt":
            return {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in padded.items()
            }
        return padded

    def save_pretrained(self, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "tiny_tokenizer.json"
        path.write_text(json.dumps({"pad_token_id": self.pad_token_id}) + "\n")
        return (str(path),)


def _fake_weight_sync() -> SimpleNamespace:
    return SimpleNamespace(
        hf_to_vllm_mapping={
            "model.layers.0.self_attn.q_proj.weight": (
                "model.layers.0.self_attn.qkv_proj.weight"
            )
        },
        hf_to_slice={"model.layers.0.self_attn.q_proj.weight": (0, 8)},
    )


class CountingTokenizer(TinyTokenizer):
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        self.encode_calls += 1
        return super().encode(*args, **kwargs)


class FakeEngine:
    plus_id = 11
    minus_id = 12

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def forward_token_logits(
        self, token_id_groups, *, labels=None, lora_ids=None, **kwargs
    ):
        active_label_rows = self._active_label_rows(token_id_groups, labels)
        active_labels = [label for row in active_label_rows for label in row]
        vocab_size = max(64, max(active_labels, default=0) + 1)
        logits = torch.zeros((len(active_labels), vocab_size), dtype=torch.float32)
        if lora_ids and set(lora_ids) == {self.plus_id}:
            target_logit = 0.0
        elif lora_ids and set(lora_ids) == {self.minus_id}:
            target_logit = 1.0
        else:
            target_logit = 0.5
        for index, label in enumerate(active_labels):
            logits[index, int(label)] = target_logit
        self.calls.append(
            {
                "token_id_groups": [list(row) for row in token_id_groups],
                "labels": None if labels is None else [list(row) for row in labels],
                "lora_ids": None
                if lora_ids is None
                else [int(item) for item in lora_ids],
                **kwargs,
            }
        )
        label_tensor = torch.tensor(active_labels, dtype=torch.long)
        return SimpleNamespace(
            logits=logits,
            target_token_ids=label_tensor,
            request_loss_token_counts=tuple(len(row) for row in active_label_rows),
            timing=ProbeTiming(),
        )

    def score_token_groups(
        self,
        token_id_groups,
        *,
        loss_token_lens=None,
        labels=None,
        lora_ids=None,
        **kwargs,
    ):
        del labels
        base_nll = [1.5, 0.5, 0.25, 2.0]
        request_nll = [
            base_nll[index % len(base_nll)] for index in range(len(token_id_groups))
        ]
        if lora_ids and set(lora_ids) == {self.plus_id}:
            request_nll = [value + 0.1 for value in request_nll]
        elif lora_ids and set(lora_ids) == {self.minus_id}:
            request_nll = [value - 0.1 for value in request_nll]
        request_num_tokens = (
            [1 for _ in token_id_groups]
            if loss_token_lens is None
            else [int(item) for item in loss_token_lens]
        )
        nll_sum = float(
            sum(
                float(nll) * float(count)
                for nll, count in zip(request_nll, request_num_tokens)
            )
        )
        num_tokens = int(sum(request_num_tokens))
        self.calls.append(
            {
                "token_id_groups": [list(row) for row in token_id_groups],
                "labels": None,
                "loss_token_lens": None
                if loss_token_lens is None
                else [int(item) for item in loss_token_lens],
                "lora_ids": None
                if lora_ids is None
                else [int(item) for item in lora_ids],
                **kwargs,
            }
        )
        return TokenScoreResult(
            loss=nll_sum / float(max(num_tokens, 1)),
            nll_sum=nll_sum,
            num_tokens=num_tokens,
            request_nll=request_nll,
            request_num_tokens=request_num_tokens,
            detail={},
            raw={},
        )

    def forward_token_request_nll(
        self,
        token_id_groups,
        *,
        loss_token_lens,
        lora_ids=None,
        **kwargs,
    ):
        score = self.score_token_groups(
            token_id_groups,
            loss_token_lens=loss_token_lens,
            lora_ids=lora_ids,
            **kwargs,
        )
        return SimpleNamespace(
            request_nll=torch.tensor(score.request_nll, dtype=torch.float32),
            timing=ProbeTiming(),
        )

    @staticmethod
    def _active_label_rows(token_id_groups, labels) -> list[list[int]]:
        if labels is None:
            return [[int(token) for token in row[1:]] for row in token_id_groups]
        return [
            [int(label) for label in row[1:] if int(label) != IGNORE_INDEX]
            for row in labels
        ]

    def set_plus_minus_directions(self, directions, *, eps: float, step: int = 0):
        del directions
        return {"eps": float(eps), "step": int(step)}


class FakeHFZORuntime:
    def __init__(self) -> None:
        self.engine = FakeEngine()
        self.direction_provider = _FakeDirectionProviderState()
        self.config = type(
            "Config",
            (),
            {"max_logits_tokens": 128, "loss_impl": "logprobs"},
        )()
        self.step_batches: list[TokenProbeBatch] = []
        self.step_losses: list[tuple[float, float]] = []
        self.applied_learning_rates: list[float] = []
        self.applied_weight_decays: list[float] = []
        self.direction_slot_state_invalidated = False

    def invalidate_direction_slot_state(self) -> None:
        self.direction_slot_state_invalidated = True

    def estimate_with_score_fn(
        self,
        batch: TokenProbeBatch,
        *,
        step: int,
        score_fn,
    ):
        self.step_batches.append(batch)
        loss_plus = float(
            score_fn(
                token_id_groups=batch.token_id_groups,
                labels=batch.labels,
                lora_ids=[self.engine.plus_id] * len(batch.token_id_groups),
            ).loss
        )
        loss_minus = float(
            score_fn(
                token_id_groups=batch.token_id_groups,
                labels=batch.labels,
                lora_ids=[self.engine.minus_id] * len(batch.token_id_groups),
            ).loss
        )
        self.step_losses.append((loss_plus, loss_minus))
        return self._pending_step(
            step=step,
            reported_loss=(loss_plus + loss_minus) / 2.0,
            loss_plus=loss_plus,
            loss_minus=loss_minus,
            projected_grad=(loss_plus - loss_minus) / 0.2,
        )

    def _pending_step(
        self,
        *,
        step: int,
        reported_loss: float,
        loss_plus: float,
        loss_minus: float,
        projected_grad: float,
    ) -> ZOPendingStep:
        def apply(learning_rate: float, weight_decay: float) -> ZOStepResult:
            self.applied_learning_rates.append(float(learning_rate))
            self.applied_weight_decays.append(float(weight_decay))
            return ZOStepResult(
                step=int(step),
                learning_rate=float(learning_rate),
                reported_loss=float(reported_loss),
                loss_plus=float(loss_plus),
                loss_minus=float(loss_minus),
                projected_grad=float(projected_grad),
                update_scale=float(projected_grad),
                direction_refreshed=False,
            )

        return ZOPendingStep(
            step=int(step),
            reported_loss=float(reported_loss),
            _apply_fn=apply,
        )


class _FakeDirectionProviderState:
    def __init__(self) -> None:
        self.loaded_state: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {"type": "fake_direction_provider", "value": 1}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("type") != "fake_direction_provider":
            raise ValueError("invalid fake direction provider state")
        self.loaded_state = dict(state)


class RecordingCheckpointHandler:
    def __init__(self, checkpoint_mode: str = "metadata") -> None:
        self.checkpoint_mode = checkpoint_mode
        self.saved_models: list[str] = []
        self.saved_checkpoints: list[dict[str, Any]] = []
        self.loaded_checkpoints: list[str] = []

    def save_checkpoint(
        self,
        output_dir: str,
        *,
        step: int,
        metrics: Mapping[str, Any] | None,
        reason: str,
    ):
        record = {
            "path": Path(output_dir).name,
            "step": int(step),
            "metrics": {} if metrics is None else dict(metrics),
            "reason": reason,
        }
        self.saved_checkpoints.append(record)
        if reason == "save_model":
            self.saved_models.append(Path(output_dir).name)
            (Path(output_dir) / "runtime_model.txt").write_text("saved\n")
        (Path(output_dir) / "runtime_checkpoint.json").write_text(
            json.dumps(record, sort_keys=True) + "\n"
        )
        return {
            "mode": self.checkpoint_mode,
            "loadable": self.checkpoint_mode != "metadata",
            **record,
        }

    def load_checkpoint(self, checkpoint_dir: str):
        self.loaded_checkpoints.append(Path(checkpoint_dir).name)
        return {
            "mode": self.checkpoint_mode,
            "loadable": True,
            "loaded": Path(checkpoint_dir).name,
        }


def _args(tmp_path: Path, **overrides: Any) -> ZOTrainerArguments:
    values = {
        "output_dir": str(tmp_path),
        "max_steps": 2,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "logging_steps": 1,
        "eval_strategy": "no",
        "save_strategy": "no",
        "disable_tqdm": True,
        "report_to": [],
        "remove_unused_columns": False,
        "learning_rate": 0.0,
    }
    values.update(overrides)
    return ZOTrainerArguments(**values)


def _tokenized_dataset(tokenizer: TinyTokenizer) -> Dataset:
    raw = Dataset.from_list([{"text": "aa"}, {"text": "bbb"}, {"text": "cccc"}])
    return raw.map(
        build_causal_lm_preprocess(tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )


def test_trainer_uses_hf_map_and_collator_for_zo_plus_minus_losses(
    tmp_path: Path,
) -> None:
    tokenizer = TinyTokenizer()
    runtime = FakeHFZORuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path, max_steps=2),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
    )

    output = trainer.train()

    assert output.global_step == 2
    assert [len(batch.token_id_groups) for batch in runtime.step_batches] == [2, 1]
    assert all(call["lora_ids"] for call in runtime.engine.calls)
    assert {tuple(call["lora_ids"]) for call in runtime.engine.calls} == {
        (runtime.engine.plus_id, runtime.engine.plus_id),
        (runtime.engine.minus_id, runtime.engine.minus_id),
        (runtime.engine.plus_id,),
        (runtime.engine.minus_id,),
    }
    assert trainer.state.log_history
    assert "zo_projected_grad" in trainer.state.log_history[0]


def test_rollout_model_uses_native_trainer_train_and_evaluate(tmp_path: Path) -> None:
    class FakeRolloutEngine:
        def __init__(self) -> None:
            self.clean_lora_ids: list[int | None] = []

        def generate_with_lora_id(self, prompts, *, lora_id, **kwargs):
            del kwargs
            self.clean_lora_ids.append(lora_id)
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text=str(prompt), token_ids=[1])]
                )
                for prompt in prompts
            ]

    class FakeRolloutRuntime(FakeHFZORuntime):
        def __init__(self) -> None:
            super().__init__()
            self.engine = FakeRolloutEngine()
            self.rollout_batches: list[RolloutProbeBatch] = []

        def clean_lora_id_for_score(self) -> int:
            return 17

        def estimate(self, batch: RolloutProbeBatch, *, step: int):
            self.rollout_batches.append(batch)
            rewards = [
                float(batch.rollout_reward_fn(prompt, target))
                for prompt, target in zip(
                    batch.rollout_prompts, batch.rollout_targets
                )
            ]
            reward = sum(rewards) / len(rewards)
            return self._pending_step(
                step=step,
                reported_loss=-reward,
                loss_plus=-reward,
                loss_minus=-reward,
                projected_grad=0.0,
            )

    def collate(rows):
        return {
            "rollout_prompts": [row["prompt"] for row in rows],
            "rollout_targets": [row["target"] for row in rows],
        }

    def reward_fn(text, target):
        return float(text == target)

    def metrics_fn(text, target, reward_result):
        del text, target
        return {"accuracy": float(reward_result)}

    def compute_metrics(prediction):
        return {
            "reward": float(prediction.predictions[:, 0].mean()),
            "accuracy": float(prediction.predictions[:, 1].mean()),
        }

    runtime = FakeRolloutRuntime()
    dataset = Dataset.from_list(
        [
            {"prompt": "correct", "target": "correct"},
            {"prompt": "wrong", "target": "expected"},
        ]
    )
    trainer = ZOTrainer(
        model=ZORolloutTrainerModel(
            runtime,
            reward_fn=reward_fn,
            max_tokens=4,
            metric_names=("reward", "accuracy"),
            metrics_fn=metrics_fn,
        ),
        args=_args(
            tmp_path,
            max_steps=1,
            learning_rate=0.1,
            eval_strategy="steps",
            eval_steps=1,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )

    result = trainer.train()

    assert result.global_step == 1
    assert len(runtime.rollout_batches) == 1
    assert runtime.applied_learning_rates == pytest.approx([0.1])
    assert runtime.engine.clean_lora_ids == [17]
    eval_metrics = next(
        row for row in trainer.state.log_history if "eval_reward" in row
    )
    assert eval_metrics["eval_loss"] == pytest.approx(-0.5)
    assert eval_metrics["eval_reward"] == pytest.approx(0.5)
    assert eval_metrics["eval_accuracy"] == pytest.approx(0.5)


def test_hf_linear_scheduler_controls_runtime_learning_rate(tmp_path: Path) -> None:
    runtime = FakeHFZORuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(
            tmp_path,
            max_steps=3,
            per_device_train_batch_size=1,
            learning_rate=0.3,
            lr_scheduler_type="linear",
            warmup_steps=0,
        ),
        train_dataset=_tokenized_dataset(TinyTokenizer()),
    )

    trainer.train()

    assert runtime.applied_learning_rates == pytest.approx([0.3, 0.2, 0.1])


def test_hf_optimizer_callbacks_bracket_real_zo_update(tmp_path: Path) -> None:
    runtime = FakeHFZORuntime()
    observations: list[tuple[str, int]] = []

    class OptimizerBoundaryCallback(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            del args, state, control, kwargs
            observations.append(("pre", len(runtime.applied_learning_rates)))

        def on_optimizer_step(self, args, state, control, **kwargs):
            del args, state, control, kwargs
            observations.append(("post", len(runtime.applied_learning_rates)))

    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path, max_steps=1, learning_rate=0.2),
        train_dataset=_tokenized_dataset(TinyTokenizer()),
        callbacks=[OptimizerBoundaryCallback()],
    )

    trainer.train()

    assert observations == [("pre", 0), ("post", 1)]


def test_hf_optimizer_passes_weight_decay_to_runtime_update(tmp_path: Path) -> None:
    runtime = FakeHFZORuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(
            tmp_path,
            max_steps=1,
            learning_rate=0.2,
            weight_decay=0.15,
        ),
        train_dataset=_tokenized_dataset(TinyTokenizer()),
    )

    trainer.train()

    assert runtime.applied_weight_decays == pytest.approx([0.15])


def test_hf_scheduler_warmup_controls_runtime_learning_rate(tmp_path: Path) -> None:
    runtime = FakeHFZORuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(
            tmp_path,
            max_steps=4,
            per_device_train_batch_size=1,
            learning_rate=0.4,
            lr_scheduler_type="linear",
            warmup_steps=2,
        ),
        train_dataset=_tokenized_dataset(TinyTokenizer()),
    )

    trainer.train()

    assert runtime.applied_learning_rates == pytest.approx([0.0, 0.2, 0.4, 0.2])


def test_hf_scheduler_resumes_at_next_runtime_step(tmp_path: Path) -> None:
    class StopAfterOneStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step == 1:
                control.should_training_stop = True
            return control

    tokenizer = TinyTokenizer()
    first_runtime = FakeHFZORuntime()
    first = ZOTrainer(
        model=first_runtime,
        checkpoint_handler=RecordingCheckpointHandler("native"),
        args=_args(
            tmp_path,
            max_steps=3,
            per_device_train_batch_size=1,
            learning_rate=0.3,
            lr_scheduler_type="linear",
            warmup_steps=0,
            save_strategy="steps",
            save_steps=1,
            zo_checkpoint_mode="native",
        ),
        train_dataset=_tokenized_dataset(tokenizer),
        callbacks=[StopAfterOneStep()],
    )
    first.train()
    assert first_runtime.applied_learning_rates == pytest.approx([0.3])
    assert first._zo_optimizer().step_count == 1

    resumed_runtime = FakeHFZORuntime()
    resumed = ZOTrainer(
        model=resumed_runtime,
        checkpoint_handler=RecordingCheckpointHandler("native"),
        args=_args(
            tmp_path,
            max_steps=3,
            per_device_train_batch_size=1,
            learning_rate=0.3,
            lr_scheduler_type="linear",
            warmup_steps=0,
            save_strategy="no",
            zo_checkpoint_mode="native",
        ),
        train_dataset=_tokenized_dataset(tokenizer),
    )
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))

    assert resumed_runtime.applied_learning_rates == pytest.approx([0.2, 0.1])
    assert resumed._zo_optimizer().step_count == 3


def test_trainer_rejects_non_sgd_optimizer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only optim='sgd'"):
        _args(tmp_path, optim="adamw_torch")


def test_trainer_rejects_autograd_gradient_clipping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires max_grad_norm=0"):
        _args(tmp_path, max_grad_norm=1.0)


def test_trainer_rejects_untyped_runtime_step_result(tmp_path: Path) -> None:
    class UntypedRuntime(FakeHFZORuntime):
        def estimate_with_score_fn(self, batch, *, step, score_fn):
            del batch, step, score_fn
            return {"loss": 1.0}

    trainer = ZOTrainer(
        model=UntypedRuntime(),
        args=_args(tmp_path, max_steps=1),
        train_dataset=_tokenized_dataset(TinyTokenizer()),
    )

    with pytest.raises(TypeError, match="stage requires ZOPendingStep"):
        trainer.train()


def test_trainer_wraps_hf_standard_model_argument(tmp_path: Path) -> None:
    tokenizer = TinyTokenizer()
    runtime = FakeHFZORuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path, max_steps=1),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
        processing_class=tokenizer,
    )

    output = trainer.train()

    assert trainer.model.zo_model is runtime
    assert output.global_step == 1
    assert len(runtime.step_batches) == 1


def test_trainer_evaluate_uses_model_forward_loss_without_lora(
    tmp_path: Path,
) -> None:
    tokenizer = TinyTokenizer()
    runtime = FakeHFZORuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path),
        eval_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
    )

    metrics = trainer.evaluate()

    assert "eval_loss" in metrics
    assert runtime.engine.calls
    assert all(call["lora_ids"] is None for call in runtime.engine.calls)


def test_trainer_compute_metrics_receives_compact_logits_and_labels(
    tmp_path: Path,
) -> None:
    tokenizer = TinyTokenizer()
    runtime = FakeHFZORuntime()
    seen: dict[str, Any] = {}

    def compute_metrics(prediction):
        seen["predictions_shape"] = tuple(prediction.predictions.shape)
        seen["labels_shape"] = tuple(prediction.label_ids.shape)
        assert prediction.predictions.shape[0] == prediction.label_ids.shape[0]
        return {"num_loss_labels": int(prediction.label_ids.shape[0])}

    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path),
        eval_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()

    assert seen["predictions_shape"][1] >= 64
    assert len(seen["labels_shape"]) == 1
    assert metrics["eval_num_loss_labels"] == seen["labels_shape"][0]


def test_target_lm_preprocess_masks_prompt_and_hf_collator_pads_labels() -> None:
    tokenizer = TinyTokenizer()
    raw = Dataset.from_list(
        [
            {"prompt": "Q:", "target": " yes"},
            {"prompt": "Long:", "target": " no"},
        ]
    )
    tokenized = raw.map(
        build_target_lm_preprocess(tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )
    batch = DataCollatorWithPadding(tokenizer)([tokenized[0], tokenized[1]])

    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"] == IGNORE_INDEX).any()
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_hf_batch_to_token_groups_rejects_dense_tensor_batches() -> None:
    with pytest.raises(TypeError, match="ragged nested sequence"):
        hf_batch_to_token_groups(
            {
                "input_ids": torch.tensor([[0, 0, 8, 9], [0, 7, 6, 5]]),
                "attention_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
                "labels": torch.tensor(
                    [
                        [IGNORE_INDEX, IGNORE_INDEX, 8, 9],
                        [IGNORE_INDEX, 7, 6, 5],
                    ]
                ),
            }
        )


def test_vllm_collator_keeps_ragged_rows_without_padding_or_copying() -> None:
    short_ids = [2, 3]
    long_ids = [4, 5, 6, 7]
    short_labels = [IGNORE_INDEX, 3]
    long_labels = [IGNORE_INDEX, IGNORE_INDEX, 6, 7]
    batch = VLLMDataCollator()(
        [
            {"input_ids": short_ids, "labels": short_labels},
            {"input_ids": long_ids, "labels": long_labels},
        ]
    )

    assert "attention_mask" not in batch
    assert [len(row) for row in batch["input_ids"]] == [2, 4]
    assert batch["input_ids"][0] is short_ids
    assert batch["input_ids"][1] is long_ids
    assert batch["labels"][0] is short_labels
    assert batch["labels"][1] is long_labels

    token_groups, token_labels = hf_batch_to_token_groups(batch)

    assert token_groups[0] is short_ids
    assert token_groups[1] is long_ids
    assert token_labels is not None
    assert token_labels[0] is short_labels
    assert token_labels[1] is long_labels


def test_hf_forward_rejects_runtime_private_batch_fields() -> None:
    model = ZOTrainerModel(FakeHFZORuntime())

    with pytest.raises(ValueError, match="runtime-private state"):
        model(input_ids=[[1, 2]], labels=[[-100, 2]], _zo_slot=[1])


def test_target_lm_forward_exposes_scorer_profile_fields() -> None:
    tokenizer = TinyTokenizer()
    runtime = FakeHFZORuntime()
    dataset = _tokenized_dataset(tokenizer)
    batch = VLLMDataCollator()([dataset[0], dataset[1]])

    outputs = ZOTrainerModel(runtime)(**batch)

    profile = outputs.timing.profile_seconds()
    assert {
        "scorer_forward_unpack",
        "scorer_engine_call",
        "scorer_forward_total",
    } <= profile.keys()
    assert "loss" not in outputs


def test_prompt_classification_forward_exposes_scorer_profile_fields() -> None:
    tokenizer = TinyTokenizer()
    raw = Dataset.from_list([{"sentence": "a great film", "label": 1}])
    tokenized = raw.map(
        build_sst2_prompt_classification_preprocess(tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )
    batch = VLLMDataCollator()([tokenized[0]])

    outputs = ZOTrainerModel(FakeHFZORuntime())(**batch)

    profile = outputs.timing.profile_seconds()
    assert {
        "scorer_forward_unpack",
        "scorer_engine_call",
        "scorer_output_postprocess",
        "scorer_forward_total",
    } <= profile.keys()
    assert "loss" not in outputs


def test_sst2_prompt_classification_returns_hf_logits_and_metrics(
    tmp_path: Path,
) -> None:
    tokenizer = TinyTokenizer()
    raw = Dataset.from_list(
        [
            {"sentence": "a great film", "label": 1},
            {"sentence": "a dull film", "label": 0},
        ]
    )
    tokenized = raw.map(
        build_sst2_prompt_classification_preprocess(tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )
    seen: dict[str, Any] = {}

    def compute_metrics(prediction):
        predictions = prediction.predictions.argmax(axis=-1)
        seen["predictions"] = predictions.tolist()
        seen["labels"] = prediction.label_ids.tolist()
        return {"accuracy": float((predictions == prediction.label_ids).mean())}

    runtime = FakeHFZORuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path),
        eval_dataset=tokenized,
        data_collator=VLLMDataCollator(),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()

    assert metrics["eval_accuracy"] == 1.0
    assert seen == {"predictions": [1, 0], "labels": [1, 0]}
    score_call = runtime.engine.calls[0]
    assert score_call["loss_token_lens"] is not None
    assert len(score_call["token_id_groups"]) == 4


def test_sst2_prompt_classification_preprocess_does_not_duplicate_option_token() -> (
    None
):
    tokenizer = TinyTokenizer()
    raw = Dataset.from_list([{"sentence": "short", "label": 1}])
    tokenized = raw.map(
        build_sst2_prompt_classification_preprocess(tokenizer, max_length=64),
        batched=True,
        remove_columns=raw.column_names,
    )

    expected = tokenizer.encode("short It was terrible", add_special_tokens=True)
    assert tokenized[0]["input_ids"][0] == expected


def test_sst2_prompt_classification_training_preserves_option_objective(
    tmp_path: Path,
) -> None:
    tokenizer = TinyTokenizer()
    raw = Dataset.from_list(
        [
            {"sentence": "a great film", "label": 1},
            {"sentence": "a dull film", "label": 0},
        ]
    )
    tokenized = raw.map(
        build_sst2_prompt_classification_preprocess(tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )
    runtime = FakeHFZORuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path, max_steps=1),
        train_dataset=tokenized,
        data_collator=VLLMDataCollator(),
    )

    trainer.train()

    batch = runtime.step_batches[0]
    assert len(batch.token_id_groups) == 4
    assert batch.labels is None
    assert batch.loss_token_lens is not None
    assert len(batch.loss_token_lens) == 4
    assert {tuple(call["lora_ids"]) for call in runtime.engine.calls} == {
        (runtime.engine.plus_id,) * 4,
        (runtime.engine.minus_id,) * 4,
    }
    assert all(call["loss_token_lens"] is not None for call in runtime.engine.calls)
    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[-1.5, -0.5], [-0.25, -2.0]]),
        torch.tensor([1, 0]),
    ).item()
    assert runtime.step_losses[0][0] == pytest.approx(expected)
    assert runtime.step_losses[0][1] == pytest.approx(expected)


def test_batched_plus_minus_slices_preserve_hf_classification_loss(
    tmp_path: Path,
) -> None:
    class BatchedPlusMinusRuntime(FakeHFZORuntime):
        def estimate_with_score_fn(self, batch, *, step, score_fn):
            request_count = len(batch.token_id_groups)
            combined = score_fn(
                token_id_groups=batch.token_id_groups + batch.token_id_groups,
                loss_token_lens=batch.loss_token_lens + batch.loss_token_lens,
                labels=None,
                lora_ids=[self.engine.plus_id] * request_count
                + [self.engine.minus_id] * request_count,
            )
            plus = slice_score_result(
                combined,
                0,
                request_count,
                loss_impl="logprobs",
            )
            minus = slice_score_result(
                combined,
                request_count,
                2 * request_count,
                loss_impl="logprobs",
            )
            self.step_losses.append((float(plus.loss), float(minus.loss)))
            projected_grad = float(plus.loss - minus.loss) / 0.2
            return self._pending_step(
                step=step,
                reported_loss=(float(plus.loss) + float(minus.loss)) / 2.0,
                loss_plus=float(plus.loss),
                loss_minus=float(minus.loss),
                projected_grad=projected_grad,
            )

    tokenizer = TinyTokenizer()
    raw = Dataset.from_list(
        [
            {"sentence": "a great film", "label": 1},
            {"sentence": "a dull film", "label": 0},
        ]
    )
    tokenized = raw.map(
        build_sst2_prompt_classification_preprocess(tokenizer),
        batched=True,
        remove_columns=raw.column_names,
    )
    runtime = BatchedPlusMinusRuntime()
    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path, max_steps=1),
        train_dataset=tokenized,
        data_collator=VLLMDataCollator(),
    )

    trainer.train()

    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[-1.5, -0.5], [-0.25, -2.0]]),
        torch.tensor([1, 0]),
    ).item()
    assert runtime.step_losses == [(pytest.approx(expected), pytest.approx(expected))]


def test_probe_losses_use_hf_compute_loss_func_without_second_forward(
    tmp_path: Path,
) -> None:
    tokenizer = TinyTokenizer()
    runtime = FakeHFZORuntime()
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def compute_loss_func(outputs, labels, num_items_in_batch=None):
        del num_items_in_batch
        seen.append((tuple(outputs["logits"].shape), tuple(labels.shape)))
        return outputs["logits"].sum() * 0.0 + 3.25

    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path, max_steps=1),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
        compute_loss_func=compute_loss_func,
    )

    trainer.train()

    assert runtime.step_losses == [(3.25, 3.25)]
    assert len(seen) == 2
    assert len(runtime.engine.calls) == 2


def test_custom_loss_accepts_generic_logits_output(tmp_path: Path) -> None:
    trainer = ZOTrainer(
        model=FakeHFZORuntime(),
        args=_args(tmp_path, max_steps=1),
        compute_loss_func=lambda outputs, labels, num_items_in_batch=None: (
            outputs["logits"].square().mean() + labels.float().mean()
        ),
    )
    outputs = ZOLogitsOutput(
        logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        loss_labels=torch.tensor([0, 1]),
        timing=ProbeTiming(),
    )

    loss = trainer._compute_loss_from_outputs(outputs)

    assert loss.item() == pytest.approx(8.0)


def test_clean_forward_uses_runtime_effective_lora_slot() -> None:
    runtime = FakeHFZORuntime()
    runtime.clean_lora_id_for_score = lambda: runtime.engine.plus_id
    model = ZOTrainerModel(runtime)

    model(input_ids=[[1, 2, 3]], labels=[[-100, 2, 3]])

    assert runtime.engine.calls[-1]["lora_ids"] == [runtime.engine.plus_id]


def test_batched_causal_probe_losses_use_contiguous_request_offsets(
    tmp_path: Path,
) -> None:
    class BatchedCausalRuntime(FakeHFZORuntime):
        def estimate_with_score_fn(self, batch, *, step, score_fn):
            request_count = len(batch.token_id_groups)
            combined = score_fn(
                token_id_groups=batch.token_id_groups + batch.token_id_groups,
                labels=batch.labels + batch.labels,
                lora_ids=[self.engine.plus_id] * request_count
                + [self.engine.minus_id] * request_count,
            )
            plus = slice_score_result(
                combined,
                0,
                request_count,
                loss_impl="logprobs",
            )
            minus = slice_score_result(
                combined,
                request_count,
                2 * request_count,
                loss_impl="logprobs",
            )
            self.step_losses.append((float(plus.loss), float(minus.loss)))
            projected_grad = float(plus.loss - minus.loss) / 0.2
            return self._pending_step(
                step=step,
                reported_loss=(float(plus.loss) + float(minus.loss)) / 2.0,
                loss_plus=float(plus.loss),
                loss_minus=float(minus.loss),
                projected_grad=projected_grad,
            )

    tokenizer = TinyTokenizer()
    runtime = BatchedCausalRuntime()
    seen_label_counts: list[int] = []

    def compute_loss_func(outputs, labels, num_items_in_batch=None):
        del outputs, num_items_in_batch
        seen_label_counts.append(int(labels.numel()))
        return labels.float().sum() * 0.0 + float(labels.numel())

    trainer = ZOTrainer(
        model=runtime,
        args=_args(tmp_path, max_steps=1),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
        compute_loss_func=compute_loss_func,
    )

    trainer.train()

    assert len(seen_label_counts) == 2
    assert seen_label_counts[0] == seen_label_counts[1]
    assert seen_label_counts[0] > 0
    assert runtime.step_losses == [
        (
            pytest.approx(float(seen_label_counts[0])),
            pytest.approx(float(seen_label_counts[1])),
        )
    ]


@pytest.mark.parametrize(
    ("task_name", "objective_name", "example"),
    [
        (
            "superglue_boolq",
            "superglue_boolq_classification",
            {"passage": "Water is wet.", "question": "Is water wet", "answer": True},
        ),
        (
            "superglue_cb",
            "superglue_cb_classification",
            {"premise": "A dog runs.", "hypothesis": "An animal moves.", "label": 0},
        ),
        (
            "superglue_copa",
            "superglue_copa_classification",
            {
                "premise": "It rained.",
                "choice1": "The road got wet.",
                "choice2": "The road dried.",
                "question": "effect",
                "label": 0,
            },
        ),
        (
            "superglue_multirc",
            "superglue_multirc_classification",
            {
                "paragraph": "The sky is blue.",
                "question": "What color is the sky?",
                "answer": "blue",
                "label": 1,
            },
        ),
        (
            "superglue_record",
            "superglue_record_nll",
            {
                "passage": "Alice arrived. @highlight\nShe smiled.",
                "query": "@placeholder smiled.",
                "entities": ["Alice", "Bob"],
                "answers": ["Alice"],
            },
        ),
        (
            "superglue_rte",
            "superglue_rte_classification",
            {"premise": "A dog runs.", "hypothesis": "An animal moves.", "label": 0},
        ),
        (
            "superglue_wic",
            "superglue_wic_classification",
            {
                "word": "bank",
                "sentence1": "The bank approved it.",
                "sentence2": "She visited the bank.",
                "label": 1,
            },
        ),
        (
            "superglue_wsc",
            "superglue_wsc_classification",
            {
                "text": "Alice thanked Mary because she helped.",
                "span1_text": "Mary",
                "span2_text": "she",
                "label": 1,
            },
        ),
    ],
)
def test_all_superglue_objectives_materialize_ragged_hf_batches(
    tmp_path: Path,
    task_name: str,
    objective_name: str,
    example: dict[str, Any],
) -> None:
    tokenizer = TinyTokenizer()
    rows = dataset_to_superglue_rows([example], task_name=task_name)
    dataset = build_objective_hf_dataset(
        rows,
        tokenizer,
        objective_name=objective_name,
        max_length=128,
        max_new_tokens=16,
    )
    batch = VLLMDataCollator()([dataset[0]])

    assert batch
    assert all(not isinstance(value, torch.Tensor) for value in batch.values())
    assert "attention_mask" not in batch
    assert "objective_batch" not in batch
    trainer = ZOTrainer(
        model=FakeHFZORuntime(),
        args=_args(tmp_path / objective_name),
        eval_dataset=dataset,
        data_collator=VLLMDataCollator(),
    )
    assert torch.isfinite(torch.tensor(trainer.evaluate()["eval_loss"]))


@pytest.mark.parametrize(
    ("objective_name", "rows", "classification"),
    [
        ("sst2_classification", [SST2Row(sentence="A good film.", label=1)], True),
        (
            "boolq_classification",
            [BoolQRow(passage="Water is wet.", question="Is water wet?", label=1)],
            True,
        ),
        (
            "squad_nll",
            [
                SquadRow(
                    title="Water",
                    context="Water is wet.",
                    question="What is wet?",
                    answers=("Water",),
                )
            ],
            False,
        ),
    ],
)
def test_non_superglue_objectives_materialize_ragged_hf_batches(
    tmp_path: Path,
    objective_name: str,
    rows: list[Any],
    classification: bool,
) -> None:
    tokenizer = TinyTokenizer()
    dataset = build_objective_hf_dataset(
        rows,
        tokenizer,
        objective_name=objective_name,
        max_length=128,
        max_new_tokens=16,
    )
    del classification
    collator = VLLMDataCollator()
    batch = collator([dataset[0]])

    assert all(not isinstance(value, torch.Tensor) for value in batch.values())
    assert "attention_mask" not in batch
    assert "objective_batch" not in batch
    trainer = ZOTrainer(
        model=FakeHFZORuntime(),
        args=_args(tmp_path / objective_name),
        eval_dataset=dataset,
        data_collator=collator,
    )
    assert torch.isfinite(torch.tensor(trainer.evaluate()["eval_loss"]))


def test_objective_dataset_is_not_retokenized_during_trainer_evaluation(
    tmp_path: Path,
) -> None:
    tokenizer = CountingTokenizer()
    dataset = build_objective_hf_dataset(
        [SST2Row(sentence="A good film.", label=1)],
        tokenizer,
        objective_name="sst2_classification",
        max_length=128,
    )
    preprocessing_calls = tokenizer.encode_calls
    trainer = ZOTrainer(
        model=FakeHFZORuntime(),
        args=_args(tmp_path / "no_retokenize"),
        eval_dataset=dataset,
    )

    trainer.evaluate()

    assert preprocessing_calls > 0
    assert tokenizer.encode_calls == preprocessing_calls


def test_zo_trainer_arguments_parse_with_hf_argument_parser(tmp_path: Path) -> None:
    parser = HfArgumentParser(ZOTrainerArguments)
    (args,) = parser.parse_args_into_dataclasses(
        [
            "--output_dir",
            str(tmp_path),
            "--max_steps",
            "1",
            "--zo_checkpoint_mode",
            "lora",
            "--zo_log_runtime_metrics",
            "False",
            "--report_to",
            "none",
        ]
    )

    assert args.output_dir == str(tmp_path)
    assert args.max_steps == 1
    assert args.zo_checkpoint_mode == "lora"
    assert args.zo_log_runtime_metrics is False


def test_zo_trainer_saves_runtime_checkpoints_and_metadata(tmp_path: Path) -> None:
    tokenizer = TinyTokenizer()
    checkpoint_handler = RecordingCheckpointHandler()
    trainer = ZOTrainer(
        model=FakeHFZORuntime(),
        checkpoint_handler=checkpoint_handler,
        args=_args(
            tmp_path,
            max_steps=2,
            save_strategy="steps",
            save_steps=1,
            zo_checkpoint_mode="metadata",
        ),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
    )

    trainer.train()

    assert [record["step"] for record in checkpoint_handler.saved_checkpoints] == [1, 2]
    assert checkpoint_handler.saved_models == []
    checkpoint = tmp_path / "checkpoint-2"
    assert (checkpoint / "runtime_checkpoint.json").exists()
    assert (checkpoint / "trainer_state.json").exists()
    assert (checkpoint / "training_args.bin").exists()
    assert (checkpoint / "rng_state.pth").exists()
    metadata = read_zo_checkpoint_metadata(checkpoint)
    assert metadata["checkpoint_type"] == "zo_trainer_checkpoint"
    assert metadata["global_step"] == 2
    assert metadata["payload"]["step"] == 2
    assert (checkpoint / ZO_CHECKPOINT_METADATA_NAME).exists()


def test_zo_trainer_uses_handler_with_prewrapped_model(tmp_path: Path) -> None:
    handler = RecordingCheckpointHandler()
    trainer = ZOTrainer(
        model=ZOTrainerModel(FakeHFZORuntime()),
        checkpoint_handler=handler,
        args=_args(tmp_path),
    )

    trainer.save_model()

    assert handler.saved_models == [tmp_path.name]


def test_zo_trainer_rejects_checkpoint_mode_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="handler mode"):
        ZOTrainer(
            model=FakeHFZORuntime(),
            checkpoint_handler=RecordingCheckpointHandler("metadata"),
            args=_args(tmp_path, zo_checkpoint_mode="native"),
        )


def test_zo_trainer_requires_handler_for_loadable_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a runtime checkpoint_handler"):
        ZOTrainer(
            model=FakeHFZORuntime(),
            args=_args(tmp_path, zo_checkpoint_mode="lora"),
        )


def test_real_checkpoint_handler_writes_one_payload_with_global_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_save_lora_bank_checkpoint(**kwargs):
        calls.append(dict(kwargs))
        return {"lora_payload_path": str(Path(kwargs["checkpoint_dir"]) / "bank.pt")}

    monkeypatch.setattr(
        "zo_trainer.runtime.save_lora_bank_checkpoint",
        fake_save_lora_bank_checkpoint,
    )
    tokenizer = TinyTokenizer()
    trainer = ZOTrainer(
        model=FakeHFZORuntime(),
        checkpoint_handler=ZOVLLMCheckpointHandler(
            checkpoint_mode="lora",
            update_state=object(),
            weight_sync=_fake_weight_sync(),
        ),
        args=_args(
            tmp_path,
            max_steps=1,
            save_strategy="steps",
            save_steps=1,
            zo_checkpoint_mode="lora",
        ),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
    )

    trainer.train()

    assert len(calls) == 1
    assert calls[0]["step"] == 1
    assert calls[0]["raw_step"] == 1
    metadata = read_zo_checkpoint_metadata(tmp_path / "checkpoint-1")
    assert metadata["global_step"] == 1
    assert metadata["payload"]["step"] == 1
    layer_mapping = metadata["payload"]["runtime_manifest"]["layer_mapping"]
    assert layer_mapping["hf_to_slice"] == {
        "model.layers.0.self_attn.q_proj.weight": [0, 8]
    }
    assert len(layer_mapping["fingerprint"]) == 64


def test_native_checkpoint_ignores_immediate_update_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_save_effective_native_checkpoint(**kwargs):
        calls.append(dict(kwargs))
        return {"num_parts": 1}

    monkeypatch.setattr(
        "zo_trainer.runtime.save_effective_native_checkpoint",
        fake_save_effective_native_checkpoint,
    )
    handler = ZOVLLMCheckpointHandler(
        checkpoint_mode="native",
        llm=object(),
        weight_sync=_fake_weight_sync(),
        update_state=object(),
    )

    payload = handler.save_checkpoint(
        str(tmp_path),
        step=1,
        metrics=None,
        reason="checkpoint",
    )

    assert calls[0]["accumulated_update_state"] is None
    assert payload["loadable"] is True


def test_native_checkpoint_detects_accumulated_update_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    accumulated = SimpleNamespace(clean_directions_for_score=lambda: {})
    monkeypatch.setattr(
        "zo_trainer.runtime.save_effective_native_checkpoint",
        lambda **kwargs: calls.append(dict(kwargs)) or {"num_parts": 1},
    )
    handler = ZOVLLMCheckpointHandler(
        checkpoint_mode="native",
        llm=object(),
        weight_sync=_fake_weight_sync(),
        update_state=accumulated,
    )

    handler.save_checkpoint(
        str(tmp_path), step=1, metrics=None, reason="checkpoint"
    )

    assert calls[0]["accumulated_update_state"] is accumulated
    assert calls[0]["use_lora_bank_update"] is True


def test_zo_trainer_save_model_writes_hf_artifacts(tmp_path: Path) -> None:
    tokenizer = TinyTokenizer()
    checkpoint_handler = RecordingCheckpointHandler()
    trainer = ZOTrainer(
        model=FakeHFZORuntime(),
        checkpoint_handler=checkpoint_handler,
        args=_args(tmp_path, max_steps=1),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()

    assert checkpoint_handler.saved_models[-1] == tmp_path.name
    assert checkpoint_handler.saved_checkpoints[-1]["step"] == 1
    assert checkpoint_handler.saved_checkpoints[-1]["reason"] == "save_model"
    assert (tmp_path / "runtime_model.txt").exists()
    assert (tmp_path / "zo_checkpoint_metadata.json").exists()
    assert (tmp_path / "training_args.bin").exists()
    assert (tmp_path / "tiny_tokenizer.json").exists()
    saved_args = torch.load(tmp_path / "training_args.bin", weights_only=False)
    assert isinstance(saved_args, ZOTrainerArguments)
    assert saved_args.output_dir == str(tmp_path)


def test_zo_trainer_rejects_metadata_only_resume_checkpoint(tmp_path: Path) -> None:
    tokenizer = TinyTokenizer()
    first = ZOTrainer(
        model=FakeHFZORuntime(),
        checkpoint_handler=RecordingCheckpointHandler(),
        args=_args(
            tmp_path,
            max_steps=1,
            save_strategy="steps",
            save_steps=1,
            zo_checkpoint_mode="metadata",
        ),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
    )
    first.train()

    resumed = ZOTrainer(
        model=FakeHFZORuntime(),
        checkpoint_handler=RecordingCheckpointHandler(),
        args=_args(
            tmp_path,
            max_steps=2,
            save_strategy="no",
            zo_checkpoint_mode="metadata",
        ),
        train_dataset=_tokenized_dataset(tokenizer),
        data_collator=VLLMDataCollator(),
    )

    with pytest.raises(RuntimeError, match="metadata-only ZO checkpoint"):
        resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))


def test_zo_trainer_resumes_loadable_runtime_checkpoint(tmp_path: Path) -> None:
    tokenizer = TinyTokenizer()
    first_handler = RecordingCheckpointHandler("native")
    first = ZOTrainer(
        model=FakeHFZORuntime(),
        checkpoint_handler=first_handler,
        args=_args(
            tmp_path,
            max_steps=1,
            save_strategy="steps",
            save_steps=1,
            zo_checkpoint_mode="native",
        ),
        train_dataset=_tokenized_dataset(tokenizer),
    )
    first.train()

    resumed_handler = RecordingCheckpointHandler("native")
    resumed = ZOTrainer(
        model=FakeHFZORuntime(),
        checkpoint_handler=resumed_handler,
        args=_args(
            tmp_path,
            max_steps=2,
            save_strategy="no",
            zo_checkpoint_mode="native",
        ),
        train_dataset=_tokenized_dataset(tokenizer),
    )
    output = resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))

    assert output.global_step == 2
    assert resumed_handler.loaded_checkpoints == ["checkpoint-1"]
    assert resumed.model.zo_model.direction_slot_state_invalidated is True


def test_zo_trainer_rejects_checkpoint_without_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    trainer = ZOTrainer(
        model=FakeHFZORuntime(),
        checkpoint_handler=RecordingCheckpointHandler("native"),
        args=_args(tmp_path, zo_checkpoint_mode="native"),
    )

    with pytest.raises(FileNotFoundError, match="metadata not found"):
        trainer.train(resume_from_checkpoint=str(checkpoint))


def test_vllm_checkpoint_handler_metadata_mode_returns_nonloadable_payload(
    tmp_path: Path,
) -> None:
    handler = ZOVLLMCheckpointHandler(checkpoint_mode="metadata")

    payload = handler.save_checkpoint(
        str(tmp_path),
        step=4,
        metrics={"eval_loss": 1.25},
        reason="checkpoint",
    )

    assert payload["mode"] == "metadata"
    assert payload["loadable"] is False
    assert payload["metrics"] == {"eval_loss": 1.25}
    with pytest.raises(RuntimeError, match="metadata-only"):
        handler.load_checkpoint(str(tmp_path))
