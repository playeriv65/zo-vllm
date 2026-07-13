"""Hugging Face Trainer adapter for ZO-vLLM runtimes."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import Trainer
from transformers.loss.loss_utils import ForCausalLMLoss, fixed_cross_entropy
from transformers.trainer import TRAINING_ARGS_NAME

from zo_vllm.training.zo_step import ZOStepResult

from .arguments import ZOTrainerArguments
from .checkpointing import (
    ZO_DIRECTION_STATE_NAME,
    ZOCheckpointMetadata,
    ZOCheckpointHandler,
    load_direction_provider_state,
    read_zo_checkpoint_metadata,
    save_direction_provider_state,
    write_zo_checkpoint_metadata,
)
from .modeling import (
    CompactCausalOutput,
    OptionClassificationOutput,
    ZOTrainerModel,
    ZOTrainerRuntime,
)
from .optimizer import ZOSGDOptimizer
from .preprocessing import VLLMDataCollator
from .runtime import ZOVLLMCheckpointHandler


class ZOTrainer(Trainer):
    """A real Hugging Face Trainer subclass backed by ZO-vLLM runtime steps."""

    args: ZOTrainerArguments

    def __init__(
        self,
        model: nn.Module | Any | None = None,
        args: ZOTrainerArguments | None = None,
        *,
        checkpoint_handler: ZOCheckpointHandler | None = None,
        **kwargs: Any,
    ) -> None:
        if model is None:
            raise ValueError("ZOTrainer requires model")
        if args is None or not isinstance(args, ZOTrainerArguments):
            raise TypeError("ZOTrainer requires ZOTrainerArguments")
        if not isinstance(model, ZOTrainerModel):
            if not isinstance(model, ZOTrainerRuntime):
                raise TypeError(
                    "ZOTrainer model must implement the ZOTrainerRuntime protocol"
                )
            model = ZOTrainerModel(model)
        if not isinstance(model.zo_model, ZOTrainerRuntime):
            raise TypeError(
                "ZOTrainerModel runtime must implement the ZOTrainerRuntime protocol"
            )

        if args.gradient_accumulation_steps != 1:
            raise ValueError(
                "ZOTrainer currently requires gradient_accumulation_steps=1"
            )
        if not args.use_cpu:
            raise ValueError("ZOTrainer requires use_cpu=True; vLLM owns GPU devices")
        if args.remove_unused_columns:
            raise ValueError("ZOTrainer requires remove_unused_columns=False")
        if checkpoint_handler is None:
            if args.zo_checkpoint_mode != "metadata":
                raise ValueError(
                    f"zo_checkpoint_mode={args.zo_checkpoint_mode!r} requires a "
                    "runtime checkpoint_handler"
                )
            checkpoint_handler = ZOVLLMCheckpointHandler(checkpoint_mode="metadata")
        if checkpoint_handler.checkpoint_mode != args.zo_checkpoint_mode:
            raise ValueError(
                "checkpoint handler mode does not match ZOTrainerArguments: "
                f"{checkpoint_handler.checkpoint_mode!r} != "
                f"{args.zo_checkpoint_mode!r}"
            )

        kwargs.setdefault("data_collator", VLLMDataCollator())
        super().__init__(model=model, args=args, **kwargs)
        if self.args.world_size != 1:
            raise RuntimeError("ZOTrainer supports one Hugging Face process only")
        self.checkpoint_handler = checkpoint_handler
        self.model_accepts_loss_kwargs = False
        self._last_zo_step_metrics: dict[str, float | int | bool] = {}
        self._last_eval_metrics: dict[str, Any] = {}

    def _wrap_model(
        self,
        model: nn.Module,
        training: bool = True,
        dataloader: Any | None = None,
    ) -> nn.Module:
        """Keep the ZO runtime facade unwrapped; vLLM owns runtime parallelism."""

        del training, dataloader
        return model

    def _prepare_input(self, data: Any) -> Any:
        """Keep token batches on CPU; vLLM owns all model-device transfers."""

        return data

    def floating_point_ops(self, inputs: dict[str, torch.Tensor | Any]) -> int:
        del inputs
        return 0

    def create_optimizer(self, model: nn.Module | None = None) -> torch.optim.Optimizer:
        del model
        if self.optimizer is None:
            self.optimizer = ZOSGDOptimizer(
                [self.model._dummy_param],  # type: ignore[attr-defined]
                lr=float(self.args.learning_rate),
                weight_decay=float(self.args.weight_decay),
                on_step_result=self._record_zo_step_result,
            )
        return self.optimizer

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        del num_items_in_batch
        model.train()
        step = int(self.state.global_step) + 1
        optimizer = self._zo_optimizer()

        def compute_probe_loss(outputs: Mapping[str, Any]) -> torch.Tensor:
            return self._compute_loss_from_outputs(outputs)

        pending_step = model.zo_estimate(  # type: ignore[attr-defined]
            inputs,
            step=step,
            compute_loss_from_outputs_fn=compute_probe_loss,
        )
        optimizer.stage(pending_step)
        return torch.tensor(float(pending_step.reported_loss), device=self.args.device)

    def _zo_optimizer(self) -> ZOSGDOptimizer:
        optimizer = self.optimizer
        while optimizer is not None and not isinstance(optimizer, ZOSGDOptimizer):
            optimizer = getattr(optimizer, "optimizer", None)
        if not isinstance(optimizer, ZOSGDOptimizer):
            raise RuntimeError("ZOTrainer requires ZOSGDOptimizer")
        return optimizer

    def _record_zo_step_result(self, result: ZOStepResult) -> None:
        if not isinstance(result, ZOStepResult):
            raise TypeError(
                f"ZO optimizer must produce a ZOStepResult, got {type(result).__name__}"
            )
        self._last_zo_step_metrics = result.metrics()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        outputs = model(**inputs)
        loss = self._compute_loss_from_outputs(
            outputs, num_items_in_batch=num_items_in_batch
        )
        return (loss, outputs) if return_outputs else loss

    def _compute_loss_from_outputs(
        self,
        outputs: Mapping[str, Any],
        *,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Apply the HF loss contract to logits already produced by vLLM."""

        logits = _output_value(outputs, "logits")
        labels = _output_value(outputs, "loss_labels")
        if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise RuntimeError(
                "logits-only ZO outputs require tensor logits and loss_labels"
            )
        if self.compute_loss_func is not None:
            return self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
            )
        if isinstance(outputs, CompactCausalOutput):
            return ForCausalLMLoss(
                logits,
                labels,
                vocab_size=int(logits.shape[-1]),
                num_items_in_batch=num_items_in_batch,
                shift_labels=labels,
            )
        if isinstance(outputs, OptionClassificationOutput):
            return fixed_cross_entropy(
                logits.view(-1, int(logits.shape[-1])),
                labels.to(logits.device).view(-1),
                num_items_in_batch=num_items_in_batch,
            )
        raise TypeError(f"unsupported ZO model output: {type(outputs).__name__}")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        del ignore_keys
        outputs = model(**inputs)
        loss = self._compute_loss_from_outputs(outputs)
        if prediction_loss_only:
            return loss, None, None
        logits = _output_value(outputs, "logits")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("ZO model output must include tensor logits")
        predictions = logits
        output_labels = _output_value(outputs, "loss_labels")
        if not isinstance(output_labels, torch.Tensor):
            raise RuntimeError("ZO model output must include tensor loss_labels")
        labels = output_labels
        return loss, predictions, labels

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self._last_eval_metrics = dict(metrics)
        return metrics

    def predict(
        self,
        test_dataset,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "test",
    ):
        output = super().predict(
            test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self._last_eval_metrics = dict(output.metrics)
        return output

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if (
            self.args.zo_log_runtime_metrics
            and "loss" in logs
            and self._last_zo_step_metrics
        ):
            merged = dict(logs)
            for key, value in self._last_zo_step_metrics.items():
                if isinstance(value, (float, int, bool)):
                    merged.setdefault(f"zo_{key}", value)
            logs = merged
            self._last_zo_step_metrics = {}
        super().log(logs, start_time=start_time)

    def save_model(
        self, output_dir: str | None = None, _internal_call: bool = False
    ) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        reason = "checkpoint" if _internal_call else "save_model"
        payload = self.checkpoint_handler.save_checkpoint(
            output_dir,
            step=int(self.state.global_step),
            metrics=self._last_eval_metrics,
            reason=reason,
        )
        payload = _validate_checkpoint_payload(
            payload,
            expected_mode=self.args.zo_checkpoint_mode,
            loading=False,
        )
        if self.args.zo_checkpoint_mode != "metadata":
            save_direction_provider_state(
                output_dir,
                self.model.zo_model.direction_provider,  # type: ignore[attr-defined]
            )
            payload["direction_state_file"] = ZO_DIRECTION_STATE_NAME
        if int(payload["step"]) != int(self.state.global_step):
            raise ValueError("checkpoint payload step does not match TrainerState")
        if payload["reason"] != reason:
            raise ValueError("checkpoint payload reason does not match save request")
        metadata = ZOCheckpointMetadata(
            checkpoint_type=(
                "zo_trainer_checkpoint" if _internal_call else "zo_trainer_model"
            ),
            checkpoint_mode=self.args.zo_checkpoint_mode,
            global_step=int(self.state.global_step),
            reason=reason,
            metrics=dict(self._last_eval_metrics),
            payload=payload,
        )
        write_zo_checkpoint_metadata(output_dir, metadata)
        self._save_hf_artifacts(output_dir)

    def _save_hf_artifacts(self, output_dir: str) -> None:
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str,
        model: nn.Module | None = None,
    ) -> None:
        del model
        metadata = _validate_loadable_runtime_checkpoint(resume_from_checkpoint)
        if metadata["checkpoint_mode"] != self.checkpoint_handler.checkpoint_mode:
            raise ValueError(
                "checkpoint mode does not match runtime handler: "
                f"{metadata['checkpoint_mode']!r} != "
                f"{self.checkpoint_handler.checkpoint_mode!r}"
            )
        result = self.checkpoint_handler.load_checkpoint(resume_from_checkpoint)
        _validate_checkpoint_payload(
            result,
            expected_mode=self.checkpoint_handler.checkpoint_mode,
            loading=True,
        )
        load_direction_provider_state(
            resume_from_checkpoint,
            self.model.zo_model.direction_provider,  # type: ignore[attr-defined]
        )
        self.model.zo_model.invalidate_direction_slot_state()  # type: ignore[attr-defined]

    def _load_best_model(self) -> None:
        if self.state.best_model_checkpoint is None:
            return
        _validate_loadable_runtime_checkpoint(
            self.state.best_model_checkpoint,
            require_native=True,
        )
        self._load_from_checkpoint(self.state.best_model_checkpoint)


def _output_value(outputs: Any, key: str) -> Any:
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"ZO model output must be a mapping, got {type(outputs).__name__}"
        )
    return outputs[key]


def _validate_loadable_runtime_checkpoint(
    checkpoint_dir: str,
    *,
    require_native: bool = False,
) -> dict[str, Any]:
    metadata = read_zo_checkpoint_metadata(checkpoint_dir)
    checkpoint_mode = str(metadata["checkpoint_mode"])
    payload = _validate_checkpoint_payload(
        metadata["payload"],
        expected_mode=checkpoint_mode,
        loading=False,
    )
    loadable = payload["loadable"]
    if checkpoint_mode == "metadata" or loadable is False:
        raise RuntimeError(
            "cannot load metadata-only ZO checkpoint: "
            f"{checkpoint_dir}. Use zo_checkpoint_mode='native' for full-model "
            "reloads or 'lora' for resumable LoRA-bank training state."
        )
    if require_native and checkpoint_mode != "native":
        raise RuntimeError(
            "load_best_model_at_end requires a native ZO checkpoint: "
            f"{checkpoint_dir} has checkpoint_mode={checkpoint_mode!r}"
        )
    return metadata


def _validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_mode: str,
    loading: bool,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint handler must return a mapping")
    result = dict(payload)
    if result["mode"] != expected_mode:
        raise ValueError(
            f"checkpoint payload mode {result['mode']!r} does not match "
            f"{expected_mode!r}"
        )
    loadable = result["loadable"]
    if not isinstance(loadable, bool):
        raise TypeError("checkpoint payload loadable must be bool")
    if loading and not loadable:
        raise RuntimeError("runtime checkpoint handler returned a non-loadable payload")
    return result
