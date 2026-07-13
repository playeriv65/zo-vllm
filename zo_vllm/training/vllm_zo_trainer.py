"""HF-like vLLM ZO trainer facade for task-owned data pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch.utils.data import DataLoader, RandomSampler

from .arguments import ZOTrainingArguments


class VLLMZOStepModel(Protocol):
    """Minimal model protocol consumed by :class:`VLLMZOTrainer`."""

    def step(self, batch: Any, *, step: int) -> Any:
        """Run one train step and return a metrics-bearing result."""


@dataclass(frozen=True)
class VLLMZOTrainOutput:
    """Return value from :meth:`VLLMZOTrainer.train`."""

    global_step: int
    metrics: dict[str, float | int | bool]


@dataclass
class VLLMZOTrainerState:
    """HF-like trainer state; JSON conversion happens only at boundaries."""

    global_step: int = 0
    raw_step: int = 0
    log_history: list[dict[str, Any]] = field(default_factory=list)
    eval_metrics: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_records: list[dict[str, Any]] = field(default_factory=list)
    best_checkpoint: dict[str, Any] | None = None
    loaded_best_checkpoint: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VLLMZOTrainerState:
        return cls(
            global_step=int(value["global_step"]),
            raw_step=int(value["raw_step"]),
            log_history=_list_of_dicts(value["log_history"]),
            eval_metrics=_list_of_dicts(value["eval_metrics"]),
            checkpoint_records=_list_of_dicts(value["checkpoint_records"]),
            best_checkpoint=_optional_dict(value["best_checkpoint"]),
            loaded_best_checkpoint=_optional_dict(value["loaded_best_checkpoint"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "raw_step": int(self.raw_step),
            "log_history": list(self.log_history),
            "eval_metrics": list(self.eval_metrics),
            "checkpoint_records": list(self.checkpoint_records),
            "best_checkpoint": self.best_checkpoint,
            "loaded_best_checkpoint": self.loaded_best_checkpoint,
        }


@dataclass
class _BestMetricTracker:
    """Track the best eval row using Trainer-style metric semantics."""

    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool | None = None
    best_value: float | None = None
    best_record: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.greater_is_better is None:
            self.greater_is_better = _infer_greater_is_better(
                self.metric_for_best_model
            )

    def update(
        self,
        eval_row: Mapping[str, Any],
        checkpoint_record: Mapping[str, Any] | None,
    ) -> bool:
        value = self.candidate_value(eval_row)
        if not self.is_better_value(value):
            return False
        self.best_value = float(value)
        self.best_record = {
            "metric": _validate_metric_name(self.metric_for_best_model),
            "metric_value": float(value),
            "greater_is_better": bool(self.greater_is_better),
            "eval": dict(eval_row),
            "checkpoint": None
            if checkpoint_record is None
            else dict(checkpoint_record),
        }
        return True

    def candidate_value(self, eval_row: Mapping[str, Any]) -> float:
        return _metric_value(eval_row, self.metric_for_best_model)

    def is_better(self, eval_row: Mapping[str, Any]) -> bool:
        return self.is_better_value(self.candidate_value(eval_row))

    def is_better_value(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if bool(self.greater_is_better):
            return float(value) > float(self.best_value)
        return float(value) < float(self.best_value)


@dataclass
class VLLMZOTrainerControl:
    """Mutable callback control flags following Hugging Face Trainer semantics."""

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_log: bool = False
    should_evaluate: bool = False
    should_save: bool = False

    def reset_step_flags(self) -> None:
        self.should_log = False
        self.should_evaluate = False
        self.should_save = False


class VLLMZOTrainerCallback:
    """HF-shaped callback surface for ZO-vLLM training loops."""

    def on_init_end(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_train_begin(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_train_end(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_epoch_begin(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_epoch_end(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_step_begin(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_substep_end(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_step_end(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_prediction_step(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_predict(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_pre_optimizer_step(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_optimizer_step(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_evaluate(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_save(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None

    def on_log(
        self,
        args: ZOTrainingArguments,
        state: VLLMZOTrainerState,
        control: VLLMZOTrainerControl,
        **kwargs: Any,
    ) -> VLLMZOTrainerControl | None:
        return None


class VLLMZOTrainer:
    """A compact Trainer-shaped loop that keeps task encoding outside zo-vllm."""

    def __init__(
        self,
        *,
        model: VLLMZOStepModel,
        args: ZOTrainingArguments,
        train_dataset: Any | None = None,
        train_dataloader: Iterable[Any] | None = None,
        data_collator: Callable[[list[Any]], Any] | None = None,
        eval_fn: Callable[[], dict[str, float | int | bool]] | None = None,
        save_model_fn: Callable[[str], Mapping[str, Any] | None] | None = None,
        load_model_fn: Callable[[str], None] | None = None,
        log_fn: Callable[[dict[str, float | int | bool], int], None] | None = None,
        callbacks: Sequence[VLLMZOTrainerCallback] | None = None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.data_collator = data_collator or (lambda batch: batch)
        self.eval_fn = eval_fn
        self.save_model_fn = save_model_fn
        self.load_model_fn = load_model_fn
        self.log_fn = log_fn
        self.callbacks = list(callbacks or [])
        self.control = VLLMZOTrainerControl()
        self.state = VLLMZOTrainerState()
        self.best_tracker = _BestMetricTracker(
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
        )
        self._call_event("on_init_end")

    def get_train_dataloader(self) -> Iterable[Any]:
        """Return a RandomSampler DataLoader matching HF Trainer's train order."""
        if self.train_dataloader is not None:
            return self.train_dataloader
        if self.train_dataset is None:
            raise ValueError("VLLMZOTrainer requires train_dataset or train_dataloader")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.args.seed))
        sampler = RandomSampler(self.train_dataset, generator=generator)
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.per_device_train_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=bool(self.args.dataloader_drop_last),
            num_workers=int(self.args.dataloader_num_workers),
        )

    def train(self, resume_from_checkpoint: str | None = None) -> VLLMZOTrainOutput:
        if resume_from_checkpoint is not None:
            self._load_state(resume_from_checkpoint)
        iterator = _cycle(self.get_train_dataloader())
        latest_metrics: dict[str, float | int | bool] = {}
        start_raw_step = int(self.state.raw_step) + 1
        for _ in range(1, start_raw_step):
            next(iterator)
        total_steps = int(self.args.warmup_steps) + int(self.args.max_steps)
        self._call_event(
            "on_train_begin", resume_from_checkpoint=resume_from_checkpoint
        )
        self._call_event("on_epoch_begin")
        for raw_step in range(start_raw_step, total_steps + 1):
            self.control.reset_step_flags()
            measured_step = raw_step - int(self.args.warmup_steps)
            self.state.raw_step = raw_step
            batch = next(iterator)
            self._call_event(
                "on_step_begin",
                raw_step=raw_step,
                measured_step=measured_step,
                batch=batch,
            )
            if self.control.should_training_stop:
                break
            self._call_event(
                "on_pre_optimizer_step",
                raw_step=raw_step,
                measured_step=measured_step,
                batch=batch,
            )
            if self.control.should_training_stop:
                break
            result = self.model.step(batch, step=raw_step)
            self._call_event(
                "on_optimizer_step",
                raw_step=raw_step,
                measured_step=measured_step,
                result=result,
            )
            latest_metrics = _result_metrics(result)
            if measured_step <= 0:
                self._call_event(
                    "on_substep_end",
                    raw_step=raw_step,
                    measured_step=measured_step,
                    result=result,
                    metrics=latest_metrics,
                )
                continue
            self.state.global_step = measured_step
            self.control.should_log = (
                measured_step == 1 or measured_step % self.args.logging_steps == 0
            )
            self.control.should_evaluate = (
                self.eval_fn is not None and measured_step % self.args.eval_steps == 0
            )
            self.control.should_save = self._should_save_steps(measured_step)
            self._call_event(
                "on_step_end",
                raw_step=raw_step,
                measured_step=measured_step,
                result=result,
                metrics=latest_metrics,
            )
            if self.control.should_log:
                self.log(latest_metrics, step=measured_step)
            eval_metrics = None
            if self.control.should_evaluate and self.eval_fn is not None:
                eval_metrics = self.evaluate()
                self.log(eval_metrics, step=measured_step)
                self._maybe_save_best(eval_metrics, step=measured_step)
            checkpoint_record = None
            if self.control.should_save:
                checkpoint_record = self._save_checkpoint(
                    step=measured_step,
                    reason="steps",
                )
            if eval_metrics is not None and checkpoint_record is not None:
                if self.best_tracker.update(eval_metrics, checkpoint_record):
                    self._write_checkpoint_state(Path(checkpoint_record["path"]))
                    self._write_state()
            if self.control.should_training_stop:
                break
        self._call_event("on_epoch_end")
        if self.args.load_best_model_at_end:
            self._load_best_model()
        self._write_state()
        self._call_event("on_train_end", metrics=latest_metrics)
        return VLLMZOTrainOutput(
            global_step=int(self.state.global_step),
            metrics=latest_metrics,
        )

    def evaluate(
        self, eval_dataset: Any | None = None
    ) -> dict[str, float | int | bool]:
        if self.eval_fn is None:
            return {}
        metrics = dict(self._call_eval_fn(eval_dataset))
        metrics.setdefault("global_step", int(self.state.global_step))
        metrics.setdefault("step", int(self.state.global_step))
        self.state.eval_metrics.append(metrics)
        self._call_event("on_evaluate", metrics=metrics)
        return metrics

    def save_model(self, output_dir: str | None = None) -> None:
        target = output_dir or self.args.output_dir
        Path(target).mkdir(parents=True, exist_ok=True)
        if self.save_model_fn is not None:
            self.save_model_fn(target)
        self._write_state()
        self._call_event("on_save", checkpoint_dir=target)

    def log(
        self, metrics: dict[str, float | int | bool], *, step: int | None = None
    ) -> None:
        step_i = int(self.state.global_step if step is None else step)
        payload = {"step": step_i, **metrics}
        self.state.log_history.append(payload)
        if self.log_fn is not None:
            self.log_fn(payload, step_i)
        self._call_event("on_log", logs=payload)

    def add_callback(self, callback: VLLMZOTrainerCallback) -> None:
        self.callbacks.append(callback)

    def remove_callback(self, callback: VLLMZOTrainerCallback) -> None:
        self.callbacks = [item for item in self.callbacks if item is not callback]

    def set_initial_state(
        self,
        *,
        global_step: int = 0,
        raw_step: int | None = None,
    ) -> None:
        """Set the trainer cursor before ``train()`` starts."""

        self.state.global_step = int(global_step)
        self.state.raw_step = int(global_step if raw_step is None else raw_step)

    def _write_state(self) -> None:
        self.state.best_checkpoint = self.best_tracker.best_record
        path = Path(self.args.output_dir) / "trainer_state.json"
        path.write_text(_json_dumps(self.state.to_dict()), encoding="utf-8")

    def _load_state(self, checkpoint_path: str) -> None:
        checkpoint = Path(checkpoint_path)
        state_path = checkpoint / "trainer_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"trainer_state.json not found in {checkpoint}")
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        self.state = VLLMZOTrainerState.from_dict(loaded)
        best_checkpoint = self.state.best_checkpoint
        if isinstance(best_checkpoint, dict):
            self.best_tracker.best_record = dict(best_checkpoint)
            metric_value = best_checkpoint.get("metric_value")
            if metric_value is not None:
                self.best_tracker.best_value = float(metric_value)

    def _should_save_steps(self, step: int) -> bool:
        return (
            self.args.save_strategy == "steps"
            and int(self.args.save_steps) > 0
            and int(step) % int(self.args.save_steps) == 0
        )

    def _maybe_save_best(
        self,
        metrics: dict[str, float | int | bool],
        *,
        step: int,
    ) -> None:
        if self.args.save_strategy != "best":
            return
        if self.best_tracker.is_better(metrics):
            record = self._save_checkpoint(step=step, reason="best")
            self.best_tracker.update(metrics, record)
            self._write_checkpoint_state(Path(record["path"]))
            self._write_state()

    def _save_checkpoint(self, *, step: int, reason: str) -> dict[str, Any]:
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{int(step):07d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_record = None
        if self.save_model_fn is not None:
            save_record = self.save_model_fn(str(checkpoint_dir))
        record = {
            "step": int(step),
            "path": str(checkpoint_dir),
            "reason": str(reason),
        }
        if isinstance(save_record, Mapping):
            record.update(dict(save_record))
            record.setdefault("reason", str(reason))
        self.state.checkpoint_records.append(record)
        self._enforce_save_total_limit(protect_path=str(checkpoint_dir))
        self._write_checkpoint_state(checkpoint_dir)
        self._write_state()
        self._call_event("on_save", checkpoint_dir=str(checkpoint_dir), record=record)
        return record

    def _write_checkpoint_state(self, checkpoint_dir: Path) -> None:
        state = self.state.to_dict()
        state["best_checkpoint"] = self.best_tracker.best_record
        (checkpoint_dir / "trainer_state.json").write_text(
            _json_dumps(state),
            encoding="utf-8",
        )

    def _enforce_save_total_limit(self, *, protect_path: str | None = None) -> None:
        limit = self.args.save_total_limit
        if limit is None:
            return
        records = self.state.checkpoint_records
        best_path = None
        best_record = self.best_tracker.best_record
        if isinstance(best_record, dict):
            checkpoint = best_record.get("checkpoint")
            if isinstance(checkpoint, dict):
                best_path = checkpoint.get("path")
        while len(records) > int(limit):
            candidate_index = next(
                (
                    index
                    for index, record in enumerate(records)
                    if record.get("path") not in {best_path, protect_path}
                ),
                0,
            )
            record = records.pop(candidate_index)
            path = record.get("path")
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)

    def _load_best_model(self) -> None:
        best_record = self.best_tracker.best_record
        if not best_record:
            raise RuntimeError(
                "load_best_model_at_end was enabled, but no best checkpoint was saved"
            )
        checkpoint = best_record.get("checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("path") is None:
            raise RuntimeError(
                "best checkpoint record does not contain a loadable path"
            )
        if self.load_model_fn is None:
            raise RuntimeError("load_best_model_at_end requires load_model_fn")
        self.load_model_fn(str(checkpoint["path"]))

    def _call_event(self, event: str, **kwargs: Any) -> VLLMZOTrainerControl:
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is None:
                continue
            new_control = method(self.args, self.state, self.control, **kwargs)
            if new_control is not None:
                self.control = new_control
        return self.control

    def _call_eval_fn(self, eval_dataset: Any | None = None) -> Mapping[str, Any]:
        if self.eval_fn is None:
            return {}
        if eval_dataset is None:
            return self.eval_fn()
        signature = inspect.signature(self.eval_fn)
        parameters = signature.parameters
        if "eval_dataset" in parameters:
            return self.eval_fn(eval_dataset=eval_dataset)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
        ):
            return self.eval_fn(eval_dataset=eval_dataset)
        positional = [
            param
            for param in parameters.values()
            if param.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if positional:
            return self.eval_fn(eval_dataset)
        return self.eval_fn()


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"expected a list of mappings, got {type(value).__name__}")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError("expected every list item to be a mapping")
    return [dict(item) for item in value]


def _optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping or None, got {type(value).__name__}")
    return dict(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _validate_metric_name(metric_name: str) -> str:
    name = str(metric_name)
    if not name:
        raise ValueError("metric_for_best_model must not be empty")
    return name


def _infer_greater_is_better(metric_name: str) -> bool:
    return "loss" not in _validate_metric_name(metric_name).lower()


def _metric_value(eval_row: Mapping[str, Any], metric_name: str) -> float:
    name = _validate_metric_name(metric_name)
    if name not in eval_row:
        raise KeyError(f"evaluation result is missing configured metric: {name}")
    return float(eval_row[name])


def _cycle(items: Iterable[Any]) -> Iterator[Any]:
    while True:
        yielded = False
        for item in items:
            yielded = True
            yield item
        if not yielded:
            raise ValueError("train_dataloader must yield at least one batch")


def _result_metrics(result: Any) -> dict[str, float | int | bool]:
    metrics_fn = getattr(result, "metrics", None)
    if callable(metrics_fn):
        return dict(metrics_fn())
    if isinstance(result, dict):
        return {
            key: value
            for key, value in result.items()
            if isinstance(value, (float, int, bool))
        }
    raise TypeError("train step result must expose metrics() or be a metrics dict")
