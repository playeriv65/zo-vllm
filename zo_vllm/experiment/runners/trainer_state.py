"""HF Trainer-like state helpers for vLLM ZO runner resume continuity."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ZO_TRAINER_STATE_NAME = "zo_trainer_state.json"


def resolve_trainer_state_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.is_dir():
        return path / ZO_TRAINER_STATE_NAME
    return path


def save_zo_trainer_state(
    checkpoint_path: str | Path,
    state: Mapping[str, Any],
) -> str:
    path = resolve_trainer_state_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(state), encoding="utf-8")
    return str(path)


def load_zo_trainer_state(
    checkpoint_path: str | Path,
) -> dict[str, Any] | None:
    path = resolve_trainer_state_path(checkpoint_path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise TypeError("ZO trainer state must be a JSON object")
    return state


def build_zo_trainer_state(
    *,
    global_step: int,
    raw_step: int,
    log_history: list[dict[str, Any]],
    eval_losses: list[dict[str, Any]],
    eval_metrics: list[dict[str, Any]],
    checkpoint_records: list[dict[str, Any]],
    best_checkpoint: dict[str, Any] | None,
    wandb_run_id: str | None,
    wandb_run_name: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a HF Trainer-like state payload for ZO resume continuity."""

    return {
        "state_format": "zo_vllm_trainer_state",
        "state_version": 1,
        "global_step": int(global_step),
        "raw_step": int(raw_step),
        "log_history": _jsonable_list(log_history),
        "eval_losses": _jsonable_list(eval_losses),
        "eval_metrics": _jsonable_list(eval_metrics),
        "checkpoint_records": _jsonable_list(checkpoint_records),
        "best_checkpoint": _jsonable(best_checkpoint),
        "wandb": {
            "run_id": wandb_run_id,
            "run_name": wandb_run_name,
        },
        "metadata": _jsonable({} if metadata is None else dict(metadata)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def merge_zo_trainer_state(
    state: Mapping[str, Any] | None,
    *,
    history: list[dict[str, Any]],
    eval_losses: list[dict[str, Any]],
    eval_metrics: list[dict[str, Any]],
    checkpoint_records: list[dict[str, Any]],
    checkpoint_paths: list[str],
    best_tracker: Any,
) -> dict[str, Any]:
    """Append restored state into live runner collections.

    The runner keeps writing new rows after these restored rows, matching the
    way HF Trainer resumes from ``trainer_state.json`` and keeps extending
    ``log_history``.
    """

    if not state:
        return {"restored": False}
    if state["state_format"] != "zo_vllm_trainer_state":
        raise ValueError(f"unknown ZO trainer state format: {state['state_format']!r}")
    if int(state["state_version"]) != 1:
        raise ValueError(
            f"unsupported ZO trainer state version: {state['state_version']!r}"
        )
    resume_step = int(state["global_step"])
    int(state["raw_step"])
    restored_history = _list_of_dicts(state["log_history"])
    restored_eval_losses = _list_of_dicts(state["eval_losses"])
    restored_eval_metrics = _list_of_dicts(state["eval_metrics"])
    restored_records = _list_of_dicts(state["checkpoint_records"])
    history[:] = restored_history + _rows_after_step(history, resume_step)
    eval_losses[:] = restored_eval_losses + _rows_after_step(eval_losses, resume_step)
    eval_metrics[:] = restored_eval_metrics + _rows_after_step(
        eval_metrics, resume_step
    )
    checkpoint_records[:] = restored_records + _rows_after_step(
        checkpoint_records,
        resume_step,
    )
    checkpoint_paths[:] = [
        str(record["path"])
        for record in checkpoint_records
        if record["path"] is not None
    ]
    best_checkpoint = state["best_checkpoint"]
    if best_checkpoint is not None:
        if not isinstance(best_checkpoint, Mapping):
            raise TypeError("best_checkpoint must be a mapping or None")
        best_tracker.best_record = dict(best_checkpoint)
        best_tracker.best_value = float(best_checkpoint["metric_value"])
    return {
        "restored": True,
        "history_rows": len(restored_history),
        "eval_loss_rows": len(restored_eval_losses),
        "eval_metric_rows": len(restored_eval_metrics),
        "checkpoint_records": len(restored_records),
    }


def jsonable(value: Any) -> Any:
    return _jsonable(value)


def json_dumps(value: Any) -> str:
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return str(value)


def _jsonable_list(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(_jsonable(value)) for value in values]


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"expected a list of mappings, got {type(value).__name__}")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError("expected every list item to be a mapping")
    return [dict(item) for item in value]


def _rows_after_step(values: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    rows = []
    for row in values:
        if not isinstance(row, Mapping):
            raise TypeError("trainer history rows must be mappings")
        if int(row["step"]) > int(step):
            rows.append(dict(row))
    return rows
