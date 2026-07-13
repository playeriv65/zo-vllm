"""Checkpoint payload helpers for vLLM ZO runners."""

from __future__ import annotations

from datetime import datetime
import os
import shutil
from typing import Any, Mapping

import torch

from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.training import lora_checkpoint
from zo_vllm.training.native_checkpoint import save_effective_native_checkpoint
from zo_vllm.experiment.runners.checkpoint_policy import BestMetricTracker
from zo_vllm.experiment.runners.trainer_state import (
    build_zo_trainer_state,
    json_dumps,
    jsonable,
    save_zo_trainer_state,
    ZO_TRAINER_STATE_NAME,
)


def prune_checkpoints(
    checkpoint_records: list[dict[str, Any]],
    *,
    save_total_limit: int,
    best_checkpoint_path: str | None,
) -> None:
    if save_total_limit <= 0:
        return
    while len(checkpoint_records) > save_total_limit:
        stale_index = None
        for index, record in enumerate(checkpoint_records):
            if record.get("path") != best_checkpoint_path:
                stale_index = index
                break
        if stale_index is None:
            stale_index = 0
        stale = checkpoint_records.pop(stale_index)
        remove_checkpoint_path(str(stale["path"]))


def remove_checkpoint_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def write_checkpoint_metadata(
    path: str | os.PathLike[str], metadata: Mapping[str, Any]
) -> str:
    path_s = str(path)
    metadata_path = (
        os.path.join(path_s, "zo_checkpoint_metadata.json")
        if os.path.isdir(path_s)
        else path_s
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        handle.write(json_dumps(jsonable(dict(metadata))))
    return metadata_path


class RuntimeCheckpointManager:
    """Own HF-like runtime checkpoint decisions for the vLLM ZO runner."""

    def __init__(
        self,
        *,
        args: Any,
        checkpoint_mode: str,
        checkpoint_root: str,
        effective_save_steps: int,
        llm: Any,
        accumulated_update_state: Any | None,
        weight_sync: WeightSync,
        use_lora_bank_update: bool,
        best_tracker: BestMetricTracker,
        timing: dict[str, list[float]],
        history: list[dict[str, Any]],
        eval_losses: list[dict[str, Any]],
        eval_metrics: list[dict[str, Any]],
        checkpoint_records: list[dict[str, Any]],
        checkpoint_paths: list[str],
        wandb_run: Any | None,
    ) -> None:
        self.args = args
        self.checkpoint_mode = checkpoint_mode
        self.checkpoint_root = checkpoint_root
        self.effective_save_steps = int(effective_save_steps)
        self.llm = llm
        self.accumulated_update_state = accumulated_update_state
        self.weight_sync = weight_sync
        self.use_lora_bank_update = bool(use_lora_bank_update)
        self.best_tracker = best_tracker
        self.timing = timing
        self.history = history
        self.eval_losses = eval_losses
        self.eval_metrics = eval_metrics
        self.checkpoint_records = checkpoint_records
        self.checkpoint_paths = checkpoint_paths
        self.wandb_run = wandb_run

    def save_runtime_checkpoint(
        self,
        measured_index: int,
        *,
        raw_step: int,
        eval_row: Mapping[str, Any] | None = None,
        reason: str,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "checkpoint_type": f"vllm_zo_{self.checkpoint_mode}",
            "step": int(measured_index),
            "raw_step": int(raw_step),
            "save_strategy": self.args.save_strategy,
            "save_reason": str(reason),
            "eval": None if eval_row is None else dict(eval_row),
            "metric_for_best_model": self.args.metric_for_best_model,
            "greater_is_better": bool(self.best_tracker.greater_is_better),
            "timing_counts": {key: len(value) for key, value in self.timing.items()},
            "history_count": len(self.history),
            "direct_update_mode": self.args.direct_update_mode,
            "quantized_update_mode": self.args.quantized_update_mode,
            "timestamp": datetime.now().isoformat(),
        }
        if self.checkpoint_mode == "native":
            ckpt_path = checkpoint_path or os.path.join(
                self.checkpoint_root,
                f"checkpoint-{int(measured_index):07d}",
            )
            os.makedirs(ckpt_path, exist_ok=True)
            save_info = save_effective_native_checkpoint(
                llm=self.llm,
                checkpoint_dir=ckpt_path,
                accumulated_update_state=self.accumulated_update_state,
                weight_sync=self.weight_sync,
                step=int(raw_step),
                use_lora_bank_update=self.use_lora_bank_update,
                precision=self.args.weight_update_precision,
            )
            metadata |= save_info
            metadata["loadable"] = True
            metadata_path = write_checkpoint_metadata(ckpt_path, metadata)
        elif self.checkpoint_mode == "lora":
            ckpt_path = checkpoint_path or os.path.join(
                self.checkpoint_root,
                f"checkpoint-{int(measured_index):07d}",
            )
            save_info = lora_checkpoint.save_lora_bank_checkpoint(
                checkpoint_dir=ckpt_path,
                accumulated_update_state=self.accumulated_update_state,
                step=int(measured_index),
                raw_step=int(raw_step),
                dtype=(
                    torch.float16
                    if self.args.u_snapshot_dtype == "float16"
                    else torch.float32
                ),
            )
            metadata |= save_info
            metadata["loadable"] = True
            metadata["load_method"] = "resume_lora_checkpoint"
            metadata_path = write_checkpoint_metadata(ckpt_path, metadata)
        else:
            ckpt_path = checkpoint_path or os.path.join(
                self.checkpoint_root,
                f"step_{int(measured_index):07d}.json",
            )
            if checkpoint_path is not None:
                os.makedirs(ckpt_path, exist_ok=True)
            metadata["loadable"] = False
            metadata_path = write_checkpoint_metadata(ckpt_path, metadata)

        record = {
            "path": ckpt_path,
            "metadata_path": metadata_path,
            "mode": self.checkpoint_mode,
            "step": int(measured_index),
            "raw_step": int(raw_step),
            "save_reason": str(reason),
            "eval": None if eval_row is None else dict(eval_row),
            "loadable": bool(metadata["loadable"]),
        }
        self.checkpoint_records.append(record)
        self.checkpoint_paths.append(ckpt_path)
        improved = False
        if eval_row is not None:
            improved = self.best_tracker.update(eval_row, record)
        best_path = None
        if self.best_tracker.best_record is not None:
            checkpoint = self.best_tracker.best_record.get("checkpoint")
            if checkpoint is not None:
                best_path = checkpoint.get("path")
        prune_checkpoints(
            self.checkpoint_records,
            save_total_limit=int(self.args.save_total_limit),
            best_checkpoint_path=best_path,
        )
        live_paths = {record["path"] for record in self.checkpoint_records}
        self.checkpoint_paths[:] = [
            path for path in self.checkpoint_paths if path in live_paths
        ]
        if os.path.isdir(ckpt_path):
            trainer_state_path = os.path.join(ckpt_path, ZO_TRAINER_STATE_NAME)
            record["trainer_state_path"] = trainer_state_path
            metadata["trainer_state_path"] = trainer_state_path
            trainer_state = build_zo_trainer_state(
                global_step=int(measured_index),
                raw_step=int(raw_step),
                log_history=self.history,
                eval_losses=self.eval_losses,
                eval_metrics=self.eval_metrics,
                checkpoint_records=self.checkpoint_records,
                best_checkpoint=self.best_tracker.best_record,
                wandb_run_id=None if self.wandb_run is None else self.wandb_run.id,
                wandb_run_name=None if self.wandb_run is None else self.wandb_run.name,
                metadata={
                    "checkpoint_mode": self.checkpoint_mode,
                    "save_strategy": self.args.save_strategy,
                    "metric_for_best_model": self.args.metric_for_best_model,
                },
            )
            save_zo_trainer_state(ckpt_path, trainer_state)
            write_checkpoint_metadata(ckpt_path, metadata)
        print(
            f"[vLLM] checkpoint_saved={ckpt_path} mode={self.checkpoint_mode} "
            f"reason={reason} best_updated={int(improved)}",
            flush=True,
        )
        return record

    def save_final_native_checkpoint(
        self,
        measured_index: int,
        *,
        raw_step: int,
    ) -> str | None:
        if not bool(int(self.args.save_final_checkpoint)):
            return None
        os.makedirs(self.checkpoint_root, exist_ok=True)
        checkpoint_dir = os.path.join(
            self.checkpoint_root,
            f"final_step_{int(measured_index):07d}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_info = save_effective_native_checkpoint(
            llm=self.llm,
            checkpoint_dir=checkpoint_dir,
            accumulated_update_state=self.accumulated_update_state,
            weight_sync=self.weight_sync,
            step=int(raw_step),
            use_lora_bank_update=self.use_lora_bank_update,
            precision=self.args.weight_update_precision,
        )
        metadata = {
            "checkpoint_type": "vllm_sharded_state",
            "step": int(measured_index),
            "direct_update_mode": self.args.direct_update_mode,
            "quantized_update_mode": self.args.quantized_update_mode,
            "timestamp": datetime.now().isoformat(),
            **save_info,
        }
        write_checkpoint_metadata(checkpoint_dir, metadata)
        self.checkpoint_paths.append(checkpoint_dir)
        print(
            f"[vLLM] final_native_checkpoint_saved={checkpoint_dir} "
            f"fold_s={float(save_info.get('fold_s', 0.0)):.6f}",
            flush=True,
        )
        return checkpoint_dir
