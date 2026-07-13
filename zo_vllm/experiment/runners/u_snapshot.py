"""Optional U-update statistics recorder for vLLM ZO runners."""

from __future__ import annotations

from datetime import datetime
import os
from typing import Any, Callable

import torch


class USnapshotRecorder:
    """Record U statistics for analysis; this does not affect training state."""

    def __init__(
        self,
        *,
        root: str,
        interval: int,
        dtype_name: str,
        total_limit: int,
        update_state: Any | None,
        log_wandb: Callable[[dict[str, Any], int], None],
    ) -> None:
        self.root = root
        self.interval = int(interval)
        self.dtype_name = dtype_name
        self.total_limit = int(total_limit)
        self.update_state = update_state
        self.log_wandb = log_wandb
        self.paths: list[str] = []
        self.metrics: list[dict[str, Any]] = []
        self._last_vec = None
        self._last_step: int | None = None
        self._last_delta_vec = None
        if self.interval > 0:
            os.makedirs(self.root, exist_ok=True)

    def capture(
        self,
        measured_index: int,
        eval_loss: float | None = None,
        eval_accuracy: float | None = None,
    ) -> dict[str, Any] | None:
        if self.interval <= 0:
            return None
        if measured_index > 0 and measured_index % self.interval != 0:
            return None
        if self.update_state is None:
            return None

        snapshot_dtype = (
            torch.float16 if self.dtype_name == "float16" else torch.float32
        )
        metrics, flat_vec, self._last_delta_vec = self.update_state.snapshot_metrics(
            step=measured_index,
            previous_flat=self._last_vec,
            previous_delta=self._last_delta_vec,
        )
        if metrics is None or flat_vec is None:
            return None
        metrics |= {
            "prev_step": self._last_step,
            "eval_loss": None if eval_loss is None else float(eval_loss),
            "eval_accuracy": None if eval_accuracy is None else float(eval_accuracy),
        }
        snapshot_path = os.path.join(self.root, f"u_step_{measured_index:07d}.pt")
        torch.save(
            {
                "metrics": metrics,
                "u_accum": self.update_state.snapshot_tensors(dtype=snapshot_dtype),
                "dtype": self.dtype_name,
                "timestamp": datetime.now().isoformat(),
            },
            snapshot_path,
        )
        self.paths.append(snapshot_path)
        self.metrics.append(metrics)
        if self.total_limit > 0 and len(self.paths) > self.total_limit:
            stale = self.paths.pop(0)
            if os.path.exists(stale):
                os.remove(stale)
        self._last_vec = flat_vec
        self._last_step = int(measured_index)
        print(
            f"[vLLM] step={measured_index} "
            f"u_total_norm={metrics['u_total_norm']:.6e} "
            f"u_delta_norm_since_prev={metrics['u_delta_norm_since_prev']} "
            f"u_cosine_to_prev={metrics['u_cosine_to_prev']} "
            f"u_angle_deg_to_prev={metrics['u_angle_deg_to_prev']} "
            f"delta_w_fro_est={metrics['delta_w_fro_est']:.6e} "
            f"u_snapshot={snapshot_path}",
            flush=True,
        )
        self.log_wandb(
            {
                "u/total_norm": metrics["u_total_norm"],
                "u/delta_norm_since_prev": metrics["u_delta_norm_since_prev"],
                "u/cosine_to_prev": metrics["u_cosine_to_prev"],
                "u/angle_deg_to_prev": metrics["u_angle_deg_to_prev"],
                "u/delta_cosine_to_prev_delta": metrics["u_delta_cosine_to_prev_delta"],
                "u/delta_angle_deg_to_prev_delta": metrics[
                    "u_delta_angle_deg_to_prev_delta"
                ],
                "u/delta_w_fro_est": metrics["delta_w_fro_est"],
            },
            int(measured_index),
        )
        return metrics
