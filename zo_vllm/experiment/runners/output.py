"""Output helpers for vLLM ZO experiment runners."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import json
import os
from typing import Any

from zo_vllm.experiment.infra.stats import summarize, summarize_tail


@dataclass(frozen=True)
class ResolvedOutputPaths:
    """Resolved output location for one runner invocation."""

    output_dir: str
    output_root: str | None
    experiment_name: str | None


def resolve_vllm_output_paths(
    *,
    args: Namespace,
    project_root: str,
    model_name: str,
    timestamp: str,
) -> ResolvedOutputPaths:
    """Resolve HF-like runner output paths without creating directories."""

    if args.output_dir is not None:
        return ResolvedOutputPaths(
            output_dir=str(args.output_dir),
            output_root=None,
            experiment_name=None,
        )
    default_output_root = (
        os.path.join(project_root, "zo_post", "results")
        if args.direction_provider in {"agzo", "uagzo", "suagzo"}
        else os.path.join(project_root, "phase3", "results")
    )
    output_root = str(args.output_root or default_output_root)
    experiment_name = str(
        args.experiment_name
        or (
            f"{args.direction_provider}_{args.train_objective}_"
            f"{model_name.replace('/', '__')}_r{args.rank}_nu{args.nu}_{timestamp}"
        )
    )
    return ResolvedOutputPaths(
        output_dir=os.path.join(output_root, experiment_name, "vllm"),
        output_root=output_root,
        experiment_name=experiment_name,
    )


def _best_eval_loss_metric(eval_metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the eval row with the lowest dev loss."""

    loss_rows = [row for row in eval_metrics if row.get("loss") is not None]
    if not loss_rows:
        return None
    best = min(loss_rows, key=lambda row: float(row["loss"]))
    payload: dict[str, Any] = {
        "step": int(best["step"]),
        "loss": float(best["loss"]),
    }
    if best.get("accuracy") is not None:
        payload["accuracy"] = float(best["accuracy"])
    if best.get("valid_accuracy") is not None:
        payload["valid_accuracy"] = float(best["valid_accuracy"])
    return payload


def write_vllm_perf_json(
    *,
    output_dir: str,
    profile_mode: str,
    timestamp: str,
    args: Namespace,
    model_name: str,
    effective_direction_scale: float,
    direction_scale_mode: str,
    direction_scale_note: str,
    direction_scale_applies_to: str,
    output_root_resolved: str | None,
    experiment_name_resolved: str | None,
    lora_slot_rank: int,
    prequant_model: str | None,
    accuracy_eval_mode: str,
    train_objective: str,
    use_unified_stepper: bool,
    initial_loss: float | None,
    final_loss: float | None,
    eval_losses: list[dict[str, Any]],
    eval_metrics: list[dict[str, Any]],
    initial_dev_acc: float | None,
    initial_valid_acc: float | None,
    final_dev_acc: float | None,
    final_valid_acc: float | None,
    history: list[dict[str, Any]],
    timing: dict[str, list[float]],
    total_s: float,
    ckpt_paths: list[str],
    checkpoint_records: list[dict[str, Any]] | None = None,
    best_checkpoint: dict[str, Any] | None = None,
    loaded_best_checkpoint: dict[str, Any] | None = None,
    u_snapshot_paths: list[str],
    u_snapshot_metrics: list[dict[str, Any]],
) -> str:
    """Write the standard vLLM performance JSON and return its path."""

    output_file = os.path.join(output_dir, f"vllm_perf_{profile_mode}_{timestamp}.json")
    aligned_update_times = [
        float(build_lora_s)
        + float(lora_update_s)
        + float(weight_update_s)
        + float(weight_fold_s)
        for build_lora_s, lora_update_s, weight_update_s, weight_fold_s in zip(
            timing["build_lora_s"],
            timing["lora_update_s"],
            timing["weight_update_s"],
            timing["weight_fold_s"],
        )
    ]
    aligned_other_times = [
        max(
            0.0,
            float(step_s) - float(score_s) - float(direction_s) - float(update_s),
        )
        for step_s, score_s, direction_s, update_s in zip(
            timing["step_s"],
            timing["score_s"],
            timing["direction_s"],
            aligned_update_times,
        )
    ]
    best_eval_loss_metric = _best_eval_loss_metric(eval_metrics)
    if best_checkpoint is not None and best_checkpoint.get("metric") == "loss":
        checkpoint_eval = best_checkpoint.get("eval")
        if checkpoint_eval is not None:
            best_eval_loss_metric = _best_eval_loss_metric([checkpoint_eval])
    with open(output_file, "w") as f:
        json.dump(
            {
                "config": vars(args)
                | {
                    "model": model_name,
                    "backend": "vllm",
                    "direction_scale_effective": effective_direction_scale,
                    "direction_scale_mode": direction_scale_mode,
                    "direction_scale_note": direction_scale_note,
                    "direction_scale_applies_to": direction_scale_applies_to,
                    "output_dir_resolved": output_dir,
                    "output_root_resolved": output_root_resolved,
                    "experiment_name_resolved": experiment_name_resolved,
                    "lora_slot_rank_resolved": lora_slot_rank,
                    "prequant_model_resolved": prequant_model,
                    "accuracy_eval_mode_resolved": accuracy_eval_mode,
                    "train_objective_resolved": train_objective,
                    "unified_stepper": bool(use_unified_stepper),
                },
                "initial_loss": None if initial_loss is None else float(initial_loss),
                "final_loss": None if final_loss is None else float(final_loss),
                "loss_change": (
                    None
                    if initial_loss is None or final_loss is None
                    else float(final_loss - initial_loss)
                ),
                "eval_losses": eval_losses,
                "eval_metrics": eval_metrics,
                "eval_metrics_accuracy_scope": "dev",
                "final_accuracy_scope": "validation",
                "initial_accuracy": initial_dev_acc,
                "initial_dev_accuracy": initial_dev_acc,
                "initial_valid_accuracy": initial_valid_acc,
                "final_accuracy": final_valid_acc,
                "final_dev_accuracy": final_dev_acc,
                "final_valid_accuracy": final_valid_acc,
                "best_eval_loss_metric": best_eval_loss_metric,
                "best_eval_loss": (
                    None
                    if best_eval_loss_metric is None
                    else best_eval_loss_metric["loss"]
                ),
                "best_eval_loss_step": (
                    None
                    if best_eval_loss_metric is None
                    else best_eval_loss_metric["step"]
                ),
                "best_eval_loss_dev_accuracy": (
                    None
                    if best_eval_loss_metric is None
                    else best_eval_loss_metric.get("accuracy")
                ),
                "best_eval_loss_valid_accuracy": (
                    None
                    if best_eval_loss_metric is None
                    else best_eval_loss_metric.get("valid_accuracy")
                ),
                "history": history,
                "timing": {
                    "total_s": float(total_s),
                    **{key: summarize(value) for key, value in timing.items()},
                    "tail_100": {
                        key: summarize_tail(value, 100) for key, value in timing.items()
                    },
                    "aligned_phase_s": {
                        "probe_forward_s": summarize(timing["score_s"]),
                        "direction_s": summarize(timing["direction_s"]),
                        "update_s": summarize(aligned_update_times),
                        "other_s": summarize(aligned_other_times),
                        "tail_100": {
                            "probe_forward_s": summarize_tail(timing["score_s"], 100),
                            "direction_s": summarize_tail(timing["direction_s"], 100),
                            "update_s": summarize_tail(aligned_update_times, 100),
                            "other_s": summarize_tail(aligned_other_times, 100),
                        },
                    },
                },
                "checkpoints": ckpt_paths,
                "checkpoint_records": [] if checkpoint_records is None else checkpoint_records,
                "best_checkpoint": best_checkpoint,
                "loaded_best_checkpoint": loaded_best_checkpoint,
                "u_snapshots": u_snapshot_paths,
                "u_snapshot_metrics": u_snapshot_metrics,
            },
            f,
            indent=2,
        )
    return output_file
