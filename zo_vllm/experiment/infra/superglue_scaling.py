"""Shared Phase 3 SST-2 and SuperGLUE scaling sweep helpers."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from zo_vllm.experiment.infra.naming import safe_model_name
from zo_vllm.tasks.registry import get_task
from zo_vllm.tasks.superglue import SUPERGLUE_TASK_SPECS

ALL_SUPERGLUE_TASKS = tuple(SUPERGLUE_TASK_SPECS)
DEFAULT_PHASE3_SCALING_TASKS = ("sst2", *ALL_SUPERGLUE_TASKS)
LOZO_SMALL_DEV_TASKS = {"superglue_cb", "superglue_copa"}


def safe_job_part(value: str) -> str:
    return (
        str(value)
        .replace("/", "__")
        .replace(":", "_")
        .replace(".", "p")
        .replace("-", "_")
    )


def job_id_for(task_name: str, model: str, batch_size: int, backend: str) -> str:
    task = get_task(task_name)
    return (
        f"{safe_job_part(task.name)}__{safe_model_name(model)}__"
        f"b{batch_size}__{backend}"
    )


def task_num_dev(task_name: str, requested_num_dev: int) -> int:
    task = get_task(task_name)
    if task.name in LOZO_SMALL_DEV_TASKS:
        return min(int(requested_num_dev), 100)
    return int(requested_num_dev)


def job_result_path(base_dir: Path, job_id: str) -> Path:
    return base_dir / "jobs" / job_id / "result.json"


def backend_order(backend: str) -> int:
    return {"lozo": 0, "vllm": 1}.get(str(backend), 99)


def job_pair_sort_key(spec: dict) -> tuple:
    return (
        int(spec.get("task_index", 0)),
        int(spec.get("model_index", 0)),
        int(spec.get("batch_index", 0)),
        backend_order(str(spec["backend"])),
    )


def build_superglue_scaling_job_specs(args, base_dir: Path) -> list[dict]:
    specs = []
    backends = ["lozo", "vllm"] if args.backend == "both" else [args.backend]
    for task_index, task_name in enumerate(args.tasks):
        task = get_task(task_name)
        effective_num_dev = task_num_dev(task.name, args.num_dev)
        for model_index, model in enumerate(args.models):
            for batch_index, batch_size in enumerate(args.batch_sizes):
                for backend in backends:
                    job_id = job_id_for(task.name, model, batch_size, backend)
                    if (
                        args.resume_existing
                        and job_result_path(base_dir, job_id).exists()
                    ):
                        continue
                    specs.append(
                        {
                            "job_id": job_id,
                            "backend": backend,
                            "vllm_runner": "hf_phase3",
                            "task_index": task_index,
                            "model_index": model_index,
                            "batch_index": batch_index,
                            "task_name": task.name,
                            "model_name": model,
                            "batch_size": batch_size,
                            "steps": args.steps,
                            "warmup_steps": 0,
                            "num_samples": args.num_samples,
                            "num_dev": effective_num_dev,
                            "eval_interval": args.eval_interval,
                            "logging_steps": args.logging_steps,
                            "seed": args.seed,
                            "train_set_seed": args.train_set_seed,
                            "rank": args.rank,
                            "lr": args.lr,
                            "eps": args.eps,
                            "nu": args.nu,
                            "direction_provider": args.direction_provider,
                            "lozo_provider_mode": args.lozo_provider_mode,
                            "direction_sampling": args.direction_sampling,
                            "perturbation_normalization": args.perturbation_normalization,
                            "train_sampler": args.train_sampler,
                            "dataloader_seed": args.dataloader_seed,
                            "task_shuffle_impl": args.task_shuffle_impl,
                            "weight_update_precision": args.weight_update_precision,
                            "direct_update_mode": args.direct_update_mode,
                            "quantized_update_mode": args.quantized_update_mode,
                            "update_bank_rank": args.update_bank_rank,
                            "gradient_accumulation_update_steps": (
                                args.gradient_accumulation_update_steps
                            ),
                            "u_beta": args.u_beta,
                            "u_norm_cap": args.u_norm_cap,
                            "qkv_weight_update": args.qkv_weight_update,
                            "gpu_memory_utilization": args.gpu_memory_utilization,
                            "max_num_batched_tokens": args.max_num_batched_tokens,
                            "score_chunk_size": args.score_chunk_size,
                            "progress_interval": args.progress_interval,
                            "save_strategy": args.save_strategy,
                            "save_steps": args.save_steps,
                            "save_total_limit": args.save_total_limit,
                            "save_checkpoint_mode": args.save_checkpoint_mode,
                            "load_best_model_at_end": args.load_best_model_at_end,
                            "metric_for_best_model": args.metric_for_best_model,
                            "greater_is_better": args.greater_is_better,
                            "save_final_checkpoint": args.save_final_checkpoint,
                            "eval_accuracy_samples": args.eval_accuracy_samples,
                            "wandb_project": args.wandb_project,
                            "wandb_entity": args.wandb_entity,
                            "wandb_mode": args.wandb_mode,
                        }
                    )
    return sorted(specs, key=job_pair_sort_key)


def load_result(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def require_step_time(result_path: Path, result: dict) -> float:
    step_time = result.get("step_time_s")
    if step_time is None:
        raise ValueError(f"missing step_time_s in {result_path}")
    step_time = float(step_time)
    if step_time <= 0:
        raise ValueError(f"non-positive step_time_s={step_time} in {result_path}")
    return step_time


def discover_superglue_scaling_rows(run_dir: Path) -> list[dict]:
    rows = []
    for result_path in sorted((run_dir / "jobs").glob("*/result.json")):
        result = load_result(result_path)
        config = result.get("config", {})
        manifest_path = result_path.parent / "manifest.json"
        manifest = load_result(manifest_path) if manifest_path.exists() else {}
        gpu = result.get("gpu")
        gpu_monitor_csv = None
        if gpu is not None:
            candidate = (
                result_path.parent
                / "logs"
                / f"gpu_util_{result_path.parent.name}_gpu{gpu}.csv"
            )
            if candidate.exists():
                gpu_monitor_csv = candidate
        rows.append(
            {
                "job_id": result.get("job_id"),
                "backend": result.get("backend"),
                "task": manifest.get("task_name") or config.get("task_name"),
                "model": manifest.get("model_name") or config.get("model_name"),
                "batch": manifest.get("batch_size") or config.get("batch_size"),
                "steps": manifest.get("steps") or config.get("steps"),
                "step_time_s": require_step_time(result_path, result),
                "steps_per_sec": result.get("steps_per_sec"),
                "wall_clock_s": result.get("wall_clock_s"),
                "initial_loss": result.get("initial_loss"),
                "final_loss": result.get("final_loss"),
                "loss_change": result.get("loss_change"),
                "final_accuracy": result.get("final_accuracy"),
                "best_eval_loss": result.get("best_eval_loss"),
                "best_eval_loss_step": result.get("best_eval_loss_step"),
                "best_eval_loss_valid_accuracy": result.get(
                    "best_eval_loss_valid_accuracy"
                ),
                "wandb_url": result.get("wandb_url"),
                "gpu_monitor_csv": gpu_monitor_csv,
                "path": result_path,
            }
        )
    return rows


def pair_key(row: dict) -> tuple:
    return (row["task"], row["model"], int(row["batch"]))


def pair_superglue_scaling_rows(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        by_key.setdefault(pair_key(row), {})[str(row["backend"])] = row
    paired = []
    for key, entries in sorted(by_key.items()):
        lozo = entries.get("lozo")
        vllm = entries.get("vllm")
        lozo_step = None if lozo is None else lozo.get("step_time_s")
        vllm_step = None if vllm is None else vllm.get("step_time_s")
        speedup = (
            float(lozo_step) / float(vllm_step)
            if lozo_step is not None and vllm_step not in {None, 0}
            else None
        )
        paired.append(
            {
                "task": key[0],
                "model": key[1],
                "batch": key[2],
                "lozo": lozo,
                "vllm": vllm,
                "lozo_step_time_s": lozo_step,
                "vllm_step_time_s": vllm_step,
                "speedup": speedup,
            }
        )
    return paired


def mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return statistics.mean(present)
