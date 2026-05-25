import argparse
import glob
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.env import configure_hf_cache
from zo_vllm.experiment.paths import project_root, resolve_path
from zo_vllm.experiment.io import load_json, write_json
from zo_vllm.experiment.run_state import (
    mark_run_completed,
    mark_run_failed,
    mark_run_running,
)


PROJECT_ROOT = project_root()


def newest(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def build_lozo_cmd(args, artifact_dir: Path) -> list[str]:
    output_dir = artifact_dir / "official_output"
    result_file = artifact_dir / "official_metrics.json"
    train_set_seed = args.seed if args.train_set_seed is None else args.train_set_seed
    cmd = [
        "third_party/LOZO/large_models/.venv/bin/python",
        "-u",
        "third_party/LOZO/large_models/run_lozo.py",
        "--seed",
        str(args.seed),
        "--model_name",
        args.model_name,
        "--task_name",
        args.task_name,
        "--output_dir",
        str(output_dir),
        "--tag",
        f"{args.job_id}",
        "--train_set_seed",
        str(train_set_seed),
        "--num_train",
        str(args.num_samples),
        "--num_dev",
        str(args.num_dev),
        "--num_eval",
        str(args.eval_accuracy_samples),
        "--logging_steps",
        str(args.logging_steps),
        "--max_steps",
        str(args.steps),
        "--trainer",
        "LOZO",
        "--load_float16",
        "--learning_rate",
        str(args.lr),
        "--zo_eps",
        str(args.eps),
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--lr_scheduler_type",
        "constant",
        "--load_best_model_at_end",
        "--evaluation_strategy",
        "steps",
        "--save_strategy",
        "steps",
        "--save_total_limit",
        str(args.save_total_limit),
        "--eval_steps",
        str(args.eval_interval),
        "--save_steps",
        str(args.save_interval if args.save_interval > 0 else args.eval_interval),
        "--train_as_classification",
        "--step_interval",
        str(args.step_interval),
        "--rank_r",
        str(args.rank),
        "--lozo_train_scope",
        args.train_scope,
        "--result_file",
        str(result_file),
        "--overwrite_output_dir",
    ]
    if args.dataloader_seed is not None:
        cmd.extend(["--data_seed", str(args.dataloader_seed)])
    return cmd


def build_vllm_cmd(args, artifact_dir: Path) -> list[str]:
    data_seed = args.seed if args.train_set_seed is None else args.train_set_seed
    return [
        ".venv/bin/python",
        "-u",
        "phase3/runners/train_vllm_perf.py",
        "--model-name",
        args.model_name,
        "--profile-mode",
        "minimal",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(args.batch_size),
        "--num-samples",
        str(args.num_samples),
        "--num-dev",
        str(args.num_dev),
        "--rank",
        str(args.rank),
        "--lr",
        str(args.lr),
        "--eps",
        str(args.eps),
        "--step-interval",
        str(args.step_interval),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--data-seed",
        str(data_seed),
        "--zo-random-device",
        args.zo_random_device,
        "--direction-sampling",
        args.direction_sampling,
        "--train-scope",
        "lora_only",
        "--train-sampler",
        args.train_sampler,
        "--dataloader-seed",
        str(args.seed if args.dataloader_seed is None else args.dataloader_seed),
        "--batch-invariant",
        "0",
        "--enforce-eager",
        "0",
        "--lora-residency",
        "gpu",
        "--lora-injection",
        "direct",
        "--weight-update",
        "direct",
        "--weight-update-precision",
        args.weight_update_precision,
        "--qkv-weight-update",
        args.qkv_weight_update,
        "--sync-weight-update",
        "0",
        "--scoring-backend",
        "direct_worker",
        "--direct-worker-max-logits-tokens",
        "8192",
        "--direct-worker-loss-impl",
        "logprobs",
        "--direct-lora-from-directions",
        "1",
        "--slot-pipeline",
        "0",
        "--base-eval-mode",
        "direct_worker",
        "--progress-interval",
        str(args.progress_interval),
        "--train-loss-interval",
        str(args.logging_steps),
        "--gpu-memory-utilization",
        "0.5",
        "--output-dir",
        str(artifact_dir),
        "--save-interval",
        str(args.save_interval),
        "--save-total-limit",
        str(args.save_total_limit),
        "--eval-accuracy-samples",
        str(args.eval_accuracy_samples),
    ]


def result_json(backend: str, artifact_dir: Path) -> Path:
    if backend == "lozo":
        candidate = str(artifact_dir / "official_metrics.json")
    else:
        candidate = newest(str(artifact_dir / "vllm_perf_*.json"))
    if candidate is None or not Path(candidate).exists():
        raise FileNotFoundError(f"result json not found in {artifact_dir}")
    return Path(candidate)


def load_backend_result(backend: str, artifact_dir: Path, train_scope: str | None = None) -> dict:
    if backend != "lozo":
        return load_json(result_json(backend, artifact_dir))

    official_metrics_file = artifact_dir / "official_metrics.json"
    official_output_dir = artifact_dir / "official_output"
    if not official_metrics_file.exists():
        raise FileNotFoundError(f"official metrics not found: {official_metrics_file}")
    metrics = load_json(official_metrics_file)

    eval_metrics = []
    for state_file in sorted(official_output_dir.glob("checkpoint-*/trainer_state.json")):
        state = load_json(state_file)
        for row in state.get("log_history", []):
            if "eval_loss" not in row:
                continue
            step = int(row.get("step", 0))
            item = {"step": step, "loss": float(row["eval_loss"])}
            if "eval_accuracy" in row:
                item["accuracy"] = float(row["eval_accuracy"])
            eval_metrics.append(item)

    dedup = {}
    for item in eval_metrics:
        dedup[item["step"]] = item
    eval_metrics = [dedup[step] for step in sorted(dedup)]
    final_accuracy = metrics.get("accuracy")
    if final_accuracy is None:
        final_accuracy = metrics.get("eval_accuracy")
    if eval_metrics and final_accuracy is not None:
        eval_metrics[-1]["accuracy"] = float(final_accuracy)

    eval_losses = [
        {"step": item["step"], "loss": item["loss"]}
        for item in eval_metrics
        if "loss" in item
    ]
    initial_loss = (
        eval_losses[0]["loss"]
        if eval_losses and int(eval_losses[0].get("step", -1)) == 0
        else None
    )
    final_loss = eval_losses[-1]["loss"] if eval_losses else None

    return {
        "config": {
            "backend": "lozo_official",
            "train_scope": f"official_{train_scope or 'unknown'}",
            "official_metrics_file": str(official_metrics_file),
            "official_output_dir": str(official_output_dir),
        },
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_change": (
            None
            if initial_loss is None or final_loss is None
            else float(final_loss - initial_loss)
        ),
        "eval_losses": eval_losses,
        "eval_metrics": eval_metrics,
        "final_accuracy": None if final_accuracy is None else float(final_accuracy),
        "official_metrics": metrics,
        "artifact_json": str(official_metrics_file),
    }


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as logf:
        logf.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        logf.write(f"\n[exit_code] {proc.returncode}\n")
        logf.flush()
        return proc.returncode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--backend", choices=["lozo", "vllm"], required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--wandb-project", default="lozo-vllm-phase4")
    parser.add_argument("--wandb-entity", default="playeriv65-university-of-minnesota")
    parser.add_argument("--model-name", default="facebook/opt-1.3b")
    parser.add_argument("--task-name", default="SST2")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-dev", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=4000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-set-seed", type=int, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--step-interval", type=int, default=100)
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--direction-sampling", choices=["exact", "flat"], default="exact")
    parser.add_argument("--train-scope", choices=["lora_only", "full"], default="lora_only")
    parser.add_argument("--train-sampler", choices=["sequential", "hf_random"], default="sequential")
    parser.add_argument("--dataloader-seed", type=int, default=None)
    parser.add_argument("--weight-update-precision", choices=["float32", "param"], default="param")
    parser.add_argument("--qkv-weight-update", choices=["separate", "batched"], default="batched")
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-accuracy-samples", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    job_dir = run_dir / "jobs" / args.job_id
    artifacts_dir = job_dir / "artifacts"
    logs_dir = job_dir / "logs"
    log_file = logs_dir / "run.log"
    result_file = job_dir / "phase4_result.json"
    manifest_file = job_dir / "manifest.json"

    mark_run_running(job_dir, note=f"backend={args.backend}", resumed=args.resume)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    effective_train_scope = f"official_{args.train_scope}" if args.backend == "lozo" else args.train_scope
    effective_train_set_seed = args.seed if args.train_set_seed is None else args.train_set_seed

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env["VLLM_BATCH_INVARIANT"] = "0"
    cache_root = configure_hf_cache(PROJECT_ROOT)
    env["HF_HOME"] = str(cache_root / "home")
    env["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    env["HF_HUB_CACHE"] = str(cache_root / "hub")
    env["HF_XET_CACHE"] = str(cache_root / "xet")
    env["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    env["WANDB_PROJECT"] = args.wandb_project
    env["WANDB_ENTITY"] = args.wandb_entity
    env["WANDB_MODE"] = env.get("WANDB_MODE", "online")

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="phase4-train",
        name=f"{args.job_id}",
        config={
            "job_id": args.job_id,
            "backend": args.backend,
            "gpu": args.gpu,
            "model_name": args.model_name,
            "task_name": args.task_name,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "num_dev": args.num_dev,
            "eval_interval": args.eval_interval,
            "logging_steps": args.logging_steps,
            "seed": args.seed,
            "train_set_seed": effective_train_set_seed,
            "rank": args.rank,
            "lr": args.lr,
            "eps": args.eps,
            "step_interval": args.step_interval,
            "zo_random_device": args.zo_random_device,
            "direction_sampling": args.direction_sampling,
            "train_scope": effective_train_scope,
            "train_sampler": args.train_sampler,
            "dataloader_seed": args.dataloader_seed,
            "weight_update_precision": args.weight_update_precision,
            "qkv_weight_update": args.qkv_weight_update,
            "progress_interval": args.progress_interval,
            "save_interval": args.save_interval,
            "save_total_limit": args.save_total_limit,
            "eval_accuracy_samples": args.eval_accuracy_samples,
            "resume": args.resume,
        },
        reinit=True,
    )

    cmd = build_lozo_cmd(args, artifacts_dir) if args.backend == "lozo" else build_vllm_cmd(args, artifacts_dir)
    manifest = {
        "job_id": args.job_id,
        "backend": args.backend,
        "train_scope": effective_train_scope,
        "save_interval": args.save_interval,
        "save_total_limit": args.save_total_limit,
        "eval_accuracy_samples": args.eval_accuracy_samples,
        "model_name": args.model_name,
        "task_name": args.task_name,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "num_dev": args.num_dev,
        "eval_interval": args.eval_interval,
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "train_set_seed": effective_train_set_seed,
        "rank": args.rank,
        "lr": args.lr,
        "eps": args.eps,
        "step_interval": args.step_interval,
        "zo_random_device": args.zo_random_device,
        "direction_sampling": args.direction_sampling,
        "train_sampler": args.train_sampler,
        "dataloader_seed": args.dataloader_seed,
        "weight_update_precision": args.weight_update_precision,
        "qkv_weight_update": args.qkv_weight_update,
        "progress_interval": args.progress_interval,
        "gpu": args.gpu,
        "command": cmd,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "resume": args.resume,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
    }
    write_json(manifest_file, manifest)

    t0 = time.time()
    rc = run_cmd(cmd, log_file, env)
    t1 = time.time()

    if rc != 0:
        mark_run_failed(job_dir, note=f"exit_code={rc}")
        wandb_run.summary["status"] = "failed"
        wandb_run.summary["exit_code"] = rc
        wandb.finish()
        raise SystemExit(rc)

    raw = load_backend_result(args.backend, artifacts_dir, args.train_scope)
    timing = raw.get("timing", {})
    eval_losses = raw.get("eval_losses", [])
    eval_metrics = raw.get("eval_metrics", [])
    for row in eval_losses:
        if "step" in row and "loss" in row:
            wandb.log({"eval/loss": row["loss"], "step": row["step"]})
    for row in eval_metrics:
        payload = {}
        if "loss" in row and row["loss"] is not None:
            payload["eval/loss"] = row["loss"]
        if "accuracy" in row and row["accuracy"] is not None:
            payload["eval/accuracy"] = row["accuracy"]
        if payload and "step" in row:
            payload["step"] = row["step"]
            wandb.log(payload)

    wall_clock_s = float(t1 - t0)
    total_step_s = float(timing.get("total_s", 0.0)) / float(args.steps) if args.steps > 0 else None
    steps_per_sec = (1.0 / total_step_s) if total_step_s and total_step_s > 0 else None
    initial_loss = raw.get("initial_loss")
    final_loss = raw.get("final_loss")
    final_accuracy = raw.get("final_accuracy")
    converged = None
    if initial_loss is not None and final_loss is not None:
        converged = float(final_loss) <= float(initial_loss)

    phase4_result = {
        "job_id": args.job_id,
        "backend": args.backend,
        "gpu": args.gpu,
        "status": "completed",
        "resume": bool(args.resume),
        "wall_clock_s": wall_clock_s,
        "timing_total_s": timing.get("total_s"),
        "step_time_s": total_step_s,
        "steps_per_sec": steps_per_sec,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_change": raw.get("loss_change"),
        "eval_losses": eval_losses,
        "eval_metrics": eval_metrics,
        "final_accuracy": final_accuracy,
        "converged": converged,
        "artifact_json": raw.get("artifact_json", str(result_json(args.backend, artifacts_dir)) if args.backend != "lozo" else None),
        "log_file": str(log_file),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_id": wandb_run.id,
        "wandb_url": wandb_run.url,
        "config": raw.get("config", {}),
    }
    write_json(result_file, phase4_result)

    wandb_run.summary["status"] = "completed"
    wandb_run.summary["wall_clock_s"] = wall_clock_s
    wandb_run.summary["step_time_s"] = total_step_s
    wandb_run.summary["steps_per_sec"] = steps_per_sec
    wandb_run.summary["initial_loss"] = initial_loss
    wandb_run.summary["final_loss"] = final_loss
    wandb_run.summary["final_accuracy"] = final_accuracy
    wandb_run.summary["converged"] = converged
    wandb.finish()

    mark_run_completed(job_dir, note="phase4_result.json generated")


if __name__ == "__main__":
    main()
