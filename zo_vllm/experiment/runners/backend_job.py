import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.infra.env import configure_hf_cache, hf_cache_env  # noqa: E402
from zo_vllm.config import DEFAULT_ZO_TASK_SHUFFLE_IMPL  # noqa: E402
from zo_vllm.utils.io import (  # noqa: E402
    load_json,
    load_json_line,
    newest_path,
    write_json,
)
from zo_vllm.experiment.infra.paths import project_root, resolve_path  # noqa: E402
from zo_vllm.experiment.infra.run_state import (  # noqa: E402
    mark_run_completed,
    mark_run_failed,
    mark_run_running,
)
from zo_vllm.tasks import TaskConfig, get_task  # noqa: E402

PROJECT_ROOT = project_root()


def best_eval_loss_metric(eval_metrics: list[dict]) -> dict | None:
    rows = [row for row in eval_metrics if row.get("loss") is not None]
    if not rows:
        return None
    best = min(rows, key=lambda row: float(row["loss"]))
    payload = {"step": int(best["step"]), "loss": float(best["loss"])}
    if best.get("accuracy") is not None:
        payload["accuracy"] = float(best["accuracy"])
    if best.get("valid_accuracy") is not None:
        payload["valid_accuracy"] = float(best["valid_accuracy"])
    return payload


def build_task_config(args) -> TaskConfig:
    data_seed = args.seed if args.train_set_seed is None else args.train_set_seed
    return TaskConfig(
        name=args.task_name,
        num_train=args.num_samples,
        num_dev=args.num_dev,
        num_eval=args.eval_accuracy_samples,
        data_seed=data_seed,
        template=args.task_template,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )


def build_lozo_cmd(args, artifact_dir: Path) -> list[str]:
    task = get_task(args.task_name)
    task_cfg = build_task_config(args)
    if not task.supports_official_lozo:
        raise ValueError(
            f"task {args.task_name!r} does not support official LOZO backend"
        )
    if args.save_strategy == "best":
        raise ValueError("official LOZO backend does not support save_strategy=best")
    if args.load_best_model_at_end == "1" and args.save_strategy == "no":
        raise ValueError("load_best_model_at_end requires checkpoint saving")
    output_dir = artifact_dir / "official_output"
    result_file = artifact_dir / "official_metrics.json"
    eval_metrics_file = artifact_dir / "lozo_eval_metrics.jsonl"
    timing_file = artifact_dir / "lozo_timing.json"
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
        task.official_task_name,
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
        "--max_length",
        str(args.max_length),
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
        "--evaluation_strategy",
        "steps",
        "--save_strategy",
        args.save_strategy,
        "--eval_steps",
        str(args.eval_interval),
        "--step_interval",
        str(args.nu),
        "--rank_r",
        str(args.rank),
        "--lozo_train_scope",
        args.train_scope,
        "--result_file",
        str(result_file),
        "--eval_at_start",
        "--eval_accuracy_during_training",
        "--eval_metrics_file",
        str(eval_metrics_file),
        "--timing_file",
        str(timing_file),
        "--timing_warmup_steps",
        str(args.warmup_steps),
        "--timing_progress_interval",
        str(args.progress_interval),
        "--report_to",
        "none",
        "--overwrite_output_dir",
    ]
    if args.save_strategy != "no":
        cmd.extend(
            [
                "--save_total_limit",
                str(args.save_total_limit),
                "--save_steps",
                str(args.save_steps if args.save_steps > 0 else args.eval_interval),
            ]
        )
    if args.load_best_model_at_end == "1":
        cmd.append("--load_best_model_at_end")
    cmd.extend(task.official_lozo_args(task_cfg))
    if args.dataloader_seed is not None:
        cmd.extend(["--data_seed", str(args.dataloader_seed)])
    return cmd


def build_vllm_cmd(args, artifact_dir: Path) -> list[str]:
    if args.vllm_runner == "hf_phase3":
        if args.save_strategy != "no" or args.resume_lora_checkpoint is not None:
            raise ValueError("hf_phase3 speed jobs do not support checkpointing")
        return [
            ".venv/bin/python",
            "-u",
            "phase3/runners/hf_trainer_speed_migration.py",
            "--model",
            args.model_name,
            "--task",
            args.task_name,
            "--task-objective",
            "registered",
            "--output-root",
            str(artifact_dir),
            "--run-id",
            "hf_native",
            "--steps",
            str(args.steps),
            "--tail-steps",
            str(min(100, args.steps)),
            "--batch-size",
            str(args.batch_size),
            "--num-train",
            str(args.num_samples),
            "--num-dev",
            str(args.num_dev),
            "--max-length",
            str(args.max_length),
            "--max-model-len",
            str(args.max_length),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--rank",
            str(args.rank),
            "--nu",
            str(args.nu),
            "--lr",
            str(args.lr),
            "--lr-scheduler-type",
            "constant",
            "--eps",
            str(args.eps),
            "--direction-provider",
            args.direction_provider,
            "--lozo-provider-mode",
            args.lozo_provider_mode,
            "--random-device",
            args.zo_random_device,
            "--direction-sampling",
            args.direction_sampling,
            "--perturbation-normalization",
            args.perturbation_normalization,
            "--seed",
            str(args.seed),
            "--data-seed",
            str(args.seed if args.train_set_seed is None else args.train_set_seed),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--direct-update-mode",
            "accumulate" if args.direct_update_mode == "accumulate" else "direct",
            "--weight-update-precision",
            args.weight_update_precision,
            "--qkv-weight-update",
            args.qkv_weight_update,
            "--gradient-accumulation-update-steps",
            str(args.gradient_accumulation_update_steps),
            "--u-beta",
            str(args.u_beta),
            "--score-chunk-size",
            str(args.score_chunk_size),
            "--logging-steps",
            str(args.logging_steps),
            *(
                []
                if args.max_num_batched_tokens is None
                else ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
            ),
            *([] if args.u_norm_cap is None else ["--u-norm-cap", str(args.u_norm_cap)]),
        ]
    if args.vllm_runner == "hf_phase4":
        if args.direction_provider != "lozo" or args.lozo_provider_mode != "fast":
            raise ValueError("hf_phase4 currently requires the migrated LOZO fast provider")
        task = get_task(args.task_name)
        checkpoint_mode = str(args.save_checkpoint_mode)
        if checkpoint_mode == "auto":
            checkpoint_mode = (
                "native" if str(args.load_best_model_at_end) == "1" else "metadata"
            )
        if args.resume_lora_checkpoint is not None:
            checkpoint_mode = "lora"
        return [
            ".venv/bin/python",
            "-u",
            "phase4/runners/hf_trainer_sst2_alignment.py",
            "--mode",
            "hf",
            "--model",
            args.model_name,
            "--task-objective",
            task.vllm_train_objective,
            "--output-root",
            str(artifact_dir),
            "--run-id",
            "hf_native",
            "--steps",
            str(args.steps),
            "--eval-steps",
            str(args.eval_interval),
            "--batch-size",
            str(args.batch_size),
            "--num-train",
            str(args.num_samples),
            "--num-dev",
            str(args.num_dev),
            "--num-eval",
            str(args.eval_accuracy_samples),
            "--max-length",
            str(args.max_length),
            "--max-model-len",
            str(args.max_length),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--rank",
            str(args.rank),
            "--nu",
            str(args.nu),
            "--lr",
            str(args.lr),
            "--eps",
            str(args.eps),
            "--seed",
            str(args.seed),
            "--data-seed",
            str(args.seed if args.train_set_seed is None else args.train_set_seed),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens or 16384),
            "--max-num-seqs",
            str(max(16, int(args.batch_size) * 4)),
            "--direct-update-mode",
            "accumulate" if args.direct_update_mode == "accumulate" else "direct",
            "--weight-update-precision",
            args.weight_update_precision,
            "--qkv-weight-update",
            args.qkv_weight_update,
            "--gradient-accumulation-update-steps",
            str(args.gradient_accumulation_update_steps),
            "--u-beta",
            str(args.u_beta),
            "--save-strategy",
            args.save_strategy,
            "--save-steps",
            str(args.save_steps or args.eval_interval),
            "--save-total-limit",
            str(args.save_total_limit),
            "--checkpoint-mode",
            checkpoint_mode,
            "--load-best-model-at-end",
            str(args.load_best_model_at_end),
            "--metric-for-best-model",
            args.metric_for_best_model,
            "--greater-is-better",
            args.greater_is_better,
            *([] if args.u_norm_cap is None else ["--u-norm-cap", str(args.u_norm_cap)]),
            *(
                []
                if args.resume_lora_checkpoint is None
                else ["--resume-from-checkpoint", str(args.resume_lora_checkpoint)]
            ),
        ]
    task = get_task(args.task_name)
    task_cfg = build_task_config(args)
    data_seed = args.seed if args.train_set_seed is None else args.train_set_seed
    cmd = [
        ".venv/bin/python",
        "-u",
        "-m",
        "zo_vllm.experiment.runners.vllm_zo_task",
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
        "--nu",
        str(args.nu),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--data-seed",
        str(data_seed),
        "--zo-random-device",
        args.zo_random_device,
        "--direction-provider",
        args.direction_provider,
        "--lozo-provider-mode",
        args.lozo_provider_mode,
        "--direction-sampling",
        args.direction_sampling,
        "--perturbation-normalization",
        args.perturbation_normalization,
        "--train-scope",
        args.train_scope,
        "--train-sampler",
        args.train_sampler,
        "--dataloader-seed",
        str(args.seed if args.dataloader_seed is None else args.dataloader_seed),
        "--task-shuffle-impl",
        args.task_shuffle_impl,
        "--enforce-eager",
        "0",
        "--weight-update",
        "direct",
        "--weight-update-precision",
        args.weight_update_precision,
        "--direct-update-mode",
        args.direct_update_mode,
        "--quantized-update-mode",
        args.quantized_update_mode,
        "--update-bank-rank",
        str(args.update_bank_rank),
        "--gradient-accumulation-update-steps",
        str(args.gradient_accumulation_update_steps),
        "--u-beta",
        str(args.u_beta),
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
        "--score-chunk-size",
        str(args.score_chunk_size),
        "--direct-lora-from-directions",
        "1",
        "--base-eval-mode",
        "direct_worker",
        "--progress-interval",
        str(args.progress_interval),
        "--train-loss-interval",
        str(args.logging_steps),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--output-dir",
        str(artifact_dir),
        "--save-strategy",
        args.save_strategy,
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--save-checkpoint-mode",
        args.save_checkpoint_mode,
        "--load-best-model-at-end",
        str(args.load_best_model_at_end),
        "--metric-for-best-model",
        args.metric_for_best_model,
        "--greater-is-better",
        args.greater_is_better,
        "--save-final-checkpoint",
        str(args.save_final_checkpoint),
        "--eval-accuracy-samples",
        str(args.eval_accuracy_samples),
        "--max-model-len",
        str(args.max_length),
    ]
    if args.max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.u_norm_cap is not None:
        cmd += ["--u-norm-cap", str(args.u_norm_cap)]
    if args.resume_lora_checkpoint is not None:
        cmd += ["--resume-lora-checkpoint", str(args.resume_lora_checkpoint)]
    cmd.extend(task.vllm_args(task_cfg))
    return cmd


def result_json(backend: str, artifact_dir: Path) -> Path:
    if backend == "lozo":
        candidate = str(artifact_dir / "official_metrics.json")
    else:
        hf_speed_result = artifact_dir / "hf_native" / "result.json"
        hf_phase4_result = artifact_dir / "hf_native" / "hf_result.json"
        candidate_path = (
            hf_phase4_result
            if hf_phase4_result.is_file()
            else hf_speed_result
            if hf_speed_result.is_file()
            else newest_path(str(artifact_dir / "vllm_perf_*.json"))
        )
        candidate = str(candidate_path) if candidate_path is not None else None
    if candidate is None or not Path(candidate).exists():
        raise FileNotFoundError(f"result json not found in {artifact_dir}")
    return Path(candidate)


def load_backend_result(
    backend: str, artifact_dir: Path, train_scope: str | None = None
) -> dict:
    if backend != "lozo":
        return load_json(result_json(backend, artifact_dir))

    official_metrics_file = artifact_dir / "official_metrics.json"
    official_output_dir = artifact_dir / "official_output"
    timing_file = artifact_dir / "lozo_timing.json"
    if not official_metrics_file.exists():
        raise FileNotFoundError(f"official metrics not found: {official_metrics_file}")
    if not timing_file.exists():
        raise FileNotFoundError(f"LOZO timing file not found: {timing_file}")
    metrics = load_json(official_metrics_file)
    timing = load_json(timing_file)
    if timing.get("total_s") is None:
        raise ValueError(f"LOZO timing file missing total_s: {timing_file}")

    eval_metrics = []
    phase_eval_file = artifact_dir / "lozo_eval_metrics.jsonl"
    if phase_eval_file.exists():
        with phase_eval_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = load_json_line(line)
                if "step" not in row:
                    continue
                item = {"step": int(row["step"])}
                if "loss" in row:
                    item["loss"] = float(row["loss"])
                if "accuracy" in row:
                    item["accuracy"] = float(row["accuracy"])
                eval_metrics.append(item)
    for state_file in sorted(
        official_output_dir.glob("checkpoint-*/trainer_state.json")
    ):
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
    best_metric = best_eval_loss_metric(eval_metrics)

    return {
        "config": {
            "backend": "lozo_official",
            "train_scope": f"official_{train_scope or 'unknown'}",
            "official_metrics_file": str(official_metrics_file),
            "official_output_dir": str(official_output_dir),
            "phase_eval_metrics_file": str(phase_eval_file),
            "timing_file": str(timing_file),
        },
        "timing": timing,
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
        "best_eval_loss_metric": best_metric,
        "best_eval_loss": None if best_metric is None else best_metric["loss"],
        "best_eval_loss_step": None if best_metric is None else best_metric["step"],
        "best_eval_loss_dev_accuracy": (
            None if best_metric is None else best_metric.get("accuracy")
        ),
        "best_eval_loss_valid_accuracy": (
            None if best_metric is None else best_metric.get("valid_accuracy")
        ),
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
    parser.add_argument("--task-template", default="default")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=50)
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
    parser.add_argument("--nu", type=int, default=100)
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--direction-provider",
        choices=["lozo", "agzo", "uagzo", "suagzo"],
        default="lozo",
    )
    parser.add_argument(
        "--lozo-provider-mode", choices=["fast", "scheduled"], default="fast"
    )
    parser.add_argument(
        "--direction-sampling", choices=["exact", "flat"], default="exact"
    )
    parser.add_argument(
        "--perturbation-normalization", choices=["rms", "none"], default="rms"
    )
    parser.add_argument(
        "--train-scope",
        choices=["lora_normal", "lora_full", "full"],
        default="lora_normal",
    )
    parser.add_argument(
        "--train-sampler", choices=["sequential", "hf_random"], default="sequential"
    )
    parser.add_argument("--dataloader-seed", type=int, default=None)
    parser.add_argument(
        "--task-shuffle-impl",
        choices=["numpy", "hf"],
        default=DEFAULT_ZO_TASK_SHUFFLE_IMPL,
    )
    parser.add_argument(
        "--weight-update-precision", choices=["float32", "param"], default="param"
    )
    parser.add_argument(
        "--direct-update-mode",
        choices=["immediate", "accumulate"],
        default="accumulate",
    )
    parser.add_argument(
        "--quantized-update-mode", choices=["none", "lora_bank"], default="none"
    )
    parser.add_argument("--update-bank-rank", default="auto")
    parser.add_argument("--gradient-accumulation-update-steps", type=int, default=0)
    parser.add_argument("--u-beta", type=float, default=1.0)
    parser.add_argument("--u-norm-cap", type=float, default=None)
    parser.add_argument(
        "--qkv-weight-update", choices=["separate", "batched"], default="batched"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--score-chunk-size", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument(
        "--save-strategy", choices=["no", "steps", "best"], default="steps"
    )
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--save-checkpoint-mode",
        choices=["auto", "metadata", "native", "lora"],
        default="auto",
    )
    parser.add_argument("--load-best-model-at-end", choices=["0", "1"], default="0")
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument(
        "--greater-is-better", choices=["auto", "0", "1"], default="auto"
    )
    parser.add_argument("--save-final-checkpoint", choices=["0", "1"], default="0")
    parser.add_argument("--resume-lora-checkpoint", default=None)
    parser.add_argument("--eval-accuracy-samples", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--vllm-runner",
        choices=["legacy", "hf_phase3", "hf_phase4"],
        default="legacy",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["ZO_TASK_SHUFFLE_IMPL"] = args.task_shuffle_impl
    if args.backend == "vllm" and args.train_scope == "full":
        raise SystemExit(
            "vLLM backend does not support train_scope=full; "
            "use lora_full for all vLLM LoRA-compatible targets"
        )
    if args.backend == "lozo" and args.train_scope == "lora_full":
        raise SystemExit(
            "official LOZO backend does not support train_scope=lora_full; "
            "use lora_normal or full"
        )
    task = get_task(args.task_name)
    task_cfg = build_task_config(args)
    effective_save_steps = args.save_steps
    if (
        args.save_strategy != "no"
        and (args.save_strategy == "best" or effective_save_steps > 0)
        and args.save_total_limit == 1
    ):
        print(
            "[phase4] save_total_limit=1 would delete the previous checkpoint; "
            "using save_total_limit=2 to keep current and previous checkpoints.",
            flush=True,
        )
        args.save_total_limit = 2
    run_dir = resolve_path(args.run_dir)
    job_dir = run_dir / "jobs" / args.job_id
    artifacts_dir = job_dir / "artifacts"
    logs_dir = job_dir / "logs"
    log_file = logs_dir / "run.log"
    result_file = job_dir / "result.json"
    manifest_file = job_dir / "manifest.json"

    mark_run_running(job_dir, note=f"backend={args.backend}", resumed=args.resume)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    effective_train_scope = (
        f"official_{args.train_scope}" if args.backend == "lozo" else args.train_scope
    )
    effective_train_set_seed = (
        args.seed if args.train_set_seed is None else args.train_set_seed
    )

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    configure_hf_cache(PROJECT_ROOT)
    env.update(hf_cache_env(PROJECT_ROOT))
    env["WANDB_PROJECT"] = args.wandb_project
    env["WANDB_ENTITY"] = args.wandb_entity
    env["WANDB_MODE"] = env.get("WANDB_MODE", "online")

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="backend-train",
        name=f"{args.job_id}",
        config={
            "job_id": args.job_id,
            "backend": args.backend,
            "gpu": args.gpu,
            "model_name": args.model_name,
            "task_name": args.task_name,
            "task_adapter": task.name,
            "official_task_name": task.official_task_name,
            "vllm_train_objective": task.vllm_train_objective,
            "task_template": task_cfg.template,
            "max_length": task_cfg.max_length,
            "max_new_tokens": task_cfg.max_new_tokens,
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
            "nu": args.nu,
            "zo_random_device": args.zo_random_device,
            "direction_provider": args.direction_provider,
            "lozo_provider_mode": args.lozo_provider_mode,
            "direction_sampling": args.direction_sampling,
            "perturbation_normalization": args.perturbation_normalization,
            "train_scope": effective_train_scope,
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
            "resume_lora_checkpoint": args.resume_lora_checkpoint,
            "load_best_model_at_end": args.load_best_model_at_end,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": args.greater_is_better,
            "save_final_checkpoint": args.save_final_checkpoint,
            "eval_accuracy_samples": args.eval_accuracy_samples,
            "resume": args.resume,
        },
        reinit=True,
    )

    cmd = (
        build_lozo_cmd(args, artifacts_dir)
        if args.backend == "lozo"
        else build_vllm_cmd(args, artifacts_dir)
    )
    manifest = {
        "job_id": args.job_id,
        "backend": args.backend,
        "train_scope": effective_train_scope,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_checkpoint_mode": args.save_checkpoint_mode,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,
        "save_final_checkpoint": args.save_final_checkpoint,
        "eval_accuracy_samples": args.eval_accuracy_samples,
        "model_name": args.model_name,
        "task_name": args.task_name,
        "task_adapter": task.name,
        "official_task_name": task.official_task_name,
        "vllm_train_objective": task.vllm_train_objective,
        "task_template": task_cfg.template,
        "max_length": task_cfg.max_length,
        "max_new_tokens": task_cfg.max_new_tokens,
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
        "nu": args.nu,
        "zo_random_device": args.zo_random_device,
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
        "gradient_accumulation_update_steps": args.gradient_accumulation_update_steps,
        "u_beta": args.u_beta,
        "u_norm_cap": args.u_norm_cap,
        "qkv_weight_update": args.qkv_weight_update,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "score_chunk_size": args.score_chunk_size,
        "progress_interval": args.progress_interval,
        "save_strategy": args.save_strategy,
        "gpu": args.gpu,
        "command": cmd,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "resume": args.resume,
        "resume_lora_checkpoint": args.resume_lora_checkpoint,
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
        if "valid_accuracy" in row and row["valid_accuracy"] is not None:
            payload["eval_valid/accuracy"] = row["valid_accuracy"]
        if payload and "step" in row:
            payload["step"] = row["step"]
            wandb.log(payload)

    wall_clock_s = float(t1 - t0)
    timing_total_s = timing.get("total_s")
    if timing_total_s is None:
        raise ValueError(f"backend result for {args.job_id} is missing timing.total_s")
    timing_source = timing.get("timing_source", "backend_timing_total_s")
    total_step_s = float(timing_total_s) / float(args.steps) if args.steps > 0 else None
    steps_per_sec = (1.0 / total_step_s) if total_step_s and total_step_s > 0 else None
    initial_loss = raw.get("initial_loss")
    final_loss = raw.get("final_loss")
    final_accuracy = raw.get("final_accuracy")
    best_eval_loss_metric = raw.get("best_eval_loss_metric")
    best_eval_loss = raw.get("best_eval_loss")
    best_eval_loss_step = raw.get("best_eval_loss_step")
    best_eval_loss_valid_accuracy = raw.get("best_eval_loss_valid_accuracy")
    converged = None
    if initial_loss is not None and final_loss is not None:
        converged = float(final_loss) <= float(initial_loss)

    result = {
        "job_id": args.job_id,
        "backend": args.backend,
        "gpu": args.gpu,
        "status": "completed",
        "resume": bool(args.resume),
        "wall_clock_s": wall_clock_s,
        "timing_total_s": timing_total_s,
        "timing_source": timing_source,
        "step_time_s": total_step_s,
        "steps_per_sec": steps_per_sec,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_change": raw.get("loss_change"),
        "eval_losses": eval_losses,
        "eval_metrics": eval_metrics,
        "final_accuracy": final_accuracy,
        "best_eval_loss_metric": best_eval_loss_metric,
        "best_eval_loss": best_eval_loss,
        "best_eval_loss_step": best_eval_loss_step,
        "best_eval_loss_dev_accuracy": raw.get("best_eval_loss_dev_accuracy"),
        "best_eval_loss_valid_accuracy": best_eval_loss_valid_accuracy,
        "checkpoint_records": raw.get("checkpoint_records", []),
        "best_checkpoint": raw.get("best_checkpoint"),
        "loaded_best_checkpoint": raw.get("loaded_best_checkpoint"),
        "converged": converged,
        "artifact_json": raw.get(
            "artifact_json",
            (
                str(result_json(args.backend, artifacts_dir))
                if args.backend != "lozo"
                else None
            ),
        ),
        "log_file": str(log_file),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_id": wandb_run.id,
        "wandb_url": wandb_run.url,
        "config": raw.get("config", {}),
    }
    write_json(result_file, result)

    wandb_run.summary["status"] = "completed"
    wandb_run.summary["wall_clock_s"] = wall_clock_s
    wandb_run.summary["step_time_s"] = total_step_s
    wandb_run.summary["steps_per_sec"] = steps_per_sec
    wandb_run.summary["initial_loss"] = initial_loss
    wandb_run.summary["final_loss"] = final_loss
    wandb_run.summary["final_accuracy"] = final_accuracy
    wandb_run.summary["best_eval_loss"] = best_eval_loss
    wandb_run.summary["best_eval_loss_step"] = best_eval_loss_step
    wandb_run.summary["best_eval_loss_valid_accuracy"] = best_eval_loss_valid_accuracy
    wandb_run.summary["converged"] = converged
    wandb.finish()

    mark_run_completed(job_dir, note="result.json generated")


if __name__ == "__main__":
    main()
