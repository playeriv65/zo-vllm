import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_PHASE3_DATA_SEED = 42

from zo_vllm.experiment.infra.job_queue import write_pending_jobs  # noqa: E402
from zo_vllm.experiment.infra.launch import (  # noqa: E402
    DEFAULT_TMUX_SESSION,
    ensure_run_dir_available,
    ensure_tmux_session,
    parse_csv,
    parse_int_csv,
    run_preflight,
    shell_join,
    tmux_new_window_command,
)
from zo_vllm.experiment.infra.manifest import (  # noqa: E402
    append_launch_record,
    write_manifest,
)
from zo_vllm.experiment.infra.naming import safe_model_name, timestamp_now  # noqa: E402
from zo_vllm.experiment.infra.superglue_scaling import (  # noqa: E402
    ALL_SUPERGLUE_TASKS,
    DEFAULT_PHASE3_SCALING_TASKS,
    build_superglue_scaling_job_specs,
    safe_job_part,
)
from zo_vllm.tasks.registry import get_task  # noqa: E402


def worker_command(args: argparse.Namespace, base_dir: Path, gpu: str) -> list[str]:
    worker_id = f"gpu{safe_job_part(gpu)}"
    worker_log = base_dir / "logs" / f"worker_{worker_id}_{args.timestamp}.log"
    cmd = [
        ".venv/bin/python",
        "-u",
        "-m",
        "zo_vllm.experiment.runners.job_queue_worker",
        "--run-dir",
        str(base_dir.relative_to(PROJECT_ROOT)),
        "--gpu",
        gpu,
        "--worker-id",
        worker_id,
    ]
    if args.stop_on_failure:
        cmd.append("--stop-on-failure")
    body = (
        f"set -o pipefail; cd {shlex.quote(str(PROJECT_ROOT))}; "
        f"{shell_join(cmd)} 2>&1 | tee {shlex.quote(str(worker_log))}"
    )
    window = f"repo-p3-scaling-gpu{safe_job_part(gpu)}"
    return tmux_new_window_command(args.tmux_session, window, body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a dynamic GPU queue for Phase 3 SST-2 then SuperGLUE scaling runs."
    )
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpus", type=parse_csv, default=None)
    parser.add_argument(
        "--tasks",
        type=parse_csv,
        default=list(DEFAULT_PHASE3_SCALING_TASKS),
        help="Ordered task list. Defaults to SST-2 first, then all SuperGLUE tasks.",
    )
    parser.add_argument(
        "--models",
        type=parse_csv,
        default=parse_csv(
            "facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,facebook/opt-13b"
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_csv,
        default=parse_int_csv("16,32,64,128"),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-dev", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-set-seed", type=int, default=DEFAULT_PHASE3_DATA_SEED)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--nu", type=int, default=50)
    parser.add_argument(
        "--direction-provider",
        choices=["lozo", "agzo", "uagzo", "suagzo"],
        default="lozo",
    )
    parser.add_argument(
        "--lozo-provider-mode", choices=["fast", "scheduled"], default="fast"
    )
    parser.add_argument(
        "--direction-sampling", choices=["exact", "flat"], default="flat"
    )
    parser.add_argument(
        "--perturbation-normalization", choices=["rms", "none"], default="rms"
    )
    parser.add_argument(
        "--train-sampler", choices=["sequential", "hf_random"], default="sequential"
    )
    parser.add_argument("--dataloader-seed", type=int, default=DEFAULT_PHASE3_DATA_SEED)
    parser.add_argument(
        "--task-shuffle-impl",
        choices=["numpy", "hf"],
        default="numpy",
        help=(
            "Dataset row shuffle implementation. Keep numpy for current Phase 3 "
            "speed results; use hf only when reproducing older HF dataset.shuffle "
            "artifacts."
        ),
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
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--score-chunk-size", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument(
        "--save-strategy", choices=["no", "steps", "best"], default="no"
    )
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--save-checkpoint-mode",
        choices=["auto", "metadata", "native", "lora"],
        default="metadata",
    )
    parser.add_argument("--load-best-model-at-end", choices=["0", "1"], default="0")
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument(
        "--greater-is-better", choices=["auto", "0", "1"], default="auto"
    )
    parser.add_argument("--save-final-checkpoint", choices=["0", "1"], default="0")
    parser.add_argument("--eval-accuracy-samples", type=int, default=512)
    parser.add_argument("--wandb-project", default="lozo-vllm-phase3-scaling-speed")
    parser.add_argument("--wandb-entity", default="playeriv65-university-of-minnesota")
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--backend", choices=["both", "lozo", "vllm"], default="both")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--allow-non-numeric-gpu", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.gpus:
        raise SystemExit("set --gpus, for example --gpus 2,3")
    if args.steps < 100:
        raise SystemExit("--steps must be at least 100 for Phase 3 tail-100 timing")
    for task_name in args.tasks:
        get_task(task_name)
    if args.num_samples < max(args.batch_sizes):
        raise SystemExit("--num-samples must cover the largest --batch-sizes value")
    if args.eval_interval <= 0:
        raise SystemExit(
            "--eval-interval must be positive for the official LOZO wrapper"
        )
    args.timestamp = timestamp_now()
    if args.run_id is None:
        task_text = (
            "sst2_allsg"
            if tuple(args.tasks) == DEFAULT_PHASE3_SCALING_TASKS
            else "allsg"
            if tuple(args.tasks) == ALL_SUPERGLUE_TASKS
            else "sst2"
            if tuple(args.tasks) == ("sst2",)
            else f"{len(args.tasks)}tasks"
        )
        model_text = "-".join(
            safe_model_name(model).removeprefix("facebook__opt-")
            for model in args.models
        )
        batch_text = "-".join(str(batch) for batch in args.batch_sizes)
        args.run_id = f"phase3_scaling_{task_text}_{model_text}_b{batch_text}_s{args.steps}_{args.timestamp}"
    return args


def main() -> None:
    args = parse_args()
    base_dir = PROJECT_ROOT / "phase3" / "results" / args.run_id
    specs = build_superglue_scaling_job_specs(args, base_dir)
    collect_command = [
        ".venv/bin/python",
        "-u",
        "-m",
        "zo_vllm.experiment.runners.collect_superglue_scaling",
        str(base_dir.relative_to(PROJECT_ROOT)),
        "--output",
        str((base_dir / "summary.md").relative_to(PROJECT_ROOT)),
    ]
    experiment_parameters = {
        "tasks": args.tasks,
        "models": args.models,
        "batch_sizes": args.batch_sizes,
        "backends": ["lozo", "vllm"] if args.backend == "both" else [args.backend],
        "gpus": args.gpus,
        "steps": args.steps,
        "warmup_steps": 0,
        "measurement": "timing.tail_100.step_s.mean",
        "num_samples": args.num_samples,
        "num_dev": args.num_dev,
        "eval_interval": args.eval_interval,
        "train_set_seed": args.train_set_seed,
        "train_sampler": args.train_sampler,
        "dataloader_seed": args.dataloader_seed,
        "task_shuffle_impl": args.task_shuffle_impl,
        "rank": args.rank,
        "lr": args.lr,
        "eps": args.eps,
        "nu": args.nu,
        "direction_provider": args.direction_provider,
        "lozo_provider_mode": args.lozo_provider_mode,
        "direction_sampling": args.direction_sampling,
        "perturbation_normalization": args.perturbation_normalization,
        "queue_policy": "zo_vllm.experiment filesystem queue with atomic rename",
        "resume_existing": args.resume_existing,
    }
    tmux_commands = [worker_command(args, base_dir, gpu) for gpu in args.gpus]
    print(f"run_dir={base_dir}", flush=True)
    print(f"job_count={len(specs)}", flush=True)
    print(
        "experiment_parameters=" + json.dumps(experiment_parameters, sort_keys=True),
        flush=True,
    )
    print("collect_command=" + shell_join(collect_command), flush=True)
    for command in tmux_commands:
        print(shell_join(command), flush=True)
    if args.dry_run:
        for spec in specs[:10]:
            print("job=" + json.dumps(spec, sort_keys=True), flush=True)
        if len(specs) > 10:
            print(f"... {len(specs) - 10} more jobs", flush=True)
        return

    if not args.resume_existing:
        ensure_run_dir_available(base_dir)
    base_dir.mkdir(parents=True, exist_ok=args.resume_existing)
    (base_dir / "logs").mkdir(parents=True, exist_ok=True)
    if not args.skip_preflight:
        preflight_report = run_preflight(
            args.gpus,
            args.allow_non_numeric_gpu,
            cwd=PROJECT_ROOT,
        )
        print(
            "preflight_report=" + json.dumps(preflight_report, sort_keys=True),
            flush=True,
        )
        if not preflight_report.get("ok", False):
            raise SystemExit("preflight failed")
    else:
        preflight_report = None
    write_pending_jobs(base_dir, specs)
    manifest = {
        "run_id": args.run_id,
        "run_dir": str(base_dir.relative_to(PROJECT_ROOT)),
        "timestamp": args.timestamp,
        "tmux_session": args.tmux_session,
        "experiment_parameters": experiment_parameters,
        "job_count": len(specs),
        "preflight_report": preflight_report,
        "collect_command": collect_command,
    }
    manifest = append_launch_record(
        manifest,
        backend=args.backend,
        run_id=args.run_id,
        tmux_commands=tmux_commands,
        config={
            "experiment_parameters": experiment_parameters,
            "preflight_report": preflight_report,
        },
    )
    write_manifest(base_dir / "manifest.json", manifest)
    created_session = ensure_tmux_session(args.tmux_session)
    if created_session:
        print(f"created_tmux_session={args.tmux_session}", flush=True)
    for command in tmux_commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
