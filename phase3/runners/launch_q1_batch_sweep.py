import argparse
import json
import os
import shlex
import subprocess
import sys
from copy import copy
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phase3.runners.launch_q1_speed_suite import (
    DEFAULT_TMUX_SESSION,
    build_lozo_command,
    build_vllm_command,
    ensure_tmux_session,
    ensure_run_dir_available,
    run_preflight,
    shell_join,
    tmux_new_window_command,
    write_manifest,
)


def parse_batch_sizes(value: str) -> list[int]:
    sizes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            raise argparse.ArgumentTypeError("batch sizes must be positive")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return sizes


def parse_optional_batch_sizes(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return parse_batch_sizes(value)


def with_batch_size(args, batch_size: int):
    batch_args = copy(args)
    batch_args.batch_size = batch_size
    return batch_args


def batch_dir(base_dir: Path, batch_size: int) -> Path:
    return base_dir / f"batch_b{batch_size}"


def sequential_body(commands: list[str]) -> str:
    return " && ".join(f"({command})" for command in commands)


def comma_join_ints(values) -> str:
    return ",".join(str(value) for value in values)


def build_lozo_sweep_body(args, base_dir: Path) -> str:
    commands = []
    for batch_size in args.batch_sizes:
        batch_args = with_batch_size(args, batch_size)
        commands.append(build_lozo_command(batch_args, batch_dir(base_dir, batch_size)))
    return sequential_body(commands)


def build_vllm_sweep_body(args, base_dir: Path) -> str:
    commands = []
    for batch_size in args.batch_sizes:
        batch_args = with_batch_size(args, batch_size)
        commands.append(build_vllm_command(batch_args, batch_dir(base_dir, batch_size), "minimal"))
        if batch_size in args.detailed_batches:
            commands.append(
                build_vllm_command(batch_args, batch_dir(base_dir, batch_size), "detailed")
            )
    return sequential_body(commands)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Phase 3 q=1 batch-size speed sweep in tmux windows."
    )
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--lozo-gpu", default=os.environ.get("PHASE3_LOZO_GPU"))
    parser.add_argument("--vllm-gpu", default=os.environ.get("PHASE3_VLLM_GPU"))
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=parse_batch_sizes("16,32,64,128"),
    )
    parser.add_argument(
        "--detailed-batches",
        type=parse_optional_batch_sizes,
        default=[],
        help="Comma-separated batch sizes that should also run vLLM detailed profiling.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Training subset size. Defaults to max(1000, max(batch_sizes)).",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--step-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument(
        "--allow-non-numeric-gpu",
        action="store_true",
        help="Allow CUDA UUID/MIG GPU identifiers in preflight.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.lozo_gpu:
        raise SystemExit("set --lozo-gpu or PHASE3_LOZO_GPU")
    if not args.vllm_gpu:
        raise SystemExit("set --vllm-gpu or PHASE3_VLLM_GPU")

    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_id is None:
        batch_text = "-".join(str(size) for size in args.batch_sizes)
        args.run_id = f"phase3_q1_batch_sweep_b{batch_text}_s{args.steps}_{args.timestamp}"
    if args.num_samples is None:
        args.num_samples = max(1000, max(args.batch_sizes))
    if args.num_samples < max(args.batch_sizes):
        raise SystemExit("--num-samples must cover the largest --batch-sizes value")
    args.detailed_batches = set(args.detailed_batches)
    return args


def main() -> None:
    args = parse_args()
    base_dir = PROJECT_ROOT / "phase3" / "results" / args.run_id
    if not args.dry_run:
        ensure_run_dir_available(base_dir)

    jobs = [
        ("zo-vllm-p3-q1-batch-lozo", build_lozo_sweep_body(args, base_dir)),
        ("zo-vllm-p3-q1-batch-vllm", build_vllm_sweep_body(args, base_dir)),
    ]
    experiment_parameters = {
        "q": 1,
        "batch_sizes": args.batch_sizes,
        "detailed_batches": sorted(args.detailed_batches),
        "steps": args.steps,
        "num_samples": args.num_samples,
        "rank": args.rank,
        "lr": args.lr,
        "eps": args.eps,
        "step_interval": args.step_interval,
        "eval_interval": args.eval_interval,
        "seed": args.seed,
        "zo_random_device": "cuda",
        "train_scope": "lora_only",
        "lozo_gpu": args.lozo_gpu,
        "vllm_gpu": args.vllm_gpu,
        "allow_non_numeric_gpu": args.allow_non_numeric_gpu,
        "vllm_batch_invariant": 0,
        "vllm_enforce_eager": int(args.enforce_eager),
        "vllm_lora_residency": "gpu",
        "vllm_lora_injection": "direct",
        "vllm_weight_update": "direct",
        "vllm_weight_update_precision": "param",
        "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
        "wandb": "disabled",
    }
    collect_command = [
        ".venv/bin/python",
        "-u",
        "phase3/runners/collect_q1_batch_sweep.py",
        str(base_dir.relative_to(PROJECT_ROOT)),
        "--output",
        str((base_dir / "summary.md").relative_to(PROJECT_ROOT)),
    ]
    validate_command = [
        ".venv/bin/python",
        "-u",
        "phase3/runners/validate_q1_batch_sweep.py",
        str(base_dir.relative_to(PROJECT_ROOT)),
        "--expected-batches",
        comma_join_ints(args.batch_sizes),
        "--output",
        str((base_dir / "validation.json").relative_to(PROJECT_ROOT)),
    ]
    if args.detailed_batches:
        validate_command.extend([
            "--require-detailed-batches",
            comma_join_ints(sorted(args.detailed_batches)),
        ])
    tmux_commands = [
        tmux_new_window_command(args.tmux_session, window, body) for window, body in jobs
    ]
    preflight_command = [
        ".venv/bin/python",
        "-u",
        "phase3/runners/preflight_q1.py",
        "--tmux-session",
        args.tmux_session,
        "--run-id",
        args.run_id,
        "--lozo-gpu",
        args.lozo_gpu,
        "--vllm-gpu",
        args.vllm_gpu,
        "--batch-sizes",
        comma_join_ints(args.batch_sizes),
        "--num-samples",
        str(args.num_samples),
    ]
    if args.allow_non_numeric_gpu:
        preflight_command.append("--allow-non-numeric-gpu")

    print(f"run_dir={base_dir}", flush=True)
    print(
        "experiment_parameters="
        + json.dumps(experiment_parameters, sort_keys=True),
        flush=True,
    )
    print("collect_command=" + shlex.join(collect_command), flush=True)
    print("validate_command=" + shlex.join(validate_command), flush=True)
    print("preflight_command=" + shlex.join(preflight_command), flush=True)
    if not args.dry_run:
        preflight_report = None
        if not args.skip_preflight:
            preflight_report = run_preflight(preflight_command)
        ensure_run_dir_available(base_dir)
        for batch_size in args.batch_sizes:
            (batch_dir(base_dir, batch_size) / "logs").mkdir(parents=True, exist_ok=True)
        write_manifest(
            base_dir / "manifest.json",
            {
                "run_id": args.run_id,
                "run_dir": str(base_dir.relative_to(PROJECT_ROOT)),
                "timestamp": args.timestamp,
                "tmux_session": args.tmux_session,
                "tmux_session_created_by_launcher": None,
                "experiment_parameters": experiment_parameters,
                "preflight_command": preflight_command,
                "preflight_report": preflight_report,
                "collect_command": collect_command,
                "validate_command": validate_command,
                "tmux_commands": [shell_join(command) for command in tmux_commands],
            },
        )
        print(f"manifest={base_dir / 'manifest.json'}", flush=True)
        created_session = ensure_tmux_session(args.tmux_session)
        manifest_path = base_dir / "manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest["tmux_session_created_by_launcher"] = created_session
        write_manifest(manifest_path, manifest)
        if created_session:
            print(f"created_tmux_session={args.tmux_session}", flush=True)
    for command in tmux_commands:
        print(shell_join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
