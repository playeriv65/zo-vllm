import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.launch import (
    ensure_run_dir_available,
    ensure_tmux_session,
    env_prefix,
    shell_join,
    tmux_new_window_command,
)
DEFAULT_TMUX_SESSION = "zo-vllm"


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run_preflight(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = {
            "ok": False,
            "errors": ["preflight did not emit valid JSON"],
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    if completed.returncode != 0 or not report.get("ok", False):
        raise SystemExit("preflight failed")
    return report


def build_lozo_command(args, base_dir: Path) -> str:
    output_dir = base_dir / "lozo_minimal"
    log_file = (
        base_dir
        / "logs"
        / f"lozo_minimal_steps{args.steps}_b{args.batch_size}_{args.timestamp}.log"
    )
    cmd = [
        "third_party/LOZO/large_models/.venv/bin/python",
        "-u",
        "phase3/runners/train_lozo_baseline_perf.py",
        "--profile-mode",
        "minimal",
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--num-samples",
        str(args.num_samples),
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
        "--accuracy-eval-mode",
        "skip",
        "--seed",
        str(args.seed),
        "--zo-random-device",
        "cuda",
        "--train-scope",
        "lora_only",
        "--progress-interval",
        str(args.progress_interval),
        "--no-wandb",
        "--output-dir",
        str(output_dir),
    ]
    body = (
        f"set -o pipefail; cd {shlex.quote(str(PROJECT_ROOT))} && "
        f"{env_prefix({'CUDA_VISIBLE_DEVICES': args.lozo_gpu})} {shell_join(cmd)} "
        f"2>&1 | tee {shlex.quote(str(log_file))}"
    )
    return body


def build_vllm_command(args, base_dir: Path, profile_mode: str) -> str:
    output_dir = base_dir / f"vllm_{profile_mode}_eager{args.enforce_eager}"
    log_file = (
        base_dir
        / "logs"
        / f"vllm_{profile_mode}_eager{args.enforce_eager}_steps{args.steps}_b{args.batch_size}_{args.timestamp}.log"
    )
    cmd = [
        ".venv/bin/python",
        "-u",
        "phase3/runners/train_vllm_perf.py",
        "--profile-mode",
        profile_mode,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--num-samples",
        str(args.num_samples),
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
        "--zo-random-device",
        "cuda",
        "--train-scope",
        "lora_only",
        "--batch-invariant",
        "0",
        "--enforce-eager",
        str(args.enforce_eager),
        "--lora-residency",
        "gpu",
        "--lora-injection",
        "direct",
        "--weight-update",
        "direct",
        "--weight-update-precision",
        "param",
        "--direct-update-mode",
        args.direct_update_mode,
        "--base-eval-mode",
        "skip",
        "--accuracy-eval-mode",
        "skip",
        "--progress-interval",
        str(args.progress_interval),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--output-dir",
        str(output_dir),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": args.vllm_gpu,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    }
    body = (
        f"set -o pipefail; cd {shlex.quote(str(PROJECT_ROOT))} && "
        f"{env_prefix(env)} {shell_join(cmd)} "
        f"2>&1 | tee {shlex.quote(str(log_file))}"
    )
    return body


def build_vllm_suite_command(args, base_dir: Path) -> str:
    commands = [build_vllm_command(args, base_dir, "minimal")]
    if args.include_detailed:
        commands.append(build_vllm_command(args, base_dir, "detailed"))
    return " && ".join(f"({command})" for command in commands)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch the Phase 3 q=1 speed suite in tmux windows."
    )
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--lozo-gpu", default=os.environ.get("PHASE3_LOZO_GPU"))
    parser.add_argument("--vllm-gpu", default=os.environ.get("PHASE3_VLLM_GPU"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--step-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--direct-update-mode", choices=["immediate", "accumulate"], default="accumulate")
    parser.add_argument(
        "--allow-non-numeric-gpu",
        action="store_true",
        help="Allow CUDA UUID/MIG GPU identifiers in preflight.",
    )
    parser.add_argument(
        "--include-detailed",
        action="store_true",
        help="Run vLLM detailed profiling after vLLM minimal finishes.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.lozo_gpu:
        raise SystemExit("set --lozo-gpu or PHASE3_LOZO_GPU")
    if not args.vllm_gpu:
        raise SystemExit("set --vllm-gpu or PHASE3_VLLM_GPU")
    if args.num_samples < args.batch_size:
        raise SystemExit("--num-samples must be greater than or equal to --batch-size")
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_id is None:
        args.run_id = f"phase3_q1_speed_b{args.batch_size}_s{args.steps}_{args.timestamp}"
    return args


def main() -> None:
    args = parse_args()
    base_dir = PROJECT_ROOT / "phase3" / "results" / args.run_id
    if not args.dry_run:
        ensure_run_dir_available(base_dir)

    jobs = [
        ("zo-vllm-p3-q1-lozo", build_lozo_command(args, base_dir)),
        ("zo-vllm-p3-q1-vllm", build_vllm_suite_command(args, base_dir)),
    ]
    experiment_parameters = {
        "q": 1,
        "steps": args.steps,
        "batch_size": args.batch_size,
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
        "vllm_direct_update_mode": args.direct_update_mode,
        "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
        "include_detailed": args.include_detailed,
        "wandb": "disabled",
    }
    collect_command = [
        ".venv/bin/python",
        "-u",
        "phase3/runners/collect_q1_speed_suite.py",
        str(base_dir.relative_to(PROJECT_ROOT)),
        "--output",
        str((base_dir / "summary.md").relative_to(PROJECT_ROOT)),
    ]
    validate_command = None
    if args.include_detailed:
        validate_command = [
            ".venv/bin/python",
            "-u",
            "phase3/runners/validate_q1_speed_suite.py",
            str(base_dir.relative_to(PROJECT_ROOT)),
            "--min-vllm-speedup",
            "1.2",
            "--output",
            str((base_dir / "validation.json").relative_to(PROJECT_ROOT)),
        ]
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
        "--batch-size",
        str(args.batch_size),
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
    if validate_command:
        print("validate_command=" + shlex.join(validate_command), flush=True)
    print("preflight_command=" + shlex.join(preflight_command), flush=True)
    if not args.dry_run:
        preflight_report = None
        if not args.skip_preflight:
            preflight_report = run_preflight(preflight_command)
        ensure_run_dir_available(base_dir)
        (base_dir / "logs").mkdir(parents=True, exist_ok=True)
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
