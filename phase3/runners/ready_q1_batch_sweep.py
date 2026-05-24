import argparse
import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str], parse_json: bool = False, quiet: bool = True) -> dict | None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout and not quiet:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr and not quiet:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
        raise SystemExit(f"command failed: {' '.join(command)}")
    if not parse_json:
        return None
    return json.loads(completed.stdout)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local readiness checks for a Phase 3 q=1 batch sweep."
    )
    parser.add_argument("--tmux-session", default="zo-vllm")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lozo-gpu", default=os.environ.get("PHASE3_LOZO_GPU"))
    parser.add_argument("--vllm-gpu", default=os.environ.get("PHASE3_VLLM_GPU"))
    parser.add_argument("--batch-sizes", default="16,32,64,128")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--detailed-batches", default="16,64,128")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--allow-non-numeric-gpu",
        action="store_true",
        help="Allow CUDA UUID/MIG GPU identifiers in preflight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.lozo_gpu:
        raise SystemExit("set --lozo-gpu or PHASE3_LOZO_GPU")
    if not args.vllm_gpu:
        raise SystemExit("set --vllm-gpu or PHASE3_VLLM_GPU")

    py_files = sorted(str(path) for path in (PROJECT_ROOT / "phase3" / "runners").glob("*.py"))
    checks = []
    run_command([".venv/bin/python", "-m", "py_compile", *py_files])
    checks.append("py_compile")
    run_command([".venv/bin/python", "-u", "phase3/runners/selftest_q1_tools.py"])
    checks.append("selftest_q1_tools")
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
        args.batch_sizes,
        "--num-samples",
        str(args.num_samples),
    ]
    if args.allow_non_numeric_gpu:
        preflight_command.append("--allow-non-numeric-gpu")
    preflight_report = run_command(
        preflight_command,
        parse_json=True,
    )
    checks.append("preflight_q1")
    launch_command = [
        ".venv/bin/python",
        "-u",
        "phase3/runners/launch_q1_batch_sweep.py",
        "--tmux-session",
        args.tmux_session,
        "--run-id",
        args.run_id,
        "--lozo-gpu",
        args.lozo_gpu,
        "--vllm-gpu",
        args.vllm_gpu,
        "--batch-sizes",
        args.batch_sizes,
        "--detailed-batches",
        args.detailed_batches,
        "--steps",
        str(args.steps),
        "--num-samples",
        str(args.num_samples),
        "--dry-run",
    ]
    if args.allow_non_numeric_gpu:
        launch_command.append("--allow-non-numeric-gpu")
    run_command(launch_command)
    checks.append("launch_q1_batch_sweep_dry_run")
    run_command(
        [
            ".venv/bin/python",
            "-u",
            "phase3/runners/validate_q1_speed_suite.py",
            "phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun",
            "--min-vllm-speedup",
            "1.2",
            "--output",
            "/tmp/phase3_ready_q1_old_validation.json",
        ]
    )
    checks.append("validate_old_q1_speed_suite")

    report = {
        "ok": True,
        "run_id": args.run_id,
        "tmux_session": args.tmux_session,
        "lozo_gpu": args.lozo_gpu,
        "vllm_gpu": args.vllm_gpu,
        "allow_non_numeric_gpu": args.allow_non_numeric_gpu,
        "batch_sizes": args.batch_sizes,
        "detailed_batches": args.detailed_batches,
        "num_samples": args.num_samples,
        "steps": args.steps,
        "checks": checks,
        "preflight_warnings": preflight_report.get("warnings", []) if preflight_report else [],
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("ready_ok=true", flush=True)


if __name__ == "__main__":
    main()
