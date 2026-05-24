import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TMUX_SESSION = "zo-vllm"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def env_prefix(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one item is required")
    return items


def parse_int_csv(value: str) -> list[int]:
    values = []
    for item in parse_csv(value):
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        values.append(parsed)
    return values


def safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def model_batch_dir(base_dir: Path, model: str, batch_size: int) -> Path:
    return base_dir / f"model_{safe_model_name(model)}" / f"batch_b{batch_size}"


def ensure_tmux_session(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return False
    subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True)
    return True


def ensure_run_dir_available(base_dir: Path) -> None:
    if base_dir.exists():
        raise SystemExit(f"run directory already exists: {base_dir}")


def tmux_new_window_command(session: str, window: str, body: str) -> list[str]:
    return ["tmux", "new-window", "-t", session, "-n", window, body]


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run_preflight(gpus: list[str], allow_non_numeric_gpu: bool) -> dict:
    errors = []
    if not allow_non_numeric_gpu:
        for gpu in gpus:
            if not gpu.isdigit():
                errors.append(f"non-numeric GPU id requires --allow-non-numeric-gpu: {gpu}")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        gpu_rows = result.stdout.strip().splitlines()
    except Exception as exc:
        return {"ok": False, "errors": [f"nvidia-smi failed: {exc}"], "gpu_rows": []}

    numeric = {row.split(",")[0].strip(): row for row in gpu_rows if row.strip()}
    for gpu in gpus:
        if gpu.isdigit() and gpu not in numeric:
            errors.append(f"GPU {gpu} not found in nvidia-smi output")
    return {"ok": not errors, "errors": errors, "gpu_rows": gpu_rows}


def monitoring_prefix(gpu_csv: Path) -> str:
    return (
        f"(while true; do nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used "
        f"--format=csv,noheader,nounits >> {shlex.quote(str(gpu_csv))}; "
        "sleep 1; done) & mon=$!; "
    )


def monitored_body(env: dict[str, str], cmd: list[str], log_file: Path, gpu_csv: Path) -> str:
    return (
        f"set -o pipefail; cd {shlex.quote(str(PROJECT_ROOT))}; "
        f"{monitoring_prefix(gpu_csv)}"
        f"{env_prefix(env)} {shell_join(cmd)} 2>&1 | tee {shlex.quote(str(log_file))}; "
        "status=${PIPESTATUS[0]}; kill $mon 2>/dev/null || true; exit $status"
    )


def build_lozo_command(args, model: str, batch_size: int, base_dir: Path) -> str:
    run_dir = model_batch_dir(base_dir, model, batch_size)
    output_dir = run_dir / "lozo_baseline"
    log_file = (
        run_dir
        / "logs"
        / f"lozo_model-{safe_model_name(model)}_b{batch_size}_s{args.steps}_{args.timestamp}.log"
    )
    gpu_csv = (
        run_dir
        / "logs"
        / f"gpu_util_lozo_model-{safe_model_name(model)}_b{batch_size}_s{args.steps}_{args.timestamp}.csv"
    )
    cmd = [
        "third_party/LOZO/large_models/.venv/bin/python",
        "-u",
        "phase3/runners/train_lozo_baseline_perf.py",
        "--model-name",
        model,
        "--profile-mode",
        "minimal",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(batch_size),
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
        "--progress-interval",
        str(args.progress_interval),
        "--no-wandb",
        "--output-dir",
        str(output_dir),
    ]
    if args.lozo_torch_compile:
        cmd.extend([
            "--torch-compile",
            "--torch-compile-mode",
            args.lozo_torch_compile_mode,
        ])
    return monitored_body(
        {"CUDA_VISIBLE_DEVICES": args.lozo_gpu},
        cmd,
        log_file,
        gpu_csv,
    )


def build_vllm_command(args, model: str, batch_size: int, base_dir: Path) -> str:
    run_dir = model_batch_dir(base_dir, model, batch_size)
    output_dir = run_dir / "vllm_optimized"
    log_file = (
        run_dir
        / "logs"
        / f"vllm_opt_model-{safe_model_name(model)}_b{batch_size}_s{args.steps}_{args.timestamp}.log"
    )
    gpu_csv = (
        run_dir
        / "logs"
        / f"gpu_util_vllm_model-{safe_model_name(model)}_b{batch_size}_s{args.steps}_{args.timestamp}.csv"
    )
    cmd = [
        ".venv/bin/python",
        "-u",
        "phase3/runners/train_vllm_perf.py",
        "--model-name",
        model,
        "--profile-mode",
        "detailed",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(batch_size),
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
        "--sync-weight-update",
        "0",
        "--scoring-backend",
        "direct_worker",
        "--direct-worker-max-logits-tokens",
        str(args.direct_worker_max_logits_tokens),
        "--direct-worker-loss-impl",
        "logprobs",
        "--direct-lora-from-directions",
        "1",
        "--slot-pipeline",
        "0",
        "--base-eval-mode",
        "skip",
        "--progress-interval",
        str(args.progress_interval),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--output-dir",
        str(output_dir),
    ]
    return monitored_body(
        {
            "CUDA_VISIBLE_DEVICES": args.vllm_gpu,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_BATCH_INVARIANT": "0",
            "WANDB_MODE": "offline",
        },
        cmd,
        log_file,
        gpu_csv,
    )


def sequential_body(commands: list[str]) -> str:
    return " && ".join(f"({command})" for command in commands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch OPT-family and batch-size scaling experiments."
    )
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--lozo-gpu", default=os.environ.get("PHASE3_LOZO_GPU"))
    parser.add_argument("--vllm-gpu", default=os.environ.get("PHASE3_VLLM_GPU"))
    parser.add_argument(
        "--models",
        type=parse_csv,
        default=parse_csv("facebook/opt-1.3b,facebook/opt-2.7b"),
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_csv,
        default=parse_int_csv("16,32,64,128"),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=5)
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
    parser.add_argument("--direct-worker-max-logits-tokens", type=int, default=8192)
    parser.add_argument("--lozo-torch-compile", action="store_true")
    parser.add_argument("--lozo-torch-compile-mode", default="default")
    parser.add_argument("--backend", choices=["both", "lozo", "vllm"], default="both")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--allow-non-numeric-gpu", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.lozo_gpu:
        raise SystemExit("set --lozo-gpu or PHASE3_LOZO_GPU")
    if not args.vllm_gpu:
        raise SystemExit("set --vllm-gpu or PHASE3_VLLM_GPU")
    if args.num_samples < max(args.batch_sizes):
        raise SystemExit("--num-samples must cover the largest --batch-sizes value")
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_id is None:
        model_text = "-".join(safe_model_name(model).removeprefix("facebook__opt-") for model in args.models)
        batch_text = "-".join(str(batch) for batch in args.batch_sizes)
        args.run_id = f"phase3_scaling_models_{model_text}_b{batch_text}_s{args.steps}_{args.timestamp}"
    return args


def main() -> None:
    args = parse_args()
    base_dir = PROJECT_ROOT / "phase3" / "results" / args.run_id
    if not args.dry_run and not args.resume_existing:
        ensure_run_dir_available(base_dir)

    lozo_commands = []
    vllm_commands = []
    for model in args.models:
        for batch_size in args.batch_sizes:
            lozo_commands.append(build_lozo_command(args, model, batch_size, base_dir))
            vllm_commands.append(build_vllm_command(args, model, batch_size, base_dir))

    jobs = []
    if args.backend in {"both", "lozo"}:
        jobs.append(("zo-vllm-p3-scale-lozo", sequential_body(lozo_commands)))
    if args.backend in {"both", "vllm"}:
        jobs.append(("zo-vllm-p3-scale-vllm", sequential_body(vllm_commands)))
    collect_command = [
        ".venv/bin/python",
        "-u",
        "phase3/runners/collect_scaling_sweep.py",
        str(base_dir.relative_to(PROJECT_ROOT)),
        "--output",
        str((base_dir / "summary.md").relative_to(PROJECT_ROOT)),
    ]
    experiment_parameters = {
        "models": args.models,
        "batch_sizes": args.batch_sizes,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "num_samples": args.num_samples,
        "rank": args.rank,
        "lr": args.lr,
        "eps": args.eps,
        "step_interval": args.step_interval,
        "eval_interval": args.eval_interval,
        "seed": args.seed,
        "lozo_gpu": args.lozo_gpu,
        "vllm_gpu": args.vllm_gpu,
        "vllm_path": "direct_worker+direct_lora_from_directions+direct_weight_update",
        "vllm_enforce_eager": int(args.enforce_eager),
        "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
        "lozo_torch_compile": bool(args.lozo_torch_compile),
        "lozo_torch_compile_mode": args.lozo_torch_compile_mode,
        "backend": args.backend,
        "resume_existing": args.resume_existing,
    }
    tmux_commands = [
        tmux_new_window_command(args.tmux_session, window, body) for window, body in jobs
    ]

    print(f"run_dir={base_dir}", flush=True)
    print(
        "experiment_parameters="
        + json.dumps(experiment_parameters, sort_keys=True),
        flush=True,
    )
    print("collect_command=" + shell_join(collect_command), flush=True)
    if not args.dry_run:
        preflight_report = None
        if not args.skip_preflight:
            preflight_report = run_preflight(
                [args.lozo_gpu, args.vllm_gpu],
                args.allow_non_numeric_gpu,
            )
            print("preflight_report=" + json.dumps(preflight_report, sort_keys=True), flush=True)
            if not preflight_report.get("ok", False):
                raise SystemExit("preflight failed")
        if not args.resume_existing:
            ensure_run_dir_available(base_dir)
            base_dir.mkdir(parents=True, exist_ok=False)
        else:
            base_dir.mkdir(parents=True, exist_ok=True)
        for model in args.models:
            for batch_size in args.batch_sizes:
                (model_batch_dir(base_dir, model, batch_size) / "logs").mkdir(
                    parents=True,
                    exist_ok=True,
                )
        manifest_path = base_dir / "manifest.json"
        manifest = {}
        if args.resume_existing and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        launch_record = {
            "timestamp": args.timestamp,
            "backend": args.backend,
            "preflight_report": preflight_report,
            "tmux_commands": [shell_join(command) for command in tmux_commands],
        }
        manifest.update({
            "run_id": args.run_id,
            "run_dir": str(base_dir.relative_to(PROJECT_ROOT)),
            "timestamp": args.timestamp,
            "tmux_session": args.tmux_session,
            "tmux_session_created_by_launcher": None,
            "experiment_parameters": experiment_parameters,
            "preflight_report": preflight_report,
            "collect_command": collect_command,
        })
        manifest.setdefault("launch_records", []).append(launch_record)
        write_manifest(manifest_path, manifest)
        created_session = ensure_tmux_session(args.tmux_session)
        manifest = json.loads(manifest_path.read_text())
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
