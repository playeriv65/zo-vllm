import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TMUX_SESSION = "zo-vllm"


def parse_int_list(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    items = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        items.append(int(item))
    return items


def run_capture(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def check_python_compile(errors: list[str]) -> None:
    returncode, stdout, stderr = run_capture(
        [".venv/bin/python", "-m", "py_compile", *sorted(str(path) for path in (PROJECT_ROOT / "phase3" / "runners").glob("*.py"))]
    )
    if returncode != 0:
        errors.append(f"py_compile failed: {stderr or stdout}")


def check_tmux(session: str, warnings: list[str]) -> None:
    returncode, stdout, stderr = run_capture(["tmux", "has-session", "-t", session])
    if returncode != 0:
        warnings.append(f"tmux session {session!r} does not exist yet: {stderr or stdout}")


def split_cuda_visible_devices(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def check_gpu_env(lozo_gpu: str | None, vllm_gpu: str | None, errors: list[str]) -> None:
    if not lozo_gpu:
        errors.append("missing PHASE3_LOZO_GPU or --lozo-gpu")
    if not vllm_gpu:
        errors.append("missing PHASE3_VLLM_GPU or --vllm-gpu")


def check_batch_shape(batch_sizes: list[int], num_samples: int | None, errors: list[str]) -> int:
    if not batch_sizes:
        errors.append("at least one batch size is required")
        return num_samples or 0
    for batch_size in batch_sizes:
        if batch_size <= 0:
            errors.append(f"invalid batch size: {batch_size}")
    resolved_num_samples = num_samples if num_samples is not None else max(1000, max(batch_sizes))
    if resolved_num_samples < max(batch_sizes):
        errors.append(
            f"num_samples={resolved_num_samples} is smaller than max batch {max(batch_sizes)}"
        )
    return resolved_num_samples


def check_run_dir(run_id: str | None, errors: list[str]) -> str | None:
    if not run_id:
        return None
    run_dir = PROJECT_ROOT / "phase3" / "results" / run_id
    if run_dir.exists():
        errors.append(f"run directory already exists: {run_dir}")
    return str(run_dir)


def check_nvidia_smi(warnings: list[str]) -> list[dict]:
    returncode, stdout, stderr = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if returncode != 0:
        warnings.append(f"nvidia-smi unavailable: {stderr or stdout}")
        return []
    rows = []
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        rows.append({
            "index": fields[0],
            "memory_used_mib": int(fields[1]),
            "memory_total_mib": int(fields[2]),
            "utilization_gpu_percent": int(fields[3]),
        })
    return rows


def check_selected_gpus(
    lozo_gpu: str | None,
    vllm_gpu: str | None,
    gpu_status: list[dict],
    errors: list[str],
    warnings: list[str],
    allow_non_numeric_gpu: bool,
) -> None:
    status_by_index = {row["index"]: row for row in gpu_status}
    for label, visible_devices in [
        ("LOZO", split_cuda_visible_devices(lozo_gpu)),
        ("vLLM", split_cuda_visible_devices(vllm_gpu)),
    ]:
        for device in visible_devices:
            if not device.isdigit():
                message = (
                    f"{label} CUDA_VISIBLE_DEVICES value {device!r} is not a numeric GPU index"
                )
                if allow_non_numeric_gpu:
                    warnings.append(message)
                else:
                    errors.append(
                        message + "; pass --allow-non-numeric-gpu only for UUID/MIG values"
                    )
                continue
            if device not in status_by_index:
                errors.append(f"{label} GPU index {device} is not present in nvidia-smi")
                continue
            status = status_by_index[device]
            if status["memory_used_mib"] > 0 or status["utilization_gpu_percent"] > 0:
                warnings.append(
                    f"{label} GPU {device} is not idle: "
                    f"memory_used_mib={status['memory_used_mib']}, "
                    f"utilization_gpu_percent={status['utilization_gpu_percent']}"
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Preflight checks for Phase 3 q=1 runs.")
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--lozo-gpu", default=os.environ.get("PHASE3_LOZO_GPU"))
    parser.add_argument("--vllm-gpu", default=os.environ.get("PHASE3_VLLM_GPU"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-sizes", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--allow-non-numeric-gpu",
        action="store_true",
        help="Allow CUDA_VISIBLE_DEVICES values that are not numeric nvidia-smi indices.",
    )
    args = parser.parse_args()
    if args.batch_sizes:
        args.batch_sizes = parse_int_list(args.batch_sizes)
    elif args.batch_size:
        args.batch_sizes = [args.batch_size]
    else:
        args.batch_sizes = [16]
    return args


def main() -> None:
    args = parse_args()
    errors = []
    warnings = []
    check_python_compile(errors)
    check_tmux(args.tmux_session, warnings)
    check_gpu_env(args.lozo_gpu, args.vllm_gpu, errors)
    resolved_num_samples = check_batch_shape(args.batch_sizes, args.num_samples, errors)
    run_dir = check_run_dir(args.run_id, errors)
    gpu_status = check_nvidia_smi(warnings)
    check_selected_gpus(
        args.lozo_gpu,
        args.vllm_gpu,
        gpu_status,
        errors,
        warnings,
        args.allow_non_numeric_gpu,
    )

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "tmux_session": args.tmux_session,
        "run_id": args.run_id,
        "run_dir": run_dir,
        "lozo_gpu": args.lozo_gpu,
        "vllm_gpu": args.vllm_gpu,
        "batch_sizes": args.batch_sizes,
        "num_samples": resolved_num_samples,
        "allow_non_numeric_gpu": args.allow_non_numeric_gpu,
        "gpu_status": gpu_status,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
