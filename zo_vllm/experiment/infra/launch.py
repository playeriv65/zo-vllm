import argparse
import shlex
import subprocess
from pathlib import Path


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
    return ["tmux", "new-window", "-a", "-t", f"{session}:", "-n", window, body]


def write_job_script(base_dir: Path, window: str, body: str) -> Path:
    scripts_dir = base_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{window}.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{body}\n"
    )
    script_path.chmod(0o755)
    return script_path


def run_preflight(
    gpus: list[str],
    allow_non_numeric_gpu: bool,
    *,
    cwd: Path,
) -> dict:
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
            cwd=cwd,
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


def monitoring_prefix(gpu_csv: Path, *, interval_s: int = 1) -> str:
    return (
        f"(while true; do nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used "
        f"--format=csv,noheader,nounits >> {shlex.quote(str(gpu_csv))}; "
        f"sleep {int(interval_s)}; done) & mon=$!; "
    )


def monitored_body(
    env: dict[str, str],
    cmd: list[str],
    log_file: Path,
    gpu_csv: Path,
    *,
    project_root: Path,
    monitor_interval_s: int = 1,
) -> str:
    return (
        f"set -o pipefail; cd {shlex.quote(str(project_root))}; "
        f"{monitoring_prefix(gpu_csv, interval_s=monitor_interval_s)}"
        f"{env_prefix(env)} {shell_join(cmd)} 2>&1 | tee {shlex.quote(str(log_file))}; "
        "status=${PIPESTATUS[0]}; kill $mon 2>/dev/null || true; exit $status"
    )


def sequential_body(commands: list[str]) -> str:
    return " && ".join(f"({command})" for command in commands)
