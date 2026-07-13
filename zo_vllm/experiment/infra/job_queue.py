"""Filesystem-backed job queue helpers for experiment launchers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from zo_vllm.utils.io import write_json


def write_pending_jobs(base_dir: Path, specs: list[dict]) -> None:
    pending_dir = base_dir / "queue" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(specs):
        filename = f"{index:05d}__{spec['job_id']}.json"
        write_json(pending_dir / filename, spec)


def claim_next_job(run_dir: Path, worker_id: str) -> Path | None:
    pending_dir = run_dir / "queue" / "pending"
    running_dir = run_dir / "queue" / "running"
    running_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(pending_dir.glob("*.json")):
        target = running_dir / f"{path.stem}__{worker_id}.json"
        try:
            path.rename(target)
            return target
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return None


def move_job_file(path: Path, run_dir: Path, state: str) -> Path:
    target_dir = run_dir / "queue" / state
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    path.rename(target)
    return target


def gpu_monitor_path(run_dir: Path, gpu: str, spec: dict) -> Path:
    return (
        run_dir
        / "jobs"
        / str(spec["job_id"])
        / "logs"
        / f"gpu_util_{spec['job_id']}_gpu{gpu}.csv"
    )


def start_gpu_monitor(path: Path, interval_s: float = 1.0) -> subprocess.Popen | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        "-lms",
        str(int(interval_s * 1000)),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc._gpu_monitor_handle = handle
        return proc
    except FileNotFoundError:
        handle.close()
        return None


def stop_gpu_monitor(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    handle = getattr(proc, "_gpu_monitor_handle", None)
    if handle is not None:
        handle.close()
