"""Run queued experiment jobs on a single GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import time

from zo_vllm.experiment.infra.job_queue import (
    claim_next_job,
    gpu_monitor_path,
    move_job_file,
    start_gpu_monitor,
    stop_gpu_monitor,
)
from zo_vllm.experiment.infra.launch import shell_join
from zo_vllm.experiment.infra.paths import project_root, resolve_path
from zo_vllm.experiment.infra.backend_jobs import build_backend_job_command
from zo_vllm.utils.io import load_json, write_json

PROJECT_ROOT = project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    print(
        f"[worker] start worker_id={args.worker_id} gpu={args.gpu} run_dir={run_dir}",
        flush=True,
    )
    completed = 0
    failed = 0
    while True:
        job_path = claim_next_job(run_dir, args.worker_id)
        if job_path is None:
            print(
                f"[worker] queue empty worker_id={args.worker_id} "
                f"completed={completed} failed={failed}",
                flush=True,
            )
            break
        spec = load_json(job_path)
        spec["claimed_by"] = args.worker_id
        spec["claimed_gpu"] = args.gpu
        spec["claimed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(job_path, spec)
        cmd = build_backend_job_command(run_dir, args.gpu, spec)
        monitor_path = gpu_monitor_path(run_dir, args.gpu, spec)
        spec["gpu_monitor_csv"] = str(monitor_path)
        write_json(job_path, spec)
        print("[worker] running " + shell_join(cmd), flush=True)
        print(f"[worker] gpu_monitor_csv={monitor_path}", flush=True)
        env = dict(os.environ)
        env["WANDB_MODE"] = str(spec.get("wandb_mode", env.get("WANDB_MODE", "online")))
        if "task_shuffle_impl" in spec:
            env["ZO_TASK_SHUFFLE_IMPL"] = str(spec["task_shuffle_impl"])
        monitor_proc = start_gpu_monitor(monitor_path)
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
            returncode = proc.returncode
        finally:
            stop_gpu_monitor(monitor_proc)
        spec["exit_code"] = returncode
        spec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(job_path, spec)
        if returncode == 0:
            completed += 1
            move_job_file(job_path, run_dir, "done")
        else:
            failed += 1
            move_job_file(job_path, run_dir, "failed")
            print(
                f"[worker] failed job_id={spec['job_id']} exit_code={returncode}",
                flush=True,
            )
            if args.stop_on_failure:
                raise SystemExit(returncode)


if __name__ == "__main__":
    main()
