import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.manifest import append_launch_record, load_manifest, write_manifest
from zo_vllm.experiment.naming import timestamp_now
from zo_vllm.experiment.paths import project_root, resolve_path
from zo_vllm.experiment.run_state import read_run_state


PROJECT_ROOT = project_root()
DEFAULT_TMUX_SESSION = "zo-vllm"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_gpu_pairs(value: str) -> list[tuple[str, str]]:
    pairs = []
    for item in parse_csv(value):
        if ":" not in item:
            raise ValueError(f"gpu pair must use left:right syntax: {item}")
        left, right = [part.strip() for part in item.split(":", 1)]
        if not left or not right:
            raise ValueError(f"invalid gpu pair: {item}")
        pairs.append((left, right))
    return pairs


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def ensure_tmux_session(session: str) -> bool:
    has = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if has.returncode == 0:
        return False
    subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True)
    return True


def should_skip_job(run_dir: Path, job_id: str, resume_existing: bool) -> tuple[bool, bool]:
    job_dir = run_dir / "jobs" / job_id
    result_file = job_dir / "phase4_result.json"
    if not resume_existing:
        return False, False
    state = read_run_state(job_dir)
    status = state.get("status")
    if status == "completed" and result_file.exists():
        return True, False
    return False, True


def build_job_cmd(job: dict, defaults: dict, wandb_cfg: dict, run_dir: Path, gpu: str, resumed: bool) -> list[str]:
    merged = dict(defaults)
    merged.update(job)
    cmd = [
        ".venv/bin/python",
        "-u",
        "phase4/runners/run_phase4_job.py",
        "--run-dir",
        str(run_dir),
        "--job-id",
        str(merged["job_id"]),
        "--backend",
        str(merged["backend"]),
        "--gpu",
        str(gpu),
        "--wandb-project",
        str(wandb_cfg.get("project", "lozo-vllm-phase4")),
        "--wandb-entity",
        str(wandb_cfg.get("entity", "playeriv65-university-of-minnesota")),
        "--model-name",
        str(merged["model_name"]),
        "--task-name",
        str(merged.get("task_name", "SST2")),
        "--steps",
        str(merged["steps"]),
        "--warmup-steps",
        str(merged["warmup_steps"]),
        "--batch-size",
        str(merged["batch_size"]),
        "--num-samples",
        str(merged["num_samples"]),
        "--num-dev",
        str(merged.get("num_dev", 500)),
        "--eval-interval",
        str(merged["eval_interval"]),
        "--seed",
        str(merged["seed"]),
        "--train-set-seed",
        str(merged.get("train_set_seed", merged["seed"])),
        "--rank",
        str(merged["rank"]),
        "--lr",
        str(merged["lr"]),
        "--eps",
        str(merged["eps"]),
        "--step-interval",
        str(merged["step_interval"]),
        "--zo-random-device",
        str(merged.get("zo_random_device", "cuda")),
        "--train-scope",
        str(merged.get("train_scope", "lora_only")),
        "--save-interval",
        str(merged.get("save_interval", 0)),
        "--save-total-limit",
        str(merged.get("save_total_limit", 3)),
        "--eval-accuracy-samples",
        str(merged.get("eval_accuracy_samples", 512)),
    ]
    if resumed:
        cmd.append("--resume")
    return cmd


def build_tmux_commands(session: str, scheduled: list[dict]) -> list[list[str]]:
    by_gpu: dict[str, list[dict]] = {}
    for item in scheduled:
        by_gpu.setdefault(str(item["gpu"]), []).append(item)
    commands: list[list[str]] = []
    for gpu, items in sorted(by_gpu.items(), key=lambda pair: pair[0]):
        window = f"zo-vllm-phase4-gpu{gpu}"[:48]
        body_parts = [f"cd {shlex.quote(str(PROJECT_ROOT))}"]
        for item in items:
            body_parts.append(shell_join(item["command"]))
        body = " && ".join(body_parts)
        commands.append(["tmux", "new-window", "-t", session, "-n", window, body])
    return commands


def render_pair_script(items: list[dict]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
    ]
    for item in items:
        left = shell_join(item["left"]["command"])
        right = shell_join(item["right"]["command"])
        lines.extend(
            [
                f"{left} &",
                "p1=$!",
                f"{right} &",
                "p2=$!",
                "wait $p1",
                "r1=$?",
                "wait $p2",
                "r2=$?",
                "if [ $r1 -ne 0 ] || [ $r2 -ne 0 ]; then",
                "  exit 1",
                "fi",
            ]
        )
    return "\n".join(lines) + "\n"


def build_tmux_pair_commands(
    session: str,
    scheduled_pairs: list[dict],
    run_dir: Path,
    dry_run: bool,
) -> list[list[str]]:
    by_pair: dict[str, list[dict]] = {}
    for item in scheduled_pairs:
        by_pair.setdefault(str(item["gpu_pair_label"]), []).append(item)
    commands: list[list[str]] = []
    scripts_dir = run_dir / "launch_scripts"
    if not dry_run:
        scripts_dir.mkdir(parents=True, exist_ok=True)
    for pair_label, items in sorted(by_pair.items(), key=lambda pair: pair[0]):
        window = f"zo-vllm-phase4-pair{pair_label}"[:48]
        script_path = scripts_dir / f"pair_{pair_label}.sh"
        if not dry_run:
            script_path.write_text(render_pair_script(items))
            script_path.chmod(0o755)
        commands.append(["tmux", "new-window", "-t", session, "-n", window, f"bash {shlex.quote(str(script_path))}"])
    return commands


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--gpu-pairs", default="0:1,6:7")
    parser.add_argument("--paired-barrier", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    defaults = cfg.get("defaults", {})
    wandb_cfg = cfg.get("wandb", {})
    jobs = list(cfg.get("jobs", []))
    if not jobs:
        raise SystemExit("no jobs in config")
    gpus = parse_csv(args.gpus)
    if not gpus:
        raise SystemExit("no GPUs provided")

    run_id = args.run_id or f"phase4_{timestamp_now()}"
    run_dir = PROJECT_ROOT / "phase4" / "results" / run_id
    manifest_path = run_dir / "manifest.json"

    if run_dir.exists() and not args.resume_existing and not args.dry_run:
        raise SystemExit(f"run dir already exists: {run_dir}")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    scheduled = []
    scheduled_pairs = []
    if args.paired_barrier:
        gpu_pairs = parse_gpu_pairs(args.gpu_pairs)
        if not gpu_pairs:
            raise SystemExit("no GPU pairs provided")
        if len(jobs) % 2 != 0:
            raise SystemExit("paired barrier requires an even number of jobs")
        for pair_idx in range(0, len(jobs), 2):
            left_job = dict(defaults)
            left_job.update(jobs[pair_idx])
            right_job = dict(defaults)
            right_job.update(jobs[pair_idx + 1])
            gpu_left, gpu_right = gpu_pairs[(pair_idx // 2) % len(gpu_pairs)]
            pair_label = f"{gpu_left}-{gpu_right}"
            pair_items = []
            for merged, gpu in ((left_job, gpu_left), (right_job, gpu_right)):
                skip, resumed = should_skip_job(run_dir, str(merged["job_id"]), args.resume_existing)
                if skip:
                    continue
                cmd = build_job_cmd(merged, {}, wandb_cfg, run_dir, gpu, resumed)
                item = {
                    "job_id": merged["job_id"],
                    "backend": merged["backend"],
                    "gpu": gpu,
                    "gpu_pair_label": pair_label,
                    "resumed": resumed,
                    "command": cmd,
                }
                scheduled.append(item)
                pair_items.append(item)
            if len(pair_items) == 2:
                scheduled_pairs.append(
                    {
                        "gpu_pair_label": pair_label,
                        "left": pair_items[0],
                        "right": pair_items[1],
                    }
                )
            elif pair_items:
                raise SystemExit(
                    "paired barrier cannot resume a half-completed pair; "
                    f"rerun the pair explicitly: {left_job['job_id']} / {right_job['job_id']}"
                )
        tmux_commands = build_tmux_pair_commands(args.tmux_session, scheduled_pairs, run_dir, args.dry_run)
        tmux_layout = "one-window-per-gpu-pair-parallel-barrier"
    else:
        for idx, job in enumerate(jobs):
            merged = dict(defaults)
            merged.update(job)
            skip, resumed = should_skip_job(run_dir, str(job["job_id"]), args.resume_existing)
            if skip:
                continue
            gpu = gpus[idx % len(gpus)]
            cmd = build_job_cmd(merged, {}, wandb_cfg, run_dir, gpu, resumed)
            scheduled.append(
                {
                    "job_id": merged["job_id"],
                    "backend": merged["backend"],
                    "gpu": gpu,
                    "resumed": resumed,
                    "command": cmd,
                }
            )
        tmux_commands = build_tmux_commands(args.tmux_session, scheduled)
        tmux_layout = "one-window-per-gpu-sequential-queue"

    manifest = load_manifest(manifest_path) if manifest_path.exists() else {}
    manifest.update(
        {
            "run_id": run_id,
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "gpus": gpus,
            "gpu_pairs": args.gpu_pairs,
            "paired_barrier": args.paired_barrier,
            "tmux_session": args.tmux_session,
            "resume_existing": args.resume_existing,
            "tmux_layout": tmux_layout,
            "scheduled_jobs": scheduled,
            "scheduled_pairs": scheduled_pairs,
        }
    )
    manifest = append_launch_record(
        manifest,
        backend="phase4",
        run_id=run_id,
        tmux_commands=tmux_commands,
        config={
            "scheduled_jobs": scheduled,
            "scheduled_pairs": scheduled_pairs,
            "tmux_layout": tmux_layout,
        },
    )
    if not args.dry_run:
        write_manifest(manifest_path, manifest)

    print(f"run_dir={run_dir}", flush=True)
    print("scheduled_jobs=" + json.dumps(scheduled, indent=2), flush=True)
    for cmd in tmux_commands:
        print(shell_join(cmd), flush=True)
    if args.dry_run:
        return

    created = ensure_tmux_session(args.tmux_session)
    if created:
        print(f"created_tmux_session={args.tmux_session}", flush=True)
    for cmd in tmux_commands:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
