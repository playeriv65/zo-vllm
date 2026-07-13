import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.infra.launch import (  # noqa: E402
    ensure_tmux_session,
    parse_csv,
    shell_join,
)
from zo_vllm.experiment.infra.manifest import (  # noqa: E402
    append_launch_record,
    load_manifest,
    write_manifest,
)
from zo_vllm.experiment.infra.naming import timestamp_now  # noqa: E402
from zo_vllm.experiment.infra.backend_jobs import build_backend_job_command  # noqa: E402
from zo_vllm.experiment.infra.paths import project_root, resolve_path  # noqa: E402
from zo_vllm.experiment.infra.run_state import read_run_state  # noqa: E402

PROJECT_ROOT = project_root()
DEFAULT_TMUX_SESSION = "zo-vllm"


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


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def should_skip_job(
    run_dir: Path, job_id: str, resume_existing: bool
) -> tuple[bool, bool]:
    job_dir = run_dir / "jobs" / job_id
    result_file = job_dir / "result.json"
    if not resume_existing:
        return False, False
    state = read_run_state(job_dir)
    status = state.get("status")
    if status == "completed" and result_file.exists():
        return True, False
    return False, True


def find_latest_lora_checkpoint(job_dir: Path) -> Path | None:
    checkpoint_root = job_dir / "artifacts" / "checkpoints"
    candidates = [
        path
        for path in checkpoint_root.glob("checkpoint-*")
        if path.is_dir() and (path / "zo_lora_bank.pt").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]


def build_job_cmd(
    job: dict, defaults: dict, wandb_cfg: dict, run_dir: Path, gpu: str, resumed: bool
) -> list[str]:
    merged = dict(defaults)
    merged.update(job)
    job_dir = run_dir / "jobs" / str(merged["job_id"])
    if resumed and merged.get("resume_lora_checkpoint") is None:
        latest_lora_checkpoint = find_latest_lora_checkpoint(job_dir)
        if latest_lora_checkpoint is not None:
            merged["resume_lora_checkpoint"] = str(latest_lora_checkpoint)
    task_cfg = dict(defaults.get("task", {}))
    task_cfg.update(job.get("task", {}))
    required_task_keys = ["name", "num_train", "num_dev", "num_eval", "data_seed"]
    missing_task_keys = [key for key in required_task_keys if key not in task_cfg]
    if missing_task_keys:
        raise ValueError(
            "phase4 config must define defaults.task with keys: "
            + ", ".join(required_task_keys)
        )
    task_name = task_cfg["name"]
    task_template = task_cfg.get("template", "default")
    max_length = task_cfg.get("max_length", 2048)
    max_new_tokens = task_cfg.get("max_new_tokens", 50)
    num_samples = task_cfg["num_train"]
    num_dev = task_cfg["num_dev"]
    eval_samples = task_cfg["num_eval"]
    train_set_seed = task_cfg["data_seed"]
    spec = {
        **merged,
        "vllm_runner": "hf_phase4",
        "wandb_project": wandb_cfg.get("project", "lozo-vllm-phase4"),
        "wandb_entity": wandb_cfg.get("entity", "playeriv65-university-of-minnesota"),
        "task_name": task_name,
        "task_template": task_template,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "num_samples": num_samples,
        "num_dev": num_dev,
        "eval_accuracy_samples": eval_samples,
        "train_set_seed": train_set_seed,
        "resume": resumed,
    }
    return build_backend_job_command(run_dir, gpu, spec)


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
        commands.append(
            [
                "tmux",
                "new-window",
                "-t",
                session,
                "-n",
                window,
                f"bash {shlex.quote(str(script_path))}",
            ]
        )
    return commands


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--gpus", default=os.environ.get("PHASE4_GPUS"))
    parser.add_argument("--gpu-pairs", default=os.environ.get("PHASE4_GPU_PAIRS"))
    parser.add_argument("--paired-barrier", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--u-beta", type=float, default=None)
    parser.add_argument("--u-norm-cap", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    defaults = cfg.get("defaults", {})
    if args.u_beta is not None:
        defaults = dict(defaults)
        defaults["u_beta"] = args.u_beta
    if args.u_norm_cap is not None:
        defaults = dict(defaults)
        defaults["u_norm_cap"] = args.u_norm_cap
    wandb_cfg = cfg.get("wandb", {})
    jobs = list(cfg.get("jobs", []))
    if not jobs:
        raise SystemExit("no jobs in config")
    run_id = args.run_id or f"phase4_{timestamp_now()}"
    run_dir = PROJECT_ROOT / "phase4" / "results" / run_id
    manifest_path = run_dir / "manifest.json"

    if run_dir.exists() and not args.resume_existing and not args.dry_run:
        raise SystemExit(f"run dir already exists: {run_dir}")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    scheduled = []
    scheduled_pairs = []
    gpus: list[str] = []
    gpu_pairs: list[tuple[str, str]] = []
    if args.paired_barrier:
        if not args.gpu_pairs:
            raise SystemExit(
                "set --gpu-pairs or PHASE4_GPU_PAIRS for paired barrier mode"
            )
        gpu_pairs = parse_gpu_pairs(args.gpu_pairs)
        if not gpu_pairs:
            raise SystemExit("no GPU pairs provided")
        if len(jobs) % 2 != 0:
            raise SystemExit("paired barrier requires an even number of jobs")
        for pair_idx in range(0, len(jobs), 2):
            left_job = jobs[pair_idx]
            right_job = jobs[pair_idx + 1]
            left_merged = dict(defaults)
            left_merged.update(left_job)
            right_merged = dict(defaults)
            right_merged.update(right_job)
            gpu_left, gpu_right = gpu_pairs[(pair_idx // 2) % len(gpu_pairs)]
            pair_label = f"{gpu_left}-{gpu_right}"
            pair_items = []
            for merged, job, gpu in (
                (left_merged, left_job, gpu_left),
                (right_merged, right_job, gpu_right),
            ):
                skip, resumed = should_skip_job(
                    run_dir,
                    str(merged["job_id"]),
                    args.resume_existing,
                )
                if skip:
                    continue
                cmd = build_job_cmd(job, defaults, wandb_cfg, run_dir, gpu, resumed)
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
                    f"rerun the pair explicitly: {left_merged['job_id']} / {right_merged['job_id']}"
                )
        tmux_commands = build_tmux_pair_commands(
            args.tmux_session, scheduled_pairs, run_dir, args.dry_run
        )
        tmux_layout = "one-window-per-gpu-pair-parallel-barrier"
    else:
        if args.gpu_pairs and not args.gpus:
            raise SystemExit("use --gpus for sequential mode, or add --paired-barrier")
        if not args.gpus:
            raise SystemExit("set --gpus or PHASE4_GPUS")
        gpus = parse_csv(args.gpus)
        if not gpus:
            raise SystemExit("no GPUs provided")
        for idx, job in enumerate(jobs):
            merged = dict(defaults)
            merged.update(job)
            skip, resumed = should_skip_job(
                run_dir, str(job["job_id"]), args.resume_existing
            )
            if skip:
                continue
            gpu = gpus[idx % len(gpus)]
            cmd = build_job_cmd(job, defaults, wandb_cfg, run_dir, gpu, resumed)
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
            "gpu_pairs": [f"{left}:{right}" for left, right in gpu_pairs],
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
