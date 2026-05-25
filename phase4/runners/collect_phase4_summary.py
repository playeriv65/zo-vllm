import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.io import load_json
from zo_vllm.experiment.paths import project_root, resolve_path
from zo_vllm.experiment.phase4_summary import summarize_phase4_job


PROJECT_ROOT = project_root()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_summary(run_dir: Path, rows: list[dict]) -> str:
    lines = [
        "# Phase4 Summary",
        "",
        f"Run directory: `{display_path(run_dir)}`",
        "",
        "## Per-Run Table",
        "",
        "| job_id | backend | scope | model | task | steps | bs | lr | eps | rank | status | resumed | converged | final_loss | final_acc | wall_clock_s | step_time_s | steps_per_sec | wandb |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {job_id} | {backend} | {train_scope} | {model_name} | {task_name} | {steps} | {batch_size} | {lr} | {eps} | {rank} | {status} | {resume_count} | {converged} | {final_metric} | {final_acc} | {wall_clock} | {step_time} | {throughput} | {wandb} |".format(
                job_id=row["job_id"],
                backend=row["backend"],
                train_scope=row["train_scope"],
                model_name=row.get("model_name") or "n/a",
                task_name=row.get("task_name") or "n/a",
                steps=row.get("steps") or "n/a",
                batch_size=row.get("batch_size") or "n/a",
                lr=row.get("lr") or "n/a",
                eps=row.get("eps") or "n/a",
                rank=row.get("rank") or "n/a",
                status=row["status"],
                resume_count=row["resume_count"],
                converged=row["converged"],
                final_metric="n/a" if row["final_metric"] is None else f"{row['final_metric']:.6f}",
                final_acc="n/a" if row.get("final_accuracy") is None else f"{row['final_accuracy']:.6f}",
                wall_clock="n/a" if row["wall_clock_s"] is None else f"{row['wall_clock_s']:.2f}",
                step_time="n/a" if row["step_time_s"] is None else f"{row['step_time_s']:.6f}",
                throughput="n/a" if row["steps_per_sec"] is None else f"{row['steps_per_sec']:.4f}",
                wandb=row["wandb_url"] or "n/a",
            )
        )

    lines.extend(["", "## Artifacts", ""])
    for row in rows:
        lines.append(f"- `{row['job_id']}`")
        lines.append(f"  - log: `{row['log_file']}`")
        lines.append(f"  - result: `{row['artifact_json']}`")
        lines.append(
            "  - config: "
            f"model={row.get('model_name')}, task={row.get('task_name')}, "
            f"steps={row.get('steps')}, bs={row.get('batch_size')}, "
            f"lr={row.get('lr')}, eps={row.get('eps')}, rank={row.get('rank')}"
        )

    lines.extend(["", "## Conclusions", ""])
    lozo_rows = [r for r in rows if r["backend"] == "lozo"]
    vllm_rows = [r for r in rows if r["backend"] == "vllm"]
    matched = 0
    for lozo_ref in lozo_rows:
        key = (
            lozo_ref.get("model_name"),
            str(lozo_ref.get("lr")),
            str(lozo_ref.get("eps")),
            str(lozo_ref.get("rank")),
        )
        vllm_ref = next(
            (
                row for row in vllm_rows
                if (
                    row.get("model_name"),
                    str(row.get("lr")),
                    str(row.get("eps")),
                    str(row.get("rank")),
                ) == key
                and _job_interval(row.get("job_id")) == _job_interval(lozo_ref.get("job_id"))
            ),
            None,
        )
        if not vllm_ref:
            continue
        matched += 1
        label = (
            f"model={key[0]}, lr={key[1]}, eps={key[2]}, "
            f"rank={key[3]}, nu={_job_interval(lozo_ref.get('job_id'))}"
        )
        if lozo_ref.get("final_metric") is not None and vllm_ref.get("final_metric") is not None:
            loss_delta = float(vllm_ref["final_metric"]) - float(lozo_ref["final_metric"])
            lines.append(
                f"- `{label}` final_loss delta = `{loss_delta:+.6f}` "
                f"(vLLM `{vllm_ref['final_metric']:.6f}` vs official LOZO `{lozo_ref['final_metric']:.6f}`)."
            )
        if lozo_ref.get("final_accuracy") is not None and vllm_ref.get("final_accuracy") is not None:
            acc_delta = float(vllm_ref["final_accuracy"]) - float(lozo_ref["final_accuracy"])
            lines.append(
                f"- `{label}` final_acc delta = `{acc_delta:+.6f}` "
                f"(vLLM `{vllm_ref['final_accuracy']:.6f}` vs official LOZO `{lozo_ref['final_accuracy']:.6f}`)."
            )
    if not matched:
        lines.append("- No matched official LOZO/vLLM rows were available for loss/accuracy comparison.")
    lines.append("- Convergence flag is derived from final loss <= initial loss when eval loss is available.")
    return "\n".join(lines) + "\n"


def _job_interval(job_id: str | None) -> str | None:
    if not job_id:
        return None
    match = re.search(r"_nu([0-9]+)_", job_id)
    return match.group(1) if match else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    jobs_root = run_dir / "jobs"
    if not jobs_root.exists():
        raise SystemExit(f"jobs dir missing: {jobs_root}")

    rows_by_id = {}
    for job_dir in sorted(path for path in jobs_root.iterdir() if path.is_dir()):
        row = summarize_phase4_job(job_dir)
        rows_by_id[row["job_id"]] = row

    run_manifest_path = run_dir / "manifest.json"
    if run_manifest_path.exists():
        run_manifest = load_json(run_manifest_path)
        scheduled_by_id = {
            item.get("job_id"): item for item in run_manifest.get("scheduled_jobs", []) if item.get("job_id")
        }
        for job_id, row in rows_by_id.items():
            scheduled = scheduled_by_id.get(job_id)
            if not scheduled:
                continue
            command = scheduled.get("command", [])
            if not isinstance(command, list):
                continue
            for i, token in enumerate(command):
                if token == "--task-name" and i + 1 < len(command) and not row.get("task_name"):
                    row["task_name"] = command[i + 1]
                if token == "--model-name" and i + 1 < len(command) and not row.get("model_name"):
                    row["model_name"] = command[i + 1]
        for item in run_manifest.get("scheduled_jobs", []):
            job_id = item.get("job_id")
            if not job_id or job_id in rows_by_id:
                continue
            rows_by_id[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "resume_count": 0,
                "attempts": 0,
                "converged": None,
                "backend": item.get("backend"),
                "train_scope": None,
                "final_metric": None,
                "final_accuracy": None,
                "initial_loss": None,
                "final_loss": None,
                "wall_clock_s": None,
                "step_time_s": None,
                "steps_per_sec": None,
                "wandb_url": None,
                "log_file": str(run_dir / "jobs" / job_id / "logs" / "run.log"),
                "artifact_json": None,
            }

    rows = [rows_by_id[k] for k in sorted(rows_by_id.keys())]

    summary = build_summary(run_dir, rows)
    output = resolve_path(args.output) if args.output else (run_dir / "summary.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary)
    print(summary, flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
