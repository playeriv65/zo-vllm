"""Collect Phase 3 SST-2 and SuperGLUE scaling sweep results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zo_vllm.experiment.infra.paths import project_root, resolve_path
from zo_vllm.experiment.infra.superglue_scaling import (
    discover_superglue_scaling_rows,
    mean_present,
    pair_superglue_scaling_rows,
)

PROJECT_ROOT = project_root()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def build_summary(run_dir: Path, rows: list[dict]) -> str:
    pairs = pair_superglue_scaling_rows(rows)
    lines = [
        "# Phase 3 Scaling Sweep Summary",
        "",
        f"Run directory: `{display_path(run_dir)}`",
        "",
        "## Speedup",
        "",
        "| task | model | batch | lozo_s/step | vllm_s/step | speedup | lozo_loss_change | vllm_loss_change | lozo_final_acc | vllm_final_acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lozo = row["lozo"] or {}
        vllm = row["vllm"] or {}
        lines.append(
            "| {task} | {model} | {batch} | {lozo_step} | {vllm_step} | {speedup} | "
            "{lozo_loss} | {vllm_loss} | {lozo_acc} | {vllm_acc} |".format(
                task=row["task"],
                model=row["model"],
                batch=row["batch"],
                lozo_step=fmt(row["lozo_step_time_s"]),
                vllm_step=fmt(row["vllm_step_time_s"]),
                speedup=fmt(row["speedup"], 4),
                lozo_loss=fmt(lozo.get("loss_change")),
                vllm_loss=fmt(vllm.get("loss_change")),
                lozo_acc=fmt(lozo.get("final_accuracy")),
                vllm_acc=fmt(vllm.get("final_accuracy")),
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| group | mean_speedup | completed_pairs |",
            "|---|---:|---:|",
        ]
    )
    by_task: dict[str, list[float | None]] = {}
    for row in pairs:
        by_task.setdefault(str(row["task"]), []).append(row["speedup"])
    for task, values in sorted(by_task.items()):
        completed = sum(value is not None for value in values)
        lines.append(f"| {task} | {fmt(mean_present(values), 4)} | {completed} |")
    lines.extend(
        [
            "",
            f"Completed result files: `{len(rows)}`",
            "",
            "## Artifacts",
            "",
            "| backend | task | model | batch | result | gpu_monitor | wandb |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["task"]),
            str(item["model"]),
            int(item["batch"]),
            str(item["backend"]),
        ),
    ):
        lines.append(
            "| {backend} | {task} | {model} | {batch} | `{path}` | {gpu_monitor} | {wandb} |".format(
                backend=row["backend"],
                task=row["task"],
                model=row["model"],
                batch=row["batch"],
                path=display_path(row["path"]),
                gpu_monitor=(
                    ""
                    if row["gpu_monitor_csv"] is None
                    else f"`{display_path(row['gpu_monitor_csv'])}`"
                ),
                wandb=row["wandb_url"] or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    run_dir = resolve_path(args.run_dir)
    rows = discover_superglue_scaling_rows(run_dir)
    summary = build_summary(run_dir, rows)
    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary)
    else:
        print(summary)
    if args.json_output:
        output = resolve_path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "paired": pair_superglue_scaling_rows(rows),
                },
                default=str,
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
