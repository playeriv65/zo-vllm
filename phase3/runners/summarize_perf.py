import argparse
import json
import os
from pathlib import Path


def load_result(path):
    with open(path) as f:
        data = json.load(f)
    config = data.get("config", {})
    timing = data.get("timing", {})
    step_timing = timing.get("step_s")
    if isinstance(step_timing, dict):
        step_mean = step_timing.get("mean", 0.0)
    else:
        step_mean = timing.get("step_s_mean", 0.0)
    steps = config.get("steps") or config.get("num_steps")
    total_s = timing.get("total_s", 0.0)
    total_s_per_step = total_s / steps if steps else 0.0
    return {
        "path": str(path),
        "backend": config.get("backend", "unknown"),
        "profile_mode": config.get("profile_mode", "unknown"),
        "steps": steps,
        "batch_size": config.get("batch_size"),
        "initial_loss": data.get("initial_loss"),
        "final_loss": data.get("final_loss"),
        "loss_drop": -(data.get("loss_change") or 0.0),
        "total_s": total_s,
        "total_s_per_step": total_s_per_step,
        "step_s_mean": step_mean,
    }


def fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generic ad-hoc summary helper. For formal q=1 results, prefer "
            "collect_q1_speed_suite.py or collect_q1_batch_sweep.py."
        )
    )
    parser.add_argument("json", nargs="+", help="Performance result JSON files.")
    parser.add_argument("--output", default=None, help="Optional markdown output path.")
    args = parser.parse_args()

    rows = [load_result(Path(path)) for path in args.json]
    baseline = rows[0] if rows else None

    lines = [
        "# Phase 3 Performance Summary",
        "",
        "| backend | profile | steps | batch | initial | final | loss_drop | total_s | total_s/step | raw_step_s_mean | total_speedup_vs_first |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speedup = None
        if baseline and row["total_s_per_step"] and baseline["total_s_per_step"]:
            speedup = baseline["total_s_per_step"] / row["total_s_per_step"]
        lines.append(
            "| {backend} | {profile_mode} | {steps} | {batch_size} | {initial} | "
            "{final} | {drop} | {total} | {total_per_step} | {step} | {speedup} |".format(
                backend=row["backend"],
                profile_mode=row["profile_mode"],
                steps=row["steps"],
                batch_size=row["batch_size"],
                initial=fmt(row["initial_loss"]),
                final=fmt(row["final_loss"]),
                drop=fmt(row["loss_drop"]),
                total=fmt(row["total_s"], 4),
                total_per_step=fmt(row["total_s_per_step"]),
                step=fmt(row["step_s_mean"]),
                speedup=fmt(speedup, 4),
            )
        )

    lines.extend([
        "",
        "Notes:",
        "",
        "- `total_s/step` is the primary speed metric because some backends time only part of a training step in their raw step timer.",
        "- `raw_step_s_mean` is preserved for component-level debugging and should not be used alone for cross-backend speed claims.",
    ])

    lines.extend(["", "Artifacts:", ""])
    for row in rows:
        lines.append(f"- `{os.path.relpath(row['path'])}`")

    output = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
    print(output)


if __name__ == "__main__":
    main()
