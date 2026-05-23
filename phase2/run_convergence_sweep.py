#!/usr/bin/env python
"""
Run the Phase 2 100-step convergence sweep and summarize candidates.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_list(raw: str, cast):
    return [cast(item) for item in raw.split(",") if item]


def run_one(args: argparse.Namespace, output_dir: Path, lr: float, rank: int, step_interval: int) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "phase2" / "run_convergence_experiment.py"),
        "--backend",
        args.backend,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--rank",
        str(rank),
        "--lr",
        str(lr),
        "--eps",
        str(args.eps),
        "--step-interval",
        str(step_interval),
        "--eval-interval",
        str(args.eval_interval),
        "--lora-residency",
        args.lora_residency,
        "--output-dir",
        str(output_dir),
        "--no-wandb",
    ]
    if args.gpu is not None:
        cmd[4:4] = ["--gpu", args.gpu]
    log_path = output_dir / "sweep_driver.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"sweep run failed with exit code {rc}: {output_dir}")


def load_summary(run_dir: Path) -> dict:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def oscillation_count(eval_losses: list[dict]) -> int:
    return sum(
        1
        for prev, cur in zip(eval_losses, eval_losses[1:])
        if float(cur["loss"]) > float(prev["loss"])
    )


def row_for(run_dir: Path, config: dict, summary: dict) -> dict:
    results = summary["results"]
    primary = results.get("vllm") or results.get("baseline")
    assert primary is not None
    timing = primary.get("timing", {})
    eval_losses = primary.get("eval_losses", [])
    row = {
        "run_dir": str(run_dir),
        "backend": primary["config"]["backend"],
        "lr": config["lr"],
        "rank": config["rank"],
        "step_interval": config["step_interval"],
        "initial_loss": primary["initial_loss"],
        "final_loss": primary["final_loss"],
        "loss_change": primary["loss_change"],
        "eval_oscillations": oscillation_count(eval_losses),
        "step_s_mean": timing.get("step_s_mean", 0.0),
        "steps_per_s": 1.0 / timing["step_s_mean"] if timing.get("step_s_mean") else 0.0,
        "score_s_mean": timing.get("score_s_mean", 0.0),
        "sync_s_mean": timing.get("sync_s_mean", 0.0),
    }
    if "alignment" in summary:
        row.update(
            {
                "initial_loss_abs_diff": summary["alignment"]["initial_loss_abs_diff"],
                "sign_match_rate": summary["alignment"]["sign_match_rate"],
                "high_signal_sign_match_rate": summary["alignment"]["high_signal_sign_match_rate"],
            }
        )
    return row


def write_sweep_summary(rows: list[dict], output_root: Path) -> None:
    rows = sorted(rows, key=lambda item: (item["loss_change"], item["eval_oscillations"]))
    csv_path = output_root / "sweep_summary.csv"
    fieldnames = [
        "lr",
        "rank",
        "step_interval",
        "initial_loss",
        "final_loss",
        "loss_change",
        "eval_oscillations",
        "steps_per_s",
        "step_s_mean",
        "score_s_mean",
        "sync_s_mean",
        "run_dir",
    ]
    optional_fields = [
        "initial_loss_abs_diff",
        "sign_match_rate",
        "high_signal_sign_match_rate",
    ]
    fieldnames.extend([name for name in optional_fields if any(name in row for row in rows)])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    lines = [
        "| rank | step_interval | lr | initial | final | change | osc | steps/s | run |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['step_interval']} | {row['lr']:.1e} | "
            f"{row['initial_loss']:.6f} | {row['final_loss']:.6f} | "
            f"{row['loss_change']:.6f} | {row['eval_oscillations']} | "
            f"{row['steps_per_s']:.2f} | `{row['run_dir']}` |"
        )
    (output_root / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "baseline", "both"], default="vllm")
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. If omitted, inherit the environment.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--lrs", default="1e-7,3e-7,1e-6")
    parser.add_argument("--ranks", default="8,16")
    parser.add_argument("--step-intervals", default="50,100")
    parser.add_argument("--lora-residency", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else PROJECT_ROOT / "phase2_results" / "convergence" / f"sweep100_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, step_interval, lr in itertools.product(
        parse_list(args.ranks, int),
        parse_list(args.step_intervals, int),
        parse_list(args.lrs, float),
    ):
        run_dir = output_root / f"r{rank}_si{step_interval}_lr{lr:.0e}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {"rank": rank, "step_interval": step_interval, "lr": lr}
        (run_dir / "sweep_config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
        if not (run_dir / "summary.json").exists():
            run_one(args, run_dir, lr=lr, rank=rank, step_interval=step_interval)
        rows.append(row_for(run_dir, config, load_summary(run_dir)))
        write_sweep_summary(rows, output_root)

    print(f"Sweep summary written to {output_root / 'sweep_summary.md'}")


if __name__ == "__main__":
    main()
