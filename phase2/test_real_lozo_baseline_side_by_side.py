#!/usr/bin/env python
"""Regression wrapper for real LOZO baseline vs vLLM side-by-side alignment.

The old version manually duplicated both training loops. The accepted path is
now the unified convergence runner, which keeps data sampling, masking, seeds,
and reporting in one place.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. If omitted, inherit the environment.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--step-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--lora-residency", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--batch-invariant", choices=["0", "1"], default="0")
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="1")
    parser.add_argument("--loss-tol", type=float, default=4e-2)
    parser.add_argument("--c-tol", type=float, default=25.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "phase2_results" / "convergence" / f"side_by_side_{timestamp}"
    )
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "phase2" / "run_convergence_experiment.py"),
        "--backend",
        "both",
        "--steps",
        str(args.steps),
        "--rank",
        str(args.rank),
        "--lr",
        str(args.lr),
        "--eps",
        str(args.eps),
        "--step-interval",
        str(args.step_interval),
        "--batch-size",
        str(args.batch_size),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--zo-random-device",
        args.zo_random_device,
        "--lora-residency",
        args.lora_residency,
        "--batch-invariant",
        args.batch_invariant,
        "--enforce-eager",
        args.enforce_eager,
        "--train-scope",
        "lora_only",
        "--output-dir",
        str(output_dir),
        "--no-wandb",
    ]
    if args.gpu is not None:
        cmd[4:4] = ["--gpu", args.gpu]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    with (output_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    alignment = summary.get("alignment")
    if alignment is None:
        raise SystemExit("missing alignment section in summary.json")

    steps_compared = int(alignment["steps_compared"])
    seed_mismatch_steps = alignment["seed_mismatch_steps"]
    digest_mismatch_steps = alignment["direction_digest_mismatch_steps"]
    loss_fail_steps = [
        item["step"]
        for item in alignment["step_pairs"]
        if item["loss_plus_diff"] > args.loss_tol
        or item["loss_minus_diff"] > args.loss_tol
    ]
    c_fail_steps = [
        item["step"]
        for item in alignment["step_pairs"]
        if item["c_diff"] > args.c_tol
    ]
    sign_fail_steps = [
        item["step"] for item in alignment["step_pairs"] if not item["sign_match"]
    ]

    print(f"steps_compared={steps_compared}")
    print(f"seed_mismatch_steps={seed_mismatch_steps}")
    print(f"direction_digest_mismatch_steps={digest_mismatch_steps}")
    print(f"loss_fail_steps@{args.loss_tol:g}={loss_fail_steps}")
    print(f"c_fail_steps@{args.c_tol:g}={c_fail_steps}")
    print(f"sign_fail_steps={sign_fail_steps}")
    print(f"max_loss_plus_diff={alignment['max_loss_plus_diff']:.6f}")
    print(f"max_loss_minus_diff={alignment['max_loss_minus_diff']:.6f}")
    print(f"max_c_diff={alignment['max_c_diff']:.6f}")

    if steps_compared != args.steps:
        raise SystemExit("not all steps were compared")
    if seed_mismatch_steps:
        raise SystemExit("per-step random seeds differ")
    if digest_mismatch_steps:
        raise SystemExit("per-step U/V direction digests differ")
    if loss_fail_steps:
        raise SystemExit("per-step plus/minus loss mismatch exceeds tolerance")
    if c_fail_steps:
        raise SystemExit("per-step projected coefficient mismatch exceeds tolerance")
    if sign_fail_steps:
        raise SystemExit("per-step projected coefficient sign differs")

    print("PASS: step-by-step baseline/vLLM perturbation alignment accepted")


if __name__ == "__main__":
    main()
