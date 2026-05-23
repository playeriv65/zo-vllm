#!/usr/bin/env python
"""
Unified convergence experiment entrypoint for strict baseline/vLLM comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP_LOSS_ABS_TOL = 4e-2
STEP_C_ABS_TOL = 25.0


def run_capture(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout


def write_git_state(output_dir: Path) -> None:
    (output_dir / "git_status.txt").write_text(
        run_capture(["git", "status", "--short"], PROJECT_ROOT),
        encoding="utf-8",
    )
    (output_dir / "git_diff_stat.txt").write_text(
        run_capture(["git", "diff", "--stat"], PROJECT_ROOT),
        encoding="utf-8",
    )


def python_for_baseline() -> str:
    baseline_python = PROJECT_ROOT / "third_party" / "LOZO" / "large_models" / ".venv" / "bin" / "python"
    if baseline_python.exists():
        return str(baseline_python)
    return sys.executable


def build_common_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    common = [
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--rank",
        str(args.rank),
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
        "--train-scope",
        args.train_scope,
        "--output-dir",
        str(output_dir),
        "--no-wandb",
    ]
    if args.gpu is not None:
        common = ["--gpu", args.gpu, *common]
    return common


def run_backend(name: str, args: argparse.Namespace, output_dir: Path) -> None:
    if name == "vllm" and args.train_scope != "lora_only":
        raise SystemExit("vLLM backend only supports --train-scope lora_only")

    env = os.environ.copy()
    cache_root = PROJECT_ROOT / ".cache" / "hf"
    env.update(
        {
            "VLLM_BATCH_INVARIANT": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "HF_HOME": str(cache_root / "home"),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "HF_HUB_CACHE": str(cache_root / "hub"),
            "HF_XET_CACHE": str(cache_root / "xet"),
            "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
            "WANDB_MODE": "offline",
            "WANDB_DISABLED": "true",
        }
    )
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    if name == "baseline":
        cmd = [
            python_for_baseline(),
            str(PROJECT_ROOT / "phase2" / "run_baseline_helper.py"),
            *build_common_args(args, output_dir),
        ]
    elif name == "vllm":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "phase2" / "train_convergence.py"),
            *build_common_args(args, output_dir),
            "--lora-residency",
            args.lora_residency,
        ]
    else:
        raise ValueError(f"unknown backend: {name}")

    log_path = output_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{name}] {line}", end="")
            log.write(line)
            log.flush()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"{name} failed with exit code {rc}; see {log_path}")


def load_backend_result(output_dir: Path, backend: str) -> dict | None:
    matches = sorted(output_dir.glob(f"{backend}_convergence_*.json"))
    if not matches:
        if backend == "vllm":
            matches = sorted(output_dir.glob("vllm_convergence_*.json"))
        elif backend == "baseline":
            matches = sorted(output_dir.glob("baseline_convergence_*.json"))
    if not matches:
        return None
    with matches[-1].open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = str(matches[-1])
    return data


def compact_summary(results: dict[str, dict | None], output_dir: Path) -> None:
    summary = {"results": results}
    baseline = results.get("baseline")
    vllm = results.get("vllm")
    if baseline and vllm:
        step_pairs = []
        for b_step, v_step in zip(baseline.get("history", []), vllm.get("history", [])):
            if b_step.get("step") != v_step.get("step"):
                continue
            b_c = float(b_step["c"])
            v_c = float(v_step["c"])
            loss_plus_diff = abs(float(b_step["loss_plus"]) - float(v_step["loss_plus"]))
            loss_minus_diff = abs(float(b_step["loss_minus"]) - float(v_step["loss_minus"]))
            c_diff = abs(b_c - v_c)
            seed_match = b_step.get("seed") == v_step.get("seed")
            direction_digest_match = (
                b_step.get("direction_digest") is not None
                and b_step.get("direction_digest") == v_step.get("direction_digest")
            )
            sign_match = (b_c == 0.0 and v_c == 0.0) or (b_c * v_c > 0.0)
            high_signal = abs(float(b_step["loss_plus"]) - float(b_step["loss_minus"])) >= 0.005
            step_pairs.append(
                {
                    "step": b_step["step"],
                    "baseline_seed": b_step.get("seed"),
                    "vllm_seed": v_step.get("seed"),
                    "seed_match": seed_match,
                    "direction_digest_match": direction_digest_match,
                    "baseline_loss_plus": float(b_step["loss_plus"]),
                    "vllm_loss_plus": float(v_step["loss_plus"]),
                    "loss_plus_diff": loss_plus_diff,
                    "baseline_loss_minus": float(b_step["loss_minus"]),
                    "vllm_loss_minus": float(v_step["loss_minus"]),
                    "loss_minus_diff": loss_minus_diff,
                    "baseline_c": b_c,
                    "vllm_c": v_c,
                    "c_diff": c_diff,
                    "sign_match": sign_match,
                    "high_signal": high_signal,
                }
            )
        high_signal_pairs = [item for item in step_pairs if item["high_signal"]]
        summary["alignment"] = {
            "initial_loss_abs_diff": abs(baseline["initial_loss"] - vllm["initial_loss"]),
            "final_loss_abs_diff": abs(baseline["final_loss"] - vllm["final_loss"]),
            "steps_compared": len(step_pairs),
            "step_pairs": step_pairs,
            "seed_mismatch_steps": [
                item["step"] for item in step_pairs if not item["seed_match"]
            ],
            "direction_digest_mismatch_steps": [
                item["step"] for item in step_pairs if not item["direction_digest_match"]
            ],
            "sign_match_rate": (
                sum(item["sign_match"] for item in step_pairs) / len(step_pairs)
                if step_pairs
                else None
            ),
            "high_signal_steps": len(high_signal_pairs),
            "high_signal_sign_match_rate": (
                sum(item["sign_match"] for item in high_signal_pairs) / len(high_signal_pairs)
                if high_signal_pairs
                else None
            ),
            "max_loss_plus_diff": max((item["loss_plus_diff"] for item in step_pairs), default=0.0),
            "max_loss_minus_diff": max((item["loss_minus_diff"] for item in step_pairs), default=0.0),
            "max_c_diff": max((item["c_diff"] for item in step_pairs), default=0.0),
            "loss_diff_fail_steps": [
                item["step"]
                for item in step_pairs
                if item["loss_plus_diff"] > STEP_LOSS_ABS_TOL
                or item["loss_minus_diff"] > STEP_LOSS_ABS_TOL
            ],
            "c_diff_fail_steps": [
                item["step"] for item in step_pairs if item["c_diff"] > STEP_C_ABS_TOL
            ],
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "| backend | initial | final | change | step_s_mean | score_s_mean | sync_s_mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, data in results.items():
        if not data:
            continue
        timing = data.get("timing", {})
        lines.append(
            f"| {backend} | {data['initial_loss']:.6f} | {data['final_loss']:.6f} | "
            f"{data['loss_change']:.6f} | {timing.get('step_s_mean', 0.0):.4f} | "
            f"{timing.get('score_s_mean', 0.0):.4f} | {timing.get('sync_s_mean', 0.0):.4f} |"
        )
    if baseline and vllm:
        alignment = summary["alignment"]
        lines.extend(
            [
                "",
                f"initial_loss_abs_diff: {alignment['initial_loss_abs_diff']:.6f}",
                f"final_loss_abs_diff: {alignment['final_loss_abs_diff']:.6f}",
                f"steps_compared: {alignment['steps_compared']}",
                f"seed_mismatch_steps: {alignment['seed_mismatch_steps']}",
                f"direction_digest_mismatch_steps: {alignment['direction_digest_mismatch_steps']}",
                f"sign_match_rate: {alignment['sign_match_rate']:.3f}",
                f"high_signal_sign_match_rate: {alignment['high_signal_sign_match_rate']:.3f}",
                f"max_loss_plus_diff: {alignment['max_loss_plus_diff']:.6f}",
                f"max_loss_minus_diff: {alignment['max_loss_minus_diff']:.6f}",
                f"max_c_diff: {alignment['max_c_diff']:.6f}",
                f"loss_diff_fail_steps@{STEP_LOSS_ABS_TOL:g}: {alignment['loss_diff_fail_steps']}",
                f"c_diff_fail_steps@{STEP_C_ABS_TOL:g}: {alignment['c_diff_fail_steps']}",
                "",
                "| step | seed | uv_digest | b_plus | v_plus | plus_diff | b_minus | v_minus | minus_diff | b_c | v_c | c_diff | sign |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in alignment["step_pairs"]:
            lines.append(
                f"| {item['step']} | {'ok' if item['seed_match'] else 'FAIL'} | "
                f"{'ok' if item['direction_digest_match'] else 'FAIL'} | "
                f"{item['baseline_loss_plus']:.6f} | {item['vllm_loss_plus']:.6f} | "
                f"{item['loss_plus_diff']:.6f} | {item['baseline_loss_minus']:.6f} | "
                f"{item['vllm_loss_minus']:.6f} | {item['loss_minus_diff']:.6f} | "
                f"{item['baseline_c']:.6f} | {item['vllm_c']:.6f} | "
                f"{item['c_diff']:.6f} | {'ok' if item['sign_match'] else 'FAIL'} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["baseline", "vllm", "both"], default="both")
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. If omitted, inherit the environment.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--step-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--train-scope", choices=["lora_only", "full"], default="lora_only")
    parser.add_argument("--lora-residency", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "phase2_results" / "convergence" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    write_git_state(output_dir)

    backends = ["baseline", "vllm"] if args.backend == "both" else [args.backend]
    for backend in backends:
        run_backend(backend, args, output_dir)

    results = {backend: load_backend_result(output_dir, backend) for backend in backends}
    compact_summary(results, output_dir)
    print(f"Summary written to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
