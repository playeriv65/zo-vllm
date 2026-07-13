"""GPU end-to-end smoke tests for real runner scripts.

These tests are opt-in because they download/load models and initialize vLLM.
Run with:

    ZO_VLLM_RUN_E2E=1 CUDA_VISIBLE_DEVICES=0,1 uv run pytest e2e_tests -m e2e
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPT_125M = "facebook/opt-125m"


def _e2e_enabled() -> bool:
    return os.environ.get("ZO_VLLM_RUN_E2E") == "1"


def _base_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env.setdefault("WANDB_MODE", "offline")
    env["PYTHONUNBUFFERED"] = "1"
    env["ZO_VLLM_E2E_TMP"] = str(tmp_path)
    return env


def _run_smoke(cmd: list[str], tmp_path: Path, timeout_s: int = 420) -> str:
    if not _e2e_enabled():
        pytest.skip("set ZO_VLLM_RUN_E2E=1 to run GPU e2e smoke tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU e2e smoke tests")
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=_base_env(tmp_path),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    assert completed.returncode == 0, output[-8000:]
    return output


@pytest.mark.e2e
@pytest.mark.gpu
def test_vllm_zo_task_opt125m_starts_and_runs_one_step(tmp_path: Path):
    output_dir = tmp_path / "vllm_zo_task"
    output = _run_smoke(
        [
            sys.executable,
            "-u",
            "-m",
            "zo_vllm.experiment.runners.vllm_zo_task",
            "--model-name",
            OPT_125M,
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--batch-size",
            "1",
            "--num-samples",
            "2",
            "--num-dev",
            "2",
            "--rank",
            "1",
            "--lr",
            "1e-7",
            "--eps",
            "1e-3",
            "--nu",
            "10",
            "--eval-interval",
            "0",
            "--progress-interval",
            "1",
            "--train-loss-interval",
            "0",
            "--base-eval-mode",
            "skip",
            "--accuracy-eval-mode",
            "skip",
            "--save-strategy",
            "no",
            "--save-final-checkpoint",
            "0",
            "--report-to",
            "none",
            "--zo-random-device",
            "cuda",
            "--enforce-eager",
            "1",
            "--gpu-memory-utilization",
            "0.25",
            "--max-model-len",
            "128",
            "--max-num-batched-tokens",
            "256",
            "--max-num-seqs",
            "4",
            "--output-dir",
            str(output_dir),
        ],
        tmp_path,
    )

    assert "[vLLM] steps=1" in output
    assert "[vLLM] saved=" in output
    assert any(output_dir.glob("vllm_perf_*.json"))


@pytest.mark.e2e
@pytest.mark.gpu
def test_phase2_train_convergence_opt125m_starts_and_runs_one_step(tmp_path: Path):
    output_dir = tmp_path / "phase2_train_convergence"
    output = _run_smoke(
        [
            sys.executable,
            "-u",
            "phase2/runners/train_convergence.py",
            "--model-name",
            OPT_125M,
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--rank",
            "1",
            "--lr",
            "1e-7",
            "--eps",
            "1e-3",
            "--nu",
            "10",
            "--eval-interval",
            "1",
            "--zo-random-device",
            "cuda",
            "--enforce-eager",
            "1",
            "--no-wandb",
            "--output-dir",
            str(output_dir),
        ],
        tmp_path,
    )

    assert "Run: zo-vllm-r1-1steps" in output
    assert "Convergence training completed" in output
    assert any(output_dir.glob("vllm_convergence_r1_*.json"))
