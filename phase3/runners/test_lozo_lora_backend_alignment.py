#!/usr/bin/env python
"""Alignment test for dense LOZO and LoRA-form LOZO backends.

This is a GPU/e2e Phase 3 check, not a unit test. It loads a small HF model and
verifies that the fast LoRA-form baseline follows the same LOZO random seed
stream, direction tensors, probe losses, projected gradients, and effective
post-update clean loss as the original dense baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
LOZO_LARGE_MODELS = REPO_ROOT / "third_party" / "LOZO" / "large_models"
sys.path.insert(0, str(LOZO_LARGE_MODELS))

from LOZOtrainer import LowRankTrainer  # noqa: E402
from run_lozo import OurArguments  # noqa: E402


METHODS = {
    "original_lozo": ("dense", "dense"),
    "lozo_lora_probe": ("lora", "dense"),
    "ours_torch": ("lora", "lazy_lora"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--rank-r", type=int, default=2)
    parser.add_argument("--step-interval", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--zo-eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-atol", type=float, default=0.0)
    parser.add_argument("--grad-atol", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def make_inputs(
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        10,
        vocab_size - 1,
        (batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def install_rng_recorder(trainer: LowRankTrainer) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    original = trainer._lozo_randn

    def wrapped(shape, target_device, dtype):
        value = original(shape, target_device, dtype)
        records.append(
            {
                "shape": list(shape),
                "device": str(value.device),
                "dtype": str(value.dtype),
                "digest": tensor_digest(value),
            }
        )
        return value

    trainer._lozo_randn = wrapped
    return records


def make_trainer(
    *,
    model: torch.nn.Module,
    method: str,
    args: argparse.Namespace,
) -> LowRankTrainer:
    perturb_backend, update_backend = METHODS[method]
    training_args = OurArguments(
        output_dir=str(args.output_json.parent / f"{method}_hf_output"),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        max_steps=args.steps,
        rank_r=args.rank_r,
        step_interval=args.step_interval,
        zo_eps=args.zo_eps,
        lozo_random_device="cuda",
        lozo_train_scope="lora_normal",
        lozo_perturbation_backend=perturb_backend,
        lozo_update_backend=update_backend,
        trainer="LOZO",
        report_to=[],
        save_strategy="no",
        evaluation_strategy="no",
        disable_tqdm=True,
    )
    trainer = LowRankTrainer(model=model, args=training_args)
    trainer.create_optimizer_and_scheduler(num_training_steps=args.steps)
    return trainer


def effective_clean_loss(
    *,
    trainer: LowRankTrainer,
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> float:
    if trainer._lozo_perturbation_backend() == "lora":
        trainer._lozo_set_lora_perturb_sign(0)
    return float(trainer.zo_forward(model, inputs))


def run_method(
    *,
    method: str,
    inputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    trainer = make_trainer(model=model, method=method, args=args)
    rng_records = install_rng_recorder(trainer)

    rows = []
    for raw_step in range(args.steps):
        start_rng_index = len(rng_records)
        loss_plus = float(trainer.lowrank_zo_step(model, inputs))
        projected_grad = float(trainer.projected_grad)
        loss_minus = loss_plus - 2.0 * args.zo_eps * projected_grad
        step_rng_records = rng_records[start_rng_index:]

        trainer.lowrank_zo_update()
        clean_loss = effective_clean_loss(trainer=trainer, model=model, inputs=inputs)

        row = {
            "raw_step": raw_step,
            "lozo_step": int(trainer.step),
            "zo_random_seed": int(trainer.zo_random_seed),
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "projected_grad": projected_grad,
            "clean_loss_after_update": clean_loss,
            "rng_record_count": len(step_rng_records),
            "rng_prefix": step_rng_records[:8],
        }
        print(f"{method} ROW " + json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    return {
        "method": method,
        "perturbation_backend": METHODS[method][0],
        "update_backend": METHODS[method][1],
        "rows": rows,
    }


def assert_close(
    *,
    failures: list[str],
    label: str,
    left: float,
    right: float,
    atol: float,
) -> None:
    if abs(left - right) > atol:
        failures.append(f"{label}: {left} != {right} (atol={atol})")


def compare_results(results: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    baseline = results["original_lozo"]["rows"]
    for method in ["lozo_lora_probe", "ours_torch"]:
        rows = results[method]["rows"]
        for idx, (base_row, row) in enumerate(zip(baseline, rows, strict=True)):
            if row["zo_random_seed"] != base_row["zo_random_seed"]:
                failures.append(
                    f"{method} step {idx}: zo_random_seed "
                    f"{row['zo_random_seed']} != {base_row['zo_random_seed']}"
                )
            if row["rng_prefix"] != base_row["rng_prefix"][: len(row["rng_prefix"])]:
                failures.append(f"{method} step {idx}: RNG prefix digest mismatch")
            assert_close(
                failures=failures,
                label=f"{method} step {idx} loss_plus",
                left=row["loss_plus"],
                right=base_row["loss_plus"],
                atol=args.loss_atol,
            )
            assert_close(
                failures=failures,
                label=f"{method} step {idx} loss_minus",
                left=row["loss_minus"],
                right=base_row["loss_minus"],
                atol=args.loss_atol,
            )
            assert_close(
                failures=failures,
                label=f"{method} step {idx} projected_grad",
                left=row["projected_grad"],
                right=base_row["projected_grad"],
                atol=args.grad_atol,
            )
            assert_close(
                failures=failures,
                label=f"{method} step {idx} clean_loss_after_update",
                left=row["clean_loss_after_update"],
                right=base_row["clean_loss_after_update"],
                atol=args.loss_atol,
            )
    return failures


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for LOZO CUDA RNG alignment")
    device = torch.device("cuda:0")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    probe_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    ).to(device)
    vocab_size = int(probe_model.config.vocab_size)
    del probe_model
    torch.cuda.empty_cache()
    inputs = make_inputs(
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed + 17,
        device=device,
    )

    results = {
        method: run_method(method=method, inputs=inputs, args=args, device=device)
        for method in ["original_lozo", "lozo_lora_probe", "ours_torch"]
    }
    failures = compare_results(results, args)
    payload = {
        "config": {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "rank_r": args.rank_r,
            "step_interval": args.step_interval,
            "learning_rate": args.learning_rate,
            "zo_eps": args.zo_eps,
            "seed": args.seed,
            "loss_atol": args.loss_atol,
            "grad_atol": args.grad_atol,
        },
        "results": results,
        "failures": failures,
        "ok": not failures,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print("OUTPUT_JSON " + str(args.output_json), flush=True)
    if failures:
        print("FAILURES " + json.dumps(failures, indent=2), flush=True)
        raise SystemExit(1)
    print("ALIGNMENT_OK", flush=True)


if __name__ == "__main__":
    main()
