#!/usr/bin/env python
"""Short Phase 3 benchmark for LOZO dense vs LoRA-form perturbation backends."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
LOZO_LARGE_MODELS = REPO_ROOT / "third_party" / "LOZO" / "large_models"
sys.path.insert(0, str(LOZO_LARGE_MODELS))


METHODS = {
    "original_lozo": ("dense", "dense", 4),
    "lozo_lora_probe": ("lora", "dense", 1),
    "ours_torch": ("lora", "lazy_lora", 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--model-name", default="facebook/opt-13b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--rank-r", type=int, default=2)
    parser.add_argument("--step-interval", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--zo-eps", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--tail-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.tail_steps <= 0:
        raise SystemExit("--tail-steps must be positive")
    if args.tail_steps > args.steps:
        raise SystemExit("--tail-steps must be less than or equal to --steps")
    from LOZOtrainer import LowRankTrainer
    from run_lozo import OurArguments

    perturb_backend, update_backend, dense_writes = METHODS[args.method]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    config = {
        "method": args.method,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "rank_r": args.rank_r,
        "step_interval": args.step_interval,
        "learning_rate": args.learning_rate,
        "zo_eps": args.zo_eps,
        "steps": args.steps,
        "tail_steps": args.tail_steps,
        "lozo_perturbation_backend": perturb_backend,
        "lozo_update_backend": update_backend,
        "dense_write_per_step": dense_writes,
    }
    print("CONFIG " + json.dumps(config, sort_keys=True), flush=True)

    load_t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_t0
    print(f"loaded model in {load_s:.3f}s", flush=True)

    training_args = OurArguments(
        output_dir=str(args.output_json.parent / f"{args.method}_hf_output"),
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

    input_ids = torch.randint(
        10,
        model.config.vocab_size - 1,
        (args.batch_size, args.seq_len),
        device=device,
        dtype=torch.long,
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }

    rows = []
    first_measured_step = max(0, int(args.steps) - int(args.tail_steps))
    for raw_step in range(args.steps):
        measured = raw_step >= first_measured_step
        torch.cuda.synchronize()
        step_t0 = time.perf_counter()
        loss = trainer.lowrank_zo_step(model, inputs)
        torch.cuda.synchronize()
        zo_step_s = time.perf_counter() - step_t0

        update_t0 = time.perf_counter()
        trainer.lowrank_zo_update()
        torch.cuda.synchronize()
        update_s = time.perf_counter() - update_t0
        total_s = zo_step_s + update_s
        row = {
            "raw_step": raw_step,
            "measured": measured,
            "loss_plus": float(loss),
            "projected_grad": float(trainer.projected_grad),
            "zo_step_s": float(zo_step_s),
            "update_s": float(update_s),
            "total_s": float(total_s),
        }
        print("ROW " + json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    measured_rows = [row for row in rows if row["measured"]]
    summary = {
        "total_s": summarize([row["total_s"] for row in measured_rows]),
        "zo_step_s": summarize([row["zo_step_s"] for row in measured_rows]),
        "update_s": summarize([row["update_s"] for row in measured_rows]),
    }
    payload = {
        "config": config,
        "load_s": load_s,
        "rows": rows,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"OUTPUT_JSON {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
