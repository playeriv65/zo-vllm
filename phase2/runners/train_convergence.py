"""
LOZO convergence training wrapper for the historical Phase 2 SST-2 prompt loss.

The experiment-specific step semantics stay here, while the outer train loop,
logging/eval cadence, and callback surface are delegated to VLLMZOTrainer.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.infra.env import configure_hf_cache

configure_hf_cache(str(PROJECT_ROOT))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import numpy as np
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM

from zo_vllm.core.direction_digest import digest_named_uv
from zo_vllm.core.lora_runtime import LoRAUpdateRuntime
from zo_vllm.core.lora_scope import DEFAULT_TRANSFORMER_TARGET_MODULES
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.experiment.scoring.generate_scorer import VLLMScorer
from zo_vllm.training import (
    VLLMZOTrainer,
    VLLMZOTrainerCallback,
    ZOTrainingArguments,
)
from zo_vllm.training.direction import (
    LOZOFastDirectionProvider,
    build_lora_runtime_tensors,
    compute_projected_grad,
)


TIMING_KEYS = (
    "step_s",
    "direction_s",
    "build_lora_s",
    "score_s",
    "score_request_build_s",
    "score_generate_s",
    "score_postprocess_s",
    "score_postprocess_plus_s",
    "score_postprocess_minus_s",
    "score_num_outputs",
    "score_num_prompt_positions",
    "score_num_loss_tokens",
    "master_update_s",
    "sync_s",
    "weight_update_s",
    "lora_update_s",
)


class SimpleDataset(Dataset):
    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


def prepare_sst2_data(num_samples: int = 1000) -> SimpleDataset:
    """Load and prepare SST-2 prompts for causal LM convergence checks."""

    dataset = load_dataset("glue", "sst2", split="train")
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)

    prompts = [f"{item['sentence']} It was" for item in dataset]
    return SimpleDataset(prompts)


@dataclass
class Phase2ConvergenceStepModel:
    direction_provider: LOZOFastDirectionProvider
    lora_runtime: LoRAUpdateRuntime
    scorer: VLLMScorer
    weight_sync: WeightSync
    eps: float
    lr: float
    nu: int
    weight_decay: float
    weight_update_precision: str
    direction_digest: bool

    def step(self, batch: list[str], *, step: int) -> dict[str, Any]:
        step_t0 = time.perf_counter()
        random_seed = np.random.randint(1000000000)

        direction_t0 = time.perf_counter()
        directions_2d = self.direction_provider.sample_direction(random_seed)
        direction_digest = None
        if self.direction_digest:
            direction_digest = digest_named_uv(
                (name, item["U"], item["V"]) for name, item in directions_2d.items()
            )
        direction_s = time.perf_counter() - direction_t0

        build_lora_t0 = time.perf_counter()
        plus_a, plus_b = build_lora_runtime_tensors(
            directions_2d,
            eps=self.eps,
            sign=+1,
            output_device="cuda",
        )
        minus_a, minus_b = build_lora_runtime_tensors(
            directions_2d,
            eps=self.eps,
            sign=-1,
            output_device="cuda",
        )
        build_lora_s = time.perf_counter() - build_lora_t0

        lora_update_t0 = time.perf_counter()
        self.lora_runtime.update_plus_minus(
            plus_a,
            plus_b,
            minus_a,
            minus_b,
            step=step,
        )
        lora_update_s = time.perf_counter() - lora_update_t0

        score_t0 = time.perf_counter()
        loss_plus, loss_minus, score_detail = self.scorer.score_plus_minus_detailed(
            batch,
            self.lora_runtime,
        )
        score_s = time.perf_counter() - score_t0
        c = compute_projected_grad(loss_plus, loss_minus, eps=self.eps)

        weight_update_t0 = time.perf_counter()
        self.weight_sync.apply_lozo_update(
            directions_2d,
            c=c,
            lr=self.lr,
            weight_decay=self.weight_decay,
            precision=self.weight_update_precision,
        )
        weight_update_s = time.perf_counter() - weight_update_t0
        step_s = time.perf_counter() - step_t0

        return {
            "step": int(step),
            "seed": int(random_seed),
            "loss_plus": float(loss_plus),
            "loss_minus": float(loss_minus),
            "c": float(c),
            "direction_digest": direction_digest,
            "step_s": float(step_s),
            "direction_s": float(direction_s),
            "build_lora_s": float(build_lora_s),
            "score_s": float(score_s),
            "score_request_build_s": float(score_detail["score_request_build_s"]),
            "score_generate_s": float(score_detail["score_generate_s"]),
            "score_postprocess_s": float(score_detail["score_postprocess_s"]),
            "score_postprocess_plus_s": float(
                score_detail["score_postprocess_plus_s"]
            ),
            "score_postprocess_minus_s": float(
                score_detail["score_postprocess_minus_s"]
            ),
            "score_num_outputs": int(score_detail["score_num_outputs"]),
            "score_num_prompt_positions": int(
                score_detail["score_num_prompt_positions"]
            ),
            "score_num_loss_tokens": int(score_detail["score_num_loss_tokens"]),
            "master_update_s": 0.0,
            "sync_s": 0.0,
            "weight_update_s": float(weight_update_s),
            "lora_update_s": float(lora_update_s),
        }


class Phase2ConvergenceCallback(VLLMZOTrainerCallback):
    def __init__(
        self,
        *,
        history: list[dict[str, Any]],
        timing: dict[str, list[float]],
        eval_losses: list[dict[str, float]],
        nu: int,
        use_wandb: bool,
    ) -> None:
        self.history = history
        self.timing = timing
        self.eval_losses = eval_losses
        self.nu = int(nu)
        self.use_wandb = bool(use_wandb)

    def on_step_end(self, args, state, control, **kwargs):
        result = dict(kwargs["result"])
        self.history.append(result)
        for key in TIMING_KEYS:
            self.timing[key].append(float(result[key]))

        step = int(state.global_step)
        if self.use_wandb:
            wandb.log(
                {
                    "step": step,
                    "loss_plus": result["loss_plus"],
                    "loss_minus": result["loss_minus"],
                    "c": result["c"],
                    "type": "train",
                }
            )
        if step % 10 == 0:
            print(f"Step {step}: c={float(result['c']):.4f}", flush=True)
        if step % self.nu == 0:
            print(f"  V cache updated at step {step}", flush=True)
            if self.use_wandb:
                wandb.log({"step": step, "v_cache_update": step})
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = dict(kwargs["metrics"])
        loss = float(metrics["loss"])
        step = int(state.global_step)
        self.eval_losses.append({"step": step, "loss": loss})
        print(f"Step {step} Eval Loss: {loss:.6f}", flush=True)
        if self.use_wandb:
            wandb.log({"step": step, "val_loss": loss})
        return control


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-7, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--eps", type=float, default=1e-3, help="ZO epsilon")
    parser.add_argument("--nu", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--model-name", default="facebook/opt-2.7b")
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. If omitted, inherit the environment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--train-scope", choices=["lora_normal"], default="lora_normal")
    parser.add_argument(
        "--enforce-eager",
        choices=["0", "1"],
        default="1",
        help="Pass enforce_eager to vLLM.",
    )
    parser.add_argument("--weight-update", choices=["direct"], default="direct")
    parser.add_argument(
        "--weight-update-precision",
        choices=["float32", "param"],
        default="param",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--direction-digest", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    return parser


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise SystemExit("train_convergence requires CUDA for GPU direct LoRA slots")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"zo-vllm-r{args.rank}-{args.steps}steps-{timestamp}"
    use_wandb = not args.no_wandb

    if use_wandb:
        wandb.init(
            project="zo-vllm",
            entity="playeriv65-university-of-minnesota",
            name=run_name,
            config={
                "model": args.model_name,
                "rank_r": args.rank,
                "lr": args.lr,
                "zo_eps": args.eps,
                "nu": args.nu,
                "batch_size": args.batch_size,
                "num_steps": args.steps,
            },
        )

    print(f"Run: {run_name}", flush=True)
    print(
        f"Config: rank={args.rank}, lr={args.lr}, eps={args.eps}, "
        f"steps={args.steps}, weight_update={args.weight_update}, "
        f"weight_update_precision={args.weight_update_precision}, "
        f"enforce_eager={args.enforce_eager}",
        flush=True,
    )

    model_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name,
        enforce_eager=bool(int(args.enforce_eager)),
        enable_lora=True,
        max_lora_rank=args.rank,
        max_loras=2,
        lora_target_modules=list(DEFAULT_TRANSFORMER_TARGET_MODULES),
        gpu_memory_utilization=0.3,
    )
    lora_runtime = LoRAUpdateRuntime.from_model_config(
        model_config,
        rank=args.rank,
        base_model_name=args.model_name,
        llm=llm,
        target_modules=list(DEFAULT_TRANSFORMER_TARGET_MODULES),
    )
    lora_runtime.register_slots()
    scorer = VLLMScorer(llm, tokenizer)
    weight_sync = WeightSync(
        llm,
        num_layers=model_config.num_hidden_layers,
        model_config=model_config,
    )
    direction_provider = LOZOFastDirectionProvider(
        param_metadata=weight_sync.get_hf_param_metadata(),
        rank=args.rank,
        nu=args.nu,
        random_device=args.zo_random_device,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = prepare_sst2_data(num_samples=1000)
    print(f"Loaded {len(dataset)} prompts from SST2", flush=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        drop_last=True,
    )
    eval_prompts = [dataset[i] for i in range(args.batch_size)]

    initial_loss = scorer.score_base(eval_prompts)
    print(f"Initial loss: {initial_loss:.6f}", flush=True)
    if use_wandb:
        wandb.log({"step": 0, "loss": initial_loss, "type": "base"})

    np.random.seed(args.seed)
    history: list[dict[str, Any]] = []
    eval_losses: list[dict[str, float]] = [
        {"step": 0, "loss": float(initial_loss)}
    ]
    timing = {key: [] for key in TIMING_KEYS}
    train_t0 = time.perf_counter()

    model = Phase2ConvergenceStepModel(
        direction_provider=direction_provider,
        lora_runtime=lora_runtime,
        scorer=scorer,
        weight_sync=weight_sync,
        eps=args.eps,
        lr=args.lr,
        nu=args.nu,
        weight_decay=args.weight_decay,
        weight_update_precision=args.weight_update_precision,
        direction_digest=args.direction_digest,
    )

    def eval_fn() -> dict[str, float]:
        loss = float(scorer.score_base(eval_prompts))
        return {"loss": loss, "eval_loss": loss}

    trainer_args = ZOTrainingArguments(
        output_dir=str(
            args.output_dir
            or PROJECT_ROOT / "phase2" / "results" / "convergence"
        ),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=1,
        eval_steps=max(1, int(args.eval_interval)),
        save_strategy="no",
        seed=args.seed,
        dataloader_drop_last=True,
        report_to=("wandb",) if use_wandb else (),
        run_name=run_name,
    )
    trainer = VLLMZOTrainer(
        model=model,
        args=trainer_args,
        train_dataloader=dataloader,
        eval_fn=eval_fn,
        callbacks=[
            Phase2ConvergenceCallback(
                history=history,
                timing=timing,
                eval_losses=eval_losses,
                nu=args.nu,
                use_wandb=use_wandb,
            )
        ],
    )
    trainer.train()

    final_loss = float(scorer.score_base(eval_prompts))
    if not eval_losses or int(eval_losses[-1]["step"]) != int(args.steps):
        eval_losses.append({"step": int(args.steps), "loss": final_loss})
    total_s = time.perf_counter() - train_t0
    print(f"\nInitial loss: {initial_loss:.4f}", flush=True)
    print(f"Final loss: {final_loss:.4f}", flush=True)
    print(f"Loss change: {final_loss - initial_loss:.4f}", flush=True)
    if use_wandb:
        wandb.log(
            {
                "final_loss": final_loss,
                "loss_change": final_loss - initial_loss,
            }
        )

    results_dir = Path(trainer_args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_file = results_dir / f"vllm_convergence_r{args.rank}_{timestamp}.json"
    history_file.write_text(
        json.dumps(
            {
                "config": {
                    "model": args.model_name,
                    "rank_r": args.rank,
                    "lr": args.lr,
                    "zo_eps": args.eps,
                    "nu": args.nu,
                    "batch_size": args.batch_size,
                    "num_steps": args.steps,
                    "backend": "vllm",
                    "seed": args.seed,
                    "zo_random_device": args.zo_random_device,
                    "train_scope": args.train_scope,
                    "weight_update": args.weight_update,
                    "weight_update_precision": args.weight_update_precision,
                    "enforce_eager": int(args.enforce_eager),
                    "direction_digest": bool(args.direction_digest),
                    "trainer": "VLLMZOTrainer",
                },
                "initial_loss": float(initial_loss),
                "final_loss": float(final_loss),
                "loss_change": float(final_loss - initial_loss),
                "eval_losses": eval_losses,
                "history": history,
                "timing": {
                    "total_s": float(total_s),
                    **{f"{key}_mean": mean_or_zero(timing[key]) for key in TIMING_KEYS},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved convergence history to {history_file}", flush=True)

    lora_runtime.cleanup()
    if use_wandb:
        wandb.finish()
    print("\nConvergence training completed!", flush=True)


if __name__ == "__main__":
    main()
