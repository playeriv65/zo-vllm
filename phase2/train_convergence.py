"""
LOZO Convergence Training - Phase 2 Milestone 4.

Train on SST2 dataset with:
- rank_r=8 (LoRA rank)
- lr=1e-7, zo_eps=1e-3
- step_interval=100
- batch_size=16
- WandB logging

Aligned with LOZO baseline (third_party/LOZO/large_models/lozo.sh).
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(PROJECT_ROOT, ".cache", "hf")
os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(CACHE_ROOT, "datasets"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE_ROOT, "hub"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(CACHE_ROOT, "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_ROOT, "transformers"))
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import sys
sys.path.insert(0, PROJECT_ROOT)

import time
import torch
import wandb
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from phase2.lozo_controller import LOZOController, LOZOConfig
from phase2.temp_lora_runtime import TempLoRARuntime
from phase2.weight_sync import WeightSync
from phase2.memory_lora_loader import install_mocks
from phase2.direction_digest import digest_named_uv


class SimpleDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def prepare_sst2_data(num_samples=1000):
    """Load and prepare SST2 dataset for causal LM training."""
    dataset = load_dataset("glue", "sst2", split="train")

    # Sample subset (consumes numpy state - same seed => same selection)
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)

    prompts = []
    for item in dataset:
        sentence = item["sentence"]
        prompt = f"{sentence} It was"
        prompts.append(prompt)

    return SimpleDataset(prompts)


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-7, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--eps", type=float, default=1e-3, help="ZO perturbation epsilon")
    parser.add_argument("--step-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. If omitted, inherit the environment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--train-scope", choices=["lora_only"], default="lora_only")
    parser.add_argument(
        "--batch-invariant",
        choices=["0", "1"],
        default=os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        help="Set VLLM_BATCH_INVARIANT before importing vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        choices=["0", "1"],
        default="1",
        help="Pass enforce_eager to vLLM. 0 enables compile/CUDA graph paths when supported.",
    )
    parser.add_argument(
        "--lora-residency",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Where temporary plus/minus LoRA tensors are loaded from.",
    )
    parser.add_argument(
        "--lora-injection",
        choices=["auto", "direct", "manager"],
        default="auto",
        help="LoRA update path. auto selects direct for GPU and manager for CPU.",
    )
    parser.add_argument(
        "--weight-update",
        choices=["copy", "direct"],
        default="direct",
        help="Base weight update path. copy keeps the external master+sync path; direct updates vLLM weights in place.",
    )
    parser.add_argument(
        "--weight-update-precision",
        choices=["float32", "param"],
        default="param",
        help="Precision for --weight-update direct. float32 matches the controller update; param uses the vLLM weight dtype in-place.",
    )
    parser.add_argument(
        "--direction-digest",
        action="store_true",
        help="Hash every step's U/V tensors for side-by-side alignment checks. Disabled for speed.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()
    os.environ["VLLM_BATCH_INVARIANT"] = args.batch_invariant
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    lora_injection = args.lora_injection
    if lora_injection == "auto":
        lora_injection = "direct" if args.lora_residency == "gpu" else "manager"

    if args.lora_residency == "gpu" and not torch.cuda.is_available():
        raise SystemExit("--lora-residency gpu requires CUDA")
    if args.lora_residency == "cpu" and lora_injection != "manager":
        raise SystemExit("--lora-injection direct requires --lora-residency gpu")
    if args.lora_residency == "cpu":
        # Install mocks before any vLLM operations on the CPU memory-LoRA path.
        install_mocks()

    from vllm import LLM
    from phase2.vllm_scorer import VLLMScorer
    
    # Configuration
    model_name = "facebook/opt-2.7b"
    rank_r = args.rank
    lr = args.lr
    zo_eps = args.eps
    step_interval = args.step_interval
    batch_size = args.batch_size
    num_steps = args.steps
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"zo-vllm-r{rank_r}-{num_steps}steps-{timestamp}"
    
    # WandB setup
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="zo-vllm",
            entity="playeriv65-university-of-minnesota",
            name=run_name,
            config={
                "model": model_name,
                "rank_r": rank_r,
                "lr": lr,
                "zo_eps": zo_eps,
                "step_interval": step_interval,
                "batch_size": batch_size,
                "num_steps": num_steps,
            }
        )
    
    print(f"Run: {run_name}")
    print(
        f"Config: rank={rank_r}, lr={lr}, eps={zo_eps}, steps={num_steps}, "
        f"lora_residency={args.lora_residency}, "
        f"lora_injection={lora_injection}, "
        f"weight_update={args.weight_update}, "
        f"weight_update_precision={args.weight_update_precision}, "
        f"batch_invariant={args.batch_invariant}, "
        f"enforce_eager={args.enforce_eager}"
    )
    
    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    num_layers = hf_model.config.num_hidden_layers
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load vLLM engine
    llm = LLM(
        model=model_name,
        enforce_eager=bool(int(args.enforce_eager)),
        enable_lora=True,
        max_lora_rank=rank_r,
        max_loras=2,
        gpu_memory_utilization=0.3,  # Reduced from 0.5
    )
    
    # Initialize components
    lozo_config = LOZOConfig(
        rank=rank_r,
        eps=zo_eps,
        lr=lr,
        step_interval=step_interval,
        master_device="cuda" if torch.cuda.is_available() else "cpu",
        random_device=args.zo_random_device,
        train_scope=args.train_scope,
    )
    
    controller = LOZOController(hf_model, lozo_config)
    
    temp_lora = TempLoRARuntime(
        rank=rank_r,
        num_layers=num_layers,
        residency=args.lora_residency,
        injection=lora_injection,
        llm=llm,
    )
    temp_lora.register_slots()
    
    scorer = VLLMScorer(llm, tokenizer)
    
    weight_sync = WeightSync(llm, num_layers=num_layers)
    
    # Load SST2 data (np.random seed before np.random.choice to align with baseline)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = prepare_sst2_data(num_samples=1000)
    print(f"Loaded {len(dataset)} prompts from SST2")

    # DataLoader with SequentialSampler (no numpy consumption for batch selection)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        drop_last=True,
    )

    # Eval prompts: first batch from dataset (fixed, same for both baseline and vLLM)
    eval_prompts = [dataset[i] for i in range(batch_size)]

    # Initial loss (base model on eval batch)
    initial_loss = scorer.score_base(eval_prompts)
    print(f"Initial loss: {initial_loss:.6f}")
    if use_wandb:
        wandb.log({"step": 0, "loss": initial_loss, "type": "base"})

    # Training loop
    np.random.seed(args.seed)
    eval_losses = [{"step": 0, "loss": float(initial_loss)}]
    history = []
    timing = {
        "step_s": [],
        "direction_s": [],
        "build_lora_s": [],
        "score_s": [],
        "score_request_build_s": [],
        "score_generate_s": [],
        "score_postprocess_s": [],
        "score_postprocess_plus_s": [],
        "score_postprocess_minus_s": [],
        "score_num_outputs": [],
        "score_num_prompt_positions": [],
        "score_num_loss_tokens": [],
        "master_update_s": [],
        "sync_s": [],
        "weight_update_s": [],
        "lora_update_s": [],
    }

    step = 0
    data_iter = iter(dataloader)
    train_t0 = time.perf_counter()

    for epoch in range(num_steps // len(dataloader) + 2):
        for batch in dataloader:
            if step >= num_steps:
                break
            step += 1
            step_t0 = time.perf_counter()

            batch_prompts = batch

            # Sample ZO seed (same as baseline: np.random.randint consumes one numpy state)
            random_seed = np.random.randint(1000000000)

            # Sample directions
            direction_t0 = time.perf_counter()
            directions_2d, directions_1d = controller.sample_direction(random_seed)
            direction_digest = None
            if args.direction_digest:
                direction_digest = digest_named_uv(
                    (name, item["U"], item["V"])
                    for name, item in directions_2d.items()
                )
            timing["direction_s"].append(time.perf_counter() - direction_t0)

            # Build LoRA tensors
            build_lora_t0 = time.perf_counter()
            lora_tensor_device = "cuda" if args.lora_residency == "gpu" else "cpu"
            plus_A, plus_B = controller.build_temp_lora_tensors(
                directions_2d,
                sign=+1,
                output_device=lora_tensor_device,
            )
            minus_A, minus_B = controller.build_temp_lora_tensors(
                directions_2d,
                sign=-1,
                output_device=lora_tensor_device,
            )
            timing["build_lora_s"].append(time.perf_counter() - build_lora_t0)

            # Update LoRA slots
            lora_update_t0 = time.perf_counter()
            temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B, step=step)
            timing["lora_update_s"].append(time.perf_counter() - lora_update_t0)

            # Compute loss
            score_t0 = time.perf_counter()
            loss_plus, loss_minus, score_detail = scorer.score_plus_minus_detailed(
                batch_prompts, temp_lora
            )
            timing["score_s"].append(time.perf_counter() - score_t0)
            for key, value in score_detail.items():
                timing[key].append(value)

            # Compute c
            c = controller.compute_c(loss_plus, loss_minus)

            # Update base weights
            weight_update_t0 = time.perf_counter()
            if args.weight_update == "copy":
                master_update_t0 = time.perf_counter()
                updated_weights = controller.apply_update_to_master(
                    directions_2d, directions_1d, c
                )
                timing["master_update_s"].append(time.perf_counter() - master_update_t0)

                sync_t0 = time.perf_counter()
                weight_sync.sync(updated_weights)
                timing["sync_s"].append(time.perf_counter() - sync_t0)
            else:
                timing["master_update_s"].append(0.0)
                weight_sync.apply_lozo_update(
                    directions_2d,
                    directions_1d,
                    c=c,
                    lr=lozo_config.lr,
                    weight_decay=lozo_config.weight_decay,
                    precision=args.weight_update_precision,
                )
                timing["sync_s"].append(0.0)
            timing["weight_update_s"].append(time.perf_counter() - weight_update_t0)
            timing["step_s"].append(time.perf_counter() - step_t0)

            # Record history
            history.append({
                "step": step,
                "seed": int(random_seed),
                "loss_plus": float(loss_plus),
                "loss_minus": float(loss_minus),
                "c": float(c),
                "direction_digest": direction_digest,
                "step_s": float(timing["step_s"][-1]),
                "direction_s": float(timing["direction_s"][-1]),
                "build_lora_s": float(timing["build_lora_s"][-1]),
                "score_s": float(timing["score_s"][-1]),
                "score_request_build_s": float(timing["score_request_build_s"][-1]),
                "score_generate_s": float(timing["score_generate_s"][-1]),
                "score_postprocess_s": float(timing["score_postprocess_s"][-1]),
                "score_postprocess_plus_s": float(timing["score_postprocess_plus_s"][-1]),
                "score_postprocess_minus_s": float(timing["score_postprocess_minus_s"][-1]),
                "score_num_outputs": int(timing["score_num_outputs"][-1]),
                "score_num_prompt_positions": int(timing["score_num_prompt_positions"][-1]),
                "score_num_loss_tokens": int(timing["score_num_loss_tokens"][-1]),
                "master_update_s": float(timing["master_update_s"][-1]),
                "sync_s": float(timing["sync_s"][-1]),
                "weight_update_s": float(timing["weight_update_s"][-1]),
                "lora_update_s": float(timing["lora_update_s"][-1]),
            })

            # Log to WandB
            if use_wandb:
                wandb.log({
                    "step": step,
                    "loss_plus": loss_plus,
                    "loss_minus": loss_minus,
                    "c": c,
                    "type": "train",
                })

            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: c={c:.4f}")

            # Periodic evaluation: forward base model on fixed eval batch
            if step % args.eval_interval == 0:
                val_loss = scorer.score_base(eval_prompts)
                eval_losses.append({"step": step, "loss": float(val_loss)})
                print(f"Step {step} Eval Loss: {val_loss:.6f}")
                if use_wandb:
                    wandb.log({"step": step, "val_loss": val_loss})

            # V cache update indicator
            if step % step_interval == 0:
                print(f"  V cache updated at step {step}")
                if use_wandb:
                    wandb.log({"step": step, "v_cache_update": step})

        if step >= num_steps:
            break

    # Final evaluation
    final_loss = scorer.score_base(eval_prompts)
    if not eval_losses or eval_losses[-1]["step"] != num_steps:
        eval_losses.append({"step": num_steps, "loss": float(final_loss)})
    total_s = time.perf_counter() - train_t0
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss change: {final_loss - initial_loss:.4f}")

    if use_wandb:
        wandb.log({
            "final_loss": final_loss,
            "loss_change": final_loss - initial_loss,
        })

    # Save local results
    results_dir = args.output_dir or os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    history_file = os.path.join(results_dir, f"vllm_convergence_r{rank_r}_{timestamp}.json")
    with open(history_file, "w") as f:
        json.dump({
            "config": {
                "model": model_name,
                "rank_r": rank_r,
                "lr": lr,
                "zo_eps": zo_eps,
                "step_interval": step_interval,
                "batch_size": batch_size,
                "num_steps": num_steps,
                "backend": "vllm",
                "seed": args.seed,
                "zo_random_device": args.zo_random_device,
                "train_scope": args.train_scope,
                "lora_residency": args.lora_residency,
                "lora_injection": lora_injection,
                "weight_update": args.weight_update,
                "weight_update_precision": args.weight_update_precision,
                "batch_invariant": int(args.batch_invariant),
                "enforce_eager": int(args.enforce_eager),
                "direction_digest": bool(args.direction_digest),
            },
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "loss_change": float(final_loss - initial_loss),
            "eval_losses": eval_losses,
            "history": history,
            "timing": {
                "total_s": float(total_s),
                "step_s_mean": float(np.mean(timing["step_s"])) if timing["step_s"] else 0.0,
                "direction_s_mean": float(np.mean(timing["direction_s"])) if timing["direction_s"] else 0.0,
                "build_lora_s_mean": float(np.mean(timing["build_lora_s"])) if timing["build_lora_s"] else 0.0,
                "score_s_mean": float(np.mean(timing["score_s"])) if timing["score_s"] else 0.0,
                "score_request_build_s_mean": float(np.mean(timing["score_request_build_s"])) if timing["score_request_build_s"] else 0.0,
                "score_generate_s_mean": float(np.mean(timing["score_generate_s"])) if timing["score_generate_s"] else 0.0,
                "score_postprocess_s_mean": float(np.mean(timing["score_postprocess_s"])) if timing["score_postprocess_s"] else 0.0,
                "score_postprocess_plus_s_mean": float(np.mean(timing["score_postprocess_plus_s"])) if timing["score_postprocess_plus_s"] else 0.0,
                "score_postprocess_minus_s_mean": float(np.mean(timing["score_postprocess_minus_s"])) if timing["score_postprocess_minus_s"] else 0.0,
                "score_num_outputs_mean": float(np.mean(timing["score_num_outputs"])) if timing["score_num_outputs"] else 0.0,
                "score_num_prompt_positions_mean": float(np.mean(timing["score_num_prompt_positions"])) if timing["score_num_prompt_positions"] else 0.0,
                "score_num_loss_tokens_mean": float(np.mean(timing["score_num_loss_tokens"])) if timing["score_num_loss_tokens"] else 0.0,
                "master_update_s_mean": float(np.mean(timing["master_update_s"])) if timing["master_update_s"] else 0.0,
                "sync_s_mean": float(np.mean(timing["sync_s"])) if timing["sync_s"] else 0.0,
                "weight_update_s_mean": float(np.mean(timing["weight_update_s"])) if timing["weight_update_s"] else 0.0,
                "lora_update_s_mean": float(np.mean(timing["lora_update_s"])) if timing["lora_update_s"] else 0.0,
            },
        }, f, indent=2)
    print(f"Saved convergence history to {history_file}")
    
    # Cleanup
    temp_lora.cleanup()
    if use_wandb:
        wandb.finish()
    
    print("\n✅ Convergence training completed!")


if __name__ == "__main__":
    main()
