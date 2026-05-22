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

# Set cache directories BEFORE any imports
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub_cache"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from vllm import LLM

from phase2.lozo_controller import LOZOController, LOZOConfig
from phase2.temp_lora_runtime import TempLoRARuntime
from phase2.vllm_scorer import VLLMScorer
from phase2.weight_sync import WeightSync
from phase2.memory_lora_loader import install_mocks


def prepare_sst2_data(tokenizer, num_samples=1000, max_length=512):
    """Load and prepare SST2 dataset for causal LM training."""
    dataset = load_dataset("glue", "sst2", split="train")
    
    # Sample subset
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)
    
    # Format as prompts for causal LM
    # SST2: sentence + " It was" -> "great"/"terrible"
    prompts = []
    for item in dataset:
        sentence = item["sentence"]
        label = item["label"]  # 0=negative, 1=positive
        # Format: "<sentence> It was" -> model should complete with sentiment
        prompt = f"{sentence} It was"
        prompts.append(prompt)
    
    return prompts, [dataset[i]['label'] for i in range(len(dataset))]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-7, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--eps", type=float, default=1e-3, help="ZO perturbation epsilon")
    args = parser.parse_args()

    # Install mocks before any vLLM operations
    install_mocks()
    
    # Configuration
    model_name = "facebook/opt-2.7b"
    rank_r = args.rank
    lr = args.lr
    zo_eps = args.eps
    step_interval = 100
    batch_size = 16
    num_steps = args.steps
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"zo-vllm-r{rank_r}-{num_steps}steps-{timestamp}"
    
    # WandB setup
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
    print(f"Config: rank={rank_r}, lr={lr}, eps={zo_eps}, steps={num_steps}")
    
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
        enforce_eager=True,
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
    )
    
    controller = LOZOController(hf_model, lozo_config)
    
    temp_lora = TempLoRARuntime(rank=rank_r, num_layers=num_layers)
    temp_lora.register_slots()
    
    scorer = VLLMScorer(llm, tokenizer)
    
    weight_sync = WeightSync(llm, num_layers=num_layers)
    
    # Load SST2 data
    prompts, labels = prepare_sst2_data(tokenizer, num_samples=1000)
    print(f"Loaded {len(prompts)} prompts from SST2")
    
    # Initial loss (base model)
    initial_prompts = prompts[:batch_size]
    initial_loss = scorer.score_base(initial_prompts)
    print(f"Initial loss: {initial_loss:.6f}")
    wandb.log({"step": 0, "loss": initial_loss, "type": "base"})
    
    # Training loop
    losses = []
    np.random.seed(42)
    
    for step in tqdm(range(1, num_steps + 1), desc="Training"):
        # Sample batch
        batch_idx = np.random.randint(0, len(prompts) - batch_size)
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        
        # Sample random seed
        random_seed = np.random.randint(1000000000)
        
        # Sample directions
        directions_2d, directions_1d = controller.sample_direction(random_seed)
        
        # Build LoRA tensors
        plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
        minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
        
        # Update LoRA slots
        temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
        
        # Compute loss
        loss_plus, loss_minus = scorer.score_plus_minus(batch_prompts, temp_lora)
        
        # Compute c
        c = controller.compute_c(loss_plus, loss_minus)
        
        # Update master weights
        updated_weights = controller.apply_update_to_master(
            directions_2d, directions_1d, c
        )
        
        # Sync to vLLM
        weight_sync.sync(updated_weights)
        
        # Compute current loss
        current_loss = scorer.score_base(batch_prompts)
        losses.append(current_loss)
        
        # Log to WandB
        wandb.log({
            "step": step,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "c": c,
            "loss": current_loss,
            "type": "train",
        })
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: loss={current_loss:.4f}, c={c:.4f}")
        
        # Periodic evaluation on the fixed validation batch (no sampling noise)
        if step % 20 == 0 or step == num_steps:
            val_loss = scorer.score_base(prompts[:batch_size])
            print(f"Step {step} Fixed Eval Loss: {val_loss:.6f}")
            wandb.log({"step": step, "val_loss": val_loss})
        
        # V cache update indicator
        if step % step_interval == 0:
            print(f"  V cache updated at step {step}")
            wandb.log({"step": step, "v_cache_update": step})
    
    # Final evaluation
    final_loss = scorer.score_base(prompts[:batch_size])
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss change: {final_loss - initial_loss:.4f}")
    print(f"Average loss: {np.mean(losses):.4f}")
    
    wandb.log({
        "final_loss": final_loss,
        "loss_change": final_loss - initial_loss,
        "avg_loss": np.mean(losses),
    })
    
    # Cleanup
    temp_lora.cleanup()
    wandb.finish()
    
    print("\n✅ Convergence training completed!")


if __name__ == "__main__":
    main()