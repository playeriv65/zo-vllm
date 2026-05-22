"""
LOZO Training Loop - Complete implementation aligned with baseline.

Data flow:
1. LOZOController samples U, V directions
2. TempLoRARuntime builds plus/minus LoRA tensors
3. VLLMScorer computes loss_plus, loss_minus
4. LOZOController computes c and updates master weights
5. WeightSync syncs updated weights to vLLM

Note: Currently only supports 2D parameters (Linear layers).
      1D params (bias, layer_norm) are skipped for initial implementation.
"""

import os
import sys
import json
import wandb
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase2.lozo_controller import LOZOController, LOZOConfig
from phase2.temp_lora_runtime import TempLoRARuntime
from phase2.vllm_scorer import VLLMScorer
from phase2.weight_sync import WeightSync


def load_sst2_dataset(
    tokenizer,
    num_train: int = 1000,
    num_dev: int = 500,
    batch_size: int = 16,
):
    """
    Load SST2 dataset for LOZO training.
    
    Returns:
        train_loader: List of (input_ids, label) batches
        dev_loader: List of (input_ids, label) batches
    """
    from datasets import load_dataset
    
    dataset = load_dataset("glue", "sst2")
    
    train_data = dataset["train"].select(range(num_train))
    dev_data = dataset["validation"].select(range(num_dev))
    
    def preprocess(example):
        text = example["sentence"]
        label = example["label"]
        
        # Format: "Sentence: {text}\nAnswer: {positive/negative}"
        if label == 1:
            answer = "positive"
        else:
            answer = "negative"
        
        prompt = f"Sentence: {text}\nAnswer: {answer}"
        
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        
        return {"input_ids": input_ids, "label": label}
    
    train_processed = [preprocess(ex) for ex in train_data]
    dev_processed = [preprocess(ex) for ex in dev_data]
    
    # Create batches
    train_batches = []
    for i in range(0, len(train_processed), batch_size):
        batch = train_processed[i:i+batch_size]
        prompts = [tokenizer.decode(ex["input_ids"]) for ex in batch]
        labels = [ex["label"] for ex in batch]
        train_batches.append((prompts, labels))
    
    dev_batches = []
    for i in range(0, len(dev_processed), batch_size):
        batch = dev_processed[i:i+batch_size]
        prompts = [tokenizer.decode(ex["input_ids"]) for ex in batch]
        labels = [ex["label"] for ex in batch]
        dev_batches.append((prompts, labels))
    
    return train_batches, dev_batches


def train_vllm_lozo(
    model_name: str = "facebook/opt-2.7b",
    rank: int = 1,
    eps: float = 1e-3,
    lr: float = 1e-7,
    step_interval: int = 100,
    num_steps: int = 2000,
    batch_size: int = 16,
    num_train: int = 1000,
    num_dev: int = 500,
    eval_steps: int = 400,
    seed: int = 0,
    gpu_id: int = 5,
    log_dir: str = "logs/phase2",
    wandb_project: str = "zo-vllm",
    wandb_entity: str = "playeriv65-university-of-minnesota",
):
    """
    Train LOZO with vLLM inference engine.
    
    Aligned with LOZO baseline (third_party/LOZO).
    """
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"lozo_train_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"vllm_lozo_rank{rank}_eps{eps}_lr{lr}",
        config={
            "model": model_name,
            "rank": rank,
            "eps": eps,
            "lr": lr,
            "step_interval": step_interval,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "seed": seed,
        },
    )
    
    # Load HF model (for master weights)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    
    # Get number of layers from config
    num_layers = hf_model.config.num_hidden_layers
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load vLLM engine
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=rank,
        max_loras=2,
        gpu_memory_utilization=0.5,
    )
    
    # Initialize components
    lozo_config = LOZOConfig(
        rank=rank,
        eps=eps,
        lr=lr,
        step_interval=step_interval,
    )
    
    controller = LOZOController(hf_model, lozo_config)
    
    temp_lora = TempLoRARuntime(rank=rank, num_layers=num_layers)
    temp_lora.register_slots()
    
    scorer = VLLMScorer(llm, tokenizer)
    
    weight_sync = WeightSync(llm, num_layers=num_layers)
    
    # Load dataset
    train_batches, dev_batches = load_sst2_dataset(
        tokenizer,
        num_train=num_train,
        num_dev=num_dev,
        batch_size=batch_size,
    )
    
    # Training loop
    for step in range(num_steps):
        # Get batch
        batch_idx = step % len(train_batches)
        prompts, labels = train_batches[batch_idx]
        
        # Sample random seed
        random_seed = np.random.randint(1000000000)
        
        # Sample directions (V cached, U fresh)
        directions_2d, directions_1d = controller.sample_direction(random_seed)
        
        # Build plus/minus LoRA tensors
        plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
        minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
        
        # Update LoRA slots
        temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
        
        # Compute loss
        loss_plus, loss_minus = scorer.score_plus_minus(prompts, temp_lora)
        
        # Compute coefficient
        c = controller.compute_c(loss_plus, loss_minus)
        
        # Update master weights
        updated_weights = controller.apply_update_to_master(
            directions_2d, directions_1d, c
        )
        
        # Sync to vLLM (only 2D params)
        weight_sync.sync(updated_weights)
        
        # Log
        wandb.log({
            "step": step,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "c": c,
        })
        
        if step % 10 == 0:
            msg = f"Step {step}: loss+={loss_plus:.4f}, loss-={loss_minus:.4f}, c={c:.4f}"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        
        # Evaluation
        if step > 0 and step % eval_steps == 0:
            eval_loss = 0.0
            for eval_prompts, eval_labels in dev_batches[:10]:
                loss = scorer.score_base(eval_prompts)
                eval_loss += loss
            eval_loss /= 10
            
            wandb.log({
                "eval_loss": eval_loss,
                "eval_step": step,
            })
            
            msg = f"Eval at step {step}: eval_loss={eval_loss:.4f}"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
    
    # Cleanup
    temp_lora.cleanup()
    wandb.finish()
    
    print(f"Training finished. Log saved to {log_file}")


if __name__ == "__main__":
    train_vllm_lozo()