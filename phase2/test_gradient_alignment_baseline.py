"""
Gradient Alignment Test - Real LOZO baseline logic.

Manual implementation of LOZO core logic (not full trainer).
Records U/V matrices and gradients for comparison with vLLM.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party/LOZO/large_models"))

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def prepare_sst2_batch(tokenizer, num_samples=16, seed=42):
    """Load SST2 batch."""
    np.random.seed(seed)
    dataset = load_dataset("glue", "sst2", split="train")
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    dataset = dataset.select(indices)
    
    prompts = []
    for item in dataset:
        sentence = item["sentence"]
        prompt = f"{sentence} It was"
        prompts.append(prompt)
    
    return prompts, dataset


def run_lozo_baseline_step(model, tokenizer, prompts, config, random_seed):
    """
    Run single LOZO step manually (aligned with baseline logic).
    
    Key: Save and restore RNG state to ensure U/V sampling is isolated.
    """
    
    rank = config["rank"]
    eps = config["eps"]
    step_interval = config["step_interval"]
    step = 0  # First step
    
    # Get 2D params (skip embeddings)
    named_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2 and "embed" not in name:
            named_params.append((name, param))
    
    # Save original weights
    original_weights = {}
    for name, param in named_params:
        original_weights[name] = param.data.clone()
    
    # CRITICAL: Save RNG state, set seed, sample U/V, then restore
    rng_state_before = torch.get_rng_state()
    torch.manual_seed(random_seed)
    
    u_matrices = {}
    v_matrices = {}
    v_cache = {}
    
    for name, param in named_params:
        out_features, in_features = param.shape
        
        # V: sample if step % step_interval == 0
        if step % step_interval == 0:
            V = torch.randn(in_features, rank, device="cpu", dtype=torch.float32)
            v_cache[name] = V
            v_matrices[name] = V.clone()
        else:
            V = v_cache[name]
            v_matrices[name] = V.clone()
        
        # U: fresh every step
        U = torch.randn(out_features, rank, device="cpu", dtype=torch.float32)
        u_matrices[name] = U.clone()
    
    # Restore RNG state immediately after sampling
    torch.set_rng_state(rng_state_before)
    
    # Apply perturbation (+eps)
    for name, param in named_params:
        U = u_matrices[name]
        V = v_matrices[name]
        param.data = param.data + eps * (U.to(param.device).to(param.dtype) @ V.to(param.device).to(param.dtype).T)
    
    # Forward pass 1
    model.eval()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    inputs["labels"] = inputs["input_ids"]
    
    with torch.inference_mode():
        outputs = model(**inputs)
        loss1_avg = outputs.loss.item()  # Average NLL per token
    
    # Perturbation (-2eps from current, i.e., -eps from original)
    for name, param in named_params:
        U = u_matrices[name].to(param.device).to(param.dtype)
        V = v_matrices[name].to(param.device).to(param.dtype)
        param.data = param.data - 2 * eps * (U @ V.T)
    
    # Forward pass 2
    with torch.inference_mode():
        outputs = model(**inputs)
        loss2_avg = outputs.loss.item()
    
    # Reset to original
    for name, param in named_params:
        param.data = original_weights[name]
    
    # Compute c (using avg loss)
    projected_grad_avg = (loss1_avg - loss2_avg) / (2 * eps)
    
    # Count tokens
    total_tokens = (inputs["input_ids"].shape[1] - 1) * inputs["input_ids"].shape[0]
    
    return {
        "loss1_avg": loss1_avg,
        "loss2_avg": loss2_avg,
        "loss1_sum": loss1_avg * total_tokens,
        "loss2_sum": loss2_avg * total_tokens,
        "projected_grad_avg": projected_grad_avg,
        "projected_grad_sum": projected_grad_avg * total_tokens,
        "total_tokens": total_tokens,
        "u_matrices": u_matrices,
        "v_matrices": v_matrices,
        "random_seed": random_seed,
    }


def main():
    model_name = "facebook/opt-2.7b"
    rank = 8
    eps = 1e-3
    step_interval = 100
    batch_size = 16
    
    print("=== Configuration ===")
    print(f"rank={rank}, eps={eps}, step_interval={step_interval}")
    print(f"Model: {model_name}")
    print("Note: Skip embeddings and 1D params (aligned with vLLM implementation)")
    
    # Load model (fp16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare batch
    prompts, dataset = prepare_sst2_batch(tokenizer, batch_size)
    print(f"\n=== Loaded {len(prompts)} prompts ===")
    
    # Generate random seed
    np.random.seed(42)
    random_seed = np.random.randint(1000000000)
    print(f"Random seed: {random_seed}")
    
    # Run baseline step
    print("\n=== Running LOZO Baseline Logic ===")
    result = run_lozo_baseline_step(
        model, tokenizer, prompts,
        {"rank": rank, "eps": eps, "step_interval": step_interval},
        random_seed
    )
    
    print(f"\nLoss1 (avg): {result['loss1_avg']:.6f}")
    print(f"Loss1 (sum): {result['loss1_sum']:.6f}")
    print(f"Loss2 (avg): {result['loss2_avg']:.6f}")
    print(f"Loss2 (sum): {result['loss2_sum']:.6f}")
    print(f"Projected grad (avg): {result['projected_grad_avg']:.6f}")
    print(f"Projected grad (sum): {result['projected_grad_sum']:.6f}")
    print(f"Total tokens: {result['total_tokens']}")
    
    # Print U/V stats
    print(f"\n=== U/V Matrices ===")
    print(f"Number of layers: {len(result['u_matrices'])}")
    
    first_name = list(result['u_matrices'].keys())[0]
    U = result['u_matrices'][first_name]
    V = result['v_matrices'][first_name]
    print(f"\nFirst layer: {first_name}")
    print(f"U shape: {U.shape}, U[0,0]: {U[0,0].item():.6f}")
    print(f"V shape: {V.shape}, V[0,0]: {V[0,0].item():.6f}")
    
    # Save result
    output_path = "/home/zelin4593/research_local/zo-vllm/results/baseline_gradient_data.json"
    
    serializable_result = {
        "loss1_avg": result["loss1_avg"],
        "loss2_avg": result["loss2_avg"],
        "loss1_sum": result["loss1_sum"],
        "loss2_sum": result["loss2_sum"],
        "projected_grad_avg": result["projected_grad_avg"],
        "projected_grad_sum": result["projected_grad_sum"],
        "total_tokens": result["total_tokens"],
        "random_seed": result["random_seed"],
        "prompts": prompts,  # Save prompts for exact alignment
        "u_matrices": {k: v.tolist() for k, v in result["u_matrices"].items()},
        "v_matrices": {k: v.tolist() for k, v in result["v_matrices"].items()},
    }
    
    with open(output_path, "w") as f:
        json.dump(serializable_result, f)
    
    print(f"\n=== Saved to {output_path} ===")
    print("Now run phase2/test_gradient_alignment_vllm.py in our vLLM environment")


if __name__ == "__main__":
    main()