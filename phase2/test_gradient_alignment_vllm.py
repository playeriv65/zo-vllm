"""
Gradient Alignment Test - Compare baseline results with our vLLM implementation.

Usage:
    1. First run test_gradient_alignment_baseline.py in LOZO environment
    2. Then run this script in our vLLM environment
    3. Compare U/V matrices and gradients

Key: Both baseline and vLLM return avg loss (not sum).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from vllm import LLM

from phase2.lozo_controller import LOZOController, LOZOConfig
from phase2.temp_lora_runtime import TempLoRARuntime
from phase2.vllm_scorer import VLLMScorer
from phase2.weight_sync import WeightSync
from phase2.memory_lora_loader import install_mocks


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


def run_vllm_step(controller, scorer, temp_lora, prompts, random_seed, baseline_u, baseline_v):
    """
    Run single LOZO step using our vLLM implementation.
    
    Use U/V from baseline results (not self-sampled) to ensure alignment.
    """
    
    # Load U/V from baseline (not sampling ourselves)
    directions_2d = {}
    for name in baseline_u:
        U = torch.tensor(baseline_u[name])
        V = torch.tensor(baseline_v[name])
        directions_2d[name] = {"U": U, "V": V}
    
    directions_1d = {}  # Skip 1D params
    
    # Build LoRA tensors
    plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
    minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
    
    # Update LoRA slots
    temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
    
    # Compute loss
    loss_plus, loss_minus = scorer.score_plus_minus(prompts, temp_lora)
    
    # Compute c
    c = controller.compute_c(loss_plus, loss_minus)
    
    return {
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "c": c,
        "directions": directions_2d,
    }


def compare_directions(baseline_u, baseline_v, vllm_dirs):
    """Compare U/V matrices."""
    
    print("\n=== Direction Comparison ===")
    
    total_params = len(baseline_u)
    matched = 0
    
    for name in baseline_u:
        if name not in vllm_dirs:
            print(f"  {name}: NOT in vLLM directions")
            continue
        
        U_base = torch.tensor(baseline_u[name])
        V_base = torch.tensor(baseline_v[name])
        U_vllm = vllm_dirs[name]["U"].float()
        V_vllm = vllm_dirs[name]["V"].float()
        
        U_match = torch.allclose(U_base, U_vllm, rtol=1e-5)
        V_match = torch.allclose(V_base, V_vllm, rtol=1e-5)
        
        if U_match and V_match:
            matched += 1
            print(f"  {name}: U ✓ V ✓")
        else:
            print(f"  {name}: U {'✓' if U_match else '✗'} V {'✓' if V_match else '✗'}")
            if not U_match:
                print(f"    U diff: max={torch.max(torch.abs(U_base - U_vllm)).item():.6f}")
                print(f"    U base[0,0]={U_base[0,0].item():.6f}, vllm[0,0]={U_vllm[0,0].item():.6f}")
            if not V_match:
                print(f"    V diff: max={torch.max(torch.abs(V_base - V_vllm)).item():.6f}")
                print(f"    V base[0,0]={V_base[0,0].item():.6f}, vllm[0,0]={V_vllm[0,0].item():.6f}")
    
    print(f"\nMatched {matched}/{total_params} parameters")
    return matched == total_params


def main():
    install_mocks()
    
    # Load baseline results
    baseline_path = "results/baseline_gradient_data.json"
    if not os.path.exists(baseline_path):
        print(f"ERROR: Baseline results not found at {baseline_path}")
        print("Please run test_gradient_alignment_baseline.py first in LOZO environment")
        return
    
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)
    
    print("=== Loaded Baseline Results ===")
    print(f"Random seed: {baseline_data['random_seed']}")
    print(f"Loss1 (avg): {baseline_data['loss1_avg']:.6f}")
    print(f"Loss2 (avg): {baseline_data['loss2_avg']:.6f}")
    print(f"Projected grad (avg): {baseline_data['projected_grad_avg']:.6f}")
    
    # Setup our implementation
    model_name = "facebook/opt-2.7b"
    rank = 8
    eps = 1e-3
    lr = 1e-8
    step_interval = 100
    batch_size = 16
    random_seed = baseline_data['random_seed']
    
    # Load HF model (for controller)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load vLLM engine
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=rank,
        max_loras=2,
        gpu_memory_utilization=0.4,
    )
    
    # Initialize components
    lozo_config = LOZOConfig(
        rank=rank,
        eps=eps,
        lr=lr,
        step_interval=step_interval,
    )
    controller = LOZOController(hf_model, lozo_config)
    temp_lora = TempLoRARuntime(rank)
    scorer = VLLMScorer(llm, tokenizer)
    num_layers = hf_model.config.num_hidden_layers
    weight_sync = WeightSync(llm, num_layers)
    
    # Prepare batch (use EXACT prompts from baseline, not re-generated)
    if 'prompts' in baseline_data:
        prompts = baseline_data['prompts']
        print(f"\n=== Using baseline prompts ===")
    else:
        prompts, dataset = prepare_sst2_batch(tokenizer, batch_size)
        print(f"\n=== Loaded {len(prompts)} prompts (re-generated) ===")
    
    # Run vLLM step
    print("\n=== Running vLLM Implementation ===")
    vllm_result = run_vllm_step(
        controller, scorer, temp_lora, prompts, random_seed,
        baseline_data['u_matrices'], baseline_data['v_matrices']
    )
    
    print(f"Loss+: {vllm_result['loss_plus']:.6f}")
    print(f"Loss-: {vllm_result['loss_minus']:.6f}")
    print(f"c: {vllm_result['c']:.6f}")
    
    # Compare results
    print("\n=== Results Comparison ===")
    
    loss1_diff = abs(baseline_data['loss1_avg'] - vllm_result['loss_plus'])
    loss2_diff = abs(baseline_data['loss2_avg'] - vllm_result['loss_minus'])
    c_diff = abs(baseline_data['projected_grad_avg'] - vllm_result['c'])
    
    print(f"Loss1_avg vs Loss+: diff={loss1_diff:.6f} ({'✓' if loss1_diff < 1e-4 else '✗'})")
    print(f"Loss2_avg vs Loss-: diff={loss2_diff:.6f} ({'✓' if loss2_diff < 1e-4 else '✗'})")
    print(f"Projected_grad_avg vs c: diff={c_diff:.6f} ({'✓' if c_diff < 10 else '✗'})")
    
    # Compare U/V matrices
    directions_match = compare_directions(
        baseline_data['u_matrices'],
        baseline_data['v_matrices'],
        vllm_result['directions']
    )
    
    # Cleanup
    temp_lora.cleanup()
    
    # Final verdict
    print("\n=== Final Verdict ===")
    if directions_match:
        print("✅ U/V matrices MATCH - RNG alignment successful!")
    else:
        print("❌ U/V matrices DO NOT MATCH - Need to investigate RNG")
    
    if loss1_diff < 1e-4 and loss2_diff < 1e-4:
        print("✅ Loss values match (within tolerance)")
    else:
        print("❌ Loss values differ significantly")
        print("   Note: Both should be avg loss for comparison")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()