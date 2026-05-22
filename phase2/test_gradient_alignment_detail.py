"""
Gradient Alignment Test - Compare manual baseline vs our vLLM implementation.

Run both on GPU 0 simultaneously to verify:
1. U/V matrices match (same random seed)
2. Loss values match
3. c coefficient match
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from vllm import LLM

from phase2.lozo_controller import LOZOController, LOZOConfig
from phase2.temp_lora_runtime import TempLoRARuntime
from phase2.vllm_scorer import VLLMScorer, compute_nll_from_prompt_logprobs
from phase2.weight_sync import WeightSync
from phase2.memory_lora_loader import install_mocks


def prepare_sst2_batch(tokenizer, num_samples=16, seed=42):
    """Load SST2 batch for testing."""
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


def run_manual_baseline(hf_model, tokenizer, prompts, config, random_seed):
    """
    Run single LOZO step using manual baseline logic.
    
    Key: Use average loss (not sum) for comparison with HF compute_loss.
    """
    
    named_params = []
    for name, param in hf_model.named_parameters():
        if param.requires_grad and param.ndim >= 2 and "embed" not in name:
            named_params.append((name, param))
    
    # V cache (aligned with baseline: step=0 for first call)
    v_cache = {}
    step = 0  # Baseline initializes step to 0 for first call
    
    # Save global RNG state and set seed (aligned with controller)
    rng_state = torch.get_rng_state()
    torch.manual_seed(random_seed)
    
    directions = {}
    rank = config["rank"]
    eps = config["eps"]
    
    for name, param in named_params:
        out_features, in_features = param.shape
        
        # V: sample new if step % step_interval == 0 (step=0 means initialize)
        if step % config["step_interval"] == 0:
            V = torch.randn(in_features, rank, device="cpu", dtype=torch.float32)
            v_cache[name] = V
        else:
            V = v_cache[name]
        
        # U: fresh every step
        U = torch.randn(out_features, rank, device="cpu", dtype=torch.float32)
        
        directions[name] = {"U": U, "V": V}
    
    # Restore global RNG state
    torch.set_rng_state(rng_state)
    
    # Save original weights
    original_weights = {}
    for name, param in named_params:
        original_weights[name] = param.data.clone()
    
    # Perturb for loss1 (+)
    for name, param in named_params:
        U = directions[name]["U"].to(param.device).to(param.dtype)
        V = directions[name]["V"].to(param.device).to(param.dtype)
        param.data = param.data + eps * (U @ V.T)
    
    # Forward pass 1 (HF compute_loss returns avg loss per token)
    hf_model.eval()
    with torch.inference_mode():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(hf_model.device)
        outputs = hf_model(**inputs, labels=inputs["input_ids"])
        loss1_avg = outputs.loss.item()  # Average NLL per token
    
    # Perturb for loss2 (-2 from current = -1 from original)
    for name, param in named_params:
        U = directions[name]["U"].to(param.device).to(param.dtype)
        V = directions[name]["V"].to(param.device).to(param.dtype)
        param.data = param.data - 2 * eps * (U @ V.T)
    
    # Forward pass 2
    with torch.inference_mode():
        outputs = hf_model(**inputs, labels=inputs["input_ids"])
        loss2_avg = outputs.loss.item()
    
    # Reset to original
    for name, param in named_params:
        param.data = original_weights[name]
    
    # Compute c (using avg loss, need to scale for comparison)
    c_avg = (loss1_avg - loss2_avg) / (2 * eps)
    
    # Count total tokens for scaling
    total_tokens = (inputs["input_ids"].shape[1] - 1) * inputs["input_ids"].shape[0]
    
    return {
        "loss1_avg": loss1_avg,
        "loss2_avg": loss2_avg,
        "loss1_sum": loss1_avg * total_tokens,
        "loss2_sum": loss2_avg * total_tokens,
        "c_avg": c_avg,
        "c_sum": c_avg * total_tokens,
        "total_tokens": total_tokens,
        "directions": directions,
    }


def run_vllm_step(controller, scorer, temp_lora, prompts, random_seed):
    """Run single LOZO step using our vLLM implementation."""
    
    # Sample directions
    directions_2d, directions_1d = controller.sample_direction(random_seed)
    
    # Build LoRA tensors
    plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
    minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
    
    # Update LoRA slots
    temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
    
    # Compute loss (returns AVG)
    loss_plus, loss_minus = scorer.score_plus_minus(prompts, temp_lora)
    
    # Compute c
    c = controller.compute_c(loss_plus, loss_minus)
    
    return {
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "c": c,
        "directions": directions_2d,
    }


def compare_directions(baseline_dirs, vllm_dirs):
    """Compare U/V matrices between baseline and vLLM."""
    
    print("\n=== Direction Comparison ===")
    
    total_params = len(baseline_dirs)
    matched = 0
    
    for name in baseline_dirs:
        if name not in vllm_dirs:
            print(f"  {name}: NOT in vLLM directions")
            continue
        
        U_base = baseline_dirs[name]["U"]
        V_base = baseline_dirs[name]["V"]
        U_vllm = vllm_dirs[name]["U"]
        V_vllm = vllm_dirs[name]["V"]
        
        U_match = torch.allclose(U_base, U_vllm.float(), rtol=1e-5)
        V_match = torch.allclose(V_base, V_vllm.float(), rtol=1e-5)
        
        if U_match and V_match:
            matched += 1
            print(f"  {name}: U ✓ V ✓")
        else:
            print(f"  {name}: U {'✓' if U_match else '✗'} V {'✓' if V_match else '✗'}")
            if not U_match:
                print(f"    U diff: max={torch.max(torch.abs(U_base - U_vllm.float())).item():.6f}")
            if not V_match:
                print(f"    V diff: max={torch.max(torch.abs(V_base - V_vllm.float())).item():.6f}")
    
    print(f"\nMatched {matched}/{total_params} parameters")


def main():
    install_mocks()
    
    model_name = "facebook/opt-2.7b"
    rank = 8
    eps = 1e-3
    lr = 1e-8
    step_interval = 100
    batch_size = 16
    
    print("=== Configuration ===")
    print(f"rank={rank}, eps={eps}, lr={lr}, step_interval={step_interval}")
    
    # Load HF model
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
    
    # Initialize our components
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
    
    # Prepare batch
    prompts, dataset = prepare_sst2_batch(tokenizer, batch_size)
    
    print(f"\n=== Loaded {len(prompts)} prompts ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random seed for this step
    random_seed = np.random.randint(1000000000)
    print(f"Random seed: {random_seed}")
    
    # Run manual baseline
    print("\n=== Running Manual Baseline ===")
    baseline_result = run_manual_baseline(
        hf_model, tokenizer, prompts,
        {"rank": rank, "eps": eps, "step_interval": step_interval},
        random_seed,
    )
    print(f"Loss1 (avg): {baseline_result['loss1_avg']:.6f}")
    print(f"Loss1 (sum): {baseline_result['loss1_sum']:.6f}")
    print(f"Loss2 (avg): {baseline_result['loss2_avg']:.6f}")
    print(f"Loss2 (sum): {baseline_result['loss2_sum']:.6f}")
    print(f"c (avg): {baseline_result['c_avg']:.6f}")
    print(f"c (sum): {baseline_result['c_sum']:.6f}")
    print(f"Total tokens: {baseline_result['total_tokens']}")
    
    # Run vLLM
    print("\n=== Running vLLM Implementation ===")
    vllm_result = run_vllm_step(
        controller, scorer, temp_lora, prompts, random_seed,
    )
    print(f"Loss+: {vllm_result['loss_plus']:.6f}")
    print(f"Loss-: {vllm_result['loss_minus']:.6f}")
    print(f"c: {vllm_result['c']:.6f}")
    
    # Compare results
    print("\n=== Results Comparison ===")
    
    # Loss comparison (use avg for comparison)
    loss1_diff = abs(baseline_result['loss1_avg'] - vllm_result['loss_plus'])
    loss2_diff = abs(baseline_result['loss2_avg'] - vllm_result['loss_minus'])
    c_diff = abs(baseline_result['c_avg'] - vllm_result['c'])
    
    print(f"Loss1_avg vs Loss+: diff={loss1_diff:.6f} ({'✓' if loss1_diff < 1e-4 else '✗'})")
    print(f"Loss2_avg vs Loss-: diff={loss2_diff:.6f} ({'✓' if loss2_diff < 1e-4 else '✗'})")
    print(f"c_avg vs c: diff={c_diff:.6f} ({'✓' if c_diff < 10 else '✗'})")
    
    # Direction comparison
    compare_directions(baseline_result['directions'], vllm_result['directions'])
    
    # Cleanup
    temp_lora.cleanup()
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()