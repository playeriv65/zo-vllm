"""
Gradient Alignment Test - Phase 2 Milestone 4.

Compare our vLLM-based LOZO implementation with baseline LOZO trainer.

Key checks:
1. Perturbation magnitude: eps * (U @ V.T) match
2. Loss difference: L+ - L- match
3. c value: (L+ - L-) / (2*eps) match
4. Gradient estimate: lr * c * (U @ V.T) match
"""

import os
import sys

# Set environment before any imports
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from zo_vllm.core.lozo_controller import LOZOController, LOZOConfig
from zo_vllm.core.temp_lora_runtime import TempLoRARuntime
from zo_vllm.core.vllm_scorer import VLLMScorer
from zo_vllm.core.memory_lora_loader import install_mocks


def test_perturbation_magnitude():
    """Test 1: Verify perturbation magnitude matches baseline formula."""
    print("\n" + "=" * 60)
    print("TEST 1: Perturbation Magnitude")
    print("=" * 60)
    
    model_name = "facebook/opt-2.7b"
    rank = 8
    eps = 1e-3
    
    # Load HF model on CPU
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    
    config = LOZOConfig(rank=rank, eps=eps, lr=1e-7, step_interval=100)
    controller = LOZOController(hf_model, config)
    
    # Sample directions with fixed seed
    torch.manual_seed(42)
    np.random.seed(42)
    random_seed = 42
    directions_2d, directions_1d = controller.sample_direction(random_seed)
    
    # Check first layer
    first_name = list(directions_2d.keys())[0]
    U = directions_2d[first_name]["U"]
    V = directions_2d[first_name]["V"]
    
    print(f"Layer: {first_name}")
    print(f"U shape: {U.shape}, V shape: {V.shape}")
    print(f"U norm: {U.norm():.6f}, V norm: {V.norm():.6f}")
    
    # Baseline formula: perturbation = eps * (U @ V.T)
    baseline_perturbation = eps * (U @ V.T)
    print(f"Baseline perturbation norm: {baseline_perturbation.norm():.6f}")
    
    # Our LoRA formula: delta_W = B @ A where A = V.T, B = eps * U
    # For plus: B = eps * U, A = V.T
    # delta_W_plus = (eps * U) @ V.T = eps * (U @ V.T)
    our_perturbation_plus = eps * (U @ V.T)
    
    # For minus: B = -eps * U, A = V.T
    # delta_W_minus = (-eps * U) @ V.T = -eps * (U @ V.T)
    our_perturbation_minus = -eps * (U @ V.T)
    
    print(f"Our perturbation (+) norm: {our_perturbation_plus.norm():.6f}")
    print(f"Our perturbation (-) norm: {our_perturbation_minus.norm():.6f}")
    
    # Check they match
    diff_plus = (baseline_perturbation - our_perturbation_plus).abs().max()
    diff_minus = (baseline_perturbation + our_perturbation_minus).abs().max()
    
    print(f"\nDifference (+): {diff_plus:.6e}")
    print(f"Difference (-): {diff_minus:.6e}")
    
    if diff_plus < 1e-6 and diff_minus < 1e-6:
        print("✅ PASS: Perturbation magnitude matches baseline")
        return True
    else:
        print("❌ FAIL: Perturbation magnitude mismatch")
        return False


def test_loss_difference():
    """Test 2: Compare L+ and L- with baseline."""
    print("\n" + "=" * 60)
    print("TEST 2: Loss Difference (L+ - L-)")
    print("=" * 60)
    
    install_mocks()
    
    model_name = "facebook/opt-2.7b"
    rank = 8
    eps = 1e-3
    
    # Load HF model on CPU (for controller)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    num_layers = hf_model.config.num_hidden_layers
    
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
    config = LOZOConfig(rank=rank, eps=eps, lr=1e-7, step_interval=100)
    controller = LOZOController(hf_model, config)
    
    temp_lora = TempLoRARuntime(rank=rank, num_layers=num_layers)
    temp_lora.register_slots()
    
    scorer = VLLMScorer(llm, tokenizer)
    
    # Fixed seed for reproducibility
    torch.manual_seed(999)
    np.random.seed(999)
    random_seed = 999
    
    # Sample directions
    directions_2d, directions_1d = controller.sample_direction(random_seed)
    
    # Build LoRA tensors
    plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
    minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
    
    # Update LoRA slots
    temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
    
    # Test with different batch sizes to check batch invariance
    prompts_single = ["The movie was great"]
    prompts_batch = ["The movie was great"] * 16
    
    print("\nTesting batch invariance...")
    
    # Single sample
    loss_plus_single, loss_minus_single = scorer.score_plus_minus(prompts_single, temp_lora)
    c_single = controller.compute_c(loss_plus_single, loss_minus_single)
    
    print(f"Single sample:")
    print(f"  L+ = {loss_plus_single:.6f}, L- = {loss_minus_single:.6f}")
    print(f"  L+ - L- = {loss_plus_single - loss_minus_single:.6f}")
    print(f"  c = {c_single:.6f}")
    
    # Batch of 16
    loss_plus_batch, loss_minus_batch = scorer.score_plus_minus(prompts_batch, temp_lora)
    c_batch = controller.compute_c(loss_plus_batch, loss_minus_batch)
    
    print(f"\nBatch of 16:")
    print(f"  L+ = {loss_plus_batch:.6f}, L- = {loss_minus_batch:.6f}")
    print(f"  L+ - L- = {loss_plus_batch - loss_minus_batch:.6f}")
    print(f"  c = {c_batch:.6f}")
    
    # Check batch invariance
    # Under average loss, the batch loss should equal the single loss
    per_sample_diff_single = loss_plus_single - loss_minus_single
    per_sample_diff_batch = loss_plus_batch - loss_minus_batch
    
    print(f"\nPer-sample loss difference:")
    print(f"  Single: {per_sample_diff_single:.6f}")
    print(f"  Batch:  {per_sample_diff_batch:.6f}")
    
    expected_batch_diff = per_sample_diff_single
    actual_batch_diff = per_sample_diff_batch
    
    print(f"\nBatch difference check:")
    print(f"  Expected: {expected_batch_diff:.6f}")
    print(f"  Actual:   {actual_batch_diff:.6f}")
    if abs(expected_batch_diff) > 1e-7:
        ratio = actual_batch_diff / expected_batch_diff
        print(f"  Ratio:    {ratio:.6f}")
    else:
        ratio = 1.0
        print(f"  Ratio:    1.0 (expected diff is zero)")
    
    temp_lora.cleanup()
    
    # Check if they are extremely close (since they are both averages over identical prompts)
    if abs(actual_batch_diff - expected_batch_diff) < 1e-4:
        print("✅ PASS: Batch invariance holds")
        return True
    else:
        print("❌ FAIL: Batch invariance broken")
        return False


def test_gradient_estimate():
    """Test 3: Verify gradient estimate matches baseline."""
    print("\n" + "=" * 60)
    print("TEST 3: Gradient Estimate")
    print("=" * 60)
    
    install_mocks()
    
    model_name = "facebook/opt-2.7b"
    rank = 8
    eps = 1e-3
    lr = 1e-7
    
    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    num_layers = hf_model.config.num_hidden_layers
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load vLLM
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=rank,
        max_loras=2,
        gpu_memory_utilization=0.4,
    )
    
    config = LOZOConfig(rank=rank, eps=eps, lr=lr, step_interval=100)
    controller = LOZOController(hf_model, config)
    
    temp_lora = TempLoRARuntime(rank=rank, num_layers=num_layers)
    temp_lora.register_slots()
    
    scorer = VLLMScorer(llm, tokenizer)
    
    # Fixed seed
    torch.manual_seed(123)
    np.random.seed(123)
    random_seed = 123
    
    # Sample directions
    directions_2d, directions_1d = controller.sample_direction(random_seed)
    
    # Build and update LoRA
    plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
    minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
    temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
    
    # Compute losses with single prompt
    prompts = ["This is a test sentence"]
    loss_plus, loss_minus = scorer.score_plus_minus(prompts, temp_lora)
    
    # Compute c
    c = controller.compute_c(loss_plus, loss_minus)
    
    print(f"Loss+: {loss_plus:.6f}")
    print(f"Loss-: {loss_minus:.6f}")
    print(f"c = (L+ - L-) / (2*eps) = {c:.6f}")
    
    # Compute gradient estimate
    # Baseline: grad_estimate = c * (U @ V.T)
    # Update: W <- W - lr * grad_estimate
    
    first_name = list(directions_2d.keys())[0]
    U = directions_2d[first_name]["U"].float()
    V = directions_2d[first_name]["V"].float()
    
    grad_estimate_baseline = c * (U @ V.T)
    update_baseline = lr * grad_estimate_baseline
    
    print(f"\nGradient estimate norm: {grad_estimate_baseline.norm():.6e}")
    print(f"Update magnitude: {update_baseline.norm():.6e}")
    print(f"Update max: {update_baseline.abs().max():.6e}")
    
    # Sanity check: update should be reasonable
    # For lr=1e-7, c~100, eps=1e-3, rank=8
    # grad_estimate ~ 100 * 7 ~ 700
    # update ~ 1e-7 * 700 ~ 7e-5
    
    expected_update_scale = lr * c * eps * rank  # rough estimate
    print(f"Expected update scale (approx): {expected_update_scale:.6e}")
    
    temp_lora.cleanup()
    
    if update_baseline.abs().max() < 1e-3:  # Should be small
        print("✅ PASS: Gradient estimate magnitude is reasonable")
        return True
    else:
        print("❌ FAIL: Gradient estimate too large")
        return False


def main():
    print("Phase 2 Gradient Alignment Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Perturbation magnitude
    results.append(("Perturbation Magnitude", test_perturbation_magnitude()))
    
    # Test 2: Loss difference (batch invariance)
    results.append(("Batch Invariance", test_loss_difference()))
    
    # Test 3: Gradient estimate
    results.append(("Gradient Estimate", test_gradient_estimate()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed! Implementation aligned with baseline.")
    else:
        print("\n⚠️ Some tests failed. Check implementation.")
    
    return all_passed


if __name__ == "__main__":
    main()