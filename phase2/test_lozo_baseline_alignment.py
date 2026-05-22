"""
Full LOZO Baseline Alignment - Compare with real LOZOtrainer.

Run both our vLLM implementation and LOZO baseline with:
- Same random seed
- Same data batch
- Compare c values, loss_plus, loss_minus
"""

import os
import sys

os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from phase2.lozo_controller import LOZOController, LOZOConfig
from phase2.temp_lora_runtime import TempLoRARuntime
from phase2.vllm_scorer import VLLMScorer
from phase2.memory_lora_loader import install_mocks


def run_lozo_baseline():
    """
    Run LOZO baseline trainer's lowrank_zo_step logic.
    
    Returns: (loss_plus, loss_minus, c, random_seed)
    """
    print("\n" + "=" * 60)
    print("LOZO BASELINE (HF model forward)")
    print("=" * 60)
    
    model_name = "facebook/opt-2.7b"
    rank_r = 8
    zo_eps = 1e-3
    
    # Load HF model on GPU (device-agnostic)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    hf_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare input
    text = "The movie was great"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Fixed random seed
    np.random.seed(42)
    random_seed = np.random.randint(1000000000)
    print(f"Random seed: {random_seed}")
    
    # V cache (simulate step_interval behavior)
    v_cache = {}
    step = 0
    
    # Get trainable parameters (matching baseline but skipping non-linear/1D)
    named_parameters_to_optim = []
    for name, param in hf_model.named_parameters():
        if param.requires_grad:
            if "embed" in name or param.ndim == 1:
                continue
            named_parameters_to_optim.append((name, param))
    
    print(f"Trainable params: {len(named_parameters_to_optim)}")
    
    # Perturb with +eps (loss_plus)
    torch.manual_seed(random_seed)
    
    for name, param in named_parameters_to_optim:
        if param.data.ndim >= 2:
            if step % 100 == 0:  # step_interval = 100
                v = torch.randn(param.data.size(1), rank_r, device="cpu", dtype=param.data.dtype).to(param.data.device)
                v_cache[name] = v
            else:
                v = v_cache[name]
            u = torch.randn(param.data.size(0), rank_r, device="cpu", dtype=param.data.dtype).to(param.data.device)
            param.data = param.data + (u @ v.t()) * zo_eps
        else:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device="cpu", dtype=param.data.dtype).to(param.data.device)
            param.data = param.data + z * zo_eps
    
    # Forward pass (loss_plus)
    with torch.no_grad():
        outputs_plus = hf_model(**inputs, labels=inputs["input_ids"])
        loss_plus = outputs_plus.loss.item()
    
    print(f"Loss+ (baseline): {loss_plus:.6f}")
    
    # Perturb with -eps (reset then apply -2)
    torch.manual_seed(random_seed)
    
    for name, param in named_parameters_to_optim:
        if param.data.ndim >= 2:
            if step % 100 == 0:  # step_interval = 100
                v = torch.randn(param.data.size(1), rank_r, device="cpu", dtype=param.data.dtype).to(param.data.device)
                v_cache[name] = v
            else:
                v = v_cache[name]
            u = torch.randn(param.data.size(0), rank_r, device="cpu", dtype=param.data.dtype).to(param.data.device)
            param.data = param.data - 2 * (u @ v.t()) * zo_eps
        else:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device="cpu", dtype=param.data.dtype).to(param.data.device)
            param.data = param.data - 2 * z * zo_eps
    
    # Forward pass (loss_minus)
    with torch.no_grad():
        outputs_minus = hf_model(**inputs, labels=inputs["input_ids"])
        loss_minus = outputs_minus.loss.item()
    
    print(f"Loss- (baseline): {loss_minus:.6f}")
    
    # Compute c
    c_baseline = (loss_plus - loss_minus) / (2 * zo_eps)
    print(f"c (baseline): {c_baseline:.6f}")
    
    # Free HF model from GPU memory to avoid OOM when vLLM starts
    del hf_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return loss_plus, loss_minus, c_baseline, random_seed


def run_our_vllm(random_seed):
    """
    Run our vLLM-based LOZO implementation with same seed.
    
    Returns: (loss_plus, loss_minus, c)
    """
    print("\n" + "=" * 60)
    print("OUR vLLM IMPLEMENTATION")
    print("=" * 60)
    
    install_mocks()
    
    model_name = "facebook/opt-2.7b"
    rank_r = 8
    zo_eps = 1e-3
    
    # Load HF model on CPU (for controller)
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
        max_lora_rank=rank_r,
        max_loras=2,
        gpu_memory_utilization=0.4,
    )
    
    config = LOZOConfig(rank=rank_r, eps=zo_eps, lr=1e-7, step_interval=100)
    controller = LOZOController(hf_model, config)
    
    temp_lora = TempLoRARuntime(rank=rank_r, num_layers=num_layers)
    temp_lora.register_slots()
    
    scorer = VLLMScorer(llm, tokenizer)
    
    # Sample directions with same seed
    directions_2d, directions_1d = controller.sample_direction(random_seed)
    
    # Build LoRA tensors
    plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
    minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
    temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
    
    # Compute losses
    prompts = ["The movie was great"]
    loss_plus, loss_minus = scorer.score_plus_minus(prompts, temp_lora)
    
    print(f"Loss+ (vLLM): {loss_plus:.6f}")
    print(f"Loss- (vLLM): {loss_minus:.6f}")
    
    c_vllm = controller.compute_c(loss_plus, loss_minus)
    print(f"c (vLLM): {c_vllm:.6f}")
    
    temp_lora.cleanup()
    
    return loss_plus, loss_minus, c_vllm


def main():
    print("Full LOZO Baseline Alignment Test")
    print("=" * 60)
    
    # Run baseline
    loss_plus_baseline, loss_minus_baseline, c_baseline, random_seed = run_lozo_baseline()
    
    # Run our vLLM with same seed
    loss_plus_vllm, loss_minus_vllm, c_vllm = run_our_vllm(random_seed)
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    print(f"Random seed: {random_seed}")
    print()
    print(f"Loss+:")
    print(f"  Baseline: {loss_plus_baseline:.6f}")
    print(f"  vLLM:     {loss_plus_vllm:.6f}")
    print(f"  Diff:     {abs(loss_plus_baseline - loss_plus_vllm):.6f}")
    print()
    print(f"Loss-:")
    print(f"  Baseline: {loss_minus_baseline:.6f}")
    print(f"  vLLM:     {loss_minus_vllm:.6f}")
    print(f"  Diff:     {abs(loss_minus_baseline - loss_minus_vllm):.6f}")
    print()
    print(f"c:")
    print(f"  Baseline: {c_baseline:.6f}")
    print(f"  vLLM:     {c_vllm:.6f}")
    print(f"  Diff:     {abs(c_baseline - c_vllm):.6f}")
    print(f"  Ratio:    {c_vllm / c_baseline:.6f}")
    
    # Check alignment
    loss_diff = abs(loss_plus_baseline - loss_plus_vllm) + abs(loss_minus_baseline - loss_minus_vllm)
    c_ratio = c_vllm / c_baseline
    
    print()
    if loss_diff < 0.05 and abs(c_ratio - 1.0) < 0.05:
        print("✅ PASS: vLLM implementation aligned with LOZO baseline")
        return True
    else:
        print("❌ FAIL: vLLM implementation NOT aligned with baseline")
        print(f"  Loss difference too large: {loss_diff:.6f}")
        print(f"  c ratio off: {c_ratio:.6f}")
        return False


if __name__ == "__main__":
    main()