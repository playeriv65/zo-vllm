import os
import sys
import json
import subprocess
import gc
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

# Set path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from phase2.lozo_controller import LOZOController, LOZOConfig
from phase2.temp_lora_runtime import TempLoRARuntime
from phase2.vllm_scorer import VLLMScorer
from phase2.weight_sync import WeightSync
from phase2.memory_lora_loader import install_mocks


def main():
    print("=" * 70)
    # 1. Prepare inputs
    prompts = [
        "The movie was great",
        "It was a terrible show",
        "I loved the acting and the plot",
        "The book is much better than the film",
        "We had a wonderful experience at the theater"
    ]
    
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    prompts_file = os.path.join(results_dir, "input_batches.json")
    with open(prompts_file, "w") as f:
        json.dump(prompts, f, indent=2)
        
    print(f"[Coordinator] Saved {len(prompts)} prompts to {prompts_file}")
    
    # 2. Run Baseline Helper via baseline venv
    baseline_venv_python = os.path.join(project_root, "third_party", "LOZO", "large_models", ".venv", "bin", "python")
    helper_script = os.path.join(project_root, "phase2", "run_baseline_helper.py")
    
    print("[Coordinator] Launching LOZO baseline helper...")
    try:
        subprocess.run([baseline_venv_python, helper_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Coordinator] Error running baseline helper: {e}")
        sys.exit(1)
        
    # Read baseline trajectory
    baseline_file = os.path.join(results_dir, "baseline_trajectory.json")
    with open(baseline_file, "r") as f:
        baseline_trajectory = json.load(f)
        
    print(f"[Coordinator] Loaded baseline trajectory with {len(baseline_trajectory)} steps.")

    # 3. Run Our vLLM implementation in current venv
    print("\n" + "=" * 70)
    print("RUNNING OUR vLLM LOZO IMPLEMENTATION")
    print("=" * 70)
    
    install_mocks()
    
    model_name = "facebook/opt-2.7b"
    rank_r = 8
    zo_eps = 1e-3
    lr = 1e-7
    step_interval = 100
    
    # Load CPU model for controller
    hf_model_cpu = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    num_layers = hf_model_cpu.config.num_hidden_layers
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
    
    config = LOZOConfig(rank=rank_r, eps=zo_eps, lr=lr, step_interval=step_interval)
    controller = LOZOController(hf_model_cpu, config)
    
    temp_lora = TempLoRARuntime(rank=rank_r, num_layers=num_layers)
    temp_lora.register_slots()
    
    scorer = VLLMScorer(llm, tokenizer)
    weight_sync = WeightSync(llm, num_layers=num_layers)
    
    vllm_trajectory = []
    
    # Run loop using the exact same seeds from baseline
    for step in range(len(prompts)):
        step_data = baseline_trajectory[step]
        random_seed = step_data["seed"]
        
        # Sample directions
        controller.step = step
        directions_2d, directions_1d = controller.sample_direction(random_seed)
        
        # Build LoRA tensors
        plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
        minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
        
        # Update LoRA slots
        temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
        
        # Compute loss base, plus, minus
        loss_base = scorer.score_base([prompts[step]])
        loss_plus, loss_minus = scorer.score_plus_minus([prompts[step]], temp_lora)
        
        # Compute c
        c = controller.compute_c(loss_plus, loss_minus)
        
        # Update master weights on CPU
        updated_weights = controller.apply_update_to_master(
            directions_2d, directions_1d, c
        )
        
        # Sync updated weights to vLLM GPU
        weight_sync.sync(updated_weights)
        
        vllm_trajectory.append({
            "step": step,
            "seed": random_seed,
            "loss_base": float(loss_base),
            "loss_plus": float(loss_plus),
            "loss_minus": float(loss_minus),
            "c": float(c)
        })
        
        print(f"[vLLM Helper] Step {step}: seed={random_seed}, base={loss_base:.6f}, plus={loss_plus:.6f}, minus={loss_minus:.6f}, c={c:.6f}")
        
    # Cleanup
    temp_lora.cleanup()
    del hf_model_cpu
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save vLLM trajectory
    vllm_file = os.path.join(results_dir, "vllm_trajectory.json")
    with open(vllm_file, "w") as f:
        json.dump(vllm_trajectory, f, indent=2)
        
    # 4. Compare Trajectories
    print("\n" + "=" * 70)
    print("TRAJECTORY COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Step':<4} | {'Metric':<10} | {'Baseline':<12} | {'Our vLLM':<12} | {'Abs Diff':<12}")
    print("-" * 60)
    
    aligned = True
    for step in range(len(prompts)):
        b_data = baseline_trajectory[step]
        v_data = vllm_trajectory[step]
        
        for metric in ["loss_base", "loss_plus", "loss_minus", "c"]:
            val_b = b_data[metric]
            val_v = v_data[metric]
            diff = abs(val_b - val_v)
            
            print(f"{step:<4} | {metric:<10} | {val_b:<12.6f} | {val_v:<12.6f} | {diff:<12.6e}")
            
            # Using 1e-4 tolerance for float16 precision variances
            if diff > 1e-4:
                aligned = False
        print("-" * 60)
        
    if aligned:
        print("✅ SUCCESS: All losses and c coefficients are consistent within 1e-4 tolerance!")
    else:
        print("❌ FAILURE: Discrepancy detected between baseline and our vLLM implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
