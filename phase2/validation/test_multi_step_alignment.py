"""
Multi-Step LOZO Alignment - Compare vLLM LOZO and HF LOZO Baseline over multiple steps.

This script runs a 5-step LOZO training loop on both HF (baseline) and vLLM (our implementation)
to verify if losses, gradient estimators c, and updated model weights align perfectly
across multiple training steps.
"""

import os
import sys
import torch
import numpy as np
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

# Set cache directories
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from zo_vllm.core.lora_scope import DEFAULT_TRANSFORMER_TARGET_MODULES
from zo_vllm.core.lora_runtime import LoRAUpdateRuntime
from zo_vllm.experiment.scoring.generate_scorer import VLLMScorer
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.training.direction import (
    LOZOFastDirectionProvider,
    build_lora_runtime_tensors,
    compute_projected_grad,
)
from zo_vllm.training.model_metadata import build_lora_param_metadata_from_hf_model


def run_hf_baseline_multi_steps(model_name, prompt, rank_r, zo_eps, lr, steps):
    print("\n" + "=" * 60)
    print("RUNNING HUGGINGFACE BASELINE (MULTIPLE STEPS)")
    print("=" * 60)

    # Load model on GPU
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    named_parameters_to_optim = []
    for name, param in hf_model.named_parameters():
        if param.requires_grad:
            if "embed" in name or param.ndim == 1:
                continue
            named_parameters_to_optim.append((name, param))

    print(f"Optimizing {len(named_parameters_to_optim)} 2D linear parameters.")

    trajectory = []
    v_cache = {}

    # Run loop
    for step in range(steps):
        # We simulate nu = 100, meaning V is cached for the whole 5 steps
        # Random seed generation for this step
        np.random.seed(42 + step)
        random_seed = np.random.randint(1000000000)

        # Step A: Perturb with +eps
        torch.manual_seed(random_seed)
        for name, param in named_parameters_to_optim:
            if step == 0:
                v = torch.randn(
                    param.data.size(1),
                    rank_r,
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
                v_cache[name] = v
            else:
                v = v_cache[name]
            u = torch.randn(
                param.data.size(0),
                rank_r,
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data = param.data + (u @ v.t()) * zo_eps

        with torch.no_grad():
            outputs_plus = hf_model(**inputs, labels=inputs["input_ids"])
            loss_plus = outputs_plus.loss.item()

        # Step B: Perturb with -eps (reset then apply -2)
        torch.manual_seed(random_seed)
        for name, param in named_parameters_to_optim:
            v = v_cache[name]
            u = torch.randn(
                param.data.size(0),
                rank_r,
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data = param.data - 2 * (u @ v.t()) * zo_eps

        with torch.no_grad():
            outputs_minus = hf_model(**inputs, labels=inputs["input_ids"])
            loss_minus = outputs_minus.loss.item()

        # Step C: Compute gradient factor c and reset parameters
        c = (loss_plus - loss_minus) / (2.0 * zo_eps)

        torch.manual_seed(random_seed)
        for name, param in named_parameters_to_optim:
            v = v_cache[name]
            u = torch.randn(
                param.data.size(0),
                rank_r,
                device=param.data.device,
                dtype=param.data.dtype,
            )
            # Reset to original W
            param.data = param.data + (u @ v.t()) * zo_eps

            # Apply gradient update to the baseline weights
            param.data = param.data - lr * c * (u @ v.t())

        # Evaluate updated base model loss
        with torch.no_grad():
            outputs_base = hf_model(**inputs, labels=inputs["input_ids"])
            loss_base = outputs_base.loss.item()

        print(
            f"Step {step}: loss+={loss_plus:.6f}, loss-={loss_minus:.6f}, c={c:.6f}, base_loss={loss_base:.6f}"
        )
        trajectory.append(
            {
                "step": step,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "c": c,
                "loss_base": loss_base,
            }
        )

    # Free HF model from GPU
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    return trajectory


def run_our_vllm_multi_steps(model_name, prompt, rank_r, zo_eps, lr, steps):
    print("\n" + "=" * 60)
    print("RUNNING OUR vLLM IMPLEMENTATION (MULTIPLE STEPS)")
    print("=" * 60)

    # Load CPU-only model for direction metadata
    hf_model_cpu = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu"
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
        lora_target_modules=list(DEFAULT_TRANSFORMER_TARGET_MODULES),
        gpu_memory_utilization=0.4,
    )

    direction_provider = LOZOFastDirectionProvider(
        param_metadata=build_lora_param_metadata_from_hf_model(hf_model_cpu),
        rank=rank_r,
        nu=100,
        random_device="cpu",
    )

    lora_runtime = LoRAUpdateRuntime(
        rank=rank_r,
        num_layers=num_layers,
        target_modules=list(DEFAULT_TRANSFORMER_TARGET_MODULES),
    )
    lora_runtime.register_slots()

    scorer = VLLMScorer(llm, tokenizer)
    weight_sync = WeightSync(llm, num_layers=num_layers)

    trajectory = []
    prompts = [prompt]

    # Run loop
    for step in range(steps):
        np.random.seed(42 + step)
        random_seed = np.random.randint(1000000000)

        # Sample directions (keep the old explicit step alignment).
        direction_provider.step = step
        directions_2d = direction_provider.sample_direction(random_seed)

        # Build LoRA tensors
        plus_A, plus_B = build_lora_runtime_tensors(
            directions_2d,
            eps=zo_eps,
            sign=+1,
        )
        minus_A, minus_B = build_lora_runtime_tensors(
            directions_2d,
            eps=zo_eps,
            sign=-1,
        )

        # Update LoRA slots
        lora_runtime.update_plus_minus(plus_A, plus_B, minus_A, minus_B)

        # Compute loss
        loss_plus, loss_minus = scorer.score_plus_minus(prompts, lora_runtime)

        # Compute c
        c = compute_projected_grad(loss_plus, loss_minus, eps=zo_eps)

        # Update vLLM weights directly
        weight_sync.apply_lozo_update(
            directions_2d,
            c=c,
            lr=lr,
            weight_decay=0.0,
            precision="param",
        )

        # Record updated base model loss
        loss_base = scorer.score_base(prompts)

        print(
            f"Step {step}: loss+={loss_plus:.6f}, loss-={loss_minus:.6f}, c={c:.6f}, base_loss={loss_base:.6f}"
        )
        trajectory.append(
            {
                "step": step,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "c": c,
                "loss_base": loss_base,
            }
        )

    lora_runtime.cleanup()

    # Free memory
    del hf_model_cpu
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return trajectory


def main():
    model_name = "facebook/opt-2.7b"
    prompt = "The movie was great"
    rank_r = 8
    zo_eps = 1e-3
    lr = 1e-7
    steps = 5

    # Run HF Baseline
    hf_traj = run_hf_baseline_multi_steps(model_name, prompt, rank_r, zo_eps, lr, steps)

    # Run Our vLLM
    vllm_traj = run_our_vllm_multi_steps(model_name, prompt, rank_r, zo_eps, lr, steps)

    print("\n" + "=" * 60)
    print("TRAJECTORY COMPARISON")
    print("=" * 60)
    print(
        f"{'Step':<6} | {'Metric':<12} | {'HF Baseline':<12} | {'Our vLLM':<12} | {'Absolute Diff':<15}"
    )
    print("-" * 65)

    success = True
    for i in range(steps):
        hf_step = hf_traj[i]
        vllm_step = vllm_traj[i]

        for metric in ["loss_plus", "loss_minus", "c", "loss_base"]:
            val_hf = hf_step[metric]
            val_vllm = vllm_step[metric]
            diff = abs(val_hf - val_vllm)

            print(
                f"{i:<6} | {metric:<12} | {val_hf:<12.6f} | {val_vllm:<12.6f} | {diff:<15.6e}"
            )

            # Use 1e-4 as tolerance due to float16 precision variation
            if diff > 1e-4:
                success = False

    print("-" * 65)
    if success:
        print(
            "✅ SUCCESS: Multi-step trajectory matches between HF Baseline and our vLLM!"
        )
    else:
        print("❌ FAILURE: Discrepancy detected in multi-step trajectory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
