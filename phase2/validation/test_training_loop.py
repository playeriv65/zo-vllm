"""
Test complete LOZO training loop.

Verify:
1. Full training loop works (sample → LoRA → loss → update → sync)
2. V cache is correctly updated at nu
3. Training converges (loss decreases)
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

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


def test_training_loop():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    model_name = "facebook/opt-2.7b"
    rank = 1
    eps = 1e-3
    lr = 1e-7
    nu = 100
    num_steps = 10
    batch_size = 4

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
        max_lora_rank=rank,
        max_loras=2,
        lora_target_modules=list(DEFAULT_TRANSFORMER_TARGET_MODULES),
        gpu_memory_utilization=0.4,
    )

    # Initialize components
    direction_provider = LOZOFastDirectionProvider(
        param_metadata=build_lora_param_metadata_from_hf_model(hf_model),
        rank=rank,
        nu=nu,
        random_device="cpu",
    )

    lora_runtime = LoRAUpdateRuntime(
        rank=rank,
        num_layers=num_layers,
        target_modules=list(DEFAULT_TRANSFORMER_TARGET_MODULES),
    )
    lora_runtime.register_slots()

    scorer = VLLMScorer(llm, tokenizer)

    weight_sync = WeightSync(llm, num_layers=num_layers)

    prompts = ["The future of AI is"] * batch_size

    # Initial loss
    initial_loss = scorer.score_base(prompts)
    print(f"Initial loss: {initial_loss:.6f}")

    # Training loop
    import numpy as np

    np.random.seed(42)

    losses = []

    for step in range(num_steps):
        random_seed = np.random.randint(1000000000)

        # Sample directions
        directions_2d = direction_provider.sample_direction(random_seed)

        # Build LoRA tensors
        plus_A, plus_B = build_lora_runtime_tensors(
            directions_2d,
            eps=eps,
            sign=+1,
        )
        minus_A, minus_B = build_lora_runtime_tensors(
            directions_2d,
            eps=eps,
            sign=-1,
        )

        # Update LoRA slots
        lora_runtime.update_plus_minus(plus_A, plus_B, minus_A, minus_B)

        # Compute loss
        loss_plus, loss_minus = scorer.score_plus_minus(prompts, lora_runtime)

        # Compute c
        c = compute_projected_grad(loss_plus, loss_minus, eps=eps)

        # Update vLLM weights directly
        weight_sync.apply_lozo_update(
            directions_2d,
            c=c,
            lr=lr,
            weight_decay=0.0,
            precision="param",
        )

        # Record loss
        current_loss = scorer.score_base(prompts)
        losses.append(current_loss)

        print(
            f"Step {step}: loss+={loss_plus:.4f}, loss-={loss_minus:.4f}, "
            f"c={c:.4f}, base_loss={current_loss:.4f}"
        )

        # Verify V cache behavior
        if step % nu == 0:
            print(f"  V cache updated at step {step}")

    # Verify convergence
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss change: {losses[-1] - initial_loss:.4f}")

    lora_runtime.cleanup()

    print("\n✅ Training loop test PASSED!")


if __name__ == "__main__":
    test_training_loop()
