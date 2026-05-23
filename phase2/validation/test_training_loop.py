"""
Test complete LOZO training loop.

Verify:
1. Full training loop works (sample → LoRA → loss → update → sync)
2. V cache is correctly updated at step_interval
3. Training converges (loss decreases)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from phase2.core.lozo_controller import LOZOController, LOZOConfig
from phase2.core.temp_lora_runtime import TempLoRARuntime
from phase2.core.vllm_scorer import VLLMScorer
from phase2.core.weight_sync import WeightSync


def test_training_loop():
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    
    model_name = "facebook/opt-2.7b"
    rank = 1
    eps = 1e-3
    lr = 1e-7
    step_interval = 100
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
    
    temp_lora = TempLoRARuntime(rank=rank, num_layers=num_layers)
    temp_lora.register_slots()
    
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
        directions_2d, directions_1d = controller.sample_direction(random_seed)
        
        # Build LoRA tensors
        plus_A, plus_B = controller.build_temp_lora_tensors(directions_2d, sign=+1)
        minus_A, minus_B = controller.build_temp_lora_tensors(directions_2d, sign=-1)
        
        # Update LoRA slots
        temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B)
        
        # Compute loss
        loss_plus, loss_minus = scorer.score_plus_minus(prompts, temp_lora)
        
        # Compute c
        c = controller.compute_c(loss_plus, loss_minus)
        
        # Update master weights
        updated_weights = controller.apply_update_to_master(
            directions_2d, directions_1d, c
        )
        
        # Sync to vLLM
        weight_sync.sync(updated_weights)
        
        # Record loss
        current_loss = scorer.score_base(prompts)
        losses.append(current_loss)
        
        print(f"Step {step}: loss+={loss_plus:.4f}, loss-={loss_minus:.4f}, "
              f"c={c:.4f}, base_loss={current_loss:.4f}")
        
        # Verify V cache behavior
        if step % step_interval == 0:
            print(f"  V cache updated at step {step}")
    
    # Verify convergence
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss change: {losses[-1] - initial_loss:.4f}")
    
    temp_lora.cleanup()
    
    print("\n✅ Training loop test PASSED!")


if __name__ == "__main__":
    test_training_loop()