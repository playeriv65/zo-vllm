"""
Test 1: Alignment Verification - Compare memory LoRA vs file LoRA outputs.

This test verifies that in-memory LoRA produces identical outputs to file-based LoRA.
"""

import os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import time
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from memory_lora_loader import register_memory_lora_cpu, clear_all_memory_loras
from memory_lora_test_utils import (
    build_lora_config,
    build_lora_tensors,
    cleanup_lora_file,
    write_lora_to_file,
)

MODEL_NAME = "facebook/opt-2.7b"
RANK = 16
LAYERS = [8, 9, 10, 11, 12, 13, 14, 15]
PROJS = ["q_proj", "v_proj"]
SEED = 42


def test_alignment():
    """Test memory LoRA vs file LoRA output alignment."""
    
    print("=" * 60)
    print("Test 1: Alignment Verification")
    print("=" * 60)
    
    # Build tensors with fixed seed
    tensors = build_lora_tensors(SEED, RANK, LAYERS, PROJS)
    config = build_lora_config(RANK, PROJS)
    
    print(f"LoRA config: rank={RANK}, layers={LAYERS}, projs={PROJS}")
    print(f"Tensors: {len(tensors)} weights")
    
    # Create vLLM engine
    print("\nInitializing vLLM engine...")
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=RANK,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=0.5,
        seed=42,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        prompt_logprobs=5,
    )
    
    # Test prompt
    prompt = [{"prompt_token_ids": [2, 133, 1234, 5678, 9012]}]
    
    # === Test A: File-based LoRA ===
    print("\n--- Test A: File-based LoRA ---")
    
    file_path = write_lora_to_file(1, config, tensors)
    print(f"Written to: {file_path}")
    
    start_time = time.time()
    outputs_file = llm.generate(
        prompt,
        sampling_params,
        lora_request=LoRARequest("file_lora", 1, file_path),
    )
    file_gen_time = time.time() - start_time
    
    token_ids_file = outputs_file[0].outputs[0].token_ids
    logprobs_file = outputs_file[0].outputs[0].logprobs
    print(f"Output tokens: {token_ids_file}")
    print(f"Generation time: {file_gen_time:.3f}s")
    
    cleanup_lora_file(file_path)
    
    # === Test B: Memory-based LoRA ===
    print("\n--- Test B: Memory-based LoRA ---")
    
    clear_all_memory_loras()
    memory_path = register_memory_lora_cpu(1, config, tensors)
    print(f"Registered at: {memory_path}")
    
    start_time = time.time()
    outputs_memory = llm.generate(
        prompt,
        sampling_params,
        lora_request=LoRARequest("memory_lora", 1, memory_path),
    )
    memory_gen_time = time.time() - start_time
    
    token_ids_memory = outputs_memory[0].outputs[0].token_ids
    logprobs_memory = outputs_memory[0].outputs[0].logprobs
    print(f"Output tokens: {token_ids_memory}")
    print(f"Generation time: {memory_gen_time:.3f}s")
    
    # === Compare ===
    print("\n--- Comparison ---")
    
    # Check token_ids
    tokens_match = token_ids_file == token_ids_memory
    print(f"Token IDs match: {tokens_match}")
    if not tokens_match:
        print(f"  File:    {token_ids_file}")
        print(f"  Memory:  {token_ids_memory}")
    
    # Check logprobs (first position)
    if logprobs_file and logprobs_memory:
        first_logprobs_file = list(logprobs_file[0].values())[:3]
        first_logprobs_memory = list(logprobs_memory[0].values())[:3]
        logprobs_match = len(first_logprobs_file) == len(first_logprobs_memory)
        if logprobs_match:
            for lp_file, lp_mem in zip(first_logprobs_file, first_logprobs_memory):
                if lp_file.logprob != lp_mem.logprob:
                    logprobs_match = False
                    break
        print(f"Logprobs match (first position): {logprobs_match}")
        if not logprobs_match:
            print(f"  File:    {[(lp.decoded_token, lp.logprob) for lp in first_logprobs_file]}")
            print(f"  Memory:  {[(lp.decoded_token, lp.logprob) for lp in first_logprobs_memory]}")
    
    # Timing comparison
    print(f"\nGeneration time comparison:")
    print(f"  File:    {file_gen_time:.3f}s")
    print(f"  Memory:  {memory_gen_time:.3f}s")
    print(f"  Diff:    {abs(file_gen_time - memory_gen_time):.3f}s")
    
    # Result
    print("\n" + "=" * 60)
    if tokens_match:
        print("TEST PASSED: Output alignment verified!")
    else:
        print("TEST FAILED: Output mismatch!")
    print("=" * 60)
    
    clear_all_memory_loras()
    return tokens_match


if __name__ == "__main__":
    success = test_alignment()
    exit(0 if success else 1)
