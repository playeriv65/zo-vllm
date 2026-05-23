"""
Test 2: Speed Comparison - Measure performance difference between memory and file LoRA.

This test measures:
1. Registration/write time
2. Generation time (with LoRA loading overhead)
"""

import os
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from phase2.core.memory_lora_loader import register_memory_lora_cpu, clear_all_memory_loras
from phase2.validation.memory_lora_test_utils import (
    build_lora_config,
    build_lora_tensors,
    cleanup_lora_file,
    write_lora_to_file,
)

MODEL_NAME = "facebook/opt-2.7b"
RANK = 16
LAYERS = [8, 9, 10, 11, 12, 13, 14, 15]
PROJS = ["q_proj", "v_proj"]
GPU_MEMORY_UTILIZATION = 0.3

TEST_CONFIGS = [
    {"num_loras": 1, "repeats": 5},
    {"num_loras": 10, "repeats": 3},
    {"num_loras": 50, "repeats": 2},
]


def measure_speed(num_loras: int, repeats: int):
    """Measure speed for given number of LoRAs."""
    
    print(f"\n{'=' * 60}")
    print(f"Testing with {num_loras} LoRAs, {repeats} repeats")
    print(f"{'=' * 60}")
    
    # Create vLLM engine
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=RANK,
        max_loras=num_loras + 10,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        seed=42,
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    prompt = [{"prompt_token_ids": [2, 133, 1234]}]
    
    # Build tensors for all LoRAs
    all_tensors = []
    all_configs = []
    for i in range(num_loras):
        tensors = build_lora_tensors(seed=42 + i, rank=RANK, layers=LAYERS, projs=PROJS)
        config = build_lora_config(RANK, PROJS)
        all_tensors.append(tensors)
        all_configs.append(config)
    
    # === Test A: File-based LoRA ===
    print("\n--- File-based LoRA ---")
    
    file_write_times = []
    file_gen_times = []
    
    for repeat in range(repeats):
        print(f"Repeat {repeat + 1}/{repeats}")
        
        # Write files
        write_start = time.time()
        file_paths = []
        for i in range(num_loras):
            path = write_lora_to_file(i + 1, all_configs[i], all_tensors[i])
            file_paths.append(path)
        write_time = time.time() - write_start
        file_write_times.append(write_time)
        
        # Generate (first LoRA only to measure loading)
        gen_start = time.time()
        output = llm.generate(
            prompt,
            sampling_params,
            lora_request=LoRARequest("file_lora", 1, file_paths[0]),
        )
        gen_time = time.time() - gen_start
        file_gen_times.append(gen_time)
        
        # Cleanup
        for path in file_paths:
            cleanup_lora_file(path)
    
    avg_file_write = sum(file_write_times) / len(file_write_times)
    avg_file_gen = sum(file_gen_times) / len(file_gen_times)
    
    print(f"Average write time: {avg_file_write:.3f}s")
    print(f"Average gen time:   {avg_file_gen:.3f}s")
    
    # === Test B: Memory-based LoRA ===
    print("\n--- Memory-based LoRA ---")
    
    memory_reg_times = []
    memory_gen_times = []
    
    for repeat in range(repeats):
        print(f"Repeat {repeat + 1}/{repeats}")
        
        clear_all_memory_loras()
        
        # Register
        reg_start = time.time()
        memory_paths = []
        for i in range(num_loras):
            path = register_memory_lora_cpu(i + 1, all_configs[i], all_tensors[i])
            memory_paths.append(path)
        reg_time = time.time() - reg_start
        memory_reg_times.append(reg_time)
        
        # Generate
        gen_start = time.time()
        output = llm.generate(
            prompt,
            sampling_params,
            lora_request=LoRARequest("memory_lora", 1, memory_paths[0]),
        )
        gen_time = time.time() - gen_start
        memory_gen_times.append(gen_time)
    
    avg_memory_reg = sum(memory_reg_times) / len(memory_reg_times)
    avg_memory_gen = sum(memory_gen_times) / len(memory_gen_times)
    
    print(f"Average register time: {avg_memory_reg:.6f}s")
    print(f"Average gen time:      {avg_memory_gen:.3f}s")
    
    # === Comparison ===
    print("\n--- Speed Comparison ---")
    
    write_speedup = avg_file_write / avg_memory_reg if avg_memory_reg > 0 else float('inf')
    gen_diff = avg_file_gen - avg_memory_gen
    gen_diff_pct = abs(gen_diff) / avg_file_gen * 100
    
    print(f"Registration/Write:")
    print(f"  File:    {avg_file_write:.3f}s")
    print(f"  Memory:  {avg_memory_reg:.6f}s")
    print(f"  Speedup: {write_speedup:.1f}x")
    
    print(f"\nGeneration (first LoRA):")
    print(f"  File:    {avg_file_gen:.3f}s")
    print(f"  Memory:  {avg_memory_gen:.3f}s")
    print(f"  Diff:    {gen_diff:.3f}s ({gen_diff_pct:.1f}%)")
    
    clear_all_memory_loras()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "num_loras": num_loras,
        "file_write": avg_file_write,
        "memory_reg": avg_memory_reg,
        "write_speedup": write_speedup,
        "file_gen": avg_file_gen,
        "memory_gen": avg_memory_gen,
        "gen_diff_pct": gen_diff_pct,
    }


def test_speed():
    """Run speed tests with different configurations."""
    
    print("=" * 60)
    print("Test 2: Speed Comparison")
    print("=" * 60)
    print(f"Config: rank={RANK}, layers={LAYERS}, projs={PROJS}")
    
    results = []
    
    for config in TEST_CONFIGS:
        result = measure_speed(config["num_loras"], config["repeats"])
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print("\n| LoRAs | File Write | Memory Reg | Speedup | File Gen | Memory Gen | Gen Diff % |")
    print("|-------|------------|------------|---------|----------|------------|------------|")
    for r in results:
        print(f"| {r['num_loras']:5d} | {r['file_write']:10.3f}s | {r['memory_reg']:10.6f}s | {r['write_speedup']:7.1f}x | {r['file_gen']:8.3f}s | {r['memory_gen']:10.3f}s | {r['gen_diff_pct']:10.1f}% |")
    
    # Check expectations
    print("\n" + "=" * 60)
    all_passed = True
    for r in results:
        if r["write_speedup"] < 5:
            print(f"FAIL: {r['num_loras']} LoRAs write speedup < 5x (got {r['write_speedup']:.1f}x)")
            all_passed = False
        if r["gen_diff_pct"] > 10:
            print(f"INFO: {r['num_loras']} LoRAs gen time diff > 10% (got {r['gen_diff_pct']:.1f}%)")
    
    if all_passed:
        print("TEST PASSED: All speed requirements met!")
    else:
        print("TEST FAILED: Some requirements not met!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_speed()
    exit(0 if success else 1)
