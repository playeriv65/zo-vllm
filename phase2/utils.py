"""
Utilities for building LoRA tensors and writing to files.
"""

import os
import json
import torch
from typing import Dict, List
from safetensors.torch import save_file

MODEL_NAME = "facebook/opt-2.7b"
HIDDEN_SIZE = 2560


def build_lora_tensors(
    seed: int,
    rank: int,
    layers: List[int],
    projs: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Build LoRA tensors (PEFT format).
    
    Args:
        seed: Random seed for reproducibility
        rank: LoRA rank
        layers: List of layer indices (e.g., [8, 9, 10])
        projs: List of projection names (e.g., ["q_proj", "v_proj"])
    
    Returns:
        tensors: Dict of tensor names to tensors (PEFT format)
    """
    torch.manual_seed(seed)
    tensors = {}
    
    for layer_idx in layers:
        for proj in projs:
            module_path = f"model.decoder.layers.{layer_idx}.self_attn.{proj}"
            lora_A = torch.randn(rank, HIDDEN_SIZE, dtype=torch.float16)
            lora_B = torch.randn(HIDDEN_SIZE, rank, dtype=torch.float16)
            tensors[f"{module_path}.lora_A.weight"] = lora_A
            tensors[f"{module_path}.lora_B.weight"] = lora_B
    
    return tensors


def build_lora_config(rank: int, target_modules: List[str]) -> dict:
    """
    Build adapter_config.json content.
    
    Args:
        rank: LoRA rank
        target_modules: List of target module names
    
    Returns:
        config: Dict for adapter_config.json
    """
    return {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank,
        "target_modules": target_modules,
        "bias": "none",
        "base_model_name_or_path": MODEL_NAME,
    }


def write_lora_to_file(
    lora_id: int,
    config: dict,
    tensors: Dict[str, torch.Tensor],
    base_dir: str = "/tmp",
) -> str:
    """
    Write LoRA to real files (for baseline comparison).
    
    Args:
        lora_id: LoRA ID
        config: adapter_config.json content
        tensors: LoRA tensors
        base_dir: Base directory for files
    
    Returns:
        path: Directory path containing adapter files
    """
    dir_path = os.path.join(base_dir, f"test_lora_{lora_id}")
    os.makedirs(dir_path, exist_ok=True)
    
    # Write config
    config_path = os.path.join(dir_path, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Write tensors
    safetensors_path = os.path.join(dir_path, "adapter_model.safetensors")
    save_file(tensors, safetensors_path)
    
    return dir_path


def cleanup_lora_file(path: str):
    """Remove LoRA file directory."""
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)