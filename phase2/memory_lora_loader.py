"""
Memory LoRA Loader - Mock safetensors file reading to support in-memory LoRA tensors.

This module intercepts vLLM's LoRA loading process by mocking:
- safetensors.safe_open() - Return in-memory tensors
- os.path.isfile(), os.path.exists(), os.path.isabs() - Pretend memory paths exist
- builtins.open() - Return adapter_config.json from memory

Usage:
    path = register_memory_lora_cpu(lora_id, config, tensors)
    llm.generate(prompts, lora_request=LoRARequest("name", lora_id, path))
"""

import os
import json
import safetensors
import torch
from typing import Dict, Optional
from io import StringIO

MEMORY_LORA_REGISTRY: Dict[str, Dict] = {}
_MOCKS_INSTALLED = False

PATH_PREFIX_CPU = "/memory_lora_cpu"


class FakeSafeFile:
    """Mock safetensors.safe_open() return object."""
    
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors
    
    def keys(self):
        return self._tensors.keys()
    
    def get_tensor(self, key: str) -> torch.Tensor:
        return self._tensors[key]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# Save original functions
_original_safe_open = safetensors.safe_open
_original_isfile = os.path.isfile
_original_isabs = os.path.isabs
_original_exists = os.path.exists
_original_open = open


def register_memory_lora_cpu(
    lora_id: int,
    config: dict,
    tensors: Dict[str, torch.Tensor],
) -> str:
    """
    Register in-memory LoRA (CPU tensors).
    
    Args:
        lora_id: Unique LoRA integer ID (must match LoRARequest.lora_int_id)
        config: adapter_config.json content (dict)
        tensors: LoRA weight tensors (PEFT format, CPU tensors)
    
    Returns:
        path: Memory path for LoRARequest (e.g., "/memory_lora_cpu/1")
    """
    if not _MOCKS_INSTALLED:
        install_mocks()
    
    path = f"{PATH_PREFIX_CPU}/{lora_id}"
    
    # Ensure tensors are on CPU and contiguous
    cpu_tensors = {k: v.cpu().contiguous() for k, v in tensors.items()}
    
    MEMORY_LORA_REGISTRY[path] = {
        "config": config,
        "tensors": cpu_tensors,
        "safetensors_path": f"{path}/adapter_model.safetensors",
        "config_path": f"{path}/adapter_config.json",
    }
    
    return path


def unregister_memory_lora(lora_id: int) -> bool:
    """Remove registered memory LoRA."""
    path = f"{PATH_PREFIX_CPU}/{lora_id}"
    return MEMORY_LORA_REGISTRY.pop(path, None) is not None


def clear_all_memory_loras():
    """Clear all registered memory LoRAs."""
    MEMORY_LORA_REGISTRY.clear()


def list_memory_loras() -> Dict[int, str]:
    """List all registered memory LoRA IDs and paths."""
    result = {}
    for path in MEMORY_LORA_REGISTRY.keys():
        lora_id = int(path.split("/")[-1])
        result[lora_id] = path
    return result


# Mock functions
def _fake_safe_open(path, framework: str = "pt"):
    """Mock safetensors.safe_open()."""
    path_str = str(path)
    for reg_path, entry in MEMORY_LORA_REGISTRY.items():
        if path_str == entry["safetensors_path"]:
            return FakeSafeFile(entry["tensors"])
    return _original_safe_open(path, framework)


def _fake_isfile(path) -> bool:
    """Mock os.path.isfile()."""
    path_str = str(path)
    for reg_path, entry in MEMORY_LORA_REGISTRY.items():
        if path_str == entry["safetensors_path"] or path_str == entry["config_path"]:
            return True
    return _original_isfile(path)


def _fake_isabs(path) -> bool:
    """Mock os.path.isabs()."""
    path_str = str(path)
    if path_str.startswith(PATH_PREFIX_CPU):
        return True
    return _original_isabs(path)


def _fake_exists(path) -> bool:
    """Mock os.path.exists()."""
    # Handle both str and pathlib.Path
    path_str = str(path)
    if path_str.startswith(PATH_PREFIX_CPU):
        return True
    return _original_exists(path)


def _fake_open(path, *args, **kwargs):
    """Mock builtins.open() for adapter_config.json."""
    path_str = str(path)
    for reg_path, entry in MEMORY_LORA_REGISTRY.items():
        if path_str == entry["config_path"]:
            return StringIO(json.dumps(entry["config"]))
    return _original_open(path, *args, **kwargs)


def install_mocks():
    """Install mock functions (called automatically on first register)."""
    global _MOCKS_INSTALLED
    if _MOCKS_INSTALLED:
        return
    
    safetensors.safe_open = _fake_safe_open
    os.path.isfile = _fake_isfile
    os.path.isabs = _fake_isabs
    os.path.exists = _fake_exists
    import builtins
    builtins.open = _fake_open
    
    _MOCKS_INSTALLED = True


def uninstall_mocks():
    """Restore original functions."""
    global _MOCKS_INSTALLED
    
    safetensors.safe_open = _original_safe_open
    os.path.isfile = _original_isfile
    os.path.isabs = _original_isabs
    os.path.exists = _original_exists
    import builtins
    builtins.open = _original_open
    
    _MOCKS_INSTALLED = False