"""
Module name mapping between HF, LoRA, and vLLM.

HF module name:
    model.decoder.layers.8.self_attn.q_proj

LoRA adapter key:
    base_model.model.model.decoder.layers.8.self_attn.q_proj.lora_A.weight

vLLM weight update name:
    model.decoder.layers.8.self_attn.q_proj.weight
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ModuleSpec:
    hf_module_name: str
    lora_module_name: str
    vllm_weight_name: str


def get_opt_module_specs(layers: List[int], projs: List[str]) -> List[ModuleSpec]:
    """
    Get module specs for OPT model.

    Args:
        layers: List of layer indices (e.g., [8, 9, 10])
        projs: List of projection/module names (e.g., ["fc1", "fc2"])

    Returns:
        List of ModuleSpec
    """
    specs = []
    for layer_idx in layers:
        for proj in projs:
            # fc1, fc2 are directly under layer, not under self_attn
            base = f"model.decoder.layers.{layer_idx}.{proj}"
            specs.append(ModuleSpec(
                hf_module_name=base,
                lora_module_name=base,
                vllm_weight_name=f"{base}.weight",
            ))
    return specs


def get_lora_target_modules(specs: List[ModuleSpec]) -> List[str]:
    """Get target_modules for LoRA config."""
    return [spec.lora_module_name for spec in specs]


def hf_name_to_vllm_name(hf_name: str, specs: List[ModuleSpec]) -> str:
    """Convert HF module name to vLLM weight name."""
    for spec in specs:
        if spec.hf_module_name == hf_name:
            return spec.vllm_weight_name
    raise ValueError(f"No matching vLLM name for HF module: {hf_name}")