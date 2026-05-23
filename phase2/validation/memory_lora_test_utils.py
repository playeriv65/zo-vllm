"""Utilities for memory LoRA regression tests."""

import json
import os
import shutil
import tempfile
from typing import Dict, Iterable

import torch
from safetensors.torch import save_file


def _module_path(layer_idx: int, proj: str) -> str:
    if proj in {"q_proj", "k_proj", "v_proj", "out_proj"}:
        return f"model.decoder.layers.{layer_idx}.self_attn.{proj}"
    return f"model.decoder.layers.{layer_idx}.{proj}"


def _module_shape(proj: str) -> tuple[int, int]:
    hidden_size = 2560
    ffn_size = 10240
    if proj == "fc1":
        return ffn_size, hidden_size
    if proj == "fc2":
        return hidden_size, ffn_size
    return hidden_size, hidden_size


def build_lora_tensors(
    seed: int,
    rank: int,
    layers: Iterable[int],
    projs: Iterable[str],
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensors: Dict[str, torch.Tensor] = {}
    for layer_idx in layers:
        for proj in projs:
            out_features, in_features = _module_shape(proj)
            module_path = _module_path(layer_idx, proj)
            prefix = f"base_model.model.{module_path}"
            tensors[f"{prefix}.lora_A.weight"] = torch.randn(
                rank,
                in_features,
                generator=generator,
                dtype=torch.float16,
            )
            tensors[f"{prefix}.lora_B.weight"] = torch.randn(
                out_features,
                rank,
                generator=generator,
                dtype=torch.float16,
            )
    return tensors


def build_lora_config(rank: int, projs: Iterable[str]) -> dict:
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "facebook/opt-2.7b",
        "bias": "none",
        "exclude_modules": [],
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": float(rank),
        "lora_dropout": 0.0,
        "megatron_core": "megatron.core",
        "megatron_config": None,
        "modules_to_save": None,
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": list(projs),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }


def write_lora_to_file(lora_id: int, config: dict, tensors: Dict[str, torch.Tensor]) -> str:
    path = tempfile.mkdtemp(prefix=f"file_lora_{lora_id}_")
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(config, f)
    save_file(
        {name: tensor.cpu().contiguous() for name, tensor in tensors.items()},
        os.path.join(path, "adapter_model.safetensors"),
    )
    return path


def cleanup_lora_file(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
