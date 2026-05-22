"""
Temporary LoRA Runtime - Manages plus/minus LoRA slots for LOZO perturbation.

Uses the in-memory LoRA interface from memory_lora_loader.py.
Supports all 2D trainable parameters (Linear layers).
"""

from typing import Dict, List
import torch

from .memory_lora_loader import register_memory_lora_cpu, unregister_memory_lora
from .module_map import ModuleSpec


# OPT model hidden size
HIDDEN_SIZE = 2560


class TempLoRARuntime:
    """
    Manages two in-memory LoRA slots for LOZO plus/minus perturbation.
    
    Slots:
    - plus_id: Perturbation +eps*U @ V^T
    - minus_id: Perturbation -eps*U @ V^T
    
    Supports all 2D trainable parameters (Linear layers).
    """
    
    def __init__(
        self,
        rank: int,
        num_layers: int = 32,
        plus_id: int = 9001,
        minus_id: int = 9002,
    ):
        self.rank = rank
        self.num_layers = num_layers
        self.plus_id = plus_id
        self.minus_id = minus_id
        self.plus_name = "lozo_plus"
        self.minus_name = "lozo_minus"
        
        self.plus_path: str = ""
        self.minus_path: str = ""
        self._registered = False
    
    def _build_target_modules(self) -> List[str]:
        """
        Build list of target modules for LoRA.
        
        Returns short names (vLLM expects this format).
        """
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    
    def _build_empty_tensors(self) -> Dict[str, torch.Tensor]:
        """Build empty tensors for registration (PEFT format, only Linear modules)."""
        tensors = {}
        
        linear_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        
        for layer_idx in range(self.num_layers):
            for module_name in linear_modules:
                if module_name in ["fc1", "fc2"]:
                    module_path = f"model.decoder.layers.{layer_idx}.{module_name}"
                else:
                    module_path = f"model.decoder.layers.{layer_idx}.self_attn.{module_name}"
                
                # Determine dimensions
                if module_name == "fc1":
                    out_features, in_features = 10240, HIDDEN_SIZE
                elif module_name == "fc2":
                    out_features, in_features = HIDDEN_SIZE, 10240
                else:
                    out_features, in_features = HIDDEN_SIZE, HIDDEN_SIZE
                
                lora_A = torch.zeros(self.rank, in_features, dtype=torch.float16)
                lora_B = torch.zeros(out_features, self.rank, dtype=torch.float16)
                
                # PEFT format keys
                tensors[f"base_model.model.{module_path}.lora_A.weight"] = lora_A
                tensors[f"base_model.model.{module_path}.lora_B.weight"] = lora_B
        
        return tensors
    
    def register_slots(self):
        """
        Register two empty LoRA slots.
        
        Note: Actual tensors will be updated via update_plus_minus().
        """
        target_modules = self._build_target_modules()
        
        config = {
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
            "lora_alpha": float(self.rank),
            "lora_dropout": 0.0,
            "megatron_core": "megatron.core",
            "megatron_config": None,
            "modules_to_save": None,
            "r": self.rank,
            "rank_pattern": {},
            "revision": None,
            "target_modules": target_modules,
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }
        
        empty_tensors = self._build_empty_tensors()
        
        self.plus_path = register_memory_lora_cpu(
            self.plus_id,
            config,
            empty_tensors,
        )
        
        self.minus_path = register_memory_lora_cpu(
            self.minus_id,
            config.copy(),
            empty_tensors.copy(),
        )
        
        self._registered = True
    
    def build_peft_tensors(
        self,
        layer_to_A: Dict[str, torch.Tensor],
        layer_to_B: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert layer_to_A/B to PEFT format tensor dict.
        
        PEFT format: base_model.model.{module_path}.lora_A.weight
        
        Args:
            layer_to_A: {module_name: [rank, in_features]}  (HF weight names with .weight suffix)
            layer_to_B: {module_name: [out_features, rank]}
        
        Returns:
            tensors: {base_model.model.{module_path}.lora_A.weight: tensor, ...}
        """
        tensors = {}
        
        for name, A in layer_to_A.items():
            # Strip ".weight" suffix to get module path
            # e.g., "model.decoder.layers.0.self_attn.k_proj.weight" -> "model.decoder.layers.0.self_attn.k_proj"
            module_path = name
            if module_path.endswith(".weight"):
                module_path = module_path[:-7]  # Remove ".weight"
            
            tensors[f"base_model.model.{module_path}.lora_A.weight"] = A
            tensors[f"base_model.model.{module_path}.lora_B.weight"] = layer_to_B[name]
        
        return tensors
    
    def update_plus_minus(
        self,
        plus_A: Dict[str, torch.Tensor],
        plus_B: Dict[str, torch.Tensor],
        minus_A: Dict[str, torch.Tensor],
        minus_B: Dict[str, torch.Tensor],
    ):
        """
        Update both plus/minus LoRA slots.
        
        Note: This re-registers the slots with new tensors.
              In the future, we can optimize with in-place update.
        """
        if self._registered:
            unregister_memory_lora(self.plus_id)
            unregister_memory_lora(self.minus_id)
        
        target_modules = self._build_target_modules()
        
        config = {
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
            "lora_alpha": float(self.rank),
            "lora_dropout": 0.0,
            "megatron_core": "megatron.core",
            "megatron_config": None,
            "modules_to_save": None,
            "r": self.rank,
            "rank_pattern": {},
            "revision": None,
            "target_modules": target_modules,
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }
        
        plus_tensors = self.build_peft_tensors(plus_A, plus_B)
        minus_tensors = self.build_peft_tensors(minus_A, minus_B)
        
        self.plus_path = register_memory_lora_cpu(self.plus_id, config, plus_tensors)
        self.minus_path = register_memory_lora_cpu(self.minus_id, config.copy(), minus_tensors)
        
        self._registered = True
    
    def get_plus_request_info(self) -> tuple[str, int, str]:
        """Get (name, id, path) for plus LoRA."""
        return self.plus_name, self.plus_id, self.plus_path
    
    def get_minus_request_info(self) -> tuple[str, int, str]:
        """Get (name, id, path) for minus LoRA."""
        return self.minus_name, self.minus_id, self.minus_path
    
    def cleanup(self):
        """Unregister LoRA slots."""
        if self._registered:
            unregister_memory_lora(self.plus_id)
            unregister_memory_lora(self.minus_id)
            self._registered = False