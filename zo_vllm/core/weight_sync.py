"""
Weight Sync - Sync updated weights from LOZO controller to vLLM engine.

Uses vLLM's collective_rpc to directly update model weights in worker process.
For packed modules (qkv_proj), updates the entire packed weight directly.

Note: vLLM LoRA wrapper modules (e.g., MergedQKVParallelLinearWithLoRA) 
      don't have weight_loader. We need to access base_layer.weight directly.
"""

from typing import Dict
import torch


def unwrap_lora_module(module):
    """
    If module is a vLLM LoRA wrapper, return its base layer.
    Otherwise return module itself.
    """
    if hasattr(module, "base_layer"):
        return module.base_layer
    return module


def should_apply_weight_decay(name: str) -> bool:
    return (
        "bias" not in name
        and "layer_norm" not in name
        and "layernorm" not in name
    )


def apply_lowrank_update_to_weight_(
    target: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    c: float,
    lr: float,
    weight_decay: float,
    precision: str,
) -> None:
    """Apply W <- W * (1 - lr * wd) - lr * c * U @ V.T in place."""
    if precision == "float32":
        U_gpu = U.to(device=target.device, non_blocking=True).float()
        V_gpu = V.to(device=target.device, non_blocking=True).float()
        beta = 1.0 - lr * weight_decay
        updated = torch.addmm(
            target.float(),
            U_gpu,
            V_gpu.T,
            beta=beta,
            alpha=-lr * c,
        )
        target.copy_(updated.to(dtype=target.dtype))
        return

    if precision == "param":
        if weight_decay:
            target.mul_(1.0 - lr * weight_decay)
        U_gpu = U.to(device=target.device, dtype=target.dtype, non_blocking=True)
        V_gpu = V.to(device=target.device, dtype=target.dtype, non_blocking=True)
        target.addmm_(U_gpu, V_gpu.T, alpha=-lr * c)
        return

    raise ValueError(f"unknown update precision: {precision}")


class WeightSync:
    """
    Sync weights from external trainer (LOZO controller) to vLLM engine.
    
    Uses collective_rpc to call worker's model parameter update directly.
    For packed modules (qkv_proj), updates the entire packed weight tensor.
    """
    
    def __init__(self, llm, num_layers: int = 32):
        self.llm = llm
        self.num_layers = num_layers
        self.hf_to_vllm_mapping = self._build_mapping()
    
    def _build_mapping(self) -> Dict[str, str]:
        """
        Build HF→vLLM parameter mapping for Linear weights only.
        
        For q/k/v_proj: map to packed qkv_proj.weight (update entire packed weight)
        For other modules: direct 1:1 mapping
        
        Returns:
            mapping: {hf_name: vllm_name}
        """
        mapping = {}
        
        for layer_idx in range(self.num_layers):
            # Packed module: qkv_proj
            # All q/k/v_proj weights map to the same qkv_proj.weight
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                hf_name = f"model.decoder.layers.{layer_idx}.self_attn.{proj_name}.weight"
                vllm_name = f"model.decoder.layers.{layer_idx}.self_attn.qkv_proj.weight"
                mapping[hf_name] = vllm_name
            
            # Non-packed modules: direct mapping
            for module_suffix in [
                "self_attn.out_proj.weight",
                "fc1.weight",
                "fc2.weight",
            ]:
                hf_name = f"model.decoder.layers.{layer_idx}.{module_suffix}"
                mapping[hf_name] = hf_name
        
        return mapping
    
    def sync(self, updated_weights_hf_names: Dict[str, torch.Tensor]):
        """
        Sync updated weights to vLLM engine.
        
        Args:
            updated_weights_hf_names: Updated weight tensors with HF module names
                (only Linear layer weights, no embeddings/1D params)
        """
        def update_weights_on_worker(worker):
            model = worker.model_runner.model
            
            # Group q/k/v_proj weights by qkv_proj
            qkv_weights = {}  # {vllm_name: {proj_key: tensor}}
            other_weights = {}
            
            for hf_name, tensor in updated_weights_hf_names.items():
                vllm_name = self.hf_to_vllm_mapping.get(hf_name)
                if vllm_name is None:
                    raise ValueError(f"No mapping for HF parameter: {hf_name}")
                
                # Check if it's a packed module
                if "qkv_proj.weight" in vllm_name:
                    # Extract which proj (q/k/v) from hf_name
                    if ".q_proj.weight" in hf_name:
                        proj_key = "q"
                    elif ".k_proj.weight" in hf_name:
                        proj_key = "k"
                    elif ".v_proj.weight" in hf_name:
                        proj_key = "v"
                    else:
                        raise ValueError(f"Unknown proj type: {hf_name}")
                    
                    if vllm_name not in qkv_weights:
                        qkv_weights[vllm_name] = {}
                    qkv_weights[vllm_name][proj_key] = tensor
                else:
                    other_weights[vllm_name] = tensor
            
            # Update packed qkv_proj weights.
            for vllm_name, proj_dict in qkv_weights.items():
                # Get module and unwrap LoRA wrapper
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = base_layer.weight
                
                # OPT packs q/k/v as [3 * hidden_size, hidden_size].
                # Copy directly into slices to avoid cloning the full packed
                # tensor for every layer on every training step.
                hidden_size = param.data.shape[0] // 3
                
                for proj_key, tensor in proj_dict.items():
                    tensor_gpu = tensor.to(
                        device=param.device,
                        dtype=param.dtype,
                        non_blocking=True,
                    )
                    if proj_key == "q":
                        param.data[0:hidden_size, :].copy_(tensor_gpu)
                    elif proj_key == "k":
                        param.data[hidden_size:2 * hidden_size, :].copy_(tensor_gpu)
                    elif proj_key == "v":
                        param.data[2 * hidden_size:3 * hidden_size, :].copy_(tensor_gpu)
            
            # Update other weights (non-packed)
            for vllm_name, tensor in other_weights.items():
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = base_layer.weight
                
                tensor_gpu = tensor.to(
                    device=param.device,
                    dtype=param.dtype,
                    non_blocking=True,
                )
                param.data.copy_(tensor_gpu)
            
            torch.cuda.synchronize()
        
        self.llm.collective_rpc(update_weights_on_worker)

    def apply_lozo_update(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        directions_1d: Dict[str, torch.Tensor],
        *,
        c: float,
        lr: float,
        weight_decay: float = 0.0,
        precision: str = "float32",
        sync_device: bool = True,
    ) -> None:
        """
        Apply the LOZO base-weight update directly inside the vLLM worker.

        This skips materializing full updated weights in the controller and
        skips copying those full tensors into vLLM. The vLLM base weights
        become the training master state.
        """
        if precision not in {"float32", "param"}:
            raise ValueError(f"unknown update precision: {precision}")
        if directions_1d:
            raise ValueError(
                "direct vLLM weight update only supports 2D LoRA-compatible "
                "parameters; use --weight-update copy for 1D/full scope"
            )

        def update_weights_on_worker(worker):
            model = worker.model_runner.model

            qkv_directions = {}
            other_directions = {}

            for hf_name, direction in directions_2d.items():
                vllm_name = self.hf_to_vllm_mapping.get(hf_name)
                if vllm_name is None:
                    raise ValueError(f"No mapping for HF parameter: {hf_name}")

                if "qkv_proj.weight" in vllm_name:
                    if ".q_proj.weight" in hf_name:
                        proj_key = "q"
                    elif ".k_proj.weight" in hf_name:
                        proj_key = "k"
                    elif ".v_proj.weight" in hf_name:
                        proj_key = "v"
                    else:
                        raise ValueError(f"Unknown proj type: {hf_name}")

                    if vllm_name not in qkv_directions:
                        qkv_directions[vllm_name] = {}
                    qkv_directions[vllm_name][proj_key] = (hf_name, direction)
                else:
                    other_directions[vllm_name] = (hf_name, direction)

            for vllm_name, proj_dict in qkv_directions.items():
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = base_layer.weight
                hidden_size = param.data.shape[0] // 3

                for proj_key, (hf_name, direction) in proj_dict.items():
                    if proj_key == "q":
                        target = param.data[0:hidden_size, :]
                    elif proj_key == "k":
                        target = param.data[hidden_size:2 * hidden_size, :]
                    elif proj_key == "v":
                        target = param.data[2 * hidden_size:3 * hidden_size, :]
                    else:
                        raise ValueError(f"Unknown proj type: {proj_key}")

                    apply_lowrank_update_to_weight_(
                        target,
                        direction["U"],
                        direction["V"],
                        c=c,
                        lr=lr,
                        weight_decay=(
                            weight_decay if should_apply_weight_decay(hf_name) else 0.0
                        ),
                        precision=precision,
                    )

            for vllm_name, (hf_name, direction) in other_directions.items():
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = base_layer.weight

                apply_lowrank_update_to_weight_(
                    param.data,
                    direction["U"],
                    direction["V"],
                    c=c,
                    lr=lr,
                    weight_decay=(
                        weight_decay if should_apply_weight_decay(hf_name) else 0.0
                    ),
                    precision=precision,
                )

            if sync_device:
                torch.cuda.synchronize()

        self.llm.collective_rpc(update_weights_on_worker)
