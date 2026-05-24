"""
Weight Sync - Sync updated weights from LOZO controller to vLLM engine.

Uses vLLM's collective_rpc to directly update model weights in worker process.
For packed modules (qkv_proj), updates the entire packed weight directly.

Note: vLLM LoRA wrapper modules (e.g., MergedQKVParallelLinearWithLoRA) 
      don't have weight_loader. We need to access base_layer.weight directly.
"""

import os
import time
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


def apply_qkv_lowrank_update_to_weight_(
    target: torch.Tensor,
    q_direction: dict[str, torch.Tensor],
    k_direction: dict[str, torch.Tensor],
    v_direction: dict[str, torch.Tensor],
    *,
    c: float,
    lr: float,
    weight_decay: float,
    precision: str,
) -> bool:
    """Apply packed q/k/v updates with one batched matmul when possible."""
    if precision != "param":
        return False
    if target.dim() != 2 or target.shape[0] % 3 != 0:
        return False
    hidden_size = target.shape[0] // 3
    if target.shape[1] != hidden_size:
        return False

    directions = [q_direction, k_direction, v_direction]
    u_tensors = [item["U"] for item in directions]
    vt_tensors = [item.get("V_T") for item in directions]
    if any(v_t is None for v_t in vt_tensors):
        vt_tensors = [item["V"].T for item in directions]
    if any(tuple(u.shape) != (hidden_size, u_tensors[0].shape[1]) for u in u_tensors):
        return False
    if any(tuple(v_t.shape) != (u_tensors[0].shape[1], hidden_size) for v_t in vt_tensors):
        return False

    beta = 1.0 - lr * weight_decay
    u_batch = torch.stack(
        [
            u.to(device=target.device, dtype=target.dtype, non_blocking=True)
            for u in u_tensors
        ],
        dim=0,
    )
    vt_batch = torch.stack(
        [
            v_t.to(device=target.device, dtype=target.dtype, non_blocking=True)
            for v_t in vt_tensors
        ],
        dim=0,
    )
    target.view(3, hidden_size, hidden_size).baddbmm_(
        u_batch,
        vt_batch,
        beta=beta,
        alpha=-lr * c,
    )
    return True


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
        self.last_update_info = {}
    
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
        qkv_update_mode: str = "separate",
    ) -> None:
        """
        Apply the LOZO base-weight update directly inside the vLLM worker.

        This skips materializing full updated weights in the controller and
        skips copying those full tensors into vLLM. The vLLM base weights
        become the training master state.
        """
        if precision not in {"float32", "param"}:
            raise ValueError(f"unknown update precision: {precision}")
        if qkv_update_mode not in {"separate", "batched"}:
            raise ValueError(f"unknown qkv_update_mode: {qkv_update_mode}")
        if directions_1d:
            raise ValueError(
                "direct vLLM weight update only supports 2D LoRA-compatible "
                "parameters; use --weight-update copy for 1D/full scope"
            )
        profile_enabled = os.environ.get("VLLM_ZO_WEIGHT_PROFILE", "0") == "1"

        def update_weights_on_worker(worker):
            profile_t0 = time.perf_counter()
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
            profile_after_group = time.perf_counter()

            for vllm_name, proj_dict in qkv_directions.items():
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = base_layer.weight
                hidden_size = param.data.shape[0] // 3

                if (
                    qkv_update_mode == "batched"
                    and all(key in proj_dict for key in ("q", "k", "v"))
                ):
                    q_name, q_direction = proj_dict["q"]
                    _k_name, k_direction = proj_dict["k"]
                    _v_name, v_direction = proj_dict["v"]
                    if apply_qkv_lowrank_update_to_weight_(
                        param.data,
                        q_direction,
                        k_direction,
                        v_direction,
                        c=c,
                        lr=lr,
                        weight_decay=(
                            weight_decay if should_apply_weight_decay(q_name) else 0.0
                        ),
                        precision=precision,
                    ):
                        continue

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

            profile_after_qkv = time.perf_counter()
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

            profile_after_other = time.perf_counter()
            if sync_device:
                torch.cuda.synchronize()
            profile_after_sync = time.perf_counter()
            if profile_enabled:
                return {
                    "profile_s": {
                        "group": profile_after_group - profile_t0,
                        "qkv_update": profile_after_qkv - profile_after_group,
                        "other_update": profile_after_other - profile_after_qkv,
                        "sync": profile_after_sync - profile_after_other,
                        "total_worker": profile_after_sync - profile_t0,
                    },
                    "num_qkv_modules": int(len(qkv_directions)),
                    "num_other_modules": int(len(other_directions)),
                }
            return None

        results = self.llm.collective_rpc(update_weights_on_worker)
        self.last_update_info = {"workers": results} if profile_enabled else {}
        return self.last_update_info
