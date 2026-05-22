"""
LOZO Controller - Manages master weights, U/V sampling, and LOZO updates.

This module handles all LOZO algorithm logic, independent of vLLM.
Aligned with LOZO baseline (third_party/LOZO/large_models/LOZOtrainer.py).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LOZOConfig:
    rank: int
    eps: float
    lr: float
    step_interval: int
    weight_decay: float = 0.0
    seed: Optional[int] = None


class LOZOController:
    """
    LOZO trainer that maintains master weights and computes updates.
    
    Aligned with LOZO baseline implementation.
    
    Responsibilities:
    1. Hold master weights (2D trainable Linear parameters from HF model)
    2. Sample U, V directions (V cached for step_interval steps)
    3. Build LoRA tensors for plus/minus perturbation (2D params only)
    4. Compute coefficient c = (L+ - L-) / (2*eps)
    5. Update master weights
    
    Note: Embeddings and 1D params (bias, layer_norm) are skipped for initial implementation.
    See IMPLEMENTATION_NOTES.md for details.
    """
    
    # Parameters to skip (vLLM LoRA limitations)
    SKIP_PARAMS = [
        "embed_tokens",      # Token embeddings (not Linear)
        "embed_positions",   # Position embeddings (not Linear)
    ]
    
    def __init__(
        self,
        hf_model,
        config: LOZOConfig,
    ):
        self.hf_model = hf_model
        self.config = config
        
        self.master: Dict[str, torch.Tensor] = {}
        self.v_cache: Dict[str, torch.Tensor] = {}
        self.step = 0
        
        self._init_master_weights()
    
    def _init_master_weights(self):
        """
        Copy trainable 2D Linear parameters from HF model.
        
        Skips:
        - Embeddings (embed_tokens, embed_positions) - vLLM LoRA doesn't support
        - 1D params (bias, layer_norm) - vLLM LoRA doesn't support
        
        Only includes Linear layer weights that can be perturbed via LoRA.
        """
        for name, param in self.hf_model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip embeddings
            if any(skip in name for skip in self.SKIP_PARAMS):
                continue
            
            # Skip 1D params (bias, layer_norm)
            if param.ndim == 1:
                continue
            
            # Only 2D Linear weights
            if param.ndim >= 2:
                self.master[name] = param.data.detach().clone().cpu()
    
    def get_trainable_2d_params(self) -> List[str]:
        """Get names of 2D trainable parameters."""
        return [name for name, W in self.master.items() if W.ndim >= 2]
    
    def get_trainable_1d_params(self) -> List[str]:
        """Get names of 1D trainable parameters (bias, layer_norm)."""
        return [name for name, W in self.master.items() if W.ndim == 1]
    
    def sample_direction(
        self,
        random_seed: int,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Sample perturbation directions for all trainable parameters.
        
        For 2D params: low-rank perturbation u @ v.T
            - V is cached for step_interval steps
            - U is freshly sampled every step
        
        For 1D params: full-rank perturbation z
            - z is freshly sampled every step
        
        Args:
            random_seed: Random seed (from np.random.randint)
        
        Returns:
            (directions_2d, directions_1d):
                directions_2d: {name: {"U": tensor, "V": tensor}}
                directions_1d: {name: z_tensor}
        """
        # Save global RNG state and set our seed (to match baseline)
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
        
        directions_2d = {}
        directions_1d = {}
        
        # Sample for 2D parameters (low-rank)
        for name, W in self.master.items():
            if W.ndim >= 2:
                out_features, in_features = W.shape
                
                # V cache logic aligned with baseline:
                # - First step (self.step=0): initialize V (0 % step_interval == 0)
                # - step % step_interval == 0: refresh V
                # - Otherwise: reuse cached V
                if self.step % self.config.step_interval == 0:
                    V = torch.randn(
                        in_features, self.config.rank,
                        dtype=W.dtype,
                        device="cpu",
                    )
                    self.v_cache[name] = V
                else:
                    V = self.v_cache[name]
                
                # U: freshly sampled every step
                U = torch.randn(
                    out_features, self.config.rank,
                    dtype=W.dtype,
                    device="cpu",
                )
                
                directions_2d[name] = {"U": U, "V": V}
        
        # Sample for 1D parameters (full-rank)
        for name, W in self.master.items():
            if W.ndim == 1:
                z = torch.randn(
                    W.shape,
                    dtype=W.dtype,
                    device="cpu",
                )
                directions_1d[name] = z
        
        # Restore global RNG state
        torch.set_rng_state(rng_state)
        
        # Increment step AFTER sampling (aligned with baseline: step++ at end of lowrank_zo_step)
        self.step += 1
        
        return directions_2d, directions_1d
    
    def perturb_weights(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        directions_1d: Dict[str, torch.Tensor],
        scaling_factor: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply perturbation to master weights.
        
        W <- W + scaling_factor * perturbation * eps
        
        Args:
            directions_2d: {name: {"U": U, "V": V}}
            directions_1d: {name: z}
            scaling_factor: +1 for plus, -1 for minus, -2 for second perturb
        
        Returns:
            perturbed_weights: Perturbed weight tensors (HF names)
        """
        perturbed = {}
        
        for name, W in self.master.items():
            if name in directions_2d:
                # 2D: low-rank perturbation u @ v.T
                U = directions_2d[name]["U"]
                V = directions_2d[name]["V"]
                delta_W = (U @ V.T) * self.config.eps
                perturbed[name] = W + scaling_factor * delta_W
            elif name in directions_1d:
                # 1D: full-rank perturbation z
                z = directions_1d[name]
                perturbed[name] = W + scaling_factor * z * self.config.eps
            else:
                # Should not happen
                perturbed[name] = W
        
        return perturbed
    
    def build_temp_lora_tensors(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        sign: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Build LoRA A/B tensors for plus or minus perturbation.
        
        Only for Linear modules (q/k/v_proj, out_proj, fc1, fc2).
        Skip embeddings (embed_tokens, embed_positions) and other non-Linear params.
        
        LoRA form: delta_W = B @ A
        - lora_A: [rank, in_features] = V^T
        - lora_B: [out_features, rank] = sign * eps * U
        
        Args:
            directions_2d: U/V matrices from sample_direction()
            sign: +1 for plus, -1 for minus
        
        Returns:
            (layer_to_A, layer_to_B): LoRA tensors
        """
        layer_to_A = {}
        layer_to_B = {}
        
        # Only build LoRA for Linear modules
        linear_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        
        for name, d in directions_2d.items():
            # Skip non-Linear modules (embeddings, etc.)
            is_linear = any(module in name for module in linear_modules)
            if not is_linear:
                continue
            
            U = d["U"]
            V = d["V"]
            
            lora_A = V.T.contiguous().half()
            lora_B = (sign * self.config.eps * U).contiguous().half()
            
            layer_to_A[name] = lora_A
            layer_to_B[name] = lora_B
        
        return layer_to_A, layer_to_B
    
    def compute_c(self, loss_plus: float, loss_minus: float) -> float:
        """
        Compute LOZO coefficient.
        
        c = (L+ - L-) / (2 * eps)
        """
        return (loss_plus - loss_minus) / (2.0 * self.config.eps)
    
    @torch.no_grad()
    def apply_update_to_master(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        directions_1d: Dict[str, torch.Tensor],
        c: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Update master weights with LOZO update.
        
        For 2D params: W <- W - lr * c * (U @ V^T) / rank
        For 1D params: W <- W - lr * c * z
        
        With weight_decay for non-bias/layer_norm params.
        
        Args:
            directions_2d: U/V matrices
            directions_1d: z vectors
            c: LOZO coefficient
        
        Returns:
            updated_weights: Updated weight tensors (HF names)
        """
        updated = {}
        
        for name, W in self.master.items():
            if name in directions_2d:
                # 2D update
                U = directions_2d[name]["U"].float()
                V = directions_2d[name]["V"].float()
                
                delta = (U @ V.T)
                
                # Check if apply weight_decay
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    new_W = W.float() - self.config.lr * (c * delta + self.config.weight_decay * W.float())
                else:
                    new_W = W.float() - self.config.lr * c * delta
                
                new_W = new_W.to(dtype=W.dtype).contiguous()
                
            elif name in directions_1d:
                # 1D update
                z = directions_1d[name].float()
                
                # Check if apply weight_decay
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    new_W = W.float() - self.config.lr * (c * z + self.config.weight_decay * W.float())
                else:
                    new_W = W.float() - self.config.lr * c * z
                
                new_W = new_W.to(dtype=W.dtype).contiguous()
            
            else:
                new_W = W
            
            self.master[name] = new_W
            updated[name] = new_W
        
        # Note: step is incremented in sample_direction(), not here
        return updated
    
    def get_master_weights(self) -> Dict[str, torch.Tensor]:
        """Get current master weights."""
        return self.master.copy()
    
    def set_step(self, step: int):
        """Set current step (for resuming)."""
        self.step = step