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
    master_device: str = "cpu"
    random_device: str = "cpu"
    train_scope: str = "lora_only"
    direction_sampling: str = "exact"
    direction_scale: float = 1.0


@dataclass(frozen=True)
class ParamMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @property
    def ndim(self) -> int:
        return len(self.shape)


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
    
    # Parameters to skip in LoRA-compatible mode (vLLM LoRA limitations)
    LORA_INCOMPATIBLE_PARAMS = [
        "embed_tokens",      # Token embeddings (not Linear)
        "embed_positions",   # Position embeddings (not Linear)
    ]
    
    def __init__(
        self,
        hf_model,
        config: LOZOConfig,
        param_metadata: Optional[Dict[str, ParamMetadata]] = None,
    ):
        self.hf_model = hf_model
        self.config = config
        if self.config.direction_sampling not in {"exact", "flat"}:
            raise ValueError(
                f"unknown direction_sampling: {self.config.direction_sampling}"
            )
        
        self.master: Dict[str, torch.Tensor | ParamMetadata] = {}
        self.v_cache: Dict[str, torch.Tensor] = {}
        self.vt_cache: Dict[str, torch.Tensor] = {}
        self.step = 0

        if param_metadata is not None:
            self.master.update(param_metadata)
        elif hf_model is not None:
            self._init_master_weights()
        else:
            raise ValueError("LOZOController requires hf_model or param_metadata")
    
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
            
            if self._should_skip_param(name, param):
                continue

            self.master[name] = (
                param.data.detach()
                .clone()
                .to(self.config.master_device)
                .contiguous()
            )

    def _should_skip_param(self, name: str, param: torch.nn.Parameter) -> bool:
        if self.config.train_scope == "full":
            return False
        if self.config.train_scope != "lora_only":
            raise ValueError(f"unknown train_scope: {self.config.train_scope}")
        if any(skip in name for skip in self.LORA_INCOMPATIBLE_PARAMS):
            return True
        return param.ndim == 1

    def _save_rng_state(self):
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return cpu_state, cuda_states

    def _restore_rng_state(self, state) -> None:
        cpu_state, cuda_states = state
        torch.set_rng_state(cpu_state)
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)

    def _seed_rng(self, random_seed: int) -> None:
        torch.manual_seed(random_seed)
        if self.config.random_device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    def _sample_device_for(self, target_device: torch.device) -> torch.device:
        if self.config.random_device == "cpu":
            return torch.device("cpu")
        if self.config.random_device == "cuda":
            if target_device.type != "cuda":
                raise ValueError("random_device='cuda' requires CUDA master weights")
            return target_device
        raise ValueError(f"unknown random_device: {self.config.random_device}")

    def _randn(self, shape, target_device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(
            shape,
            dtype=dtype,
            device=self._sample_device_for(target_device),
        ).to(target_device)

    def _randn_flat(
        self,
        numel: int,
        target_device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(
            (numel,),
            dtype=dtype,
            device=self._sample_device_for(target_device),
        ).to(target_device)
    
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
        rng_state = self._save_rng_state()
        self._seed_rng(random_seed)
        
        directions_2d = {}
        directions_1d = {}
        
        # The exact path samples per parameter in the SAME order as baseline,
        # preserving the randn call sequence for step-by-step alignment. The
        # flat path is a Phase 3 performance mode: it preserves distribution
        # and deterministic seeding but not the baseline's per-call RNG digest.
        try:
            if self.config.direction_sampling == "flat":
                params_2d = [
                    (name, W)
                    for name, W in self.master.items()
                    if W.ndim >= 2
                ]
                params_1d = [
                    (name, W)
                    for name, W in self.master.items()
                    if W.ndim == 1
                ]
                v_refreshed = self.step % self.config.step_interval == 0
                if params_2d:
                    sample_device = params_2d[0][1].device
                    sample_dtype = params_2d[0][1].dtype
                    if any(
                        W.device != sample_device or W.dtype != sample_dtype
                        for _, W in params_2d
                    ):
                        raise RuntimeError(
                            "flat direction sampling requires uniform 2D parameter "
                            "device and dtype"
                        )
                    if v_refreshed:
                        total_v = sum(
                            W.shape[1] * self.config.rank for _, W in params_2d
                        )
                        flat_v = self._randn_flat(total_v, sample_device, sample_dtype)
                        offset = 0
                        for name, W in params_2d:
                            in_features = W.shape[1]
                            numel = in_features * self.config.rank
                            V = flat_v[offset : offset + numel].view(
                                in_features, self.config.rank
                            )
                            self.v_cache[name] = V
                            self.vt_cache[name] = V.T.contiguous()
                            offset += numel

                    total_u = sum(
                        W.shape[0] * self.config.rank for _, W in params_2d
                    )
                    flat_u = self._randn_flat(total_u, sample_device, sample_dtype)
                    offset = 0
                    for name, W in params_2d:
                        out_features = W.shape[0]
                        numel = out_features * self.config.rank
                        U = flat_u[offset : offset + numel].view(
                            out_features, self.config.rank
                        )
                        directions_2d[name] = {
                            "U": U,
                            "V": self.v_cache[name],
                            "V_T": self.vt_cache[name],
                            "v_refreshed": v_refreshed,
                            "scale": float(self.config.direction_scale),
                        }
                        offset += numel

                if params_1d:
                    sample_device = params_1d[0][1].device
                    sample_dtype = params_1d[0][1].dtype
                    if any(
                        W.device != sample_device or W.dtype != sample_dtype
                        for _, W in params_1d
                    ):
                        raise RuntimeError(
                            "flat direction sampling requires uniform 1D parameter "
                            "device and dtype"
                        )
                    total_z = sum(W.numel() for _, W in params_1d)
                    flat_z = self._randn_flat(total_z, sample_device, sample_dtype)
                    offset = 0
                    for name, W in params_1d:
                        numel = W.numel()
                        directions_1d[name] = flat_z[offset : offset + numel].view_as(W)
                        offset += numel
            else:
                for name, W in self.master.items():
                    if W.ndim >= 2:
                        out_features, in_features = W.shape

                        # V cache logic aligned with baseline:
                        # - First step (self.step=0): initialize V (0 % step_interval == 0)
                        # - step % step_interval == 0: refresh V
                        # - Otherwise: reuse cached V
                        v_refreshed = self.step % self.config.step_interval == 0
                        if v_refreshed:
                            V = self._randn(
                                (in_features, self.config.rank), W.device, W.dtype
                            )
                            self.v_cache[name] = V
                            self.vt_cache[name] = V.T.contiguous()
                        else:
                            V = self.v_cache[name]

                        # U: freshly sampled every step
                        U = self._randn(
                            (out_features, self.config.rank), W.device, W.dtype
                        )

                        directions_2d[name] = {
                            "U": U,
                            "V": V,
                            "V_T": self.vt_cache[name],
                            "v_refreshed": v_refreshed,
                            "scale": float(self.config.direction_scale),
                        }
                    elif W.ndim == 1:
                        directions_1d[name] = self._randn(
                            tuple(W.shape), W.device, W.dtype
                        )
        finally:
            # Restore global RNG state
            self._restore_rng_state(rng_state)
        
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
                scale = float(directions_2d[name].get("scale", self.config.direction_scale))
                delta_W = (U @ V.T) * (self.config.eps * scale)
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
        output_device: str | torch.device | None = "cpu",
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
            output_device: Device for returned LoRA tensors. The historical
                memory LoRA path uses CPU tensors; GPU-resident LoRA passes
                "cuda" to avoid the host round trip.

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
            scale = float(d.get("scale", self.config.direction_scale))
            
            lora_A = V.T.contiguous().half()
            lora_B = (sign * self.config.eps * scale * U).contiguous().half()
            if output_device is not None:
                lora_A = lora_A.to(output_device).contiguous()
                lora_B = lora_B.to(output_device).contiguous()
            
            layer_to_A[name] = lora_A
            layer_to_B[name] = lora_B
        
        return layer_to_A, layer_to_B

    def build_temp_lora_pair_tensors(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
        output_device: str | torch.device | None = "cpu",
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Build plus/minus LoRA tensors in one pass.

        The A matrix is identical for both signs, so this avoids building and
        copying V.T twice on every ZO step.
        """
        plus_A = {}
        plus_B = {}
        minus_A = {}
        minus_B = {}
        linear_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

        for name, d in directions_2d.items():
            if not any(module in name for module in linear_modules):
                continue

            U = d["U"]
            V = d["V"]
            scale = float(d.get("scale", self.config.direction_scale))
            lora_A = V.T.contiguous().half()
            lora_B = (self.config.eps * scale * U).contiguous().half()
            if output_device is not None:
                lora_A = lora_A.to(output_device).contiguous()
                lora_B = lora_B.to(output_device).contiguous()

            plus_A[name] = lora_A
            minus_A[name] = lora_A
            plus_B[name] = lora_B
            minus_B[name] = lora_B.neg()

        return plus_A, plus_B, minus_A, minus_B
    
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
                
                scale = float(directions_2d[name].get("scale", self.config.direction_scale))
                delta = (U @ V.T) * scale
                
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
