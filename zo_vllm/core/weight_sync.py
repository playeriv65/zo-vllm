"""
Weight Sync - Sync updated weights from ZO trainer to vLLM engine.

Uses vLLM's collective_rpc to directly update model weights in worker process.
For packed modules (qkv_proj), updates the entire packed weight directly.

Note: vLLM LoRA wrapper modules (e.g., MergedQKVParallelLinearWithLoRA)
      don't have weight_loader. We need to access base_layer.weight directly.
"""

import os
import time
from typing import Dict
import torch

from zo_vllm.core.param_metadata import ParamMetadata


PACKED_MODEL_LAYERS_MODEL_TYPES = {
    "gemma",
    "gemma2",
    "gemma3",
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
}


def unwrap_lora_module(module):
    """
    If module is a vLLM LoRA wrapper, return its base layer.
    Otherwise return module itself.
    """
    if hasattr(module, "base_layer"):
        return module.base_layer
    return module


def should_apply_weight_decay(name: str) -> bool:
    return "bias" not in name and "layer_norm" not in name and "layernorm" not in name


def apply_lowrank_update_to_weight_(
    target: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    c: float,
    lr: float,
    weight_decay: float,
    precision: str,
    direction_scale: float = 1.0,
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
            alpha=-lr * c * direction_scale,
        )
        target.copy_(updated.to(dtype=target.dtype))
        return

    if precision == "param":
        if weight_decay:
            target.mul_(1.0 - lr * weight_decay)
        U_gpu = U.to(device=target.device, dtype=target.dtype, non_blocking=True)
        V_gpu = V.to(device=target.device, dtype=target.dtype, non_blocking=True)
        target.addmm_(U_gpu, V_gpu.T, alpha=-lr * c * direction_scale)
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
    scales = [float(item.get("scale", 1.0)) for item in directions]
    if any(v_t is None for v_t in vt_tensors):
        vt_tensors = [item["V"].T for item in directions]
    if any(tuple(u.shape) != (hidden_size, u_tensors[0].shape[1]) for u in u_tensors):
        return False
    if any(
        tuple(v_t.shape) != (u_tensors[0].shape[1], hidden_size) for v_t in vt_tensors
    ):
        return False

    beta = 1.0 - lr * weight_decay
    u_batch = torch.stack(
        [
            u.to(device=target.device, dtype=target.dtype, non_blocking=True) * scale
            for u, scale in zip(u_tensors, scales)
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


def apply_transposed_lowrank_update_to_weight_(
    target: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    c: float,
    lr: float,
    weight_decay: float,
    precision: str,
    direction_scale: float = 1.0,
) -> None:
    """Apply target += alpha * V @ U.T for embedding weights stored vocab-first."""
    if precision == "float32":
        U_gpu = U.to(device=target.device, non_blocking=True).float()
        V_gpu = V.to(device=target.device, non_blocking=True).float()
        beta = 1.0 - lr * weight_decay
        updated = torch.addmm(
            target.float(),
            V_gpu,
            U_gpu.T,
            beta=beta,
            alpha=-lr * c * direction_scale,
        )
        target.copy_(updated.to(dtype=target.dtype))
        return
    if precision == "param":
        if weight_decay:
            target.mul_(1.0 - lr * weight_decay)
        U_gpu = U.to(device=target.device, dtype=target.dtype, non_blocking=True)
        V_gpu = V.to(device=target.device, dtype=target.dtype, non_blocking=True)
        target.addmm_(V_gpu, U_gpu.T, alpha=-lr * c * direction_scale)
        return
    raise ValueError(f"unknown update precision: {precision}")


def apply_embedding_lowrank_update_to_weight_(
    target: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    c: float,
    lr: float,
    weight_decay: float,
    precision: str,
    direction_scale: float = 1.0,
) -> None:
    """Apply an embedding update while tolerating vocab padding differences."""
    target_shape = tuple(int(dim) for dim in target.shape)
    normal_shape = (int(U.shape[0]), int(V.shape[0]))
    transposed_shape = (normal_shape[1], normal_shape[0])

    def decay_full_target_once() -> None:
        if weight_decay:
            target.mul_(1.0 - lr * weight_decay)

    if target_shape == normal_shape:
        apply_lowrank_update_to_weight_(
            target,
            U,
            V,
            c=c,
            lr=lr,
            weight_decay=weight_decay,
            precision=precision,
            direction_scale=direction_scale,
        )
        return

    if target_shape == transposed_shape:
        apply_transposed_lowrank_update_to_weight_(
            target,
            U,
            V,
            c=c,
            lr=lr,
            weight_decay=weight_decay,
            precision=precision,
            direction_scale=direction_scale,
        )
        return

    if (
        target.dim() == 2
        and target.shape[1] == V.shape[0]
        and target.shape[0] <= U.shape[0]
    ):
        apply_lowrank_update_to_weight_(
            target,
            U[: target.shape[0], :],
            V,
            c=c,
            lr=lr,
            weight_decay=weight_decay,
            precision=precision,
            direction_scale=direction_scale,
        )
        return

    if (
        target.dim() == 2
        and target.shape[0] == V.shape[0]
        and target.shape[1] <= U.shape[0]
    ):
        apply_transposed_lowrank_update_to_weight_(
            target,
            U[: target.shape[1], :],
            V,
            c=c,
            lr=lr,
            weight_decay=weight_decay,
            precision=precision,
            direction_scale=direction_scale,
        )
        return

    if (
        target.dim() == 2
        and target.shape[1] == U.shape[0]
        and target.shape[0] >= V.shape[0]
    ):
        decay_full_target_once()
        apply_lowrank_update_to_weight_(
            target[: V.shape[0], :],
            V,
            U,
            c=c,
            lr=lr,
            weight_decay=0.0,
            precision=precision,
            direction_scale=direction_scale,
        )
        return

    if (
        target.dim() == 2
        and target.shape[0] == U.shape[0]
        and target.shape[1] >= V.shape[0]
    ):
        decay_full_target_once()
        apply_lowrank_update_to_weight_(
            target[:, : V.shape[0]],
            U,
            V,
            c=c,
            lr=lr,
            weight_decay=0.0,
            precision=precision,
            direction_scale=direction_scale,
        )
        return

    raise RuntimeError(
        "embedding update shape mismatch: "
        f"target={target_shape} direction={normal_shape}"
    )


class WeightSync:
    """
    Sync weights from external trainer (ZO trainer) to vLLM engine.

    Uses collective_rpc to call worker's model parameter update directly.
    For packed modules (qkv_proj), updates the entire packed weight tensor.
    """

    def __init__(self, llm, num_layers: int = 32, model_config=None):
        self.llm = llm
        self.num_layers = num_layers
        self.model_config = model_config
        self.model_type = str(getattr(model_config, "model_type", "opt"))
        self.hf_to_vllm_mapping = self._build_mapping()
        self.hf_to_slice = self._build_slice_mapping()
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
        if self._uses_packed_model_layers_mapping():
            for layer_idx in range(self.num_layers):
                prefix = f"model.layers.{layer_idx}"
                for proj_name in ["q_proj", "k_proj", "v_proj"]:
                    mapping[f"{prefix}.self_attn.{proj_name}.weight"] = (
                        f"{prefix}.self_attn.qkv_proj.weight"
                    )
                for proj_name in ["gate_proj", "up_proj"]:
                    mapping[f"{prefix}.mlp.{proj_name}.weight"] = (
                        f"{prefix}.mlp.gate_up_proj.weight"
                    )
                mapping[f"{prefix}.self_attn.o_proj.weight"] = (
                    f"{prefix}.self_attn.o_proj.weight"
                )
                mapping[f"{prefix}.mlp.down_proj.weight"] = (
                    f"{prefix}.mlp.down_proj.weight"
                )
            mapping["model.embed_tokens.weight"] = "model.embed_tokens.weight"
            mapping["lm_head.weight"] = "lm_head.weight"
            return mapping

        for layer_idx in range(self.num_layers):
            # Packed module: qkv_proj
            # All q/k/v_proj weights map to the same qkv_proj.weight
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                hf_name = (
                    f"model.decoder.layers.{layer_idx}.self_attn.{proj_name}.weight"
                )
                vllm_name = (
                    f"model.decoder.layers.{layer_idx}.self_attn.qkv_proj.weight"
                )
                mapping[hf_name] = vllm_name

            # Non-packed modules: direct mapping
            for module_suffix in [
                "self_attn.out_proj.weight",
                "fc1.weight",
                "fc2.weight",
            ]:
                hf_name = f"model.decoder.layers.{layer_idx}.{module_suffix}"
                mapping[hf_name] = hf_name

        mapping["model.decoder.embed_tokens.weight"] = (
            "model.decoder.embed_tokens.weight"
        )
        mapping["lm_head.weight"] = "lm_head.weight"
        return mapping

    def _build_slice_mapping(self) -> Dict[str, tuple[int, int]]:
        if not self._uses_packed_model_layers_mapping():
            return {}
        hidden_size = int(getattr(self.model_config, "hidden_size"))
        num_heads = int(getattr(self.model_config, "num_attention_heads"))
        num_kv_heads = int(getattr(self.model_config, "num_key_value_heads", num_heads))
        configured_head_dim = getattr(self.model_config, "head_dim", None)
        head_dim = int(configured_head_dim or (hidden_size // num_heads))
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        intermediate_size = int(getattr(self.model_config, "intermediate_size"))

        slices: Dict[str, tuple[int, int]] = {}
        for layer_idx in range(self.num_layers):
            prefix = f"model.layers.{layer_idx}"
            q_start = 0
            k_start = q_size
            v_start = q_size + kv_size
            slices[f"{prefix}.self_attn.q_proj.weight"] = (q_start, k_start)
            slices[f"{prefix}.self_attn.k_proj.weight"] = (k_start, v_start)
            slices[f"{prefix}.self_attn.v_proj.weight"] = (
                v_start,
                v_start + kv_size,
            )
            slices[f"{prefix}.mlp.gate_proj.weight"] = (0, intermediate_size)
            slices[f"{prefix}.mlp.up_proj.weight"] = (
                intermediate_size,
                2 * intermediate_size,
            )
        return slices

    def _uses_packed_model_layers_mapping(self) -> bool:
        return self.model_type in PACKED_MODEL_LAYERS_MODEL_TYPES

    def _infer_hf_param_shape_from_config(self, hf_name: str) -> tuple[int, int]:
        """Infer unpacked HF Linear weight shape when quantized vLLM lacks weight."""
        hidden_size = int(getattr(self.model_config, "hidden_size"))
        ffn_dim = getattr(
            self.model_config,
            "ffn_dim",
            getattr(self.model_config, "intermediate_size", None),
        )
        if ffn_dim is None:
            raise AttributeError(
                "model_config must define ffn_dim or intermediate_size "
                "to infer quantized metadata"
            )
        ffn_dim = int(ffn_dim)

        if self._uses_packed_model_layers_mapping():
            if hf_name == "model.embed_tokens.weight":
                vocab_size = int(getattr(self.model_config, "vocab_size"))
                return (hidden_size, vocab_size)
            if hf_name == "lm_head.weight":
                vocab_size = int(getattr(self.model_config, "vocab_size"))
                return (vocab_size, hidden_size)
            num_heads = int(getattr(self.model_config, "num_attention_heads"))
            num_kv_heads = int(
                getattr(self.model_config, "num_key_value_heads", num_heads)
            )
            configured_head_dim = getattr(self.model_config, "head_dim", None)
            head_dim = int(configured_head_dim or (hidden_size // num_heads))
            if ".self_attn.q_proj.weight" in hf_name:
                return (num_heads * head_dim, hidden_size)
            if (
                ".self_attn.k_proj.weight" in hf_name
                or ".self_attn.v_proj.weight" in hf_name
            ):
                return (num_kv_heads * head_dim, hidden_size)
            if ".self_attn.o_proj.weight" in hf_name:
                return (hidden_size, num_heads * head_dim)
            if ".mlp.gate_proj.weight" in hf_name or ".mlp.up_proj.weight" in hf_name:
                return (ffn_dim, hidden_size)
            if ".mlp.down_proj.weight" in hf_name:
                return (hidden_size, ffn_dim)
        else:
            embed_dim = int(
                getattr(self.model_config, "word_embed_proj_dim", hidden_size)
            )
            if hf_name == "model.decoder.embed_tokens.weight":
                vocab_size = int(getattr(self.model_config, "vocab_size"))
                return (embed_dim, vocab_size)
            if hf_name == "lm_head.weight":
                vocab_size = int(getattr(self.model_config, "vocab_size"))
                return (vocab_size, embed_dim)
            if ".self_attn." in hf_name:
                return (hidden_size, hidden_size)
            if hf_name.endswith(".fc1.weight"):
                return (ffn_dim, hidden_size)
            if hf_name.endswith(".fc2.weight"):
                return (hidden_size, ffn_dim)

        raise ValueError(f"cannot infer quantized metadata shape for {hf_name}")

    def get_hf_param_metadata(
        self,
        *,
        include_embeddings: bool = False,
        include_lm_head: bool = False,
    ) -> Dict[str, ParamMetadata]:
        """
        Read LoRA-compatible parameter metadata from the vLLM worker.

        This avoids loading a second HF model just to discover tensor shapes.
        For OPT q/k/v weights, vLLM stores one packed qkv_proj tensor, so the
        returned HF-style metadata exposes each slice with the unpacked shape.
        """
        ordered_hf_names = [
            name
            for name in self.hf_to_vllm_mapping
            if include_embeddings or "embed_tokens" not in name
            if include_lm_head or not name.startswith("lm_head.")
        ]

        def read_metadata_on_worker(worker):
            model = worker.model_runner.model
            metadata = []
            for hf_name in ordered_hf_names:
                vllm_name = self.hf_to_vllm_mapping[hf_name]
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = getattr(base_layer, "weight", None)
                if "embed_tokens" in hf_name or hf_name == "lm_head.weight":
                    shape = self._infer_hf_param_shape_from_config(hf_name)
                    if param is None:
                        quant_param = getattr(base_layer, "qweight", None)
                        device = (
                            quant_param.device
                            if quant_param is not None
                            else torch.device(
                                "cuda" if torch.cuda.is_available() else "cpu"
                            )
                        )
                        dtype = torch.float16
                elif param is None:
                    shape = self._infer_hf_param_shape_from_config(hf_name)
                    quant_param = getattr(base_layer, "qweight", None)
                    device = (
                        quant_param.device
                        if quant_param is not None
                        else torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                    )
                    dtype = torch.float16
                elif hf_name in self.hf_to_slice:
                    start, end = self.hf_to_slice[hf_name]
                    shape = (end - start, param.data.shape[1])
                elif "qkv_proj.weight" in vllm_name:
                    hidden_size = param.data.shape[0] // 3
                    shape = (hidden_size, param.data.shape[1])
                else:
                    shape = tuple(param.data.shape)
                if param is not None:
                    dtype = param.data.dtype
                    device = param.data.device
                metadata.append(
                    {
                        "name": hf_name,
                        "shape": tuple(int(dim) for dim in shape),
                        "dtype": str(dtype).replace("torch.", ""),
                        "device": str(device),
                    }
                )
            return metadata

        worker_results = self.llm.collective_rpc(read_metadata_on_worker)
        if not worker_results:
            raise RuntimeError("vLLM worker returned no parameter metadata")
        if len(worker_results) != 1:
            raise RuntimeError(
                "metadata-only ZO trainer currently expects one vLLM worker"
            )

        metadata = {}
        for item in worker_results[0]:
            dtype = getattr(torch, item["dtype"])
            metadata[item["name"]] = ParamMetadata(
                name=item["name"],
                shape=tuple(item["shape"]),
                dtype=dtype,
                device=torch.device(item["device"]),
            )
        return metadata

    def apply_lozo_update(
        self,
        directions_2d: Dict[str, Dict[str, torch.Tensor]],
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

        This skips materializing full updated weights in Python master weights and
        skips copying those full tensors into vLLM. The vLLM base weights
        become the training master state.
        """
        if precision not in {"float32", "param"}:
            raise ValueError(f"unknown update precision: {precision}")
        if qkv_update_mode not in {"separate", "batched"}:
            raise ValueError(f"unknown qkv_update_mode: {qkv_update_mode}")
        profile_enabled = os.environ.get("VLLM_ZO_WEIGHT_PROFILE", "0") == "1"

        def update_weights_on_worker(worker):
            profile_t0 = time.perf_counter()
            model = worker.model_runner.model

            packed_directions = {}
            qkv_directions = {}
            other_directions = {}

            for hf_name, direction in directions_2d.items():
                vllm_name = self.hf_to_vllm_mapping.get(hf_name)
                if vllm_name is None:
                    raise ValueError(f"No mapping for HF parameter: {hf_name}")

                if hf_name in self.hf_to_slice:
                    packed_directions.setdefault(vllm_name, []).append(
                        (hf_name, self.hf_to_slice[hf_name], direction)
                    )
                elif "qkv_proj.weight" in vllm_name:
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

            for vllm_name, items in packed_directions.items():
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = base_layer.weight
                for hf_name, (start, end), direction in items:
                    target = param.data[start:end, :]
                    expected_shape = (
                        int(direction["U"].shape[0]),
                        int(direction["V"].shape[0]),
                    )
                    if tuple(target.shape) != expected_shape:
                        raise RuntimeError(
                            "packed slice shape mismatch for "
                            f"{hf_name}: target={tuple(target.shape)} "
                            f"direction={expected_shape} packed={tuple(param.data.shape)} "
                            f"slice=({start}, {end})"
                        )
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
                        direction_scale=float(direction.get("scale", 1.0)),
                    )

            for vllm_name, proj_dict in qkv_directions.items():
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = base_layer.weight
                hidden_size = param.data.shape[0] // 3

                if qkv_update_mode == "batched" and all(
                    key in proj_dict for key in ("q", "k", "v")
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
                        target = param.data[hidden_size : 2 * hidden_size, :]
                    elif proj_key == "v":
                        target = param.data[2 * hidden_size : 3 * hidden_size, :]
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
                        direction_scale=float(direction.get("scale", 1.0)),
                    )

            profile_after_qkv = time.perf_counter()
            for vllm_name, (hf_name, direction) in other_directions.items():
                module_path = vllm_name.replace(".weight", "")
                module = model.get_submodule(module_path)
                base_layer = unwrap_lora_module(module)
                param = base_layer.weight
                expected_shape = (
                    int(direction["U"].shape[0]),
                    int(direction["V"].shape[0]),
                )
                if "embed_tokens" in hf_name:
                    apply_embedding_lowrank_update_to_weight_(
                        param.data,
                        direction["U"],
                        direction["V"],
                        c=c,
                        lr=lr,
                        weight_decay=(
                            weight_decay if should_apply_weight_decay(hf_name) else 0.0
                        ),
                        precision=precision,
                        direction_scale=float(direction.get("scale", 1.0)),
                    )
                    continue

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
                    direction_scale=float(direction.get("scale", 1.0)),
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
