"""Model metadata helpers for ZO update backends."""

from __future__ import annotations

from typing import Any

import torch

from zo_vllm.core.lora_scope import DEFAULT_TRANSFORMER_TARGET_MODULES
from zo_vllm.core.param_metadata import ParamMetadata


LORA_INCOMPATIBLE_PARAM_SUBSTRINGS = (
    "embed_tokens",
    "embed_positions",
)


def build_lora_param_metadata_from_hf_model(
    hf_model: Any,
    *,
    device: torch.device | str | None = None,
) -> dict[str, ParamMetadata]:
    """Read trainable 2D LoRA-compatible Linear metadata from an HF model."""

    metadata: dict[str, ParamMetadata] = {}
    override_device = None if device is None else torch.device(device)
    for name, param in hf_model.named_parameters():
        if not param.requires_grad:
            continue
        if any(item in name for item in LORA_INCOMPATIBLE_PARAM_SUBSTRINGS):
            continue
        if int(param.ndim) == 1:
            continue
        metadata[name] = ParamMetadata(
            name=name,
            shape=tuple(int(dim) for dim in param.shape),
            dtype=param.dtype,
            device=override_device or param.device,
        )
    return metadata


def build_lora_param_metadata_from_config(
    model_config: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    target_modules: list[str] | None = None,
) -> dict[str, ParamMetadata]:
    """Build HF-style LoRA-compatible Linear metadata from a model config."""

    model_type = str(getattr(model_config, "model_type", "")).lower()
    num_layers = int(getattr(model_config, "num_hidden_layers"))
    hidden_size = int(getattr(model_config, "hidden_size"))
    ffn_dim = int(
        getattr(
            model_config,
            "ffn_dim",
            getattr(model_config, "intermediate_size", 0),
        )
    )
    if ffn_dim <= 0:
        raise ValueError("model config must expose ffn_dim or intermediate_size")

    metadata: dict[str, ParamMetadata] = {}
    effective_target_modules = target_modules or DEFAULT_TRANSFORMER_TARGET_MODULES

    def add(name: str, out_features: int, in_features: int) -> None:
        if not _matches_target(name.removesuffix(".weight"), effective_target_modules):
            return
        metadata[name] = ParamMetadata(
            name=name,
            shape=(int(out_features), int(in_features)),
            dtype=dtype,
            device=device,
        )

    vocab_size = int(getattr(model_config, "vocab_size", 0))
    tie_word_embeddings = bool(getattr(model_config, "tie_word_embeddings", False))

    if model_type in {"qwen2", "qwen3"}:
        num_heads = int(getattr(model_config, "num_attention_heads"))
        num_kv_heads = int(getattr(model_config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(model_config, "head_dim", hidden_size // num_heads))
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}"
            add(f"{prefix}.self_attn.q_proj.weight", q_size, hidden_size)
            add(f"{prefix}.self_attn.k_proj.weight", kv_size, hidden_size)
            add(f"{prefix}.self_attn.v_proj.weight", kv_size, hidden_size)
            add(f"{prefix}.self_attn.o_proj.weight", hidden_size, q_size)
            add(f"{prefix}.mlp.gate_proj.weight", ffn_dim, hidden_size)
            add(f"{prefix}.mlp.up_proj.weight", ffn_dim, hidden_size)
            add(f"{prefix}.mlp.down_proj.weight", hidden_size, ffn_dim)
        add("model.embed_tokens.weight", hidden_size, vocab_size)
        if not tie_word_embeddings or _matches_target(
            "lm_head", effective_target_modules
        ):
            add("lm_head.weight", vocab_size, hidden_size)
        return metadata

    if model_type == "opt":
        embed_dim = int(getattr(model_config, "word_embed_proj_dim", hidden_size))
        for layer_idx in range(num_layers):
            prefix = f"model.decoder.layers.{layer_idx}"
            add(f"{prefix}.self_attn.q_proj.weight", hidden_size, hidden_size)
            add(f"{prefix}.self_attn.k_proj.weight", hidden_size, hidden_size)
            add(f"{prefix}.self_attn.v_proj.weight", hidden_size, hidden_size)
            add(f"{prefix}.self_attn.out_proj.weight", hidden_size, hidden_size)
            add(f"{prefix}.fc1.weight", ffn_dim, hidden_size)
            add(f"{prefix}.fc2.weight", hidden_size, ffn_dim)
        add("model.decoder.embed_tokens.weight", embed_dim, vocab_size)
        if not tie_word_embeddings or _matches_target(
            "lm_head", effective_target_modules
        ):
            add("lm_head.weight", vocab_size, embed_dim)
        return metadata

    raise ValueError(
        "ZO LoRA metadata currently supports qwen2, qwen3, and opt model configs; "
        f"got model_type={model_type!r}"
    )


def _matches_target(module_name: str, target_modules: list[str]) -> bool:
    suffix = module_name.rsplit(".", 1)[-1]
    target_set = set(target_modules)
    return module_name in target_set or suffix in target_set


def metadata_specs(metadata: dict[str, ParamMetadata]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "shape": list(item.shape),
        }
        for item in metadata.values()
    ]


def estimate_lora_bank_state_memory_bytes(
    metadata: dict[str, ParamMetadata],
    *,
    update_bank_rank: int,
    include_probe_u: bool = True,
    include_pending_u: bool = False,
) -> int:
    """Estimate worker-resident update-bank memory in bytes.

    This intentionally excludes vLLM's own LoRA slot stacks, which are allocated
    by the LoRA manager from ``max_loras`` and ``max_lora_rank``. It only counts
    the extra ZO update-bank tensors owned by ``BlockLoRAUpdateBankState``:

    - bank_a: rank x in_features
    - accumulated_u: out_features x rank
    - zero_u: out_features x rank
    - probe_u clone during prepare_for_score, when requested
    - pending_u, only when gradient accumulation keeps pending updates
    """

    rank = int(update_bank_rank)
    if rank <= 0:
        raise ValueError("update_bank_rank must be positive")
    total = 0
    for item in metadata.values():
        if len(item.shape) != 2:
            continue
        out_features, in_features = (int(item.shape[0]), int(item.shape[1]))
        dtype_bytes = torch.empty((), dtype=item.dtype).element_size()
        u_copies = 2
        if include_probe_u:
            u_copies += 1
        if include_pending_u:
            u_copies += 1
        elements = rank * (in_features + u_copies * out_features)
        total += elements * dtype_bytes
    return int(total)


def torch_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("direction_device='cuda' requires CUDA")
    return torch.device(value)


def torch_dtype(value: str) -> torch.dtype:
    normalized = value.lower().replace("torch.", "")
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported direction dtype: {value}")


def torch_dtype_name(value: torch.dtype) -> str:
    return str(value).replace("torch.", "")


def resolve_direction_dtype(value: str, model_config: Any) -> torch.dtype:
    normalized = value.lower().replace("torch.", "")
    if normalized != "auto":
        return torch_dtype(normalized)
    dtype = getattr(model_config, "torch_dtype", None)
    if dtype is None:
        dtype = getattr(model_config, "dtype", None)
    if isinstance(dtype, str):
        dtype = torch_dtype(dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        return dtype
    return torch.float16
