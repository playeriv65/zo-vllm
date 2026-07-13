"""GPU memory reservation estimates for ZO training scratch tensors."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from zo_vllm.core.param_metadata import ParamMetadata


def dtype_nbytes(dtype: torch.dtype) -> int:
    """Return element size for a torch dtype without allocating model tensors."""

    return int(torch.empty((), dtype=dtype).element_size())


def estimate_factorized_direction_bytes(
    param_metadata: Mapping[str, ParamMetadata],
    *,
    rank: int,
) -> int:
    """Estimate bytes for one materialized factorized direction map."""

    rank_i = int(rank)
    if rank_i <= 0:
        raise ValueError("rank must be positive")
    total = 0
    for metadata in param_metadata.values():
        if metadata.ndim < 2:
            continue
        out_features, in_features = int(metadata.shape[0]), int(metadata.shape[1])
        total += (out_features + in_features) * rank_i * dtype_nbytes(metadata.dtype)
    return int(total)


def estimate_zo_training_reserved_gpu_bytes(
    param_metadata: Mapping[str, ParamMetadata],
    *,
    rank: int,
    estimator: str,
    multi_query_direction_mode: str,
    query_microbatch_size: int,
    safety_factor: float = 1.25,
) -> int:
    """Estimate non-KV ZO scratch memory that vLLM should reserve.

    vLLM already budgets its registered LoRA slot tensors. This estimate covers
    transient Python-side direction tensors used before they are copied into
    those slots.
    """

    direction_bytes = estimate_factorized_direction_bytes(
        param_metadata,
        rank=int(rank),
    )
    estimator_name = str(estimator).replace("-", "_")
    direction_mode = str(multi_query_direction_mode).replace("-", "_")
    if estimator_name in {"multi_query", "evolution_strategy"}:
        if direction_mode == "independent":
            concurrent_directions = max(1, int(query_microbatch_size))
        else:
            concurrent_directions = max(1, int(query_microbatch_size)) + 1
    else:
        concurrent_directions = 1
    return int(direction_bytes * concurrent_directions * float(safety_factor))


__all__ = [
    "dtype_nbytes",
    "estimate_factorized_direction_bytes",
    "estimate_zo_training_reserved_gpu_bytes",
]
