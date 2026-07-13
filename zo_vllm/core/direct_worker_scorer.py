from __future__ import annotations

import time
from typing import Any

from zo_vllm.core.binary_option_objective import (
    accuracy_from_option_nll,
    accuracy_from_multi_option_nll,
    classification_loss_from_option_nll,
    classification_loss_from_multi_option_nll,
    option_lengths,
    predictions_from_multi_option_nll,
    regroup_flat_option_values,
)


__all__ = [
    "accuracy_from_option_nll",
    "accuracy_from_multi_option_nll",
    "classification_loss_from_option_nll",
    "classification_loss_from_multi_option_nll",
    "collect_activation_stats",
    "collect_agzo_directions",
    "collect_agzo_directions_chunked",
    "direct_worker_detail",
    "option_lengths",
    "predictions_from_multi_option_nll",
    "regroup_flat_option_values",
    "score_direct_worker_detailed",
    "forward_token_id_logits",
    "forward_token_id_request_nll",
    "score_prompt_nll_plus_minus_direct_worker",
    "score_token_id_groups",
]


def score_token_id_groups(
    llm,
    token_id_groups: list[list[int]],
    *,
    lora_ids: list[int] | None,
    loss_token_lens: list[int] | None = None,
    labels: list[list[int]] | None = None,
    max_logits_tokens: int,
    loss_impl: str,
) -> dict[str, Any]:
    kwargs = {
        "prompt_token_ids": token_id_groups,
        "lora_ids": lora_ids,
        "labels": labels,
        "max_logits_tokens": max_logits_tokens,
        "loss_impl": loss_impl,
    }
    if loss_token_lens is not None:
        kwargs["loss_token_lens"] = loss_token_lens
    result = llm.llm_engine.model_executor.collective_rpc(
        "zo_score_prompt_token_ids",
        kwargs=kwargs,
        single_value=True,
    )
    return _with_request_mean_nll(result)


def forward_token_id_logits(
    llm,
    token_id_groups: list[list[int]],
    *,
    lora_ids: list[int] | None,
    loss_token_lens: list[int] | None = None,
    labels: list[list[int]] | None = None,
    max_logits_tokens: int,
    loss_impl: str,
    return_device_tensors: bool = False,
) -> dict[str, Any]:
    kwargs = {
        "prompt_token_ids": token_id_groups,
        "lora_ids": lora_ids,
        "labels": labels,
        "max_logits_tokens": max_logits_tokens,
        "loss_impl": loss_impl,
        "return_logits": True,
        "compute_token_nll": False,
        "return_device_tensors": bool(return_device_tensors),
    }
    if loss_token_lens is not None:
        kwargs["loss_token_lens"] = loss_token_lens
    result = llm.llm_engine.model_executor.collective_rpc(
        "zo_score_prompt_token_ids",
        kwargs=kwargs,
        single_value=True,
    )
    return result


def forward_token_id_request_nll(
    llm,
    token_id_groups: list[list[int]],
    *,
    lora_ids: list[int] | None,
    loss_token_lens: list[int],
    max_logits_tokens: int,
    loss_impl: str,
    return_device_tensors: bool = False,
) -> dict[str, Any]:
    """Return per-request mean NLL as a tensor for HF classification loss."""

    result = llm.llm_engine.model_executor.collective_rpc(
        "zo_score_prompt_token_ids",
        kwargs={
            "prompt_token_ids": token_id_groups,
            "lora_ids": lora_ids,
            "loss_token_lens": loss_token_lens,
            "labels": None,
            "max_logits_tokens": max_logits_tokens,
            "loss_impl": loss_impl,
            "compute_token_nll": True,
            "return_request_nll_tensor": True,
            "return_device_tensors": bool(return_device_tensors),
        },
        single_value=True,
    )
    request_nll = result.get("request_nll_tensor")
    if not hasattr(request_nll, "device"):
        raise RuntimeError("worker did not return request_nll_tensor")
    return result


def _with_request_mean_nll(result: dict[str, Any]) -> dict[str, Any]:
    """Return package-facing per-request mean NLL from worker raw weighted NLL."""
    weighted_nll = [float(item) for item in result["request_weighted_nll"]]
    num_tokens = [int(item) for item in result["request_num_tokens"]]
    if len(weighted_nll) != len(num_tokens):
        raise ValueError(
            "request_weighted_nll and request_num_tokens must have the same length"
        )
    zero_indices = [index for index, count in enumerate(num_tokens) if int(count) <= 0]
    if zero_indices:
        raise ValueError(
            "request mean NLL requires positive token counts; zero-token "
            f"indices: {zero_indices[:8]}"
        )
    normalized = {
        key: value for key, value in result.items() if key != "request_weighted_nll"
    }
    normalized["request_nll"] = [
        float(nll_sum) / float(count)
        for nll_sum, count in zip(weighted_nll, num_tokens)
    ]
    return normalized


def collect_activation_stats(
    llm,
    token_id_groups: list[list[int]],
    *,
    lora_ids: list[int] | None = None,
    loss_token_lens: list[int] | None = None,
    labels: list[list[int]] | None = None,
    max_logits_tokens: int = 8192,
    loss_impl: str = "logprobs",
    activation_max_layers: int = 16,
    activation_force_eager: bool = True,
) -> dict[str, Any]:
    """Collect worker-side Linear input activation stats through direct scoring."""
    return llm.llm_engine.model_executor.collective_rpc(
        "zo_collect_activation_stats",
        kwargs={
            "prompt_token_ids": token_id_groups,
            "lora_ids": lora_ids,
            "loss_token_lens": loss_token_lens,
            "labels": labels,
            "max_logits_tokens": max_logits_tokens,
            "loss_impl": loss_impl,
            "activation_max_layers": activation_max_layers,
            "activation_force_eager": activation_force_eager,
        },
        single_value=True,
    )


def collect_agzo_directions(
    llm,
    token_id_groups: list[list[int]],
    *,
    lora_ids: list[int] | None = None,
    loss_token_lens: list[int] | None = None,
    labels: list[list[int]] | None = None,
    max_logits_tokens: int = 8192,
    loss_impl: str = "logprobs",
    activation_force_eager: bool = True,
    agzo_rank: int = 1,
    agzo_power_iter_steps: int = 5,
    agzo_low_rank_oversample: int = 4,
    agzo_basis_seed: int | None = None,
    agzo_perturb_seed: int | None = None,
    agzo_basis_method: str = "power_iter",
) -> dict[str, Any]:
    """Build AGZO directions inside the vLLM worker direct forward path."""
    return llm.llm_engine.model_executor.collective_rpc(
        "zo_collect_agzo_directions",
        kwargs={
            "prompt_token_ids": token_id_groups,
            "lora_ids": lora_ids,
            "loss_token_lens": loss_token_lens,
            "labels": labels,
            "max_logits_tokens": max_logits_tokens,
            "loss_impl": loss_impl,
            "activation_force_eager": activation_force_eager,
            "agzo_rank": agzo_rank,
            "agzo_power_iter_steps": agzo_power_iter_steps,
            "agzo_low_rank_oversample": agzo_low_rank_oversample,
            "agzo_basis_seed": agzo_basis_seed,
            "agzo_perturb_seed": agzo_perturb_seed,
            "agzo_basis_method": agzo_basis_method,
        },
        single_value=True,
    )


def collect_agzo_directions_chunked(
    llm,
    token_id_group_batches: list[list[list[int]]],
    *,
    activation_force_eager: bool = True,
    agzo_rank: int = 1,
    agzo_power_iter_steps: int = 5,
    agzo_low_rank_oversample: int = 4,
    agzo_basis_seed: int | None = None,
    agzo_perturb_seed: int | None = None,
    agzo_basis_method: str = "power_iter",
) -> dict[str, Any]:
    """Build AGZO directions after collecting activations chunk by chunk."""
    return llm.llm_engine.model_executor.collective_rpc(
        "zo_collect_agzo_directions_chunked",
        kwargs={
            "prompt_token_id_batches": token_id_group_batches,
            "activation_force_eager": activation_force_eager,
            "agzo_rank": agzo_rank,
            "agzo_power_iter_steps": agzo_power_iter_steps,
            "agzo_low_rank_oversample": agzo_low_rank_oversample,
            "agzo_basis_seed": agzo_basis_seed,
            "agzo_perturb_seed": agzo_perturb_seed,
            "agzo_basis_method": agzo_basis_method,
        },
        single_value=True,
    )


def fused_agzo_step(
    llm,
    token_id_groups: list[list[int]],
    *,
    token_labels: list[list[int]],
    labels: list[int],
    plus_lora_id: int,
    minus_lora_id: int,
    zo_eps: float,
    learning_rate: float,
    weight_decay: float = 0.0,
    max_logits_tokens: int = 8192,
    loss_impl: str = "logprobs",
    activation_force_eager: bool = True,
    agzo_rank: int = 1,
    agzo_power_iter_steps: int = 5,
    agzo_low_rank_oversample: int = 4,
    agzo_basis_seed: int | None = None,
    agzo_perturb_seed: int | None = None,
    agzo_basis_method: str = "power_iter",
    direction_scale: float = 1.0,
) -> dict[str, Any]:
    """Run one binary-option AGZO update inside the vLLM worker."""
    return llm.llm_engine.model_executor.collective_rpc(
        "zo_fused_agzo_step",
        kwargs={
            "prompt_token_ids": token_id_groups,
            "token_labels": token_labels,
            "labels": labels,
            "plus_lora_id": plus_lora_id,
            "minus_lora_id": minus_lora_id,
            "zo_eps": zo_eps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_logits_tokens": max_logits_tokens,
            "loss_impl": loss_impl,
            "activation_force_eager": activation_force_eager,
            "agzo_rank": agzo_rank,
            "agzo_power_iter_steps": agzo_power_iter_steps,
            "agzo_low_rank_oversample": agzo_low_rank_oversample,
            "agzo_basis_seed": agzo_basis_seed,
            "agzo_perturb_seed": agzo_perturb_seed,
            "agzo_basis_method": agzo_basis_method,
            "direction_scale": direction_scale,
        },
        single_value=True,
    )


def direct_worker_detail(
    result: dict[str, Any],
    score_s: float,
    *,
    loss_impl: str,
) -> dict[str, float | int | str | bool]:
    detail: dict[str, float | int | str | bool] = {
        "score_direct_worker_s": float(score_s),
        "score_num_outputs": int(result["num_reqs"]),
        "score_num_prompt_positions": int(result["num_prompt_tokens"]),
        "score_num_loss_tokens": int(result["num_tokens"]),
        "score_num_tokens_padded": int(result["num_tokens_padded"]),
        "score_num_active_loras": int(result["num_active_loras"]),
        "score_cudagraph_mode": str(result["cudagraph_mode"]),
        "score_loss_impl": str(result.get("loss_impl", loss_impl)),
        "score_num_logits_positions": int(result.get("num_logits_positions", 0)),
        "score_cache_hit": bool(result.get("cache_hit", False)),
    }
    for key, value in result.get("profile_s", {}).items():
        detail[f"score_worker_{key}"] = float(value)
    for key, value in result.get("profile_cuda_ms", {}).items():
        detail[f"score_worker_cuda_{key}"] = float(value)
    return detail


def score_direct_worker_detailed(
    llm,
    token_id_groups: list[list[int]],
    *,
    lora_ids: list[int] | None,
    loss_token_lens: list[int] | None = None,
    labels: list[list[int]] | None = None,
    max_logits_tokens: int,
    loss_impl: str,
) -> tuple[float, dict[str, float | int | str | bool]]:
    score_t0 = time.perf_counter()
    result = score_token_id_groups(
        llm,
        token_id_groups,
        lora_ids=lora_ids,
        loss_token_lens=loss_token_lens,
        labels=labels,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
    )
    score_s = time.perf_counter() - score_t0
    return float(result["loss"]), direct_worker_detail(
        result,
        score_s,
        loss_impl=loss_impl,
    )


def split_direct_worker_losses(
    result: dict[str, Any],
    split: int,
) -> tuple[float, float]:
    request_nll = result["request_nll"]
    num_tokens = result["request_num_tokens"]
    if len(request_nll) != len(num_tokens):
        raise ValueError("request_nll and request_num_tokens must have the same length")
    if split <= 0 or split >= len(request_nll):
        raise ValueError(
            f"split must divide non-empty plus/minus groups; split={split}, "
            f"num_requests={len(request_nll)}"
        )
    plus_nll = sum(
        float(nll) * float(tokens)
        for nll, tokens in zip(request_nll[:split], num_tokens[:split])
    )
    plus_tokens = int(sum(num_tokens[:split]))
    minus_nll = sum(
        float(nll) * float(tokens)
        for nll, tokens in zip(request_nll[split:], num_tokens[split:])
    )
    minus_tokens = int(sum(num_tokens[split:]))
    if plus_tokens <= 0 or minus_tokens <= 0:
        raise ValueError(
            "plus and minus groups must each contain at least one loss token; "
            f"plus_tokens={plus_tokens}, minus_tokens={minus_tokens}"
        )
    return plus_nll / plus_tokens, minus_nll / minus_tokens


def score_prompt_nll_plus_minus_direct_worker(
    llm,
    prompt_token_ids: list[list[int]],
    *,
    plus_id: int,
    minus_id: int,
    max_logits_tokens: int,
    loss_impl: str,
) -> tuple[float, float, dict[str, float | int | str | bool]]:
    score_t0 = time.perf_counter()
    result = score_token_id_groups(
        llm,
        prompt_token_ids + prompt_token_ids,
        lora_ids=[plus_id] * len(prompt_token_ids) + [minus_id] * len(prompt_token_ids),
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
    )
    score_s = time.perf_counter() - score_t0
    loss_plus, loss_minus = split_direct_worker_losses(result, len(prompt_token_ids))
    return (
        loss_plus,
        loss_minus,
        direct_worker_detail(
            result,
            score_s,
            loss_impl=loss_impl,
        ),
    )
