"""Reusable runtime components for ZO experiments with vLLM."""

from .binary_option_objective import (
    accuracy_from_option_nll,
    classification_loss_from_option_nll,
    option_lengths,
)
from .direct_worker_scorer import (
    collect_agzo_directions,
    collect_agzo_directions_chunked,
    collect_activation_stats,
    direct_worker_detail,
    forward_token_id_logits,
    score_direct_worker_detailed,
    score_prompt_nll_plus_minus_direct_worker,
    score_token_id_groups,
    split_direct_worker_losses,
)
from .direction_digest import digest_named_uv
from .param_metadata import ParamMetadata
from .lora_runtime import AsyncLoRASlotRegistry, LoRAUpdateRuntime
from .token_scores import (
    merge_token_scores,
    score_clean_token_groups,
    slice_token_score,
)
from .weight_sync import WeightSync

__all__ = [
    "ParamMetadata",
    "LoRAUpdateRuntime",
    "AsyncLoRASlotRegistry",
    "WeightSync",
    "accuracy_from_option_nll",
    "classification_loss_from_option_nll",
    "collect_agzo_directions",
    "collect_agzo_directions_chunked",
    "collect_activation_stats",
    "digest_named_uv",
    "direct_worker_detail",
    "forward_token_id_logits",
    "merge_token_scores",
    "option_lengths",
    "score_clean_token_groups",
    "score_direct_worker_detailed",
    "score_prompt_nll_plus_minus_direct_worker",
    "score_token_id_groups",
    "slice_token_score",
    "split_direct_worker_losses",
]
