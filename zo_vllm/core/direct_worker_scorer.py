from __future__ import annotations

import time
from typing import Any

from zo_vllm.experiment.sst2_official import (
    accuracy_from_nll,
    classification_loss_from_nll,
    encode_sst2_vllm_prompts,
    single_token_id,
)


def score_token_id_groups(
    llm,
    token_id_groups: list[list[int]],
    *,
    lora_ids: list[int] | None,
    loss_token_lens: list[int] | None = None,
    max_logits_tokens: int,
    loss_impl: str,
) -> dict[str, Any]:
    return llm.llm_engine.model_executor.collective_rpc(
        "zo_score_prompt_token_ids",
        kwargs={
            "prompt_token_ids": token_id_groups,
            "lora_ids": lora_ids,
            "loss_token_lens": loss_token_lens,
            "max_logits_tokens": max_logits_tokens,
            "loss_impl": loss_impl,
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
    max_logits_tokens: int,
    loss_impl: str,
) -> tuple[float, dict[str, float | int | str | bool]]:
    score_t0 = time.perf_counter()
    result = score_token_id_groups(
        llm,
        token_id_groups,
        lora_ids=lora_ids,
        loss_token_lens=loss_token_lens,
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
    nll_sums = result["request_nll_sums"]
    num_tokens = result["request_num_tokens"]
    plus_nll = float(sum(nll_sums[:split]))
    plus_tokens = int(sum(num_tokens[:split]))
    minus_nll = float(sum(nll_sums[split:]))
    minus_tokens = int(sum(num_tokens[split:]))
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
    return loss_plus, loss_minus, direct_worker_detail(
        result,
        score_s,
        loss_impl=loss_impl,
    )


def option_lengths(
    stem_ids: list[list[int]],
    neg_ids: list[list[int]],
    pos_ids: list[list[int]],
) -> tuple[list[int], list[int]]:
    neg_lens = [len(neg) - len(stem) for stem, neg in zip(stem_ids, neg_ids)]
    pos_lens = [len(pos) - len(stem) for stem, pos in zip(stem_ids, pos_ids)]
    if min(neg_lens + pos_lens) <= 0:
        raise ValueError("classification verbalizer must add at least one option token")
    return neg_lens, pos_lens


def classification_loss_from_option_nll(
    neg_nll: list[float],
    pos_nll: list[float],
    labels: list[int],
) -> float:
    return classification_loss_from_nll([0.0] * len(labels), neg_nll, pos_nll, labels)


def accuracy_from_option_nll(
    neg_nll: list[float],
    pos_nll: list[float],
    labels: list[int],
) -> float:
    return accuracy_from_nll([0.0] * len(labels), neg_nll, pos_nll, labels)


def eval_sst2_accuracy_direct_worker(
    llm,
    tokenizer,
    rows,
    *,
    max_logits_tokens: int,
    loss_impl: str,
    max_prompts_per_call: int = 32,
) -> float | None:
    if not rows:
        return None
    single_token_id(tokenizer, " terrible")
    single_token_id(tokenizer, " great")
    correct = 0
    total = len(rows)
    for start in range(0, total, max_prompts_per_call):
        sub_rows = rows[start : start + max_prompts_per_call]
        stem_ids, neg_ids, pos_ids, labels = encode_sst2_vllm_prompts(
            sub_rows,
            tokenizer,
        )
        neg_lens, pos_lens = option_lengths(stem_ids, neg_ids, pos_ids)
        token_groups = neg_ids + pos_ids
        result = score_token_id_groups(
            llm,
            token_groups,
            lora_ids=None,
            loss_token_lens=neg_lens + pos_lens,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
        )
        n = len(sub_rows)
        nll = result["request_nll_sums"]
        correct += int(accuracy_from_option_nll(nll[:n], nll[n : 2 * n], labels) * n)
    return float(correct / total)


def score_sst2_classification_direct_worker(
    llm,
    rows,
    tokenizer,
    *,
    max_logits_tokens: int,
    loss_impl: str,
    lora_id: int | None = None,
    max_rows_per_call: int = 32,
) -> float | None:
    if not rows:
        return None
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(rows), max_rows_per_call):
        sub_rows = rows[start : start + max_rows_per_call]
        stem_ids, neg_ids, pos_ids, labels = encode_sst2_vllm_prompts(
            sub_rows,
            tokenizer,
        )
        neg_lens, pos_lens = option_lengths(stem_ids, neg_ids, pos_ids)
        token_groups = neg_ids + pos_ids
        lora_ids = None if lora_id is None else [int(lora_id)] * len(token_groups)
        result = score_token_id_groups(
            llm,
            token_groups,
            lora_ids=lora_ids,
            loss_token_lens=neg_lens + pos_lens,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
        )
        n = len(sub_rows)
        loss = classification_loss_from_option_nll(
            result["request_nll_sums"][:n],
            result["request_nll_sums"][n : 2 * n],
            labels,
        )
        total_loss += loss * n
        total_count += n
    return total_loss / max(total_count, 1)


def score_sst2_classification_plus_minus_direct_worker(
    llm,
    rows,
    tokenizer,
    *,
    plus_id: int,
    minus_id: int,
    max_logits_tokens: int,
    loss_impl: str,
) -> tuple[float, float, dict[str, float | int | str | bool]]:
    stem_ids, neg_ids, pos_ids, labels = encode_sst2_vllm_prompts(rows, tokenizer)
    neg_lens, pos_lens = option_lengths(stem_ids, neg_ids, pos_ids)
    one_side = neg_ids + pos_ids
    one_side_loss_lens = neg_lens + pos_lens
    n = len(rows)
    score_t0 = time.perf_counter()
    result = score_token_id_groups(
        llm,
        one_side + one_side,
        lora_ids=[plus_id] * len(one_side) + [minus_id] * len(one_side),
        loss_token_lens=one_side_loss_lens + one_side_loss_lens,
        max_logits_tokens=max_logits_tokens,
        loss_impl=loss_impl,
    )
    score_s = time.perf_counter() - score_t0
    nll = result["request_nll_sums"]
    plus_loss = classification_loss_from_option_nll(
        nll[:n],
        nll[n : 2 * n],
        labels,
    )
    offset = len(one_side)
    minus_loss = classification_loss_from_option_nll(
        nll[offset : offset + n],
        nll[offset + n : offset + 2 * n],
        labels,
    )
    return plus_loss, minus_loss, direct_worker_detail(
        result,
        score_s,
        loss_impl=loss_impl,
    )
