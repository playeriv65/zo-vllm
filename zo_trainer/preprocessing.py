"""Hugging Face-native preprocessing helpers for prompt-objective training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

IGNORE_INDEX = -100


def build_causal_lm_preprocess(
    tokenizer: Any,
    *,
    text_field: str = "text",
    max_length: int | None = None,
    truncation: bool = True,
) -> Callable[[Mapping[str, Sequence[Any]]], dict[str, list[list[int]]]]:
    """Return a ``Dataset.map`` function that emits HF causal-LM features."""

    def preprocess(examples: Mapping[str, Sequence[Any]]) -> dict[str, list[list[int]]]:
        texts = [str(item) for item in examples[text_field]]
        encoded = tokenizer(
            texts,
            max_length=max_length,
            truncation=bool(truncation),
            padding=False,
        )
        input_ids = _int_rows(encoded["input_ids"])
        attention_mask = _int_rows(
            encoded.get(
                "attention_mask",
                [[1] * len(row) for row in input_ids],
            )
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [list(row) for row in input_ids],
        }

    return preprocess


def build_target_lm_preprocess(
    tokenizer: Any,
    *,
    prompt_field: str = "prompt",
    target_field: str = "target",
    max_length: int | None = None,
    truncation: bool = True,
    add_special_tokens: bool = True,
) -> Callable[[Mapping[str, Sequence[Any]]], dict[str, list[list[int]]]]:
    """Return a ``Dataset.map`` function with prompt tokens masked to ``-100``."""

    def preprocess(examples: Mapping[str, Sequence[Any]]) -> dict[str, list[list[int]]]:
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for prompt_value, target_value in zip(
            examples[prompt_field],
            examples[target_field],
        ):
            prompt = str(prompt_value)
            prompt_ids = _encode(
                tokenizer,
                prompt,
                max_length=None,
                truncation=False,
                add_special_tokens=bool(add_special_tokens),
            )
            target_ids = _encode(
                tokenizer,
                str(target_value),
                max_length=None,
                truncation=False,
                add_special_tokens=False,
            )
            if not target_ids:
                raise ValueError("target-LM preprocessing produced an empty target")
            if max_length is not None and bool(truncation):
                prompt_ids, target_ids = _truncate_prompt_target(
                    prompt_ids,
                    target_ids,
                    max_length=int(max_length),
                )
            full_ids = list(prompt_ids) + list(target_ids)
            row_labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)
            input_ids.append(full_ids)
            attention_mask.append([1] * len(full_ids))
            labels.append(row_labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return preprocess


@dataclass
class VLLMDataCollator:
    """Build ragged vLLM batches through the native HF collator extension point."""

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("VLLMDataCollator requires at least one feature")
        if _is_nested_rows(features[0]["input_ids"]):
            return self._classification_batch(features)
        return self._causal_lm_batch(features)

    @staticmethod
    def _causal_lm_batch(features: list[Mapping[str, Any]]) -> dict[str, Any]:
        input_ids: list[list[int]] = []
        if any(
            "labels" not in feature or feature["labels"] is None for feature in features
        ):
            raise ValueError("causal-LM features require labels")
        labels: list[list[int]] = []
        for feature in features:
            ids, active_indices = _active_token_row(
                feature["input_ids"],
                feature.get("attention_mask"),
            )
            input_ids.append(ids)
            row_labels = feature["labels"]
            if len(row_labels) != len(feature["input_ids"]):
                raise ValueError("causal-LM labels must match input_ids")
            labels.append(
                row_labels
                if active_indices is None and isinstance(row_labels, list)
                else [
                    int(row_labels[index])
                    for index in (
                        range(len(row_labels))
                        if active_indices is None
                        else active_indices
                    )
                ]
            )
        return {"input_ids": input_ids, "labels": labels}

    @staticmethod
    def _classification_batch(features: list[Mapping[str, Any]]) -> dict[str, Any]:
        token_groups: list[list[int]] = []
        option_loss_token_counts: list[int] = []
        row_option_counts: list[int] = []
        labels: list[int] = []
        for feature in features:
            row_input_ids = feature["input_ids"]
            row_attention_mask = feature.get("attention_mask")
            row_option_lens = feature["option_loss_token_counts"]
            if not _is_nested_rows(row_input_ids):
                raise ValueError("prompt classification input_ids must be nested")
            row_count = len(row_input_ids)
            if "row_option_counts" not in feature:
                raise ValueError("classification features require row_option_counts")
            if int(feature["row_option_counts"]) != row_count:
                raise ValueError("row_option_counts must match candidate count")
            if len(row_option_lens) != row_count:
                raise ValueError("option_loss_token_counts must match candidate count")
            row_option_counts.append(row_count)
            labels.append(int(feature["labels"]))
            for option_index, ids in enumerate(row_input_ids):
                mask = (
                    None
                    if row_attention_mask is None
                    else row_attention_mask[option_index]
                )
                active_ids, _ = _active_token_row(
                    ids,
                    mask,
                )
                token_groups.append(active_ids)
                option_loss_token_counts.append(int(row_option_lens[option_index]))
        return {
            "input_ids": token_groups,
            "option_loss_token_counts": option_loss_token_counts,
            "row_option_counts": row_option_counts,
            "labels": labels,
        }


def _active_token_row(
    input_ids: Sequence[int],
    attention_mask: Sequence[int] | None,
) -> tuple[list[int], list[int] | None]:
    if attention_mask is None:
        if not input_ids:
            raise ValueError("input_ids must contain at least one active token")
        if isinstance(input_ids, list):
            return input_ids, None
        return [int(token) for token in input_ids], None
    else:
        if len(attention_mask) != len(input_ids):
            raise ValueError("attention_mask must match input_ids")
        active_indices = [
            index for index, active in enumerate(attention_mask) if int(active) != 0
        ]
    if not active_indices:
        raise ValueError("input_ids must contain at least one active token")
    if len(active_indices) == len(input_ids) and isinstance(input_ids, list):
        return input_ids, None
    return [int(input_ids[index]) for index in active_indices], active_indices


def _truncate_prompt_target(
    prompt_ids: list[int],
    target_ids: list[int],
    *,
    max_length: int,
) -> tuple[list[int], list[int]]:
    if max_length < 2:
        raise ValueError("target-LM max_length must be at least 2")
    if len(target_ids) >= max_length:
        return prompt_ids[:1], target_ids[: max_length - 1]
    prompt_budget = max_length - len(target_ids)
    return prompt_ids[:prompt_budget], target_ids


def _truncate_option_prompt(
    input_ids: list[int],
    *,
    option_loss_token_count: int,
    max_length: int,
) -> tuple[list[int], int]:
    if max_length < 2:
        raise ValueError("prompt classification max_length must be at least 2")
    option_loss_token_count = int(option_loss_token_count)
    if len(input_ids) <= max_length:
        return input_ids, option_loss_token_count
    if option_loss_token_count >= max_length:
        return input_ids[-max_length:], max_length
    prompt_budget = max_length - option_loss_token_count
    return input_ids[:prompt_budget] + input_ids[
        -option_loss_token_count:
    ], option_loss_token_count


def _encode(
    tokenizer: Any,
    text: str,
    *,
    max_length: int | None,
    truncation: bool,
    add_special_tokens: bool,
) -> list[int]:
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=bool(truncation),
        add_special_tokens=bool(add_special_tokens),
        padding=False,
    )
    return [int(item) for item in encoded["input_ids"]]


def _int_rows(rows: Sequence[Sequence[Any]]) -> list[list[int]]:
    return [[int(item) for item in row] for row in rows]


def _is_nested_rows(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    )
