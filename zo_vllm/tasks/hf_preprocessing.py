"""Task-specific Hugging Face dataset preprocessing for ZO experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from datasets import Dataset, load_dataset

from zo_vllm.core.binary_option_objective import option_lengths
from zo_vllm.tasks.boolq import dataset_to_boolq_rows, encode_boolq_vllm_prompts
from zo_vllm.tasks.squad import dataset_to_squad_rows, encode_squad_train_prompts
from zo_vllm.tasks.sst2 import dataset_to_sst2_rows, encode_sst2_vllm_prompts
from zo_vllm.tasks.superglue import (
    SUPERGLUE_OBJECTIVE_TO_TASK,
    dataset_to_superglue_rows,
    encode_superglue_option_prompts,
)
from zo_vllm.tasks.superglue.record import (
    OBJECTIVE_NAME as RECORD_NLL_OBJECTIVE,
)
from zo_vllm.tasks.superglue.record import encode_gold_prompts


IGNORE_INDEX = -100


def build_sst2_prompt_classification_preprocess(
    tokenizer: Any,
    *,
    sentence_field: str = "sentence",
    label_field: str = "label",
    max_length: int | None = None,
    truncation: bool = True,
    add_special_tokens: bool = True,
) -> Callable[[Mapping[str, Sequence[Any]]], dict[str, list[Any]]]:
    """Return the MeZO-style SST-2 prompt classification map function."""

    verbalizers = ("terrible", "great")

    def preprocess(examples: Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
        option_input_ids: list[list[list[int]]] = []
        option_attention_mask: list[list[list[int]]] = []
        option_lens: list[list[int]] = []
        labels: list[int] = []
        num_options: list[int] = []
        for sentence_value, label_value in zip(
            examples[sentence_field], examples[label_field]
        ):
            sentence = str(sentence_value).strip()
            stem = f"{sentence} It was"
            stem_ids = _encode(
                tokenizer,
                stem,
                add_special_tokens=bool(add_special_tokens),
            )
            row_ids: list[list[int]] = []
            row_masks: list[list[int]] = []
            row_option_lens: list[int] = []
            for verbalizer in verbalizers:
                ids = _encode(
                    tokenizer,
                    f"{stem} {verbalizer}",
                    add_special_tokens=bool(add_special_tokens),
                )
                option_len = len(ids) - len(stem_ids)
                if option_len <= 0:
                    raise ValueError("SST-2 verbalizer produced an empty option")
                if max_length is not None and bool(truncation):
                    ids, option_len = _truncate_option_prompt(
                        ids,
                        option_len=option_len,
                        max_length=int(max_length),
                    )
                row_ids.append(ids)
                row_masks.append([1] * len(ids))
                row_option_lens.append(int(option_len))
            option_input_ids.append(row_ids)
            option_attention_mask.append(row_masks)
            option_lens.append(row_option_lens)
            labels.append(int(label_value))
            num_options.append(len(verbalizers))
        return {
            "input_ids": option_input_ids,
            "attention_mask": option_attention_mask,
            "option_loss_token_counts": option_lens,
            "row_option_counts": num_options,
            "labels": labels,
        }

    return preprocess


def load_sst2_prompt_classification_datasets(
    tokenizer: Any,
    *,
    data_seed: int,
    num_train: int,
    num_dev: int,
) -> tuple[Dataset, Dataset]:
    """Load, split, and map SST-2 into HF prompt-classification features."""

    raw_train = load_dataset("glue", "sst2", split="train")
    requested = int(num_train) + int(num_dev)
    if requested > len(raw_train):
        raise ValueError(
            f"SST-2 train split has {len(raw_train)} rows, requested {requested}"
        )
    selected = raw_train.shuffle(seed=int(data_seed)).select(range(requested))
    train_raw = selected.select(range(int(num_train)))
    dev_raw = selected.select(range(int(num_train), requested))
    preprocess = build_sst2_prompt_classification_preprocess(tokenizer)

    def encode(dataset: Dataset) -> Dataset:
        return dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing SST-2 prompt classification",
        )

    return encode(train_raw), encode(dev_raw)


def build_objective_hf_dataset(
    rows: Sequence[Any],
    tokenizer: Any,
    *,
    objective_name: str,
    max_length: int,
    max_new_tokens: int = 50,
) -> Dataset:
    """Encode a registered experiment objective into ragged HF features."""

    objective = str(objective_name)
    normalized_rows = _objective_rows(rows, objective=objective)
    if objective in {"sst2_classification", "boolq_classification"}:
        encode = (
            encode_sst2_vllm_prompts
            if objective == "sst2_classification"
            else encode_boolq_vllm_prompts
        )
        stem_ids, negative_ids, positive_ids, labels = encode(normalized_rows, tokenizer)
        negative_lens, positive_lens = option_lengths(
            stem_ids, negative_ids, positive_ids
        )
        return Dataset.from_list(
            [
                _classification_feature(
                    [negative_ids[index], positive_ids[index]],
                    [negative_lens[index], positive_lens[index]],
                    label=int(labels[index]),
                )
                for index in range(len(rows))
            ]
        )
    if objective in SUPERGLUE_OBJECTIVE_TO_TASK and objective != RECORD_NLL_OBJECTIVE:
        encoded = encode_superglue_option_prompts(normalized_rows, tokenizer)
        return Dataset.from_list(
            [
                _classification_feature(
                    encoded.option_ids[index],
                    encoded.option_lens[index],
                    label=int(encoded.labels[index]),
                )
                for index in range(len(rows))
            ]
        )
    if objective == "squad_nll":
        token_groups, suffix_lens = encode_squad_train_prompts(
            list(normalized_rows),
            tokenizer,
            max_length=int(max_length),
            max_new_tokens=int(max_new_tokens),
        )
    elif objective == RECORD_NLL_OBJECTIVE:
        token_groups, suffix_lens = encode_gold_prompts(
            normalized_rows,
            tokenizer,
            max_length=int(max_length),
            max_new_tokens=int(max_new_tokens),
        )
    else:
        raise ValueError(f"unsupported objective: {objective}")
    return Dataset.from_list(
        [
            _masked_lm_feature(token_ids, suffix_len)
            for token_ids, suffix_len in zip(token_groups, suffix_lens)
        ]
    )


def _objective_rows(rows: Sequence[Any], *, objective: str) -> list[Any]:
    values = list(rows)
    if not values or not isinstance(values[0], Mapping):
        return values
    if objective == "sst2_classification":
        return dataset_to_sst2_rows(Dataset.from_list(values))
    if objective == "boolq_classification":
        return dataset_to_boolq_rows(Dataset.from_list(values))
    if objective == "squad_nll":
        return dataset_to_squad_rows(Dataset.from_list(values))
    task_name = SUPERGLUE_OBJECTIVE_TO_TASK.get(objective)
    if task_name is not None:
        return dataset_to_superglue_rows(values, task_name=task_name)
    return values


def _classification_feature(
    option_ids: Sequence[Sequence[int]],
    option_lens: Sequence[int],
    *,
    label: int,
) -> dict[str, Any]:
    token_groups = [[int(token) for token in group] for group in option_ids]
    loss_lens = [int(value) for value in option_lens]
    if not token_groups or len(token_groups) != len(loss_lens):
        raise ValueError("classification options and lengths must match")
    return {
        "input_ids": token_groups,
        "option_loss_token_counts": loss_lens,
        "row_option_counts": len(token_groups),
        "labels": int(label),
    }


def _masked_lm_feature(
    token_ids: Sequence[int],
    suffix_len: int,
) -> dict[str, Any]:
    tokens = [int(token) for token in token_ids]
    keep = max(0, min(int(suffix_len), len(tokens)))
    if keep <= 0:
        raise ValueError("masked-LM feature requires at least one loss token")
    return {
        "input_ids": tokens,
        "labels": [IGNORE_INDEX] * (len(tokens) - keep) + tokens[-keep:],
    }


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return [
            int(token) for token in encode(text, add_special_tokens=add_special_tokens)
        ]
    encoded = tokenizer(
        text,
        padding=False,
        truncation=False,
        add_special_tokens=add_special_tokens,
    )
    return [int(token) for token in encoded["input_ids"]]


def _truncate_option_prompt(
    ids: list[int], *, option_len: int, max_length: int
) -> tuple[list[int], int]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    keep_option = min(int(option_len), int(max_length))
    return ids[-int(max_length) :], keep_option


__all__ = [
    "build_objective_hf_dataset",
    "build_sst2_prompt_classification_preprocess",
    "load_sst2_prompt_classification_datasets",
]
