"""Task row loading and ZO batch construction for ZO training loops."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from datasets import load_dataset

from zo_vllm.core.binary_option_objective import (
    accuracy_from_option_nll,
    accuracy_from_multi_option_nll,
    classification_loss_from_option_nll,
    classification_loss_from_multi_option_nll,
    option_lengths,
    predictions_from_multi_option_nll,
    regroup_flat_option_values,
)
from zo_vllm.core.label_masks import suffix_lm_labels
from zo_vllm.engine import TokenScoreResult
from zo_vllm.tasks import TaskConfig, get_task
from zo_vllm.tasks.boolq import dataset_to_boolq_rows, encode_boolq_vllm_prompts
from zo_vllm.tasks.squad import dataset_to_squad_rows, encode_squad_train_prompts
from zo_vllm.tasks.sst2 import dataset_to_sst2_rows, encode_sst2_vllm_prompts
from zo_vllm.tasks.superglue import (
    SUPERGLUE_OBJECTIVE_TO_TASK,
    dataset_to_superglue_rows,
    encode_superglue_option_prompts,
)
from zo_vllm.tasks.superglue.record import OBJECTIVE_NAME as RECORD_NLL_OBJECTIVE
from zo_vllm.tasks.superglue.record import (
    encode_gold_prompts as encode_record_gold_prompts,
)
from .direction import SubspaceTokenProbeBatch, TokenProbeBatch


SUPPORTED_OBJECTIVES = {
    "sst2_classification",
    "boolq_classification",
    "squad_nll",
    "prompt_nll",
    *SUPERGLUE_OBJECTIVE_TO_TASK.keys(),
}


@dataclass(frozen=True)
class BinaryClassificationScoringBatch:
    """Tokenized binary-option classification batch with scoring helpers."""

    token_id_groups: list[list[int]]
    labels: list[int]
    loss_token_lens: list[int]

    @property
    def token_labels(self) -> list[list[int]]:
        return suffix_lm_labels(self.token_id_groups, self.loss_token_lens)

    @property
    def num_rows(self) -> int:
        return len(self.labels)

    def loss(self, score: TokenScoreResult) -> float:
        n = self.num_rows
        return classification_loss_from_option_nll(
            score.request_nll[:n],
            score.request_nll[n : 2 * n],
            self.labels,
        )

    def accuracy(self, score: TokenScoreResult) -> float:
        n = self.num_rows
        return accuracy_from_option_nll(
            score.request_nll[:n],
            score.request_nll[n : 2 * n],
            self.labels,
        )

    def predictions(self, score: TokenScoreResult) -> list[int]:
        return [
            1 if pos_nll < neg_nll else 0
            for neg_nll, pos_nll in self.row_request_nlls(score)
        ]

    def row_request_nlls(self, score: TokenScoreResult) -> list[list[float]]:
        n = self.num_rows
        if len(score.request_nll) < 2 * n:
            raise ValueError(
                "binary score does not contain two option NLLs per row: "
                f"got {len(score.request_nll)} requests for {n} rows"
            )
        return [
            [float(score.request_nll[index]), float(score.request_nll[n + index])]
            for index in range(n)
        ]

    def row_request_num_tokens(self, score: TokenScoreResult) -> list[list[int]]:
        n = self.num_rows
        if len(score.request_num_tokens) < 2 * n:
            raise ValueError(
                "binary score does not contain two option token counts per row: "
                f"got {len(score.request_num_tokens)} requests for {n} rows"
            )
        return [
            [
                int(score.request_num_tokens[index]),
                int(score.request_num_tokens[n + index]),
            ]
            for index in range(n)
        ]

    def per_row_losses(self, score: TokenScoreResult) -> list[float]:
        return _classification_losses_from_option_nlls(
            self.row_request_nlls(score),
            self.labels,
        )


@dataclass(frozen=True)
class OptionClassificationScoringBatch:
    """Tokenized variable-width option classification batch."""

    token_id_groups: list[list[int]]
    labels: list[int]
    label_sets: list[tuple[int, ...]]
    option_counts: list[int]
    loss_token_lens: list[int]

    @property
    def token_labels(self) -> list[list[int]]:
        return suffix_lm_labels(self.token_id_groups, self.loss_token_lens)

    @property
    def num_rows(self) -> int:
        return len(self.labels)

    def _option_nll(self, score: TokenScoreResult) -> list[list[float]]:
        return regroup_flat_option_values(
            score.request_nll,
            self.option_counts,
        )

    def loss(self, score: TokenScoreResult) -> float:
        return classification_loss_from_multi_option_nll(
            self._option_nll(score),
            self.labels,
        )

    def accuracy(self, score: TokenScoreResult) -> float:
        return accuracy_from_multi_option_nll(
            self._option_nll(score),
            self.label_sets,
        )

    def predictions(self, score: TokenScoreResult) -> list[int]:
        return predictions_from_multi_option_nll(self._option_nll(score))

    def row_request_nlls(self, score: TokenScoreResult) -> list[list[float]]:
        return self._option_nll(score)

    def row_request_num_tokens(self, score: TokenScoreResult) -> list[list[int]]:
        return regroup_flat_option_values(
            [int(value) for value in score.request_num_tokens],
            self.option_counts,
        )

    def per_row_losses(self, score: TokenScoreResult) -> list[float]:
        return _classification_losses_from_option_nlls(
            self._option_nll(score),
            self.labels,
        )


@dataclass(frozen=True)
class MaskedLMScoringBatch:
    """Tokenized causal-LM batch whose labels already encode the loss mask."""

    token_id_groups: list[list[int]]
    loss_token_lens: list[int]

    @property
    def token_labels(self) -> list[list[int]]:
        return suffix_lm_labels(self.token_id_groups, self.loss_token_lens)

    @property
    def num_rows(self) -> int:
        return len(self.token_id_groups)

    def loss(self, score: TokenScoreResult) -> float:
        return float(score.loss)

    def row_request_nlls(self, score: TokenScoreResult) -> list[list[float]]:
        if len(score.request_nll) != self.num_rows:
            raise ValueError(
                "masked-LM score must contain one request per row: "
                f"got {len(score.request_nll)} requests for {self.num_rows} rows"
            )
        return [[float(value)] for value in score.request_nll]

    def row_request_num_tokens(self, score: TokenScoreResult) -> list[list[int]]:
        if len(score.request_num_tokens) != self.num_rows:
            raise ValueError(
                "masked-LM score must contain one token count per row: "
                f"got {len(score.request_num_tokens)} requests for {self.num_rows} rows"
            )
        return [[int(value)] for value in score.request_num_tokens]

    def per_row_losses(self, score: TokenScoreResult) -> list[float]:
        return [float(value) for value in score.request_nll]


def _classification_losses_from_option_nlls(
    option_nlls: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> list[float]:
    losses: list[float] = []
    if len(option_nlls) != len(labels):
        raise ValueError(f"got {len(option_nlls)} option rows for {len(labels)} labels")
    for row_nlls, label in zip(option_nlls, labels):
        if not row_nlls:
            raise ValueError("classification row has no options")
        label_i = int(label)
        if label_i < 0 or label_i >= len(row_nlls):
            raise ValueError(
                f"label index {label_i} out of range for {len(row_nlls)} options"
            )
        logits = [-float(value) for value in row_nlls]
        max_logit = max(logits)
        log_denom = max_logit + math.log(
            sum(math.exp(logit - max_logit) for logit in logits)
        )
        losses.append(float(log_denom - logits[label_i]))
    return losses


def resolve_objective_name(
    objective_name: str | None = None,
    *,
    dataset_name: str | None = None,
    dataset_config_name: str | None = None,
    task_name: str | None = None,
) -> str:
    """Resolve HF-like dataset/task flags to the internal training objective."""

    if task_name:
        return _objective_from_task_name(task_name)
    if dataset_name:
        return _objective_from_dataset_name(
            dataset_name,
            dataset_config_name=dataset_config_name,
        )
    if objective_name:
        return _validate_objective_name(objective_name)
    return "sst2_classification"


def _validate_objective_name(value: str) -> str:
    if value not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported objective_name: {value}")
    return value


def _objective_from_task_name(value: str) -> str:
    task = str(value).strip().lower().replace("-", "_")
    if task in {"sst2", "glue_sst2"}:
        return "sst2_classification"
    if task in {"boolq", "boolq_classification"}:
        return "boolq_classification"
    if task in {"squad", "squad_nll"}:
        return "squad_nll"
    if task == "prompt_nll":
        return "prompt_nll"
    if task.startswith("superglue_"):
        task = task.removeprefix("superglue_")
    if task.startswith("super_glue_"):
        task = task.removeprefix("super_glue_")
    superglue_objective = f"superglue_{task}_classification"
    if superglue_objective in SUPERGLUE_OBJECTIVE_TO_TASK:
        return superglue_objective
    record_objective = "superglue_record_nll"
    if (
        task in {"record", "record_nll"}
        and record_objective in SUPERGLUE_OBJECTIVE_TO_TASK
    ):
        return record_objective
    raise ValueError(f"unsupported task name or objective: {value}")


def _objective_from_dataset_name(
    dataset_name: str,
    *,
    dataset_config_name: str | None,
) -> str:
    dataset = str(dataset_name).strip().lower().replace("-", "_")
    dataset = dataset.replace("/", "_")
    config = (
        None
        if dataset_config_name is None
        else str(dataset_config_name).strip().lower().replace("-", "_")
    )
    if dataset in {"glue"}:
        if config == "sst2":
            return "sst2_classification"
        raise ValueError(f"unsupported GLUE dataset config: {dataset_config_name}")
    if dataset in {"super_glue", "superglue"}:
        if not config:
            raise ValueError("--dataset-config-name is required for super_glue")
        return _objective_from_task_name(f"superglue_{config}")
    if dataset in {"stanfordnlp_sst2", "sst2"}:
        return "sst2_classification"
    if dataset in {"google_boolq", "boolq"}:
        return "boolq_classification"
    if dataset in {"squad"}:
        return "squad_nll"
    raise ValueError(
        "unsupported dataset. Pass --train-objective for a custom objective or "
        f"add a task adapter for dataset={dataset_name!r}"
    )


def prepare_prompt_nll_sst2_data(seed: int, num_samples: int = 1000) -> list[str]:
    np.random.seed(seed)
    dataset = load_dataset("glue", "sst2", split="train")
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)
    return [f"{item['sentence']} It was" for item in dataset]


def tokenize_prompts(tokenizer: Any, prompts: Sequence[str]) -> list[list[int]]:
    return [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]


def load_objective_rows(
    objective_name: str,
    *,
    data_seed: int,
    num_train: int,
    num_dev: int,
    num_eval: int,
    shuffle_impl: str | None = None,
):
    objective_name = resolve_objective_name(objective_name)
    if objective_name == "sst2_classification":
        task = get_task("sst2")
        convert = dataset_to_sst2_rows
    elif objective_name == "boolq_classification":
        task = get_task("boolq")
        convert = dataset_to_boolq_rows
    elif objective_name == "squad_nll":
        task = get_task("squad")
        convert = dataset_to_squad_rows
    elif objective_name in SUPERGLUE_OBJECTIVE_TO_TASK:
        task_name = SUPERGLUE_OBJECTIVE_TO_TASK[objective_name]
        task = get_task(task_name)

        def convert(dataset, *, task_name=task_name):
            return dataset_to_superglue_rows(dataset, task_name=task_name)
    else:
        raise ValueError(f"unsupported task objective: {objective_name}")
    splits = task.load_splits(
        TaskConfig(
            name=task.name,
            num_train=num_train,
            num_dev=num_dev,
            num_eval=num_eval,
            data_seed=data_seed,
            shuffle_impl=shuffle_impl,
        )
    )
    return convert(splits.train), convert(splits.dev), convert(splits.eval)


def build_prompt_nll_zo_batch(
    token_groups: Sequence[Sequence[int]],
) -> TokenProbeBatch:
    return TokenProbeBatch(token_id_groups=token_groups)


def _zo_batch_from_scoring_batch(scoring_batch: Any) -> TokenProbeBatch:
    return TokenProbeBatch(
        token_id_groups=scoring_batch.token_id_groups,
        loss_token_lens=scoring_batch.loss_token_lens,
        objective=scoring_batch.loss,
    )


def _build_binary_classification_scoring_batch(
    rows: Sequence[Any],
    tokenizer: Any,
    *,
    objective_name: str,
) -> BinaryClassificationScoringBatch:
    if objective_name == "sst2_classification":
        stem_ids, neg_ids, pos_ids, labels = encode_sst2_vllm_prompts(rows, tokenizer)
    elif objective_name == "boolq_classification":
        stem_ids, neg_ids, pos_ids, labels = encode_boolq_vllm_prompts(rows, tokenizer)
    else:
        raise ValueError(f"unsupported binary objective: {objective_name}")
    neg_lens, pos_lens = option_lengths(stem_ids, neg_ids, pos_ids)
    token_groups = neg_ids + pos_ids
    suffix_lengths = neg_lens + pos_lens
    return BinaryClassificationScoringBatch(
        token_id_groups=token_groups,
        loss_token_lens=suffix_lengths,
        labels=labels,
    )


def _build_classification_scoring_batch(
    rows: Sequence[Any],
    tokenizer: Any,
    *,
    objective_name: str,
) -> BinaryClassificationScoringBatch | OptionClassificationScoringBatch:
    if objective_name in {"sst2_classification", "boolq_classification"}:
        return _build_binary_classification_scoring_batch(
            rows,
            tokenizer,
            objective_name=objective_name,
        )
    if objective_name in SUPERGLUE_OBJECTIVE_TO_TASK:
        return _build_option_classification_scoring_batch(
            rows,
            tokenizer,
            objective_name=objective_name,
        )
    raise ValueError(f"unsupported classification objective: {objective_name}")


def _build_masked_lm_scoring_batch(
    rows: Sequence[Any],
    tokenizer: Any,
    *,
    objective_name: str,
    max_length: int,
    max_new_tokens: int = 50,
) -> MaskedLMScoringBatch:
    if objective_name == "squad_nll":
        token_groups, suffix_lengths = encode_squad_train_prompts(
            rows,
            tokenizer,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
    elif objective_name == RECORD_NLL_OBJECTIVE:
        token_groups, suffix_lengths = encode_record_gold_prompts(
            rows,
            tokenizer,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"unsupported masked-LM objective: {objective_name}")
    return MaskedLMScoringBatch(
        token_id_groups=token_groups,
        loss_token_lens=suffix_lengths,
    )


def _build_option_classification_scoring_batch(
    rows: Sequence[Any],
    tokenizer: Any,
    *,
    objective_name: str,
) -> OptionClassificationScoringBatch:
    if objective_name in SUPERGLUE_OBJECTIVE_TO_TASK:
        encoded = encode_superglue_option_prompts(rows, tokenizer)
    else:
        raise ValueError(f"unsupported option objective: {objective_name}")
    return OptionClassificationScoringBatch(
        token_id_groups=encoded.flat_option_ids,
        loss_token_lens=encoded.flat_option_lens,
        labels=encoded.labels,
        label_sets=encoded.label_sets,
        option_counts=encoded.option_counts,
    )


def build_zo_task_batch(
    rows: Sequence[Any],
    tokenizer: Any,
    *,
    objective_name: str,
    max_length: int | None = None,
    max_new_tokens: int = 50,
) -> TokenProbeBatch:
    if objective_name == "prompt_nll":
        return build_prompt_nll_zo_batch(rows)
    if objective_name in {"squad_nll", RECORD_NLL_OBJECTIVE}:
        if max_length is None:
            raise ValueError(f"max_length is required for {objective_name}")
        return _zo_batch_from_scoring_batch(
            _build_masked_lm_scoring_batch(
                rows,
                tokenizer,
                objective_name=objective_name,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
        )
    if objective_name in {"sst2_classification", "boolq_classification"}:
        return _zo_batch_from_scoring_batch(
            _build_classification_scoring_batch(
                rows,
                tokenizer,
                objective_name=objective_name,
            )
        )
    if objective_name in SUPERGLUE_OBJECTIVE_TO_TASK:
        return _zo_batch_from_scoring_batch(
            _build_classification_scoring_batch(
                rows,
                tokenizer,
                objective_name=objective_name,
            )
        )
    raise ValueError(f"unsupported generic ZO objective: {objective_name}")


def with_subspace_token_batch_factory(
    batch: TokenProbeBatch,
    subspace_token_id_group_batch_factory,
    *,
    subspace_num_rows: int | None = None,
) -> SubspaceTokenProbeBatch:
    return SubspaceTokenProbeBatch(
        token_id_groups=batch.token_id_groups,
        loss_token_lens=batch.loss_token_lens,
        labels=batch.labels,
        objective=batch.objective,
        subspace_num_rows=subspace_num_rows,
        subspace_token_id_group_batch_factory=subspace_token_id_group_batch_factory,
    )


def cyclic_rows(rows: Sequence[Any], *, start: int, count: int) -> list[Any]:
    if not rows:
        raise ValueError("rows must be non-empty")
    if int(count) <= 0:
        raise ValueError("count must be positive")
    size = len(rows)
    return [rows[(int(start) + offset) % size] for offset in range(int(count))]


def split_rows_for_agzo_subspace(
    rows: Sequence[Any],
    *,
    chunk_size: int = 16,
) -> list[list[Any]]:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if len(rows) <= int(chunk_size):
        return [list(rows)]
    return [
        list(rows[start : start + int(chunk_size)])
        for start in range(0, len(rows), int(chunk_size))
    ]
