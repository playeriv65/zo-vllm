"""SQuAD task adapter, prompt encoding, and metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import string
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from .base import TaskConfig, TaskSplits, build_task_data_collator, shuffled_select


@dataclass(frozen=True)
class SquadRow:
    title: str
    context: str
    question: str
    answers: tuple[str, ...]


class SquadTaskAdapter:
    name = "squad"
    official_task_name = "SQuAD"
    vllm_train_objective = "squad_nll"
    metric_names = ("eval_loss", "eval_f1", "eval_em")
    primary_metric = "eval_loss"
    greater_is_better = False
    supports_official_lozo = True

    def load_splits(self, cfg: TaskConfig) -> TaskSplits:
        raw = load_dataset("squad")
        train_raw = raw["train"].filter(lambda item: bool(item["answers"]["text"]))
        validation_raw = raw["validation"].filter(lambda item: bool(item["answers"]["text"]))
        train_dev = shuffled_select(
            train_raw,
            seed=cfg.data_seed,
            num=int(cfg.num_train) + int(cfg.num_dev),
            shuffle_impl=cfg.shuffle_impl,
        )
        train = train_dev.select(range(min(int(cfg.num_train), len(train_dev))))
        dev_start = min(int(cfg.num_train), len(train_dev))
        dev_end = min(dev_start + int(cfg.num_dev), len(train_dev))
        dev = train_dev.select(range(dev_start, dev_end))
        eval_split = shuffled_select(
            validation_raw,
            seed=cfg.data_seed,
            num=cfg.num_eval,
            shuffle_impl=cfg.shuffle_impl,
        )
        return TaskSplits(train=train, dev=dev, eval=eval_split)

    def official_lozo_args(self, cfg: TaskConfig) -> list[str]:
        return ["--max_new_tokens", str(cfg.max_new_tokens)]

    def vllm_args(self, cfg: TaskConfig) -> list[str]:
        return [
            "--train-objective",
            self.vllm_train_objective,
            "--max-new-tokens",
            str(cfg.max_new_tokens),
        ]

    def data_collator(self, tokenizer: Any, cfg: TaskConfig | None = None):
        return build_task_data_collator(
            tokenizer,
            objective_name=self.vllm_train_objective,
            row_converter=dataset_to_squad_rows,
            cfg=cfg,
        )


def squad_stem(row: SquadRow) -> str:
    question = row.question.strip()
    return (
        f"Title: {row.title}\n"
        f"Context: {row.context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def squad_verbalized(row: SquadRow) -> str:
    return f"{squad_stem(row)} {row.answers[0]}\n"


def dataset_to_squad_rows(dataset: HFDataset) -> list[SquadRow]:
    """Convert HF Dataset rows into SQuAD scoring rows."""

    rows = []
    for item in dataset:
        answers = tuple(str(answer) for answer in item["answers"]["text"])
        if not answers:
            continue
        rows.append(
            SquadRow(
                title=str(item["title"]),
                context=str(item["context"]),
                question=str(item["question"]),
                answers=answers,
            )
        )
    return rows


def encode_squad_train_prompts(
    rows: list[SquadRow],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    max_new_tokens: int,
) -> tuple[list[list[int]], list[int]]:
    token_groups: list[list[int]] = []
    answer_lens: list[int] = []
    effective_max_length = max_length - max_new_tokens
    if effective_max_length <= 0:
        raise ValueError("max_length must be larger than max_new_tokens for SQuAD")
    for row in rows:
        stem_ids = tokenizer.encode(squad_stem(row), add_special_tokens=True)
        full_ids = tokenizer.encode(squad_verbalized(row), add_special_tokens=True)
        answer_len = len(full_ids) - len(stem_ids)
        if answer_len <= 0:
            raise ValueError("SQuAD answer must add at least one token")
        token_groups.append(left_truncate(full_ids, effective_max_length, tokenizer))
        answer_lens.append(answer_len)
    return token_groups, answer_lens


def encode_squad_generation_prompts(
    rows: list[SquadRow],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    max_new_tokens: int,
) -> list[list[int]]:
    effective_max_length = max_length - max_new_tokens
    if effective_max_length <= 0:
        raise ValueError("max_length must be larger than max_new_tokens for SQuAD")
    return [
        left_truncate(
            tokenizer.encode(squad_stem(row), add_special_tokens=True),
            effective_max_length,
            tokenizer,
        )
        for row in rows
    ]


def left_truncate(
    token_ids: list[int],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    if len(token_ids) <= max_length:
        return token_ids
    add_bos = bool(getattr(tokenizer, "add_bos_token", False))
    if add_bos and token_ids:
        return token_ids[:1] + token_ids[1:][-(max_length - 1) :]
    return token_ids[-max_length:]


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def squad_f1(prediction: str, answers: tuple[str, ...]) -> float:
    scores = []
    prediction_tokens = normalize_answer(prediction).split()
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if num_same == 0 or not prediction_tokens or not answer_tokens:
            scores.append(0.0)
            continue
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(answer_tokens)
        scores.append((2 * precision * recall) / (precision + recall))
    return float(max(scores) if scores else 0.0)


def squad_exact_match(prediction: str, answers: tuple[str, ...]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalize_answer(answer) == normalized for answer in answers))
