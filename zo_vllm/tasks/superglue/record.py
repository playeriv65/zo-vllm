"""SuperGLUE ReCoRD prompt adapter and entity-candidate metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .base import SuperGLUERow, SuperGLUETaskSpec, example_idx


TASK_NAME = "superglue_record"
OBJECTIVE_NAME = "superglue_record_nll"


def row_from_example(example: dict[str, Any]) -> SuperGLUERow:
    candidates = tuple(str(entity) for entity in example["entities"])
    answers = tuple(str(answer) for answer in example["answers"])
    answer_set = set(answers)
    label_indices = tuple(
        index for index, candidate in enumerate(candidates) if candidate in answer_set
    )
    if not label_indices:
        raise ValueError("ReCoRD row has no answer entity among candidates")
    return SuperGLUERow(
        task_name=TASK_NAME,
        data=dict(example),
        candidates=candidates,
        label_indices=label_indices,
        answers=answers,
        idx=example_idx(example),
    )


def stem(row: SuperGLUERow) -> str:
    passage = str(row.data["passage"]).replace("@highlight\n", "- ")
    return f"{passage}\n-"


def verbalized(row: SuperGLUERow, candidate: Any) -> str:
    passage = str(row.data["passage"]).replace("@highlight\n", "- ")
    query = str(row.data["query"]).replace("@placeholder", str(candidate))
    return f"{passage}\n- {query}"


def encode_gold_prompts(
    rows: Sequence[SuperGLUERow],
    tokenizer: Any,
    *,
    max_length: int,
    max_new_tokens: int,
) -> tuple[list[list[int]], list[int]]:
    token_groups: list[list[int]] = []
    answer_lens: list[int] = []
    effective_max_length = int(max_length) - int(max_new_tokens)
    if effective_max_length <= 0:
        raise ValueError("max_length must be larger than max_new_tokens for ReCoRD")
    for row in rows:
        gold = row.answers[0] if row.answers else row.candidates[row.label]
        stem_ids = tokenizer.encode(stem(row), add_special_tokens=True)
        full_ids = tokenizer.encode(verbalized(row, gold), add_special_tokens=True)
        answer_len = len(full_ids) - len(stem_ids)
        if answer_len <= 0:
            raise ValueError("ReCoRD gold answer must add at least one token")
        token_groups.append(left_truncate(full_ids, effective_max_length, tokenizer))
        answer_lens.append(answer_len)
    return token_groups, answer_lens


def left_truncate(
    token_ids: list[int],
    max_length: int,
    tokenizer: Any,
) -> list[int]:
    if len(token_ids) <= max_length:
        return token_ids
    add_bos = bool(getattr(tokenizer, "add_bos_token", False))
    if add_bos and token_ids:
        return token_ids[:1] + token_ids[1:][-(max_length - 1) :]
    return token_ids[-max_length:]


def prediction_metric(
    rows: Sequence[SuperGLUERow],
    predictions: Sequence[int],
) -> dict[str, float]:
    total_em = 0.0
    total_f1 = 0.0
    for row, prediction in zip(rows, predictions):
        candidate = str(row.candidates[int(prediction)])
        total_em += exact_match(candidate, row.answers)
        total_f1 += f1(candidate, row.answers)
    return {
        "em": float(total_em / len(rows)),
        "f1": float(total_f1 / len(rows)),
    }


def exact_match(prediction: str, answers: Sequence[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(answer) for answer in answers))


def f1(prediction: str, answers: Sequence[str]) -> float:
    return max((_token_f1(prediction, answer) for answer in answers), default=0.0)


def _token_f1(prediction: str, answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(answer_tokens)
    return float(2 * precision * recall / (precision + recall))


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    without_punc = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punc)
    return " ".join(without_articles.split())


TASK_SPEC = SuperGLUETaskSpec(
    name=TASK_NAME,
    dataset_config="record",
    official_task_name="ReCoRD",
    objective_name=OBJECTIVE_NAME,
    row_builder=row_from_example,
    stem=stem,
    verbalized=verbalized,
    official_train_as_classification=False,
)
