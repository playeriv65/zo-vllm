from __future__ import annotations

import math
from collections.abc import Sequence


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


def multi_option_lengths(
    stem_ids: Sequence[Sequence[int]],
    option_ids: Sequence[Sequence[Sequence[int]]],
) -> list[list[int]]:
    lengths: list[list[int]] = []
    for row_index, (stem, row_options) in enumerate(zip(stem_ids, option_ids)):
        if not row_options:
            raise ValueError(f"row {row_index} has no answer options")
        row_lengths = [len(option) - len(stem) for option in row_options]
        if min(row_lengths) <= 0:
            raise ValueError(
                "classification verbalizer must add at least one option token; "
                f"bad row={row_index}, lengths={row_lengths}"
            )
        lengths.append(row_lengths)
    if len(lengths) != len(stem_ids):
        raise ValueError(
            "stem_ids and option_ids must have identical row counts; "
            f"got {len(stem_ids)} and {len(option_ids)}"
        )
    return lengths


def classification_loss_from_option_nll(
    neg_nll: list[float],
    pos_nll: list[float],
    labels: list[int],
) -> float:
    _validate_binary_option_inputs(neg_nll, pos_nll, labels)
    total = 0.0
    for neg, pos, label in zip(neg_nll, pos_nll, labels):
        logits = [-float(neg), -float(pos)]
        max_logit = max(logits)
        log_denom = max_logit + math.log(
            math.exp(logits[0] - max_logit) + math.exp(logits[1] - max_logit)
        )
        total += log_denom - logits[int(label)]
    return float(total / len(labels))


def accuracy_from_option_nll(
    neg_nll: list[float],
    pos_nll: list[float],
    labels: list[int],
) -> float:
    _validate_binary_option_inputs(neg_nll, pos_nll, labels)
    correct = 0
    for neg, pos, label in zip(neg_nll, pos_nll, labels):
        pred = 1 if float(pos) < float(neg) else 0
        correct += int(pred == int(label))
    return float(correct / len(labels))


def classification_loss_from_multi_option_nll(
    option_nll: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> float:
    _validate_multi_option_inputs(option_nll, labels)
    total = 0.0
    for row_nll, label in zip(option_nll, labels):
        logits = [-float(value) for value in row_nll]
        max_logit = max(logits)
        log_denom = max_logit + math.log(
            sum(math.exp(logit - max_logit) for logit in logits)
        )
        total += log_denom - logits[int(label)]
    return float(total / len(labels))


def predictions_from_multi_option_nll(
    option_nll: Sequence[Sequence[float]],
) -> list[int]:
    predictions: list[int] = []
    for row_index, row_nll in enumerate(option_nll):
        if not row_nll:
            raise ValueError(f"row {row_index} has no option NLLs")
        predictions.append(min(range(len(row_nll)), key=lambda idx: float(row_nll[idx])))
    return predictions


def accuracy_from_multi_option_nll(
    option_nll: Sequence[Sequence[float]],
    labels: Sequence[int | Sequence[int]],
) -> float:
    _validate_multi_option_inputs(option_nll, labels)
    predictions = predictions_from_multi_option_nll(option_nll)
    correct = 0
    for prediction, label in zip(predictions, labels):
        if isinstance(label, Sequence) and not isinstance(label, (str, bytes)):
            correct += int(int(prediction) in {int(value) for value in label})
        else:
            correct += int(int(prediction) == int(label))
    return float(correct / len(labels))


def regroup_flat_option_values(
    values: Sequence[float],
    option_counts: Sequence[int],
) -> list[list[float]]:
    grouped: list[list[float]] = []
    offset = 0
    for count in option_counts:
        count = int(count)
        if count <= 0:
            raise ValueError(f"option count must be positive, got {count}")
        grouped.append([float(value) for value in values[offset : offset + count]])
        offset += count
    if offset != len(values):
        raise ValueError(
            "option_counts do not cover flat values; "
            f"covered={offset}, values={len(values)}"
        )
    return grouped


def _validate_binary_option_inputs(
    neg_nll: list[float],
    pos_nll: list[float],
    labels: list[int],
) -> None:
    if not labels:
        raise ValueError("labels must not be empty")
    if len(neg_nll) != len(pos_nll) or len(neg_nll) != len(labels):
        raise ValueError(
            "neg_nll, pos_nll, and labels must have identical lengths; "
            f"got {len(neg_nll)}, {len(pos_nll)}, {len(labels)}"
        )
    bad_labels = [
        index for index, label in enumerate(labels) if int(label) not in {0, 1}
    ]
    if bad_labels:
        raise ValueError(f"labels must be 0 or 1; bad indices: {bad_labels[:8]}")
    for name, values in (("neg_nll", neg_nll), ("pos_nll", pos_nll)):
        bad_values = [
            index
            for index, value in enumerate(values)
            if not math.isfinite(float(value))
        ]
        if bad_values:
            raise ValueError(f"{name} must be finite; bad indices: {bad_values[:8]}")


def _validate_multi_option_inputs(
    option_nll: Sequence[Sequence[float]],
    labels: Sequence[int | Sequence[int]],
) -> None:
    if not labels:
        raise ValueError("labels must not be empty")
    if len(option_nll) != len(labels):
        raise ValueError(
            "option_nll and labels must have identical row counts; "
            f"got {len(option_nll)} and {len(labels)}"
        )
    for row_index, (row_nll, label) in enumerate(zip(option_nll, labels)):
        if not row_nll:
            raise ValueError(f"row {row_index} has no option NLLs")
        bad_values = [
            index
            for index, value in enumerate(row_nll)
            if not math.isfinite(float(value))
        ]
        if bad_values:
            raise ValueError(
                f"option_nll values must be finite; row={row_index}, "
                f"bad indices={bad_values[:8]}"
            )
        if isinstance(label, Sequence) and not isinstance(label, (str, bytes)):
            label_values = [int(value) for value in label]
        else:
            label_values = [int(label)]
        bad_labels = [value for value in label_values if value < 0 or value >= len(row_nll)]
        if bad_labels:
            raise ValueError(
                f"label out of range for row {row_index}: labels={label_values}, "
                f"num_options={len(row_nll)}"
            )
