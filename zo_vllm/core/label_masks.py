"""Helpers for HF-style causal-LM label masks."""

from __future__ import annotations

from collections.abc import Sequence


def suffix_lm_labels(
    token_groups: Sequence[Sequence[int]],
    suffix_lengths: Sequence[int],
) -> list[list[int]]:
    """Build labels that score only each request suffix after causal shift."""

    if len(token_groups) != len(suffix_lengths):
        raise ValueError("token_groups and suffix_lengths must have same length")
    labels: list[list[int]] = []
    for index, (token_ids, suffix_len) in enumerate(zip(token_groups, suffix_lengths)):
        ids = [int(item) for item in token_ids]
        suffix_len_i = int(suffix_len)
        if suffix_len_i <= 0:
            raise ValueError(f"suffix_lengths must be positive; index={index}")
        if suffix_len_i > len(ids) - 1:
            raise ValueError(
                "suffix_lengths cannot exceed prompt length minus one; "
                f"index={index}, suffix_len={suffix_len_i}, prompt_len={len(ids)}"
            )
        row = [-100] * len(ids)
        start = len(ids) - suffix_len_i
        row[start:] = ids[start:]
        labels.append(row)
    return labels


__all__ = ["suffix_lm_labels"]
