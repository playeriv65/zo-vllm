"""Token-container normalization at public runtime boundaries."""

from __future__ import annotations

from collections.abc import Sequence


def borrow_or_copy_token_groups(
    rows: Sequence[Sequence[int]] | None,
) -> list[list[int]] | None:
    """Borrow normalized lists and copy other sequence implementations once."""

    if rows is None:
        return None
    if isinstance(rows, list) and all(isinstance(row, list) for row in rows):
        return rows
    return [[int(item) for item in row] for row in rows]


def borrow_or_copy_int_list(values: Sequence[int] | None) -> list[int] | None:
    """Borrow a normalized integer list or materialize an external sequence."""

    if values is None:
        return None
    if isinstance(values, list):
        return values
    return [int(value) for value in values]


def borrow_or_copy_token_group_batches(
    batches: Sequence[Sequence[Sequence[int]]],
) -> list[list[list[int]]]:
    """Apply token-group normalization without copying normalized batches."""

    if isinstance(batches, list) and all(
        isinstance(batch, list) and all(isinstance(row, list) for row in batch)
        for batch in batches
    ):
        return batches
    return [borrow_or_copy_token_groups(batch) or [] for batch in batches]


__all__ = [
    "borrow_or_copy_int_list",
    "borrow_or_copy_token_group_batches",
    "borrow_or_copy_token_groups",
]
