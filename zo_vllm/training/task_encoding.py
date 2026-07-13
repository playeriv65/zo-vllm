"""HF-like tokenizer and task collation helpers for ZO training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer

from zo_vllm.tasks.tokenization import (
    configure_opt_tokenizer,
    tokenizer_from_pretrained_kwargs,
)

from .direction import TokenProbeBatch
from .task_batches import (
    build_zo_task_batch,
    cyclic_rows,
    split_rows_for_agzo_subspace,
    with_subspace_token_batch_factory,
)


@dataclass(frozen=True)
class ZOTaskEncodingConfig:
    """Tokenizer-owned task encoding settings for row-to-ZO batch collation."""

    objective_name: str
    max_length: int | None = None
    max_new_tokens: int = 50
    direction_provider: str = "lozo"
    agzo_kappa: int = 1
    agzo_subspace_chunk_size: int = 16

    def __post_init__(self) -> None:
        if int(self.max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if int(self.agzo_kappa) <= 0:
            raise ValueError("agzo_kappa must be positive")
        if int(self.agzo_subspace_chunk_size) <= 0:
            raise ValueError("agzo_subspace_chunk_size must be positive")


def build_tokenizer(
    model_name: str,
    *,
    opt_bos_mode: str = "native",
    use_fast: bool = False,
):
    """Build the tokenizer used by ZO task encoders."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast,
        **tokenizer_from_pretrained_kwargs(
            model_name,
            opt_bos_mode=opt_bos_mode,
        ),
    )
    configure_opt_tokenizer(
        tokenizer,
        model_name,
        opt_bos_mode=opt_bos_mode,
    )
    return tokenizer


class ZOTaskDataCollator:
    """Collate task rows into the token-id batch consumed by ``VLLMZOModel``."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        config: ZOTaskEncodingConfig,
        subspace_rows: Sequence[Any] | None = None,
        step_getter: Callable[[], int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.subspace_rows = list(subspace_rows or [])
        self.step_getter = step_getter

    def __call__(self, rows: Sequence[Any]) -> TokenProbeBatch:
        row_list = list(rows)
        batch = build_zo_task_batch(
            row_list,
            self.tokenizer,
            objective_name=self.config.objective_name,
            max_length=self.config.max_length,
            max_new_tokens=self.config.max_new_tokens,
        )
        if self.config.direction_provider not in {"agzo", "uagzo", "suagzo"}:
            return batch
        if not self.subspace_rows:
            raise ValueError("AGZO task collation requires non-empty subspace_rows")
        batch = with_subspace_token_batch_factory(
            batch,
            self._build_subspace_token_id_group_batches,
            subspace_num_rows=int(self.config.agzo_kappa),
        )
        return batch

    def _build_subspace_token_id_group_batches(self):
        step = 1 if self.step_getter is None else int(self.step_getter())
        start = (max(1, step) - 1) * int(self.config.agzo_kappa)
        subspace_rows = cyclic_rows(
            self.subspace_rows,
            start=start,
            count=int(self.config.agzo_kappa),
        )
        return [
            build_zo_task_batch(
                chunk_rows,
                self.tokenizer,
                objective_name=self.config.objective_name,
                max_length=self.config.max_length,
                max_new_tokens=self.config.max_new_tokens,
            ).token_id_groups
            for chunk_rows in split_rows_for_agzo_subspace(
                subspace_rows,
                chunk_size=int(self.config.agzo_subspace_chunk_size),
            )
        ]


__all__ = [
    "ZOTaskDataCollator",
    "ZOTaskEncodingConfig",
    "build_tokenizer",
]
