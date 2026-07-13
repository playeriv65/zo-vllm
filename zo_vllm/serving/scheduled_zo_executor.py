"""Scheduled vLLM executor for async antithetic ZO probe plans."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any

from zo_vllm.core.label_masks import suffix_lm_labels
from zo_vllm.training.direction import TokenProbeBatch
from zo_vllm.training.estimator import AntitheticProbePlan

from .compact_nll_scorer import ScheduledCompactNLLScorer
from .worker_client import AsyncWorkerUpdateBankClient


class AsyncZOStepCancelled(RuntimeError):
    """Raised when serving shutdown interrupts a step before probe submission."""


@dataclass(frozen=True)
class CleanScoreExecution:
    """Clean scheduled score returned to the synchronous trainer thread."""

    score: Any
    lora_update_s: float
    score_s: float
    clean_slot_info: Any


@dataclass(frozen=True)
class ScheduledProbeScores:
    """Raw scheduled scores for one estimator-owned probe plan."""

    scores: tuple[Any, ...]
    direction_refreshed: bool
    direction_info: dict[str, Any]
    observations: dict[str, Any]


class ScheduledWorkerZOExecutor:
    """Keep serving-only admission and RPC details behind one async executor."""

    def __init__(
        self,
        *,
        worker_bank: AsyncWorkerUpdateBankClient,
        scorer: ScheduledCompactNLLScorer,
        pair_lock: asyncio.Lock,
        wait_for_admission: Callable[[], Awaitable[tuple[float, int, int]]],
        stop_requested: Callable[[], bool],
        foreground_load: Callable[[], int],
        admission_extra: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.worker_bank = worker_bank
        self.scorer = scorer
        self.pair_lock = pair_lock
        self.wait_for_admission = wait_for_admission
        self.stop_requested = stop_requested
        self.foreground_load = foreground_load
        self.admission_extra = admission_extra

    async def score_probe_plan(
        self,
        plan: AntitheticProbePlan,
        batch: TokenProbeBatch,
        *,
        step: int,
    ) -> ScheduledProbeScores:
        """Execute probes without owning the task loss calculation."""

        labels = _score_labels(batch)
        async with self.pair_lock:
            self._raise_if_stopped()
            foreground_load_at_slot_write = int(self.foreground_load())
            slot_write_start_perf = time.perf_counter()
            prepare_info = await self.worker_bank.prepare_slots(
                step=int(step),
                eps=float(plan.eps),
            )
            slot_write_end_perf = time.perf_counter()
            (
                score_admission_wait_s,
                score_admission_samples,
                foreground_load_at_score_submit,
            ) = await self.wait_for_admission()
            self._raise_if_stopped()
            score_start_perf = time.perf_counter()
            plus_task = asyncio.create_task(
                self.scorer.score(
                    batch.token_id_groups,
                    labels=labels,
                    lora_request=self.worker_bank.lora_request(sign="plus"),
                    tag=f"s{int(step)}-{plan.probe_names[0]}",
                )
            )
            minus_task = asyncio.create_task(
                self.scorer.score(
                    batch.token_id_groups,
                    labels=labels,
                    lora_request=self.worker_bank.lora_request(sign="minus"),
                    tag=f"s{int(step)}-{plan.probe_names[1]}",
                )
            )
            scores = tuple(await asyncio.gather(plus_task, minus_task))
            score_end_perf = time.perf_counter()

        score_s = score_end_perf - score_start_perf
        observations = {
            "refresh_fold_s": float(prepare_info.get("refresh_fold_s", 0.0)),
            "direction_s": float(prepare_info.get("direction_s", 0.0)),
            "slot_info": dict(prepare_info.get("slot_info", {})),
            "slot_write_start_perf": slot_write_start_perf,
            "slot_write_end_perf": slot_write_end_perf,
            "score_start_perf": score_start_perf,
            "score_end_perf": score_end_perf,
            "lora_update_s": slot_write_end_perf - slot_write_start_perf,
            "score_s": score_s,
            "score_admission_wait_s": float(score_admission_wait_s),
            "score_admission_samples": int(score_admission_samples),
            "foreground_load_at_score_submit": int(foreground_load_at_score_submit),
            "foreground_load_at_slot_write": foreground_load_at_slot_write,
            "plus_num_tokens": int(scores[0].num_tokens),
            "minus_num_tokens": int(scores[1].num_tokens),
        }
        if self.admission_extra is not None:
            observations.update(dict(self.admission_extra()))
        submitter = getattr(self.scorer, "submitter", None)
        if submitter is not None:
            observations["zo_queue"] = submitter.stats()
        return ScheduledProbeScores(
            scores=scores,
            direction_refreshed=bool(prepare_info.get("direction_refreshed", False)),
            direction_info=dict(prepare_info.get("direction_info", {})),
            observations=observations,
        )

    async def apply_update(
        self,
        *,
        step: int,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[dict[str, Any], float]:
        return await self.worker_bank.apply_update(
            step=int(step),
            projected_grad=float(projected_grad),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    async def score_clean(
        self, batch: TokenProbeBatch, *, step: int
    ) -> CleanScoreExecution:
        """Write the effective clean slot and score one evaluation batch."""

        t0 = time.perf_counter()
        clean_info = await self.worker_bank.write_clean(step=int(step))
        lora_update_s = time.perf_counter() - t0
        lora_request = (
            self.worker_bank.lora_request(sign="plus")
            if clean_info.get("has_update")
            else None
        )
        score_t0 = time.perf_counter()
        score = await self.scorer.score(
            batch.token_id_groups,
            labels=_score_labels(batch),
            lora_request=lora_request,
            tag=f"eval-s{int(step)}",
        )
        return CleanScoreExecution(
            score=score,
            lora_update_s=lora_update_s,
            score_s=time.perf_counter() - score_t0,
            clean_slot_info=clean_info.get("slot_info"),
        )

    async def close(self) -> None:
        """Close event-loop-owned serving resources."""

        submitter = getattr(self.scorer, "submitter", None)
        if submitter is not None:
            await submitter.close()

    def _raise_if_stopped(self) -> None:
        if self.stop_requested():
            raise AsyncZOStepCancelled("serving ZO step cancelled before scoring")


def _score_labels(batch: TokenProbeBatch) -> list[list[int]]:
    if batch.labels is not None:
        labels = [[int(value) for value in row] for row in batch.labels]
        if len(labels) != len(batch.token_id_groups):
            raise ValueError("probe labels must match token_id_groups")
        return labels
    if batch.loss_token_lens is None:
        raise ValueError("scheduled NLL scoring requires labels or loss_token_lens")
    return suffix_lm_labels(batch.token_id_groups, batch.loss_token_lens)


__all__ = [
    "AsyncZOStepCancelled",
    "CleanScoreExecution",
    "ScheduledProbeScores",
    "ScheduledWorkerZOExecutor",
]
