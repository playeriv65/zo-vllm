"""Scheduled compact prompt-NLL scorer for serving-time ZO."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any
import uuid

from zo_vllm.engine import TokenScoreResult


@dataclass
class _QueuedCompactNLLItem:
    call: Any
    estimated_tokens: int
    future: asyncio.Future


class QueuedCompactNLLSubmitter:
    """FIFO admission queue for low-priority compact-NLL scheduler requests."""

    def __init__(
        self,
        *,
        token_rate: float,
        burst_tokens: int,
        max_inflight: int,
        max_admitted_tokens: int,
        poll_s: float,
    ) -> None:
        self.token_rate = float(token_rate)
        self.burst_tokens = int(burst_tokens)
        self.max_inflight = int(max_inflight)
        self.max_admitted_tokens = int(max_admitted_tokens)
        self.poll_s = float(poll_s)
        if self.token_rate <= 0:
            raise ValueError("token_rate must be positive")
        if self.burst_tokens <= 0:
            raise ValueError("burst_tokens must be positive")
        if self.max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if self.max_admitted_tokens <= 0:
            raise ValueError("max_admitted_tokens must be positive")
        if self.poll_s < 0:
            raise ValueError("poll_s must be non-negative")

        self._queue: asyncio.Queue[_QueuedCompactNLLItem | None] = asyncio.Queue()
        self._inflight = asyncio.Semaphore(self.max_inflight)
        self._admitted_cond = asyncio.Condition()
        self._admitted_tokens = 0
        self._tokens = float(self.burst_tokens)
        self._last_refill = time.perf_counter()
        self._worker_task: asyncio.Task | None = None
        self._closed = False

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._worker_loop(),
                name="serving-zo-submit-queue",
            )

    async def close(self) -> None:
        self._closed = True
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task

    def stats(self) -> dict[str, Any]:
        return {
            "queue_policy": "queued",
            "queue_size": self._queue.qsize(),
            "token_rate": self.token_rate,
            "burst_tokens": self.burst_tokens,
            "max_inflight": self.max_inflight,
            "max_admitted_tokens": self.max_admitted_tokens,
            "admitted_tokens": self._admitted_tokens,
            "poll_s": self.poll_s,
        }

    async def submit(self, call: Any, *, estimated_tokens: int) -> Any:
        if self._closed:
            raise RuntimeError("queued compact-NLL submitter is closed")
        self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(
            _QueuedCompactNLLItem(
                call=call,
                estimated_tokens=max(1, int(estimated_tokens)),
                future=future,
            )
        )
        return await future

    async def _worker_loop(self) -> None:
        running_tasks: set[asyncio.Task] = set()
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                if item.future.cancelled():
                    continue
                if item.estimated_tokens > self.max_admitted_tokens:
                    if not item.future.done():
                        item.future.set_exception(
                            RuntimeError(
                                "queued compact-NLL request estimated_tokens "
                                f"({item.estimated_tokens}) exceeds "
                                "max_admitted_tokens "
                                f"({self.max_admitted_tokens})"
                            )
                        )
                    continue
                await self._reserve_tokens(item.estimated_tokens)
                await self._inflight.acquire()
                await self._acquire_admitted_tokens(item.estimated_tokens)
                task = asyncio.create_task(
                    self._run_item(item, admitted_tokens=item.estimated_tokens)
                )
                running_tasks.add(task)
                task.add_done_callback(running_tasks.discard)
        finally:
            if running_tasks:
                await asyncio.gather(*running_tasks, return_exceptions=True)

    async def _run_item(
        self,
        item: _QueuedCompactNLLItem,
        *,
        admitted_tokens: int,
    ) -> None:
        try:
            result = await item.call()
        except Exception as exc:
            if not item.future.done():
                item.future.set_exception(exc)
        else:
            if not item.future.done():
                item.future.set_result(result)
        finally:
            await self._release_admitted_tokens(admitted_tokens)
            self._inflight.release()

    async def _acquire_admitted_tokens(self, estimated_tokens: int) -> None:
        async with self._admitted_cond:
            while self._admitted_tokens + estimated_tokens > self.max_admitted_tokens:
                await self._admitted_cond.wait()
            self._admitted_tokens += estimated_tokens

    async def _release_admitted_tokens(self, estimated_tokens: int) -> None:
        async with self._admitted_cond:
            self._admitted_tokens = max(0, self._admitted_tokens - estimated_tokens)
            self._admitted_cond.notify_all()

    async def _reserve_tokens(self, estimated_tokens: int) -> None:
        cost = min(max(1, int(estimated_tokens)), self.burst_tokens)
        while True:
            now = time.perf_counter()
            elapsed = max(0.0, now - self._last_refill)
            self._tokens = min(
                float(self.burst_tokens),
                self._tokens + elapsed * self.token_rate,
            )
            self._last_refill = now
            if self._tokens >= cost:
                self._tokens -= cost
                return
            missing = cost - self._tokens
            wait_s = missing / self.token_rate
            await asyncio.sleep(max(self.poll_s, wait_s))


class ScheduledCompactNLLScorer:
    """Score token-id prompts through the vLLM scheduler."""

    def __init__(
        self,
        *,
        engine_client: Any,
        priority: int,
        submitter: QueuedCompactNLLSubmitter | None = None,
    ) -> None:
        self.engine_client = engine_client
        self.priority = int(priority)
        self.submitter = submitter

    async def score(
        self,
        token_id_groups: Sequence[Sequence[int]],
        *,
        labels: Sequence[Sequence[int]],
        lora_request: Any | None,
        tag: str,
    ) -> TokenScoreResult:
        if len(token_id_groups) != len(labels):
            raise ValueError("token_id_groups and labels must have same length")
        tasks = [
            asyncio.create_task(
                self._score_one(
                    list(token_ids),
                    labels=list(map(int, label_row)),
                    lora_request=lora_request,
                    request_id=f"serving-zo-{tag}-{uuid.uuid4().hex}",
                )
            )
            for token_ids, label_row in zip(token_id_groups, labels)
        ]
        per_request = await asyncio.gather(*tasks)
        weighted_request_nll = [float(item[0]) for item in per_request]
        request_num_tokens = [int(item[1]) for item in per_request]
        request_nll = [
            float(weighted_nll) / float(num_tokens)
            for weighted_nll, num_tokens in zip(
                weighted_request_nll,
                request_num_tokens,
            )
        ]
        nll_sum = float(sum(weighted_request_nll))
        num_tokens = int(sum(request_num_tokens))
        loss = nll_sum / max(1, num_tokens)
        return TokenScoreResult(
            loss=loss,
            nll_sum=nll_sum,
            num_tokens=num_tokens,
            request_nll=request_nll,
            request_num_tokens=request_num_tokens,
            detail={
                "backend": "scheduled_compact_nll",
                "priority": self.priority,
                "num_requests": len(request_nll),
                "submitter": (
                    None if self.submitter is None else self.submitter.stats()
                ),
            },
            raw={},
        )

    async def _score_one(
        self,
        token_ids: list[int],
        *,
        labels: list[int],
        lora_request: Any | None,
        request_id: str,
    ) -> tuple[float, int]:
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=0,
            prompt_logprobs=0,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            skip_clone=True,
            extra_args={
                "zo_direct_prompt_nll": True,
                "zo_loss_labels": labels,
            },
        )

        async def submit() -> Any:
            final_output = None
            async for output in self.engine_client.generate(
                {"prompt_token_ids": token_ids},
                sampling_params,
                request_id,
                lora_request=lora_request,
                priority=self.priority,
            ):
                final_output = output
            return final_output

        if self.submitter is None:
            final_output = await submit()
        else:
            final_output = await self.submitter.submit(
                submit,
                estimated_tokens=len(token_ids),
            )
        if final_output is None:
            raise RuntimeError("scheduled compact NLL request returned no output")
        prompt_logprobs = final_output.prompt_logprobs
        if not (
            isinstance(prompt_logprobs, dict)
            and prompt_logprobs.get("__zo_prompt_nll__")
        ):
            raise RuntimeError(
                "scheduled compact NLL request did not return compact prompt NLL"
            )
        return float(prompt_logprobs["nll_sum"]), int(prompt_logprobs["num_tokens"])
