"""Event-loop-owned serving engine operations for background ZO training."""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any, Callable, Sequence

import torch

from zo_vllm.core.lora_runtime import AsyncLoRASlotRegistry
from zo_vllm.serving.compact_nll_scorer import (
    QueuedCompactNLLSubmitter,
    ScheduledCompactNLLScorer,
)
from zo_vllm.serving.schemas import (
    SUPPORTED_SCORE_ADMISSION_POLICIES,
    ServingZOStartRequest,
)
from zo_vllm.serving.scheduled_zo_executor import ScheduledWorkerZOExecutor
from zo_vllm.serving.worker_client import AsyncWorkerUpdateBankClient


class AsyncZOEngineService:
    """Own all asyncio and scheduler-specific work on the server event loop."""

    def __init__(
        self,
        *,
        executor: ScheduledWorkerZOExecutor,
        pair_lock: asyncio.Lock,
    ) -> None:
        self._executor = executor
        self._pair_lock = pair_lock

    @classmethod
    async def create(
        cls,
        *,
        engine_client: Any,
        state: Any,
        model_name: str,
        request: ServingZOStartRequest,
        model_config: Any,
        metadata: Any,
        direction_dtype: torch.dtype,
        resolved_target_modules: Sequence[str],
        stop_requested: Callable[[], bool],
    ) -> "AsyncZOEngineService":
        pair_lock = asyncio.Lock()
        admission = _ScoreAdmissionController(
            state=state,
            request=request,
            stop_requested=stop_requested,
        )
        slot_registry = AsyncLoRASlotRegistry.from_model_config(
            engine_client=engine_client,
            model_config=model_config,
            rank=request.update_bank_rank,
            base_model_name=model_name,
            plus_id=request.plus_id,
            minus_id=request.minus_id,
            target_modules=resolved_target_modules,
        )
        worker_bank = AsyncWorkerUpdateBankClient(
            engine_client=engine_client,
            slot_registry=slot_registry,
            metadata=metadata,
            request=request,
            direction_dtype=direction_dtype,
        )
        await worker_bank.initialize()
        submitter = None
        if validate_score_admission_policy(request.score_admission_policy) == "queued":
            submitter = QueuedCompactNLLSubmitter(
                token_rate=float(request.zo_queue_token_rate),
                burst_tokens=int(request.zo_queue_burst_tokens),
                max_inflight=int(request.zo_queue_max_inflight),
                max_admitted_tokens=int(request.zo_queue_max_admitted_tokens),
                poll_s=float(request.zo_queue_poll_s),
            )
        scorer = ScheduledCompactNLLScorer(
            engine_client=engine_client,
            priority=request.priority,
            submitter=submitter,
        )
        return cls(
            executor=ScheduledWorkerZOExecutor(
                worker_bank=worker_bank,
                scorer=scorer,
                pair_lock=pair_lock,
                wait_for_admission=admission.wait,
                stop_requested=stop_requested,
                foreground_load=lambda: int(getattr(state, "server_load_metrics", 0)),
                admission_extra=admission.extra,
            ),
            pair_lock=pair_lock,
        )

    @property
    def pair_inflight(self) -> bool:
        return self._pair_lock.locked()

    async def score_probe_plan(self, plan: Any, batch: Any, *, step: int):
        return await self._executor.score_probe_plan(plan, batch, step=step)

    async def apply_update(self, **kwargs: Any):
        return await self._executor.apply_update(**kwargs)

    async def score_clean(self, batch: Any, *, step: int):
        return await self._executor.score_clean(batch, step=step)

    async def close(self) -> None:
        await self._executor.close()


class _ScoreAdmissionController:
    def __init__(
        self,
        *,
        state: Any,
        request: ServingZOStartRequest,
        stop_requested: Callable[[], bool],
    ) -> None:
        self.state = state
        self.request = request
        self.stop_requested = stop_requested
        self._extra: dict[str, Any] = {}

    def extra(self) -> dict[str, Any]:
        return dict(self._extra)

    async def wait(self) -> tuple[float, int, int]:
        policy = validate_score_admission_policy(self.request.score_admission_policy)
        load = int(getattr(self.state, "server_load_metrics", 0))
        self._extra = {
            "score_admission_gpu_utilization": None,
            "score_admission_gpu_utilization_threshold": None,
            "score_admission_gpu_device": None,
        }
        if policy in {"scheduler_only", "queued"}:
            return 0.0, 1, load
        if policy == "gpu_utilization":
            return await self._wait_for_gpu_utilization()
        if policy != "idle_gap":
            raise RuntimeError(f"unknown score admission policy: {policy!r}")

        start = asyncio.get_running_loop().time()
        samples = 0
        while True:
            load = int(getattr(self.state, "server_load_metrics", 0))
            samples += 1
            elapsed = asyncio.get_running_loop().time() - start
            if self.stop_requested() or load <= int(
                self.request.max_score_admission_foreground_load
            ):
                return elapsed, samples, load
            timeout_s = float(self.request.score_admission_timeout_s)
            if timeout_s > 0 and elapsed >= timeout_s:
                raise RuntimeError(
                    "serving-time ZO idle_gap score admission timed out: "
                    f"load={load} "
                    f"max_load={self.request.max_score_admission_foreground_load} "
                    f"timeout_s={timeout_s}"
                )
            await asyncio.sleep(float(self.request.score_admission_poll_s))

    async def _wait_for_gpu_utilization(self) -> tuple[float, int, int]:
        start = asyncio.get_running_loop().time()
        samples = 0
        threshold = float(self.request.max_score_admission_gpu_utilization)
        device = resolve_gpu_utilization_device(self.request.score_admission_gpu_device)
        while True:
            load = int(getattr(self.state, "server_load_metrics", 0))
            util = await asyncio.to_thread(query_gpu_utilization_percent, device)
            samples += 1
            self._extra = {
                "score_admission_gpu_utilization": float(util),
                "score_admission_gpu_utilization_threshold": threshold,
                "score_admission_gpu_device": device,
            }
            elapsed = asyncio.get_running_loop().time() - start
            if self.stop_requested() or util <= threshold:
                return elapsed, samples, load
            timeout_s = float(self.request.score_admission_timeout_s)
            if timeout_s > 0 and elapsed >= timeout_s:
                raise RuntimeError(
                    "serving-time ZO gpu_utilization score admission timed out: "
                    f"gpu_utilization={util} threshold={threshold} "
                    f"device={device!r} timeout_s={timeout_s}"
                )
            await asyncio.sleep(float(self.request.score_admission_poll_s))


def validate_score_admission_policy(value: str) -> str:
    if value not in SUPPORTED_SCORE_ADMISSION_POLICIES:
        raise RuntimeError(
            "serving-time ZO score_admission_policy must be "
            f"one of {sorted(SUPPORTED_SCORE_ADMISSION_POLICIES)}; got {value!r}"
        )
    return value


def resolve_gpu_utilization_device(request_device: str | None) -> str | None:
    if request_device is not None and str(request_device).strip():
        return str(request_device).strip()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return None
    first_device = visible_devices.split(",", 1)[0].strip()
    return first_device or None


def query_gpu_utilization_percent(device: str | None) -> float:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    if device is not None:
        cmd.append(f"--id={device}")
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return float(stripped)
    raise RuntimeError("nvidia-smi returned no GPU utilization samples")


__all__ = ["AsyncZOEngineService", "validate_score_admission_policy"]
