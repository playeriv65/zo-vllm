from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from datasets import Dataset
from fastapi import FastAPI

from zo_trainer import ZOTrainer, ZOTrainerArguments, ZOTrainerModel
from zo_vllm.engine import TokenScoreResult
from zo_vllm.serving.hf_callbacks import ServingObservationCallback
from zo_vllm.serving.schemas import ServingZOStartRequest, ServingZOStopRequest
from zo_vllm.serving.scheduled_runtime import ScheduledServingRuntime
from zo_vllm.serving.scheduled_zo_executor import (
    CleanScoreExecution,
    ScheduledProbeScores,
)
from zo_vllm.serving.thread_bridge import BlockingAsyncBridge
from zo_vllm.serving.zo_training_api import attach_router
import zo_vllm.serving.zo_training_api as zo_training_api_module
from zo_vllm.serving.async_engine_service import AsyncZOEngineService


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        assert self.ready.wait(timeout=2.0)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()
        self.loop.close()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2.0)
        assert not self.thread.is_alive()


def test_blocking_async_bridge_returns_values_and_propagates_errors() -> None:
    owner = _LoopThread()
    bridge = BlockingAsyncBridge(
        owner.loop,
        owner_thread_id=owner.thread.ident,
    )

    async def value() -> int:
        await asyncio.sleep(0)
        return 7

    async def fail() -> None:
        raise ValueError("probe failed")

    try:
        assert bridge.call(value()) == 7
        with pytest.raises(ValueError, match="probe failed"):
            bridge.call(fail())
        assert bridge.active_calls == 0
    finally:
        owner.close()


def test_blocking_async_bridge_rejects_event_loop_thread() -> None:
    owner = _LoopThread()
    bridge = BlockingAsyncBridge(
        owner.loop,
        owner_thread_id=owner.thread.ident,
    )

    async def invoke_on_owner() -> str:
        with pytest.raises(RuntimeError, match="event-loop thread"):
            bridge.call(asyncio.sleep(0))
        return "guarded"

    try:
        future = asyncio.run_coroutine_threadsafe(invoke_on_owner(), owner.loop)
        assert future.result(timeout=2.0) == "guarded"
    finally:
        owner.close()


def test_serving_api_starts_one_hf_trainer_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = object()
    state = SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(policy="priority"),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        ),
        server_load_metrics=0,
        enable_server_load_tracking=True,
    )
    args = SimpleNamespace(
        model="fake/model",
        enable_lora=True,
        max_loras=2,
        max_lora_rank=16,
    )
    state.zo_vllm_serving_engine_client = SimpleNamespace(
        get_tokenizer=lambda: tokenizer
    )
    state.zo_vllm_serving_server_args = args
    state.zo_vllm_background_training = None
    run_thread_ids: list[int] = []

    def fake_run(**kwargs) -> None:
        assert kwargs["tokenizer"] is tokenizer
        run_thread_ids.append(threading.get_ident())
        kwargs["handle"].stop_event.wait(timeout=2.0)

    monkeypatch.setattr(
        "zo_vllm.serving.zo_training_api.configure_opt_tokenizer",
        lambda tokenizer, model: None,
    )
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setattr(zo_training_api_module, "_run_background_training", fake_run)

    async def exercise() -> None:
        status = await zo_training_api_module._start_background_training(
            state,
            ServingZOStartRequest(
                steps=1,
                update_bank_rank=16,
                output_dir=str(tmp_path),
            ),
        )
        assert status["running"] is True
        handle = state.zo_vllm_background_training
        assert handle.thread is not None
        assert handle.thread.name == "serving-zo-hf-trainer"
        stopped = await zo_training_api_module._stop_background_training(
            state, ServingZOStopRequest(wait=True, timeout_s=2.0)
        )
        assert stopped["running"] is False
        assert stopped["stop_requested"] is True

    asyncio.run(exercise())

    assert len(run_thread_ids) == 1
    assert run_thread_ids[0] != threading.get_ident()


def test_serving_state_contains_dependencies_and_passive_job_only() -> None:
    engine_client = object()
    server_args = object()
    state = SimpleNamespace()

    asyncio.run(
        zo_training_api_module.init_serving_zo_state(
            engine_client,
            state,
            server_args,
        )
    )

    assert state.zo_vllm_serving_engine_client is engine_client
    assert state.zo_vllm_serving_server_args is server_args
    assert state.zo_vllm_background_training is None
    assert not hasattr(state, "zo_vllm_serving_zo_controller")


def test_only_engine_service_operations_are_async() -> None:
    assert not inspect.iscoroutinefunction(
        zo_training_api_module._run_background_training
    )
    assert not inspect.iscoroutinefunction(zo_training_api_module._run_hf_trainer)
    assert inspect.iscoroutinefunction(AsyncZOEngineService.score_probe_plan)
    assert inspect.iscoroutinefunction(AsyncZOEngineService.apply_update)
    assert inspect.iscoroutinefunction(AsyncZOEngineService.score_clean)


def test_router_shutdown_waits_for_background_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    attach_router(app)
    requests: list[ServingZOStopRequest] = []

    async def fake_stop(state, request: ServingZOStopRequest) -> None:
        del state
        requests.append(request)

    monkeypatch.setattr(
        zo_training_api_module,
        "_stop_background_training",
        fake_stop,
    )
    app.state.zo_vllm_background_training = object()
    assert app.router.on_shutdown

    asyncio.run(app.router.on_shutdown[-1]())

    assert len(requests) == 1
    assert requests[0].wait is True
    assert requests[0].timeout_s == 60.0


def _score(request_nll: list[float]) -> TokenScoreResult:
    return TokenScoreResult(
        loss=sum(request_nll) / len(request_nll),
        nll_sum=sum(request_nll),
        num_tokens=len(request_nll),
        request_nll=request_nll,
        request_num_tokens=[1] * len(request_nll),
        detail={"backend": "fake_scheduled"},
        raw={},
    )


class _FakeScheduledService:
    def __init__(self) -> None:
        self.score_thread_ids: list[int] = []
        self.apply_thread_ids: list[int] = []
        self.apply_calls: list[dict[str, float | int]] = []

    async def score_probe_plan(self, plan, batch, *, step):
        self.score_thread_ids.append(threading.get_ident())
        assert plan.probe_names == ("plus", "minus")
        assert len(batch.token_id_groups) == 4
        return ScheduledProbeScores(
            scores=(
                _score([1.0, 0.0, 0.0, 1.0]),
                _score([0.0, 1.0, 1.0, 0.0]),
            ),
            direction_refreshed=True,
            direction_info={"direction_provider": "worker"},
            observations={"score_s": 0.02, "slot_info": {"source": "worker"}},
        )

    async def apply_update(self, **kwargs):
        self.apply_thread_ids.append(threading.get_ident())
        self.apply_calls.append(kwargs)
        return {"source": "worker"}, 0.01

    async def score_clean(self, batch, *, step):
        self.score_thread_ids.append(threading.get_ident())
        return CleanScoreExecution(
            score=_score([1.0, 0.0, 0.0, 1.0]),
            lora_update_s=0.003,
            score_s=0.02,
            clean_slot_info={"source": "worker"},
        )


def test_hf_trainer_owns_serving_loop_loss_optimizer_and_scheduler(tmp_path) -> None:
    owner = _LoopThread()
    bridge = BlockingAsyncBridge(
        owner.loop,
        owner_thread_id=owner.thread.ident,
    )
    service = _FakeScheduledService()
    runtime = ScheduledServingRuntime(
        service=service,  # type: ignore[arg-type]
        bridge=bridge,
        eps=0.1,
    )
    records: list[dict] = []
    callback = ServingObservationCallback(
        runtime=runtime,
        record=lambda metrics, step: records.append({"step": step, **metrics}),
        batch_size=2,
        priority=1000,
        score_admission_policy="scheduler_only",
    )
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [[1, 2], [1, 3]],
                "option_loss_token_counts": [1, 1],
                "row_option_counts": 2,
                "labels": 1,
            },
            {
                "input_ids": [[4, 5], [4, 6]],
                "option_loss_token_counts": [1, 1],
                "row_option_counts": 2,
                "labels": 0,
            },
        ]
    )
    args = ZOTrainerArguments(
        output_dir=str(tmp_path),
        max_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=0.3,
        lr_scheduler_type="linear",
        warmup_steps=0,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
        max_grad_norm=0.0,
        optim="sgd",
    )
    trainer = ZOTrainer(
        model=ZOTrainerModel(runtime),
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        callbacks=[callback],
    )

    try:
        output = trainer.train()
        assert output.global_step == 1
        assert trainer.lr_scheduler is not None
        assert service.apply_calls[0]["learning_rate"] == pytest.approx(0.3)
        assert service.score_thread_ids == [owner.thread.ident]
        assert service.apply_thread_ids == [owner.thread.ident]
        assert records[0]["event"] == "train_step"
        assert records[0]["loss_plus"] != records[0]["loss_minus"]

        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert service.score_thread_ids[-1] == owner.thread.ident
        with pytest.raises(RuntimeError, match="option NLL only"):
            runtime.engine.forward_token_logits([])
    finally:
        owner.close()
