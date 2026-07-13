"""Serving lifecycle API for HF-native ZO training on vLLM servers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import time
from typing import Any

import torch
from fastapi import APIRouter, HTTPException, Request
from transformers import SchedulerType

from zo_trainer import (
    StopOnSignalCallback,
    ZOTrainer,
    ZOTrainerArguments,
    ZOTrainerModel,
)
from zo_vllm.utils.io import append_jsonl, write_json
from zo_vllm.tasks.hf_preprocessing import (
    load_sst2_prompt_classification_datasets,
)
from zo_vllm.tasks.tokenization import configure_opt_tokenizer
from zo_vllm.serving.async_engine_service import (
    AsyncZOEngineService,
    validate_score_admission_policy,
)
from zo_vllm.serving.schemas import (
    ServingZOStartRequest,
    ServingZOStopRequest,
)
from zo_vllm.core.lora_scope import (
    resolve_lora_target_modules,
    resolve_update_bank_rank,
)
from zo_vllm.training.model_metadata import (
    build_lora_param_metadata_from_config,
    resolve_direction_dtype,
    torch_device,
)
from zo_vllm.serving.hf_callbacks import (
    InterStepDelayCallback,
    ServingObservationCallback,
)
from zo_vllm.serving.scheduled_zo_executor import AsyncZOStepCancelled
from zo_vllm.serving.scheduled_runtime import ScheduledServingRuntime
from zo_vllm.serving.thread_bridge import BlockingAsyncBridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BackgroundTrainingHandle:
    """Passive state for one synchronous HF Trainer running in a thread."""

    stop_event: threading.Event
    started_at: float
    run_dir: Path
    jsonl_path: Path
    bridge: BlockingAsyncBridge
    thread: threading.Thread | None = None
    trainer: ZOTrainer | None = None
    engine_service: AsyncZOEngineService | None = None
    last_metrics: dict[str, Any] | None = None
    last_error: str | None = None
    step: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


async def _start_background_training(
    state: Any,
    request: ServingZOStartRequest,
) -> dict[str, Any]:
    active = _job_from_state(state)
    if active is not None and active.thread is not None and active.thread.is_alive():
        raise RuntimeError("serving-time ZO training is already running")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
        raise RuntimeError(
            "VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required for internal "
            "LoRA slot update RPC callables"
        )

    engine_client, server_args = _serving_dependencies(state)
    request = _resolve_auto_update_bank_rank(request)
    _validate_runtime_config(state, server_args, request)
    tokenizer = engine_client.get_tokenizer()
    configure_opt_tokenizer(tokenizer, str(getattr(server_args, "model", "")))
    model_config = _hf_config(state)
    run_dir = _make_run_dir(server_args, request)
    handle = BackgroundTrainingHandle(
        stop_event=threading.Event(),
        started_at=time.time(),
        run_dir=run_dir,
        jsonl_path=run_dir / "zo_metrics.jsonl",
        bridge=BlockingAsyncBridge(
            asyncio.get_running_loop(),
            owner_thread_id=threading.get_ident(),
        ),
    )
    state.zo_vllm_background_training = handle
    write_json(
        run_dir / "start_config.json",
        {
            "request": request.model_dump(),
            "model": getattr(server_args, "model", None),
            "scheduler_policy": getattr(
                state.vllm_config.scheduler_config, "policy", None
            ),
            "server_args": _server_args_status(server_args),
        },
    )
    handle.thread = threading.Thread(
        target=_run_background_training,
        kwargs={
            "handle": handle,
            "engine_client": engine_client,
            "state": state,
            "server_args": server_args,
            "request": request,
            "tokenizer": tokenizer,
            "model_config": model_config,
        },
        name="serving-zo-hf-trainer",
        daemon=True,
    )
    handle.thread.start()
    return _background_training_status(state)


async def _stop_background_training(
    state: Any,
    request: ServingZOStopRequest,
) -> dict[str, Any]:
    handle = _job_from_state(state)
    if handle is None:
        return _background_training_status(state)
    handle.stop_event.set()
    thread = handle.thread
    if request.wait and thread is not None:
        await asyncio.to_thread(thread.join, float(request.timeout_s))
        if thread.is_alive():
            raise RuntimeError("serving-time ZO training did not stop before timeout")
    return _background_training_status(state)


def _background_training_status(state: Any) -> dict[str, Any]:
    handle = _job_from_state(state)
    _, server_args = _serving_dependencies(state)
    if handle is None:
        return {
            "enabled": True,
            "running": False,
            "step": 0,
            "started_at": None,
            "run_dir": None,
            "jsonl_path": None,
            "inflight_zo_pair": False,
            "stop_requested": False,
            "last_metrics": None,
            "last_error": None,
            "server_load_metrics": int(getattr(state, "server_load_metrics", 0)),
            "enable_server_load_tracking": bool(
                getattr(state, "enable_server_load_tracking", False)
            ),
            "scheduler_policy": _scheduler_policy_name(state),
            "server_args": _server_args_status(server_args),
        }
    thread = handle.thread
    with handle.lock:
        engine_service = handle.engine_service
        step = int(handle.step)
        last_metrics = handle.last_metrics
        last_error = handle.last_error
    return {
        "enabled": True,
        "running": thread is not None and thread.is_alive(),
        "step": step,
        "started_at": handle.started_at,
        "run_dir": str(handle.run_dir),
        "jsonl_path": str(handle.jsonl_path),
        "inflight_zo_pair": bool(engine_service and engine_service.pair_inflight),
        "stop_requested": handle.stop_event.is_set(),
        "last_metrics": last_metrics,
        "last_error": last_error,
        "server_load_metrics": int(getattr(state, "server_load_metrics", 0)),
        "enable_server_load_tracking": bool(
            getattr(state, "enable_server_load_tracking", False)
        ),
        "scheduler_policy": _scheduler_policy_name(state),
        "server_args": _server_args_status(server_args),
    }


def _run_background_training(
    *,
    handle: BackgroundTrainingHandle,
    engine_client: Any,
    state: Any,
    server_args: Any,
    request: ServingZOStartRequest,
    tokenizer: Any,
    model_config: Any,
) -> None:
    try:
        _run_hf_trainer(
            handle=handle,
            engine_client=engine_client,
            state=state,
            server_args=server_args,
            request=request,
            tokenizer=tokenizer,
            model_config=model_config,
        )
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        with handle.lock:
            handle.last_error = error
        _append_job_jsonl(handle, {"event": "error", "error": error})


def _run_hf_trainer(
    *,
    handle: BackgroundTrainingHandle,
    engine_client: Any,
    state: Any,
    server_args: Any,
    request: ServingZOStartRequest,
    tokenizer: Any,
    model_config: Any,
) -> None:
    train_dataset, dev_dataset = load_sst2_prompt_classification_datasets(
        tokenizer,
        data_seed=request.seed if request.data_seed is None else request.data_seed,
        num_train=int(request.num_train),
        num_dev=int(request.num_dev),
    )
    direction_dtype = resolve_direction_dtype(request.direction_dtype, model_config)
    resolved_target_modules = resolve_lora_target_modules(
        request.target_modules,
        include_lm_head=request.include_lm_head,
        include_embeddings=request.include_embeddings,
    )
    metadata = build_lora_param_metadata_from_config(
        model_config,
        device=torch_device(request.direction_device),
        dtype=direction_dtype,
        target_modules=resolved_target_modules,
    )
    engine_service = handle.bridge.call(
        AsyncZOEngineService.create(
            engine_client=engine_client,
            state=state,
            model_name=str(getattr(server_args, "model", "")),
            request=request,
            model_config=model_config,
            metadata=metadata,
            direction_dtype=direction_dtype,
            resolved_target_modules=resolved_target_modules,
            stop_requested=handle.stop_event.is_set,
        )
    )
    with handle.lock:
        handle.engine_service = engine_service
    runtime = ScheduledServingRuntime(
        service=engine_service,
        bridge=handle.bridge,
        eps=float(request.eps),
    )
    callbacks = [
        StopOnSignalCallback(stop_requested=handle.stop_event.is_set),
        ServingObservationCallback(
            runtime=runtime,
            record=lambda metrics, step: _record_job(handle, metrics, step=step),
            batch_size=int(request.batch_size),
            priority=int(request.priority),
            score_admission_policy=request.score_admission_policy,
        ),
    ]
    if float(request.inter_step_delay_s) > 0:
        callbacks.append(
            InterStepDelayCallback(
                stop_event=handle.stop_event,
                delay_s=float(request.inter_step_delay_s),
            )
        )
    training_args = ZOTrainerArguments(
        output_dir=str(handle.run_dir / "hf"),
        max_steps=int(request.steps) if int(request.steps) > 0 else 2**31 - 1,
        per_device_train_batch_size=int(request.batch_size),
        per_device_eval_batch_size=int(request.batch_size),
        learning_rate=float(request.learning_rate),
        weight_decay=float(request.weight_decay),
        lr_scheduler_type=request.lr_scheduler_type,
        warmup_steps=int(request.warmup_steps),
        seed=int(request.seed),
        data_seed=(
            int(request.seed) if request.data_seed is None else int(request.data_seed)
        ),
        eval_strategy="steps" if int(request.eval_interval) > 0 else "no",
        eval_steps=max(1, int(request.eval_interval)),
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"] if request.enable_wandb else [],
        run_name=handle.run_dir.name,
        use_cpu=True,
        remove_unused_columns=False,
        max_grad_norm=0.0,
        optim="sgd",
        disable_tqdm=True,
        zo_checkpoint_mode="metadata",
    )
    if request.enable_wandb:
        os.environ["WANDB_PROJECT"] = str(request.wandb_project)
        os.environ["WANDB_ENTITY"] = str(request.wandb_entity)
    trainer = ZOTrainer(
        model=ZOTrainerModel(runtime),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        compute_metrics=_classification_metrics,
        callbacks=callbacks,
    )
    with handle.lock:
        handle.trainer = trainer
    try:
        if request.initial_eval and not handle.stop_event.is_set():
            trainer.evaluate()
        if not handle.stop_event.is_set():
            train_output = trainer.train()
            _record_job(
                handle,
                {"event": "train", **dict(train_output.metrics)},
                step=int(trainer.state.global_step),
            )
        with handle.lock:
            handle.step = int(trainer.state.global_step)
    except AsyncZOStepCancelled:
        with handle.lock:
            handle.step = int(trainer.state.global_step)
    finally:
        try:
            handle.bridge.call(engine_service.close())
        finally:
            with handle.lock:
                handle.engine_service = None

    with handle.lock:
        stopped_step = int(handle.step)
    _append_job_jsonl(handle, {"event": "stopped", "step": stopped_step})


def _validate_runtime_config(
    state: Any,
    server_args: Any,
    request: ServingZOStartRequest,
) -> None:
    policy = _scheduler_policy_name(state)
    if policy != "priority":
        raise RuntimeError(
            "serving-time ZO training requires vLLM --scheduling-policy priority; "
            f"got {policy!r}"
        )
    if not bool(getattr(server_args, "enable_lora", False)):
        raise RuntimeError("serving-time ZO training requires --enable-lora")
    max_loras = getattr(server_args, "max_loras", None)
    if max_loras is not None and int(max_loras) < 2:
        raise RuntimeError(
            f"serving-time ZO training requires --max-loras >= 2; got {int(max_loras)}"
        )
    resolved_update_bank_rank = _resolve_auto_update_bank_rank(request).update_bank_rank
    max_lora_rank = getattr(server_args, "max_lora_rank", None)
    if max_lora_rank is not None and int(max_lora_rank) < int(
        resolved_update_bank_rank
    ):
        raise RuntimeError(
            "serving-time ZO training requires --max-lora-rank >= "
            f"update_bank_rank; got {int(max_lora_rank)} < "
            f"{int(resolved_update_bank_rank)}"
        )
    validate_score_admission_policy(request.score_admission_policy)
    if float(request.weight_decay) != 0.0:
        raise ValueError("serving worker-bank training does not support weight_decay")
    try:
        scheduler_type = SchedulerType(request.lr_scheduler_type)
    except ValueError as exc:
        raise ValueError(
            f"unsupported serving scheduler: {request.lr_scheduler_type!r}"
        ) from exc
    if scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
        raise ValueError(
            "serving training does not support metric-driven plateau scheduler"
        )
    if int(request.steps) == 0 and scheduler_type != SchedulerType.CONSTANT:
        raise ValueError(
            "open-ended serving training requires lr_scheduler_type='constant'"
        )


def _resolve_auto_update_bank_rank(
    request: ServingZOStartRequest,
) -> ServingZOStartRequest:
    resolved, auto = resolve_update_bank_rank(
        request.update_bank_rank,
        rank=int(request.rank),
        steps=int(request.steps),
        nu=int(request.nu),
    )
    if resolved < int(request.rank):
        raise ValueError("update_bank_rank must be >= rank")
    if not auto and int(resolved) == int(request.update_bank_rank):
        return request
    return request.model_copy(update={"update_bank_rank": int(resolved)})


def _serving_dependencies(state: Any) -> tuple[Any, Any]:
    engine_client = getattr(state, "zo_vllm_serving_engine_client", None)
    server_args = getattr(state, "zo_vllm_serving_server_args", None)
    if engine_client is None or server_args is None:
        raise HTTPException(
            status_code=503,
            detail="serving-time ZO API dependencies are not initialized",
        )
    return engine_client, server_args


def _job_from_state(state: Any) -> BackgroundTrainingHandle | None:
    job = getattr(state, "zo_vllm_background_training", None)
    if job is not None and not isinstance(job, BackgroundTrainingHandle):
        raise TypeError("zo_vllm_background_training has an invalid value")
    return job


def _hf_config(state: Any) -> Any:
    model_config = getattr(state.vllm_config, "model_config", None)
    return getattr(model_config, "hf_config", model_config)


def _make_run_dir(server_args: Any, request: ServingZOStartRequest) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_slug = str(getattr(server_args, "model", "model")).replace("/", "__")
    name = request.run_name or (
        f"serving_sst2_r{request.rank}_bank{request.update_bank_rank}_{ts}"
    )
    output_root = Path(request.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / f"{model_slug}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _server_args_status(server_args: Any) -> dict[str, Any]:
    return {
        "enable_lora": getattr(server_args, "enable_lora", None),
        "max_lora_rank": getattr(server_args, "max_lora_rank", None),
        "max_loras": getattr(server_args, "max_loras", None),
    }


def _record_job(
    handle: BackgroundTrainingHandle,
    metrics: dict[str, Any],
    *,
    step: int,
) -> None:
    with handle.lock:
        handle.step = int(step)
        handle.last_metrics = metrics
    _append_job_jsonl(handle, metrics)


def _append_job_jsonl(
    handle: BackgroundTrainingHandle,
    payload: dict[str, Any],
) -> None:
    row = {
        "time": time.time(),
        "perf_time": time.perf_counter(),
        **_jsonable(payload),
    }
    append_jsonl(handle.jsonl_path, row)


def _classification_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    predicted_labels = predictions.argmax(axis=-1)
    return {"accuracy": float((predicted_labels == labels).mean())}


def attach_router(app: Any) -> None:
    router = APIRouter(prefix="/zo_vllm/serving_zo", tags=["zo-vllm-serving-zo"])

    @router.post("/start")
    async def start_serving_zo(payload: ServingZOStartRequest, raw_request: Request):
        payload.target_modules = resolve_lora_target_modules(
            payload.target_modules,
            include_lm_head=payload.include_lm_head,
            include_embeddings=payload.include_embeddings,
        )
        try:
            return await _start_background_training(raw_request.app.state, payload)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/stop")
    async def stop_serving_zo(payload: ServingZOStopRequest, raw_request: Request):
        try:
            return await _stop_background_training(raw_request.app.state, payload)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/status")
    async def status_serving_zo(raw_request: Request):
        return _background_training_status(raw_request.app.state)

    app.include_router(router)

    async def shutdown_serving_zo() -> None:
        if getattr(app.state, "zo_vllm_background_training", None) is not None:
            await _stop_background_training(
                app.state,
                ServingZOStopRequest(wait=True, timeout_s=60.0),
            )

    app.add_event_handler("shutdown", shutdown_serving_zo)


async def init_serving_zo_state(
    engine_client: Any,
    state: Any,
    args: Any,
    request_logger: Any | None = None,
) -> None:
    del request_logger
    state.zo_vllm_serving_engine_client = engine_client
    state.zo_vllm_serving_server_args = args
    state.zo_vllm_background_training = None


def _scheduler_policy_name(state: Any) -> str | None:
    scheduler_config = getattr(
        getattr(state, "vllm_config", None),
        "scheduler_config",
        None,
    )
    policy = getattr(scheduler_config, "policy", None)
    if policy is None:
        return None
    normalized = str(getattr(policy, "value", policy)).lower()
    if normalized.endswith(".priority"):
        return "priority"
    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return str(value)
