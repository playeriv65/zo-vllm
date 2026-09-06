from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import torch

from zo_vllm.core.lora_runtime import AsyncLoRASlotRegistry
from zo_vllm.serving.schemas import ServingZOStartRequest
from zo_vllm.training.model_metadata import metadata_specs, torch_dtype_name
from zo_vllm.training.worker_update_bank import (
    SOURCE_NAME as WORKER_UPDATE_BANK_SOURCE,
    apply_worker_update_bank_update,
    init_worker_update_bank,
    prepare_worker_update_bank_slots,
    worker_update_bank_key,
    write_clean_worker_update_bank,
)


class AsyncWorkerUpdateBankClient:
    """Async client for the worker-resident ZO update bank backend."""

    def __init__(
        self,
        *,
        engine_client: Any,
        slot_registry: AsyncLoRASlotRegistry,
        metadata: dict[str, Any],
        request: ServingZOStartRequest,
        direction_dtype: torch.dtype,
    ) -> None:
        self.engine_client = engine_client
        self.slot_registry = slot_registry
        self.state_key = worker_update_bank_key(
            plus_id=request.plus_id,
            minus_id=request.minus_id,
        )
        self.slot_write_stream = request.slot_write_stream
        self.metadata_specs = metadata_specs(metadata)
        self.lozo_config = {
            "rank": int(request.rank),
            "eps": float(request.eps),
            "nu": int(request.nu),
            "seed": int(request.seed),
            "random_device": request.random_device,
            "direction_dtype": torch_dtype_name(direction_dtype),
            "direction_sampling": request.direction_sampling,
            "direction_scale": request.direction_scale,
            "perturbation_normalization": request.perturbation_normalization,
            "v_normalization": request.v_normalization,
        }
        self.bank_config = {
            "update_bank_rank": int(request.update_bank_rank),
            "u_beta": float(request.u_beta),
            "u_momentum": float(getattr(request, "u_momentum", 0.0)),
            "u_norm_cap": request.u_norm_cap,
            "gradient_accumulation_update_steps": int(
                request.gradient_accumulation_update_steps
            ),
        }
        self._initialized = False
        self.last_update_info: dict[str, Any] = {}

    async def initialize(self) -> dict[str, Any]:
        if self._initialized:
            return self.last_update_info
        await self.slot_registry.register_slots()
        results = await self.engine_client.collective_rpc(
            "apply_model",
            args=(
                partial(
                    init_worker_update_bank,
                    state_key=self.state_key,
                    metadata_specs=self.metadata_specs,
                    lozo_config=self.lozo_config,
                    bank_config=self.bank_config,
                    plus_id=self.slot_registry.plus_id,
                    minus_id=self.slot_registry.minus_id,
                ),
            ),
        )
        if not results:
            raise RuntimeError("vLLM apply_model returned no results initializing bank")
        self.last_update_info = {"workers": results}
        self._initialized = True
        return dict(self.last_update_info)

    async def prepare_slots(self, *, step: int, eps: float) -> dict[str, Any]:
        await self.initialize()
        results = await self.engine_client.collective_rpc(
            "apply_model",
            args=(
                partial(
                    prepare_worker_update_bank_slots,
                    state_key=self.state_key,
                    step=int(step),
                    eps=float(eps),
                    copy_stream=self.slot_write_stream,
                ),
            ),
        )
        if not results:
            raise RuntimeError("vLLM apply_model returned no results preparing bank")
        merged = merge_worker_bank_results(results)
        self.last_update_info = {"workers": results, **merged}
        return dict(self.last_update_info)

    async def apply_update(
        self,
        *,
        step: int,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[dict[str, Any], float]:
        await self.initialize()
        results = await self.engine_client.collective_rpc(
            "apply_model",
            args=(
                partial(
                    apply_worker_update_bank_update,
                    state_key=self.state_key,
                    step=int(step),
                    projected_grad=float(projected_grad),
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                ),
            ),
        )
        if not results:
            raise RuntimeError("vLLM apply_model returned no results applying bank")
        first = dict(results[0])
        update_info = dict(first.get("update_info", {}))
        update_info["source"] = first.get("source", WORKER_UPDATE_BANK_SOURCE)
        return update_info, float(first.get("apply_s", 0.0))

    async def write_clean(self, *, step: int) -> dict[str, Any]:
        await self.initialize()
        results = await self.engine_client.collective_rpc(
            "apply_model",
            args=(
                partial(
                    write_clean_worker_update_bank,
                    state_key=self.state_key,
                    step=int(step),
                    copy_stream=self.slot_write_stream,
                ),
            ),
        )
        if not results:
            raise RuntimeError(
                "vLLM apply_model returned no results writing clean bank"
            )
        return {"workers": results, **merge_worker_bank_results(results)}

    def lora_request(self, *, sign: str):
        return self.slot_registry.lora_request(sign=sign)


def merge_worker_bank_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    first = dict(results[0])
    slot_info = first.get("slot_info")
    if slot_info is None:
        slot_info = {"workers": results}
    else:
        slot_info = {"workers": [slot_info]}
    return {
        "source": first.get("source", WORKER_UPDATE_BANK_SOURCE),
        "direction_refreshed": bool(first.get("direction_refreshed", False)),
        "direction_info": dict(first.get("direction_info", {})),
        "refresh_fold_s": float(first.get("refresh_fold_s", 0.0)),
        "direction_s": float(first.get("direction_s", 0.0)),
        "bank_prepare_s": float(first.get("bank_prepare_s", 0.0)),
        "worker_slot_write_s": float(first.get("worker_slot_write_s", 0.0)),
        "slot_info": slot_info,
        "has_update": bool(first.get("has_update", True)),
    }
