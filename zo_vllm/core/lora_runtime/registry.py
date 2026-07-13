"""Async LoRA slot registration helpers."""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

from zo_vllm.core.lora_runtime.runtime import (
    LoRAUpdateRuntime,
    _load_empty_lora_slots_from_runtime_config,
)


class AsyncLoRASlotRegistry:
    """Async wrapper for registering persistent direct LoRA slots."""

    def __init__(
        self,
        *,
        engine_client: Any,
        runtime: LoRAUpdateRuntime,
        plus_name: str = "serving_zo_plus",
        minus_name: str = "serving_zo_minus",
    ) -> None:
        self.engine_client = engine_client
        self.runtime = runtime
        self.plus_name = plus_name
        self.minus_name = minus_name
        self.last_update_info: dict[str, Any] = {}

    @classmethod
    def from_model_config(
        cls,
        *,
        engine_client: Any,
        model_config: Any,
        rank: int,
        base_model_name: str,
        plus_id: int,
        minus_id: int,
        target_modules: Sequence[str] | None = None,
        plus_name: str = "serving_zo_plus",
        minus_name: str = "serving_zo_minus",
    ) -> "AsyncLoRASlotRegistry":
        runtime = LoRAUpdateRuntime.from_model_config(
            model_config,
            rank=int(rank),
            base_model_name=base_model_name,
            llm=None,
            plus_id=int(plus_id),
            minus_id=int(minus_id),
            target_modules=target_modules,
        )
        return cls(
            engine_client=engine_client,
            runtime=runtime,
            plus_name=plus_name,
            minus_name=minus_name,
        )

    @property
    def plus_id(self) -> int:
        return int(self.runtime.plus_id)

    @property
    def minus_id(self) -> int:
        return int(self.runtime.minus_id)

    async def register_slots(self) -> None:
        if self.runtime._registered:
            return
        self.runtime._set_ids_for_step(step=0)
        results = await self.engine_client.collective_rpc(
            "apply_model",
            args=(
                partial(
                    _load_empty_lora_slots_from_runtime_config,
                    runtime_config=self.runtime.worker_registration_config(),
                ),
            ),
        )
        if not results:
            raise RuntimeError(
                "vLLM apply_model returned no results while loading LoRA"
            )
        self.last_update_info = {"workers": results}
        self.runtime._registered = True

    def lora_request(self, *, sign: str):
        from vllm.lora.request import LoRARequest

        if sign == "plus":
            return LoRARequest(
                self.plus_name,
                self.runtime.plus_id,
                self.runtime.plus_path,
            )
        if sign == "minus":
            return LoRARequest(
                self.minus_name,
                self.runtime.minus_id,
                self.runtime.minus_path,
            )
        raise ValueError(f"unknown LoRA sign: {sign}")
