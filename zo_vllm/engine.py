"""Public engine API for zeroth-order scoring with vLLM."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
import threading
import time
from typing import Any

import torch

from zo_vllm.config import ZOVLLMEngineConfig
from zo_vllm.core.direct_worker_scorer import (
    collect_agzo_directions,
    collect_agzo_directions_chunked,
    collect_activation_stats,
    direct_worker_detail,
    forward_token_id_logits,
    forward_token_id_request_nll,
    fused_agzo_step,
    score_token_id_groups,
)
from zo_vllm.core.lora_runtime import LoRAUpdateRuntime
from zo_vllm.core.probe_results import ProbeTiming
from zo_vllm.core.token_scores import (
    score_clean_token_groups,
    slice_score_result,
)
from zo_vllm.core.token_groups import (
    borrow_or_copy_int_list,
    borrow_or_copy_token_group_batches,
    borrow_or_copy_token_groups,
)

DirectionMap = Mapping[str, Mapping[str, torch.Tensor]]
ObjectiveFn = Callable[["TokenScoreResult"], float]


@dataclass(frozen=True)
class TokenScoreResult:
    """Per-request token scoring result returned by the direct vLLM worker path."""

    loss: float
    nll_sum: float
    num_tokens: int
    request_nll: list[float]
    request_num_tokens: list[int]
    detail: dict[str, float | int | str | bool]
    raw: dict[str, Any]


@dataclass(frozen=True)
class RuntimeProbeSlots:
    """Opaque handle for runtime-owned LoRA probe slots."""

    _lora_ids: tuple[int, ...]


@dataclass(frozen=True)
class TokenLogitsResult:
    """Compact logits for HF causal-LM loss positions."""

    logits: torch.Tensor
    target_token_ids: torch.Tensor
    request_loss_token_counts: tuple[int, ...]
    timing: ProbeTiming


@dataclass(frozen=True)
class TokenRequestNLLResult:
    """Per-request mean NLL tensor used as HF classification model output."""

    request_nll: torch.Tensor
    timing: ProbeTiming


@dataclass(frozen=True)
class PlusMinusScoreResult:
    """Plus/minus objective values for one zeroth-order direction."""

    loss_plus: float
    loss_minus: float
    projected_grad: float
    plus: Any
    minus: Any
    update_info: dict[str, Any]


@dataclass(frozen=True)
class FusedAGZOStepResult:
    """Result from a worker-side fused binary-option AGZO step."""

    loss_plus: float
    loss_minus: float
    projected_grad: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class ActivationStatsResult:
    """Worker-side Linear input activation stats from one direct vLLM forward."""

    layers: list[dict[str, Any]]
    num_linear_inputs_seen: int
    max_layers: int
    detail: dict[str, float | int | str | bool]
    raw: dict[str, Any]


class ZOVLLMEngine:
    """
    Reusable vLLM-backed engine for zeroth-order experiments.

    The engine owns vLLM initialization, persistent plus/minus LoRA slots, direct
    worker scoring, and direction-slot updates. Callers provide token IDs,
    optional loss-token lengths, and optional objective functions; task-specific
    aggregation stays outside this class.
    """

    def __init__(
        self,
        *,
        model: str,
        rank: int,
        config: ZOVLLMEngineConfig | None = None,
        llm: Any | None = None,
        model_config: Any | None = None,
    ) -> None:
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        resolved_config = config or ZOVLLMEngineConfig()
        self.config = resolved_config
        self.model = model
        self.rank = int(rank)
        self.lora_rank = int(
            rank if resolved_config.lora_rank is None else resolved_config.lora_rank
        )
        self.plus_id = int(resolved_config.slot.plus_id)
        self.minus_id = int(resolved_config.slot.minus_id)
        self.target_modules = list(resolved_config.target_modules)
        self._slots_registered = False
        self._owns_llm = llm is None
        self._closed = False
        self._call_lock = threading.RLock()

        if model_config is None:
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(model)
        self.model_config = model_config

        if llm is None:
            from vllm import LLM

            kwargs: dict[str, Any] = {
                "model": model,
                "dtype": resolved_config.dtype,
                "gpu_memory_utilization": float(resolved_config.gpu_memory_utilization),
                "enforce_eager": bool(resolved_config.enforce_eager),
                "enable_lora": True,
                "max_lora_rank": self.lora_rank,
                "max_loras": int(resolved_config.slot.max_loras),
                "lora_target_modules": self.target_modules,
            }
            if resolved_config.max_model_len is not None:
                kwargs["max_model_len"] = int(resolved_config.max_model_len)
            if resolved_config.llm_kwargs:
                kwargs.update(dict(resolved_config.llm_kwargs))
            previous_reserved = os.environ.get("VLLM_ZO_RESERVED_GPU_BYTES")
            if resolved_config.zo_reserved_gpu_bytes:
                os.environ["VLLM_ZO_RESERVED_GPU_BYTES"] = str(
                    int(resolved_config.zo_reserved_gpu_bytes)
                )
            try:
                llm = LLM(**kwargs)
            finally:
                if resolved_config.zo_reserved_gpu_bytes:
                    if previous_reserved is None:
                        os.environ.pop("VLLM_ZO_RESERVED_GPU_BYTES", None)
                    else:
                        os.environ["VLLM_ZO_RESERVED_GPU_BYTES"] = previous_reserved
        self.llm = llm

        self.runtime = LoRAUpdateRuntime.from_model_config(
            model_config,
            rank=self.lora_rank,
            base_model_name=model,
            llm=self.llm,
            plus_id=self.plus_id,
            minus_id=self.minus_id,
            target_modules=self.target_modules,
        )

    def __enter__(self) -> "ZOVLLMEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def register_direction_slots(self) -> None:
        """Create persistent plus/minus LoRA slots once."""
        self._ensure_open()
        if not self._slots_registered:
            self.runtime.register_slots()
            self._slots_registered = True

    def set_plus_minus_directions(
        self,
        directions: DirectionMap,
        *,
        eps: float,
        step: int = 0,
    ) -> dict[str, Any]:
        """Write plus/minus low-rank directions into persistent GPU LoRA slots."""
        self._validate_eps(eps)
        with self._call_lock:
            self.register_direction_slots()
            self.runtime.update_plus_minus_from_directions(
                dict(directions),
                eps=float(eps),
                step=int(step),
            )
            return dict(self.runtime.last_update_info)

    def set_slot_direction(
        self,
        lora_id: int,
        directions: DirectionMap,
        *,
        eps: float,
        sign: float = 1.0,
    ) -> dict[str, Any]:
        """Write one low-rank direction into one persistent GPU LoRA slot."""
        self._validate_eps(eps)
        with self._call_lock:
            self.register_direction_slots()
            return dict(
                self.runtime.update_slot_from_direction(
                    lora_id=int(lora_id),
                    directions_2d=dict(directions),
                    eps=float(eps),
                    sign=float(sign),
                )
            )

    @property
    def probe_slot_capacity(self) -> int:
        """Return the number of reusable runtime probe slots."""

        return 2

    def set_probe_directions(
        self,
        directions: Sequence[DirectionMap],
        *,
        eps: float,
        sign: float = 1.0,
    ) -> tuple[RuntimeProbeSlots, list[dict[str, Any]]]:
        """Write directions into runtime-owned slots and return an opaque handle."""

        if not directions or len(directions) > self.probe_slot_capacity:
            raise ValueError("probe direction count exceeds runtime slot capacity")
        slot_ids = (self.plus_id, self.minus_id)[: len(directions)]
        update_infos = [
            self.set_slot_direction(
                slot_id,
                direction,
                eps=float(eps),
                sign=float(sign),
            )
            for slot_id, direction in zip(slot_ids, directions)
        ]
        return RuntimeProbeSlots(tuple(slot_ids)), update_infos

    def score_probe_slots(
        self,
        slots: RuntimeProbeSlots,
        token_id_groups: Sequence[Sequence[int]],
        *,
        loss_token_lens: Sequence[int] | None = None,
        labels: Sequence[Sequence[int]] | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
    ):
        """Score one complete token batch through every opaque probe slot."""

        return self.score_probe_slots_with_scorer(
            self,
            slots,
            token_id_groups,
            loss_token_lens=loss_token_lens,
            labels=labels,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
        )

    def score_probe_slots_with_scorer(
        self,
        scorer: Any,
        slots: RuntimeProbeSlots,
        token_id_groups: Sequence[Sequence[int]],
        *,
        loss_token_lens: Sequence[int] | None = None,
        labels: Sequence[Sequence[int]] | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
    ):
        """Score an opaque slot batch through a caller-provided objective scorer."""

        token_groups = borrow_or_copy_token_groups(token_id_groups) or []
        loss_lens = borrow_or_copy_int_list(loss_token_lens)
        label_rows = borrow_or_copy_token_groups(labels)
        slot_count = len(slots._lora_ids)
        return scorer.score_token_groups(
            token_groups * slot_count,
            loss_token_lens=None if loss_lens is None else loss_lens * slot_count,
            labels=None if label_rows is None else label_rows * slot_count,
            lora_ids=[
                slot_id for slot_id in slots._lora_ids for _ in range(len(token_groups))
            ],
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
        )

    def generate_probe_slots(
        self,
        slots: RuntimeProbeSlots,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> list[list[Any]]:
        """Generate through runtime slots without exposing vLLM request details."""

        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        from zo_vllm.core.lora_runtime import DIRECT_SLOT_PATH_PREFIX

        prompt_rows = list(prompts)
        sampling_params = SamplingParams(
            n=1,
            seed=None if seed is None else int(seed),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        flat_prompts = prompt_rows * len(slots._lora_ids)
        requests = [
            LoRARequest(
                f"zo_es_query_{slot_id}",
                slot_id,
                f"{DIRECT_SLOT_PATH_PREFIX}/{slot_id}",
            )
            for slot_id in slots._lora_ids
            for _ in prompt_rows
        ]
        outputs = self.llm.generate(
            flat_prompts,
            sampling_params=sampling_params,
            lora_request=requests,
            use_tqdm=False,
        )
        width = len(prompt_rows)
        return [
            list(outputs[start : start + width])
            for start in range(0, len(outputs), width)
        ]

    def score_token_groups(
        self,
        token_id_groups: Sequence[Sequence[int]],
        *,
        loss_token_lens: Sequence[int] | None = None,
        labels: Sequence[Sequence[int]] | None = None,
        lora_ids: Sequence[int] | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
    ) -> TokenScoreResult:
        """Score arbitrary token-id prompts with optional LoRA ids."""
        self._ensure_open()
        token_groups = borrow_or_copy_token_groups(token_id_groups)
        loss_lens = borrow_or_copy_int_list(loss_token_lens)
        label_rows = borrow_or_copy_token_groups(labels)
        lora_id_list = borrow_or_copy_int_list(lora_ids)
        self._validate_score_inputs(token_groups, loss_lens, label_rows, lora_id_list)
        with self._call_lock:
            result = score_token_id_groups(
                self.llm,
                token_groups,
                lora_ids=lora_id_list,
                loss_token_lens=loss_lens,
                labels=label_rows,
                max_logits_tokens=int(max_logits_tokens),
                loss_impl=loss_impl,
            )
        return self._token_score_from_raw(result, loss_impl=loss_impl)

    def forward_token_logits(
        self,
        token_id_groups: Sequence[Sequence[int]],
        *,
        labels: Sequence[Sequence[int]] | None = None,
        lora_ids: Sequence[int] | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
    ) -> TokenLogitsResult:
        """Return compact logits for the active causal-LM label positions."""

        self._ensure_open()
        token_groups = borrow_or_copy_token_groups(token_id_groups)
        label_rows = borrow_or_copy_token_groups(labels)
        lora_id_list = borrow_or_copy_int_list(lora_ids)
        self._validate_score_inputs(token_groups, None, label_rows, lora_id_list)
        with self._call_lock:
            return_device_tensors = self._can_return_device_tensors()
            raw = forward_token_id_logits(
                self.llm,
                token_groups,
                lora_ids=lora_id_list,
                labels=label_rows,
                max_logits_tokens=int(max_logits_tokens),
                loss_impl=loss_impl,
                return_device_tensors=return_device_tensors,
            )
        self._wait_for_device_tensors(raw)
        logits = raw.get("logits")
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, dtype=torch.float32)
        target_token_ids = raw.get("target_token_ids")
        if not isinstance(target_token_ids, torch.Tensor):
            target_token_ids = torch.as_tensor(target_token_ids, dtype=torch.long)
        return TokenLogitsResult(
            logits=logits.float(),
            target_token_ids=target_token_ids.long(),
            request_loss_token_counts=tuple(
                int(value) for value in raw["request_loss_token_counts"]
            ),
            timing=self._probe_timing_from_raw(raw),
        )

    def forward_token_request_nll(
        self,
        token_id_groups: Sequence[Sequence[int]],
        *,
        loss_token_lens: Sequence[int],
        lora_ids: Sequence[int] | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
    ) -> TokenRequestNLLResult:
        """Return request-level NLL tensors without a device-to-host round trip."""

        self._ensure_open()
        token_groups = borrow_or_copy_token_groups(token_id_groups)
        loss_lens = borrow_or_copy_int_list(loss_token_lens)
        lora_id_list = borrow_or_copy_int_list(lora_ids)
        self._validate_score_inputs(token_groups, loss_lens, None, lora_id_list)
        with self._call_lock:
            raw = forward_token_id_request_nll(
                self.llm,
                token_groups,
                lora_ids=lora_id_list,
                loss_token_lens=loss_lens,
                max_logits_tokens=int(max_logits_tokens),
                loss_impl=loss_impl,
                return_device_tensors=self._can_return_device_tensors(),
            )
        self._wait_for_device_tensors(raw)
        request_nll = raw["request_nll_tensor"]
        if not isinstance(request_nll, torch.Tensor):
            request_nll = torch.as_tensor(request_nll, dtype=torch.float32)
        return TokenRequestNLLResult(
            request_nll=request_nll.float(),
            timing=self._probe_timing_from_raw(raw),
        )

    def _can_return_device_tensors(self) -> bool:
        """CUDA tensors can cross the scorer boundary only in one process."""

        executor = getattr(
            getattr(self.llm, "llm_engine", None), "model_executor", None
        )
        return bool(getattr(executor, "supports_device_tensor_rpc", False))

    @staticmethod
    def _wait_for_device_tensors(raw: Mapping[str, Any]) -> None:
        ready_event = raw.get("device_tensors_ready_event")
        if ready_event is None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("received a CUDA readiness event without CUDA")
        torch.cuda.current_stream().wait_event(ready_event)

    @staticmethod
    def _probe_timing_from_raw(raw: Mapping[str, Any]) -> ProbeTiming:
        profile_s = raw.get("profile_s")
        profile_cuda_ms = raw.get("profile_cuda_ms")
        cuda_events = raw.get("profile_cuda_events")
        return ProbeTiming.from_backend(
            profile_s if isinstance(profile_s, Mapping) else None,
            profile_cuda_ms if isinstance(profile_cuda_ms, Mapping) else None,
            cuda_events if isinstance(cuda_events, Mapping) else None,
        )

    def collect_activation_stats(
        self,
        token_id_groups: Sequence[Sequence[int]],
        *,
        loss_token_lens: Sequence[int] | None = None,
        labels: Sequence[Sequence[int]] | None = None,
        lora_ids: Sequence[int] | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
        activation_max_layers: int = 16,
        activation_force_eager: bool = True,
    ) -> ActivationStatsResult:
        """Run one direct worker forward and collect Linear input activation stats."""
        self._ensure_open()
        token_groups = borrow_or_copy_token_groups(token_id_groups)
        loss_lens = borrow_or_copy_int_list(loss_token_lens)
        label_rows = borrow_or_copy_token_groups(labels)
        lora_id_list = borrow_or_copy_int_list(lora_ids)
        self._validate_score_inputs(token_groups, loss_lens, label_rows, lora_id_list)
        with self._call_lock:
            raw = collect_activation_stats(
                self.llm,
                token_groups,
                lora_ids=lora_id_list,
                loss_token_lens=loss_lens,
                labels=label_rows,
                max_logits_tokens=int(max_logits_tokens),
                loss_impl=loss_impl,
                activation_max_layers=int(activation_max_layers),
                activation_force_eager=bool(activation_force_eager),
            )
        stats = raw.get("activation_stats", {})
        return ActivationStatsResult(
            layers=list(stats.get("layers", [])),
            num_linear_inputs_seen=int(stats.get("num_linear_inputs_seen", 0)),
            max_layers=int(stats.get("max_layers", activation_max_layers)),
            detail=direct_worker_detail(raw, 0.0, loss_impl=loss_impl),
            raw=raw,
        )

    def collect_agzo_directions(
        self,
        token_id_groups: Sequence[Sequence[int]],
        *,
        loss_token_lens: Sequence[int] | None = None,
        labels: Sequence[Sequence[int]] | None = None,
        lora_ids: Sequence[int] | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
        activation_force_eager: bool = True,
        agzo_rank: int | None = None,
        agzo_power_iter_steps: int = 5,
        agzo_low_rank_oversample: int = 4,
        agzo_basis_seed: int | None = None,
        agzo_perturb_seed: int | None = None,
        agzo_basis_method: str = "power_iter",
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
        """Run one worker forward and build AGZO directions in the vLLM worker."""
        self._ensure_open()
        token_groups = borrow_or_copy_token_groups(token_id_groups)
        loss_lens = borrow_or_copy_int_list(loss_token_lens)
        label_rows = borrow_or_copy_token_groups(labels)
        lora_id_list = borrow_or_copy_int_list(lora_ids)
        self._validate_score_inputs(token_groups, loss_lens, label_rows, lora_id_list)
        if int(self.rank if agzo_rank is None else agzo_rank) <= 0:
            raise ValueError("agzo_rank must be positive")
        if int(agzo_power_iter_steps) <= 0:
            raise ValueError("agzo_power_iter_steps must be positive")
        if agzo_basis_method not in {"power_iter", "svd", "low_rank_svd"}:
            raise ValueError(f"unsupported agzo_basis_method: {agzo_basis_method}")
        with self._call_lock:
            raw = collect_agzo_directions(
                self.llm,
                token_groups,
                lora_ids=lora_id_list,
                loss_token_lens=loss_lens,
                labels=label_rows,
                max_logits_tokens=int(max_logits_tokens),
                loss_impl=loss_impl,
                activation_force_eager=bool(activation_force_eager),
                agzo_rank=int(self.rank if agzo_rank is None else agzo_rank),
                agzo_power_iter_steps=int(agzo_power_iter_steps),
                agzo_low_rank_oversample=int(agzo_low_rank_oversample),
                agzo_basis_seed=agzo_basis_seed,
                agzo_perturb_seed=agzo_perturb_seed,
                agzo_basis_method=agzo_basis_method,
            )
        if "agzo_directions" not in raw:
            raise RuntimeError("vLLM worker did not return agzo_directions")
        return raw["agzo_directions"], raw

    def collect_agzo_directions_chunked(
        self,
        token_id_group_batches: Sequence[Sequence[Sequence[int]]],
        *,
        activation_force_eager: bool = True,
        agzo_rank: int | None = None,
        agzo_power_iter_steps: int = 5,
        agzo_low_rank_oversample: int = 4,
        agzo_basis_seed: int | None = None,
        agzo_perturb_seed: int | None = None,
        agzo_basis_method: str = "power_iter",
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
        """Collect AGZO activations one caller-provided chunk at a time."""
        self._ensure_open()
        batches = borrow_or_copy_token_group_batches(token_id_group_batches)
        if not batches:
            raise ValueError("token_id_group_batches must be non-empty")
        for batch in batches:
            self._validate_score_inputs(batch, None, None, None)
        if int(self.rank if agzo_rank is None else agzo_rank) <= 0:
            raise ValueError("agzo_rank must be positive")
        if int(agzo_power_iter_steps) <= 0:
            raise ValueError("agzo_power_iter_steps must be positive")
        if agzo_basis_method not in {"power_iter", "svd", "low_rank_svd"}:
            raise ValueError(f"unsupported agzo_basis_method: {agzo_basis_method}")
        with self._call_lock:
            raw = collect_agzo_directions_chunked(
                self.llm,
                batches,
                activation_force_eager=bool(activation_force_eager),
                agzo_rank=int(self.rank if agzo_rank is None else agzo_rank),
                agzo_power_iter_steps=int(agzo_power_iter_steps),
                agzo_low_rank_oversample=int(agzo_low_rank_oversample),
                agzo_basis_seed=agzo_basis_seed,
                agzo_perturb_seed=agzo_perturb_seed,
                agzo_basis_method=agzo_basis_method,
            )
        if "agzo_directions" not in raw:
            raise RuntimeError("vLLM worker did not return agzo_directions")
        return raw["agzo_directions"], raw

    def score_plus_minus_directions(
        self,
        directions: DirectionMap,
        token_id_groups: Sequence[Sequence[int]],
        *,
        eps: float,
        loss_token_lens: Sequence[int] | None = None,
        labels: Sequence[Sequence[int]] | None = None,
        objective: ObjectiveFn | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
        score_chunk_size: int = 0,
        step: int = 0,
    ) -> PlusMinusScoreResult:
        """
        Update plus/minus slots, score both sides, and return a projected gradient.

        If ``objective`` is omitted, the objective is mean token NLL. For
        classification or preference losses, pass a callable that consumes a
        ``TokenScoreResult`` and returns a scalar objective.
        """
        return self.score_plus_minus_directions_with_scorer(
            self,
            directions,
            token_id_groups,
            eps=eps,
            loss_token_lens=loss_token_lens,
            labels=labels,
            objective=objective,
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
            score_chunk_size=score_chunk_size,
            step=step,
        )

    def score_plus_minus_directions_with_scorer(
        self,
        scorer: Any,
        directions: DirectionMap,
        token_id_groups: Sequence[Sequence[int]],
        *,
        eps: float,
        loss_token_lens: Sequence[int] | None = None,
        labels: Sequence[Sequence[int]] | None = None,
        objective: ObjectiveFn | None = None,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
        score_chunk_size: int = 0,
        step: int = 0,
    ) -> PlusMinusScoreResult:
        """Update plus/minus slots and score them through a token scorer."""

        self._ensure_open()
        self._validate_eps(eps)
        token_groups = borrow_or_copy_token_groups(token_id_groups)
        loss_lens = borrow_or_copy_int_list(loss_token_lens)
        label_rows = borrow_or_copy_token_groups(labels)
        self._validate_score_inputs(token_groups, loss_lens, label_rows, None)
        with self._call_lock:
            total_t0 = time.perf_counter()
            update_info = self.set_plus_minus_directions(directions, eps=eps, step=step)
            update_s = time.perf_counter() - total_t0
            score_t0 = time.perf_counter()
            if int(score_chunk_size) > 0 and len(token_groups) > int(score_chunk_size):
                plus = self._score_lora_side(
                    scorer,
                    token_groups,
                    lora_id=self.plus_id,
                    loss_token_lens=loss_lens,
                    labels=label_rows,
                    max_logits_tokens=max_logits_tokens,
                    loss_impl=loss_impl,
                    chunk_size=int(score_chunk_size),
                )
                minus = self._score_lora_side(
                    scorer,
                    token_groups,
                    lora_id=self.minus_id,
                    loss_token_lens=loss_lens,
                    labels=label_rows,
                    max_logits_tokens=max_logits_tokens,
                    loss_impl=loss_impl,
                    chunk_size=int(score_chunk_size),
                )
                slice_s = 0.0
            else:
                score = scorer.score_token_groups(
                    token_groups + token_groups,
                    lora_ids=[self.plus_id] * len(token_groups)
                    + [self.minus_id] * len(token_groups),
                    loss_token_lens=None
                    if loss_lens is None
                    else loss_lens + loss_lens,
                    labels=None if label_rows is None else label_rows + label_rows,
                    max_logits_tokens=max_logits_tokens,
                    loss_impl=loss_impl,
                )
                worker_profile_s = {}
                worker_cuda_profile_s = {}
                raw_score = None
                score_timing = getattr(score, "timing", None)
                if isinstance(score_timing, ProbeTiming):
                    worker_profile_s = score_timing.profile_seconds()
                else:
                    raw_score = getattr(score, "raw", None)
                    if not isinstance(raw_score, dict):
                        raw_score = None
                if not worker_profile_s and isinstance(raw_score, dict):
                    raw_worker_profile = raw_score.get("profile_s", {})
                    if isinstance(raw_worker_profile, dict):
                        worker_profile_s = {}
                        for key, value in raw_worker_profile.items():
                            if not isinstance(value, (int, float)):
                                continue
                            output_key = (
                                str(key)
                                if str(key).startswith("scorer_")
                                else f"worker_{key}"
                            )
                            worker_profile_s[output_key] = float(value)
                    raw_cuda_profile = raw_score.get("profile_cuda_ms", {})
                    if isinstance(raw_cuda_profile, dict):
                        worker_cuda_profile_s = {
                            f"worker_cuda_{key.removesuffix('_ms')}_s": float(value)
                            / 1000.0
                            for key, value in raw_cuda_profile.items()
                            if isinstance(value, (int, float))
                        }
                slice_t0 = time.perf_counter()
                result_factory = self._token_score_from_raw if scorer is self else None
                plus = slice_score_result(
                    score,
                    0,
                    len(token_groups),
                    loss_impl=loss_impl,
                    result_factory=result_factory,
                )
                minus = slice_score_result(
                    score,
                    len(token_groups),
                    2 * len(token_groups),
                    loss_impl=loss_impl,
                    result_factory=result_factory,
                )
                slice_s = time.perf_counter() - slice_t0
            score_s = time.perf_counter() - score_t0
            objective_t0 = time.perf_counter()
            loss_plus = float(plus.loss if objective is None else objective(plus))
            loss_minus = float(minus.loss if objective is None else objective(minus))
            objective_s = time.perf_counter() - objective_t0
            total_s = time.perf_counter() - total_t0
            update_info = dict(update_info)
            profile_s = dict(update_info.get("profile_s", {}))
            profile_s.update(
                {
                    "set_plus_minus_directions": update_s,
                    "score_token_groups": score_s,
                    "score_chunk_size": int(score_chunk_size),
                    "slice_score": slice_s,
                    "objective": objective_s,
                    "score_plus_minus_total": total_s,
                    **worker_profile_s,
                    **worker_cuda_profile_s,
                }
            )
            update_info["profile_s"] = profile_s
        if not (math.isfinite(loss_plus) and math.isfinite(loss_minus)):
            raise ValueError(
                f"objective returned non-finite values: plus={loss_plus}, "
                f"minus={loss_minus}"
            )
        return PlusMinusScoreResult(
            loss_plus=loss_plus,
            loss_minus=loss_minus,
            projected_grad=(loss_plus - loss_minus) / (2.0 * float(eps)),
            plus=plus,
            minus=minus,
            update_info=update_info,
        )

    def _score_lora_side(
        self,
        scorer: Any,
        token_groups: list[list[int]],
        *,
        lora_id: int,
        loss_token_lens: list[int] | None,
        labels: list[list[int]] | None,
        max_logits_tokens: int,
        loss_impl: str,
        chunk_size: int,
    ) -> TokenScoreResult:
        return score_clean_token_groups(
            scorer,
            token_groups,
            loss_token_lens=loss_token_lens,
            labels=labels,
            lora_ids=[int(lora_id)] * len(token_groups),
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
            score_chunk_size=chunk_size,
            result_factory=self._token_score_from_raw if scorer is self else None,
        )

    def fused_agzo_step(
        self,
        token_id_groups: Sequence[Sequence[int]],
        *,
        token_labels: Sequence[Sequence[int]],
        labels: Sequence[int],
        eps: float,
        learning_rate: float,
        weight_decay: float = 0.0,
        max_logits_tokens: int = 8192,
        loss_impl: str = "logprobs",
        activation_force_eager: bool = True,
        agzo_rank: int | None = None,
        agzo_power_iter_steps: int = 5,
        agzo_low_rank_oversample: int = 4,
        agzo_basis_seed: int | None = None,
        agzo_perturb_seed: int | None = None,
        agzo_basis_method: str = "power_iter",
        direction_scale: float = 1.0,
    ) -> FusedAGZOStepResult:
        """Run one binary-option AGZO step inside the vLLM worker."""
        self._ensure_open()
        self._validate_eps(eps)
        token_groups = borrow_or_copy_token_groups(token_id_groups)
        label_rows = borrow_or_copy_token_groups(token_labels)
        label_list = borrow_or_copy_int_list(labels)
        self._validate_score_inputs(token_groups, None, label_rows, None)
        if len(token_groups) != 2 * len(label_list):
            raise ValueError("fused_agzo_step expects two token groups per label")
        if int(self.rank if agzo_rank is None else agzo_rank) <= 0:
            raise ValueError("agzo_rank must be positive")
        if int(agzo_power_iter_steps) <= 0:
            raise ValueError("agzo_power_iter_steps must be positive")
        if agzo_basis_method not in {"power_iter", "svd", "low_rank_svd"}:
            raise ValueError(f"unsupported agzo_basis_method: {agzo_basis_method}")
        with self._call_lock:
            self.register_direction_slots()
            raw = fused_agzo_step(
                self.llm,
                token_groups,
                token_labels=label_rows,
                labels=label_list,
                plus_lora_id=self.plus_id,
                minus_lora_id=self.minus_id,
                zo_eps=float(eps),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                max_logits_tokens=int(max_logits_tokens),
                loss_impl=loss_impl,
                activation_force_eager=bool(activation_force_eager),
                agzo_rank=int(self.rank if agzo_rank is None else agzo_rank),
                agzo_power_iter_steps=int(agzo_power_iter_steps),
                agzo_low_rank_oversample=int(agzo_low_rank_oversample),
                agzo_basis_seed=agzo_basis_seed,
                agzo_perturb_seed=agzo_perturb_seed,
                agzo_basis_method=agzo_basis_method,
                direction_scale=float(direction_scale),
            )
        return FusedAGZOStepResult(
            loss_plus=float(raw["loss_plus"]),
            loss_minus=float(raw["loss_minus"]),
            projected_grad=float(raw["projected_grad"]),
            raw=raw,
        )

    def cleanup(self) -> None:
        if self._closed:
            return
        with self._call_lock:
            try:
                self.runtime.cleanup()
            finally:
                self._slots_registered = False
                if self._owns_llm:
                    self._shutdown_owned_llm()
                self._closed = True

    def _shutdown_owned_llm(self) -> None:
        llm_engine = getattr(self.llm, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
        model_executor = getattr(llm_engine, "model_executor", None)
        shutdown = getattr(model_executor, "shutdown", None)
        if callable(shutdown):
            shutdown()

    @staticmethod
    def _token_score_from_raw(
        raw: dict[str, Any],
        *,
        loss_impl: str,
    ) -> TokenScoreResult:
        if int(raw["num_tokens"]) <= 0:
            raise ValueError("direct worker score returned zero loss tokens")
        request_nll = [float(x) for x in raw["request_nll"]]
        request_num_tokens = [int(x) for x in raw["request_num_tokens"]]
        if len(request_nll) != len(request_num_tokens):
            raise ValueError(
                "direct worker score returned mismatched request_nll and "
                "request_num_tokens lengths"
            )
        zero_indices = [
            index
            for index, num_tokens in enumerate(request_num_tokens)
            if num_tokens <= 0
        ]
        if zero_indices:
            raise ValueError(
                "direct worker score returned request(s) with zero loss tokens: "
                f"{zero_indices[:8]}"
            )
        return TokenScoreResult(
            loss=float(raw["loss"]),
            nll_sum=float(raw["nll_sum"]),
            num_tokens=int(raw["num_tokens"]),
            request_nll=request_nll,
            request_num_tokens=request_num_tokens,
            detail=direct_worker_detail(raw, 0.0, loss_impl=loss_impl),
            raw=raw,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ZOVLLMEngine is closed")

    @staticmethod
    def _validate_eps(eps: float) -> None:
        eps_f = float(eps)
        if eps_f <= 0.0 or not math.isfinite(eps_f):
            raise ValueError("eps must be positive and finite")

    @staticmethod
    def _validate_score_inputs(
        token_groups: list[list[int]],
        loss_token_lens: list[int] | None,
        labels: list[list[int]] | None,
        lora_ids: list[int] | None,
    ) -> None:
        if not token_groups:
            raise ValueError("token_id_groups must not be empty")
        short_indices = [
            index for index, token_ids in enumerate(token_groups) if len(token_ids) < 2
        ]
        if short_indices:
            raise ValueError(
                "every token_id_group must contain at least two tokens; short "
                f"indices: {short_indices[:8]}"
            )
        if loss_token_lens is not None and labels is not None:
            raise ValueError("loss_token_lens and labels are mutually exclusive")
        if loss_token_lens is not None:
            if len(loss_token_lens) != len(token_groups):
                raise ValueError("loss_token_lens must match token_id_groups length")
            bad_indices = [
                index
                for index, (token_ids, length) in enumerate(
                    zip(token_groups, loss_token_lens)
                )
                if int(length) <= 0 or int(length) > len(token_ids) - 1
            ]
            if bad_indices:
                raise ValueError(
                    "loss_token_lens must be in [1, len(tokens)-1]; bad "
                    f"indices: {bad_indices[:8]}"
                )
        if labels is not None:
            if len(labels) != len(token_groups):
                raise ValueError("labels must match token_id_groups length")
            bad_shape_indices = []
            bad_value_indices = []
            empty_indices = []
            for index, (token_ids, row) in enumerate(zip(token_groups, labels)):
                if len(row) != len(token_ids):
                    bad_shape_indices.append(index)
                    continue
                active_count = 0
                for label in row[1:]:
                    if label == -100:
                        continue
                    if label < 0:
                        bad_value_indices.append(index)
                        break
                    active_count += 1
                if active_count <= 0:
                    empty_indices.append(index)
            if bad_shape_indices:
                raise ValueError(
                    "labels rows must match token_id_groups lengths; bad "
                    f"indices: {bad_shape_indices[:8]}"
                )
            if bad_value_indices:
                raise ValueError(
                    "labels may only use -100 for ignored positions; bad "
                    f"indices: {bad_value_indices[:8]}"
                )
            if empty_indices:
                raise ValueError(
                    "labels must leave at least one active loss token per request; "
                    f"indices: {empty_indices[:8]}"
                )
        if lora_ids is not None and len(lora_ids) != len(token_groups):
            raise ValueError("lora_ids must match token_id_groups length")


__all__ = [
    "ActivationStatsResult",
    "DirectionMap",
    "ObjectiveFn",
    "PlusMinusScoreResult",
    "TokenScoreResult",
    "TokenLogitsResult",
    "TokenRequestNLLResult",
    "ZOVLLMEngine",
]
