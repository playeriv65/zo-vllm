"""Runtime model facade used by the Hugging Face ZO trainer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import time
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

import torch
from torch import nn
from transformers.utils import ModelOutput

from zo_vllm.core.token_groups import (
    borrow_or_copy_int_list,
    borrow_or_copy_token_groups,
)
from zo_vllm.core.probe_results import ProbeLossResult, ProbeTiming
from zo_vllm.training.direction import TokenProbeBatch

if TYPE_CHECKING:
    from zo_vllm.training.zo_step import ZOPendingStep


@runtime_checkable
class ZOTrainerRuntime(Protocol):
    """Runtime capabilities required by the Hugging Face model facade."""

    engine: Any
    config: Any
    direction_provider: Any

    def invalidate_direction_slot_state(self) -> None: ...

    def estimate_with_score_fn(
        self,
        batch: TokenProbeBatch,
        *,
        step: int,
        score_fn: Callable[..., object],
    ) -> ZOPendingStep: ...


@dataclass
class CompactCausalOutput(ModelOutput):
    """Compact active-token logits for the Hugging Face causal-LM loss."""

    logits: torch.Tensor | None = None
    loss_labels: torch.Tensor | None = None
    request_loss_token_counts: tuple[int, ...] | None = None
    timing: ProbeTiming | None = None


@dataclass
class OptionClassificationOutput(ModelOutput):
    """Per-row option logits for the Hugging Face classification loss."""

    logits: torch.Tensor | None = None
    loss_labels: torch.Tensor | None = None
    timing: ProbeTiming | None = None


class ZOTrainerModel(nn.Module):
    """HF-facing module facade around a ZO-vLLM runtime model.

    The public forward path accepts Hugging Face-native tokenized batches. The
    vLLM backend returns compact logits for the active causal-LM loss
    positions. Loss is computed from those logits through the same Transformers
    causal-LM loss helper used by native model implementations.
    """

    main_input_name = "input_ids"

    def __init__(self, zo_model: ZOTrainerRuntime) -> None:
        super().__init__()
        self.zo_model = zo_model
        self._dummy_param = nn.Parameter(torch.zeros(()))

    def forward(
        self, **inputs: Any
    ) -> CompactCausalOutput | OptionClassificationOutput:
        private_fields = sorted(key for key in inputs if key.startswith("_zo_"))
        if private_fields:
            raise ValueError(
                "HF batches cannot select runtime-private state: "
                + ", ".join(private_fields)
            )
        if self.zo_model.engine is None:
            raise RuntimeError("ZOTrainerModel.forward requires a runtime engine")
        forward_t0 = time.perf_counter()
        unpack_t0 = time.perf_counter()
        if _is_prompt_classification_batch(inputs):
            token_groups, option_loss_token_counts, row_option_counts, labels = (
                hf_prompt_classification_batch_to_token_groups(inputs)
            )
            unpack_s = time.perf_counter() - unpack_t0
            outputs = self._forward_prompt_classification_token_groups(
                token_groups,
                option_loss_token_counts=option_loss_token_counts,
                row_option_counts=row_option_counts,
                labels=labels,
                lora_ids=None,
            )
        else:
            token_groups, labels = hf_batch_to_token_groups(inputs)
            unpack_s = time.perf_counter() - unpack_t0
            outputs = self._forward_token_groups(
                token_groups,
                labels=labels,
                lora_ids=None,
            )
        if not isinstance(outputs.timing, ProbeTiming):
            raise RuntimeError("ZO model output must include ProbeTiming")
        return replace(
            outputs,
            timing=outputs.timing.updated(
                forward_unpack_s=unpack_s,
                forward_total_s=time.perf_counter() - forward_t0,
            ),
        )

    def zo_estimate(
        self,
        inputs: Mapping[str, Any],
        *,
        step: int,
        compute_loss_from_outputs_fn: Callable[[Mapping[str, Any]], torch.Tensor],
    ) -> ZOPendingStep:
        batch_unpack_t0 = time.perf_counter()
        classification_context = None
        if _is_prompt_classification_batch(inputs):
            token_groups, option_loss_token_counts, row_option_counts, class_labels = (
                hf_prompt_classification_batch_to_token_groups(inputs)
            )
            labels = None
            classification_context = {
                "option_loss_token_counts": option_loss_token_counts,
                "row_option_counts": row_option_counts,
                "labels": class_labels,
                "device_label_cache": {},
            }
        else:
            token_groups, labels = hf_batch_to_token_groups(inputs)
            option_loss_token_counts = None
        batch_unpack_s = time.perf_counter() - batch_unpack_t0

        def score_fn(
            *,
            token_id_groups,
            loss_token_lens=None,
            labels=None,
            lora_ids=None,
            max_logits_tokens: int = 8192,
            loss_impl: str = "logprobs",
        ) -> ProbeLossResult:
            score_fn_t0 = time.perf_counter()
            if classification_context is None:
                if labels is None:
                    raise RuntimeError("causal-LM probe scoring requires labels")
                outputs = self._forward_token_groups(
                    token_id_groups,
                    labels=labels,
                    lora_ids=lora_ids,
                    max_logits_tokens=max_logits_tokens,
                    loss_impl=loss_impl,
                )
            else:
                repeated_context = _repeat_classification_context(
                    classification_context,
                    actual_token_groups=len(token_id_groups),
                )
                outputs = self._forward_prompt_classification_token_groups(
                    token_id_groups,
                    option_loss_token_counts=repeated_context[
                        "option_loss_token_counts"
                    ],
                    row_option_counts=repeated_context["row_option_counts"],
                    labels=repeated_context["labels"],
                    lora_ids=lora_ids,
                    max_logits_tokens=max_logits_tokens,
                    loss_impl=loss_impl,
                    device_label_cache=classification_context["device_label_cache"],
                )
            compute_loss_t0 = time.perf_counter()
            grouped_losses = _compute_grouped_probe_losses(
                outputs,
                compute_loss_from_outputs_fn=compute_loss_from_outputs_fn,
                num_requests=len(token_id_groups),
                requests_per_group=len(token_groups),
                classification_rows_per_group=(
                    None
                    if classification_context is None
                    else len(classification_context["labels"])
                ),
            )
            loss_dispatch_s = time.perf_counter() - compute_loss_t0
            probe_result = _probe_loss_from_forward_outputs(
                grouped_losses,
                outputs,
                num_requests=len(token_id_groups),
                requests_per_group=len(token_groups),
            )
            return replace(
                probe_result,
                timing=probe_result.timing.updated(
                    batch_unpack_s=batch_unpack_s,
                    loss_dispatch_s=loss_dispatch_s,
                    total_s=time.perf_counter() - score_fn_t0,
                ),
            )

        return self.zo_model.estimate_with_score_fn(
            TokenProbeBatch(
                token_id_groups=token_groups,
                loss_token_lens=option_loss_token_counts,
                labels=labels,
            ),
            step=int(step),
            score_fn=score_fn,
        )

    def _forward_token_groups(
        self,
        token_groups: Sequence[Sequence[int]],
        *,
        labels: Sequence[Sequence[int]] | None,
        lora_ids: Sequence[int] | None,
        max_logits_tokens: int | None = None,
        loss_impl: str | None = None,
    ) -> CompactCausalOutput:
        engine = self.zo_model.engine
        config = self.zo_model.config
        forward_t0 = time.perf_counter()
        engine_t0 = time.perf_counter()
        result = engine.forward_token_logits(
            token_groups,
            labels=labels,
            lora_ids=lora_ids,
            max_logits_tokens=int(
                config.max_logits_tokens
                if max_logits_tokens is None
                else max_logits_tokens
            ),
            loss_impl=str(config.loss_impl if loss_impl is None else loss_impl),
        )
        return CompactCausalOutput(
            logits=result.logits,
            loss_labels=result.target_token_ids,
            request_loss_token_counts=result.request_loss_token_counts,
            timing=result.timing.updated(
                engine_call_s=time.perf_counter() - engine_t0,
                forward_total_s=time.perf_counter() - forward_t0,
            ),
        )

    def _forward_prompt_classification_token_groups(
        self,
        token_groups: Sequence[Sequence[int]],
        *,
        option_loss_token_counts: Sequence[int],
        row_option_counts: Sequence[int],
        labels: Sequence[int],
        lora_ids: Sequence[int] | None,
        max_logits_tokens: int | None = None,
        loss_impl: str | None = None,
        device_label_cache: dict[tuple[str, tuple[int, ...]], torch.Tensor]
        | None = None,
    ) -> OptionClassificationOutput:
        engine = self.zo_model.engine
        config = self.zo_model.config
        forward_t0 = time.perf_counter()
        engine_t0 = time.perf_counter()
        result = engine.forward_token_request_nll(
            token_groups,
            loss_token_lens=option_loss_token_counts,
            lora_ids=lora_ids,
            max_logits_tokens=int(
                config.max_logits_tokens
                if max_logits_tokens is None
                else max_logits_tokens
            ),
            loss_impl=str(config.loss_impl if loss_impl is None else loss_impl),
        )
        engine_s = time.perf_counter() - engine_t0
        postprocess_t0 = time.perf_counter()
        logits = _classification_logits_from_request_nll(
            result.request_nll,
            row_option_counts,
        )
        label_key = (str(logits.device), tuple(int(value) for value in labels))
        label_tensor = None
        if device_label_cache is not None:
            label_tensor = device_label_cache.get(label_key)
        if label_tensor is None:
            label_tensor = torch.as_tensor(
                labels, dtype=torch.long, device=logits.device
            )
            if device_label_cache is not None:
                device_label_cache[label_key] = label_tensor
        return OptionClassificationOutput(
            logits=logits,
            loss_labels=label_tensor,
            timing=result.timing.updated(
                engine_call_s=engine_s,
                output_postprocess_s=time.perf_counter() - postprocess_t0,
                forward_total_s=time.perf_counter() - forward_t0,
            ),
        )


def hf_batch_to_token_groups(
    inputs: Mapping[str, Any],
) -> tuple[list[list[int]], list[list[int]]]:
    """Convert an HF batch into ragged vLLM prompt ids and label masks."""

    input_rows, token_groups, active_indices_rows = _trim_hf_input_rows(inputs)
    if "labels" not in inputs or inputs["labels"] is None:
        raise ValueError("HF-native ZO batches must contain labels")
    label_rows = _to_nested_int_lists(inputs["labels"], name="labels")
    if len(label_rows) != len(input_rows):
        raise ValueError("labels must have the same batch size as input_ids")
    token_labels: list[list[int]] = []
    for ids, labels, active_indices in zip(input_rows, label_rows, active_indices_rows):
        if len(labels) != len(ids):
            raise ValueError("labels rows must match input_ids rows")
        token_labels.append(
            labels
            if active_indices is None
            else [int(labels[pos]) for pos in active_indices]
        )
    return token_groups, token_labels


def hf_prompt_classification_batch_to_token_groups(
    inputs: Mapping[str, Any],
) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    """Convert a flattened prompt-classification HF batch for vLLM scoring."""

    if "input_ids" not in inputs:
        raise ValueError("prompt classification batches must contain input_ids")
    _, token_groups, _ = _trim_hf_input_rows(
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
        }
    )
    option_loss_token_counts = _to_flat_int_list(
        inputs["option_loss_token_counts"], name="option_loss_token_counts"
    )
    row_option_counts = _to_flat_int_list(
        inputs["row_option_counts"], name="row_option_counts"
    )
    if "labels" not in inputs:
        raise ValueError("prompt classification batches require labels")
    labels = _to_flat_int_list(inputs["labels"], name="labels")
    if len(option_loss_token_counts) != len(token_groups):
        raise ValueError(
            "option_loss_token_counts must match flattened candidate count"
        )
    if sum(row_option_counts) != len(token_groups):
        raise ValueError("sum(row_option_counts) must match candidate count")
    if len(labels) != len(row_option_counts):
        raise ValueError("classification labels must match row count")
    for row_index, (label, count) in enumerate(zip(labels, row_option_counts)):
        if int(count) <= 0:
            raise ValueError(f"row {row_index} has no options")
        if int(label) < 0 or int(label) >= int(count):
            raise ValueError(
                f"classification label out of range at row {row_index}: "
                f"label={label}, row_option_counts={count}"
            )
    return token_groups, option_loss_token_counts, row_option_counts, labels


def _trim_hf_input_rows(
    inputs: Mapping[str, Any],
) -> tuple[list[list[int]], list[list[int]], list[list[int] | None]]:
    if "input_ids" not in inputs:
        raise ValueError("HF-native ZO batches must contain input_ids")
    input_rows = _to_nested_int_lists(inputs["input_ids"], name="input_ids")
    attention_rows = None
    if inputs.get("attention_mask") is not None:
        attention_rows = _to_nested_int_lists(
            inputs["attention_mask"], name="attention_mask"
        )
        if len(attention_rows) != len(input_rows):
            raise ValueError("attention_mask must match input_ids batch size")
    token_groups: list[list[int]] = []
    active_indices_rows: list[list[int] | None] = []
    for index, ids in enumerate(input_rows):
        if attention_rows is None:
            active_indices = None
            trimmed_ids = ids
        else:
            mask = attention_rows[index]
            if len(mask) != len(ids):
                raise ValueError("attention_mask rows must match input_ids rows")
            active_indices = [pos for pos, item in enumerate(mask) if int(item) != 0]
            trimmed_ids = [int(ids[pos]) for pos in active_indices]
        if not trimmed_ids:
            raise ValueError("input_ids rows must contain at least one active token")
        token_groups.append(trimmed_ids)
        active_indices_rows.append(active_indices)
    return input_rows, token_groups, active_indices_rows


def _to_nested_int_lists(value: Any, *, name: str) -> list[list[int]]:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a ragged nested sequence, not a tensor")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a ragged nested sequence")
    if not value:
        return []
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        return borrow_or_copy_token_groups(value) or []
    raise TypeError(f"{name} must contain one sequence per batch row")


def _to_flat_int_list(value: Any, *, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a sequence, not a tensor")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return borrow_or_copy_int_list(value) or []


def _is_prompt_classification_batch(inputs: Mapping[str, Any]) -> bool:
    return "option_loss_token_counts" in inputs and "row_option_counts" in inputs


def _classification_logits_from_request_nll(
    request_nll: Sequence[float] | torch.Tensor,
    row_option_counts: Sequence[int],
) -> torch.Tensor:
    option_counts = [int(item) for item in row_option_counts]
    flat_nll = (
        request_nll.float().view(-1)
        if isinstance(request_nll, torch.Tensor)
        else torch.tensor([float(item) for item in request_nll], dtype=torch.float32)
    )
    if sum(option_counts) != int(flat_nll.numel()):
        raise ValueError("request_nll does not match row_option_counts")
    if option_counts and len(set(option_counts)) == 1:
        return -flat_nll.view(len(option_counts), option_counts[0])
    max_options = max(option_counts)
    logits = torch.full(
        (len(option_counts), max_options),
        float("-inf"),
        dtype=torch.float32,
        device=flat_nll.device,
    )
    start = 0
    for row_index, count in enumerate(option_counts):
        logits[row_index, :count] = -flat_nll[start : start + count]
        start += count
    return logits


def _repeat_classification_context(
    context: Mapping[str, list[int]],
    *,
    actual_token_groups: int,
) -> dict[str, list[int]]:
    base_option_lens = [int(item) for item in context["option_loss_token_counts"]]
    base_num_options = [int(item) for item in context["row_option_counts"]]
    base_labels = [int(item) for item in context["labels"]]
    base_count = len(base_option_lens)
    if base_count <= 0 or int(actual_token_groups) % base_count != 0:
        raise ValueError(
            "classification probe token group count must be a multiple of "
            "the original flattened candidate count"
        )
    repeats = int(actual_token_groups) // base_count
    return {
        "option_loss_token_counts": base_option_lens * repeats,
        "row_option_counts": base_num_options * repeats,
        "labels": base_labels * repeats,
    }


def _probe_loss_from_forward_outputs(
    grouped_losses: Sequence[Any],
    outputs: Any,
    *,
    num_requests: int,
    requests_per_group: int,
) -> ProbeLossResult:
    loss_values, loss_to_host_s = _float_losses(grouped_losses)
    if not loss_values:
        raise ValueError("grouped probe losses must not be empty")
    if int(requests_per_group) <= 0:
        raise ValueError("requests_per_group must be positive")
    if len(loss_values) * int(requests_per_group) != int(num_requests):
        raise ValueError("grouped probe losses do not cover all requests")

    timing = _output_value(outputs, "timing")
    if not isinstance(timing, ProbeTiming):
        raise RuntimeError("ZO model output must include ProbeTiming")
    return ProbeLossResult(
        group_losses=tuple(loss_values),
        requests_per_group=int(requests_per_group),
        timing=timing.resolve_cuda_events().updated(loss_to_host_s=loss_to_host_s),
    )


def _compute_grouped_probe_losses(
    outputs: Mapping[str, Any],
    *,
    compute_loss_from_outputs_fn: Callable[[Mapping[str, Any]], torch.Tensor],
    num_requests: int,
    requests_per_group: int,
    classification_rows_per_group: int | None,
) -> list[torch.Tensor]:
    if int(requests_per_group) <= 0 or int(num_requests) % int(requests_per_group):
        raise ValueError("probe requests must contain complete objective groups")
    group_count = int(num_requests) // int(requests_per_group)
    if group_count == 1:
        return [compute_loss_from_outputs_fn(outputs)]

    logits = _output_value(outputs, "logits")
    labels = _output_value(outputs, "loss_labels")
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise RuntimeError("grouped probe loss requires tensor logits and labels")

    losses: list[torch.Tensor] = []
    if classification_rows_per_group is not None:
        rows_per_group = int(classification_rows_per_group)
        if rows_per_group <= 0 or logits.shape[0] != group_count * rows_per_group:
            raise ValueError("classification logits do not match objective groups")
        for group_index in range(group_count):
            start = group_index * rows_per_group
            end = start + rows_per_group
            grouped_outputs = OptionClassificationOutput(
                logits=logits[start:end],
                loss_labels=labels[start:end],
                timing=_output_value(outputs, "timing"),
            )
            losses.append(compute_loss_from_outputs_fn(grouped_outputs))
        return losses

    request_loss_token_counts = _output_value(outputs, "request_loss_token_counts")
    if not isinstance(request_loss_token_counts, Sequence) or isinstance(
        request_loss_token_counts, (str, bytes)
    ):
        raise RuntimeError("grouped causal-LM loss requires request_loss_token_counts")
    token_counts = [int(value) for value in request_loss_token_counts]
    if len(token_counts) != int(num_requests) or any(
        value <= 0 for value in token_counts
    ):
        raise ValueError("request_loss_token_counts must cover every probe request")
    token_offsets = [0]
    for token_count in token_counts:
        token_offsets.append(token_offsets[-1] + token_count)
    if token_offsets[-1] != int(logits.shape[0]) or logits.shape[0] != labels.shape[0]:
        raise ValueError("compact logits do not match request_loss_token_counts")
    for group_index in range(group_count):
        request_start = group_index * int(requests_per_group)
        request_end = request_start + int(requests_per_group)
        loss_start = token_offsets[request_start]
        loss_end = token_offsets[request_end]
        grouped_outputs = CompactCausalOutput(
            logits=logits[loss_start:loss_end],
            loss_labels=labels[loss_start:loss_end],
            request_loss_token_counts=tuple(token_counts[request_start:request_end]),
            timing=_output_value(outputs, "timing"),
        )
        losses.append(compute_loss_from_outputs_fn(grouped_outputs))
    return losses


def _float_loss(loss: Any) -> float:
    if isinstance(loss, torch.Tensor):
        return float(loss.detach().cpu().item())
    return float(loss)


def _float_losses(losses: Sequence[Any]) -> tuple[list[float], float]:
    sync_t0 = time.perf_counter()
    tensor_losses = [loss for loss in losses if isinstance(loss, torch.Tensor)]
    if len(tensor_losses) == len(losses) and tensor_losses:
        devices = {loss.device for loss in tensor_losses}
        if len(devices) == 1:
            values = torch.stack(
                [loss.detach().reshape(()) for loss in tensor_losses]
            ).float()
            output = [float(value) for value in values.cpu().tolist()]
            return output, time.perf_counter() - sync_t0
    output = [_float_loss(loss) for loss in losses]
    return output, time.perf_counter() - sync_t0


def _output_value(outputs: Any, key: str) -> Any:
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"ZO model output must be a mapping, got {type(outputs).__name__}"
        )
    return outputs[key]
