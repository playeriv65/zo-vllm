from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from zo_vllm.core.direct_worker_scorer import (
    forward_token_id_logits,
    forward_token_id_request_nll,
)
from zo_vllm.core.probe_results import ProbeTiming


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def collective_rpc(self, fn, *, kwargs, single_value):
        self.calls.append(
            {
                "fn": fn,
                "kwargs": dict(kwargs),
                "single_value": bool(single_value),
            }
        )
        return {
            "loss": 1.2,
            "nll_sum": 6.0,
            "num_tokens": 5,
            "request_weighted_nll": [3.0, 3.0],
            "request_num_tokens": [2, 3],
            "logits": torch.ones((5, 7), dtype=torch.float32),
            "target_token_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "loss_segments": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            "request_loss_token_counts": [2, 3],
            "request_nll_tensor": torch.tensor([1.5, 1.0]),
        }


class FakeCudaEvent:
    def __init__(self, elapsed_ms: float) -> None:
        self.elapsed_ms = float(elapsed_ms)

    def elapsed_time(self, end) -> float:
        assert isinstance(end, FakeCudaEvent)
        return end.elapsed_ms - self.elapsed_ms


def test_probe_timing_resolves_pending_cuda_events_after_host_sync() -> None:
    timing = ProbeTiming(
        pending_cuda_events=(
            ("model_forward_ms", FakeCudaEvent(2.0), FakeCudaEvent(5.5)),
        )
    )

    resolved = timing.resolve_cuda_events()

    assert resolved.pending_cuda_events == ()
    assert resolved.profile_seconds()["worker_cuda_model_forward"] == pytest.approx(
        0.0035
    )


def test_forward_token_id_logits_requests_only_worker_logits():
    executor = RecordingExecutor()
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(model_executor=executor),
    )

    result = forward_token_id_logits(
        llm,
        [[1, 2, 3], [4, 5, 6, 7]],
        lora_ids=[11, 12],
        labels=[[-100, 2, 3], [-100, -100, 6, 7]],
        max_logits_tokens=128,
        loss_impl="logprobs",
    )

    call = executor.calls[0]
    assert call["fn"] == "zo_score_prompt_token_ids"
    assert call["single_value"] is True
    assert call["kwargs"] == {
        "prompt_token_ids": [[1, 2, 3], [4, 5, 6, 7]],
        "lora_ids": [11, 12],
        "labels": [[-100, 2, 3], [-100, -100, 6, 7]],
        "max_logits_tokens": 128,
        "loss_impl": "logprobs",
        "return_logits": True,
        "compute_token_nll": False,
        "return_device_tensors": False,
    }
    assert "request_nll" not in result
    assert result["logits"].shape == (5, 7)
    assert result["target_token_ids"].tolist() == [1, 2, 3, 4, 5]


def test_forward_token_id_request_nll_keeps_tensor_output() -> None:
    executor = RecordingExecutor()
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(model_executor=executor),
    )

    result = forward_token_id_request_nll(
        llm,
        [[1, 2, 3], [4, 5, 6, 7]],
        lora_ids=[11, 12],
        loss_token_lens=[2, 3],
        max_logits_tokens=128,
        loss_impl="logprobs",
        return_device_tensors=True,
    )

    assert executor.calls[0]["kwargs"] == {
        "prompt_token_ids": [[1, 2, 3], [4, 5, 6, 7]],
        "lora_ids": [11, 12],
        "loss_token_lens": [2, 3],
        "labels": None,
        "max_logits_tokens": 128,
        "loss_impl": "logprobs",
        "compute_token_nll": True,
        "return_request_nll_tensor": True,
        "return_device_tensors": True,
    }
    assert result["request_nll_tensor"].tolist() == [1.5, 1.0]
