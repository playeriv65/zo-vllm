"""Probe scoring must submit device work and consume it in separate phases.

The submit phase queues the forward and must not read a device value, because
any read blocks the host on the whole forward and charges that wait to whichever
timing field encloses it. The wait belongs in ``scorer_device_wait``, once, at
the first host read. These tests pin the two properties that keep that true on a
CPU-only runtime: labels are staged before the engine call, and the wait field
is reported.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from zo_trainer.modeling import (
    ZOTrainerModel,
    _compute_grouped_probe_losses,
    _probe_loss_from_forward_outputs,
)
from zo_vllm.core.probe_results import ProbeTiming


class _OrderRecordingEngine:
    """Fake classification backend that records host operation order."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def forward_token_request_nll(
        self,
        token_id_groups,
        *,
        loss_token_lens,
        lora_ids=None,
        max_logits_tokens=8192,
        loss_impl="logprobs",
    ):
        self.events.append("engine_call")
        return SimpleNamespace(
            request_nll=torch.arange(len(token_id_groups), dtype=torch.float32).div(
                10.0
            ),
            timing=ProbeTiming(),
        )


def _model(engine: _OrderRecordingEngine) -> ZOTrainerModel:
    return ZOTrainerModel(
        SimpleNamespace(
            engine=engine,
            config=SimpleNamespace(max_logits_tokens=8192, loss_impl="logprobs"),
            direction_provider=None,
            invalidate_direction_slot_state=lambda: None,
            estimate_with_score_fn=None,
        )
    )


# Captured before any patching so repeated patches cannot nest.
_REAL_AS_TENSOR = torch.as_tensor

# Antithetic layout: two rows of two options each, scored once per LoRA slot,
# so the flattened request list and every per-row list is repeated twice.
_BASE_GROUPS = [[5, 6, 7], [5, 6, 8], [5, 9, 7], [5, 9, 8]]
_PROBE_GROUPS = _BASE_GROUPS * 2
_PROBE_KWARGS = {
    "option_loss_token_counts": [1, 1, 1, 1] * 2,
    "row_option_counts": [2, 2] * 2,
    "labels": [0, 1] * 2,
    "lora_ids": [1] * 4 + [2] * 4,
}


def _record_label_copies(monkeypatch, engine: _OrderRecordingEngine) -> None:
    def as_tensor(data, *args, **kwargs):
        if kwargs.get("dtype") is torch.long:
            engine.events.append("label_to_device")
        return _REAL_AS_TENSOR(data, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", as_tensor)


def _score(model: ZOTrainerModel, engine: _OrderRecordingEngine):
    engine.events.clear()
    return model._forward_prompt_classification_token_groups(
        _PROBE_GROUPS, **_PROBE_KWARGS
    )


def test_labels_reach_the_device_before_the_forward_is_queued(monkeypatch) -> None:
    engine = _OrderRecordingEngine()
    model = _model(engine)
    _record_label_copies(monkeypatch, engine)

    _score(model, engine)
    # The first call cannot know the backend device yet, so it still stages
    # labels afterwards and learns the device from the returned tensors.
    assert engine.events == ["engine_call", "label_to_device"]
    assert model._probe_label_device == torch.device("cpu")

    _score(model, engine)
    assert engine.events == ["label_to_device", "engine_call"]


def test_probe_loss_reports_a_device_wait_field() -> None:
    engine = _OrderRecordingEngine()
    model = _model(engine)
    outputs = _score(model, engine)
    grouped = _compute_grouped_probe_losses(
        outputs,
        compute_loss_from_outputs_fn=lambda out: torch.nn.functional.cross_entropy(
            out["logits"].float(), out["loss_labels"]
        ),
        num_requests=len(_PROBE_GROUPS),
        requests_per_group=len(_BASE_GROUPS),
        classification_rows_per_group=2,
    )
    result = _probe_loss_from_forward_outputs(
        grouped,
        outputs,
        num_requests=len(_PROBE_GROUPS),
        requests_per_group=len(_BASE_GROUPS),
    )
    # CPU losses need no wait, so the field is zero here; the contract this
    # pins is that the field exists and never reports a negative interval.
    assert result.timing.device_wait_s == 0.0
    assert "scorer_device_wait" not in result.timing.profile_seconds()
    assert (
        result.timing.updated(device_wait_s=0.5).profile_seconds()["scorer_device_wait"]
        == 0.5
    )
