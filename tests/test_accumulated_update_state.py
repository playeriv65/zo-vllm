import pytest
import torch

from zo_vllm.training.update_state import AccumulatedLowRankUpdateState


class FakeWeightSync:
    def __init__(self):
        self.calls = []

    def apply_lozo_update(self, directions, **kwargs):
        self.calls.append((directions, kwargs))
        return {"weight_update_s": 0.0}


def test_accumulated_update_does_not_fold_by_step_number():
    weight_sync = FakeWeightSync()
    state = AccumulatedLowRankUpdateState(weight_sync=weight_sync)
    directions = {
        "layer.weight": {
            "U": torch.ones((2, 1), dtype=torch.float32),
            "V": torch.ones((3, 1), dtype=torch.float32),
        }
    }

    info = state.apply(
        directions,
        projected_grad=2.0,
        learning_rate=0.5,
        weight_decay=0.0,
        step=50,
    )

    assert info["fold_s"] == 0.0
    assert weight_sync.calls == []
    assert torch.linalg.vector_norm(state.accumulated_u["layer.weight"]).item() > 0.0


def test_accumulated_update_folds_only_on_refresh_request():
    weight_sync = FakeWeightSync()
    state = AccumulatedLowRankUpdateState(weight_sync=weight_sync)
    directions = {
        "layer.weight": {
            "U": torch.ones((2, 1), dtype=torch.float32),
            "V": torch.ones((3, 1), dtype=torch.float32),
        }
    }
    state.apply(
        directions,
        projected_grad=2.0,
        learning_rate=0.5,
        weight_decay=0.0,
        step=50,
    )

    state.fold_before_direction_refresh(step=51)

    assert len(weight_sync.calls) == 1
    assert torch.linalg.vector_norm(state.accumulated_u["layer.weight"]).item() == 0.0


def test_accumulated_update_rejects_unsupported_weight_decay():
    state = AccumulatedLowRankUpdateState(weight_sync=FakeWeightSync())
    directions = {
        "layer.weight": {
            "U": torch.ones((2, 1), dtype=torch.float32),
            "V": torch.ones((3, 1), dtype=torch.float32),
        }
    }

    with pytest.raises(ValueError, match="does not support weight_decay"):
        state.apply(
            directions,
            projected_grad=2.0,
            learning_rate=0.5,
            weight_decay=0.1,
            step=1,
        )
