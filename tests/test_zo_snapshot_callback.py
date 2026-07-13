from types import SimpleNamespace

import torch

from zo_trainer import ZOUSnapshotCallback
from zo_vllm.experiment.runners.u_snapshot import USnapshotRecorder
from zo_vllm.training import inspect_training_artifact


class SnapshotState:
    def snapshot_metrics(self, *, step, previous_flat, previous_delta):
        del previous_flat, previous_delta
        return (
            {
                "step": int(step),
                "u_total_norm": 1.0,
                "u_delta_norm_since_prev": None,
                "u_cosine_to_prev": None,
                "u_angle_deg_to_prev": None,
                "delta_w_fro_est": 2.0,
                "u_delta_cosine_to_prev_delta": None,
                "u_delta_angle_deg_to_prev_delta": None,
            },
            torch.tensor([1.0]),
            None,
        )

    def snapshot_tensors(self, *, dtype):
        return {"layer": torch.tensor([1.0], dtype=dtype)}


def test_hf_snapshot_callback_preserves_study_artifact_schema(tmp_path) -> None:
    callback = ZOUSnapshotCallback(
        recorder=USnapshotRecorder(
            root=str(tmp_path),
            interval=5,
            dtype_name="float16",
            total_limit=2,
            update_state=SnapshotState(),
            log_wandb=lambda metrics, step: None,
        ),
    )
    control = SimpleNamespace(should_evaluate=True)
    state = SimpleNamespace(global_step=5)

    callback.on_step_end(None, state, control)
    callback.on_evaluate(
        None,
        state,
        control,
        metrics={"eval_loss": 0.4, "eval_accuracy": 0.75},
    )

    path = tmp_path / "u_step_0000005.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metrics"]["eval_loss"] == 0.4
    assert payload["metrics"]["eval_accuracy"] == 0.75
    assert inspect_training_artifact(path).kind == "u_snapshot"
