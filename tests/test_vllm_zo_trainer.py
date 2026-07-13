import json
from pathlib import Path

from zo_vllm.training import (
    VLLMZOTrainer,
    VLLMZOTrainerCallback,
    VLLMZOTrainerControl,
    ZOPendingStep,
    ZOStepResult,
    ZOTrainingArguments,
)


class CountingModel:
    def __init__(self) -> None:
        self.steps = []

    def estimate(self, batch, *, step: int):
        self.steps.append((batch, step))
        return ZOPendingStep(
            step=step,
            reported_loss=float(10 - step),
            _apply_fn=lambda learning_rate, weight_decay: ZOStepResult(
                step=step,
                learning_rate=learning_rate,
                reported_loss=float(10 - step),
                loss_plus=float(10 - step),
                loss_minus=float(10 - step),
                projected_grad=0.0,
                update_scale=0.0,
                direction_refreshed=False,
            ),
        )


def test_vllm_zo_trainer_saves_step_checkpoints_and_prunes(tmp_path: Path):
    saved = []

    def save_model(path: str) -> None:
        saved.append(Path(path).name)
        (Path(path) / "model.txt").write_text("saved\n", encoding="utf-8")

    args = ZOTrainingArguments(
        output_dir=str(tmp_path),
        max_steps=4,
        logging_steps=10,
        eval_steps=2,
        save_steps=2,
        save_strategy="steps",
        save_total_limit=1,
    )
    trainer = VLLMZOTrainer(
        model=CountingModel(),
        args=args,
        train_dataloader=["a"],
        save_model_fn=save_model,
    )

    output = trainer.train()

    assert output.global_step == 4
    assert saved == ["checkpoint-0000002", "checkpoint-0000004"]
    assert not (tmp_path / "checkpoint-0000002").exists()
    assert (tmp_path / "checkpoint-0000004" / "trainer_state.json").exists()
    state = json.loads((tmp_path / "trainer_state.json").read_text(encoding="utf-8"))
    assert [record["step"] for record in state["checkpoint_records"]] == [4]


def test_vllm_zo_trainer_records_save_model_checkpoint_metadata(tmp_path: Path):
    def save_model(path: str):
        (Path(path) / "payload.txt").write_text("saved\n", encoding="utf-8")
        return {
            "path": path,
            "mode": "native",
            "loadable": True,
            "metadata_path": str(Path(path) / "metadata.json"),
        }

    args = ZOTrainingArguments(
        output_dir=str(tmp_path),
        max_steps=1,
        logging_steps=10,
        eval_steps=1,
        save_steps=1,
        save_strategy="steps",
    )
    trainer = VLLMZOTrainer(
        model=CountingModel(),
        args=args,
        train_dataloader=["a"],
        save_model_fn=save_model,
    )

    trainer.train()

    assert trainer.state.checkpoint_records[0]["mode"] == "native"
    assert trainer.state.checkpoint_records[0]["loadable"] is True


def test_vllm_zo_trainer_best_checkpoint_loads_at_end(tmp_path: Path):
    eval_values = iter([0.8, 0.4, 0.6])
    loaded = []

    def evaluate():
        return {"eval_loss": next(eval_values)}

    def save_model(path: str) -> None:
        (Path(path) / "model.txt").write_text("saved\n", encoding="utf-8")

    def load_model(path: str) -> None:
        loaded.append(Path(path).name)

    args = ZOTrainingArguments(
        output_dir=str(tmp_path),
        max_steps=3,
        logging_steps=10,
        eval_steps=1,
        save_strategy="best",
        load_best_model_at_end=True,
    )
    trainer = VLLMZOTrainer(
        model=CountingModel(),
        args=args,
        train_dataloader=["a"],
        eval_fn=evaluate,
        save_model_fn=save_model,
        load_model_fn=load_model,
    )

    trainer.train()

    assert loaded == ["checkpoint-0000002"]
    state = json.loads((tmp_path / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["best_checkpoint"]["metric_value"] == 0.4
    assert state["best_checkpoint"]["checkpoint"]["path"].endswith(
        "checkpoint-0000002"
    )


def test_vllm_zo_trainer_resume_from_checkpoint_state(tmp_path: Path):
    args = ZOTrainingArguments(
        output_dir=str(tmp_path),
        max_steps=3,
        logging_steps=10,
        eval_steps=3,
        save_steps=2,
        save_strategy="steps",
    )
    model = CountingModel()
    trainer = VLLMZOTrainer(
        model=model,
        args=args,
        train_dataloader=["a"],
    )
    trainer.train()

    resumed_model = CountingModel()
    resumed_args = ZOTrainingArguments(
        output_dir=str(tmp_path / "resumed"),
        max_steps=4,
        logging_steps=10,
        eval_steps=4,
        save_steps=2,
        save_strategy="steps",
    )
    resumed = VLLMZOTrainer(
        model=resumed_model,
        args=resumed_args,
        train_dataloader=["a"],
    )

    output = resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-0000002"))

    assert output.global_step == 4
    assert [step for _, step in resumed_model.steps] == [3, 4]


def test_vllm_zo_trainer_runs_warmup_as_raw_steps(tmp_path: Path):
    collator_steps = []
    model = CountingModel()
    args = ZOTrainingArguments(
        output_dir=str(tmp_path),
        max_steps=2,
        warmup_steps=1,
        logging_steps=1,
        eval_steps=2,
        save_strategy="no",
    )
    trainer = VLLMZOTrainer(
        model=model,
        args=args,
        train_dataset=["a"],
        data_collator=lambda rows: collator_steps.append(list(rows)) or rows[0],
    )

    output = trainer.train()

    assert output.global_step == 2
    assert [step for _, step in model.steps] == [1, 2, 3]
    assert trainer.state.raw_step == 3
    assert trainer.state.global_step == 2
    assert len(collator_steps) == 3


def test_vllm_zo_trainer_exposes_hf_like_callback_points(tmp_path: Path):
    events = []

    class RecordingCallback(VLLMZOTrainerCallback):
        def on_init_end(self, args, state, control, **kwargs):
            events.append(("init", state.global_step))

        def on_train_begin(self, args, state, control, **kwargs):
            events.append(("train_begin", state.global_step))

        def on_step_begin(self, args, state, control, **kwargs):
            events.append(("step_begin", kwargs["raw_step"], kwargs["measured_step"]))

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            events.append(("pre_optimizer", kwargs["raw_step"]))

        def on_optimizer_step(self, args, state, control, **kwargs):
            events.append(("optimizer", kwargs["raw_step"]))

        def on_step_end(self, args, state, control, **kwargs):
            events.append(("step_end", kwargs["raw_step"], state.global_step))

        def on_log(self, args, state, control, **kwargs):
            events.append(("log", kwargs["logs"]["step"]))

        def on_evaluate(self, args, state, control, **kwargs):
            events.append(("evaluate", kwargs["metrics"]["step"]))

        def on_save(self, args, state, control, **kwargs):
            events.append(("save", Path(kwargs["checkpoint_dir"]).name))

        def on_train_end(self, args, state, control, **kwargs):
            events.append(("train_end", state.global_step))

    args = ZOTrainingArguments(
        output_dir=str(tmp_path),
        max_steps=2,
        logging_steps=1,
        eval_steps=1,
        save_steps=2,
        save_strategy="steps",
    )
    trainer = VLLMZOTrainer(
        model=CountingModel(),
        args=args,
        train_dataloader=["a"],
        eval_fn=lambda: {"eval_loss": 1.0},
        callbacks=[RecordingCallback()],
    )

    trainer.train()

    assert events[0] == ("init", 0)
    assert ("train_begin", 0) in events
    assert ("step_begin", 1, 1) in events
    assert ("pre_optimizer", 1) in events
    assert ("optimizer", 1) in events
    assert ("step_end", 2, 2) in events
    assert ("log", 1) in events
    assert ("evaluate", 2) in events
    assert ("save", "checkpoint-0000002") in events
    assert events[-1] == ("train_end", 2)
    assert events.index(("pre_optimizer", 1)) < events.index(("optimizer", 1))
    assert events.index(("optimizer", 1)) < events.index(("step_end", 1, 1))


def test_vllm_zo_trainer_callback_can_request_stop(tmp_path: Path):
    class StopAfterFirstStep(VLLMZOTrainerCallback):
        def on_step_end(
            self,
            args,
            state,
            control: VLLMZOTrainerControl,
            **kwargs,
        ):
            control.should_training_stop = True
            return control

    model = CountingModel()
    args = ZOTrainingArguments(
        output_dir=str(tmp_path),
        max_steps=4,
        logging_steps=1,
        eval_steps=4,
        save_strategy="no",
    )
    trainer = VLLMZOTrainer(
        model=model,
        args=args,
        train_dataloader=["a"],
        callbacks=[StopAfterFirstStep()],
    )

    output = trainer.train()

    assert output.global_step == 1
    assert [step for _, step in model.steps] == [1]


def test_vllm_zo_trainer_callback_has_hf_callback_method_names():
    expected_methods = {
        "on_init_end",
        "on_train_begin",
        "on_train_end",
        "on_epoch_begin",
        "on_epoch_end",
        "on_step_begin",
        "on_substep_end",
        "on_step_end",
        "on_prediction_step",
        "on_predict",
        "on_evaluate",
        "on_save",
        "on_log",
        "on_pre_optimizer_step",
        "on_optimizer_step",
    }

    assert expected_methods <= set(dir(VLLMZOTrainerCallback))


def test_vllm_zo_trainer_callback_is_top_level_public_api():
    import zo_vllm
    import zo_vllm.training

    assert zo_vllm.VLLMZOTrainerCallback is zo_vllm.training.VLLMZOTrainerCallback
    assert zo_vllm.VLLMZOTrainerControl is zo_vllm.training.VLLMZOTrainerControl
    assert zo_vllm.VLLMZOTrainerState is zo_vllm.training.VLLMZOTrainerState
    assert zo_vllm.ZOStepCallback is zo_vllm.training.ZOStepCallback
    assert zo_vllm.ZOStepControl is zo_vllm.training.ZOStepControl
    assert "VLLMZOTrainerCallback" in zo_vllm.__all__
    assert "VLLMZOTrainerControl" in zo_vllm.__all__
    assert "VLLMZOTrainerState" in zo_vllm.__all__
    assert "ZOStepCallback" in zo_vllm.__all__
    assert "ZOStepControl" in zo_vllm.__all__


def test_vllm_zo_trainer_state_uses_hf_like_attribute_api(tmp_path: Path):
    trainer = VLLMZOTrainer(
        model=CountingModel(),
        args=ZOTrainingArguments(output_dir=str(tmp_path), max_steps=1),
        train_dataloader=["a"],
    )

    trainer.set_initial_state(global_step=3, raw_step=5)

    assert trainer.state.global_step == 3
    assert trainer.state.raw_step == 5
    assert trainer.state.to_dict()["global_step"] == 3
    try:
        trainer.state["global_step"]  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("trainer state must not expose dict-style access")


def test_vllm_zo_trainer_evaluate_accepts_dataset_override(tmp_path: Path):
    calls = []

    def evaluate(eval_dataset=None):
        calls.append(list(eval_dataset or []))
        return {"eval_loss": 1.5}

    trainer = VLLMZOTrainer(
        model=CountingModel(),
        args=ZOTrainingArguments(output_dir=str(tmp_path), max_steps=1),
        train_dataloader=["a"],
        eval_fn=evaluate,
    )

    metrics = trainer.evaluate(eval_dataset=["x", "y"])

    assert metrics["eval_loss"] == 1.5
    assert metrics["global_step"] == 0
    assert calls == [["x", "y"]]
