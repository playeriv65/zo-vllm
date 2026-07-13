from types import SimpleNamespace
import sys

from zo_vllm.experiment.runners.wandb_logging import (
    WandbRunLogger,
    init_wandb_logger,
)


class _FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, payload, step):
        self.logged.append((payload, step))

    def finish(self):
        self.finished = True


def test_wandb_logger_noops_without_run():
    logger = WandbRunLogger()

    logger.log({"x": 1}, step=3)
    logger.finish()

    assert logger.run is None


def test_init_wandb_logger_skips_when_not_requested(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", object())

    logger = init_wandb_logger(
        args=SimpleNamespace(report_to="none", wandb_project="p", run_name=None),
        resume_trainer_state=None,
    )

    assert logger.run is None


def test_init_wandb_logger_resumes_saved_run(monkeypatch):
    calls = []
    fake_run = _FakeRun()

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            calls.append(kwargs)
            return fake_run

    args = SimpleNamespace(
        report_to="tensorboard,wandb",
        wandb_project="proj",
        run_name=None,
    )

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    logger = init_wandb_logger(
        args=args,
        resume_trainer_state={"wandb": {"run_id": "abc", "run_name": "old-name"}},
    )
    logger.log({"metric": 1.0}, step=7)
    logger.finish()

    assert logger.run is fake_run
    assert calls == [
        {
            "project": "proj",
            "name": "old-name",
            "config": vars(args),
            "id": "abc",
            "resume": "allow",
        }
    ]
    assert fake_run.logged == [({"metric": 1.0}, 7)]
    assert fake_run.finished is True
