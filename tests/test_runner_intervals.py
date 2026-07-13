from types import SimpleNamespace

from zo_vllm.experiment.runners.intervals import resolve_runner_intervals


def _args(**overrides):
    defaults = dict(
        eval_interval=100,
        progress_interval=10,
        train_loss_interval=25,
        save_steps=None,
        eval_interval_epochs=0.0,
        progress_interval_epochs=0.0,
        train_loss_interval_epochs=0.0,
        batch_size=8,
        dataloader_drop_last=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_runner_intervals_use_step_defaults():
    intervals = resolve_runner_intervals(
        args=_args(save_steps=40),
        num_train_items=33,
    )

    assert intervals.eval == 100
    assert intervals.progress == 10
    assert intervals.train_loss == 25
    assert intervals.save == 40


def test_runner_intervals_apply_epoch_overrides():
    intervals = resolve_runner_intervals(
        args=_args(
            eval_interval_epochs=1.0,
            progress_interval_epochs=0.5,
            train_loss_interval_epochs=2.0,
            save_steps=70,
            batch_size=8,
            dataloader_drop_last=1,
        ),
        num_train_items=33,
    )

    assert intervals.eval == 4
    assert intervals.progress == 2
    assert intervals.train_loss == 8
    assert intervals.save == 70
