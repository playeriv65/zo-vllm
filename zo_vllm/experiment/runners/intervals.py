"""Step interval resolution for experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zo_vllm.experiment.infra.batching import epoch_interval_to_steps


@dataclass(frozen=True)
class RunnerIntervals:
    """Effective step intervals after applying epoch-based overrides."""

    eval: int
    progress: int
    train_loss: int
    save: int


def resolve_runner_intervals(
    *,
    args: Any,
    num_train_items: int,
) -> RunnerIntervals:
    """Resolve runner intervals from step and epoch CLI settings."""

    eval_interval = int(args.eval_interval)
    progress_interval = int(args.progress_interval)
    train_loss_interval = int(args.train_loss_interval)
    save_steps = 0 if args.save_steps is None else int(args.save_steps)
    if float(args.eval_interval_epochs) > 0.0:
        eval_interval = epoch_interval_to_steps(
            args.eval_interval_epochs,
            num_items=num_train_items,
            batch_size=args.batch_size,
            drop_last=bool(args.dataloader_drop_last),
        )
    if float(args.progress_interval_epochs) > 0.0:
        progress_interval = epoch_interval_to_steps(
            args.progress_interval_epochs,
            num_items=num_train_items,
            batch_size=args.batch_size,
            drop_last=bool(args.dataloader_drop_last),
        )
    if float(args.train_loss_interval_epochs) > 0.0:
        train_loss_interval = epoch_interval_to_steps(
            args.train_loss_interval_epochs,
            num_items=num_train_items,
            batch_size=args.batch_size,
            drop_last=bool(args.dataloader_drop_last),
        )
    return RunnerIntervals(
        eval=eval_interval,
        progress=progress_interval,
        train_loss=train_loss_interval,
        save=save_steps,
    )
