"""W&B logging helpers for experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WandbRunLogger:
    """Small wrapper that keeps runner code independent from wandb imports."""

    run: Any | None = None

    def log(self, payload: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(payload, step=int(step))

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def init_wandb_logger(
    *,
    args: Any,
    resume_trainer_state: dict[str, Any] | None,
) -> WandbRunLogger:
    """Initialize W&B when requested by --report-to."""

    report_targets = {
        item.strip() for item in str(args.report_to).split(",") if item.strip()
    }
    if "wandb" not in report_targets:
        return WandbRunLogger()

    import wandb

    saved_wandb = (
        resume_trainer_state.get("wandb", {})
        if isinstance(resume_trainer_state, dict)
        else {}
    )
    saved_wandb_id = saved_wandb.get("run_id")
    saved_wandb_name = saved_wandb.get("run_name")
    wandb_kwargs = {}
    if saved_wandb_id:
        wandb_kwargs |= {"id": str(saved_wandb_id), "resume": "allow"}
    return WandbRunLogger(
        run=wandb.init(
            project=args.wandb_project,
            name=args.run_name or saved_wandb_name,
            config=vars(args),
            **wandb_kwargs,
        )
    )
