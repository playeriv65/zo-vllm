"""Registry for supported ZO experiment tasks."""

from __future__ import annotations

from .boolq import BoolQTaskAdapter
from .squad import SquadTaskAdapter
from .sst2 import SST2TaskAdapter
from .superglue import SUPERGLUE_TASK_SPECS, SuperGLUETaskAdapter


_TASKS = {
    "sst2": SST2TaskAdapter(),
    "boolq": BoolQTaskAdapter(),
    "squad": SquadTaskAdapter(),
}
for _task_name, _spec in SUPERGLUE_TASK_SPECS.items():
    _module = __import__(
        f"zo_vllm.tasks.superglue.{_task_name.removeprefix('superglue_')}",
        fromlist=["TASK_ADAPTER"],
    )
    _adapter = getattr(_module, "TASK_ADAPTER", SuperGLUETaskAdapter(_spec))
    _TASKS[_task_name] = _adapter


def get_task(name: str):
    if name not in _TASKS:
        choices = ", ".join(sorted(_TASKS))
        raise KeyError(f"unsupported task {name!r}; supported tasks: {choices}")
    return _TASKS[name]


def list_tasks() -> list[str]:
    return sorted(_TASKS)
