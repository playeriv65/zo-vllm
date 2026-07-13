import json
from pathlib import Path
from typing import Any

from .naming import timestamp_now


def _state_path(run_dir: Path) -> Path:
    return run_dir / "run_state.json"


def read_run_state(run_dir: Path) -> dict[str, Any]:
    path = _state_path(run_dir)
    if not path.exists():
        return {
            "status": "pending",
            "updated_at": None,
            "attempts": 0,
            "resume_count": 0,
            "history": [],
        }
    with path.open() as f:
        return json.load(f)


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _state_path(run_dir).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def mark_run_running(run_dir: Path, *, note: str | None = None, resumed: bool = False) -> None:
    state = read_run_state(run_dir)
    attempts = int(state.get("attempts", 0)) + 1
    resume_count = int(state.get("resume_count", 0)) + (1 if resumed else 0)
    history = list(state.get("history", []))
    history.append({"timestamp": timestamp_now(), "event": "running", "note": note})
    state.update(
        {
            "status": "running",
            "updated_at": timestamp_now(),
            "attempts": attempts,
            "resume_count": resume_count,
            "history": history,
        }
    )
    _write_state(run_dir, state)


def mark_run_completed(run_dir: Path, *, note: str | None = None) -> None:
    state = read_run_state(run_dir)
    history = list(state.get("history", []))
    history.append({"timestamp": timestamp_now(), "event": "completed", "note": note})
    state.update({"status": "completed", "updated_at": timestamp_now(), "history": history})
    _write_state(run_dir, state)


def mark_run_failed(run_dir: Path, *, note: str | None = None) -> None:
    state = read_run_state(run_dir)
    history = list(state.get("history", []))
    history.append({"timestamp": timestamp_now(), "event": "failed", "note": note})
    state.update({"status": "failed", "updated_at": timestamp_now(), "history": history})
    _write_state(run_dir, state)
