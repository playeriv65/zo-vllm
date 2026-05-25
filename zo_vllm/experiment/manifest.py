import json
from pathlib import Path
from typing import Any

from .naming import timestamp_now


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def append_launch_record(
    manifest: dict[str, Any],
    *,
    backend: str,
    run_id: str,
    tmux_commands: list[list[str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    records = list(manifest.get("launch_records", []))
    records.append(
        {
            "timestamp": timestamp_now(),
            "backend": backend,
            "run_id": run_id,
            "tmux_commands": tmux_commands,
            "config": config,
        }
    )
    manifest["launch_records"] = records
    return manifest
