import json
import glob
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_json_line(line: str) -> dict:
    return json.loads(line)


def newest_path(pattern: str) -> Path | None:
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None
