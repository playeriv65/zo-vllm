from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return project_root() / path
