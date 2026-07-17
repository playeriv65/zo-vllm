"""Shared storage path resolution for experiment artifacts."""

from __future__ import annotations

import os
from pathlib import Path


ZO_ARTIFACT_ROOT_ENV = "ZO_ARTIFACT_ROOT"


def configured_artifact_root() -> Path | None:
    """Return the machine-level artifact root when one is configured."""

    value = os.environ.get(ZO_ARTIFACT_ROOT_ENV)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def project_artifact_root(
    *,
    project_name: str,
    fallback_root: str | Path,
) -> Path:
    """Return the artifact root for one repository."""

    shared_root = configured_artifact_root()
    if shared_root is None:
        return Path(fallback_root).expanduser().resolve()
    return shared_root / project_name


def resolve_artifact_path(
    path: str | Path,
    *,
    project_name: str,
    fallback_root: str | Path,
) -> Path:
    """Resolve a runner path while preserving explicit absolute paths."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (
        project_artifact_root(
            project_name=project_name,
            fallback_root=fallback_root,
        )
        / candidate
    ).resolve()
