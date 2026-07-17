from types import SimpleNamespace

from zo_vllm.experiment.infra.storage import (
    project_artifact_root,
    resolve_artifact_path,
)
from zo_vllm.experiment.runners.output import resolve_vllm_output_paths


def test_artifact_paths_use_project_root_without_shared_storage(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ZO_ARTIFACT_ROOT", raising=False)

    assert (
        project_artifact_root(
            project_name="zo-vllm",
            fallback_root="/repo",
        ).as_posix()
        == "/repo"
    )
    assert (
        resolve_artifact_path(
            "runs/example",
            project_name="zo-vllm",
            fallback_root="/repo",
        ).as_posix()
        == "/repo/runs/example"
    )


def test_artifact_paths_use_machine_root_for_relative_paths(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ZO_ARTIFACT_ROOT", "/shared/artifacts")

    assert (
        project_artifact_root(
            project_name="zo-vllm",
            fallback_root="/repo",
        ).as_posix()
        == "/shared/artifacts/zo-vllm"
    )
    assert (
        resolve_artifact_path(
            "runs/example",
            project_name="zo-post",
            fallback_root="/repo",
        ).as_posix()
        == "/shared/artifacts/zo-post/runs/example"
    )


def test_artifact_paths_preserve_explicit_absolute_output(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ZO_ARTIFACT_ROOT", "/shared/artifacts")

    assert (
        resolve_artifact_path(
            "/explicit/run",
            project_name="zo-vllm",
            fallback_root="/repo",
        ).as_posix()
        == "/explicit/run"
    )


def test_vllm_runner_defaults_follow_machine_artifact_root(monkeypatch) -> None:
    monkeypatch.setenv("ZO_ARTIFACT_ROOT", "/shared/artifacts")
    args = SimpleNamespace(
        output_dir=None,
        output_root=None,
        experiment_name="example",
        direction_provider="lozo",
        train_objective="sst2",
        rank=2,
        nu=50,
    )

    paths = resolve_vllm_output_paths(
        args=args,
        project_root="/repo",
        model_name="facebook/opt-2.7b",
        timestamp="20260716_120000",
    )

    assert paths.output_root == "/shared/artifacts/zo-vllm/phase3/results"
    assert paths.output_dir == ("/shared/artifacts/zo-vllm/phase3/results/example/vllm")
