import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def newest_json(directory: Path, pattern: str) -> str | None:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        return None
    return display_path(matches[-1])


def load_manifest(run_dir: Path) -> dict | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open() as f:
        return json.load(f)


def tmux_windows(session: str | None) -> list[dict]:
    if not session:
        return []
    completed = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_name} #{pane_current_command}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        if not line:
            continue
        if "p3-q1" not in line:
            continue
        parts = line.split(maxsplit=1)
        rows.append({
            "window": parts[0],
            "command": parts[1] if len(parts) > 1 else "",
        })
    return rows


def log_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return [display_path(path) for path in sorted(directory.glob("*.log"))]


def batch_dirs(run_dir: Path) -> list[Path]:
    dirs = []
    for path in run_dir.glob("batch_b*"):
        if path.is_dir():
            dirs.append(path)
    return sorted(dirs, key=batch_sort_key)


def batch_sort_key(path: Path) -> int:
    try:
        return int(path.name.removeprefix("batch_b"))
    except ValueError:
        return 10**9


def fixed_suite_status(run_dir: Path) -> dict:
    return {
        "lozo_minimal": newest_json(run_dir / "lozo_minimal", "lozo_perf_minimal_*.json"),
        "vllm_minimal_eager0": newest_json(
            run_dir / "vllm_minimal_eager0", "vllm_perf_minimal_*.json"
        ),
        "vllm_detailed_eager0": newest_json(
            run_dir / "vllm_detailed_eager0", "vllm_perf_detailed_*.json"
        ),
        "logs": log_files(run_dir / "logs"),
    }


def batch_sweep_status(run_dir: Path) -> dict:
    rows = {}
    for path in batch_dirs(run_dir):
        rows[path.name] = {
            "lozo_minimal": newest_json(path / "lozo_minimal", "lozo_perf_minimal_*.json"),
            "vllm_minimal_eager0": newest_json(
                path / "vllm_minimal_eager0", "vllm_perf_minimal_*.json"
            ),
            "vllm_detailed_eager0": newest_json(
                path / "vllm_detailed_eager0", "vllm_perf_detailed_*.json"
            ),
            "logs": log_files(path / "logs"),
        }
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a Phase 3 q=1 run directory.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    args.run_dir = resolve_path(args.run_dir)
    return args


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    manifest = load_manifest(run_dir)
    tmux_session = manifest.get("tmux_session") if manifest else None
    batches = batch_dirs(run_dir)
    run_type = "batch_sweep" if batches else "speed_suite"
    artifacts = batch_sweep_status(run_dir) if batches else fixed_suite_status(run_dir)
    report = {
        "exists": run_dir.exists(),
        "run_dir": display_path(run_dir),
        "run_type": run_type,
        "manifest_exists": manifest is not None,
        "summary_exists": (run_dir / "summary.md").exists(),
        "validation_exists": (run_dir / "validation.json").exists(),
        "tmux_session": tmux_session,
        "tmux_windows": tmux_windows(tmux_session),
        "artifacts": artifacts,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
