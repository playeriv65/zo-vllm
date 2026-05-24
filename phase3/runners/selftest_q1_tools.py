import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "phase3"
    / "results"
    / "phase3_q1_speed_noeval_b16_s1000_20260523_rerun"
)


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def newest_json(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise SystemExit(f"missing JSON {pattern} under {directory}")
    return matches[-1]


def run_command(command: list[str]) -> dict | None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(command)}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def copy_result(source_run: Path, fixture_dir: Path) -> None:
    batch_dir = fixture_dir / "batch_b16"
    for subdir, pattern in [
        ("lozo_minimal", "lozo_perf_minimal_*.json"),
        ("vllm_minimal_eager0", "vllm_perf_minimal_*.json"),
        ("vllm_detailed_eager0", "vllm_perf_detailed_*.json"),
    ]:
        target_dir = batch_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(newest_json(source_run / subdir, pattern), target_dir)


def write_manifest(fixture_dir: Path) -> None:
    manifest = {
        "collect_command": [
            ".venv/bin/python",
            "-u",
            "phase3/runners/collect_q1_batch_sweep.py",
            str(fixture_dir),
            "--output",
            str(fixture_dir / "finalize_summary.md"),
        ],
        "validate_command": [
            ".venv/bin/python",
            "-u",
            "phase3/runners/validate_q1_batch_sweep.py",
            str(fixture_dir),
            "--expected-batches",
            "16",
            "--require-detailed-batches",
            "16",
            "--min-speedup",
            "1.2",
            "--output",
            str(fixture_dir / "finalize_validation.json"),
        ],
    }
    (fixture_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-test Phase 3 q=1 helper tools.")
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--keep-fixture", action="store_true")
    args = parser.parse_args()
    args.source_run = resolve_path(args.source_run)
    return args


def main() -> None:
    args = parse_args()
    if not args.source_run.exists():
        raise SystemExit(f"source run does not exist: {args.source_run}")

    with tempfile.TemporaryDirectory(prefix="phase3_q1_tools_") as tmpdir:
        fixture_dir = Path(tmpdir) / "batch_fixture"
        copy_result(args.source_run, fixture_dir)
        write_manifest(fixture_dir)

        run_command([
            ".venv/bin/python",
            "-u",
            "phase3/runners/collect_q1_batch_sweep.py",
            str(fixture_dir),
            "--output",
            str(fixture_dir / "summary.md"),
        ])
        validate_report = run_command([
            ".venv/bin/python",
            "-u",
            "phase3/runners/validate_q1_batch_sweep.py",
            str(fixture_dir),
            "--expected-batches",
            "16",
            "--require-detailed-batches",
            "16",
            "--min-speedup",
            "1.2",
            "--output",
            str(fixture_dir / "validation.json"),
        ])
        status_report = run_command([
            ".venv/bin/python",
            "-u",
            "phase3/runners/status_q1_run.py",
            str(fixture_dir),
        ])
        run_command([
            ".venv/bin/python",
            "-u",
            "phase3/runners/finalize_q1_run.py",
            str(fixture_dir),
        ])

        if not validate_report or not validate_report.get("ok"):
            raise SystemExit("validator did not report ok=true")
        if not status_report or status_report.get("run_type") != "batch_sweep":
            raise SystemExit("status tool did not recognize the batch sweep fixture")

        if args.keep_fixture:
            kept_dir = PROJECT_ROOT / "phase3" / "results" / "selftest_q1_tools_fixture"
            if kept_dir.exists():
                raise SystemExit(f"kept fixture already exists: {kept_dir}")
            shutil.copytree(fixture_dir, kept_dir)
            print(f"kept_fixture={kept_dir}", flush=True)

    print("selftest_ok=true", flush=True)


if __name__ == "__main__":
    main()
